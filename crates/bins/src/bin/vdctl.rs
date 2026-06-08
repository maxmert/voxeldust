//! `vdctl` — the agent-facing dev-control CLI (HR6). It builds ONE `DevRequest`
//! from the command line, sends it as a JSON line to a client's loopback
//! dev-control listener, prints the single `DevResponse` line, and maps the
//! response to an exit code (Ack/State → 0, Timeout → 3, Error → 1) so Bash
//! scenarios can branch. The request/response shapes are `vd-devproto`'s serde
//! types — the SAME the client decodes — so the two cannot drift.
//!
//! Usage: `vdctl [--port P] <command>` (port falls back to `VD_DEVCTL_PORT`):
//!   move <fwd> <strafe> <vert> | look <yaw> <pitch> | action <index> <on|off>
//!   close | reset | state | wait <field> <op> <value> [max_ticks]
//! where `<index>` is a 0-based action-bit index (converted to a single-bit mask, so
//!       you can never accidentally press two), field ∈ active|entity_count|
//!       own_entity_set|snapshots_applied and op ∈ eq|ne|ge|le|gt|lt.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::ExitCode;

use vd_bins::loopback;
use vd_devproto::{DevRequest, DevResponse, WaitField, WaitOp, WaitPredicate};

/// Action bits are a u32 mask, so a 0-based index must be in `0..32`.
const MAX_ACTION_INDEX: u32 = 31;

/// Default `wait-until` budget when none is given (step-ticks; ~10 s at 20 Hz).
const DEFAULT_WAIT_TICKS: u64 = 200;

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("vdctl: {e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let port = take_port(&mut args)?;
    let request = parse_command(&args)?;
    let response = round_trip(port, &request)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())?
    );
    Ok(match response {
        DevResponse::Ack | DevResponse::State { .. } => ExitCode::SUCCESS,
        DevResponse::Timeout { .. } => ExitCode::from(3),
        DevResponse::Error { .. } => ExitCode::FAILURE,
    })
}

/// Take `--port P` out of the args (so the remainder is the command), falling back
/// to `VD_DEVCTL_PORT` (set by `client.sh`/`vd-slot`).
fn take_port(args: &mut Vec<String>) -> Result<u16, String> {
    if let Some(i) = args.iter().position(|a| a == "--port") {
        let raw = args.get(i + 1).ok_or("--port requires a value")?.clone();
        args.drain(i..=i + 1);
        return raw
            .parse()
            .map_err(|_| format!("--port: bad value {raw:?}"));
    }
    match std::env::var("VD_DEVCTL_PORT") {
        Ok(raw) => raw
            .parse()
            .map_err(|_| format!("VD_DEVCTL_PORT: bad value {raw:?}")),
        Err(_) => Err("no --port given and VD_DEVCTL_PORT not set".to_owned()),
    }
}

fn parse_command(args: &[String]) -> Result<DevRequest, String> {
    let cmd = args
        .first()
        .ok_or("missing command: move|look|action|close|reset|state|wait")?;
    let rest = &args[1..];
    match cmd.as_str() {
        "move" => Ok(DevRequest::Move {
            axes: floats::<3>(rest, "move <fwd> <strafe> <vert>")?,
        }),
        "look" => Ok(DevRequest::Look {
            delta: floats::<2>(rest, "look <yaw> <pitch>")?,
        }),
        "action" => {
            let usage = "action <index> <on|off>";
            exact_arity(rest, 2, usage)?;
            let index: u32 = rest[0]
                .parse()
                .map_err(|_| "action: <index> must be an integer".to_owned())?;
            if index > MAX_ACTION_INDEX {
                return Err(format!("action: <index> must be 0..={MAX_ACTION_INDEX}"));
            }
            let pressed = parse_bool(&rest[1])?;
            // Convert the 0-based index to a single-bit mask so the agent can never
            // set two action bits at once (the wire field is a u32 bitmask).
            Ok(DevRequest::Action {
                bit: 1u32 << index,
                pressed,
            })
        }
        "close" => no_args(rest, "close").map(|()| DevRequest::Close),
        "reset" => no_args(rest, "reset").map(|()| DevRequest::ResetInput),
        "state" => no_args(rest, "state").map(|()| DevRequest::State),
        "wait" => {
            let usage = "wait <field> <op> <value> [max_ticks]";
            if rest.len() < 3 || rest.len() > 4 {
                return Err(usage.to_owned());
            }
            let field = parse_field(rest.first().ok_or(usage)?)?;
            let op = parse_op(rest.get(1).ok_or(usage)?)?;
            let value = rest
                .get(2)
                .ok_or(usage)?
                .parse()
                .map_err(|_| "wait: <value> must be an integer".to_owned())?;
            let predicate = WaitPredicate { field, op, value };
            // A 0/1 flag compared so it can NEVER hold (e.g. `eq 2`, `gt 1`, `lt 0`) is a
            // typo that would silently only ever time out; reject it loudly. The
            // operator-aware rule lives in vd-devproto so client and vdctl can't drift.
            if field.is_boolean() && !predicate.is_satisfiable_for_boolean() {
                return Err(format!(
                    "wait: `{} {} {value}` can never hold for a 0/1 flag",
                    rest[0], rest[1]
                ));
            }
            let max_ticks = match rest.get(3) {
                Some(raw) => raw
                    .parse()
                    .map_err(|_| "wait: <max_ticks> must be an integer".to_owned())?,
                None => DEFAULT_WAIT_TICKS,
            };
            Ok(DevRequest::WaitUntil {
                predicate,
                max_ticks,
            })
        }
        other => Err(format!("unknown command: {other}")),
    }
}

/// Parse EXACTLY `N` floats — rejects too few AND too many (a fat-fingered extra arg
/// is a loud error, never silently dropped).
fn floats<const N: usize>(args: &[String], usage: &str) -> Result<[f32; N], String> {
    exact_arity(args, N, usage)?;
    let mut out = [0.0f32; N];
    for (slot, raw) in out.iter_mut().zip(args) {
        *slot = raw
            .parse()
            .map_err(|_| format!("{usage}: bad number {raw:?}"))?;
    }
    Ok(out)
}

/// Reject a command given the wrong number of arguments (honest CLI: no silent drop).
fn exact_arity(args: &[String], n: usize, usage: &str) -> Result<(), String> {
    if args.len() == n {
        Ok(())
    } else {
        Err(format!("{usage} (expected {n} args, got {})", args.len()))
    }
}

fn no_args(args: &[String], cmd: &str) -> Result<(), String> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(format!("{cmd} takes no arguments"))
    }
}

fn parse_bool(raw: &str) -> Result<bool, String> {
    match raw {
        "on" | "true" | "1" => Ok(true),
        "off" | "false" | "0" => Ok(false),
        other => Err(format!("expected on|off, got {other:?}")),
    }
}

/// Parse a `WaitField`/`WaitOp` through the SAME serde derive the client uses (the
/// snake_case names) — no second mapping to drift.
fn parse_field(raw: &str) -> Result<WaitField, String> {
    serde_json::from_value(serde_json::Value::String(raw.to_owned())).map_err(|_| {
        format!("bad field {raw:?} (active|entity_count|own_entity_set|snapshots_applied)")
    })
}

fn parse_op(raw: &str) -> Result<WaitOp, String> {
    serde_json::from_value(serde_json::Value::String(raw.to_owned()))
        .map_err(|_| format!("bad op {raw:?} (eq|ne|ge|le|gt|lt)"))
}

/// Send one request line, read one response line. The client always answers (a
/// `wait` terminates by `max_ticks`), so a closed/empty read means the client died.
fn round_trip(port: u16, request: &DevRequest) -> Result<DevResponse, String> {
    let addr = loopback(port);
    let stream = TcpStream::connect(addr).map_err(|e| format!("connect {addr}: {e}"))?;
    let mut writer = stream
        .try_clone()
        .map_err(|e| format!("clone stream: {e}"))?;
    let mut line = serde_json::to_string(request).map_err(|e| e.to_string())?;
    line.push('\n');
    writer
        .write_all(line.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer.flush().ok();

    let mut reply = String::new();
    BufReader::new(stream)
        .read_line(&mut reply)
        .map_err(|e| format!("read: {e}"))?;
    if reply.trim().is_empty() {
        return Err("no response (client closed the connection)".to_owned());
    }
    serde_json::from_str(reply.trim()).map_err(|e| format!("decode response: {e}"))
}
