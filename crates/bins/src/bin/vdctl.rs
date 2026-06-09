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
//!   screenshot [--at-tick <N>] [label]   (capture a PNG — a `--capture` client only)
//! where `<index>` is a 0-based action-bit index (converted to a single-bit mask, so
//!       you can never accidentally press two), and `<field>`/`<op>` are a `WaitField`/
//!       `WaitOp` — the live lists are shown in the `bad field` / `bad op` errors,
//!       derived from `WaitField::ALL` / `WaitOp::ALL` so they can never go stale.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::ExitCode;

use vd_bins::loopback;
use vd_devproto::{DevRequest, DevResponse, WaitField, WaitOp, WaitPredicate};

/// Default `wait-until` budget when none is given, in STEP-TICKS (≈10 s at the default
/// 20 Hz step cadence; shorter/longer if the client runs a non-default `--step-hz`).
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
        DevResponse::Ack
        | DevResponse::State { .. }
        | DevResponse::Captured { .. }
        | DevResponse::Recorded { .. } => ExitCode::SUCCESS,
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
            // The index→single-bit-mask transform and its bound are the SHARED
            // vd_devproto::action_bit (the same the windowed keymap uses), so an agent
            // can never set two bits at once and both injection paths reject an
            // out-of-range index identically — no local `1 << index`.
            let bit = vd_devproto::action_bit(index).ok_or_else(|| {
                format!(
                    "action: <index> must be 0..={}",
                    vd_devproto::MAX_ACTION_INDEX
                )
            })?;
            let pressed = parse_bool(&rest[1])?;
            Ok(DevRequest::Action { bit, pressed })
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
            // A predicate that can NEVER hold — a 0/1 flag like `active eq 2`/`gt 1`, OR
            // a count like `universe_tick lt 0` — is a typo that would silently only ever
            // time out; reject it loudly. The field-class-aware rule lives in vd-devproto
            // so the client and vdctl can't drift.
            if !predicate.is_satisfiable(field.is_boolean()) {
                return Err(format!(
                    "wait: `{} {} {value}` can never hold",
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
        "screenshot" => {
            // screenshot [--at-tick <N>] [label] — capture a PNG (Capture-mode client).
            let mut at_tick = None;
            let mut label = None;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--at-tick" => {
                        let raw = rest.get(i + 1).ok_or("--at-tick requires a value")?;
                        at_tick =
                            Some(raw.parse().map_err(|_| {
                                "screenshot: --at-tick must be an integer".to_owned()
                            })?);
                        i += 2;
                    }
                    other if label.is_none() => {
                        label = Some(other.to_owned());
                        i += 1;
                    }
                    other => return Err(format!("screenshot: unexpected argument {other:?}")),
                }
            }
            Ok(DevRequest::Screenshot { at_tick, label })
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
        // The valid-field list is derived from WaitField::ALL — it can never go stale.
        let fields: Vec<&str> = WaitField::ALL.iter().map(|f| f.name()).collect();
        format!("bad field {raw:?} ({})", fields.join("|"))
    })
}

fn parse_op(raw: &str) -> Result<WaitOp, String> {
    serde_json::from_value(serde_json::Value::String(raw.to_owned())).map_err(|_| {
        // The valid-op list is derived from WaitOp::ALL — it can never go stale.
        let ops: Vec<&str> = WaitOp::ALL.iter().map(|o| o.name()).collect();
        format!("bad op {raw:?} ({})", ops.join("|"))
    })
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
