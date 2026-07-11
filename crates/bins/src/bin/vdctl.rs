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
//!   screenshot [--at-tick <N>] [label]       (capture a PNG — a `--capture` client only)
//!   record [--fps <N>] [--secs <D>] [label]  (capture a frame sequence — `--capture` only)
//! where `<index>` is a 0-based action-bit index (converted to a single-bit mask, so
//!       you can never accidentally press two), and `<field>`/`<op>` are a `WaitField`/
//!       `WaitOp` — the live lists are shown in the `bad field` / `bad op` errors,
//!       derived from `WaitField::ALL` / `WaitOp::ALL` so they can never go stale.

use std::process::ExitCode;

use vd_bins::dev_roundtrip;
use vd_devproto::{DevRequest, DevResponse, WaitField, WaitOp, WaitPredicate};

/// Default `wait-until` budget when none is given, in STEP-TICKS (≈10 s at the default
/// 20 Hz step cadence; shorter/longer if the client runs a non-default `--step-hz`).
const DEFAULT_WAIT_TICKS: u64 = 200;

/// `record` defaults when not given: 30 fps for 2 s (a short transition clip). The client
/// clamps both to safe bounds, so a typo is corrected there, not silently honored.
const DEFAULT_RECORD_FPS: u32 = 30;
const DEFAULT_RECORD_SECS: f64 = 2.0;

/// Closed-loop `walk_to`/`look_at` defaults. `arrive_epsilon` = 0.5 m sits comfortably above
/// one sim step (so the fixed-magnitude Move never overshoots into oscillation — the contract
/// `nav::walk_to` pins); `align_epsilon` = 0.02 rad (~1.1°) is the true 3-D facing tolerance.
/// The tick budgets bound a drive (≈20 s / 10 s at 20 Hz) so an unreachable target times out.
const DEFAULT_ARRIVE_EPSILON: f64 = 0.5;
const DEFAULT_ALIGN_EPSILON: f64 = 0.02;
const DEFAULT_WALK_TICKS: u64 = 400;
const DEFAULT_LOOK_TICKS: u64 = 200;

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
        .ok_or(
            "missing command: move|look|action|close|reset|state|wait|screenshot|record|walk_to|look_at",
        )?;
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
        "record" => {
            // record [--fps <N>] [--secs <D>] [label] — a reduced-rate frame sequence
            // (Capture-mode client). The client paces + clamps; defaults fill in the rest.
            let mut fps = DEFAULT_RECORD_FPS;
            let mut secs = DEFAULT_RECORD_SECS;
            let mut label = None;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--fps" => {
                        let raw = rest.get(i + 1).ok_or("--fps requires a value")?;
                        fps = raw
                            .parse()
                            .map_err(|_| "record: --fps must be a positive integer".to_owned())?;
                        i += 2;
                    }
                    "--secs" => {
                        let raw = rest.get(i + 1).ok_or("--secs requires a value")?;
                        secs = raw
                            .parse()
                            .map_err(|_| "record: --secs must be a number".to_owned())?;
                        i += 2;
                    }
                    other if label.is_none() => {
                        label = Some(other.to_owned());
                        i += 1;
                    }
                    other => return Err(format!("record: unexpected argument {other:?}")),
                }
            }
            Ok(DevRequest::Record { fps, secs, label })
        }
        "walk_to" => {
            let usage = "walk_to <x> <y> <z> [arrive_epsilon] [max_ticks]";
            let (target, arrive_epsilon, max_ticks) =
                parse_target(rest, usage, DEFAULT_ARRIVE_EPSILON, DEFAULT_WALK_TICKS)?;
            Ok(DevRequest::WalkTo {
                target,
                arrive_epsilon,
                max_ticks,
            })
        }
        "look_at" => {
            let usage = "look_at <x> <y> <z> [align_epsilon] [max_ticks]";
            let (target, align_epsilon, max_ticks) =
                parse_target(rest, usage, DEFAULT_ALIGN_EPSILON, DEFAULT_LOOK_TICKS)?;
            Ok(DevRequest::LookAt {
                target,
                align_epsilon,
                max_ticks,
            })
        }
        other => Err(format!("unknown command: {other}")),
    }
}

/// Shared parse for the two closed loops: EXACTLY three world coords, then an OPTIONAL
/// epsilon and OPTIONAL max_ticks (each defaulting) — `walk_to`/`look_at` differ only in the
/// defaults + the request they build. Rejects too few / too many args loudly.
fn parse_target(
    rest: &[String],
    usage: &str,
    default_epsilon: f64,
    default_ticks: u64,
) -> Result<([f64; 3], f64, u64), String> {
    if rest.len() < 3 || rest.len() > 5 {
        return Err(usage.to_owned());
    }
    let target = floats64::<3>(&rest[..3], usage)?;
    let epsilon = match rest.get(3) {
        Some(raw) => raw
            .parse()
            .map_err(|_| format!("{usage}: bad epsilon {raw:?}"))?,
        None => default_epsilon,
    };
    let max_ticks = match rest.get(4) {
        Some(raw) => raw
            .parse()
            .map_err(|_| format!("{usage}: bad max_ticks {raw:?}"))?,
        None => default_ticks,
    };
    Ok((target, epsilon, max_ticks))
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

/// Parse EXACTLY `N` f64s — like [`floats`] but for the closed-loop WORLD targets, which
/// need f64 range/precision (world coordinates dwarf f32's exact-integer band).
fn floats64<const N: usize>(args: &[String], usage: &str) -> Result<[f64; N], String> {
    exact_arity(args, N, usage)?;
    let mut out = [0.0f64; N];
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

/// Send one request line, read one response line — the SHARED `vd_bins::dev_roundtrip`
/// (one wire framing for vdctl, the load test, and the render-smoke gate; read bounded by
/// `DEVCTL_READ_TIMEOUT`). The client always answers (a `wait` terminates by `max_ticks`),
/// so a closed/empty read means the client died.
fn round_trip(port: u16, request: &DevRequest) -> Result<DevResponse, String> {
    dev_roundtrip(port, request)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|a| (*a).to_owned()).collect()
    }

    #[test]
    fn walk_to_parses_coords_and_defaults_the_rest() {
        let r = parse_command(&args(&["walk_to", "1", "2.5", "-3"])).expect("parse");
        assert_eq!(
            r,
            DevRequest::WalkTo {
                target: [1.0, 2.5, -3.0],
                arrive_epsilon: DEFAULT_ARRIVE_EPSILON,
                max_ticks: DEFAULT_WALK_TICKS,
            }
        );
    }

    #[test]
    fn look_at_takes_explicit_epsilon_and_budget() {
        let r = parse_command(&args(&["look_at", "0", "1", "0", "0.05", "120"])).expect("parse");
        assert_eq!(
            r,
            DevRequest::LookAt {
                target: [0.0, 1.0, 0.0],
                align_epsilon: 0.05,
                max_ticks: 120,
            }
        );
    }

    #[test]
    fn closed_loop_rejects_too_few_and_bad_numbers() {
        // Fewer than three coords is a loud usage error, never a silent partial.
        assert!(parse_command(&args(&["walk_to", "1", "2"])).is_err());
        // A non-numeric coordinate is rejected, not defaulted to 0.
        assert!(parse_command(&args(&["look_at", "1", "x", "3"])).is_err());
        // Too many trailing args (past epsilon + max_ticks) is rejected.
        assert!(parse_command(&args(&["walk_to", "1", "2", "3", "0.5", "400", "extra"])).is_err());
    }
}
