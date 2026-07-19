//! `vd-slot` — the dev-cluster port helper (HR6). Resolves the canonical
//! [`vd_devproto::DevPortScheme`] for a worktree `slot` (and optional client
//! `agent`) into shell-eval-able `KEY=VALUE` lines, so `scripts/dev-cluster.sh`
//! and `scripts/client.sh` NEVER hand-compute a port (single source of truth).
//!
//! ```text
//! vd-slot --worktree /path/to/wt   # -> VD_SLOT=… (the stable per-worktree slot)
//! vd-slot --slot 0                 # -> VD_GW_ADDR=… VD_ADMIN_ADDR=… …
//! vd-slot --slot 0 --agent 1       # adds VD_DEVCTL_PORT / VD_CLIENT_QUIC_PORT
//! ```
//! Every emitted value is POSIX single-quoted (`vd_bins::sh_quote`) so a path with
//! spaces or metacharacters can never word-split or inject when sourced. All port
//! math lives in vd-devproto (covered); this shim is a Tier-B bin.

use std::process::ExitCode;

use vd_bins::sh_quote;
use vd_devproto::{DevPortScheme, slot_for_worktree};

const HOST: &str = "127.0.0.1";

fn main() -> ExitCode {
    match run(std::env::args().skip(1)) {
        Ok(lines) => {
            print!("{lines}");
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("vd-slot: {msg}");
            eprintln!("usage: vd-slot (--worktree <path> | --slot <N> [--agent <M>])");
            ExitCode::FAILURE
        }
    }
}

fn run(mut args: impl Iterator<Item = String>) -> Result<String, String> {
    let mut slot: Option<u16> = None;
    let mut agent: Option<u16> = None;
    let mut worktree: Option<String> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--slot" => slot = Some(parse_u16("--slot", args.next())?),
            "--agent" => agent = Some(parse_u16("--agent", args.next())?),
            "--worktree" => worktree = Some(args.next().ok_or("--worktree needs a path")?),
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    // `--worktree` resolves the stable per-worktree slot and nothing else.
    if let Some(path) = worktree {
        if slot.is_some() {
            return Err("--worktree and --slot are mutually exclusive".to_owned());
        }
        return Ok(emit(&[("VD_SLOT", slot_for_worktree(&path).to_string())]));
    }

    let slot = slot.ok_or("--slot (or --worktree) is required")?;
    let ports = DevPortScheme::DEFAULT
        .slot_ports(slot)
        .map_err(|e| e.to_string())?;

    let mut out = vec![
        ("VD_SLOT", slot.to_string()),
        ("VD_ORCH_ADDR", format!("{HOST}:{}", ports.orchestrator)),
        ("VD_GW_ADDR", format!("{HOST}:{}", ports.gateway)),
        ("VD_SHARD_ADDR", format!("{HOST}:{}", ports.shard)),
        ("VD_ADMIN_ADDR", format!("{HOST}:{}", ports.admin)),
        // The k8s probe (/healthz + /readyz) surfaces — so bash/S4 tooling reads the ports from this
        // covered helper, never hand-computing the DevPortScheme probe offsets.
        (
            "VD_ORCH_PROBE_ADDR",
            format!("{HOST}:{}", ports.probe_orchestrator),
        ),
        (
            "VD_GW_PROBE_ADDR",
            format!("{HOST}:{}", ports.probe_gateway),
        ),
        (
            "VD_SHARD_PROBE_ADDR",
            format!("{HOST}:{}", ports.probe_shard),
        ),
        // Track R / 1d.2 (M-2 scope: the LOCAL 2-process crossing playground): the DEST shard's QUIC +
        // probe addrs, surfaced from the covered scheme so a dual `client.sh`/S4 scenario never
        // hand-computes the shard-b offset. Emitted for every slot; bound only by a dual `up`.
        ("VD_SHARD_B_ADDR", format!("{HOST}:{}", ports.shard_b)),
        (
            "VD_SHARD_B_PROBE_ADDR",
            format!("{HOST}:{}", ports.probe_shard_b),
        ),
        // S5b (the acceptance capstone: the LOCAL 3-process seed-forest crossing): the GALAXY
        // between-space shard's QUIC + probe addrs, surfaced from the covered scheme so a `--triple`
        // scenario never hand-computes the galaxy offset. Emitted for every slot; bound only by a
        // `--triple` up.
        ("VD_GALAXY_ADDR", format!("{HOST}:{}", ports.galaxy)),
        (
            "VD_GALAXY_PROBE_ADDR",
            format!("{HOST}:{}", ports.probe_galaxy),
        ),
        // NODE-PER-REALM (Forest): the Planet/Station/Area 7 realm-shards' QUIC + probe addrs, surfaced
        // from the covered scheme so a `--forest` client.sh/S4 scenario never hand-computes an offset.
        // Emitted for every slot; bound only by a `--forest` up.
        ("VD_PLANET_ADDR", format!("{HOST}:{}", ports.planet)),
        (
            "VD_PLANET_PROBE_ADDR",
            format!("{HOST}:{}", ports.probe_planet),
        ),
        ("VD_STATION_ADDR", format!("{HOST}:{}", ports.station)),
        (
            "VD_STATION_PROBE_ADDR",
            format!("{HOST}:{}", ports.probe_station),
        ),
        ("VD_AREA_ADDR", format!("{HOST}:{}", ports.area)),
        ("VD_AREA_PROBE_ADDR", format!("{HOST}:{}", ports.probe_area)),
    ];
    if let Some(agent) = agent {
        let devctl = ports.dev_control(agent).map_err(|e| e.to_string())?;
        let quic = ports.client_quic(agent).map_err(|e| e.to_string())?;
        out.push(("VD_AGENT", agent.to_string()));
        out.push(("VD_DEVCTL_PORT", devctl.to_string()));
        out.push(("VD_DEVCTL_ADDR", format!("{HOST}:{devctl}")));
        out.push(("VD_CLIENT_QUIC_PORT", quic.to_string()));
        out.push(("VD_CLIENT_QUIC_ADDR", format!("{HOST}:{quic}")));
    }
    Ok(emit(&out))
}

/// Render `KEY=VALUE` lines with every value sh-quoted for safe sourcing.
fn emit(pairs: &[(&str, String)]) -> String {
    let mut out = String::new();
    for (key, value) in pairs {
        out.push_str(key);
        out.push('=');
        out.push_str(&sh_quote(value));
        out.push('\n');
    }
    out
}

fn parse_u16(flag: &str, value: Option<String>) -> Result<u16, String> {
    value
        .ok_or_else(|| format!("{flag} needs a value"))?
        .parse::<u16>()
        .map_err(|_| format!("{flag} value must be a u16"))
}
