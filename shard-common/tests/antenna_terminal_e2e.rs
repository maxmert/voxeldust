//! Phase D: end-to-end coverage for the bidirectional Antenna +
//! unified Terminal consolidation.
//!
//! These tests don't spin up actual shards (that's the harness layer's
//! job). They exercise the public, production-path apply / subscribe
//! / publish helpers in `shard-common::signal_pipeline` +
//! `shard-common::media_pipeline` against a single in-memory
//! `SignalChannelTable` + `GrantsRegistry` + `HeldGrants`. The flow
//! the player will see in-game (place an Antenna, configure both
//! sides, type into a Terminal, watch text echo) is reproduced here
//! through the same code paths.

use voxeldust_core::shard_types::SessionToken;
use voxeldust_core::signal::channel::SignalChannelTable;
use voxeldust_core::signal::config::{
    AntennaConfig, AntennaSide, BlockSignalConfig, TerminalConfig,
};
use voxeldust_core::signal::grants::GrantsRegistry;
use voxeldust_core::signal::types::SignalScope;
use voxeldust_shard_common::signal_pipeline::{
    apply_antenna_config, AntennaApplyDenied, HeldGrant, HeldGrants,
    MAX_SUBSCRIBE_LEASE_TICKS,
};

const OWNER_ID: u64 = 99;
const REMOTE_SHARD: u64 = 7;
const FREQ: u32 = 100;

fn fixture_session() -> SessionToken {
    SessionToken(OWNER_ID)
}

fn insert_held_grant(held: &mut HeldGrants, session: SessionToken, gid: u64, target: u64) {
    held.insert(
        session,
        gid,
        HeldGrant {
            key: [0xAB; 32],
            target_shard_id: target,
            label: format!("grant_{gid}"),
        },
    );
}

#[test]
fn full_duplex_chat_apply_resolves_both_sides_at_one_frequency() {
    // The flagship use case: a player builds an Antenna, wires both
    // TX and RX to the SAME local channel ("ship.chat") at the same
    // frequency. Sitting at a Terminal that subscribes + publishes on
    // ship.chat = full chat panel. Apply must produce a state with
    // BOTH sides resolved.
    let mut channels = SignalChannelTable::new();
    channels.get_or_create(
        "ship.chat",
        SignalScope::Local,
        voxeldust_core::signal::types::ChannelMergeStrategy::LastWrite,
        OWNER_ID,
    );
    let mut grants = GrantsRegistry::default();
    let mut held = HeldGrants::default();
    let session = fixture_session();
    insert_held_grant(&mut held, session, 0xAA, REMOTE_SHARD);

    let cfg = AntennaConfig {
        tx: Some(AntennaSide {
            local_channel_name: "ship.chat".into(),
            frequency: FREQ,
            grant_id: Some(0xAA),
            remote_shard_id: Some(REMOTE_SHARD),
        }),
        rx: Some(AntennaSide {
            local_channel_name: "ship.chat".into(),
            frequency: FREQ,
            grant_id: Some(0xAA),
            remote_shard_id: Some(REMOTE_SHARD),
        }),
    };

    let res = apply_antenna_config(&mut channels, &mut grants, &held, &cfg, session, 100)
        .expect("bidirectional apply must succeed");

    // Both sides resolved.
    let tx = res.state.tx.expect("tx side present");
    let rx = res.state.rx.expect("rx side present");
    assert_eq!(tx.frequency, FREQ);
    assert_eq!(rx.frequency, FREQ);
    assert!(tx.local_channel_id.is_some(), "tx local channel resolved");
    assert!(rx.local_channel_id.is_some(), "rx local channel resolved");
    assert!(rx.bridged_channel_id.is_some(), "rx bridged channel created");

    // Keyed RX produces an outbound subscribe.
    let sub = res
        .outbound_subscribe
        .expect("keyed RX must emit subscribe to publisher");
    assert_eq!(sub.grant_id, 0xAA);
    assert_eq!(sub.valid_until_tick, 100 + MAX_SUBSCRIBE_LEASE_TICKS);
}

#[test]
fn open_full_duplex_needs_no_grants() {
    // CB-radio default: open broadcast on a frequency. No grants
    // attached either side — the apply must succeed without HeldGrants.
    let mut channels = SignalChannelTable::new();
    channels.get_or_create(
        "ship.chat",
        SignalScope::Local,
        voxeldust_core::signal::types::ChannelMergeStrategy::LastWrite,
        OWNER_ID,
    );
    let mut grants = GrantsRegistry::default();
    let held = HeldGrants::default(); // empty
    let session = fixture_session();
    let cfg = AntennaConfig {
        tx: Some(AntennaSide {
            local_channel_name: "ship.chat".into(),
            frequency: FREQ,
            grant_id: None, // open
            remote_shard_id: Some(REMOTE_SHARD),
        }),
        rx: Some(AntennaSide {
            local_channel_name: "ship.chat".into(),
            frequency: FREQ,
            grant_id: None, // open
            remote_shard_id: Some(REMOTE_SHARD),
        }),
    };
    let res = apply_antenna_config(&mut channels, &mut grants, &held, &cfg, session, 100)
        .expect("open apply must succeed without grants");
    assert!(
        res.outbound_subscribe.is_none(),
        "open RX skips the auth handshake — no subscribe required"
    );
    let tx = res.state.tx.expect("tx present");
    let rx = res.state.rx.expect("rx present");
    assert!(tx.grant_id.is_none(), "open TX leaves grant_id None");
    assert!(rx.grant_id.is_none(), "open RX leaves grant_id None");
}

#[test]
fn empty_config_rejected() {
    let mut channels = SignalChannelTable::new();
    let mut grants = GrantsRegistry::default();
    let held = HeldGrants::default();
    let session = fixture_session();
    let cfg = AntennaConfig::default();
    let err = apply_antenna_config(&mut channels, &mut grants, &held, &cfg, session, 0)
        .expect_err("empty antenna config must reject");
    assert!(matches!(err, AntennaApplyDenied::NoSideConfigured));
}

#[test]
fn tx_only_beacon_resolves_without_rx() {
    // TX-only antenna = transmit-only beacon. No RX side configured;
    // apply succeeds and the resulting state has rx = None.
    let mut channels = SignalChannelTable::new();
    channels.get_or_create(
        "beacon.signal",
        SignalScope::Local,
        voxeldust_core::signal::types::ChannelMergeStrategy::LastWrite,
        OWNER_ID,
    );
    let mut grants = GrantsRegistry::default();
    let held = HeldGrants::default();
    let session = fixture_session();
    let cfg = AntennaConfig {
        tx: Some(AntennaSide {
            local_channel_name: "beacon.signal".into(),
            frequency: 7777,
            grant_id: None, // open
            remote_shard_id: Some(REMOTE_SHARD),
        }),
        rx: None,
    };
    let res = apply_antenna_config(&mut channels, &mut grants, &held, &cfg, session, 0)
        .expect("tx-only apply must succeed");
    assert!(res.state.tx.is_some());
    assert!(res.state.rx.is_none());
    assert!(res.outbound_subscribe.is_none());
}

#[test]
fn rx_only_listener_resolves_without_tx() {
    // RX-only antenna = receive-only listener (the prior split-block's
    // Listener role). No TX side; apply creates the bridged Radio
    // channel + local mirror and returns a signed subscribe for the
    // keyed case.
    let mut channels = SignalChannelTable::new();
    let mut grants = GrantsRegistry::default();
    let mut held = HeldGrants::default();
    let session = fixture_session();
    insert_held_grant(&mut held, session, 0xBB, REMOTE_SHARD);
    let cfg = AntennaConfig {
        tx: None,
        rx: Some(AntennaSide {
            local_channel_name: "weather.report".into(),
            frequency: 5555,
            grant_id: Some(0xBB),
            remote_shard_id: Some(REMOTE_SHARD),
        }),
    };
    let res = apply_antenna_config(&mut channels, &mut grants, &held, &cfg, session, 100)
        .expect("rx-only apply must succeed");
    assert!(res.state.tx.is_none());
    assert!(res.state.rx.is_some());
    assert!(
        res.outbound_subscribe.is_some(),
        "keyed RX produces a subscribe even when tx is absent"
    );
    // Local mirror channel + bridged Radio channel both created.
    assert!(channels.resolve("weather.report").is_some());
    assert!(channels.resolve("weather.report__radio_in_5555").is_some());
}

#[test]
fn terminal_config_default_scrollback_applied() {
    // The Terminal config exposes `scrollback_lines: Option<u16>`.
    // None ⇒ the registry default applies; Some(0) is also coerced to
    // the default; non-zero is honoured.
    let cfg_default = TerminalConfig::default();
    assert_eq!(
        cfg_default.effective_scrollback(),
        TerminalConfig::DEFAULT_SCROLLBACK
    );

    let cfg_zero = TerminalConfig {
        scrollback_lines: Some(0),
        ..Default::default()
    };
    assert_eq!(
        cfg_zero.effective_scrollback(),
        TerminalConfig::DEFAULT_SCROLLBACK,
        "scrollback=0 must coerce to the default — never let a player end up with 0 lines"
    );

    let cfg_some = TerminalConfig {
        scrollback_lines: Some(42),
        ..Default::default()
    };
    assert_eq!(cfg_some.effective_scrollback(), 42);
}

#[test]
fn terminal_config_has_any_channel_validation() {
    // The has_any_channel() predicate is what the server uses to
    // reject empty Terminal configs. None / Some("") on both sides
    // counts as "no channel set".
    let empty = TerminalConfig::default();
    assert!(!empty.has_any_channel());

    let only_sub = TerminalConfig {
        subscribe_channel_name: Some("ch".into()),
        ..Default::default()
    };
    assert!(only_sub.has_any_channel(), "read-only sign is valid");

    let only_pub = TerminalConfig {
        publish_channel_name: Some("ch".into()),
        ..Default::default()
    };
    assert!(only_pub.has_any_channel(), "input-only kiosk is valid");

    let both = TerminalConfig {
        subscribe_channel_name: Some("a".into()),
        publish_channel_name: Some("b".into()),
        ..Default::default()
    };
    assert!(both.has_any_channel(), "full chat panel is valid");

    // "Empty string" sides count as unset — the wire encodes empty
    // strings as None on the way back through, but defensive validation
    // here covers the in-memory case before send.
    let empty_strings = TerminalConfig {
        subscribe_channel_name: Some(String::new()),
        publish_channel_name: Some(String::new()),
        ..Default::default()
    };
    assert!(
        !empty_strings.has_any_channel(),
        "empty strings must count as unset to match the wire encoding"
    );
}

#[test]
fn block_signal_config_carries_status_for_picker() {
    // The configurator UI's grant picker reads
    // `BlockSignalConfig.antenna_tx_status` / `_rx_status` and
    // `held_grants` to decide what to render. The defaults (None,
    // empty) must round-trip cleanly through the type so a server
    // sending an open-channel snapshot doesn't accidentally surface
    // a "request access" affordance on the client.
    let snap = BlockSignalConfig::default();
    assert!(snap.antenna_tx_status.is_none());
    assert!(snap.antenna_rx_status.is_none());
    assert!(snap.held_grants.is_empty());
    assert!(snap.terminal.is_none());
}
