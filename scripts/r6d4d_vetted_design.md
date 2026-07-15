# R-6d4-D Vetted Design of Record (wf_bec6d1a6)

## ADJUDICATION + CORRECTED DESIGN OF RECORD

The task is a design task (read-only); I'll leave the task tracking as-is and deliver the adjudication directly. Here is my adjudication.

---

# R-6d4-D ADJUDICATION + CORRECTED DESIGN OF RECORD (HEAD c2095a2)

## Findings adjudicated against code

**F1 (HIGH) — observability rationale wrong (fresh B delivers via first-contact Accept, not boot-counter Reset): CONFIRMED.** `classify_and_deliver` seeds a never-seen peer's `RecvState` from the *incoming* frame — `classes.entry(frame.class).or_insert(RecvState { incarnation: frame.incarnation, epoch: frame.epoch, hw: 0, primed: false })` (mesh.rs:1066-1071). Then `classify_reliable` sees `incarnation == st.incarnation` (A1 false, mesh.rs:364), `epoch == st.epoch` (A3/A4 false, mesh.rs:379/384), and routes to `prime_or_contiguous(st, seq, /*fresh_incarnation=*/false)` (mesh.rs:387). At seq0 that returns `Verdict::Accept`, NOT `Verdict::Reset` (mesh.rs:396-403). A fresh B has no prior ledger and cannot `Dedup` (Dedup requires `primed` + `seq <= hw`, mesh.rs:420). The JSON's claim "Reset-then-Accept… the reason the re-drive is OBSERVABLE at all" is FALSE for this topology. **Fold: correct the prose; make v2==v1+1 an INDEPENDENT exactly-once check.**

**F-CRITICAL (boot-1 seed-and-idle never increments the counter): CONFIRMED, and it is the design's central defect.** `increment_on_boot` (boot.rs:193) is called ONLY by `resolve_process_incarnation` (lib.rs:243), which is called ONLY by `boot_mesh_and_replay` (lib.rs:323). The JSON mandates boot-1 = "open_node_outbox → seed → confirm → idle" with NO `boot_mesh_and_replay` (open_questions #2). Therefore boot-1 never touches the counter: (a) the file does not exist, so the pre-kill read returns `Ok(None)` (boot.rs:210) — there is no v1; (b) boot-2's `boot_mesh_and_replay` mints *genesis* = `launch_incarnation().max(1)` (boot.rs:196), not v1+1. `v2 == v1+1` is unprovable as written. **Fold: boot-1 MUST resolve+increment the counter (Option A below).**

**F-HIGH (BootCounter::current conflates absent Ok(None) with a value): CONFIRMED.** `read` returns `Ok(None)` on `NotFound` (boot.rs:208-210), `Ok(Some(n))` valid, `Err` on corrupt (checksum at boot.rs:232). A `current(path) -> Result<Option<u64>, BootCounterError>` wrapper faithfully propagates None, but the JSON's `v2 == v1 + 1` never states how None is handled. **Fold: reader returns `Result<Option<u64>,_>`; the test asserts `Some(_)` on both reads (valid only under the Option-A fix), then `v2.unwrap() == v1.unwrap()+1`.**

**F-HIGH (seed_reliable_row body uncompilable — `from: self`): CONFIRMED.** `NodeOutbox { store, durability }` has NO NodeId field (outbox.rs:221-224). `ReliableFrame` requires `from: NodeId` (lib.rs:71). The JSON's `from: self` and the signature that omits `from` disagree with open_questions #0. **Fold: pin `pub fn seed_reliable_row(&mut self, from: NodeId, peer: NodeId, class: MsgClass, incarnation: u64, seq: u64, payload: &[u8]) -> u64`.** (The stored `from` is inert on replay — `decode_value_payload` strips the header, outbox.rs:385-389, and replay re-frames from payload only, outbox.rs:518 — but take the param for honesty, matching `framed_row`'s `NodeId(1)`, outbox.rs:927.)

**F-MEDIUM (return watermark = BATCH seq, not the frame-seq param): CONFIRMED.** `NodeOutbox::commit()` calls `wait_durable_through(store.last_submitted())` (outbox.rs:253-255); `DurabilityHandle::last_submitted()` and `RedbStore::last_submitted()` read the same atomic (store.rs:184-186 / 655-657). The method's `seq` PARAM is the OutboxKey frame-seq (outbox.rs:112), distinct from the store batch-seq. **Fold: return the batch seq; document `caller asserts ob.durability().is_durable_through(returned)`.**

**F-MEDIUM (.seeded marker must be written strictly AFTER commit returns / post-fsync): CONFIRMED sound-but-must-state.** `NodeOutbox::commit()` blocks until durable (outbox.rs:249-256 → store.rs:786-788 block-on-prior + wait). So the row is fsynced on `seed_reliable_row` return. Orchestrator_crash's marker is the INVERSE (a pre-fsync LOST window written by the paused writer, orchestrator_crash.rs:65). **Fold: strict happens-before — (1) seed returns, (2) assert durable, (3) THEN write `.seeded`, (4) idle.** The anti-theater twin writes NO seed ⇒ needs a distinct `.ready` marker (written after `open_node_outbox`) as its deterministic kill trigger.

**F-crash-window LOW (durable-confirm redundancy): CONFIRMED.** Belt-and-suspenders `assert!(ob.durability().is_durable_through(batch))` after the seed, before the marker, is honest fail-loud. **Fold.**

**F-observability LOW (deadline-not-sleep drain for BOTH tests): CONFIRMED.** `replay_outbox` fences on LOCAL durability (`wait_durable_through(target)`, outbox.rs:549), NOT B delivery; `boot_mesh_and_replay` returns after `replay_outbox` (lib.rs:346). The source→B QUIC dial + delivery are async. `mesh_redelivery.rs` proves the correct discipline: `wait_until` (mesh_redelivery.rs:75-81, DEADLINE 30s at :20) draining `drain_inbound()` (mesh.rs:1953). **Fold: positive test polls B via a 30s wait_until; the twin asserts empty over a bounded settle "long enough the positive path would have landed" (borrow the boot_counter_crashloop.rs:181-191 "RED-proves-the-window" framing).**

**F-observability LOW (anti-theater twin should also assert v2==v1+1): CONFIRMED valuable.** orchestrator_crash's twin negates the exact predicate (orchestrator_crash.rs:209-260). The positive test makes TWO assertions; a twin negating only the re-drive leaves the counter half untested for vacuity. **Fold: the twin ALSO reads v1/v2 and asserts +1 — proving the 0-re-drive is over a genuine restart.**

**F-observability NIT (bind B before boot-2): CONFIRMED trivial.** `spawn_mesh` binds synchronously before returning (mesh.rs boot; `boot_mesh_and_replay.rs:54` asserts `local_addr` up on return). **Fold: spawn B, assert `ctl_b.local_addr().is_ok()`, THEN reap-and-rebind the source.**

**F-addr LOW (reap-then-rebind-same-addr via kill_and_reap): CONFIRMED SOUND.** `kill_and_reap` does `kill()` + `wait()` (reap, frees port) + remove-from-set (lib.rs:727-736); boot_counter_crashloop uses exactly this then `push` same env (boot_counter_crashloop.rs:171-172), proven green in `orch-crash`. orchestrator_crash's fresh-addr restart (orchestrator_crash.rs:129-136) is correctly rejected for D. **Refinement: use Cluster uniformly for the source (not the KillOnDrop/Cluster mix); B + runtime held separately.**

**F-SCOPE MEDIUM (D's novel surface is ONLY process-tier SIGKILL + disk-reopen; re-drive already proven): CONFIRMED.** `replay_outbox_redrives_retained_rows_delivers_and_gcs_the_prior_incarnation` (mesh.rs:3589) already pre-seeds a real `NodeOutbox`, spawns A at incarnation 2, `replay_outbox` re-drives, B receives both payloads, the roster-gone row is quarantined — all in-process. R-6d4-A proves it purely. Every in-process test uses explicit `VD_PROCESS_INCARNATION` (boot_mesh_and_replay.rs:35) which SHORT-CIRCUITS the durable counter (lib.rs:225-227). **Fold: state D's non-duplicative properties are (1) the row survives a REAL SIGKILL + redb reopen, (2) the disk BootCounter increments by exactly 1 across a real restart — the ONE path the in-process tests bypass. B is the OBSERVATION mechanism, not a novel re-drive proof.**

**F-DRY MEDIUM/LOW (frame-encode home): PARTIALLY CONFIRMED, scope honestly.** There are already three frame-encode sites: `framed_row` (`#[cfg(test)]`, outbox.rs:925-935), the decode-test inline encode (outbox.rs:747), and `expected_encoded` (`#[cfg(test)]` in mesh.rs:3353-3369). `seed_reliable_row` (store-test-hooks pub) is a fourth. The `#[cfg(test)]` helpers CANNOT be called from a store-test-hooks integration test in another crate (visibility mismatch), and they build `Vec<u8>` for mocks, not a real store. **Fold the HONEST claim: `seed_reliable_row` is THE single real-store frame-and-retain builder; where crate-visible, `framed_row` may delegate to a shared private `fn frame_reliable(from,class,incarnation,epoch,seq,payload)->Vec<u8>` that `seed_reliable_row` also calls. Do NOT promise collapsing all three tiers.** Pin `epoch = u32::MAX` for consistency (framed_row/expected_encoded both use it; inert since replay re-frames).

**F-new-bin HIGH (under-justified; boot-2 needs nothing new): CONFIRMED.** shard.rs:33 ALREADY calls `boot_mesh_and_replay`. The real forcing function is boot-1's seed-and-idle branch (which must NOT replay — else the just-seeded row is re-driven+gc'd before the crash) PLUS the shard bin's snapshot-budget assert (shard.rs:24-29) + register_stub_shard/self-fence (shard.rs:43-79) needing realm/orch peers. **DECISION: a new minimal `vd-outbox-testnode` bin IS justified — but on the TRUE forcing function (boot-1 seed-and-idle cannot live in the shard's clean boot path), and boot-2 reuses `boot_mesh_and_replay` verbatim.**

**F-profile MEDIUM (orch-crash-cov must instrument BOTH fns AND the twin): CONFIRMED + adds a real gap.** `cargo test --test <name>` runs the whole file, so `orch-crash` (functional) covers both. But two facts the JSON got wrong: (a) `orch-crash-cov` currently accumulates ONLY orchestrator_crash + boot_guard (justfile:116-119) — it does NOT even include boot_counter_crashloop; and (b) **`gate` (justfile:149) runs `orch-crash` + `coverage`, NOT `orch-crash-cov`** — so the JSON's "gate already runs orch-crash-cov" is FALSE; orch-crash-cov is a standalone recipe. **Fold: add the new test to `orch-crash` (gate path, functional) AND `orch-crash-cov` (the %c continuous merge); both the positive test and twin live in ONE file so one `--test outbox_sigkill_restart` line covers both.**

**F-profile LOW (%c wiring correct; %m/%p wording): CONFIRMED.** `spawn_node` does `cmd.env(k,v)` with no `env_clear` (lib.rs:645-649), so the SIGKILLed child inherits `LLVM_PROFILE_FILE`; %c continuous mode is the only way a killed child's counters survive (no atexit flush). The io-prod library regions (`seed_reliable_row` in the child boot-1, `BootCounter::current` in the test process, `replay_outbox` in the child boot-2) merge by profdata regardless of pid. **Fold the precise wording.**

**F-profile LOW (required-features vs CARGO_BIN_EXE): CONFIRMED as a real hazard.** No `[[bin]]` in bins/Cargo.toml uses `required-features` today (Cargo.toml:9-44); the two proven crash tests reference bins with NO required-features (they always build) and gate only the test with `#![cfg]`. A `required-features` bin whose feature is off is skipped ⇒ `CARGO_BIN_EXE_*` undefined ⇒ compile error. **DECISION: make `vd-outbox-testnode` a plain `[[bin]]` whose `main()` body is `#[cfg(feature="store-test-hooks")]`-gated (empty/err main without the feature), so `CARGO_BIN_EXE_vd-outbox-testnode` is ALWAYS defined — removing the interaction risk. Release-hygiene is preserved by the gated body (no seed logic compiles without the feature).**

**spawn_mesh 2-peer guard: CONFIRMED non-issue.** `spawn_mesh` rejects `outbox.is_some() && peers.len() > OUTBOX_WRITER_CHANNEL_DEPTH` (256) (mesh.rs:824). A 2-peer book passes.

**redb reopen-after-SIGKILL: CONFIRMED safe.** orchestrator_crash proves kill-9-then-reopen recovery for the same RedbStore (orchestrator_crash.rs:148-159); NodeOutbox uses the same RedbStore. This is THE load-bearing property D exists to prove at the process tier.

---

## CORRECTED DESIGN OF RECORD — R-6d4-D

### 1. SEED SEAM (io-prod)
Add to `outbox.rs`, beside `NodeOutbox::open` (outbox.rs:226-238), a store-test-hooks pub method:
```
#[cfg(any(test, feature = "store-test-hooks"))]
pub fn seed_reliable_row(&mut self, from: NodeId, peer: NodeId, class: MsgClass,
                         incarnation: u64, seq: u64, payload: &[u8]) -> u64 {
    let framed = frame_reliable(from, class, incarnation, u32::MAX, seq, payload); // the ONE encode home
    self.retain(&OutboxKey { peer, class, incarnation, seq }, &framed); // outbox.rs:241 store.put
    self.commit();                                                       // outbox.rs:249-256: durable-on-return
    self.durability.last_submitted()                                     // the durable BATCH seq
}
```
- `frame_reliable(from,class,incarnation,epoch,seq,payload)->Vec<u8>` is a new private helper (`vd_wire::framing::encode_frame(&ReliableFrame{..})`, keeping `ReliableFrame` pub(crate)); `framed_row` (outbox.rs:925, crate-visible) delegates to it — DRY, honestly scoped (mesh.rs `expected_encoded` stays separate; different crate/mock tier).
- Durability is guaranteed by `commit()` alone (outbox.rs:253-255) — NO pause hook, NO extra wait.

### 2. BootCounter reader (io-prod)
Add `#[cfg(any(test, feature="store-test-hooks"))] pub fn BootCounter::current(path) -> Result<Option<u64>, BootCounterError>` — a thin wrapper over the private `read` (boot.rs:207), keeping the MAGIC + checksum guard in ONE place (prefer over a raw 24-byte read). Returns `Ok(None)` for absent, `Ok(Some(n))` valid, `Err` corrupt.

### 3. Purpose-built bin: `vd-outbox-testnode` (bins)
Plain `[[bin]]` (NO required-features), `main()` body gated `#[cfg(feature="store-test-hooks")]`. Reads `VD_NODE_ID/VD_BIND/VD_PEERS/VD_OUTBOX_PATH(+EPHEMERAL_OK)/VD_BOOT_STATE_DIR(+EPHEMERAL_OK)/VD_OUTBOX_TEST_SEED`. Branch on the seed env:
- **boot-1 (VD_OUTBOX_TEST_SEED SET):** `resolve_process_incarnation(&env)` FIRST (mints + durably writes v1 — this is the F-CRITICAL Option-A fix) → `open_node_outbox` → `ob.seed_reliable_row(from=VD_NODE_ID, peer=B, class=Saga, incarnation=v1, seq=0, payload)` → `assert!(ob.durability().is_durable_through(batch))` → write `.seeded` marker → idle (park; keep the outbox open). NO spawn_mesh, NO replay.
- **boot-2 (seed UNSET):** `boot_mesh_and_replay(&env, ...)` (resolves incarnation ONCE → v2 = v1+1, opens outbox, replays) → tick/drain loop keeping the mesh alive.
- The env MUST NOT carry `VD_PROCESS_INCARNATION` (else the durable counter is bypassed, lib.rs:225-227) — the test builds it by hand (like boot_mesh_and_replay.rs:30-41) with `VD_BOOT_STATE_DIR` + `VD_BOOT_STATE_EPHEMERAL_OK=1` instead.
- The seed spec mirrors the `VD_STORE_TEST_SENTINEL_SEED` u64-parse precedent (orchestrator.rs:175-186); parse `"class_byte,payload_hex"` (peer=B, incarnation=v1, seq=0 are fixed by the test topology).

Cargo.toml: `store-test-hooks` already forwards `vd-io-prod/store-test-hooks` (Cargo.toml:59) — no feature edit needed; the bin body gating provides release-hygiene.

### 4. The test: `crates/bins/tests/outbox_sigkill_restart.rs` (`#![cfg(feature="store-test-hooks")]`)
Topology: source = NodeId(1) (child bin), B = NodeId(2) (in-process real `MeshTransport`). Reserve both addrs (reserve_udp_addr, lib.rs:473). Shared book `[(1, addr_src), (2, addr_b)]` (book, lib.rs:153) → source via VD_PEERS, B via MeshConfig.peers.
1. Spawn B via `spawn_mesh(rt.handle(), &trust, &cfg_b, None)` (mesh_redelivery.rs:50-52); assert `ctl_b.local_addr().is_ok()`. Hold `(b, ctl_b, rt)` to end-of-test.
2. Spawn boot-1 source under a `Cluster` (RAII reap). Poll `.seeded` marker to existence (deterministic, orchestrator_crash.rs:96-105). Read `v1 = BootCounter::current(boot_dir/boot.counter)?` and `assert!(v1.is_some())`.
3. `cluster.kill_and_reap(SOURCE)` (SIGKILL + reap → frees addr_src + redb lock), then `cluster.push(SOURCE_RESTART, spawn_node(bin, common, env_no_seed))` on the SAME env/addr (boot_counter_crashloop.rs:171-172).
4. Drain B via `wait_until(|| { for ev in b.drain_inbound() { if Inbound::Wire{bytes,..} => collect } ; got.contains(&payload_byte) }, ...)` with the 30s DEADLINE (mesh_redelivery.rs:56-81). Assert the seed payload arrived (delivered via first-contact Accept, mesh.rs:1066-1071 + 394-403 — NOT dependent on the boot counter).
5. Read `v2 = BootCounter::current(...)?` AFTER B drains (boot-2's write is complete); `assert_eq!(v2.unwrap(), v1.unwrap() + 1)` — the INDEPENDENT exactly-once-incarnation check.

Anti-theater twin `fresh_store_re_drives_zero_rows` (same file): boot-1 with NO seed writes a `.ready` marker after `open_node_outbox`; read v1; SIGKILL + reap; boot-2 on the SAME empty outbox + fresh B; assert B's inbound stays EMPTY over a bounded settle (chosen so the positive path would have landed — replay_outbox returns `ReplayCounts::default()` on empty, outbox.rs:482-484) AND `v2 == v1 + 1` (proving the 0-re-drive is over a genuine restart). Mirror orchestrator_crash.rs:209-260.

### 5. Profile merge + justfile
- `orch-crash` (justfile:90-93): add `cargo test -p vd-bins --features store-test-hooks --test outbox_sigkill_restart -- --test-threads=1`. In the GATE path (justfile:149 runs `orch-crash`).
- `orch-crash-cov` (justfile:111-119): add a matching `cargo +TOOLCHAIN llvm-cov --no-report --branch -p vd-bins --features store-test-hooks --test outbox_sigkill_restart -- --test-threads=1` block under the `vd-orchcrash-%p-%m%c.profraw` `LLVM_PROFILE_FILE` export (justfile:115). Both fns (positive + twin) run under the one `--test` selector. Correct wording: `%m` separates the `vd-outbox-testnode` bin from the vd-bins test harness; `%p` separates the two source-bin boots; the io-prod regions merge by profdata regardless of pid. NOTE: `orch-crash-cov` is NOT in `gate` today — that is unchanged (it is the deeper owed %c merge, DEFERRED.md D-40).
- Serial `--test-threads=1` (real process + real UDP ports).
- Tier-B honesty: a SIGKILL loses the child's final counter flush after the last %c mmap sync — a genuine floor, not a bug. The new test ratchets the io-prod store-test-hooks floor (coverage-io-prod-hooks, justfile:48-49) UP as `seed_reliable_row` / `BootCounter::current` / `replay_outbox` execute at the process tier; never claims 100%.

### 6. Files touched
- `crates/io-prod/src/outbox.rs` — `frame_reliable` private helper + store-test-hooks `NodeOutbox::seed_reliable_row(from,peer,class,incarnation,seq,payload)->u64`; `framed_row` delegates.
- `crates/io-prod/src/boot.rs` — store-test-hooks/test `BootCounter::current(path)->Result<Option<u64>,BootCounterError>`.
- `crates/bins/src/bin/vd-outbox-testnode.rs` — NEW; plain `[[bin]]`, `main()` body store-test-hooks-gated; boot-1 resolve+seed+idle / boot-2 boot_mesh_and_replay+tick.
- `crates/bins/Cargo.toml` — new `[[bin]] name="vd-outbox-testnode"` (NO required-features).
- `crates/bins/tests/outbox_sigkill_restart.rs` — NEW; positive test + anti-theater twin.
- `justfile` — add to `orch-crash` + `orch-crash-cov`.

### 7. Open questions / API-insufficiencies
- **Seed spec format**: `VD_OUTBOX_TEST_SEED = "class_byte,payload_hex"` (peer=2/incarnation=v1/seq=0 fixed) vs a fuller tuple. Recommend the minimal form; the topology fixes the rest.
- **Idle mechanism for boot-1**: `std::thread::park()` vs a sleep loop. Recommend `park()` — the test SIGKILLs it; keep the outbox open (do not drop → no graceful flush/join timing).
- **v2 read timing**: read v2 AFTER B drains the re-drive (boot-2's counter write is complete + quiescent), not during the boot-2 rename window.
- **Cleanup**: TempPaths RAII (boot_counter_crashloop.rs:35-46) for outbox + boot dir + trust dir; `Cluster` Drop reaps the source on panic; B `ctl_b` + `rt` held to end.
- **CA-1 boundary honest**: the SIGKILL-source-in-AwaitAdopt real-QUIC e2e stays OUT (AwaitAdopt egress empty, saga.rs:1474; needs CA-1 + M3). D proves the RESTART machinery (survives-a-kill + counter+1), NOT the never-restart closure.

---

**VERDICT: SOUND_TO_IMPLEMENT** (with the folded must-fixes: (1) boot-1 MUST call `resolve_process_incarnation` to mint v1 [F-CRITICAL Option A] so v2==v1+1 is provable; (2) `seed_reliable_row` takes a `from: NodeId` param + returns the durable BATCH seq; (3) `BootCounter::current` returns `Result<Option<u64>,_>` and the test asserts `Some` on both reads; (4) correct the observability prose — fresh B delivers via first-contact Accept, v2==v1+1 is an INDEPENDENT exactly-once check, not a delivery precondition; (5) `.seeded` written strictly post-`commit`; the twin uses a `.ready` marker + asserts v2==v1+1; (6) 30s deadline-drain for both tests, never a bare sleep; (7) new `vd-outbox-testnode` bin justified on the boot-1-seed-and-idle forcing function, made a plain `[[bin]]` with a feature-gated `main()` body [NOT required-features] so `CARGO_BIN_EXE_*` is always defined; (8) add the test to BOTH `orch-crash` and `orch-crash-cov`; scope D to its novel process-tier SIGKILL+disk-reopen + disk-counter surface, not re-proving mesh.rs:3589.)

### DESIGNER.seed_seam
SEED SEAM — recommend a `store-test-hooks` pub `NodeOutbox::seed_reliable_row` on `crates/io-prod/src/outbox.rs` (add beside `NodeOutbox::open`, outbox.rs:226-238), NOT a free function, because it must build+frame a `ReliableFrame` (pub(crate) at lib.rs:70) and call the private-ish `retain`/`commit` on `self` while keeping `ReliableFrame` pub(crate).

Exact signature + body (all pieces exist in-tree):
```
#[cfg(feature = "store-test-hooks")]
pub fn seed_reliable_row(&mut self, peer: NodeId, class: MsgClass, incarnation: u64, seq: u64, payload: &[u8]) -> u64 {
    // 1. Build+frame internally (keeps ReliableFrame pub(crate) — no bin touches it).
    //    epoch=u32::MAX matches the write-path re-stamp the A-proptest already uses (framed_row, outbox.rs:925-935 / DedupLedger note outbox.rs:1430).
    let framed = vd_wire::framing::encode_frame(&crate::ReliableFrame {
        from: self /*local id not stored on NodeOutbox*/, ... }).expect("encode seed frame");
    // NodeOutbox has no local NodeId today — pass `from` as a param OR stamp NodeId(0) (the stored `from` is never read on replay: replay_outbox re-frames from decode_value_payload's PAYLOAD only, outbox.rs:385-389/518). Simplest: seed `from = peer`'s counterpart is irrelevant; take a `from: NodeId` arg for honesty.
    let key = OutboxKey { peer, class, incarnation, seq };
    self.retain(&key, &framed);   // outbox.rs:241 — stages via store.put
    self.commit();                // outbox.rs:249-256 — submit_barrier + wait_durable_through: RETURNS ONLY WHEN DURABLE
    self.durability.last_submitted()  // return the durable seq so the caller can assert is_durable_through
}
```
KEY INSIGHT: `NodeOutbox::commit()` (outbox.rs:249-256) already does submit + `wait_durable_through(last_submitted())` — so the row is FSYNCED on return. That is the whole durability guarantee; no pause hook, no extra wait needed. `retain` uses `encode_value` (format-version envelope, outbox.rs:143-149) and `decode_value_payload` (outbox.rs:385) recovers the PAYLOAD on replay, so the seeded payload is exactly what peer B observes.

DRY: refactor B2's 2-row pre-seed (currently the MockOutboxSink path) is a MOCK, so no real overlap; but the A-proptest's `framed_row` (outbox.rs:925) and B2 could both call this if they used a real NodeOutbox — recommend the seed method call `framed_row`-equivalent internally so the frame encoding lives in ONE place. Minimum: the seed method is the ONE real-store frame-and-retain builder.

INVOCATION VIA ENV (the bin side): add a `store-test-hooks`-gated seed path to the SOURCE bin. Env var `VD_OUTBOX_TEST_SEED` = a compact spec the bin parses, e.g. `"peer,class_byte,incarnation,seq,payload_hex"` (mirror the orchestrator's `VD_STORE_TEST_SENTINEL_SEED` u64-parse precedent, orchestrator.rs:175-186). The bin, AFTER `open_node_outbox` but BEFORE (or instead of) handing the outbox to `boot_mesh_and_replay`, calls `ob.seed_reliable_row(...)` then confirms `ob.durability().is_durable_through(seq)`. See seed_seam→bin flag under files.

FLAG: a MINIMAL PURPOSE-BUILT TEST NODE BIN IS NEEDED (do NOT reuse vd-shard). Reasons: (1) the shard bin (shard.rs:12-85) hardcodes register_stub_shard + register_clock_follower + a snapshot-budget assert + realm/self-fence validates — all irrelevant noise that would need a peer/orchestrator to not error; (2) boot 1 must SEED then confirm durable then let boot replay run on boot 2 — the seed belongs in a bin whose ONLY job is open_node_outbox → (boot1: seed+confirm+idle) / (boot2: boot_mesh_and_replay → tick-drain forever). A ~40-line `vd-outbox-testnode` bin under `crates/bins/src/bin/` gated `#![cfg(feature="store-test-hooks")]` reading VD_NODE_ID/VD_BIND/VD_PEERS/VD_OUTBOX_PATH/VD_BOOT_STATE_DIR/VD_OUTBOX_TEST_SEED, calling `boot_mesh_and_replay` (which already resolves incarnation ONCE + opens outbox + replays, lib.rs:311-356), is DRY-correct and keeps the shard bin clean. It needs `required-features = ["store-test-hooks"]` in Cargo.toml so release never builds it.

### DESIGNER.crash_window
CRASH WINDOW — guarantee DURABLE-and-unacked BEFORE the SIGKILL (must NOT be the pause window = a lost row).

The DOR's must-fix is already structurally satisfied by the seed method design: `NodeOutbox::commit()` (outbox.rs:249-256) calls `self.store.commit()` then `self.durability.wait_durable_through(self.store.last_submitted())` — it BLOCKS until the batch is fsynced. So on `seed_reliable_row` return the row is durable-on-disk. This is the OPPOSITE of orchestrator_crash.rs, which deliberately uses `pause_on_key_prefix` (store.rs:85-92) to catch a submitted-but-PRE-fsync window (a LOST row). For D we want a SURVIVING row, so:

1. Boot-1 bin: `let seq = ob.seed_reliable_row(B, MsgClass::Saga, incarnation, 0, payload);` then `assert!(ob.durability().is_durable_through(seq))` — a hard in-bin confirmation the write reached disk (fail-loud if not).
2. Boot-1 bin then writes a "seed durable" MARKER file (path = outbox path with `.seeded` extension, mirroring orchestrator_crash.rs:65's `marker = store.with_extension("paused")`) so the TEST knows the durable window is open before it kills. This is the honest, decoupled durability signal (the bin, unlike orchestrator_crash, is NOT blocked, so IT can write its own marker — simpler than the writer-thread marker).
3. The row is unacked BY CONSTRUCTION: nothing ever calls `release`/`gc` in boot 1 (the bin just seeds and idles). peer B does not exist yet in boot 1 (see observability) so no ack path retires it.
4. Boot-1 bin then IDLES (park forever / sleep loop) holding the outbox open — do NOT drop it (a graceful Drop would flush+join, which is fine, but keeping it open is closer to the crash reality and avoids any writer-join timing). The test SIGKILLs it in this idle state.
5. TEST sequence: spawn boot-1 → poll `.seeded` marker to existence (deterministic, exactly like orchestrator_crash.rs:96-105) → read BootCounter v1 → SIGKILL + reap (Cluster::kill_and_reap or KillOnDrop.take()+kill()+wait(), lib.rs:727-736 / orchestrator_crash.rs:122-126) → the redb file lock releases on reap → spawn boot-2.

WHY the seed must NOT overlap the pause hook: the pause hook parks the writer FOREVER pre-fsync (store.rs:458-460, maybe_pause_before_fsync) and makes Drop's join hang (store.rs:1224 `std::mem::forget` precedent) — that models a LOST row (orchestrator_crash asserts grant B is LOST, orchestrator_crash.rs:167-172). For D the row must SURVIVE, so `pause_on_key_prefix`/`fail_fsync_on_key_prefix` are LEFT None (StoreTuning::default under store-test-hooks, store.rs:111-124). The confirm-durable-then-kill window is the entire point.

ANTI-VACUITY: also confirm the row is durable via the reopen semantics the outbox already proves (`retained_frames_survive_a_reopen`, outbox.rs:1008-1026) — the seed+commit path is the same code, so durability across the SIGKILL is the load-bearing property under test at the process tier.

### DESIGNER.observability
OBSERVABILITY — in-process receiver B (the killed source has NO admin surface; must-fix HIGH, DOR line 25).

Peer B is a REAL `MeshTransport` held BY THE TEST PROCESS, bound to B's addr, spawned via `spawn_mesh(rt.handle(), &trust, &cfg_b, None)` exactly as mesh_redelivery.rs:50-52 does. Design:

1. TOPOLOGY: source = NodeId(1) (the child bin), B = NodeId(2) (in-process). Reserve both addrs up front (reserve_udp_addr, lib.rs:473). The BOOK both sides share: `[(NodeId(1), addr_src), (NodeId(2), addr_b)]` (book() formatter, lib.rs:153). The source bin gets this via `VD_PEERS`; B gets it via its MeshConfig.peers. Both must agree so the restarted source can DIAL B by book (the mesh dials by address book — mesh_redelivery relies on this).

2. B is spawned ONCE in the test process and STAYS UP across both source boots (it is not the crash subject). Hold `(mut b_transport, _b_control)` for the whole test; keep the owning `tokio::runtime::Runtime` alive (drop tears down peer-writers — see boot_mesh_and_replay.rs:56-57 discipline).

3. WIRING SO THE RESTARTED SOURCE DIALS B: the seeded OutboxKey.peer MUST be NodeId(2) (B). On boot 2, `boot_mesh_and_replay` (lib.rs:344-354) calls `replay_outbox` which, for each retained row, checks `peers.contains_key(&key.peer)` (outbox.rs:494) — B(2) IS in VD_PEERS ⇒ routable ⇒ `send_durable(B, class, payload, Retained)` (outbox.rs:518). The mesh peer-writer for lane B dials addr_b (which the in-process B is listening on) and delivers the ReliableFrame. B's serve_connection decodes + delivers it to B's inbox.

4. DRAIN TO OBSERVE (mesh_redelivery drain pattern, mesh_redelivery.rs:56-62): after spawning boot 2, `wait_until` (poll loop with a DEADLINE, mesh_redelivery.rs:75-81) draining `b_transport.drain_inbound()` (mesh.rs:1953-1958) filtering `Inbound::Wire { bytes, .. }` (mesh_redelivery.rs:57-59) and asserting the re-driven payload byte(s) match the seed. The drain must run on the SIM/test thread (drain_inbound is sync). Deadline 30s (mesh_redelivery.rs:20).

5. ANTI-THEATER TWIN (must-fix, DOR line 97): a second test `fresh_store_re_drives_zero_rows` — boot 1 seeds NOTHING (no VD_OUTBOX_TEST_SEED), SIGKILL, boot 2 on the SAME (empty) outbox + a fresh B; assert B's inbound stays EMPTY over a bounded settle (replay_outbox returns `ReplayCounts::default()` on an empty scan, outbox.rs:482-484). This falsifies the positive test (an empty outbox re-drives 0 rows). Mirror `grant_a_recovery_assertion_fails_against_a_fresh_store` (orchestrator_crash.rs:209-260).

NOTE on incarnation across the two source boots: the seeded row carries `incarnation` = the boot-1 incarnation. Boot 2 comes up at a HIGHER incarnation (boot-counter +1); replay re-frames at the FRESH incarnation (the send path re-stamps — decode_value_payload strips the stored frame, outbox.rs:382-389, and the write path stamps new incarnation/epoch/seq). So B's receiver sees a fresh-incarnation frame at seq0 ⇒ classify_reliable Reset-then-Accept (mesh.rs:360, the R-3' A1 ladder) ⇒ delivered. This is exactly the boot_counter_crashloop GREEN mechanism (boot_counter_crashloop.rs:215-224). If the source were to come up at an EQUAL incarnation, B could DEDUP — so the boot-counter-by-1 assertion (below) is not merely a counter check, it is the reason the re-drive is OBSERVABLE at all.

### DESIGNER.boot_counter
v2 == v1 + 1; see the boot_counter field for the full read-across-crash sequence (store-test-hooks pub BootCounter::current wrapper over the private read, boot.rs:207; or a direct 24-byte le read of bytes[8..16]).

### DESIGNER.addr_discipline
ADDR DISCIPLINE — reap-then-rebind-SAME-addr so B's book entry stays valid + handle the reserve_udp_addr TOCTOU race.

The restarted source→B re-drive requires B to keep dialing/receiving on a STABLE topology. Since B dials nothing (the SOURCE dials B) and B stays up, the constraint is: the RESTARTED SOURCE must rebind the SAME addr it had in boot 1, because B's peer book (and the source's own VD_PEERS + VD_BIND) is fixed at test setup. So follow boot_counter_crashloop, NOT orchestrator_crash:
- orchestrator_crash.rs:128-136 restarts on a FRESH addr (orch2/admin2) — WRONG for D (B's book would go stale).
- boot_counter_crashloop.rs:170-172 does the CORRECT pattern: `cluster.kill_and_reap("vd-shard")` THEN `cluster.push("vd-shard-restart", spawn_shard())` with the SAME `shard_env`/addr. kill_and_reap (lib.rs:727-736) SIGKILLs + `wait()`s (reaps) + REMOVES from the reap set so the bind port frees for the immediate restart.

SEQUENCE for D:
1. `reserve_udp_addr()` (lib.rs:473) for both addr_src and addr_b ONCE at setup. Bind B immediately (spawn_mesh binds addr_b) — this CLOSES the TOCTOU window for B (B now owns the port). 
2. addr_src is reserved-then-dropped (TOCTOU-racy, lib.rs:471 doc). Boot-1 source binds it. On kill_and_reap the source's UDP port is released by the `wait()` reap (the OS frees the socket on process death + reap). 
3. Boot 2 rebinds addr_src — this is the SAME reserve_udp_addr TOCTOU race the whole test tier lives with, but the window is small: reap completes (port freed) → immediate respawn. To make it robust: use `Cluster` (lib.rs:685) so boot-1 source is RAII-reaped on panic; call `cluster.kill_and_reap(SOURCE_NAME)` (synchronous reap = port freed) then `cluster.push(SOURCE_RESTART_NAME, spawn_node(...))` on the SAME env. The `wait()` inside kill_and_reap guarantees the kernel has torn down the boot-1 socket before the respawn binds — the same guarantee boot_counter_crashloop relies on and which is green in CI.
4. Serial `--test-threads=1` (justfile:91 precedent) — the test spawns a real process + binds real UDP ports; concurrent crash tests contend for ports/CPU (boot_counter_crashloop.rs / orch-crash comment justfile:88-89).

RAII SAFETY: use the KillOnDrop pattern (orchestrator_crash.rs:37-45) for boot-1 source OR the Cluster reap set (lib.rs:748-755 Drop kills+reaps all) so a panic between spawn and the explicit kill never leaks the source child holding addr_src. B's `_b_control` + the runtime are held to end-of-test (dropping either kills B). Clean up the outbox file + boot dir + trust dir on exit (TempPaths RAII, boot_counter_crashloop.rs:35-46).

NOTE: because B never restarts and the SOURCE always rebinds the same addr, there is no second TOCTOU race beyond the single source-rebind — strictly simpler than orchestrator_crash (which reserved fresh addrs) and identical in shape to boot_counter_crashloop (proven green).

### DESIGNER.profile_merge
PROFILE MERGE — %p-%m%c continuous-mode wiring + add the test to orch-crash + orch-crash-cov.

The SIGKILLed child loses its atexit profile flush; only %c CONTINUOUS mode (mmapped profraw updated in place) captures a killed child's counters. The existing orch-crash-cov recipe (justfile:111-119) already sets this up:
```
source <(cargo +TOOLCHAIN llvm-cov show-env --export-prefix --branch)
export LLVM_PROFILE_FILE="$(dirname "$LLVM_PROFILE_FILE")/vd-orchcrash-%p-%m%c.profraw"
```
`%p` (pid) + `%m` (binary signature) keep the two SOURCE boots (+ in-process B, which is the test process itself) distinct; `%c` = continuous mode. The child INHERITS LLVM_PROFILE_FILE because `spawn_node` (lib.rs:640-650) forwards the parent env (it only sets the explicit VD_* pairs via cmd.env, and Command inherits the rest of the parent environment by default) — same mechanism orch-crash-cov relies on (justfile:109-110 comment).

CHANGES:
1. `orch-crash` recipe (justfile:90-93): add a line `cargo test -p vd-bins --features store-test-hooks --test outbox_sigkill_restart -- --test-threads=1`. The new test file is `#![cfg(feature="store-test-hooks")]` (like orchestrator_crash.rs:18) and needs the purpose-built test node bin, which is also gated `store-test-hooks` + `required-features` — so `--features store-test-hooks` builds both the test AND the bin (Cargo builds required-features bins for the test target when the feature is on). Serial (--test-threads=1) for the port/process reasons above.
2. `orch-crash-cov` recipe (justfile:111-119): add a matching `cargo +TOOLCHAIN llvm-cov --no-report --branch -p vd-bins --features store-test-hooks --test outbox_sigkill_restart -- --test-threads=1` block under the same LLVM_PROFILE_FILE export, so the killed source-node child's outbox/boot/mesh coverage is captured %p-%m%c. This accumulates into the Tier-B floor.
3. `gate` (justfile:149) already runs `orch-crash` + `coverage`, so adding to orch-crash wires it into the pre-merge gate with no gate edit. `coverage` (justfile:55) = coverage-fast + coverage-io-prod + coverage-io-prod-hooks; the store-test-hooks surface (seed_reliable_row, BootCounter::current) is floored by coverage-io-prod-hooks (justfile:48-49). The seed method + BootCounter wrapper EXECUTE under this process-tier test — ratcheting the io-prod Tier-B floor UP.

TIER-B HONESTY: io-prod/bins are Tier-B (ratcheted floor, honestly never 100%). A SIGKILL can lose the final counter flush of any code that ran after the last %c mmap sync (a genuine floor, not a bug). The new test ratchets the floor up as its store-test-hooks surface executes at the process tier; it never claims 100%. Document this in the recipe comment (mirror justfile:107-110).

### open_questions
- NodeOutbox has NO local NodeId field (outbox.rs:221-224) — seed_reliable_row must take `from: NodeId` as a param (the stored frame's `from` is never read on replay: replay re-frames from decode_value_payload's PAYLOAD only, outbox.rs:518, and decode_value_payload strips the header, outbox.rs:385). Recommend a `from` param for honesty; NodeId(local) from VD_NODE_ID in the bin. Alternatively stamp NodeId(0) — functionally inert but less honest. DECIDE: add `from` param.
- The purpose-built bin's boot-1 path must NOT call boot_mesh_and_replay (that would replay the just-seeded row to a not-yet-existent lane — B is bound but the SOURCE lane to B on boot 1 has nothing to send and replay would try to re-drive+gc the seed we WANT to survive). Boot-1 must ONLY open_node_outbox + seed + confirm + idle (no spawn_mesh, no replay). Boot-2 calls boot_mesh_and_replay. So the bin branches on VD_OUTBOX_TEST_SEED presence: SET ⇒ seed-and-idle; UNSET ⇒ boot_mesh_and_replay-and-tick. CONFIRM this branch is the intended contract (it is the only way the seeded row survives to boot 2).
- spawn_mesh guards peer count <= OUTBOX_WRITER_CHANNEL_DEPTH (256) when an outbox is wired (mesh.rs:824-831) — a 2-peer book passes trivially. No action, but noted so a future larger-roster D-variant knows the ceiling.
- boot_mesh_and_replay resolves incarnation via resolve_process_incarnation which, with VD_BOOT_STATE_DIR set + no VD_PROCESS_INCARNATION, uses the durable counter (lib.rs:239-247) — but the shard/gateway bins today get VD_PROCESS_INCARNATION from common_env (lib.rs:365). The purpose-built bin must NOT receive VD_PROCESS_INCARNATION (else the counter is bypassed, lib.rs:225-227). CONFIRM the test builds the source env WITHOUT common_env's VD_PROCESS_INCARNATION (build the env by hand, like boot_mesh_and_replay.rs:30-41, adding VD_BOOT_STATE_DIR instead).
- epoch for the seeded frame: recommend u32::MAX to match the write-path re-stamp the A-proptest already assumes (outbox.rs:1430 note, framed_row uses epoch=u32::MAX outbox.rs:932). Since replay re-frames anyway, the stored epoch is inert — but pin it to u32::MAX for consistency with the existing frame builders. LOW priority.
- Timing of the boot-2 re-drive vs. B readiness: B is up before boot 2 spawns, but the source→B QUIC dial + reliable delivery is async. The test must poll-drain B with a DEADLINE (mesh_redelivery.rs:75-81 wait_until, 30s) — NOT a fixed sleep. This is standard for the mesh tests; confirm the DEADLINE budget (30s) is generous enough for a cold dial + replay fence (boot_mesh_and_replay's replay fences on DURABILITY not delivery, lib.rs:303-306, so it does not hang on the dial — the row is re-driven then B receives asynchronously).
- Cleanup ordering on the crash path: the boot-1 source is SIGKILLed while holding the outbox redb file open; the reap (wait()) releases the OS file lock, and boot 2 reopens the SAME file (RedbStore::open = Database::create opens-or-creates, store.rs:551). CONFIRM redb tolerates a reopen after a SIGKILL of the prior writer (the orchestrator_crash test proves exactly this for Store A — a kill-9 then reopen recovers, orchestrator_crash.rs:148-159; NodeOutbox uses the same RedbStore, so it inherits the guarantee). No new risk, but it is THE load-bearing durability property the test exists to prove at the process tier.