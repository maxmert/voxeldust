Confirmed all key claims. `RedbStore::open` uses redb's `Database::create` — no reusable std atomic-write-file helper exists in io-prod (reviewer 3's "no existing helper to lean on" is accurate; the counter must be std-only new code). identity_persistence.md:122 confirms the durable-outbox/retain-until-acked sketch and §3.3's `(correlation_id, step_id)` idempotency the outbox would compose with.

Now I have everything I need to adjudicate. Let me verify one final claim — reviewer 1's HIGH on the genesis/corrupt conflation, by checking the store.rs fail-loud discipline reference (lines 468-481 "refusing to degrade to genesis"). That's already confirmed above (store.rs:473-479). And the L4 on_ack guard at mesh.rs:540-541.

I have full grounding. Let me deliver the final vetted design.

---

# R-6 — FINAL VETTED, IMPLEMENT-READY DESIGN (adjudicated against the code)

## Adjudication summary

I read the load-bearing code. Verdict corrections vs the three reviewers:

- **The core M3 boot-counter (R-6a) is SOUND** — every load-bearing claim checks out: `classify_reliable` A1 resets dedup only on strictly-greater incarnation (mesh.rs:362, `>`), an EQUAL incarnation + restarted seq0 falls to `seq<=hw ⇒ Dedup` (mesh.rs:418) = the exact silent loss; the durable strictly-monotone counter is the correct cure; `launch_incarnation()` (bins/lib.rs:173) is genuinely wall-clock-ms and rewind-unsafe. No frozen-seam touch (incarnation is a `MeshConfig` field, mesh.rs:79, stamped on the io-prod-private `ReliableFrame`, lib.rs:70). No HR3 fork (all three bins use the identical `env.parse_or("VD_PROCESS_INCARNATION", 0)?` 5th arg — orchestrator.rs:55, gateway.rs:31, shard.rs:31). No new dep (no reusable atomic-file helper exists; `RedbStore::open` uses `Database::create` — std-only is the right call).

- **Reviewer 2's CRITICAL is UPHELD and is the headline correction** — the design's "near-term k3d CrashLoop/reschedule proof gated only on M3" is WRONG. Confirmed against code: `peer_book` parses a literal `SocketAddr` (runtime.rs:108), NO DNS/service-name resolution; DEFERRED.md D-12:1712 is **explicitly BINDING**: *"the deployment story is loopback-only until CA-1 lands."* Two k3d pods literally cannot address each other. D-6 #1's AwaitAdopt trigger is gated on **M3 AND L5** (DEFERRED.md:955-957), and the `kubectl delete pod` reschedule test exercises L5 behavior the transport provably lacks (`PeerWriter.addr` captured once, mesh.rs:1081). Admin `/metrics` has no auth (D-13, loopback-only). **The k3d manifests + real-cloud proof must be DE-SCOPED out of R-6** behind a CA-1+L5+M3 triple.

- **The L4 HIGH (both R1 and R2) is UPHELD and under-stated in the original design** — `ack_egress` does NOT mint a uniform scalar "by construction"; it **overwrites** `incarnation = st.incarnation` every loop iteration (mesh.rs:1049), so the emitted `AckFrame.incarnation` is the last-sorted `(peer,class)`'s value, then the sender fan-out applies that ONE scalar to every entry's `on_ack` (mesh.rs:1127). Correct today only because all incarnations are equal. The per-class move removes a **latent silent-misfire**, not a theoretical one.

- **Reviewer 1's HIGH (genesis-vs-corrupt conflation) is UPHELD** — the design's "torn/foreign file ⇒ conservatively low from genesis" is a **silent monotonicity break** on a present-but-unreadable file, and it directly contradicts the codebase's own fail-loud discipline (store.rs:473-479 panics rather than "degrade to genesis").

- **The durable outbox is correctly DEFERRED** (R-6d), and landing M3 alone leaves DEFERRED.md D-6 #1 honestly 🟧.

Net: **FIX_BEFORE_IMPL on the scoping (de-scope k3d), then R-6a+R-6c are SOUND_TO_IMPLEMENT.** The first sub-slice to implement is **R-6a** (the boot-counter core), proven by the **loopback** CrashLoop test — no pod network required.

---

## (a) Resolved sub-slicing (ordered, each independently gate-able + commit-able)

| Slice | Scope | Gate | Deploy-blocking? |
|---|---|---|---|
| **R-6a** | The M3 boot-counter mechanism: a std-only `BootCounter` sidecar in `vd-io-prod`, the shared `reject_ephemeral_path` helper (lifted from orchestrator.rs:141-181) + a durable-root **ALLOW-list**, `bins` `boot_incarnation(state_dir)` → `MeshConfig::process_incarnation` with the `VD_BOOT_STATE_DIR`-else-`VD_PROCESS_INCARNATION` precedence. Unit tests: read-increment-fsync, monotone-across-crash, rewind-survival, **present-but-corrupt ⇒ fail-loud**, ephemeral-reject + allow-list. | `cargo test -p vd-io-prod` + `just gate` | **YES — the loopback-CrashLoop precondition** |
| **R-6c** | **L4**: `incarnation` moves from `AckFrame` into `AckEntry` (per-class); `ack_egress` stamps per-entry (mesh.rs:1050); the sender fan-out reads `e.incarnation` (mesh.rs:1127). Land WITH R-6a — cheap, wire-local, removes a latent silent-misfire. | `cargo test -p vd-io-prod` | No (correctness-forward) |
| **R-6b** | The **loopback** CrashLoop e2e proof: `crates/bins/tests/boot_counter_crashloop.rs` (Tier-B) — fixed `VD_BOOT_STATE_DIR` per node, SIGKILL + sub-second restart, assert via `/metrics` that `dedup_drop`/`stale_incarnation_drop` do NOT climb + `reliable_acked` advances, with a **fixed-incarnation red control**. Ledgers **M3 🟩 on the loopback strength**. | new Tier-B process test | **YES — the M3 proof (loopback)** |
| **~~k3d~~** | **DE-SCOPED from R-6.** The k3d StatefulSet+PVC manifests + real-cloud CrashLoop/reschedule proof are a **CA-1 + L5 + M3 combined slice** (two pods cannot address each other + reschedule needs addr-reread + admin needs auth). Do NOT author polished manifests in R-6. | — | gated on CA-1+L5 |
| **R-6d** | **Durable outbox** — retain-until-acked backed by the `Store` seam, composed with the RAM `ReliableLaneSender.retry`. Its own gate-able prerequisite: **gateway/shard open a per-node `RedbStore`** (they have none today; only orchestrator.rs:226). Acceptance gate: a both-ends-restart replay-interleave proptest. Closes D-6 #1's AwaitAdopt source-crash window. | new io-prod tests + a saga-recovery process test + the proptest | **NO — follow-slice** |

**R-6a is the sole near-term deploy precondition; R-6b is its loopback proof.** L4 (R-6c) lands with R-6a. k3d and the durable outbox both leave the tree honest and are correctly owed in DEFERRED.md.

---

## (b) THE EXACT M3 boot-counter mechanism

**Where the u64 lives:** a dedicated tiny durable **sidecar file** (`$VD_BOOT_STATE_DIR/boot.counter`), **NOT** the redb `Store`, **NOT** a new `Store` key family.

Decisive rationale (grounded): gateway.rs and shard.rs open **no** `Store` today — only orchestrator.rs:226 does. Reusing redb forces every node kind to open a database (the whole off-tick writer thread + `DurabilityHandle` + block-on-prior) for one u64, and couples the incarnation read to the redb open path. HR3 wants **one mechanism for every node kind**; the common denominator is "a tiny file on a persistent path," not "a redb database." A dedicated `BootCounter` is the DRY common mechanism.

**File format** — a fixed 24-byte record so a torn write is *detectable*:
```
[0..8]   magic     = b"VDBOOT\0\0"   (a FOREIGN file ⇒ fail-loud; see below)
[8..16]  counter    u64 LE
[16..24] checksum   u64 LE           (std DefaultHasher of magic||counter; NO new dep)
```
All layout values (`BOOT_MAGIC`, `RECORD_LEN=24`, field offsets, `BOOT_COUNTER_NAME="boot.counter"`) are **named consts** in `io-prod/src/boot.rs` with rationale comments (matching store.rs:45-56 discipline) — **no bare literal at any use site** (HR: no magic numbers). The two new env keys (`VD_BOOT_STATE_DIR`, `VD_BOOT_STATE_EPHEMERAL_OK`) register in bins/src/lib.rs next to the `VD_*` contract (bump the "23-key" doc count to 25).

**The read-increment-write-fsync-on-boot flow** (`BootCounter::increment_on_boot(path) -> Result<u64, BootCounterError>`) — the ONE function every bin calls at boot, before `spawn_mesh`:

1. **Read** the file, classify into **THREE distinct cases** (folding R1's HIGH):
   - **ABSENT** ⇒ true genesis, `prev = 0`. Safe.
   - **PRESENT, magic-wrong OR checksum-fails** ⇒ **`Err(BootCounterError::Corrupt)` — FAIL LOUD, refuse to boot.** A present-but-unreadable durable counter is data loss; silently resetting to 0 would hand out a LOWER incarnation than a peer's surviving ledger ⇒ `classify_reliable` A2 StaleIncarnation-drops ALL the node's traffic (mesh.rs:372) — the exact k8s hazard M3 exists to kill, reintroduced by a torn REAL file. This mirrors the store's own discipline (store.rs:473-479 "refusing to degrade to genesis"). Recovery is an operator action, never a silent low reset.
   - **PRESENT, valid** ⇒ `prev = counter`.
2. **`next = prev.checked_add(1)`** — `checked_add` fails loud (`BootCounterError::Overflow`) rather than wrap to 0 (a wrap would be a silent monotonicity break; a u64 boot counter never realistically overflows).
3. **Atomic-durable replace** (all three steps checked — never `let _ =`, folding R1's MEDIUM):
   a. write the 24-byte record to `path.tmp`;
   b. **`File::sync_all()` on the temp file** (fsync the DATA) — propagate the error;
   c. **`std::fs::rename(tmp, path)`** (atomic on POSIX within a filesystem) — propagate;
   d. **fsync the containing DIRECTORY** (`File::open(dir)?.sync_all()`) so the rename itself is durable — propagate. This is load-bearing on k3d local-path (see risks); a dropped dir-fsync re-opens the host-crash revert window.
   Any of b/c/d erroring ⇒ typed error, **fail loud** — a boot that cannot durably persist its incarnation must not proceed to stamp frames.
4. **Return `next`** — the bin uses it as `process_incarnation`. The value is durable **before** `spawn_mesh`, so a crash after stamping frames can never re-use it.

**ATOMIC:** write-temp→fsync→rename→fsync-dir means `path` is *only ever* the old value or the fully-written new value; a crash mid-`write` corrupts only `.tmp` (ignored next boot; the real `path` is untouched).

**MONOTONE + crash-during-increment safety:**
- **Crash after reading `prev`, before the rename completes** ⇒ next boot re-reads the same `prev`, hands out the same `next = prev+1` again. **SAFE** — the crash was *before* `spawn_mesh`, so **no frame ever carried that incarnation**; a re-used-but-never-wired value is indistinguishable from a first use. Invariant: *a value is durable in the file before any frame carries it.*
- **Crash after the rename completes** ⇒ next boot reads the new value, hands out `+1`. **Skip-forward is impossible** (exactly one higher), re-use is impossible (the durable value moved). Steady case: **strictly monotone increase every real boot.**
- So the counter either **re-uses a never-wired value (safe)** or **advances by exactly one (safe)** — never skips, never re-uses a wired value. Exactly the strictly-higher property `classify_reliable`'s A1 ladder needs (mesh.rs:362): a restarted sender always presents a **strictly higher** incarnation than any surviving peer ledger holds ⇒ A1 resets to not-primed ⇒ the restarted seq0.. primes + is Accepted, never silent-Deduped.

**A higher incarnation never wrongly resets a LIVE peer:** A1 is per-`(peer,class)` and fires only on a frame that actually carries a higher incarnation (mesh.rs:362, per-key `RecvState`). A non-restarted peer keeps stamping its same value; its lane never resets. Only the restarted node's outbound lanes carry the bumped value.

**Survives a wall-clock REWIND:** the counter is derived only from the file's prior value + 1 — **no `SystemTime::now()` anywhere in the path**. NTP stepping backward cannot lower it. (The whole point vs `launch_incarnation()`.)

**Threading into `MeshConfig` (prod) while dev/tests keep the cheap path.** New bins helper:
```rust
// crates/bins/src/lib.rs — replaces launch_incarnation() for PROD; dev keeps the clock path.
pub fn boot_incarnation(state_dir: &Path) -> Result<u64, BootCounterError> {
    vd_io_prod::boot::BootCounter::increment_on_boot(&state_dir.join(BOOT_COUNTER_NAME))
}
```
**Precedence (folding R2's MEDIUM on forward-compat):** each bin resolves, in order:
1. `VD_PROCESS_INCARNATION` if explicitly set (the **future orchestrator-issued** path + the dev/test/loopback path — wins);
2. else `VD_BOOT_STATE_DIR` present ⇒ the durable self-counter;
3. else ⇒ **fail loud unless `VD_BOOT_STATE_EPHEMERAL_OK=1`** (folding R1's LOW: a prod pod that mounts a PVC but forgets `VD_BOOT_STATE_DIR` must NOT silently revert to wall-clock-ms — the unsafe clock path is **opt-IN, never a silent default**).

This keeps `vd-devcluster`/`process_parity` on the cheap `launch_incarnation()` path (they set `VD_PROCESS_INCARNATION` today) and lets the future provisioning slice supersede cleanly (the orchestrator hands a value instead of the file) at **zero rework** — one mechanism, opt-in by env presence, exactly the shape `VD_STORE_PATH` already uses.

**k3d per-pod persistence + the fail-loud guard (folding R2's MEDIUM — the deny-list is a shortcut here).** The counter lives on a per-pod PVC (StatefulSet `volumeClaimTemplate`). The guard is **NOT** the orchestrator's prefix DENY-list (orchestrator.rs:143-147 documents it as inadequate — a k8s `emptyDir` mounts under `/var/lib/kubelet/...`, not `/tmp`, and slips through, silently making the counter non-durable). Instead R-6a **lands the owed durable-root ALLOW-list** the orchestrator comment already owes: `reject_ephemeral_path(path, durable_root_env, ephemeral_ok_env) -> Result<()>` — the path must canonicalize UNDER an explicitly-declared `VD_BOOT_DURABLE_ROOT` (the PVC mount) or refuse to boot, with `VD_BOOT_STATE_EPHEMERAL_OK=1` the strict-parse dev escape (reusing the orchestrator's exact `1/true/yes | 0/false/no | error` ladder, factored so it's not a second copy). Lift the block from **orchestrator.rs:136-188** (NOT store.rs — R3's LOW correctly caught the design's mis-attribution) into the shared helper; both `VD_STORE_PATH` and `VD_BOOT_STATE_DIR` call it. The empirical durability self-check (R-6b's "boot.counter incremented N times across restarts") is promoted from a runbook step to a **required test-code assertion**.

---

## (c) Frozen seam + HR3 — CONFIRMED

**No frozen `sim::io` seam touch.** Incarnation is stamped on the io-prod-private `ReliableFrame` (lib.rs:70), sourced from `MeshConfig.process_incarnation` (mesh.rs:79) — a bin-level config. `sim::io::{Transport, Store, MsgClass, Inbound}` are untouched; sim/node never see incarnation or the boot counter. The boot counter is deliberately **not** a `Store`. **No HR3 fork:** `BootCounter` is one type, zero match-on-node-kind; all three bins call the identical `boot_incarnation(state_dir)`. Shard *kind* never enters it.

---

## (d) Durable outbox — scope + design

**Scope: NOT needed for the first (loopback) M3 proof; a follow-slice (R-6d).** R-6b proves the M3 CrashLoop-restart-no-silent-loss property — that is what M3 unlocks now. The durable outbox closes a **narrower, pre-existing** window (a source crashing inside `BatchHandoff::AwaitAdopt` before its `TransientBatch` reached the dest — DEFERRED.md D-6 #1) that is (i) gated behind M3+L5 for its confirm-dead self-promote trigger, and (ii) not introduced by R-6. Landing M3 alone leaves D-6 #1 honestly 🟧.

**Design when it lands (R-6d):** retain-until-acked backed by the `Store` seam, **composed with — not duplicating** — the RAM `ReliableLaneSender.retry` (mesh.rs:470). The RAM `retry` map stays the hot path; the durable outbox is a **write-through mirror keyed `(peer, class, incarnation, seq) → framed bytes`** persisted in the *same* `commit()` txn as the producing saga step (mirroring identity_persistence.md §3.3's `(correlation_id, step_id)` idempotency — the outbox and `applied_steps` share the redb barrier). On boot, `BootCounter` bumps the incarnation, then the outbox replays every un-acked entry **under a FRESH lane (epoch 0, ascending seq re-based from base)** — the receiver's A1 reset + seq0 prime gate (mesh.rs:394) fires deterministically, so replayed frames are Accepted contiguously, never re-injected out of order (folding R1's LOW on the both-ends-restart reorder). `ReliableLaneSender` gains a `durable: Option<&dyn OutboxSink>` that `assign_and_retain` write-throughs and `on_ack` deletes-through — ONE FSM, a durable backend behind the same seam.

**R-6d carries two first-class hidden costs** (folding R3's LOW): (i) **gateway/shard open a per-node `RedbStore`** (they have none today) — its own gate-able step, which also lets their store path reuse the SAME `reject_ephemeral_path` helper (more DRY leverage); (ii) a **both-ends-restart replay-interleave-under-bumped-epoch proptest** is a NAMED R-6d acceptance gate, not a "corner I'd want hammered."

---

## (e) L4 AckFrame resolution — MOVE `incarnation` INTO `AckEntry` (hard cutover, greenfield)

The invariant *holds today* (`on_ack` guard at mesh.rs:540-541 `ack_incarnation != self.incarnation`), but the code does NOT mint a uniform scalar "by construction" — `ack_egress` **overwrites** `incarnation = st.incarnation` every loop iteration (mesh.rs:1049), emitting whatever `(peer,class)` sorted LAST, and the fan-out applies that ONE scalar to every entry (mesh.rs:1127). Correct only because all incarnations are equal. M3 makes sender restart first-class, so **eliminate the invariant dependency now** (per DEFERRED.md:1294-1296's ledgered acceptance item):

- **Remove** `AckFrame.incarnation` (lib.rs:106); **add** `incarnation: u64` to `AckEntry` (lib.rs:113).
- `ack_egress` (mesh.rs:1050) stamps each entry from *that* `(peer,class)`'s `st.incarnation` (drop the loop-overwritten scalar at 1039/1049).
- The sender fan-out (mesh.rs:1127) reads `e.incarnation` per entry: `lane.on_ack(e.incarnation, e.epoch, e.ack_through)`.
- **Hard cutover** (greenfield, single deploy — no rolling version mix pre-first-deploy; R2's "StatefulSet RollingUpdate mixes versions" concern is real only once the k3d slice lands, and that slice is CA-1+L5-gated anyway). Do NOT keep a deprecated-0 scalar the sender still reads (on_ack against a live lane at incarnation=N with incarnation=0 retires nothing ⇒ silent send stall via `AssignReject::BufferFull`). `on_ack`'s tests already take a per-call `ack_incarnation` (mesh.rs:2212+), so the FSM is ready — only the frame decode + fan-out plumbing changes.
- **New test:** two classes on one connection at DIFFERENT incarnations retire independently (the invariant-elimination proof).

If the k3d rolling-update path ever needs mixed-version acks, route the grow through the reserved `codec_flags` bit (lib.rs:47-50) — deferred to the CA-1+L5 k3d slice, not R-6.

---

## (f) The CrashLoop e2e test + PASS assertion (LOOPBACK — no pod network)

**What M3 unlocks:** a CrashLoop proof — restart a node repeatedly, sub-second, at the same wall-clock ms, and assert its reliable traffic is never silent-Deduped/Stale-dropped.

**R-6b — `crates/bins/tests/boot_counter_crashloop.rs` (Tier-B, mirrors `dev_cluster_smoke.rs`):**
- Bring up a 3-node **loopback** cluster with a **fixed** `VD_BOOT_STATE_DIR` per node (a scratch dir NOT reaped between restarts).
- SIGKILL the shard; restart it within the same wall-clock ms (tight restart loop).
- **PASS assertion** (via the orchestrator `/metrics` R-4d `MeshStats` surface, orchestrator.rs:261): after the restart, `dedup_drop` and `stale_incarnation_drop` for the shard's lanes do **NOT** climb on the restarted seq0.. traffic, and `reliable_acked` advances (clear-on-ack fires). Additionally assert `boot.counter` on disk **incremented** across restarts (the empirical dir-fsync-honesty check, now a required code assertion).
- **RED control:** the same test with `VD_PROCESS_INCARNATION` forced to a fixed constant (the wall-clock-equal hazard) must show `dedup_drop`/`stale_incarnation_drop` **climbing** — proving the durable counter demonstrably fixes what the fixed version breaks. This is the honest e2e (real QUIC mesh + real classify ladder across a real process kill), not theater.

This test needs **no** Docker/k3d and runs in CI (the harness has no Docker). **On its strength, ledger M3 🟩.**

**k3d (DE-SCOPED, gated on CA-1+L5+M3):** author NEW manifests under a fresh `deploy/` (not `scripts/`, which is the loopback launcher — note: no k3d manifests exist in this worktree today; `scripts/dev-cluster.sh:13` is explicitly "needs no Docker/kubectl", R3's LOW confirmed). The StatefulSet+PVC + real-cloud CrashLoop/reschedule proof land only when CA-1 (reply-on-connection, so two pods can address each other) + L5 (`PeerWriter` re-reads `addr`) + D-13 (admin auth) land. DEFERRED.md records: M3 alone unblocks the LOOPBACK CrashLoop hazard; the k3d proof is a CA-1+L5+M3 triple.

---

## (g) Residual risks (least-sure, flagged for the user)

1. **PVC-per-pod self-count vs orchestrator-issued incarnation.** The self-count file is correct + DRY for the fixed P1 roster, but the end goal (hundreds of dynamically-provisioned shards, NodeId reuse) fits the StatefulSet-sticky-PVC model poorly — a re-provisioned shard with a fresh PVC re-genesises at 1, and under NodeId reuse could re-mint a LOW incarnation below a peer's surviving ledger. **The R-6a precedence (`VD_PROCESS_INCARNATION` wins) already makes the orchestrator-issued path a zero-rework supersede** — but whether the eventual model is self-count-PVC or orchestrator-issued is a **jointly-investigated choice** (memory: no-unilateral-adoption). Flag before investing in polished k3d manifests (which are CA-1+L5-gated regardless).
2. **fsync honesty on the k3d local-path provisioner** (hostPath under Docker/overlayfs). The atomic-replace dance assumes honest `sync_all` on the mount. The allow-list catches tmpfs by declaration; a subtly-non-durable bind mount surviving a *host* power-loss (not a pod restart) is the residual. R-6b's empirical increment-across-restarts assertion is the check; a lying host FS is out of scope for the loopback proof.
3. **PVC-loss re-genesis vs a peer's surviving ledger** (R3's LOW) — narrower than a pod restart. Belt-and-suspenders option worth a conscious pick in R-6a: seed genesis-only from a wall-clock floor (`max(1, now_ms)` as the FIRST-EVER value) so a re-provisioned node still exceeds a wall-clock-era ledger, keeping steady-state purely counter-driven. Name it in DEFERRED.md.
4. **Both-ends-simultaneous-restart replay reorder** (R-6d only) — cured by the fresh-lane/seq0-rebase replay design above; gated behind the named proptest before shipping the outbox.

---

## VERDICT

**FIX_BEFORE_IMPL** on the original design's scoping (the k3d CrashLoop/reschedule proof is NOT M3-gated — it is blocked by CA-1 + L5 + D-13, all BINDING per DEFERRED.md D-12:1712; de-scope the manifests), and fold the four upheld findings: (1) present-but-corrupt counter ⇒ **fail loud, never reset to 0**; (2) checked, propagated fsync/rename/dir-fsync; (3) L4 is a real latent-misfire fix (`ack_egress` is last-class-wins, not uniform); (4) the ephemeral guard for the boot path must be a **durable-root ALLOW-list**, not the inadequate prefix deny-list. **With those folded, the M3 boot-counter core (R-6a) + L4 (R-6c) are SOUND_TO_IMPLEMENT.**

**Recommended FIRST sub-slice: R-6a** — the std-only `BootCounter` in `crates/io-prod/src/boot.rs` (three-way read classification with fail-loud-on-corrupt, checked atomic-durable replace, the shared `reject_ephemeral_path` durable-root allow-list lifted from orchestrator.rs:136-188), plus the `bins` `boot_incarnation` + `VD_BOOT_STATE_DIR`-precedence plumbing into the three `MeshConfig::new` calls. Land **R-6c (L4)** with it. Prove both with the **loopback** `boot_counter_crashloop.rs` (R-6b). No pod network, no new dependency, no frozen-seam change.

**Files the implementer touches:** new `crates/io-prod/src/boot.rs` (`BootCounter`, `BootCounterError`, named layout consts, std-only); the ephemeral/allow-list block lifted from **orchestrator.rs:136-188** into a shared io-prod helper (both `VD_STORE_PATH` and `VD_BOOT_STATE_DIR` call it); `crates/io-prod/src/lib.rs:105-117` (L4 `AckEntry` grows, `AckFrame` scalar removed); `crates/io-prod/src/mesh.rs:1039-1058` (per-entry stamp) + `:1127` (per-entry `on_ack`); `crates/bins/src/lib.rs:163-192` (`boot_incarnation` + the two `VD_BOOT_*` keys, doc count 23→25); `crates/bins/src/bin/{orchestrator.rs:55, gateway.rs:31, shard.rs:31}` (the precedence resolve feeding the 5th `MeshConfig::new` arg); new `crates/bins/tests/boot_counter_crashloop.rs`. **NO k3d manifests in R-6** (deferred to the CA-1+L5+M3 slice).