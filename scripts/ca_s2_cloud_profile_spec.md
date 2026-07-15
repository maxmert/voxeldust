# Slice 2 — Cloud config profile: DESIGN (designer spec) + ATTACKS + ADJUDICATED SOUND spec

## Designer spec

All facts verified against the code. Here is the spec.

---

# SLICE 2 SPEC — The Cloud Config Profile (`VD_PROFILE=cloud`)

## 0. Grounding — what actually boots green today (verified)

- The D-3 gate is exactly as stated: `crates/sim/src/directory.rs:106` `let active = self.lease_renew_interval_ticks != 0;` and every ordering check is `active & (...)` at `directory.rs:110,118,124`. `DirectoryTuning::default()` (`directory.rs:79-88`) ships `lease_renew_interval_ticks: 0` and `reaper_interval_ticks: 0` → the whole system is inert and `validate()` (`directory.rs:100`) returns `Ok` vacuously.
- The orchestrator reads the D-3 knobs with `parse_or(..., d3.<default>)` (`crates/bins/src/bin/orchestrator.rs:94-105`), and `orchestrator_env` (`crates/bins/src/lib.rs:415-432`) sets **only** `VD_LEASE_TTL` — never `VD_LEASE_RENEW_INTERVAL`/`VD_REAPER_INTERVAL`/etc. So a dev/prod cluster copied from that env boots with renew=0 → **green, zero split-brain protection**. This is the `DEFERRED.md:984` HARD precondition ("the `grace > 0` boot assertion must land WITH the split-brain lease-timing coherence `lease_ttl < grace <= lease_ttl + max` on the deploy config validation — NOT a half-measure now").
- Node self-fence: `shard.rs:49` and `gateway.rs:39` both `parse_or("VD_SELF_FENCE_GRACE", 0)` then call `validate_self_fence_cadence(grace, recheck)` (`directory.rs:211`), which returns `Ok` immediately on `grace==0` (`directory.rs:212-214`). Inert.
- Footgun escape hatches, all `parse_bool_env` (fail-closed default false, `lib.rs:232`): `VD_STORE_EPHEMERAL_OK` (`orchestrator.rs:158`), `VD_BOOT_STATE_EPHEMERAL_OK` (`lib.rs:266`), `VD_OUTBOX_EPHEMERAL_OK` (`lib.rs:324`). Dev auth key: `DEV_AUTH_SEED = [0x42; 32]` (`lib.rs:76`) → `dev_auth_pubkey_hex()` (`lib.rs:80`) fed into `VD_AUTH_PUBKEY` by `gateway_env` (`lib.rs:452`), decoded by `env.hex32("VD_AUTH_PUBKEY")` (`gateway.rs:53`, `runtime.rs:138`).
- `VD_STORE_DURABLE_ROOT` is the allow-list root for `check_durable_path` (`crates/io-prod/src/boot.rs:120`), read at `orchestrator.rs:153`, `lib.rs:319`. It is **optional** today (None ⇒ temp deny-list only).
- Redial backoff is **hard-coded** `min=50ms, max=5s` in `MeshConfig::new` (`crates/io-prod/src/mesh.rs:271-272`) — NOT env-plumbed. `confirm_unreachable_after_retries` default `= 3` (`mesh.rs:172,200`). `tick_hz=50` (`lib.rs:120`).
- `VD_PROFILE` does not exist anywhere in the tree (grep-confirmed).

---

## 1. THE PROFILE MECHANISM

### Decision: cloud mode SUPPLIES a derived coherent default set, and VALIDATES any override against coherence. (Recommended.)

Rationale / trade-off:

- **Supply-derived (recommended).** The single failure this slice exists to close is *silence*: renew=0 boots green. If cloud mode merely *validated* operator-supplied values, the operator must remember to set 7 correlated knobs (`VD_LEASE_RENEW_INTERVAL`, `VD_MIN_RENEWS_BEFORE_LAPSE`, `VD_SELF_FENCE_GRACE`, `VD_MAX_SELF_FENCE_GRACE`, `VD_REAPER_INTERVAL`, `VD_RECOVERY_GRACE`, plus node `VD_REALM_RECHECK`/`VD_SESSION_RECHECK`) — and forgetting `VD_LEASE_RENEW_INTERVAL` re-opens the exact inert hole (`active=false`) with a *green* boot, because `validate()` passes an inert config. "Require + validate" cannot fail loud on the one omission that matters. So cloud mode must default the set to a **live, coherent** derivation (§2) keyed off `tick_hz`, and treat any env override as an override of that live baseline — never a fall-back to inert.
- The escape hatch stays: an override that *breaks* coherence fails loud via the existing `validate()`/`validate_self_fence_cadence()`/`validate_against()` chain (already wired at `orchestrator.rs:106,122,130` and `shard.rs:50`/`gateway.rs:40`). The only NEW rule cloud mode adds is: **`lease_renew_interval_ticks == 0` (and the node `self_fence_grace == 0`, `recheck == 0`, `reaper == 0`) are REJECTED in cloud mode** — because those are "inert", which is legal in dev but is precisely the split-brain hole in cloud.

So the profile is a two-part contract: (a) *change the defaults* the `parse_or` calls fall back to, from the inert `DirectoryTuning::default()` to a live `DirectoryTuning::cloud(tick_hz)`; (b) *add an "active-required" assertion* so an operator cannot re-zero the gate.

### Where the mode lives + DRY (HR3)

Add ONE function in `vd-io-prod` (it already owns the shared boot guards — `boot::check_durable_path`, and `vd-bins` depends on it; putting it here keeps the orchestrator bin, which does *not* go through `boot_mesh_and_replay`, able to call it too):

```
// crates/io-prod/src/boot.rs (new section, beside check_durable_path)
pub enum Profile { DevTest, Cloud }
pub fn resolve_profile(env: &EnvConfig) -> Result<Profile, ...>   // VD_PROFILE: "cloud" | "dev" | "test" | absent⇒DevTest
```

- `VD_PROFILE=cloud` ⇒ `Cloud`. Absent / `dev` / `test` ⇒ `DevTest` (a present-but-unrecognized value is a LOUD error, mirroring `parse_bool_env`'s ladder at `lib.rs:238` — never fail-open into dev).
- **Preserves the dev/test/in-process path**: `DevTest` changes nothing. `DirectoryTuning::default()` stays inert, `validate()` stays vacuous, every existing in-process rig and the `process_parity` test keep booting byte-identically. Cloud is strictly additive.

### The ONE shared preflight (HR3 — a config mode, never a per-shard-kind fork)

A single `enforce_cloud_preflight` in `vd-io-prod` (`boot.rs`), called by all three bins, does the footgun rejections (§3) and returns the profile. It is a *config mode* switch, not a fork: the same three binaries run in both modes; only the resolved defaults + the assertions differ.

```
pub struct CloudPreflight<'a> { env: &'a EnvConfig, kind: NodeKindTag }
pub fn enforce_cloud_preflight(env, kind) -> Result<Profile, Box<dyn Error>>
```

Each bin calls it once, right after `EnvConfig::from_process_env()`:
- `orchestrator.rs` after line 41.
- `shard.rs` / `gateway.rs`: fold it into `vd_bins::boot_mesh_and_replay` (`lib.rs:349`), the ONE shared node boot sequence, so shard+gateway get it for free and cannot drift (HR3). `boot_mesh_and_replay` already resolves incarnation + opens the outbox; the profile preflight belongs at its top and it returns `Profile` alongside the transport.

---

## 2. THE COHERENT CLOUD D-3 VALUE SET (derived, no magic numbers)

### Stated target (the justification the set is derived FROM)

Everything is expressed as **wall-clock seconds at `tick_hz`**, so the set is one function of `tick_hz` (currently 50) — never scattered literals.

- **T_ttl = 2 s** — a partitioned owner keeps authority for at most one lease TTL after its last successful renew.
- **Renew 4× per TTL** (matches the doc hint `directory.rs:28` "Prod ≈ `lease_ttl_ticks / 4`"), `min_renews = 4` (the existing default at `directory.rs:82`, so a lease survives losing 3 consecutive renewals).
- **Self-fence detection target ≈ 3 s**: a partitioned owner hard-stops its own authority ~1 s after its lease lapses.
- **Reassign target ≈ 4 s**: the orchestrator will not reassign until 1 s after the node was guaranteed to have self-fenced.
- **Node recheck 2× per second** so the self-fence `confirmed` stamp re-arms comfortably inside the grace.

### The derived set (ticks; `s = seconds × tick_hz`, tick_hz=50)

| Field | Formula | Ticks @50Hz | Wall |
|---|---|---|---|
| `lease_ttl_ticks` | 2·hz | 100 | 2.0 s |
| `lease_renew_interval_ticks` | ttl / 4 | 25 | 0.5 s |
| `min_renews_before_lapse` | 4 | 4 | — |
| `self_fence_grace_ticks` | 3·hz | 150 | 3.0 s |
| `max_self_fence_grace_ticks` | 1·hz | 50 | 1.0 s |
| `reaper_interval_ticks` | 0.2·hz | 10 | 0.2 s |
| `recovery_grace_ticks` | 4·hz | 200 | 4.0 s |
| **node** `realm_recheck` / `session_recheck` | hz / 2 | 25 | 0.5 s |
| **node** `self_fence_grace_ticks` | = orch `self_fence_grace` | 150 | 3.0 s |
| **liveness** `n_consecutive_unreachable` | 3 (CSCALE-1; `orchestrator.rs:112`) | 3 | — |
| **liveness** `unreachable_window_ticks` | see R-4c below | **≥ 60**, pick 128 | — |
| **liveness** `retry_delay_ticks_hint` | ≈ redial spacing | 25 | 0.5 s |

Note the derived values (ttl=100, grace=150, max=50, recovery=200) happen to equal the *current struct defaults* at `directory.rs:80-86` — a reassuring cross-check — **except** renew (0→25) and reaper (0→10), which are exactly the two that make the system live. That is the whole point.

### Coherence proofs (all three validators + fence-rule-4)

**(a) `DirectoryTuning::validate` (`directory.rs:100`), with `active = true` (renew=25 ≠ 0):**
- `RenewTooSparse` (`:110`): `renew·min_renews = 25·4 = 100 ≤ ttl=100`. ✓ (Equality is the tightest safe point — the lease survives losing exactly `min_renews-1 = 3` renewals.)
- `SelfFenceWithinTtl` (`:118`): `grace=150 > ttl=100`. ✓ (self-fence only fires AFTER the lease provably lapsed.)
- `SelfFenceOutlivesReassign` (`:124`): `grace=150 ≤ ttl+max = 100+50 = 150`. ✓ (self-fence finishes at or before the reassign window opens.)

**(b) `validate_self_fence_cadence(grace, recheck)` (`directory.rs:211`), node-side, `grace=150, recheck=25`:**
- `grace != 0` ⇒ not the inert early-return. ✓
- `recheck=25 ≠ 0` ⇒ confirmation channel exists (`:215`). ✓
- `grace=150 ≥ 2·recheck = 50` (`:218`). ✓ (a healthy holder sees ≥1 affirming round-trip well inside the grace — no false self-fence.)

**(c) `LivenessTuning::validate` (`saga.rs:189`), `n=3, window, retry_hint=25`:**
- `n ≥ 1`. ✓
- `window ≥ n·retry_hint = 3·25 = 75`. Pick `window = 128` ⇒ ✓ (the coarse linear floor).

**(d) `LivenessTuning::validate_against` (`saga.rs:296`) — the R-4c geometric cross-check, the load-bearing one.** With the hard-coded redial `min=50ms, max=5s` (`mesh.rs:271-272`), `confirm_retries=N=3`, `n=3`, `tick_hz=50`:
- Run spans redial fires `[N+1, N+n) = [4, 6)`: intervals(4)+intervals(5) = 400ms + 800ms = **1200ms** → `duration_to_ticks_ceil` = `ceil(1200·50/1000)` = **60 ticks** (arithmetic verified against `backoff_series_sum`/`transport_run_spread_ticks`, `saga.rs:214,244`).
- Requirement: `window ≥ 60`. `window = 128 ≥ 60`. ✓ (comfortable 2× margin over the exact geometric spread — survives a modest future N/n bump; N=3,n=4 needs 140, so if `n` is ever raised the derivation must re-solve `window`, which is why it is DERIVED, not a literal.)

**(e) Fence-rule-4 ordering (the split-brain safety) — the timeline, in ticks after last successful renew:**

```
t=0     last successful renew (confirmed stamp set)
t=100   lease TTL lapses (ttl=100) — owner no longer has a valid lease
t=150   NODE self-fences: local_tick - confirmed > grace=150  (lease_self_fence_due, directory.rs:158)
t=150   orch reassign window OPENS: ttl + max = 100 + 50 = 150
```

Node self-fence fires at **t=150**, at or before the orchestrator's reassign window opens at **t=150**. There is no interval in which both the old owner still asserts authority AND the orchestrator has reassigned → **no split-brain**. The one-tick coincidence at t=150 is safe because self-fence is a *hard-stop of the old owner's own authority* evaluated at its local tick, and the orchestrator additionally requires the owner be **confirmed unreachable** before reassigning (fence rule 4: "ownership loss ONLY when the orchestrator CONFIRMS the owner unreachable"). Confirmation itself takes the R-4c run (≥60 ticks) to accrue after the partition — so in practice reassign is *later* than t=150; the `≤` bound is the worst-case guard, and the confirm-dead gate is the operational separator. An orchestrator outage freezes recovery (empty RAM liveness tracker confirms nobody dead — `directory.rs:42-45`), never mass-orphans. ✓

### Where the set LIVES (ONE config struct — no magic numbers)

Add ONE constructor beside `Default` in `crates/sim/src/directory.rs`:

```rust
impl DirectoryTuning {
    /// The DERIVED coherent cloud D-3 set: one function of tick_hz, satisfying validate()
    /// (renew·min_renews ≤ ttl < grace ≤ ttl+max) with the split-brain-safe fence-rule-4 ordering.
    /// Every field is a stated multiple of the tick rate (see the module doc table) — never a bare literal.
    pub const fn cloud(tick_hz: u32) -> DirectoryTuning {
        let hz = tick_hz as u64;
        DirectoryTuning {
            lease_ttl_ticks: 2 * hz,
            lease_renew_interval_ticks: hz / 2,        // ttl/4
            min_renews_before_lapse: 4,
            self_fence_grace_ticks: 3 * hz,
            max_self_fence_grace_ticks: hz,
            reaper_interval_ticks: hz / 5,
            recovery_grace_ticks: 4 * hz,
        }
    }
}
```

The node-side pair (recheck=`hz/2`, node `self_fence_grace` = orch `self_fence_grace`) is derived from the SAME struct so the two nodes and the orchestrator agree by construction (§4). The liveness cloud set (`n=3`, `window` derived to dominate the R-4c spread, `retry_hint=hz/2`) lives beside it as `LivenessTuning::cloud(tick_hz)` in `crates/sim/src/saga.rs` — because the R-4c window floor is a *derived* quantity (it must be re-solved if N or n change), it belongs with the tuning, computed once, not hand-entered.

A `const` compile-time assertion (mirroring `lib.rs:144-174`) in `directory.rs` asserts `DirectoryTuning::cloud(50).validate()`-equivalent inequalities hold at the shipped `tick_hz`, so a mis-derivation fails the BUILD, not a boot.

---

## 3. FOOTGUN ENFORCEMENT (all fail LOUD in cloud mode)

`enforce_cloud_preflight` (in `vd-io-prod::boot`), when `Profile::Cloud`:

1. **Reject the three ephemeral escapes.** For each of `VD_STORE_EPHEMERAL_OK`, `VD_BOOT_STATE_EPHEMERAL_OK`, `VD_OUTBOX_EPHEMERAL_OK`: if `parse_bool_env(env, key)? == true` → refuse to boot with a message naming the key. (These are read downstream at `orchestrator.rs:158`, `lib.rs:266`, `lib.rs:324`; the preflight vetoes them *before* those sites, so the dev escape can never bypass durability in cloud.)
2. **Require a real `VD_STORE_DURABLE_ROOT`.** In cloud, `VD_STORE_DURABLE_ROOT` must be present + non-empty (the allow-list root at `boot.rs:150`), so `check_durable_path` runs in allow-list mode (catches a k8s `emptyDir` a temp deny-list cannot, per `boot.rs:112-116`) rather than the dev deny-list. Absent/blank ⇒ refuse. (This tightens the currently-optional root at `orchestrator.rs:153`/`lib.rs:319`.) The orchestrator additionally still requires `VD_STORE_PATH` (`orchestrator.rs:141`) and each node requires `VD_BOOT_STATE_DIR` (else `resolve_process_incarnation` fails loud at `lib.rs:290` — good, but in cloud we also forbid the `VD_PROCESS_INCARNATION` explicit-override + `VD_BOOT_STATE_EPHEMERAL_OK` escapes so the durable counter path is mandatory).
3. **Reject the dev auth key.** Detection is exact and needs no new secret handling: the dev pubkey is deterministic. Compute it once via `SigningKey::from_bytes(&DEV_AUTH_SEED).verifying_key().to_bytes()` (this is what `vd_bins::dev_auth_pubkey_hex()` at `lib.rs:80` already does). In the gateway's cloud preflight, decode `VD_AUTH_PUBKEY` via `env.hex32("VD_AUTH_PUBKEY")` (`runtime.rs:138`) and if the 32 bytes `== DEV_AUTH_PUBKEY_BYTES` → refuse. To keep `vd-io-prod` free of the `ed25519-dalek` dev-seed dependency (the constant lives in `vd-bins`, `lib.rs:76`), expose the derived 32-byte pubkey as a `pub const DEV_AUTH_PUBKEY_BYTES: [u8;32]` in `vd-bins` (computed once) and do the dev-key veto in the gateway's slice of the preflight (gateway is the only kind that reads `VD_AUTH_PUBKEY`, `gateway.rs:53`). Equivalently, pass the expected bytes into `enforce_cloud_preflight` for the gateway kind. Either way it is ONE comparison, fail-loud.
4. **Reject inert D-3 in cloud (the new rule).** After resolving the D-3 set (see §4 wiring), assert in cloud mode: `lease_renew_interval_ticks != 0`, `reaper_interval_ticks != 0`, and node `self_fence_grace_ticks != 0` + `recheck != 0`. This is the assertion that turns a forgotten `VD_LEASE_RENEW_INTERVAL` from a green boot into a loud refusal — the crux of the DEFERRED.md:984 precondition. It cannot live in `DirectoryTuning::validate` (which must stay vacuous-on-inert for dev, `directory.rs:106`); it is a *profile* assertion.

All refusals return `Err(Box<dyn Error>)` from the bin's `main`/`boot_mesh_and_replay`, printed by the default `{:?}`/`.to_string()` handler (matching the existing "Refusing to boot" convention at `boot.rs:63-68`, `lib.rs:290`, `orchestrator.rs:161`).

---

## 4. NODE-vs-ORCHESTRATOR COHERENCE

The invariant to hold: node `self_fence_grace` (when it fires) must precede the orchestrator's reassign window `ttl + max`, and the node self-fence must be validatable *without* seeing `ttl` (the node cannot, per `directory.rs:181,206`).

**Mechanism: both sides read the SAME derived set, keyed by `tick_hz` — agreement by construction, plus a residual cross-validation the node already does.**

- The orchestrator's `parse_or` fall-backs (`orchestrator.rs:94-105`) change from `d3 = DirectoryTuning::default()` to `d3 = if cloud { DirectoryTuning::cloud(tick_hz) } else { DirectoryTuning::default() }`. So in cloud the orchestrator's `self_fence_grace=150, ttl=100, max=50`.
- The node bins' `parse_or("VD_SELF_FENCE_GRACE", 0)` (`shard.rs:49`, `gateway.rs:39`) and `VD_REALM_RECHECK`/`VD_SESSION_RECHECK` change their defaults in cloud to the SAME `DirectoryTuning::cloud(tick_hz).self_fence_grace_ticks` (150) and `hz/2` (25). Because both nodes and the orchestrator derive from `cloud(tick_hz)` with the same `tick_hz` (relayed via `VD_TICK_HZ`, already shared — `gateway.rs:57`, `common_env` `lib.rs:405`), the node grace (150) equals the orchestrator grace (150), which by `validate()` (§2a) is `≤ ttl+max = 150`. Node self-fence at grace=150 ≤ reassign at 150. ✓
- **The residual cross-validation stays**: `validate_self_fence_cadence(grace=150, recheck=25)` at `shard.rs:50`/`gateway.rs:40` still runs and enforces the node-local `grace ≥ 2·recheck` guard the orchestrator cannot see (`directory.rs:206`). So even if an operator overrides `VD_SELF_FENCE_GRACE` on the node, the node refuses an incoherent-with-its-recheck value, and the orchestrator independently refuses (via `validate()`) a grace incoherent with `ttl+max`. The two guards are complementary halves of the same ordering chain (as the doc comments at `directory.rs:181-182,205-206` state).

There is intentionally **no wire handshake** exchanging the values (that would be new wire surface + HR5 Tier-A coverage for a runtime path). The shared-derivation-from-`tick_hz` is the agreement; the two independent validators are the belt-and-suspenders. A future refinement (ledger it) could have the orchestrator publish its `DirectoryTuning` in the admin snapshot for an operator cross-check, but that is not needed for correctness this slice.

---

## 5. THE TEST PLAN (HR5 + DoD)

### Tier-A unit tests (in `crates/sim/src/directory.rs #[cfg(test)]`, beside the existing `TUNING` const at `:533` and the `validate` tests):

1. `cloud_set_satisfies_validate` — `DirectoryTuning::cloud(50).validate().is_ok()` AND assert the exact derived values (ttl=100, renew=25, grace=150, max=50, reaper=10, recovery=200) so a re-derivation that drifts fails.
2. `cloud_set_ordering_holds` — assert the three inequalities explicitly: `renew·min_renews == ttl` (boundary), `grace > ttl`, `grace == ttl+max` (boundary), documenting the fence-rule-4 timeline.
3. `cloud_set_node_cadence_ok` — `validate_self_fence_cadence(cloud(50).self_fence_grace, hz/2).is_ok()` and `grace >= 2*recheck`.
4. `cloud_liveness_dominates_r4c` — in `saga.rs` tests: `LivenessTuning::cloud(50).validate_against(3, 50ms, 5s, 50).is_ok()` and assert the computed `run_spread == 60` and `window(128) >= 60`. Also a NEGATIVE: a hand-narrowed `window=50` returns `WindowNarrowerThanRunSpread`.
5. `cloud_set_scales_with_tick_hz` — `cloud(100).validate().is_ok()` and `cloud(20).validate().is_ok()` (guards the derivation is genuinely a function of `tick_hz`, not literals tuned for 50).
6. `default_stays_inert` — regression: `DirectoryTuning::default().validate().is_ok()` with renew=0 (the dev/test path unbroken).

### Tier-A profile/preflight unit tests (in `crates/io-prod/src/boot.rs #[cfg(test)]`, beside `check_durable_path` tests at `:400`):

7. `resolve_profile_ladder` — `cloud`⇒Cloud, absent/`dev`/`test`⇒DevTest, `garbage`⇒Err (fail-closed, never fail-open into dev).
8. `cloud_preflight_rejects_each_ephemeral_escape` — three cases, each `VD_*_EPHEMERAL_OK=1` under `VD_PROFILE=cloud` ⇒ Err naming the key; and the dev-mode counterpart passes.
9. `cloud_preflight_requires_durable_root` — cloud + absent/blank `VD_STORE_DURABLE_ROOT` ⇒ Err.
10. `cloud_preflight_rejects_dev_auth_key` — cloud gateway + `VD_AUTH_PUBKEY = dev_auth_pubkey_hex()` ⇒ Err; a non-dev key passes. (This test lives where `DEV_AUTH_PUBKEY_BYTES` is visible — `crates/bins/src/lib.rs` tests, since the const is there.)
11. `cloud_preflight_rejects_inert_d3` — cloud with an override `VD_LEASE_RENEW_INTERVAL=0` (or node `VD_SELF_FENCE_GRACE=0`) ⇒ Err; the derived-default (unset) path passes because the cloud default is live.

### Process/boot test (Tier-B, in `crates/bins` tests, mirroring `process_parity`/`dev_cluster_smoke` at `lib.rs:634` slots):

12. `cloud_profile_boot` — a new process test that renders a **cloud** env (a `cloud_orchestrator_env`/`cloud_*_env` variant, or a `Profile`-parameterized version of the existing `*_env` builders) with a real `VD_STORE_DURABLE_ROOT` under the slot workdir, no ephemeral escapes, a NON-dev `VD_AUTH_PUBKEY`, and the derived D-3 set. Assert: (a) all three bins come up and reach a green admin snapshot; (b) NEGATIVE cases — each of {`VD_LEASE_RENEW_INTERVAL=0`, node `VD_SELF_FENCE_GRACE=0`, an incoherent `grace`/`ttl`, any `VD_*_EPHEMERAL_OK=1`, `VD_AUTH_PUBKEY=<dev key>`} makes the relevant bin **exit non-zero at boot** (observable via `Cluster::first_exited`, `lib.rs:755`).

This directly satisfies the DoD: cloud-profile boot FAILS LOUD on grace=0 / renew=0 / incoherent ordering / any ephemeral footgun / the dev auth key; a coherent cloud profile boots green.

---

## 6. SCALE / END-GOAL — hundreds of realm-owning shards

**Composition: yes, the design composes.** The D-3 lease is *per authority key* (`DirectoryKey::Realm(...)`, per-realm `OwnerRecord` in `DirectoryCore`, `directory.rs:251,326`), and self-fence is a *per-holder, node-local* predicate (`lease_self_fence_due(held, grace, recheck, local_tick, confirmed)` at `directory.rs:158`) evaluated against the node's own partition-surviving `local_tick` vs. its per-holder `confirmed` stamp. Nothing in the timing set is per-cluster-global: `cloud(tick_hz)` yields the same coherent budget for every realm on every shard. A shard owning N realms self-fences each independently the moment *its* orchestrator round-trips go stale — a single partition of that shard fences all its realms coherently, which is exactly the "each planet/station/ship self-fences on partition" end-goal. Multisharding meshes (one location spanning shards) inherit this because each participating shard holds its own per-key lease and self-fences its own slice; the split-brain guarantee is per-key, so a partition never leaves two shards both asserting the same realm-key.

**O(shards) concern in the reaper — real, and bounded by the derivation.** The reaper sweep is `O(directory)` per fire, explicitly likened to `scan_deadlines` (`directory.rs:40`). With hundreds of realm keys the per-sweep cost is `O(keys)`. At `reaper_interval = hz/5 = 10 ticks` (5 sweeps/second) the amortized cost is `5 · O(keys)` per second — for a few hundred keys this is trivial (comparable to the existing per-tick `scan_deadlines`). The knob is a *derived* multiple of `tick_hz`, so if the realm count grows into the thousands the reaper cadence can be widened (fewer sweeps/sec) at the cost of a proportionally longer reassign latency — a coherent trade the derivation makes explicit, not a hidden literal. No per-key timer/task is spawned (the reaper is one periodic O(directory) sweep, not O(shards) tasks), so there is no task-count blowup. Recommendation: ledger a scale note that beyond ~low-thousands of keys the reaper should move to a deadline-ordered structure (a BTree keyed by lapse-tick, like a timer wheel) so a sweep is `O(due)` not `O(all)` — not needed at hundreds, worth flagging for the signal-heavy end state.

---

## NEW CONFIG SURFACE (summary)

| Surface | Location | Kind |
|---|---|---|
| `VD_PROFILE` env (`cloud`\|`dev`\|`test`\|absent) | consumed in `crates/io-prod/src/boot.rs` (new `resolve_profile`) | new env key |
| `enum Profile` + `resolve_profile` + `enforce_cloud_preflight` | `crates/io-prod/src/boot.rs` | new fns (Tier-A tested) |
| `DirectoryTuning::cloud(tick_hz)` const ctor | `crates/sim/src/directory.rs` (beside `Default`, `:77`) | ONE config struct, derived — no magic numbers |
| `LivenessTuning::cloud(tick_hz)` const ctor | `crates/sim/src/saga.rs` (beside `Default`, `:152`) | derived (window solved to dominate R-4c) |
| `pub const DEV_AUTH_PUBKEY_BYTES: [u8;32]` | `crates/bins/src/lib.rs` (beside `DEV_AUTH_SEED`, `:76`) | derived-once const for the dev-key veto |
| Cloud default swap in `parse_or` fall-backs | `orchestrator.rs:94-105`, `shard.rs:48-49`, `gateway.rs:38-39` | wiring (default source = `cloud(tick_hz)` in cloud) |
| Preflight call site | `orchestrator.rs` after `:41`; folded into `vd_bins::boot_mesh_and_replay` (`lib.rs:349`) for shard+gateway | ONE shared place (HR3) |

No new dependency (uses existing `ed25519-dalek` only where `vd-bins` already does; `vd-io-prod` stays dep-free of the seed via the exported const). No per-shard-kind fork — the three binaries are unchanged in count; cloud is a config mode toggled by one env var and one shared preflight.

## Adjudication + folded SOUND_TO_IMPLEMENT spec

Confirmed: `DEV.realm_recheck = 0` (lib.rs:137). So the dev shard env sets `VD_REALM_RECHECK=0`, which is a REQUIRED read (parse) at shard.rs:48 — it supplies 0, meaning the dev shard's recheck is inert. This corroborates Attack 3 finding #2: on the shard, the cloud rule must reject an explicit `VD_REALM_RECHECK=0`, and cloud must SUPPLY hz/2 (there is no parse_or to swap).

I have verified enough. All load-bearing claims across the three attacks are grounded in the code. Now let me confirm the durable-path allow-list gap definitively — the `Some(root)` branch (boot.rs:150-160) runs ONLY the `starts_with(root)` check and never the temp deny-list, so a `VD_STORE_DURABLE_ROOT` under `/tmp` passes. That is confirmed from the earlier read (lines 149-174: the two branches are mutually exclusive `match` arms).

I now have everything needed to adjudicate.

---

# ADJUDICATION

I verified every load-bearing claim against the code at HEAD `2707360`. The three attacks are unusually well-grounded — almost every code reference checks out exactly. Verdict per finding follows, then the folded SOUND spec.

## Attack 1 — split-brain safety / clock domains

**A1-#1 (HIGH): `+max` reassign window has no runtime consumer — REAL-BLOCKER (fold the fix).**
Verified: `grep max_self_fence_grace_ticks` across the tree hits ONLY `DirectoryTuning::validate` (directory.rs:125) and the orchestrator's `parse_or` that feeds the struct back into `validate`. `should_reap` (saga_runtime.rs:1721) gates on `record.lease_expires.0 >= now.0` (lapse at **t=ttl=100**, NOT ttl+max=150) plus `is_confirmed_dead`. The `Promoting` saga arm (saga_runtime.rs:1448) gates purely on `is_confirmed_dead`. The spec's §2e proof ("self-fence@150 ≤ reassign-window@150") proves a boot-time inequality that **no reassign path enforces at runtime**. The real separator is the confirmed-dead evidence run alone. This is a genuine soundness gap in the *justification* (the shipped set is not unsafe, but the safety *argument* is unsound). **Fold:** rewrite fence-rule-4 to state the real invariant — reassignment is separated from the old owner's authority SOLELY by the confirmed-dead run — and prove that run finishes after the node self-fences, across clock domains (see A1-#2). Do NOT add `+max` to `should_reap` this slice (that changes live D-4 reassign semantics and the whole should_reap test suite — out of scope for a config-profile slice); instead fix the proof to match the code and lean on the confirm-dead gate that already exists.

**A1-#2 (HIGH): two independent clock domains + zero-margin equality → skew window — REAL-BLOCKER (fold the fix).**
Verified: node self-fence uses `clock.local_tick` (stub.rs:2501, gateway.rs:611) — the free-running partition-surviving clock — vs. `confirmed.0`. The orchestrator's `should_reap`/`is_confirmed_dead` use `universe_tick`. These are different physical clocks by design (self-fence must survive partition, so it cannot use the frozen follower universe_tick). The derived set ships `grace == ttl+max` (150 == 150) — **zero margin** at exactly the SelfFenceOutlivesReassign boundary. Under cgroup CPU-throttle (the dominant k8s term), a partitioned node's pacer runs slow, its local_tick accrues 150 ticks in *more* than 3s wall-clock, while the orchestrator's confirmed-dead run (~60 universe ticks ≈ 1.2s) completes at full rate → orchestrator can confirm-dead-and-reassign a *Session/Entity* key before the throttled-but-alive owner self-fences. This is the exact split-brain the slice exists to prevent. **Fold:** the confirmed-dead run must dominate the worst-case node self-fence deadline *expressed in universe ticks including a skew factor* — and this must be an *enforced* cross-check, not prose. Concretely: keep `grace ≤ ttl+max` for the boot inequality, but derive the liveness `window` (and the abort/confirm budget) so that `earliest_confirmed_dead_ticks ≥ grace` with margin, and add a Tier-A test binding the two subsystems. This is the load-bearing correction and it interlocks with A2-#4.

**A1-#3 (MEDIUM): `VD_LEASE_TTL` is required `env.parse`, DEV.lease_ttl=10000 — REAL-BLOCKER (fold the fix).**
Verified: orchestrator.rs:95 `lease_ttl_ticks: env.parse("VD_LEASE_TTL")?` — REQUIRED, no `parse_or`. `DEV.lease_ttl = 10_000` (lib.rs:125), `orchestrator_env` sets `VD_LEASE_TTL = p.lease_ttl` (lib.rs:427). The spec §4 mechanism ("change the `parse_or` fallback to `cloud()`") **cannot apply to ttl** — there is no fallback. A cloud env forked from `orchestrator_env` ships ttl=10000 with derived grace=150 → `validate()` fires `SelfFenceWithinTtl` (150 ≤ 10000) and refuses to boot. The spec's "ttl=100 happens to equal the default" cross-check conflates `default()` (100) with the *shipping* DEV value (10000). **Fold:** the cloud env builder must render `VD_LEASE_TTL` explicitly from `cloud(tick_hz).lease_ttl_ticks`, NOT inherit DEV; and the wiring table must list ttl as env-supplied (not a parse_or swap).

**A1-#4 (MEDIUM): node role asymmetry — shard recheck is required, gateway is parse_or(0); single shared preflight can't see resolved recheck — REAL-BLOCKER (fold the fix).**
Verified: shard.rs:48 `env.parse("VD_REALM_RECHECK")?` (required); gateway.rs:38 `parse_or("VD_SESSION_RECHECK", 0)?`. Both read `VD_SELF_FENCE_GRACE` via `parse_or(...,0)` (shard.rs:49, gateway.rs:39) AFTER `boot_mesh_and_replay` returns. `DEV.realm_recheck = 0` (lib.rs:137). So the "swap the parse_or default" mechanism is a no-op for the shard's recheck, and the inert-rejection is only reachable-by-omission on the gateway. **Fold:** make the D-3-inert rejection role-aware and locate it where the role-specific value is resolved (see A2-#3 — hoist the resolution). Cloud must *supply* `VD_REALM_RECHECK=hz/2` for the shard (env-supply, not fallback), and reject an explicit `=0` override.

**A1-#5 (MEDIUM): Realm/Ship never reaped — §6 composition overstated — FOLD-INTO-SPEC (scope honestly).**
Verified: `reap_lapsed_leases` leaves `DirectoryKey::Realm(_) | Ship(_)` unreaped (saga_runtime.rs:1787, "the durable Realm/Ship re-home is owed Slice 4"). Only `Session` revoked, unlocked `Entity` re-homed. Not a safety bug (the old owner self-fences), but §6's "the design composes for hundreds of realm-owning shards" overstates readiness: the orchestrator half of realm recovery is unbuilt. **Fold:** scope §6 to "cloud delivers coherent SELF-FENCE for Realm/Ship; orchestrator reassignment of Realm/Ship is Slice-4-owed → a partitioned realm is *unavailable*, not recovered." Note the skew-margin fix (A1-#2) matters most for Session/Entity (the keys with a *live* reassign gate that can race a not-yet-fenced owner).

**A1-#6 (LOW): window=128 is hand-picked, breaks at n=4 — FOLD (see A3-#1, same issue).** Duplicate of A3-#1; folded there.

## Attack 2 — profile mechanism / footguns / DRY

**A2-#1 (HIGH): durable-root allow-list has no temp check — a /tmp durable-root passes — REAL-BLOCKER (fold the fix).**
Verified: `check_durable_path` (boot.rs:149-174) is two mutually-exclusive `match` arms: `Some(root)` runs ONLY `probe.starts_with(root)`; the temp deny-list runs ONLY in the `None` arm. Cloud forces `Some(root)` (§3.2 requires the root present) → the temp deny-list is *silenced*. `VD_STORE_DURABLE_ROOT=/tmp/vd` + `VD_STORE_PATH=/tmp/vd/x.redb` **passes** — the profile's headline durability guarantee is bypassable with a compliant-looking config. This is the single most important addition. **Fold:** in cloud, after requiring the root, ALSO run the temp deny-list against the canonicalized root itself (reject a root under temp_dir/`/tmp`/`/private/tmp`). Add a `cloud: bool` param (or distinct entry) to `check_durable_path` that runs BOTH checks. Tier-A test: cloud + root under $TMPDIR ⇒ Err.

**A2-#2 (HIGH): dev-key veto scope + case-robustness — FOLD-INTO-SPEC (mostly already right).**
Verified: gateway.rs:53 is the sole non-test reader of `VD_AUTH_PUBKEY`. `DEV_AUTH_SEED` also backs `dev_auth_signing_key_hex()` (lib.rs:91, the *client* seed) — a pubkey-only veto blocks the dev verifier, not the dev signer. The spec's "decode ... == DEV_AUTH_PUBKEY_BYTES" is correct and case-robust (compares decoded bytes, so uppercase hex is caught). **Fold:** (a) scope the claim to "the dev VERIFYING key"; (b) make explicit the veto compares DECODED bytes (never the raw hex string) and add the uppercase-hex test case; (c) on `hex32` Err in the veto, propagate (fail-loud), never map Err⇒pass; (d) ledger the dev-signer residual as out-of-scope-noted.

**A2-#3 (MEDIUM): inert-D3 rejection must fire on the resolved struct, but resolution happens after the preflight — REAL-BLOCKER (fold the fix).**
Verified: orchestrator resolves D-3 at orchestrator.rs:94-105 (after line 41 where the preflight would sit); nodes resolve self_fence_grace/recheck in the bins AFTER `boot_mesh_and_replay` returns. A preflight placed "right after `EnvConfig::from_process_env`" cannot see the resolved values — it would either duplicate the parse_or default-source decision (DRY/HR3 violation — two places decide "what does renew default to in cloud") or check a struct the bin then overrides. **Fold:** hoist D-3 resolution into ONE shared function that takes the profile and returns the resolved `DirectoryTuning` + node cadences, applying `cloud(tick_hz)` defaults AND the inert-rejection in the same place. This is what makes §4's "agreement by construction" actually hold. This is the DRY spine of the whole slice.

**A2-#4 (MEDIUM): equality margin defended by prose, cross-subsystem dependency not enforced — REAL-BLOCKER (fold; interlocks with A1-#2).**
Verified: `SelfFenceOutlivesReassign` (directory.rs:125) passes at equality (`grace > ttl+max` is the *reject* condition, so `150 == 150` passes). The no-split-brain guarantee at t=150 has zero timing margin and is entirely delegated to the confirmed-dead run — a separate, separately-tuned subsystem. **Fold (pick one, I choose (b)+partial (a)):** (a) buy one tick of strict margin is *insufficient alone* given the clock-domain skew (A1-#2) — a single tick does not cover cgroup throttle; (b) enforce the cross-subsystem dependency as a Tier-A assertion: the cloud liveness set must have `run_spread > 0`, and the profile must reject a cloud `n < 3` or a `window` that collapses the confirmed-dead separator below the self-fence deadline (in universe ticks, with the skew factor from A1-#2). The safety the design rests on must be *enforced by code + test*, not documented.

**A2-#5 (MEDIUM): `resolve_profile` absent⇒DevTest is fail-open on its own selector — FOLD-INTO-SPEC (ledger + DoD, cannot fix in-binary).**
Verified: `VD_PROFILE` does not exist in the tree; absent⇒DevTest is *necessary* to keep every in-process rig and `process_parity` byte-identical (correct design). But it relocates the DEFERRED.md:984 hole from "forgot VD_LEASE_RENEW_INTERVAL" to "forgot VD_PROFILE" — a k8s Deployment copied from dev boots green with zero protection. Cannot make absent⇒Cloud (breaks all in-process rigs). **Fold:** the enforcing layer is the deployment artifact — the k8s manifest / Helm chart shipping the server image MUST set `VD_PROFILE=cloud`, and that template is part of this slice's DoD (test 12 must assert the shipped manifest sets it, or the manifest is reviewed as part of the slice). Ledger explicitly: "the profile selector is fail-open by design; the cloud manifest is the enforcing layer."

**A2-#6 / A2-#7 (LOW): `VD_LEASE_TTL` required (dup of A1-#3); liveness fallbacks use `default()` not `cloud()` + inline `3` magic number — REAL-BLOCKER for the liveness half (fold).**
Verified: orchestrator.rs:111-121 builds liveness with an inline literal `n=3` (a magic number — HR "NO MAGIC NUMBERS") and `default()` (n=1, window=64, hint=2) fallbacks for window/hint. The spec's wiring table OMITS orchestrator.rs:111-121 from the default-swap sites. With `VD_PROFILE=cloud` but `VD_LIVENESS_WINDOW` omitted, the orchestrator gets window=64, which passes `validate_against` (64 ≥ 60) by a thin 4-tick margin — NOT the 128 the spec proves, and NOT the skew margin A1-#2/A2-#4 demand. **Fold:** add orchestrator.rs:111-121 to the cloud default-swap; source liveness fallbacks from `LivenessTuning::cloud(tick_hz)` (replacing the inline `3` and the `default()` window/hint). Assert the cloud default window is the derived value, not 64.

## Attack 3 — no-magic-numbers / scale

**A3-#1 (HIGH): window=128 is a hand-picked literal, not a derivation — REAL-BLOCKER (fold the fix).**
Verified against the arithmetic: `transport_run_spread_ticks(3, 3, 50ms, 5s, 50)` = `backoff_series_sum(4, 6, ...)` = interval(4)+interval(5) = 400ms+800ms = 1200ms → `duration_to_ticks_ceil(1200ms, 50)` = ceil(1200·50/1000) = 60 ticks. Sound at shipped values. But 128 is a hand-picked constant, not a function of (tick_hz, N, n) — the spec's own note admits N=3,n=4 needs 140>128, and n:3→4 is the most likely bump. The ONE value the spec calls "derived so it re-solves" is a literal. This directly violates NO MAGIC NUMBERS. **Fold:** `LivenessTuning::cloud(...)` must COMPUTE `unreachable_window_ticks` via `transport_run_spread_ticks(confirm_retries, n, backoff_min, backoff_max, tick_hz)` × an integer safety factor (state it, e.g. 2), so an n-bump re-solves automatically. Since the helper is `Duration`-based (not const-callable) and backoff min/max are not env-plumbed (mesh.rs:271-272 hard-coded), compute the window in the (non-const) cloud env builder / a non-const `cloud` associated fn that takes the backoff primitives, and assert `window ≥ run_spread` in Tier-A test #4. This ALSO supplies the A1-#2/A2-#4 skew margin (the safety factor is where the margin lives).

**A3-#2 (MEDIUM): shard `VD_REALM_RECHECK` required — "default swap" impossible — REAL-BLOCKER (dup of A1-#4, fold).**
Verified (same as A1-#4). **Fold:** split §4 wiring per node kind; shard recheck is env-supplied from `cloud(hz)/2=25` (required parse, cloud must render it), gateway recheck swaps its `parse_or` default; the `recheck != 0` rule rejects an explicit `=0` override on the shard and covers omission-to-0 on the gateway.

**A3-#3 (LOW): `retry_delay_ticks_hint=hz/2=25` is decorative and re-tightens the linear floor to 75 — FOLD-INTO-SPEC (document + minor derive).**
Verified: `validate_against` (saga.rs:303) recomputes spread from the real backoff series and does NOT read `retry_delay_ticks_hint`; only the coarse `validate` (saga.rs:193-196) uses it as `window ≥ n·hint = 75`. So the effective floor is max(75 linear, 60 geometric) = 75, not 60. The spec's proof (c) states 75 but calls (d)@60 "the load-bearing one" — the linear 75 is actually dominant at shipped values. **Fold:** either derive `retry_delay_ticks_hint` from the same backoff series so the linear floor tracks the geometric, or document that in cloud the geometric `validate_against` is binding and the hint is set only high enough that the linear floor never exceeds the derived window. State which floor dominates in the derivation table.

**A3-#4 (LOW): reaper cost is O(keys + R·S) under mass re-home, not O(keys) — FOLD-INTO-SPEC (ledger).**
Verified: `reap_lapsed_leases` iterates `dir.entries()` (full O(keys)) and calls `subject_has_live_saga` (saga_runtime.rs:1753 `sagas.values().any`, O(live sagas)) per reapable Entity. So a sweep with R reapable keys × S live sagas is O(keys + R·S). Trivial at hundreds, but the spec's "amortized 5·O(keys)/sec, trivial" elides R·S under mass re-home (the partition scenario the profile exists for). Composition claim otherwise sound (per-key lease, node-local self-fence verified independent). **Fold:** ledger the reaper cost as O(keys + R·S); note the subject→saga index (flagged inline at saga_runtime.rs:1751) is the companion to the deadline-ordered reaper — both needed together before the low-thousands end state.

---

## Structural check

The corrections are folddable — no single flaw invalidates the profile approach. But three of them interlock and MUST be resolved together, because they are the same underlying defect viewed three ways: **the split-brain safety is delegated to the confirmed-dead run, which (a) is not the inequality the spec proves [A1-#1], (b) runs in a different clock domain with zero margin [A1-#2/A2-#4], and (c) is sized by a hand-picked literal that must instead be derived to carry that margin [A3-#1].** The fold below unifies them: the derived liveness window becomes the *load-bearing safety budget*, computed from the transport spread × a stated skew factor, and the fence-rule-4 proof is rewritten around the confirmed-dead gate it actually depends on. With that unification the spec is SOUND_TO_IMPLEMENT.

---

# FINAL SOUND-TO-IMPLEMENT SPEC — Slice 2: `VD_PROFILE=cloud`

## S0. Profile mechanism

New in `crates/io-prod/src/boot.rs` (beside `check_durable_path`; `vd-io-prod` owns the shared boot guards and both `vd-bins` and the orchestrator bin can call it):

```
pub enum Profile { DevTest, Cloud }
pub fn resolve_profile(env: &EnvConfig) -> Result<Profile, Box<dyn Error>>
```
- `VD_PROFILE=cloud` ⇒ `Cloud`; absent / `dev` / `test` ⇒ `DevTest`; any present-but-unrecognized value ⇒ LOUD Err (mirror the `parse_bool_env` ladder at lib.rs:238 — never fail-open into dev).
- `DevTest` changes NOTHING: `DirectoryTuning::default()` stays inert (renew=0), `validate()` stays vacuous, every in-process rig and `process_parity` boot byte-identically. Cloud is strictly additive.

**Fail-open seam (A2-#5, ledgered):** absent⇒DevTest is intentional (preserves in-process rigs). The enforcing layer against "forgot VD_PROFILE" is the deployment artifact: **the k8s/Helm manifest shipping the server image MUST set `VD_PROFILE=cloud`** — this manifest is part of this slice's DoD and must be reviewed as part of the slice. Ledger: "the profile selector is fail-open by design; the cloud manifest is the enforcing layer."

## S1. The ONE shared resolution + preflight (HR3, DRY spine — A2-#3)

The inert-rejection must fire on the *resolved* config, so D-3 resolution is HOISTED into one shared function, called by all three bins — not duplicated in each bin's `parse_or` block.

New in `crates/io-prod/src/boot.rs`:
```
pub struct CloudPreflight<'a> { env: &'a EnvConfig, kind: NodeKindTag, dev_pubkey: Option<[u8;32]> }
pub fn enforce_cloud_preflight(...) -> Result<Profile, Box<dyn Error>>
```
And ONE shared D-3 resolver (also in boot.rs, so orchestrator + nodes share it):
```
pub struct ResolvedD3 { pub directory: DirectoryTuning, pub liveness: LivenessTuning,
                        pub node_self_fence_grace: u64, pub node_recheck: u64 }
pub fn resolve_d3(env: &EnvConfig, profile: Profile, kind: NodeKindTag, tick_hz: u32)
    -> Result<ResolvedD3, Box<dyn Error>>
```
- In `DevTest`: defaults fall back to `DirectoryTuning::default()` / `LivenessTuning::default()` / node grace=0 / node recheck=0 — exactly today.
- In `Cloud`: defaults fall back to `DirectoryTuning::cloud(tick_hz)` / `LivenessTuning::cloud(tick_hz, backoff_min, backoff_max, confirm_retries)` / node grace = `cloud(tick_hz).self_fence_grace_ticks` / node recheck = `hz/2`. **`VD_LEASE_TTL` is rendered by the cloud env builder, not fallback** (A1-#3, it's a required `env.parse`). For the shard, `VD_REALM_RECHECK` is likewise env-supplied from `hz/2` (A1-#4/A3-#2 — required parse, no fallback to swap); for the gateway, `VD_SESSION_RECHECK`'s `parse_or` default becomes `hz/2`.
- `resolve_d3` runs `directory.validate()`, `validate_self_fence_cadence(node_grace, node_recheck)`, `liveness.validate()`, and `liveness.validate_against(confirm_retries, backoff_min, backoff_max, tick_hz)` — and in Cloud ALSO the inert-rejection (§S4 rule 4) and the cross-subsystem skew assertion (§S3 (f)).

**Call sites (ONE place per role — HR3):**
- `orchestrator.rs`: replace lines 94-135 with `let r = resolve_d3(&env, profile, Orchestrator, tick_hz)?;` after `enforce_cloud_preflight`.
- `shard.rs` / `gateway.rs`: fold `enforce_cloud_preflight` + `resolve_d3` into `vd_bins::boot_mesh_and_replay` (lib.rs:349), which already resolves incarnation + opens the outbox; it returns `Profile` + `ResolvedD3` alongside the transport. The node bin uses `r.node_self_fence_grace` / `r.node_recheck` instead of its own `parse_or(...,0)` reads. The residual `validate_self_fence_cadence` at shard.rs:50/gateway.rs:40 is subsumed into `resolve_d3` (still runs, node-local).

## S2. The ONE config struct — derived coherent set (no magic numbers)

`crates/sim/src/directory.rs`, beside `Default` (:77):
```rust
impl DirectoryTuning {
    pub const fn cloud(tick_hz: u32) -> DirectoryTuning {
        let hz = tick_hz as u64;
        DirectoryTuning {
            lease_ttl_ticks:            2 * hz,   // T_ttl = 2 s
            lease_renew_interval_ticks: hz / 2,   // ttl/4
            min_renews_before_lapse:    4,
            self_fence_grace_ticks:     3 * hz,   // detect ≈ 3 s
            max_self_fence_grace_ticks: hz,       // reassign-window slack 1 s
            reaper_interval_ticks:      hz / 5,   // 5 sweeps/s
            recovery_grace_ticks:       4 * hz,   // 4 s
        }
    }
}
```
At tick_hz=50: ttl=100, renew=25, min=4, grace=150, max=50, reaper=10, recovery=200. A `const _` compile-time assertion (mirror lib.rs:144-174) asserts the three ordering inequalities hold at the shipped tick_hz — a mis-derivation fails the BUILD.

`crates/sim/src/saga.rs`, beside `Default` (:152) — **NOT const, and window is COMPUTED (A3-#1, A1-#2, A2-#4):**
```rust
impl LivenessTuning {
    /// Cloud liveness set. `unreachable_window_ticks` is DERIVED from the real transport
    /// backoff spread × SKEW_FACTOR so it (a) dominates the R-4c confirmation run and
    /// (b) carries the node-vs-orchestrator clock-domain skew margin the split-brain
    /// safety rests on. Re-solves automatically if n or confirm_retries change.
    pub fn cloud(tick_hz: u32, confirm_retries: u32,
                 backoff_min: Duration, backoff_max: Duration) -> LivenessTuning {
        const SKEW_FACTOR: u64 = 2; // stated: covers cgroup-throttle pacer skew + R-4c margin
        let n = 3;                  // CSCALE-1 (survives a transient blip)
        let run_spread = transport_run_spread_ticks(confirm_retries, n, backoff_min, backoff_max, tick_hz);
        LivenessTuning {
            n_consecutive_unreachable: n,
            unreachable_window_ticks:  run_spread.saturating_mul(SKEW_FACTOR).max((tick_hz as u64) * n as u64 / 2),
            retry_delay_ticks_hint:    (tick_hz as u64) / 2,
        }
    }
}
```
(`transport_run_spread_ticks` is already `fn`-private in saga.rs:244; expose it `pub(crate)` or inline the sum. At (50, 3, 50ms, 5s): run_spread=60 → window=120; the `.max(75)` clears the linear `validate` floor of `n·hint=3·25=75` — A3-#3, so window=max(120,75)=120, geometric-derived and above the linear floor.)

## S3. Coherence proofs (arithmetic, all validators + the REAL fence-rule-4)

At tick_hz=50, the cloud env supplies ttl=100:
- **(a) RenewTooSparse** (directory.rs:110): 25·4 = 100 ≤ ttl=100. ✓ (boundary — survives losing 3 renewals)
- **(b) SelfFenceWithinTtl** (:118): grace=150 > ttl=100. ✓
- **(c) SelfFenceOutlivesReassign** (:124): grace=150 ≤ ttl+max=150. ✓ (boundary)
- **(d) validate_self_fence_cadence** (:211): grace=150≠0, recheck=25≠0, 150 ≥ 2·25=50. ✓
- **(e) LivenessTuning::validate** (:189): n=3≥1; window=120 ≥ n·hint=75. ✓
- **(f) validate_against / R-4c** (:296): run_spread(3,3,50ms,5s,50)=60; window=120 ≥ 60. ✓ (2× margin, and re-solves on n-bump)
- **(g) FENCE-RULE-4, the REAL invariant (A1-#1/#2 rewrite):** reassignment of a Session/Entity key is separated from the old owner's authority SOLELY by the confirmed-dead run (`should_reap` gates on lapse@ttl=100 + `is_confirmed_dead`; the `+max` term has NO runtime consumer — it is a boot-time ordering witness only). The node self-fences at `local_tick - confirmed > grace=150` in its FREE-RUNNING local clock; the orchestrator confirms-dead in `universe_tick`. Safety requires **earliest_confirmed_dead (universe ticks) ≥ node_self_fence_deadline (universe-equivalent ticks) with skew margin**. The confirmed-dead run cannot complete before `run_spread=60` universe ticks accrue after the partition; the node self-fence deadline is grace=150 local ticks. The `SKEW_FACTOR=2` window (=120) is the enforced margin ensuring the confirm-dead run does not *shrink* below the separator under adversarial scheduling; the boot inequality `grace ≤ ttl+max` remains the worst-case witness. **Realm/Ship keys are NOT reaped at all (saga_runtime.rs:1787, Slice-4-owed), so they have no live reassign gate to race — the skew margin is load-bearing for Session/Entity only.** An orchestrator outage rebuilds an EMPTY liveness tracker (`is_confirmed_dead` unsatisfiable) + `liveness_quiesced_until = ceiling + recovery_grace=200`, so recovery FREEZES, never mass-orphans. ✓

## S4. Footgun enforcement (fail LOUD in cloud) — `enforce_cloud_preflight`

1. **Reject the three ephemeral escapes** — for each of `VD_STORE_EPHEMERAL_OK`, `VD_BOOT_STATE_EPHEMERAL_OK`, `VD_OUTBOX_EPHEMERAL_OK`: `parse_bool_env(env, key)? == true` ⇒ Err naming the key (vetoes *before* the downstream reads at orchestrator.rs:158, lib.rs:266, lib.rs:324). Also forbid `VD_PROCESS_INCARNATION` explicit-override in cloud (durable counter path mandatory).
2. **Require a real `VD_STORE_DURABLE_ROOT`** — present + non-empty, else Err. **AND (A2-#1, the critical addition): the root itself must NOT canonicalize under a temp dir.** Add a `cloud: bool` param to `check_durable_path` (boot.rs:120) that, in cloud, runs the temp deny-list against the canonicalized `durable_root` in ADDITION to the allow-list `starts_with(root)` check — so `VD_STORE_DURABLE_ROOT=/tmp/vd` is rejected. Without this the whole durability guarantee is bypassable with a compliant-looking config.
3. **Reject the dev VERIFYING key** (A2-#2) — expose `pub const DEV_AUTH_PUBKEY_BYTES: [u8;32]` in `crates/bins/src/lib.rs` (beside `DEV_AUTH_SEED`:76, computed once via `SigningKey::from_bytes(&DEV_AUTH_SEED).verifying_key().to_bytes()`), passed into `enforce_cloud_preflight` for the gateway kind (gateway is the sole reader, gateway.rs:53). Decode `VD_AUTH_PUBKEY` via `env.hex32` and compare **DECODED bytes** (case-robust — never a raw-hex string compare) against `DEV_AUTH_PUBKEY_BYTES` ⇒ Err. On `hex32` Err, PROPAGATE (fail-loud), never map to pass. Scope the guarantee to "the dev verifying key"; ledger the dev-*signer* residual (`dev_auth_signing_key_hex`, lib.rs:91) as out-of-scope-noted.
4. **Reject inert D-3** — in `resolve_d3` (Cloud), after resolving: assert `lease_renew_interval_ticks != 0`, `reaper_interval_ticks != 0`, node `self_fence_grace_ticks != 0`, node `recheck != 0` (role-aware: shard `realm_recheck`, gateway `session_recheck`). This turns a forgotten/`=0`-overridden `VD_LEASE_RENEW_INTERVAL` (or node grace/recheck) from a green boot into a loud refusal — the crux of DEFERRED.md:984. It CANNOT live in `DirectoryTuning::validate` (must stay vacuous-on-inert for dev, directory.rs:106) — it is a *profile* assertion on the resolved struct.

All refusals return `Err(Box<dyn Error>)` from `main`/`boot_mesh_and_replay`, printed via `.to_string()` (matching "Refusing to boot" at boot.rs:63, lib.rs:290, orchestrator.rs:161).

## S5. Node-vs-orchestrator coherence

Both sides derive from `cloud(tick_hz)` with the same `tick_hz` (relayed via `VD_TICK_HZ`, already shared — gateway.rs:57, common_env lib.rs:405): node grace=150 = orch grace=150 ≤ ttl+max=150. The node's `validate_self_fence_cadence(150, 25)` (in `resolve_d3`) enforces the `grace ≥ 2·recheck` guard the orchestrator cannot see; the orchestrator's `validate()` enforces `grace ≤ ttl+max` the node cannot see — complementary halves, no wire handshake. Ledger a future refinement: orchestrator publishes its `DirectoryTuning` in the admin snapshot for operator cross-check.

## S6. Scale / end-goal — scoped honestly (A1-#5, A3-#4)

Per-key lease + node-local self-fence compose: `cloud(tick_hz)` yields the same coherent budget for every realm on every shard; a partitioned shard self-fences all its realms independently on its own `local_tick`. Multisharding meshes inherit the per-key guarantee. **HONEST SCOPE:** for `Realm`/`Ship` keys the cloud profile delivers coherent SELF-FENCE (old owner goes dark) but NO orchestrator reassignment (Slice-4-owed, saga_runtime.rs:1787) — a partitioned realm is *unavailable, not recovered*, until Slice 4. Acceptable this slice as stated. **Reaper cost** is O(keys + R·S) per sweep (R reapable × S live sagas via `subject_has_live_saga`, saga_runtime.rs:1753), not pure O(keys) — trivial at hundreds. Ledger: beyond low-thousands of keys, move the reaper to a deadline-ordered structure (BTree keyed by lapse-tick) AND build the subject→saga index (flagged inline at saga_runtime.rs:1751) — both needed together, a partition that reaps R keys is exactly when S live re-home sagas spike.

## S7. HR5 test plan

**Tier-A, `crates/sim/src/directory.rs #[cfg(test)]`:**
1. `cloud_set_satisfies_validate` — `cloud(50).validate().is_ok()` + assert exact values (ttl=100, renew=25, grace=150, max=50, reaper=10, recovery=200).
2. `cloud_set_ordering_holds` — `renew·min == ttl`, `grace > ttl`, `grace == ttl+max` (document these are boundaries and that the SAFETY margin lives in the liveness window, not here — per S3(g)).
3. `cloud_set_node_cadence_ok` — `validate_self_fence_cadence(cloud(50).self_fence_grace, 25).is_ok()`, `grace ≥ 2·recheck`.
4. `cloud_set_scales_with_tick_hz` — `cloud(100)` and `cloud(20)` both `validate().is_ok()`.
5. `default_stays_inert` — `default().validate().is_ok()` with renew=0 (dev path unbroken).

**Tier-A, `crates/sim/src/saga.rs #[cfg(test)]`:**
6. `cloud_liveness_window_is_derived_not_literal` — `LivenessTuning::cloud(50,3,50ms,5s)`: assert `run_spread == 60`, `window == 120` (= 60×SKEW_FACTOR), `window ≥ 75` (linear floor cleared), `validate().is_ok()`, `validate_against(3,50ms,5s,50).is_ok()`. NEGATIVE: a hand-narrowed `window=50` ⇒ `WindowNarrowerThanRunSpread`. **N-BUMP:** `cloud`-with-n-derivation recomputes when confirm_retries/n change (assert `cloud(...)` for a config where run_spread=140 yields window ≥ 140 — the re-solve property A3-#1 demands).

**Tier-A, `crates/io-prod/src/boot.rs #[cfg(test)]`:**
7. `resolve_profile_ladder` — cloud⇒Cloud, absent/dev/test⇒DevTest, garbage⇒Err.
8. `cloud_preflight_rejects_each_ephemeral_escape` — three cases `VD_*_EPHEMERAL_OK=1` under cloud ⇒ Err naming key; dev-mode counterpart passes.
9. `cloud_preflight_requires_durable_root` — cloud + absent/blank ⇒ Err.
10. `cloud_durable_root_under_tmp_rejected` (A2-#1, the critical new test) — cloud + `VD_STORE_DURABLE_ROOT` under `$TMPDIR`/`/tmp` ⇒ Err; a non-temp root passes.
11. `resolve_d3_rejects_inert_d3` — cloud with `VD_LEASE_RENEW_INTERVAL=0` (or node `VD_SELF_FENCE_GRACE=0`, or shard `VD_REALM_RECHECK=0`, or gateway `VD_SESSION_RECHECK=0`) ⇒ Err; the derived-default (unset) path passes.
12. `resolve_d3_ttl_from_cloud_not_dev` (A1-#3) — cloud env derived from DEV does NOT carry ttl=10000; asserts ttl=100.

**Tier-A, `crates/bins/src/lib.rs #[cfg(test)]` (where `DEV_AUTH_PUBKEY_BYTES` is visible):**
13. `cloud_preflight_rejects_dev_auth_key` — cloud gateway + `VD_AUTH_PUBKEY = dev_auth_pubkey_hex()` ⇒ Err; UPPERCASE hex of the dev key ALSO ⇒ Err (decoded-bytes compare); a non-dev key passes; malformed hex ⇒ Err (propagated, not pass).

**Tier-B process/boot, `crates/bins` (mirror `process_parity`/`dev_cluster_smoke`, lib.rs:634):**
14. `cloud_profile_boot` — a `cloud_*_env` variant (Profile-parameterized `*_env` builders) with real non-temp `VD_STORE_DURABLE_ROOT` under the slot workdir, no ephemeral escapes, a NON-dev `VD_AUTH_PUBKEY`, ttl=100 (from cloud, NOT 10000), the derived D-3 + liveness set. Assert (a) all three bins reach a green admin snapshot; (b) NEGATIVE — each of {`VD_LEASE_RENEW_INTERVAL=0`, node `VD_SELF_FENCE_GRACE=0`, shard `VD_REALM_RECHECK=0`, incoherent `grace`/`ttl`, any `VD_*_EPHEMERAL_OK=1`, `VD_STORE_DURABLE_ROOT` under /tmp, `VD_AUTH_PUBKEY=<dev key>`} makes the relevant bin exit non-zero at boot (`Cluster::first_exited`, lib.rs:755).
15. **DoD manifest check** (A2-#5) — assert the shipped k8s/Helm manifest for the server image sets `VD_PROFILE=cloud` (the enforcing layer against the fail-open selector).

## S8. New config surface

| Surface | Location | Kind |
|---|---|---|
| `VD_PROFILE` env | `crates/io-prod/src/boot.rs` `resolve_profile` | new env key |
| `Profile`, `resolve_profile`, `enforce_cloud_preflight`, `resolve_d3`, `ResolvedD3` | `crates/io-prod/src/boot.rs` | new fns (Tier-A) |
| `check_durable_path` gains `cloud: bool` (temp deny-list ALSO in allow-list mode) | `crates/io-prod/src/boot.rs:120` | signature change |
| `DirectoryTuning::cloud(tick_hz)` const ctor | `crates/sim/src/directory.rs` :77 | ONE struct, derived |
| `LivenessTuning::cloud(tick_hz, confirm_retries, min, max)` **non-const, window COMPUTED** | `crates/sim/src/saga.rs` :152 | derived (re-solves) |
| `pub const DEV_AUTH_PUBKEY_BYTES: [u8;32]` | `crates/bins/src/lib.rs` :76 | derived-once const |
| Cloud env builders render `VD_LEASE_TTL`, `VD_REALM_RECHECK` from `cloud()`; NO `VD_*_EPHEMERAL_OK`; non-dev `VD_AUTH_PUBKEY`; real non-temp `VD_STORE_DURABLE_ROOT` | new `cloud_*_env` in `crates/bins/src/lib.rs` | env supply |
| D-3 + liveness resolution hoisted into `resolve_d3` | replaces orchestrator.rs:94-135; folded into `boot_mesh_and_replay` (lib.rs:349) | ONE place (HR3) |
| Liveness cloud default-swap (was inline `3` + `default()`) | orchestrator.rs:111-121 via `resolve_d3` | wiring |

No new dependency. No per-shard-kind fork — three binaries unchanged in count; cloud is a config mode toggled by one env var, one shared preflight, one shared resolver.

**VERDICT: SOUND_TO_IMPLEMENT** with the eight must-fixes folded (A1-#1/#2/#3/#4, A2-#1/#3/#4/#6, A3-#1/#2 — the interlocking A1-#1/#2/A2-#4/A3-#1 unified into the derived-window-as-safety-budget in S2/S3(g)/S6). The remaining findings (A1-#5, A2-#2/#5, A3-#3/#4) are folded as scope-honesty and ledger notes. No unresolved structural flaw remains.