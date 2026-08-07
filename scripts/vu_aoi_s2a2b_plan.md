# VU AoI S2a-2b — child EMIT of OccupantInterest (vetted implementation contract)

Vetted via the `vu-aoi-s2a2b-emit-design` workflow (Opus, 2 understand → lead plan → adversarial
verify, 2026-07-28). Verdict: **NEEDS-REVISION** — one MAJOR (folded below) + 3 MINORs (folded).
Core machinery CONFIRMED sound (HR1/HR3/byte-identity/scale). All `file:line` = `crates/sim/src/stub.rs`.

## ⚠ The MAJOR the review caught (would have shipped a false self-fence)

Routing the **parent**-realm HeadRead reply through the existing `DirectoryReply::Head{Realm}` arm calls
`affirm_realm_head`, which branches on `realm == config.realm`. `DirectoryKey::Realm` keys on the **lossy**
`RealmId` (`lowered()`), and `RealmLevel::to_realm_id` collapses **Galaxy → System(1)** — so a System(seed 1)
hosted under a Galaxy has `own.lowered() == parent.lowered() == System(1)`, and the parent-Head reply (owned
by a foreign node) drives the **PRIMARY-FOREIGN self-fence branch** → drops `RealmAuthority`, declares
transients lost, `realm_self_fenced` — a **false self-fence of a healthy shard, from reading its own parent.**
Violates realm_coord.rs's emphatic "NEVER use `lowered()` as a directory/dedup key at multi-galaxy scale".

**Fix (folded):** the colliding universe is *unrepresentable* in today's lowered-keyed directory, so this is
not a hard stop — but arm it LOUD: (a) boot `debug_assert!(own.parent().map(lowered) != Some(config.realm))`
in `register_stub_shard`; (b) a `DEFERRED.md` note that this parent resolve keys on `lowered()` like the
directory and must migrate to `path()`-keying together; (c) keep `update_parent_node`'s own parent-match guard.

## The step list (revisions folded)

1. **Resource + threading.** Add `ParentRealmNode(pub Option<NodeId>)` by `RealmConfirmedAt` (~288);
   `insert_resource(ParentRealmNode::default())` (~1250). Thread `ResMut<ParentRealmNode>` as the 16th param
   of `process_inbound` (at the bevy ceiling — flag it; future adds need bundling) → `&mut parent_node` into
   `on_directory_reply` (1575); add the `&mut ParentRealmNode` param to `on_directory_reply`.
2. **`update_parent_node`** monomorphic helper next to `affirm_realm_head`: `if Some(realm) ==
   own.parent().map(|p| p.lowered()) { parent_node.0 = match record.map(|r| r.authority) {
   Some(AuthorityRef::Shard(n)) => Some(n), _ => None } }`. Call **after** `affirm_realm_head` in the
   `Head{Realm}` arm (after 4095). **[MAJOR]** Add the boot `debug_assert` + the DEFERRED ledger note.
3. **`aoi_live()`** `pub(crate)` accessor on `RealmRegions` (~620):
   `self.regions.iter().any(|r| r.aoi.spin_up_r_m() > 0.0)` (inert bands are 0 → false at walk/static).
4. **`parent_headread_due(config, regions, clock) -> Option<RealmCoord>`** near `aoi_decide`:
   `let parent = own_coord.parent()?; (regions.aoi_live() & due_this_tick(config.realm_recheck_interval,
   clock.local_tick.0)).then_some(parent)`. **[MINOR-3] use BITWISE `&`** (not `&&`) — no short-circuit
   region to leave uncovered, matching `AoiConfig::in_range`.
5. **`push_occupant_interest(outbox, parent, observer, to_realm, occupant)`** near `push_demand` (4746):
   3-arg `push_flow(parent, MsgClass::SignalDelta, &InterShardFlow::OccupantInterest{observer, to_realm,
   occupant, coarsen_level: 0})` (Ephemeral; FireAndForget/Unreliable, exempt from both push_flow_durable
   debug-asserts).
6. **Emit block** in `aoi_decide` after the render-delta loop (4644), before `retain_live` (4647):
   the `parent_headread_due` HeadRead, then `for (_, d) in dots.0.iter().filter(|(_,d)| d.authority.simulates())
   { push_occupant_interest(outbox, parent, d.account, parent_coord.clone(), StampedPose{ universe_tick: tick,
   ..d.pose }) }` — **spread `..d.pose`** (carries `orient`; a field-list literal omitting `orient` won't
   compile) and read **`d.pose` directly** (the reduced `observers` tuple `.offset()`-zeroes the cell tier).
   Add `parent_node: Option<NodeId>` to `aoi_decide`; add `parent: Res<ParentRealmNode>` to
   `evaluate_realm_aoi` and pass `parent.0`.
7. **Tests T1–T6** (unit, Tier-A 100%; `decode_flows` at 5111). T1 parent-HeadRead armed / inert-at-walk /
   due-false / root-none (covers `parent_headread_due` + `aoi_live`). T2 Head reply caches node / re-home
   overwrite / revoke-clears / non-parent-unchanged / Gateway-record→None (covers `update_parent_node`).
   T3 emit per dot, right fields (full pos cell + orient), coarsen 0 — **[MINOR-2] assert raw-outbox
   `MsgClass::SignalDelta` BEFORE `decode_flows` (which empties the box)**. T4 root never emits. T5
   parent-unresolved skips (armed → HeadRead yes, no OccupantInterest). T6 transients+zero-dots.
   `#[should_panic]` test for the boot debug_assert on a colliding coord.
8. Confirm the schedule runs `process_inbound` before `evaluate_realm_aoi` (same-tick resolve; else it
   converges one tick later — harmless, `ParentRealmNode` is persistent).

## Deferred to S2b (noted, not S2a-2b work)
- Stale-cache → dead/**recycled** node: across a parent re-home, up to `realm_recheck_interval` ticks of
  OccupantInterest go to the old node. Inert now (parent only decode-drops); at S2b (once the parent FOLDS),
  a recycled NodeId could spuriously keep a cousin warm → invalidate the cache on own-realm-recheck instability.

Confirmed sound: HR1 (only AccountId+StampedPose+coord cross; both push_flow_durable asserts inert for
OccupantInterest), HR3 (reuses HeadRead/DirectoryReply/due_this_tick/the aoi_decide pass — correctly NOT
reusing `request_pending_grants`'s ungated own-realm recheck), byte-identity (3 independent inert gates),
scale (O(local dots) + 1 HeadRead/cadence), design match (dedicated arm, no fence, unreliable, coarsen 0).
