# VU-AoI S2c — DRAW the warmed neighbour realm on the traveller's client, before the crossing

**Status:** vetted + adversarially hardened (3 lenses: leakage-and-sealing, scale-and-topology, correctness-cross-teardown), synthesized 2026-07-28.
**Predecessor:** `scripts/vu_aoi_s2b_plan.md` (the up-relay + warm-ahead — LANDED; DO NOT redesign it).
**Ledger:** flips `D-RLM-11` on landing; opens `D-RLM-13` (the crossing scene-reconcile residual → VU-6).

---

## (A) Plain-language summary (for the human)

Today, when you fly toward the edge of the region you're standing in, the game quietly wakes the neighbouring region *ahead of time* so there's no loading pause when you arrive. That already works. But there's a gap: the neighbour is awake, yet you can't *see* it until the exact instant you cross the border. Removing that last visual pop is this slice.

The obstacle is plumbing, not data. The region you're in is run by one server; the *parent* region that contains both your region and the neighbour is run by another. The parent is the one that knows the neighbour's shape and does the wake-ahead work — but it has no idea which screen you're looking at. Your screen's connection is known only to the server running the ground under your feet.

The fix: the parent never tries to find your screen. It hands the neighbour's outline *down* to the server already drawing your world — the one that unmistakably knows your connection — and that server passes it to you exactly the way it passes you everything else you see. The parent never learns anything about your connection; your connection detail never leaves the one server that already holds it.

One thing we got right only after stress-testing: the parent doesn't send "the neighbour just appeared / just left" as one-off nudges. It continuously tells the home server *the full list of neighbours currently in view*, and the home server works out the difference and updates your screen. This matters because when you actually cross the border you switch servers mid-motion — and a one-off "it left" nudge sent to the server you just left would vanish into thin air, leaving a ghost box on your screen forever. Sending the whole current list instead means whoever is drawing your world can always reconcile it to the truth, even right after a hand-off or after a server restart. It self-corrects.

After this slice: the neighbour region is **visible on approach**, drawn a beat early, and stays seamless as you continue in or turn back. This is the last piece before the pixel-level "watch a dot cross a boundary with no pop" acceptance test can pass. Two stale-teardown corners at the exact hand-off instant (a neighbour that drops out of view during the sub-second server switch; the fine interior detail of the region you just left) are handed to the warp/scene-reset slice, which is the natural owner of "redraw the world when authority changes."

---

## (B) The option decision — **Option C, CONFIRMED. A and B both lose.**

> Option C, one line: the parent reflects the sibling outline **down to the home shard it already relays with** (the up-relay's own return address), and the home shard — the sole holder of the client route — forwards it; **zero client-connection-detail leakage, zero new topology edges, no client-facing wire growth**, which A (leaks the gateway up + a new parent→gateway edge + frozen up-relay growth) and B (needs a new shared-directory `Account→Gateway` record that publishes the connection detail cluster-wide) both violate.

All three adversarial lenses attacked the choice and **could not refute it**; the leakage lens proved C is Pareto-minimal (the only shard that knows the client's gateway is the home shard, so the outline reaching the client through the home shard is the *minimum* necessary cross-realm leak, and A/B leak strictly more). The verdicts against A and B are grounded in code:

- **B is refuted on the facts.** `DirectoryKey` has exactly `Session | Entity | Realm | Ship` (`crates/wire/src/seams/directory.rs:16-26`) — **no Account key**; an `Entity` head resolves to `AuthorityRef::Shard(NodeId)`, never a gateway (`directory.rs:38-49`; `crates/sim/src/stub.rs:1565-1567`). The parent's proxy holds an `AccountId` (`RetainedOccupants` keyed by `AccountId`, `stub.rs:309`), and the only `account → session` lookup in the whole system is a **gateway-local RAM scan** unreadable by any shard (`crates/connection-plane/src/gateway.rs:2884`). The parent's periodic HeadRead is keyed `DirectoryKey::Realm(parent_coord.lowered())` (`stub.rs:4818`) — it returns the parent realm's shard, not a gateway. So B's "no wire growth" premise (as tentatively recorded in `DEFERRED.md:3179` and `vu_aoi_s2b_plan.md`) is **false**: B needs either a `SessionId` carried up the relay (Option A's exact wire cost, plus semantically broken — `SessionId` is ephemeral, minted per-connection at `gateway.rs:1902-1906`, revoked at Bye `gateway.rs:1967-1974`, and stale the instant the traveller re-homes, which is the whole S2c scenario) OR a brand-new durable `DirectoryKey::Account → Gateway` record that **publishes the client-connection detail into the shared cluster-wide authority directory** — the exact opposite of the user-decisive constraint. **Rejected.**
- **Option A leaks the connection detail up the tree.** It adds the client gateway (`d.gateway`, `stub.rs:4714`) to `OccupantInterest` and has the *parent* emit `RealmSceneDelta` straight to that gateway: the home realm's connection detail leaks upward into a frozen, high-frequency, unreliable relay; it creates a new parent→client-gateway edge; at 100K the parent becomes a second render-emitter to gateways it doesn't own, splitting render-to-client authority. **Rejected.**
- **Option C wins on every axis.** The parent already knows the relay **sender** `NodeId` — the transport-populated `from` on the inbound `OccupantInterest` frame (`stub.rs:1623`, in scope at the receive arm `stub.rs:1679`) — and it is **provably the home shard** hosting the dot: the only emitter of `OccupantInterest` is `push_occupant_interest` (`stub.rs:4834`), called only from the up-relay loop over `dots.0.filter(|(_,d)| d.authority.simulates())` (`stub.rs:4833`). Replying **down** to that `from` adds **no new edge** (it's the relay's own return address); the home shard already holds `d.gateway` for this dot and already streams `RealmSceneDelta` to it (`stub.rs:4710-4715, 4791-4805`), so it splices the sibling outline into that existing stream. The parent never learns the gateway; `d.gateway` never leaves the home shard; one render-to-client authority; `forward_realm_scene_delta` (`gateway.rs:2877`) reused unchanged; **no PROTO_MINOR bump** (the client sees the byte-identical `ServerControlMsg::RealmSceneDelta` at the existing minor 6, `channels.rs:147`). The sole new wire is one appended intra-cluster `InterShardFlow` arm.

---

## (C) The critical revision the adversaries forced — LEVEL, not EDGE

The original S2c design shipped an **edge-triggered** `ProxySceneDelta { added, removed }` addressed to the stored home. **All three lenses independently found the same HIGH defect**, and I confirmed it against the code:

**The route (`home`) goes dead mid-crossing, and an edge sent to it is delivered-then-dropped and never replayed to the new home.** Because every crossing is a cross-node re-home (memory: node-per-realm), the failing window is not an edge case — it fires on *every* boundary crossing, the exact acceptance scenario:

1. The dot re-homes A→B. The source retains the dot as `Frozen`/`Ghost` until dest Committed (fence discipline). `simulates()` is true only for `Owned`, so A stops relaying the instant it self-fences (`stub.rs:4833` filters `simulates()`), and B does not relay until it promotes to `Owned`. There is a multi-tick window where **neither endpoint relays** and P's stored `home` is still **A**.
2. During that window P's proxy cull keeps running against **moving** child placements (`child_placements(..., tick)`, `stub.rs:4721`), so a sibling can cross an AoI edge from orbital motion alone. An edge fires, addressed to `home = A`.
3. If the helper matches the dot with `d.authority.simulates()`, A finds no simulated dot → the edge is **dropped**. A missing `removed` → the client's sticky `BTreeMap` keeps a **permanent ghost outline** (`crates/client/src/realm_scene.rs:245-278` removes only ids explicitly in `removed`); a missing `added` → an invisible neighbour.
4. B promotes, `home` flips A→B via last-wins (`retain_occupant`, `stub.rs:5011` — a plain overwrite, **not** a prune), but the proxy's membership latch is keyed by durable `AccountId` and **survives the flip**, so P computes no new edge and **never re-emits**. `retain_live` (`stub.rs:4848`) never clears the latch because the proxy never left `observers`. The smoother the hand-off, the more permanent the bug.

A second lens proved the **same edge feed also loses `removed` on a parent crash or a byte-cap shed**: the TTL-refill self-heal the original cited recovers *adds* (re-latch from `was_in=false`, add re-emits) but **not removes** (a departed sibling is out of range post-refill, `was_in` reset to `false`, the remove-branch needs `was_in=true` → never re-derived). "RAM retry suffices, no outbox" is false for removes.

**The fix folded in: make the parent→home feed LEVEL-triggered (a self-healing full current set), and let the home shard reconcile it to the client edge.** This is the exact pattern the codebase already blesses for `RealmDemand` (level, so drops self-heal). Concretely:

- The parent ships **the full current in-range sibling set** for a proxy (`ProxySceneSet { observer, realms: Vec<RealmShape> }`), reliably, **only when the set changes for that observer or the home changed** (send-on-change — reliable-buffer-lean, so the correlated-burst reliable-retry pressure a lens flagged stays bounded and any shed message is reconciled by the next change).
- The home shard keeps a per-account **forwarded baseline** and reconciles: `added = set − baseline`, `removed = baseline − set`, forwards the byte-identical `ShardToGateway::RealmSceneDelta { observer, added, removed }` (client wire unchanged), then stores `baseline = set`.
- The home forward matches the dot by **account alone**, NOT gated on `simulates()` — so during the pre-commit re-home window (dot still in the source's `dots.0` as `Frozen`/`Ghost`, `d.gateway` still valid) the source **still reconciles and forwards P's shrinking set**, catching the departed sibling as a `removed` before it can ghost.
- On the flip A→B, B's baseline is empty; P's full set (auto-followed to `home=B` by last-wins) reconciles at B → the still-in-range siblings re-assert (idempotent by `RealmId`, sticky client → no flicker), the entered neighbour stays continuous.

**Why LEVEL closes every hole EDGE left open:**
- **Crash / shed:** P's per-observer last-sent cache is RAM; on restart (or after refill) it is empty → the next set is treated as changed → P resends the full set → the home reconciles a `removed` for the departed sibling. Self-heals both adds and removes.
- **Re-home added-half:** last-wins moves `home` to B automatically; a lens-requested explicit "new-home resync" is achieved for free — a home-change clears P's last-sent cache for that observer, forcing a full resend to B.
- **Re-home removed-half (common):** forward-by-account keeps the source reconciling through the pre-commit window.

**The residual (ledgered, NOT hidden):** a sibling that drops out of range in the narrow `[dest-Committed, home-flip-to-B]` sub-window — where the source has already dropped the dot from `dots.0` but P's last-wins has not yet moved `home` to B — is known to neither the old home (dot gone) nor the new home (never told). This is the **same systemic stale-render-across-authority-change corner the dot render path already has on a shard crash**, whose real answer is the VU-6 authoritative scene re-stream on a scene change. Ledgered `D-RLM-13`, deferred to VU-6. It is a bounded (≤ one relay cadence) transient, not the every-crossing permanent ghost EDGE produced.

**Correcting the original's "double-draw dedup" narrative (a lens found the premise false, and it was concealing the defect):** the original claimed "B's own shard also begins streaming B as one of its children" so two copies of B collapse by `RealmId`. **A shard streams its CHILDREN, never itself** (`child_placements(config.realm, …)` → `child_shape`, `stub.rs:4721-4759`); B is not a child of B. At single hop there is exactly **one** source per `RealmId` (a realm ⇢ its parent). The seamless-through-crossing property is delivered by **client-map stickiness** (`realm_scene.rs:258` insert-once) **plus last-wins home re-target**, NOT by two-source dedup. The plan below states the real mechanism.

---

## (D) The winner's wire shape, classes, lifecycle (Q2)

**One appended arm** in the single reviewed file `crates/wire/src/intershard.rs`:

- **`intershard.rs:281`** — after `OccupantInterest(OccupantInterest)`, append `ProxySceneSet(ProxySceneSet)`.
- **New struct beside `OccupantInterest`** (~`intershard.rs:619`):
  ```rust
  /// VU AoI S2c — the parent's LEVEL-triggered reflection of a proxy occupant's CURRENT in-range sibling
  /// set, sent DOWN to the home shard that relays the occupant (the up-relay's own return address). Public
  /// parent-authored geometry only — NO NodeId / gateway / session. `realms` is the FULL current set (not an
  /// edge): the home shard diffs it against its per-account forwarded baseline to derive the client
  /// add/remove, so a lost/shed message or a re-home hand-off self-heals on the next set — an edge would be
  /// delivered-then-dropped to a route that changes under the crossing. `observer` is the DURABLE traveller
  /// id — the SAME key the home shard already streams `RealmSceneDelta` under. Empty `realms` = "the proxy's
  /// AoI holds no sibling now" ⇒ the home reconciles all forwarded ids for the account to `removed`.
  pub struct ProxySceneSet {
      pub observer: AccountId,
      pub realms:   Vec<crate::channels::RealmShape>,
  }
  ```
  (`RealmShape` = `{ realm, frame, center, shape, parent }`, `channels.rs:224` — public geometry, "NEVER a NodeId/SubId/shard … NEVER server-only config".)
- **`effect_class` arm** (add beside `intershard.rs:478`): `InterShardFlow::ProxySceneSet(_) => EffectClass::FireAndForget`. No fence, no transfer trigger, no home-shard state mutation beyond render bookkeeping; a re-delivered set is idempotent (full-set reconcile).
- **`durability_class` arm** (add beside `intershard.rs:553`): `InterShardFlow::ProxySceneSet(_) => FlowDurabilityClass::ReDriven`. Reliable (the reconcile must not silently lose a set) and **re-driven, not producer-less** — on any loss/crash/shed P re-sends the full set from its refilled `retained`, and the home reconciles. **No durable outbox.** The golden producer-less pin stays exactly TWO (`crates/wire/tests/intershard_closed.rs:434,440,452`; shadow classifier `_ => ReDriven` already covers the new arm; `producer_less.len() == 2` unchanged).

**Carrier:** `MsgClass::Saga` (reliable/ordered — the lane every reliable `InterShardFlow` rides), received at the home shard's `MsgClass::Saga => on_directory_reply` dispatch (`stub.rs:1647`), decoded in the match at `stub.rs:4032`.

**Send-on-change lifecycle (parent side):** the parent already folds each proxy as `ObserverId::Proxy(acct)` and computes its per-child post-hysteresis membership (`stub.rs:4661-4769`). Accumulate, per proxy observer, the set of child realms whose `next_in` is true this tick (reusing the SAME cull — no new AoI math, HR3). Diff against a P-side per-observer last-sent id cache; emit `ProxySceneSet` to the observer's stored `home` only when the id set differs OR the home changed. The client wire (`ServerControlMsg::RealmSceneDelta`) is byte-identical; only the intra-cluster P→home arm is new.

---

## (E) Byte-identity / inertness + the cross & teardown (Q3)

**Inert at walk/static — byte-identical.** With `AoiConfig::inert()` (spin-up radius 0) the whole up-flow is dormant: `aoi_live()` is false → `parent_headread_due` returns `None` (`stub.rs:4813`) → `ParentRealmNode` stays `None` → the up-relay never fires → no `OccupantInterest` is emitted → `RetainedOccupants` stays empty (`stub.rs:4653,4661`) → `proxy_observers` is empty → no proxy in-range set accumulates → the P-side last-sent cache stays empty → **no `ProxySceneSet` is ever emitted**; the home's `ForwardedProxyScene` stays empty; the S2c decode arm is reached only by a `ProxySceneSet` frame (none). `home` is written into `RetainedOccupant` only on a received relay (none). Every S2c path is gated on "the retained store is non-empty," exactly the S2b inertness argument (`vu_aoi_s2b_plan.md` §E). Zero new emitted bytes; the whole-cluster walk-scale golden byte-identity gate stays green.

**The cross (traveller enters sibling B, a child of parent P) — seamless, self-consistent:**
1. The dot re-homes off home shard A onto B. **Pre-commit:** the dot is `Frozen`/`Ghost` in A's `dots.0` with `d.gateway` still valid; `home` in P is still A. P's cull emits its shrinking sibling set to A; A reconciles by-account (no `simulates()` gate) and forwards — so any sibling that leaves range during this window is removed on the client cleanly.
2. **Commit → B promotes → B relays `OccupantInterest` up to the same parent P** → last-wins flips `home` A→B and clears P's last-sent cache for the account → P re-sends the full current set to B.
3. B's forwarded baseline is empty → B reconciles the full set as `added` → the still-in-range siblings (already on the client, sticky) are re-asserted idempotently by `RealmId` (`realm_scene.rs:250-268`, `boxes.insert(s.realm, …)`) → **no flicker, no double-draw** (exactly one source per `RealmId`). B itself — the realm the traveller is now inside — is P's child, still in the proxy's AoI (distance ≈ 0), so P keeps forwarding B's outline down through B; continuous.

**Teardown on back-away (no cross).** The sibling leaves the proxy's AoI → P's next set omits it → the set changed → P emits the smaller `ProxySceneSet` → the home reconciles `removed = baseline − set` → the client's `with_delta` drops that `RealmId` (`realm_scene.rs:251-253`). Same path as add; self-heals a lost message on the next change.

**Exploiting the RealmId union.** The client unions every source into one `BTreeMap<RealmId, RealmBox>` with no central assembler, so (a) the forwarded sibling and any authoritative copy collapse to one box, and (b) re-asserting the full set after a hand-off is idempotent. S2c never sends a second source for the same id; the union is what makes the hand-off re-assert safe.

**Two ledgered teardown residuals (→ VU-6, D-RLM-13):**
- The `[Committed, home-flip]` departed-sibling sub-window (§C) — a bounded transient ghost, the same class as the dot path's shard-crash stale-render.
- **A's own interior children after the cross** (a lens confirmed this, and it is **pre-existing S1b**, not introduced by S2c): while the traveller is in A, A streams A's *children* (P's grandchildren) via `render_routes`; on re-home the dot leaves `dots.0` so the render emit loop (`stub.rs:4791`, iterates present dots only) sends no `removed`, and `retain_live` silently drops the membership keys (`stub.rs:4848`). P never knew A's children (out of single-hop range) and B does not. So A's interior sub-boxes linger until the VU-6 scene re-stream. **Confirmed real, pre-existing, ledgered — not silently assumed covered by the RealmId union (no `removed` is ever sent, so the union cannot drop them).**

Both residuals are the natural property of VU-6 ("authoritative re-draw on a scene/authority change"). S2c's crisp deliverable — the neighbour drawn on approach, continuous through the crossing for the entered neighbour and still-in-range siblings, torn down on back-away — is met without them.

---

## (F) The gated sub-slices (ordered; each: unit + integration tests to Tier-A 100%, `just gate` clean, then commit)

### S2c-i — WIRE the level arm + STORE the return address. Pure additive, provably inert.
- Append `ProxySceneSet(ProxySceneSet)` + the struct + the two classifier arms (`intershard.rs:281, 478, 553, ~619`).
- Add `home: NodeId` to `RetainedOccupant` (`stub.rs:315`). Grow `retain_occupant` by one `home: NodeId` param (`stub.rs:4997`); store it in the insert (`stub.rs:5011`). **On a home change** for an existing account (stored `home != incoming`), this is where the P-side last-sent cache is cleared in S2c-ii — for S2c-i just store `home` (last-wins). Thread `*from` at the call site (`stub.rs:1680` → `retain_occupant(&mut retained, &config, oi, clock.local_tick, *from, &mut stats)`); `from` is transport metadata already at the receive edge — `OccupantInterest` stays byte-frozen, no wire grows.
- **Tests to 100%:** add `ProxySceneSet` to the all-flows roundtrip census (`intershard_closed.rs:296` list + the `| InterShardFlow::… => {}` fixture arm at `:347`); an `assert_eq` equality test for both classifier arms mirroring `OccupantInterest`; golden pin unchanged (`producer_less.len()==2`); extend `retain_occupant_matches_misroutes_and_is_last_wins` (`stub.rs:12775`) to assert the stored `home` and that a second relay from a different node last-wins-overwrites it; walk-scale golden byte-identity gate stays green (arm defined, nothing emits it; `home` written only on a received relay → none at walk/static → inert).
- **Gate + commit.**

### S2c-ii — EMIT the level set down (parent side). Send-on-change. Inert at walk/static.
- New resource `ProxySentScene(BTreeMap<AccountId, BTreeSet<RealmId>>)` — P-side render bookkeeping (NOT authority: no fence, never persisted, `BTreeMap`), the per-observer last-sent id set. Bundled into the existing VU-AoI tuple `SystemParam` if the 16-param ceiling requires (mirror `vu_aoi` at `stub.rs:1616`).
- In the cull, build `proxy_home: BTreeMap<AccountId, NodeId>` from `retained` (`acct → entry.home`), non-empty only when `retained` is non-empty (empty at walk/static → byte-identical).
- In the per-observer transition loop (`stub.rs:4737-4769`), for an `ObserverId::Proxy(acct)` observer accumulate `proxy_now: BTreeMap<AccountId, Vec<RealmShape>>` — push `child_shape(region)` when `next_in` is true. **Disjoint from the dot render path**: the dot branch stays gated on `render_routes.contains_key(obs)` (`stub.rs:4754`), the proxy branch on `proxy_home.contains_key`-by-account; the dot path's bytes are unchanged.
- On a home change detected in `retain_occupant` (S2c-i), clear `ProxySentScene` for that account so the next tick treats the set as changed (forces a full resend to the new home). *(Wire the clear here in S2c-ii; the detection point is the S2c-i insert.)*
- After the dot emit loop (`stub.rs:4806`), add a proxy emit loop: for each proxy account, `new_ids = proxy_now[acct].map(.realm)`; if `new_ids != ProxySentScene[acct]`, `outbox.push_flow(home, MsgClass::Saga, &InterShardFlow::ProxySceneSet(ProxySceneSet { observer: acct, realms: proxy_now[acct] }))` and set `ProxySentScene[acct] = new_ids`. Unchanged set → skip (no emit). A proxy that TTL-expired from `retained` this tick → drop its `ProxySentScene` entry (no client-facing consequence; the home drops its baseline on dot-leave).
- **Tests to 100%:** an armed-proxy unit test driving a proxy across a sibling AoI edge → exactly one `ProxySceneSet { realms:[sibling] }` to the stored `home`; back-away → the smaller set (sibling absent); unchanged tick → no emit (send-on-change); a **home-change** test (relay from node A then node B for the same account) → the cache clears and the full set resends to B; walk-scale byte-identity green (proxy set empty → no accumulation, no emit).
- **Gate + commit.**

### S2c-iii — FORWARD + reconcile at the home (receive side) + end-to-end.
- New resource `ForwardedProxyScene(BTreeMap<AccountId, BTreeSet<RealmId>>)` — home-shard render bookkeeping (NOT authority), the per-account forwarded baseline. Drop an account's entry when the dot leaves `dots.0` (checked in the existing dot-lifecycle path or lazily on the next reconcile miss).
- In `on_directory_reply`'s decode match (`stub.rs:4032`), add `Ok(InterShardFlow::ProxySceneSet(pss)) => { on_proxy_scene_set(pss, &dots, &mut forwarded, &mut stats, &mut outbox); return; }`, keeping the generic `postcard::from_bytes` arm straight (HR5 — all branching in the helper).
- New monomorphic helper `on_proxy_scene_set`: find the dot by **account alone** (`dots.0.iter().find(|(_, d)| d.account == pss.observer)` — NO `simulates()` gate, so the pre-commit re-home window still forwards); if absent → `stats.<counter> += 1`, drop, and clear any `forwarded` entry for the account (a departed dot). If found: `new_ids = pss.realms.map(.realm)`; `added = pss.realms.filter(|s| !baseline.contains(&s.realm))`; `removed = baseline.iter().filter(|r| !new_ids.contains(r))`; if either non-empty, `push_session_reply(outbox, d.gateway, &ShardToGateway::RealmSceneDelta { observer: d.account, added, removed })` — the existing shape (`session_flow.rs:137`), the unchanged gateway forward (`gateway.rs:2861,2877`); set `baseline = new_ids`.
- **Tests to 100%:** found + fresh baseline → forwards the full set as `added`; found + shrinking set → forwards the departed as `removed`; found + unchanged → no emit; not-found → counter + no emit + baseline cleared; a **re-home reconcile** test — set forwarded via node A (baseline built), then the same account's dot moves to node B with an empty baseline and a full set → B forwards the full `added` (no flicker); an **end-to-end** scenario: up-relay → P folds + emits `ProxySceneSet` down → home reconciles + forwards `RealmSceneDelta` → gateway forwards `ServerControlMsg::RealmSceneDelta` → client `with_delta` draws the sibling box; cross → last-wins re-home + full-set reconcile at B → `with_delta` union dedup by `RealmId` (no double-draw); a **crash/refill** test — clear P's `ProxySentScene`, re-emit the full set, assert the home reconciles a `removed` for a now-absent sibling; walk-scale golden byte-identity gate stays green.
- **Gate + commit.**

---

## (G) HR-conformance, scale, deferrals, ledger (Q4, Q5)

**HR1 — the home shard forwarding an outline for a realm it does NOT own is a legitimate relay, not a breach.** The payload is `RealmShape` — public parent-authored geometry (`channels.rs:224`, "NEVER a NodeId/SubId/shard … NEVER server-only config"), the identical bytes the parent already ships to its own dot observers. The home shard treats it as **opaque render payload**: it does not parse it into `RealmRegions`/`Dots`/`OwnedTransients`, does not persist it, mints no region, authors no fence, mutates no sim state — it only re-emits the `RealmSceneDelta` it already emits to a gateway it already owns. This is the mirror of `OccupantInterest` riding the seam upward, and structurally identical to the gateway forwarding opaque `RealmFrame` bytes it never decodes (`session_flow.rs:116-128`). One appended `InterShardFlow` arm in the one reviewed file — the HR1 seam. No shard reads another's private World/rapier/redb.

**HR5 — no uncoverable region.** The two classifier arms are single equality-tested expressions (mirroring `OccupantInterest`); `on_proxy_scene_set` is a monomorphic helper with all branches (found→reconcile→emit / found→no-change→skip / not-found→count+clear) unit-covered; the generic `postcard::from_bytes` decode arm stays a straight call; the proxy emit loop's changed/unchanged/expire branches are all driven by tests. The golden pin's `_ => ReDriven` covers the new arm with `producer_less.len()==2` unchanged.

**Frozen wire.** The sole new wire is an APPENDED `InterShardFlow` variant (intra-cluster, single-binary greenfield — safe postcard append, no negotiation). No trailing struct-field append; `OccupantInterest` is untouched; the client-facing `ServerControlMsg::RealmSceneDelta` is byte-identical at the existing minor 6 → **no PROTO_MINOR bump**. Strictly less wire surface than A (grows the frozen/negotiated up-relay) or B (grows the shared directory).

**Scale (100K).** Fan-out is tree-preserving and bounded: the parent sends `ProxySceneSet` **only** to each proxy's stored `home` — its direct-child home shards (occupants inside its own direct children, a TTL-bounded local crowd, `stub.rs:4652-4653`). Send-on-change (not per-tick) keeps the reliable-retry buffer bounded under a correlated burst (a lens's concern) — bursts occur only on actual set changes, and a shed message is reconciled by the next change (no un-re-driven remove, unlike the edge design). Two O(n) scans, both bounded and both matching an existing ledgered pattern, **folded under the one existing scale ledger item `VU-AoI-scale`** rather than re-normalized per-site: (a) the home-shard `dots.0` by-account find in `on_proxy_scene_set` (bounded by dots-per-shard; `Dots` is `BTreeMap<SessionId, Dot>`, `stub.rs:263`, so the account find is linear — the same latent lookup as the gateway's `sessions.by_session.values().find(|s| s.account == observer)` at `gateway.rs:2882`, already self-documented as owing an `AccountId → SessionId` index); (b) the parent's `proxy_home`/`proxy_now` build (bounded by the local crowd). **The `AccountId → SessionId` (and `AccountId → SessionId` on `Dots`) secondary index is the discharge of `VU-AoI-scale`, committed when the density fixture forces it — this slice adds the third caller, it does not add a new scale wall.**

**Q4 — what S2c proves vs what rides S3.** The up-relay iterates `dots.0` (`stub.rs:4833`) — single hop, direct-child only. **S2c proves single-hop sibling render:** a direct parent reflects its own children's outlines down (as a self-healing level set) to the one direct-child home shard it already exchanges `OccupantInterest` with, and the home reconciles them onto the client. A deep traveller (district→city→planet) whose **grandparent** wants to reflect a **grand-sibling** needs (a) the coarsen ladder to feed the grandparent's proxy and (b) a two-hop down-reflect (grandparent→city→home). Both ride **S3**; `ProxySceneSet` is the reusable primitive, re-forwarded + re-reconciled down each level.

**Deferrals / ledger:**
- **Flip `D-RLM-11` 🟥 → 🟩** on S2c-iii landing — single-hop sibling render to a non-hosted client is done. In the flip, **correct the recorded prose**: `DEFERRED.md:3179` and `vu_aoi_s2b_plan.md` list "(B, recommended) … NO wire growth"; the S2c directory verdict **refutes** it (no `Account→Gateway` resolution exists; B needs a new shared-directory record that un-seals the connection detail). Record **Option C chosen** with the seal/topology/wire justification, and that the s2b plan explicitly deferred the final call to S2c.
- **Open `D-RLM-13` 🟥 → VU-6:** the crossing scene-reconcile residual — (i) a sibling that departs range in the `[dest-Committed, home-flip]` sub-window (bounded transient ghost), and (ii) the old home's own interior children lingering after re-home (pre-existing S1b render-teardown hole). Both are the authoritative-re-draw-on-scene-change that VU-6 (warp re-stream / scene-swap) owns; the same class as the dot render path's shard-crash stale-render. Pin: this plan §C/§E + the `on_proxy_scene_set` and `ForwardedProxyScene` doc comments.
- **Multi-hop / grand-sibling render → S3** (coarsen ladder + two-hop down-reflect; `ProxySceneSet` re-forwarded per level).
- **`D-RLM-12`** (coarsen-ladder proxy compose-`Err` anti-flicker) → S3, unchanged.
- **`D-RLM-10`** (`path()`-keyed directory) unaffected — the proxy store already keys on durable `AccountId`.

**Naive-implementation traps flagged (do NOT):**
- carry the gateway/session up the relay (Option A / B1 — leaks the connection detail, staleness under re-home);
- add an `Account→Gateway` directory record (B2 — un-seals the connection detail cluster-wide);
- have the **parent** emit `RealmSceneDelta` to a gateway (a new parent→gateway edge, two render authorities);
- ship the feed **edge-triggered** (`{added, removed}`) — the every-crossing route-goes-dead ghost + the crash-loses-removes hole; ship the **level set** (`{realms}`) reconciled at the home;
- gate the home forward on `d.authority.simulates()` — drops the pre-commit re-home removes; match by **account alone**;
- flush the old home's forwarded set on dot-leave to "clean up" — risks a remove-then-readd flicker of still-in-range siblings; rely on the new-home full-set reconcile + client stickiness, and hand the genuine residual to VU-6;
- parse the forwarded `RealmShape` into the home's own sim / persist it (HR1 breach — opaque re-emit only);
- branch inside the generic `postcard::from_bytes` arm rather than `on_proxy_scene_set` (HR5);
- merge the dot and proxy render-route maps into one (risks changing the dot path's bytes — keep them disjoint);
- classify `ProxySceneSet` as `Unreliable` (a lost set-change blinks the neighbour) or `ProducerLessReliable` (breaks the golden-pin-2 and forces a needless durable outbox — the level reconcile + RAM re-send is the correct recovery).

---

## (H) Adversary disposition (each finding: CONFIRMED-folded or REFUTED)

- **[leakage lens] C is Pareto-minimal; A/B leak strictly more — CONFIRMED, no change** (the option decision stands; the plan cites the proof).
- **[all three] HIGH: edge feed addressed to a route that dies mid-crossing → dropped, never replayed → permanent ghost / invisible neighbour — CONFIRMED, folded.** Fix: level-triggered `ProxySceneSet` (full set) reconciled at the home + forward-by-account (no `simulates()` gate) + last-wins home auto-follow with a cache-clear resync. New sub-slice structure (S2c-ii is now the level emitter; S2c-iii the reconciler).
- **[scale lens] MEDIUM: `ReDriven`/TTL-refill recovers adds but NOT removes on crash/shed — CONFIRMED, folded.** The level full-set reconcile self-heals removes (P's empty last-sent cache after restart → full resend → home reconciles the departed to `removed`); a crash/refill test is required. The "RAM retry suffices" prose is corrected.
- **[scale lens] MEDIUM: O(T²) by-account scans under a correlated burst — CONFIRMED, folded as scope-note.** Send-on-change bounds the emit rate; the by-account find is the third instance of the ledgered `VU-AoI-scale` unindexed lookup and is discharged by the `AccountId→SessionId` index under that existing item — not re-normalized per-site.
- **[scale lens] LOW: reliable-retry buffer burst / shed re-creates the stale remove — CONFIRMED, folded.** Send-on-change keeps the buffer bounded; a shed set-change is reconciled by the next change (level self-heal), so a shed message no longer produces a permanent stale outline.
- **[scale lens + correctness lens] the "B streams B as its child" double-draw premise is FALSE — CONFIRMED, folded.** The plan states the real mechanism (client-map stickiness + last-wins home re-target; exactly one source per `RealmId` at single hop).
- **[correctness lens] MEDIUM: the old home never emits `removed` for its OWN children on re-home (pre-existing S1b) — CONFIRMED, ledgered.** Real, pre-existing, not introduced by S2c; opened as `D-RLM-13`(ii) → VU-6; explicitly NOT assumed covered by the RealmId union.
- **[leakage lens] LOW: reliability rationale prose inconsistency ("clean no-op" vs "reliable so it's delivered") — CONFIRMED, folded.** §C states reliability guarantees delivery to the *addressed* node, not the *currently-hosting* node; the level reconcile is what makes correctness independent of which node was addressed.
- **REFUTED: none.** No adversary showed Option C wrong; all three affirmed it and attacked only the (edge) feed mechanism, which is now replaced by the level feed.
