# VU AoI S2b — parent RETAINS the relayed occupant + FOLDS it as a proxy (the up-flow payoff)

Vetted via the `vu-aoi-s2b-design` workflow (Opus, 2026-07-28): 4 parallel readers → lead design →
4 adversary lenses (correctness-frames / flicker-TTL / render-path / hr1-hr5-scale) → synthesis. The
adversarial pass caught a CRITICAL flaw in the draft (the Empty early-return skipped the proxy cull —
the district-sibling payoff would never have shipped) which the synthesis folded. All `file:line` =
`crates/sim/src/stub.rs` unless noted. This is the implementer-ready contract.

All contested claims verified against HEAD. Key confirmations: the `observers.is_empty()` early-return at stub.rs:4622-4632 fires on the local-only set; `retain_live` (stub.rs:3127-3129) is an unconditional hard-evict with no grace path; `ancestor_close`/`!has_desired_descendant` (rlm.rs:485,498-512) keeps a parent alive for any desired descendant independent of its Empty report; `GRACE_TICKS_FLOOR = 1` not 2 (worldgen.rs:884); `child_shape` draws the box from the parent's own seed roster (stub.rs:4648,4776); `frame_context`/`child_placements` share the `parent.is_none()` root frame (stub.rs:683-702,743-752); `ObserverId` has no exhaustive match (only keyed at 4610/4616/4641). Here is the final synthesized contract.

---

# S2b — Parent USES the relayed occupant (retain + fold-as-proxy). FINAL implementer contract.

*Synthesized from the base design + four adversary lenses (correctness-and-frames, flicker-and-ttl, render-path-to-client, hr1-hr5-scale-rehome), each finding re-verified against HEAD. This file supersedes the draft.*

---

## 0. Adversary finding ledger (CONFIRMED → folded / REFUTED → why)

**F-A1 / F-C1 / F-H1 (three lenses, independently) — CRITICAL: the `Empty` early-return skips the proxy cull; the district-sibling payoff never ships. — CONFIRMED.**
Verified: stub.rs:4622-4632 is `if observers.is_empty() { push_demand(Empty); return; }`. The traveller is a dot on the *child* shard, never in the parent's `dots.0` (the up-relay ships the child's own dots, stub.rs:4755). So in the lone-traveller scenario the parent's local observer set is empty at exactly the acting tick. The draft's Q4 "test emptiness on `local_observers`, early-return, *then* chain proxies" returns before the child-placement cull loop (stub.rs:4648-4707) ever runs — no `SpinUp` for the sibling → the neighbour is never warmed. The draft is also self-contradictory: Q4 says a proxy must NOT suppress Empty, Q6 says it MUST. The fear behind Q4 (a proxy-only parent "never tears down") is refuted by real code: `ancestor_close` (rlm.rs:498-512) pulls every desired child's whole parent chain into the desired closure, and teardown is gated on `!facts.has_desired_descendant` (rlm.rs:485) — a parent with a desired child is never reaped *even while it self-reports Empty*. **Fold:** adopt Q6. Fold alive proxies into `observers` **before** the emptiness gate; test `observers.is_empty()` on the combined set. Delete the Q4 "early-return on local_observers" prescription and its "Empty excludes proxy" test; replace with "a live proxy participates in the union; Empty is reported only when local ∪ proxies is empty; the Empty report is harmless under ancestor-closure."

**F-B1 — HIGH: "grace ages the skipped-proxy latch" is false; `retain_live` hard-evicts. — CONFIRMED (rationale defect); behavioral blink is OUT OF S2b SCOPE.**
Verified: `retain_live` = `map.retain(|k,_| live.contains(k))` (stub.rs:3127-3129), no grace branch. A proxy absent from `live_keys` for one tick (a compose `Err`, or the TTL-lapse tick) has its `(Proxy, child_path)` latches hard-deleted, not aged through `aoi_transition`. **Fold the rationale fix:** delete every draft sentence claiming `aoi_transition` grace protects a skipped proxy. **On the behavioral blink:** it requires a compose `Err` that *oscillates* tick-to-tick. In S2b's actual scope this cannot occur: an alive proxy's child is the realm the player *occupies*, hence within the parent's spin-up radius, hence always in the parent's region roster → `transfer_frame` always returns `Ok` (proven by a unit test, S2b-iii). `Err` is reachable only for out-of-scope relays (a coarsen-ladder grand-child the parent doesn't place → S3; a stale post-re-home relay → correctly dropped). **Fold:** on `Err`, skip the proxy this tick and accept the hard-evict; state plainly it is unreachable in scope; **defer the coarsen-ladder anti-flicker** (carry-forward / last-good-composed-position) to S3 with a ledger entry (D-RLM-S2b-flicker). Do NOT claim grace covers it.

**F-B2 — MEDIUM: "grace ramps demand down after departure / ~1 s warm-hold" is wrong. — CONFIRMED.**
Verified: on genuine departure the child stops relaying, the store keeps `occupant` unchanged (never re-relayed), and the fold re-composes the same in-range offset every tick → `now_in` stays true → the `aoi_transition` `(true,false)` countdown (stub.rs:4683-4690 branch) is never taken. The sibling is held at full `KeepAlive` for the whole TTL window, then hard-cut at TTL-lapse. **Fold:** restate the warm-hold bound as **TTL + the reconciler's own empty-grace** (`empty_grace_ticks`, rlm.rs:482), not the region AoI `grace_ticks`; note the hard-cut at TTL-lapse — not `aoi_transition` — is what ends the proxy's demand. Still bounded (TTL prune + non-durable restart), so the anti-warm-forever bar holds; only the stated mechanism was wrong.

**F-B3 — LOW: `RETAIN_TTL_FLOOR = 2` mis-described as "analogous to `GRACE_TICKS_FLOOR`". — CONFIRMED.**
Verified: `GRACE_TICKS_FLOOR = 1` (worldgen.rs:884), not 2. Also `retain_ttl_ticks = max(grace_ticks_from_seconds(1.0, dt), 2)` = 20 @ 50 Hz / 50 @ 20 Hz (worldgen.rs test rates), so the floor never operates at a real cluster rate — the 1 s derived term is what bridges packet loss. **Fold:** drop the false "analogous to :884"; credit the derived 1 s term for the survive-one-loss guarantee; justify the floor of 2 *independently* as "≥ 2 so a one-tick datagram gap is bridged regardless of intra-tick prune-vs-receive ordering; operative only under a degenerate test `dt`."

**F-C2 — MEDIUM: Q5 "seamless bars hold, only draw-ahead defers" over-claims for the district-sibling case. — CONFIRMED.**
Verified render routing: deltas emit only for `render_routes` entries, which is dots-only (stub.rs:4637-4642, 4713); `OccupantInterest` carries no gateway (intershard.rs:606-619); a proxy has no route. So while the player is inside district A, the parent cannot draw sibling B to A's client — B first renders when the player crosses into B (becomes a real dot whose shard/ancestors route it). For adjacent tiling districts that is "invisible until crossed, then pop," not "a beat early vs at the crossing." **Crucial nuance (verified):** `child_shape` (stub.rs:4776) builds B's box from the *parent's own seed-derived region roster* (`placements`, stub.rs:4648) — it never needed B's node. So S2b's warm-ahead buys B's **node/interior readiness** at descent, NOT B's box outline. The box outline is the render concern S2c owns. **Fold:** keep lifecycle-only slicing (a legitimate gate boundary), but (a) reclassify **S2c from optional polish to a REQUIRED companion** before any rendering client exercises this path; (b) gate the pixel-visible dot-crosses-a-boundary acceptance test on **S2b + S2c together**; (c) correct the plain-language + Q5 prose to state the no-pop bar for the sibling is **not met until S2c**.

**F-H-secondary — a proxy-only parent's OWN survival above one hop rides the S3 coarsen ladder. — CONFIRMED as a scoping note.** The up-relay iterates `dots.0` (stub.rs:4755), so a proxy is not re-relayed to the grandparent; S2b's single-hop sibling-warm is self-contained, but the parent's own keep-alive beyond one hop depends on S3. **Fold:** state it explicitly; S2b's gates do not prove multi-hop.

**REFINED (not adopted verbatim): the F-H bitwise-`&` emptiness gate.** F-H proposed `local.is_empty() & proxies.is_empty()`. **Refuted as unnecessary:** building `observers = local ∪ alive_proxies` and testing the single `observers.is_empty()` (stub.rs:4622) is one already-coverable branch with no compound boolean — strictly simpler and HR5-clean. Adopt the single-set test, not the compound `&`.

**Everything else SOUND (attacked, survived):** the retention store shape/key/clock (`BTreeMap<AccountId,…>`, `clock.local_tick`); frame composition via `transfer_frame` with the `parent.is_none()` root frame commensurate with `child_placements` (stub.rs:687/746); moving-child zero-special-code via live `orbital_state`; `Err`→drop (NOT `rebind_pose_to_dest`'s `unwrap_or(pose)` pass-through, frame.rs:272); HR1 structural sealing (proxy never in `Dots`/`OwnedTransients`, never persisted, never a transfer subject); the `render_routes.contains_key(obs)` render-accumulation guard; no re-relay loop-back; the `to_realm.lowered()` mis-route guard; `ObserverId::Proxy` adds no exhaustive-match arm; 100K scale (a parent's store holds only occupants inside *its own direct children*, a local crowd); and full inertness at walk/static.

---

## (A) Plain-language summary (for a human)

Right now, every tick a child realm's shard tells its parent "my player is right here," and the parent counts the note and throws it away. This slice makes the parent **keep the last note for a short, self-erasing moment** and **treat that remembered player as a stand-in observer of its own** — so the parent warms up the neighbouring district and the rest of the star system *before* the traveller physically reaches them, and never lets an edge realm blink out because one note got lost.

What visibly works after this slice: a player walking toward the edge of their district makes the **neighbour district next door spin up ahead of them**, driven entirely by the parent's own culling. The neighbour's world is *running and ready* the instant the player crosses in — no cold load, no hitch.

What this slice deliberately does **not** do yet: it does not **draw** that neighbour on the traveller's screen while they are still inside their own district. Warming a realm (so its contents are ready) and painting its outline on a screen the parent doesn't own are two different jobs; the second needs the parent to learn how to address a client it isn't hosting, which is a clean, separately-shippable follow-on. Being honest about the consequence: until that follow-on lands, the neighbour district's outline appears at the moment the player crosses the boundary rather than as they approach it. So the two must be treated as a pair — **the "no pop" promise for the neighbour district is only fully met once both this slice and its render companion are in.** This slice delivers the readiness half (the neighbour is never *cold*); the companion delivers the visible half (the neighbour is *seen* as you approach).

Three safety bars, all held and all re-checked against the real code:
- **Invisible during ordinary play.** At walking scale the interest radius is zero, so no notes are ever sent; the parent's memory box is never written and folds nothing. Byte-for-byte identical output, proven by the existing walk-scale gate plus a new "the box stays empty" test.
- **No blinking under a lost note.** The remembering window is derived from the one loiter-duration value already in the code (one second), never invented, and can never fall below "survive one missed note." A lost note simply leaves the last remembered position in place — the parent keeps warming, nothing blinks.
- **A stand-in is never a second boss.** The remembered player is pure culling input — never stored as an owned entity, never triggers a hand-off, never written to disk. If the parent restarts, the box is simply empty and refills within one window.

---

## (B) The six hard questions — decisions AS REVISED (one line + rationale each)

**Q1 — Retention store.** A new in-memory bevy `Resource RetainedOccupants(BTreeMap<AccountId, RetainedOccupant>)` holding `{ occupant: StampedPose (as relayed, child frame), last_seen: TickId (parent local_tick) }`, keyed by the durable `AccountId`, written from the `SignalDelta` receive arm (stub.rs:1639-1642) through a monomorphic `retain_occupant` helper. *Rationale:* durable key survives re-home and matches the render-route identity; last-wins collapses the brief dual-relay during a cross; `BTreeMap` for sim determinism; a foreign pose only — no fence, never an owned entity, never persisted (HR1).

**Q2 — TTL / anti-flicker.** `ttl_ticks = max(grace_ticks_from_seconds(WALK_DEMAND_AOI_GRACE_S, config.tick_dt_s), RETAIN_TTL_FLOOR=2)`, aged on `clock.local_tick`, alive iff `local_tick − last_seen ≤ ttl`. *Rationale:* reuses the one loiter constant (1 s, worldgen.rs:878) so it is never a magic number and is automatically consistent with the region grace an armed shard carries; the derived 1 s term (20-50 ticks) is what actually bridges a lost `Unreliable` datagram; the floor of 2 is an independent degenerate-`dt` guard for one-tick prune-ordering, **not** analogous to `GRACE_TICKS_FLOOR`=1. A lost datagram never hits `retain_live`'s hard-evict because the store still holds the last pose and the fold re-composes it successfully; the true warm-hold after a *genuine departure* is **TTL + the reconciler's empty-grace**, ended by the hard-cut at TTL-lapse (not by `aoi_transition`).

**Q3 — Frame composition.** Per tick, build `ctx = regions.frame_context(tick_hz)` once and lift each retained occupant with `transfer_frame(&entry.occupant, root_frame, &ctx)` where `root_frame` is the `parent.is_none()` region's frame (the same frame `child_placements` stamps children in, stub.rs:687/746), feeding `result.pos.offset()` + `result.vel` as the proxy `(pos, vel)`; movers need zero special code (live `orbital_state`); **`Err(UnknownSourceFrame)` ⇒ drop the proxy this tick** (not `unwrap_or(pose)` pass-through). *Rationale:* commensurate with the sibling cull metric; a raw child-frame offset fed into the parent metric would spuriously warm the wrong siblings — the exact forbidden failure; the `?`/match live in the already-monomorphic `transfer_frame_resolved` (frame.rs:117-127), so the generic shim adds no per-mono branch. In S2b scope `Err` is unreachable (an occupied child is always in the parent roster — unit-tested); the coarsen-ladder anti-flicker for out-of-scope `Err` relays is deferred to S3.

**Q4 — Proxy identity + fold.** Add `ObserverId::Proxy(AccountId)` and fold alive+composable proxies into the `observers` union **before** the emptiness gate (stub.rs:4622), via a `.chain()` mapped through the monomorphic `proxy_observer` (skipping `None`); gate the render-accumulation lines (stub.rs:4677-4682) on `render_routes.contains_key(obs)` so a proxy runs the AoI/latch math but no render delta. *Rationale:* one insertion carries the proxy through `was_demanded`, the cull loop, its own hysteresis latch, and `retain_live` for free; folding **before** the gate is what makes the payoff ship (the parent's local set is empty in the target scenario); the proxy never enters `Dots`/`OwnedTransients` so it can never re-home or emit a mis-routed delta (HR1, structural); a live proxy legitimately keeps `observers` non-empty, and its Empty-suppression is harmless because ancestor-closure independently keeps the parent alive.

**Q5 — Sibling render scope.** S2b is **LIFECYCLE-ONLY**; sibling render is a **required companion S2c**, not optional polish. *Rationale:* the parent structurally cannot address the in-A client (no gateway in `OccupantInterest`; a proxy has no `render_routes` entry); the contract's named payoff — "the district sibling case ships end-to-end" — *is* the lifecycle warm-ahead, which ships whole here with zero wire growth; but the neighbour's *box outline* (drawn by the parent from its own roster via `child_shape`, needing no child node) is invisible until the crossing without S2c, so the no-pop bar for the sibling is met only by S2b+S2c together, and the pixel-visible acceptance test gates on both. S2c options: **(B, recommended)** resolve the observer's gateway on the parent via a directory HeadRead reusing the `ParentRealmNode` cadence machinery (no wire growth), vs **(A)** grow the wire additively (`OccupantInterestV2` variant or a `PROTO_MINOR`-gated gateway field — never a silently-appended trailing field). Decide finally at S2c.

**Q6 — Byte-identity / HR1 / scale / re-home.** Structurally inert at walk/static (no interest emitted ⇒ store never written ⇒ zero folded rows ⇒ byte-identical); HR1 held (pure AoI input, no fence, no persistence); 100K-safe (a local per-parent crowd, `BTreeMap` + last-wins + TTL prune); re-home handled two ways — the non-durable store starts empty on restart (no stale resurrect), and a recycled-NodeId mis-delivery is rejected on receive by `oi.to_realm.lowered() == config.own_coord.lowered()` (+ the compose-`Err` as a second implicit guard). *Rationale:* the non-durability *is* the re-home guard (persisting would be the HR1 violation); TTL expiry is the sole thing that re-enables an idle parent's Empty→teardown, so one derived value discharges both anti-flicker and anti-warm-forever.

---

## (C) Gated sub-slices (ordered; each: `cargo test -p vd-sim` + touched-crate tests, `cargo clippy --workspace -- -D warnings`, `just coverage-fast` to Tier-A 100% region+branch, then commit)

### S2b-i — RETAIN (store + receive write + mis-route guard). No fold yet — pure additive, provably inert.

**Contract (file:line):**
- Declare `RetainedOccupants` + `RetainedOccupant` adjacent to `AoiMembership` (stub.rs:~586). `#[derive(Resource, Debug, Default)] pub struct RetainedOccupants(pub BTreeMap<AccountId, RetainedOccupant>);` and `#[derive(Clone, Debug)] pub struct RetainedOccupant { pub occupant: StampedPose, pub last_seen: TickId }`.
- `world.insert_resource(RetainedOccupants::default())` beside the existing inserts in `register_stub_shard` (stub.rs:~1279-1281).
- Add `misrouted_interest` counter to `StubStats` (after the existing `occupant_interest_received` field, near stub.rs:1124).
- `process_inbound` is at the 16-param ceiling (stub.rs:1576-1578) — bundle `&mut RetainedOccupants` into an existing tuple param (the `ghost_state`/`crossing` bundling pattern, stub.rs:1569/1574-1581), NOT a 17th top-level param; destructure at the fn head.
- Convert the drop arm (stub.rs:1639-1642): `Ok(InterShardFlow::OccupantInterest(oi)) => retain_occupant(&mut retained, config, oi, clock.local_tick),` keeping the `_ => stats.undecodable += 1` arm straight (no branching inside the generic `postcard::from_bytes` arm — HR5).
- Monomorphic helper `fn retain_occupant(store: &mut RetainedOccupants, config: &StubConfig, oi: OccupantInterest, now: TickId, stats: &mut StubStats)`: early-return `if oi.to_realm.lowered() != config.own_coord.lowered() { stats.misrouted_interest += 1; return; }` (single compare, no compound guard); else `stats.occupant_interest_received += 1; store.0.insert(oi.observer, RetainedOccupant { occupant: oi.occupant, last_seen: now });`.

**Tests (unit, `stub.rs mod tests` — `tests/` is coverage-excluded, use `assert_eq`/`expect_err`):**
- matching `to_realm` → `store.0.len() == 1`, `occupant_interest_received == 1`.
- mis-routed `to_realm` → not inserted, `misrouted_interest == 1`.
- undecodable bytes → `undecodable == 1`.
- last-wins: two inserts same account, differing `last_seen` → `len == 1`, newer `last_seen`.
- inert: N ticks with zero interest → `store.0.len() == 0`.

**Gate + commit.**

### S2b-ii — EXPIRE (TTL predicate + prune sweep). Store self-erases; still no fold.

**Contract:**
- `const RETAIN_TTL_FLOOR: TickId = 2;` and `fn retain_ttl_ticks(config: &StubConfig) -> TickId` = `max(grace_ticks_from_seconds(WALK_DEMAND_AOI_GRACE_S, config.tick_dt_s) as TickId, RETAIN_TTL_FLOOR)` (derived — no literal window). Re-export `WALK_DEMAND_AOI_GRACE_S`/`grace_ticks_from_seconds` from `vd-core` if not already visible.
- Monomorphic `fn proxy_alive(last_seen: TickId, now: TickId, ttl: TickId) -> bool { now.saturating_sub(last_seen) <= ttl }`.
- Prune at the head of `aoi_decide` (before the observer build, stub.rs:~4606): `retained.0.retain(|_, e| proxy_alive(e.last_seen, clock.local_tick, ttl));` (bounds memory; `BTreeMap::retain`, alive test in the monomorphic helper).

**Tests:**
- `proxy_alive`: `now-last_seen == ttl` → true; `== ttl+1` → false.
- `retain_ttl_ticks`: real `dt` → derived term dominates (≥ 20); degenerate `dt` → floor 2.
- prune: entry at `last_seen`, advance `> ttl` → pruned; advance `≤ ttl` → retained.
- single-loss survival: skip one tick's insert, entry still alive next tick.

**Gate + commit.**

### S2b-iii — FOLD (ObserverId::Proxy + frame compose + fold-before-gate + 3 guards). The payoff.

**Contract:**
- Add `Proxy(AccountId)` to `ObserverId` (stub.rs:~577). Confirm `Copy + Ord` derive still holds; no exhaustive match to update (grep-confirmed only-keyed at 4610/4616/4641).
- Monomorphic `fn proxy_observer(entry: &RetainedOccupant, root_frame: FrameRef, ctx: &LocalFrames) -> Option<(DVec3, DVec3)>` wrapping `transfer_frame(&entry.occupant, root_frame, ctx)`: `Ok(p) => Some((p.pos.offset(), p.vel))`, `Err(_) => None`. Hold the match here; never in the generic body.
- In `aoi_decide`: build `ctx = regions.frame_context(tick_hz)` and `root_frame = regions.regions.iter().find(|r| r.parent.is_none()).map_or(FrameRef::GalaxySpace, |r| r.frame)` and `ttl` once, near the observer build (stub.rs:~4606). **Fold BEFORE the gate:** extend the `observers` builder (stub.rs:4606-4618) with a `.chain()` over `retained.0.iter().filter(|(_,e)| proxy_alive(e.last_seen, clock.local_tick, ttl)).filter_map(|(acct,e)| proxy_observer(e, root_frame, &ctx).map(|(p,v)| (ObserverId::Proxy(*acct), p, v)))`. The existing `if observers.is_empty() { push Empty; return; }` (stub.rs:4622-4632) now tests the combined set — unchanged code, correct behaviour.
- Gate render accumulation (stub.rs:4677-4682) on `render_routes.contains_key(obs)` (a proxy runs AoI/latch, not render).
- Leave untouched: `render_routes` (4637-4642), the emit loop (4713-4728), the parent HeadRead (4735-4743), the up-relay (4752-4767, iterates `dots.0` → no proxy re-relay), `retain_live` (4770), and the boundary/transfer path (proxy never in `Dots`/`OwnedTransients`).

**Tests:**
- **sibling warm-ahead (the payoff):** armed parent, **zero local dots**, one retained proxy positioned so a sibling child enters range → that child's `push_demand(SpinUp/KeepAlive)` fires; the `(Proxy(account), child_path)` latch is set. (This is the test the draft's Q4 made impossible; it passes only because the fold is before the gate.)
- **Empty when truly empty:** parent, zero local dots, zero alive proxies → `DemandVerb::Empty` for `own_coord`; with a real local dot → not Empty.
- **live proxy is not Empty:** parent, zero local dots, one alive proxy in range of some child → NOT Empty; the sibling is demanded.
- **direct-child compose always Ok (Err unreachable in scope):** proxy whose `occupant.frame` is the occupied direct child, that child in the roster → `proxy_observer` returns `Some` (`expect`), demand fires.
- **Err safe-degrade (out-of-scope path):** proxy whose `occupant.frame` is absent from `ctx` → `proxy_observer` returns `None` (`expect_err` on the raw `transfer_frame`), the proxy is not folded, no demand emitted, the entry stays in the store with TTL running.
- **no render for proxy:** proxy-only observer whose child acquires → **no** `RealmSceneDelta` in the outbox (proxy absent from `render_routes`); a real dot in the same tick does get one.
- **moving child:** proxy inside a mover → the composed distance tracks the live `orbital_state` placement.
- **anti-flicker (lost datagram):** proxy folded, one relay tick skipped (still alive by TTL, store unchanged) → the fold re-composes the last pose, the latch survives (no hard-evict), sibling demand held; after TTL lapses → pruned, `retain_live` evicts, sibling releases.
- **re-home invalidation:** fresh (simulated-restart) store → empty, no stale proxy demand.
- **inert byte-identity:** full armed cull vs walk-scale → walk emits byte-identical (no proxy path taken); mirror `parent_headread_due_resolves_only_when_parented_armed_and_on_cadence` (stub.rs:~12847) and its inert assertion.

**Gate + commit.** After S2b-iii, run the existing whole-cluster walk-scale golden byte-identity gate — must stay green.

---

## (D) Byte-identity / inertness argument (re-verified)

Structural, every link load-bearing already for the S2a up-flow:
- Regions default `AoiConfig::inert()` at walk/canonical scale — `spin_up_r_m == 0`, produced branchlessly when `spin_up_factor <= 0.0` (worldgen.rs:928-929).
- `RealmRegions::aoi_live()` is `false` at walk/static (stub.rs:670) ⇒ `parent_headread_due` returns `None` (stub.rs:4735) ⇒ `ParentRealmNode` stays `None` ⇒ the up-relay guard `if let Some(parent) = parent_node` (stub.rs:4753) never enters ⇒ **no `OccupantInterest` is ever emitted**.
- No interest ⇒ the `SignalDelta` write site never runs ⇒ `RetainedOccupants` stays empty.
- Empty store ⇒ the prune `retain` is a no-op; the `.chain()` proxy arm yields zero rows ⇒ `observers` is byte-identical to today ⇒ the `observers.is_empty()` gate sees the same set ⇒ zero new demands, zero render deltas.

**Proof gates:** (1) the existing whole-cluster walk-scale golden byte-identity gate stays green; (2) the S2b-i inert unit test (`store.0.len() == 0` after N walk ticks); (3) the S2b-iii inert test (armed vs walk byte-identical, no proxy path).

**Flagged naive breakages to avoid:** making the store or prune run/branch on anything other than "is the map non-empty" (risks a walk-scale write — keep it purely input-driven); branching inside the `postcard::from_bytes` arm or `transfer_frame`'s generic body (delegate to `retain_occupant`/`transfer_frame_resolved`); persisting the store or mirroring the proxy into owned entities (HR1); adding a gateway field to `OccupantInterest` here (frozen wire — that is S2c); any `match` on shard kind (none introduced).

---

## (E) Explicit deferrals + DEFERRED.md ledger

- **S2c — sibling RENDER to the traveller's client (REQUIRED companion, not optional).** S2b warms the neighbour node; S2c draws the neighbour's box outline (from the parent's own seed roster via `child_shape`) on a client the parent does not host. Owner: S2c. Path: **Option B (recommended)** — directory gateway-resolve reusing the `ParentRealmNode` HeadRead cadence (stub.rs:4735-4743, `on_directory_reply`/`update_parent_node`), no wire growth; vs **Option A** — additive wire growth (`OccupantInterestV2` variant with both exhaustive `effect_class`/`durability_class` arms, OR a `PROTO_MINOR`-gated gateway field). **The pixel-visible dot-crosses-a-boundary acceptance test gates on S2b + S2c together**, not S2b alone. Consequence until S2c: the neighbour's outline appears at the crossing instant, not on approach — the readiness (no cold-load) bar holds, the visible-no-pop bar does not.
- **D-RLM-S2b-flicker — coarsen-ladder proxy `Err` anti-flicker → S3.** For an out-of-scope grand-child relay whose frame the parent cannot place, a hard-evict-then-respin could blink if the `Err` oscillates. Unreachable in S2b's direct-child scope (proven by the direct-child-always-Ok test). When S3 makes the coarsen ladder real, add carry-forward of the proxy's existing latch keys (or a cached last-good composed position) so a transient un-measurable tick freeze-holds rather than evicts. Owner: S3.
- **Multi-hop parent survival → S3.** A proxy is not re-relayed to the grandparent (up-relay iterates `dots.0`, stub.rs:4755); a proxy-only parent's own keep-alive above one hop rides the S3 coarsen ladder. S2b's gates prove only the single-hop sibling-warm.
- **Coarsen ladder (`coarsen_level`, intershard.rs:616-618) → S3;** S2b consumes only full-pose (deepest-parent) relays; a coarsened grand-child relay the parent doesn't place `Err`-drops safely.
- **Exact-integer cross-tier rebase in `transfer_frame` (frame.rs:129-133, D-41 plant-item 2) → P4/P5;** S2b uses the cell-0 offset path like every current caller.

**DEFERRED.md ledger edits owed:** flip **D-RLM** (parent USES relayed interest) from received-and-dropped to **retained-and-folded (lifecycle)**; add **D-RLM-S2c** (sibling render to a non-hosted client — the required render companion, Options A/B above); add **D-RLM-S2b-flicker** (coarsen-ladder `Err` anti-flicker, owner S3).