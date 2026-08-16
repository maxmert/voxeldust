# THE WINDOW LANE — FINAL SYNTHESIZED DESIGN (2026-08-15)

Synthesis of three competing designs (wire-minimal, composition-first, client-contract-first) and
three verdicts (law 90/89/75 — WM wins; feasibility 56/53/45 — CC wins; experience — WM wins,
CF disqualified). Base = **wire-minimal**. Every graft the judges mandated is folded in. Every
judge-found hole is resolved in §6 or promoted to §5 (open questions — none are buried).

Discharges owner decisions 2026-08-15 items 1, 9, 10 (docs/design/owner_decisions_2026-08-15.md)
and requirements R1–R8. Worktree: new-system. File references relative to repo root.

---

# 1. PLAIN-LANGUAGE OWNER SUMMARY

(Short sentences. Active voice. No code names.)

Today the picture a player sees travels through a bucket chain. Each world level re-computes the
scenery for the level below it, every tick. This chain is heavy. It mixes different moments of
time. It presses on the law that a place must never learn its own position. And it still cannot
show the player the world they stand in, or the stars they could fly to.

We replace the chain with a window. Each world level states only what it lawfully owns. It states
where its children are, in its own terms. It states what it itself looks like. It states one point
of light for each sleeping child. It sends these short statements straight to the player's doorman
— the connection process that already serves that player. No statement ever passes between worlds.

The doorman stacks the four to six statements on the player's chain of containers. It stacks them
at one single moment of world time. It marks which world is the center of the picture. It ships
one finished picture. The player's screen only draws. It computes nothing. This is the standing
law: the server composes, the client renders.

The doorman cannot invent anything. It has no world model, no seed, and no motion code — the
motion code is not even linked into it. It can only stack statements that arrived from their true
authors. Each statement is checked against the ownership record. A missing statement means the
thing is not drawn. So every pixel has exactly one lawful author, by construction, not by care.

What you see: sleeping stars are real points of light, placed by the galaxy, with real parallax.
No server runs for a sleeping star. When you fly toward one, the system wakes ahead of you. Its
own self-description takes over the drawing at the exact spot the point occupied. The position
never changes author, so a jump is impossible. Behind you, the departed system shrinks to a dot
and goes back to sleep. That is warp, in pixels, with no loading screen and no flicker. That is
the acceptance bar of this work.

Three cleanups ride along, all approved before. The test file of boxes the client could load at
boot is deleted. The position field inside shape messages is removed, in one cut, because no
players are deployed. And the shape-passing between worlds ends: a world's look is authored only
by that world; its position only by its parent.

If one distant level lags, the sky above the fault holds still while your local world keeps
moving. Nothing vanishes. Nothing tears. If the level dies for good, its part of the sky is
removed cleanly after a fixed, derived wait.

## 1.1 THE FORMAL ASK (the standing gate: ask before new data crosses; default NO)

No new data crosses any world-to-world boundary. Four world-to-doorman statements and one
doorman-to-world control are asked for. One item needs a separate ruling (marked).

1. **The hop statement.** From me, for each occupied direct child, every tick: my own body,
   expressed in that child's terms. I do the one subtraction myself, because I author that child's
   placement. The doorman cannot compute this: it holds no placements and cannot evaluate motion,
   by construction. Nobody else may author it: the child must never hear it. Cost of doing
   without: no picture beyond the current world — no sky, no star map, no warp.
2. **The placement statements.** From me, every tick: where each of my direct children is, in my
   own terms — the full list, sleeping children included. I already author these today. The
   doorman cannot compute them: only I run my physics.
3. **The body statements**, two exclusive kinds. (a) MY OWN LOOK: my outline and appearance. It
   carries no position, by its very shape. Only I may state it — a world draws itself. (b) A
   POINT-OF-LIGHT DATUM per direct child: brightness and color, computed from the child's seed,
   which I hold as that child's generator. I name this plainly: for a SLEEPING child, I author a
   small piece of its appearance. Your ruling of 15 Aug allows this. The moment the child runs
   and states its own look, my datum is superseded by the presence of its statement.
4. **The membership verdict.** From me: which of my direct children are inside the interest band
   of an occupant I hold, or of an occupied child standing in for its occupants. Ids only. This
   keeps the visibility decision mine, as the law says. Points of light are NOT filtered by it —
   the full child list is always in the sky.
5. **Inbound control (not a world boundary).** The doorman may ask me: serve the picture for my
   own occupants; or serve it for the occupants under my child so-and-so. No account, no pose.
   This tells me nothing beyond the occupancy bit that already crosses.
   **5b — needs your separate ruling (Open Question 2):** the doorman may also ask a LIVE world
   the observer is NOT inside: let me watch one level of your interior, for an outside observer.
   This tells that world one new thing: some doorman watches it from outside. The honest
   alternative costs one relay hop of delay at the exact moment a star wakes.

What does NOT cross: no occupant pose, in any direction. Nothing new enters any world. No world
learns anything about itself. The one statement that contains a reversed placement cannot enter
a world's process — its message type only exists on the world-to-doorman leg.

---

# 2. TECHNICAL DESIGN

## 2.1 The model in one page

For an observer (logged-in dot) standing in realm `L` with ancestor chain `L ⊂ P1 ⊂ … ⊂ G`
(depth 4–6):

- Each chain level `A` ships, directly to the observer's gateway (never through another realm):
  (i) its authored per-tick child-placement rows — FULL direct-child roster, dormant included;
  (ii) ONE hop row per occupied direct child: "my frame expressed in that child's frame at this
  tick", pre-inverted by `A`, who authors that placement (R1; SL1's "conversion in the parent"
  holds hop-by-hop); (iii) its self-look (no position — by type); (iv) one marker datum per
  direct child (photometric point-of-light, R4); (v) its SL7 membership verdict (ids only), which
  gates BODIES and INTERIORS — never markers.
- The gateway composes per observer at one universe tick, using the EXISTING `transfer_frame` +
  `PlacementBook` core (`crates/core/src/frame.rs:133`): mixed-tick composition is refused by the
  already-covered `FrameError::InstantMismatch` — the shear law is type-level, not tested-for.
  The gateway inverts nothing, subtracts nothing, evaluates no motion (it structurally cannot:
  no `vd-physics` dependency).
- Body selection per drawn realm: self-look if one was received (only a running realm can ship
  one), else the parent's marker. THE DRAW LAW by absence of data — no if-running check exists,
  and a third pixel source is unrepresentable in the types (a look cannot carry a position; a
  marker cannot carry a look).
- The composed scene ships stamped with an explicit origin marker `(origin: RealmId,
  origin_epoch)` riding the scene level itself (R3, owner option b). The client is a pure
  passthrough: pose + tagged skip-unknown TLV bag per row, no realm-kind branch, interpolation
  buffer unchanged, no prediction.
- The re-proven two-level visibility bound (boot fence `guard_grandchildren_invisible_outside`,
  `crates/bins/src/bin/shard.rs` — verified real) bounds the drawn set: chain levels' bodies +
  each level's direct children + at most one level of interior for a LIVE in-band child. The
  live-sibling interior lane is CONTINGENT on Open Question 1.
- Consequently the entire inter-realm scenery relay (`RealmCascade`, `RealmObservation`,
  `RealmShapeObservation`, `ChildSceneSet`) becomes producer-less and is tombstoned. **This
  design adds ZERO new realm-to-realm data.** Sealing strictly increases: four inter-shard arms
  removed, none added.

## 2.2 Wire rows, shard → gateway (mesh minor 15; `crates/wire/src/session_flow.rs`)

Appended `ShardToGateway` variants. Mesh-only ⇒ ledger minor entry, client floor unmoved. Every
row carries `realm_fence` (zombie guard) and a universe-tick stamp. Attestation on every row:
the gateway drops + counts any row whose sender node ≠ the `ShardRoster` head for the stating
realm (fail-closed; measured now, refused at cloud mTLS — owner item 11 pattern).

**`ShardToGateway::WindowFrame`** — the per-tick unit of one window level. ONE message per tick
per open window: intra-level same-tickness is by construction, never by matching.

```
WindowFrame {
    realm_fence: Fence,
    window: WindowId,              // gateway-issued subscription id
    at: UniverseTick,              // the one stamp for everything inside
    hop: Option<HopRow>,           // None for the observer's-own-level window
    rows: Vec<RealmSnap>,          // TYPED authored child rows, sender's own frame
}                                  // (no nested serialize-inside-serialize — judge fix)
HopRow {
    child: RealmId,
    inv: FramePlacement,           // "my frame expressed in the child's frame at `at`" —
}                                  // full rigid transform, PRE-INVERTED by the author
```

- **INV-BODY-AT-ORIGIN (named invariant, pinned):** a realm's own body sits at its own frame
  origin. `inv.origin` therefore IS "the parent's body in the child's frame" (R1 literal). A pin
  test asserts the invariant on THE world and fails if it ever stops holding (closes hole H9).
- Rows = the FULL direct-child roster (markers are never membership-filtered — graft from CC,
  closes A-2: stars always visible by construction, no band-edge flicker, no wait on the owed
  photometric derivation).
- Classification: EffectClass FireAndForget; durability Unreliable, full-state latest-wins per
  tick (owner law 3(b): only full state rides a drop lane). Cadence: the realm-lane tick (20 Hz),
  stamped off the follower clock.
- Hop inversion at the author: through P3 cells are zero (small-pose case). A rotated-frame-
  across-cells inversion inherits `transfer_frame`'s refusal semantics (dropped + counted, owed
  with P10 cell math). "Cannot occur in THE world today" is NOT argued — a pin measurement lands
  with Slice A and fails when it stops being true (never-assume law).

**`ShardToGateway::WindowBody`** — the look/marker lane. **ReDriven, reliable** (rides the
session-reply lane; a lost look is an invisible realm at exactly the no-flicker moment — closes
hole H6; mirrors `ShardRoster`'s reasoning).

```
WindowBody {
    realm_fence: Fence,
    window: WindowId,
    subject: RealmId,
    stmt: BodyStmt,
    authored_at: UniverseTick,
}
BodyStmt =                          // CF's type graft: third pixel source unrepresentable
  | SelfLook { bag: TlvBag }        // TAG_LOOK outline + display tags. NO position field EXISTS.
  | Marker   { luma: TlvBag }       // TAG_LUMA photometric scalars. NO look field EXISTS.
```

Two structurally exclusive authorships, enforced by attestation, never a flag:
- `SelfLook`: legal only when `subject == sender's own realm` (a realm states a look ONLY about
  itself — R7/SL3/D-LANE-4).
- `Marker`: legal only when `subject ∈ sender's direct children` (roster check). Luma drawn from
  the same seed stream the parent generated the child from (`sample_imf_mass` /
  `classify_spectral` / `main_sequence_luminosity` in `crates/physics` taxonomy — the missing
  per-system draw is added to `generate_system_forest` in Slice 0, pinned f(seed) values).
A mis-authored body (look about a child, marker about a non-child, non-head sender) is dropped +
counted (`window_misauthored_body`), never patched.

**`ShardToGateway::WindowMembership`** — `{window, added: Vec<RealmId>, removed: Vec<RealmId>}`,
ids only. The parent's SL7 band/hysteresis verdict shipped to the gateway so the gateway NEVER
re-derives AoI. Scope (grafted refinement): membership gates BODIES and live-child INTERIOR
windows only; marker rows always ship (full roster). Per-dot fold for own-level windows (today's
per-account fold); per-occupied-child fold (the existing `child_visible` SL7 proxy set) for
child-scope windows — shared by every observer under that child. Reliable, on the AoI cadence.

## 2.3 Subscription control, gateway → shard (same mesh minor)

```
GatewayToShard::WindowOpen  { window: WindowId, scope: WindowScope }
GatewayToShard::WindowClose { window: WindowId }
WindowScope =
  | Occupants            // serve the picture for my own occupants (own-level window)
  | Child(RealmId)       // serve occupants under my direct child c (hop row for c + rows
                         //   + bodies + membership scoped to c's proxy band)
  | Observed             // CONTINGENT on Open Questions 1+2: serve ONE level of my interior
                         //   to an external observer's gateway (live-sibling window)
```

- The typed `WindowScope` dissolves the `child: None` overload (closes hole H7): each scope has
  stated shard-side gating semantics. `Occupants`/`Child` are information-equivalent to the
  occupancy bit that already crosses (SL2 intact). `Observed` is NOT — it tells a realm "some
  gateway watches me from outside". It is therefore its OWN honest SL6 item (5b in §1.1), put to
  the owner as Open Question 2, default-NO respected (closes hole H3 — the law verdict's exact
  prescribed remedy). Fallback if refused: the parent-held one-hop relay (CC's `WindowHeld`
  shape) replaces the direct window, costing one relay hop of look latency, budgeted in
  G-HANDOVER.
- Carries NO account, NO pose. Re-asserted on a derived keepalive cadence; a shard drops a window
  not refreshed within the derived TTL (2 beats + 1 — owner law 3(a): retention forever derived).
  A dead gateway can never leak a fan (chaos-tested in Slice A).

## 2.4 Client-facing wire — THE ONE-CUT FLAG DAY (minor 16, floor 8 → 16)

Owner item 9: zero deployed clients ⇒ reshape in place, no shims, no dual-decode. Mesh additions
(§2.2–2.3) ledger as minor 15; the flag day as minor 16; `PROTO_MINOR_FLOOR` moves to 16. Both
ledger entries carry the owner citation (items 1/9/10) — the citation build gate at
`crates/wire/src/version.rs` enforces this. (Numbers re-derived at landing if the ledger moved.)

- **`RealmShape` loses `center`** (R3/R5 companion; `channels.rs` one-meaning docblock rewritten;
  the known-violator paragraph replaced by the cure note). A shape is pure self-description.
- **`ServerControlMsg::RealmRegistry`** becomes the composed LEVEL:
  `{origin: RealmId, origin_epoch: u64, rows: Vec<SceneRow>}`. `pin` deleted; the origin marker
  is `pin`'s lawful successor; the unread `render_pin` (gateway.rs:739–762, read-site :3160-3172)
  is deleted — origin derivation becomes the one meaning. The pin-vs-home disagreement settles:
  the origin follows the avatar; the lease-time home stays routing/diagnostic only.
- **`ServerControlMsg::RealmSceneDelta`** becomes
  `{origin, origin_epoch, added: Vec<SceneRow>, removed: Vec<RealmId>}` — reliable incremental.
- **`SceneRow`** — the VU streaming contract realized: *a pose + a bag of signals*:

```
SceneRow {
    realm: RealmId,
    parent: Option<RealmId>,   // hierarchy identity only
    pose: StampedPose,         // in the ORIGIN frame; stamp explicit PER ROW (held strata
                               //   lawfully carry an older stamp — §2.6.4)
    bag: TlvBag,               // TAG_LOOK present  => body (self-authored)
                               // TAG_LOOK absent + TAG_LUMA => marker point
}                              // unknown tags SKIPPED — signals extend forever, zero client change
```

  Levels/deltas carry the pose, so a row is drawable the instant it arrives — the "shape before
  first pose row" gap ceases to exist; there is no undrawn window at login or crossing (closes
  CC's designed blink, hole C-1, structurally).
- **`RealmSnapshotDatagram` gains `origin_epoch: u64`** (same flag day). Composed per-tick rows
  reuse `RealmSnap` (head = drawn realm's own frame, tail = origin frame — head≠tail law holds;
  the origin realm never ships a row: it draws from its look at the origin marker). The gateway
  stamps a per-session monotone `frame_id` — lawful: the composed row is a NEW row authored by
  the gateway from attested inputs. Client per-RealmId high-water collapses to one feed counter
  + epoch (single author).
- The `RealmBox.tier` gap closes: tier rides `pose.frame` explicitly; the tier-agreement pin test
  (realm_scene.rs:1216) retires with an owner-cited ledger note.

## 2.5 Tombstones (`crates/wire/src/intershard.rs`, Slice C2 mesh minor)

`RealmCascade` (disc 26), `RealmObservation` (31), `RealmShapeObservation` (32), `ChildSceneSet`
(33): variants + payload structs REMAIN, discriminants reserved forever, both exhaustive-match
classifications frozen, received frames count `undecodable` — the established discipline (minors
12/13/15 precedent). `ChildLive` (30) stays (liveness/demand, SL7 — not draw). `ShardRoster` (29)
stays and gains its second consumer (window attestation + routing). Positional pin tests in
`crates/wire/tests/intershard_closed.rs` extended: producer-less golden pin grows by four arms.
**No new `InterShardFlow` arm is added anywhere** (CF's arm 34 rejected by all verdicts).

## 2.6 Gateway composition (`crates/connection-plane/src/window.rs`, new Tier-A module)

### 2.6.1 Structural guards (the T1 resolution — build-level, not care)

1. **Dependency gate**: `vd-connection-plane` must not depend on `vd-physics` — asserted by a
   manifest-parsing test + clippy `disallowed-types` on motion/worldgen symbols. The gateway is
   structurally unable to evaluate a placement or generate a world. This replaces the old
   router-converter doc-scan test (version.rs:503–618), which guarded the OLD rejected model
   (world-wide graph); the replacement is ledgered with the owner citation.
2. **Provenance gate**: every composed row's inputs are attested rows from roster-head senders;
   a WireMonitor check asserts a forged non-head `WindowFrame` is dropped + counted, and no
   composed row exists without an attested input chain.
3. **No shard-bound sender**: the module's signature takes inbound rows, returns client-bound
   emissions; the wiring gives it no handle that can produce an `InterShardFlow` (CF graft).
4. **Zero-state proof**: a structural test asserts the gateway holds ZERO window state with zero
   sessions (CC graft) — the anti-central invariant: nothing global ever accumulates.
5. **AoI separation**: the demand/AoI fold has no data path into `window.rs` (module dependency
   rule) — SL7 proxies stay demand-only; no approximation on the render chain (R6).

### 2.6.2 Data structures

Per window, keyed `(shard_node, scope)`, SHARED across all sessions whose chain passes through it:

```
WindowState {
    window: WindowId,
    author_realm: RealmId,                 // roster-checked
    levels: TickRing<WindowLevel>,         // K = 2·cadence_ratio + 1 (DERIVED)
    body_of:   BTreeMap<RealmId, TlvBag>,  // latest SelfLook per subject (only ever self-shipped)
    marker_of: BTreeMap<RealmId, TlvBag>,  // latest Marker per direct child
    members: BTreeSet<RealmId>,            // parent's SL7 verdict (bodies/interiors only)
    freshness: UniverseTick,               // TTL-pruned (2 cadences + 1)
}
WindowLevel { at: UniverseTick, hop: Option<HopRow>, rows: Vec<RealmSnap> }  // decoded ONCE
```

Per session: `chain: Vec<WindowRef>` (leaf→root), `live_children: BTreeMap<RealmId, WindowRef>`
(contingent on Q1/Q2), `origin`, `origin_epoch`, `composed_frame_id`, `last_level_sent`.

**Chain derivation is stream/session-only — NO seed table at the gateway** (CF graft, closes hole
H10): the lineage comes from the session's login descent (which already derives the home lineage)
and is updated at each crossing's `SubscriptionReady`; parenthood is confirmed by the attested
hop rows themselves (a `Child(c)`-scope `WindowFrame` from realm A is A's attested claim
A = parent(c)); realm→node from `ShardRoster`. The flagged "seed-forest second source"
(gateway.rs:2024–2028) is DELETED, not shrunk. Cycle-safe (visited set; a cycle counts
`window_chain_cycle` and truncates, fail-closed).

### 2.6.3 Same-tick composition (the shear law — CC's core, grafted verbatim)

- Every `WindowFrame` is internally same-tick by construction (one message, one `at`).
- Cross-level: compose at `T = max { t : every FRESH chain window's ring holds a level stamped
  t }`. Composition runs through the existing `transfer_frame` + `PlacementBook` machinery, with
  per-level books built from hop rows at exactly T. `FrameError::InstantMismatch` REFUSES any
  mixed-tick fold — the shear guarantee is enforced by the type system that five slices of prior
  coverage already prove, not by new bespoke math (closes WM's biggest unproven claim).
- The gateway NEVER interpolates, NEVER extrapolates, NEVER evaluates ephemeris (CF's gateway
  interpolation explicitly NOT grafted — it made a non-author manufacture shipped positions).
- **Anti-vacuity (CF graft)**: the shear gate includes a deliberately mixed-tick compose that
  MUST fail — proving the gate can fail.
- `T` is monotone per chain (CF graft): a would-be rewind freezes and counts
  `window_t_monotone_stalled`.
- **Exact-cadence precondition, PINNED (judge fix)**: common ticks exist because all shards stamp
  realm-lane levels at identical universe ticks off the one orchestrator clock. This lands as a
  boot/CI invariant that fails loudly if per-realm emission cadence ever diverges — an
  assumption converted to a gate.

### 2.6.4 Per-stratum hold (the A-1 resolution — sky holds, local moves, nothing vanishes)

If hop `j` (counting leaf→root) cannot serve the common tick within the ring:
- Strata BELOW `j` (the local world, closer levels) continue composing at fresh `T` computed over
  the fresh hops only.
- Stratum `j` and everything ABOVE hold at their LAST COMPOSED origin-frame poses — held, not
  removed, not truncated. Each held row keeps its old stamp explicitly (`SceneRow.pose` carries
  it); held strata are contiguous-above by construction. Counted `compose_hold_ticks{hop}`.
- Honest exactness note: a held stratum is stale-by-declaration relative to the moving origin —
  declared, per-row stamped, counted, bounded; never silently mixed into a fresh fold (fresh rows
  in one datagram still share exactly one tick; the shear gate asserts both halves).
- **Dead-hop exit (the missing exit condition, closes A-1's second half)**: when the hop's realm
  leaves the roster, or the hold exceeds a DERIVED long bound (roster-loss confirmation window),
  the held strata are removed via the reliable delta — clean removal after a fence, counted
  `window_hop_dead`. Truncation is the last resort with a fence, never the first response
  (rejects CF's eager truncation, hole B-2). Heal: fresh rows resume; strata re-add reliably.

### 2.6.5 The compose step (per session, per tick)

1. Pick fresh `T` (§2.6.3) and the hold boundary (§2.6.4).
2. Walk the chain leaf→root accumulating `X_k = inv_1 ∘ … ∘ inv_k` — pure composition of
   parent-authored, parent-inverted transforms via the frame core. The gateway inverts and
   subtracts NOTHING; every minus happened in the lawful parent (SL1's letter, hop-by-hop).
3. Per fresh level `A_k`: child rows (in `A_k`'s frame) map through `X_k` into the origin frame;
   `A_k`'s own body row = `X_k` applied to the frame origin (INV-BODY-AT-ORIGIN), look =
   `A_k`'s self-look.
4. Per live-child window `c` (contingent Q1/Q2): `c`'s rows map through `X_k ∘ placement(c)` at T.
5. Membership-filter bodies/interiors by `members` ∪ chain bodies; markers pass unfiltered.
6. Body selection: `body_of[r]` if held, ELSE `marker_of[r]` — presence gate; types make a third
   source unrepresentable (§2.2).
7. Dedup one entry per RealmId (chain-body/child-row overlap resolved by construction; the f64
   agreement between a pre-inverted hop and the parent's child row under rotation is MEASURED
   inside the Slice-B parity gate, not argued — judge fix).
8. Emit: per-tick composed `RealmSnapshotDatagram{origin_epoch, rows}` (MTU-partitioned via the
   shared `partition_rows` ladder, session `frame_id++`); reliable `RealmSceneDelta` on
   membership/body change; full `RealmRegistry` level on origin change or (re)login.

### 2.6.6 Failure-mode table (CF graft — the coverage checklist; every arm both-ways covered)

| failure | behavior | counter |
|---|---|---|
| non-head sender | drop row | window_sender_mismatch |
| stale fence | drop row | window_stale_fence |
| mis-authored body (look≠self / marker∉children) | drop row | window_misauthored_body |
| hop lags beyond ring | per-stratum hold (§2.6.4) | compose_hold_ticks |
| hop dead (roster loss / derived bound) | reliable removal of held strata | window_hop_dead |
| would-be T rewind | freeze emission | window_t_monotone_stalled |
| rotated cross-cell hop inversion (pre-P10) | drop + refuse | window_tier_refused |
| chain cycle | truncate, fail closed | window_chain_cycle |
| unresolved standing realm (login race) | withhold composed feed | window_unresolved_standing |
| mixed-tick fold attempt | refused by InstantMismatch | window_instant_mismatch (asserted 0) |
| client: stale-epoch datagram | drop row | stale_epoch_rows |

### 2.6.7 Cost + load gate

Ingest is per-WINDOW, shared; decode once per level per tick. Compose per session ≤6 transform
compositions + ~10–40 row rebases through `transfer_frame` (real, monomorphic; grounded estimate
≈10–40 µs at chain 6 × 64 rows). Budget < 10 µs/player/tick typical on THE world; memory per
window ≈ K×rows×~100 B. **G-COMPOSE-LOAD gates BOTH sides (judge fix)**: compose p99 latency AND
the ingest/decode side at "many live windows × 20 Hz" (the previously unmodeled decode cost),
plus the hold-tick histogram, against `WindowTuning` bounds. One `WindowTuning` struct holds
every derived bound (ring K = 2 cadences + 1; window TTL = 2 beats + 1; look re-send policy;
membership cadence coupling; hold-alarm + dead-hop thresholds) — no magic numbers.

### 2.6.8 Interpolation interaction

Composed rows are stamped T ≤ newest authored tick; alignment gap typically ≤1 realm tick (50 ms
at 20 Hz), ring-bounded worst case — inside the client's 100–150 ms buffer. The interp
`EntityTrack` machinery is reused verbatim; freeze-never-coast on stall. NO prediction anywhere.

## 2.7 The origin marker and crossing semantics (R3 — WM's tight flip, kept whole)

- Stamped by the gateway at the composition point: `(origin: RealmId, origin_epoch: u64)` ON the
  scene level itself, every delta, and every composed datagram (epoch, 8 bytes). No separate
  origin message exists that could desync from the level (rejects CC's split-message flip).
- **Scene-swap on crossing**: at `TransferControl::CommitAuthority`/`SubscriptionReady` the
  gateway recomputes the chain, opens/closes windows (holding BOTH chains through the hand-off
  overlap — the existing dual-sub pattern), bumps `origin_epoch`, and emits — ordered on the SAME
  reliable control stream, after `AuthorityChanged` — one full `RealmRegistry` level in the new
  origin, composed at the SAME T as the last old-epoch datagram. The client swaps atomically on
  the epoch: the old scene keeps rendering until the new level lands; early new-epoch datagrams
  are held one beat. Every body's position across the swap is the exact same-tick re-expression
  under the new chain — screen deltas bounded by one tick of true motion (the crossing no-flicker
  gate measures exactly this). This REPLACES the client's `forget_space` inference (net.rs:317–
  330) and its one-straggler residual; the crossing re-stream and the origin marker are one
  mechanism (discharges the warp re-stream / D-RLM-14). No gap and no double-draw, mechanically.

## 2.8 The star map and the map-to-live handover (R4, R6)

- **Points of light**: the galaxy shard ships its FULL direct-child roster as placement rows
  (real positions ⇒ real parallax; STARMAP-IS-REAL kept) + one `Marker` per child (luma from the
  generation stream — the per-system taxonomy draw added in Slice 0). Dormant systems get NO
  servers; their pixels are 100 % parent-authored marker + placement.
- **Precision**: `LatticePos` integer-cell + f64 residual end-to-end; the ONE lossy point is the
  client's `draw_center()` flatten — sub-pixel at stellar distances. FINE i64 span is ±0.476 ly
  (corrected: CELL_DOMAIN_MAX = i64::MAX/2); at near-real scale galaxy rows ride COARSE ly-cells,
  converted by the exact `FINE_CELLS_PER_LY` ratio. The tier decision is REPRESENTABILITY (a
  saturation refusal), never a distance heuristic or kind fork; cross-tier compose activates with
  P10's `convert_tier`; until then refused + counted.
- **Handover — one position author for life**: the drawn body of system S is positioned by its
  parent's placement row before, during, and after spin-up. Only the LOOK payload upgrades:
  marker ⇒ self-look the moment S's `WindowBody` reaches the gateway, by data presence, never
  both, never zero. Departure mirrors: teardown-behind ⇒ roster loss ⇒ `body_of[S]` pruned after
  the derived TTL ⇒ the ever-present marker resumes ⇒ the system shrinks to a dot, literally.
- **No pop**: AoI spin-up leads visibility (θ_min band + velocity lead + boot-horizon
  prediction), so S runs and its look has arrived while S subtends ≈ a point. Gate preconditions
  pinned from THE world's solved geometry (CF graft), RE-DERIVED against the live world at Slice D
  and read from the shipped config rather than restated: the 1.5° swap line is
  **11 458.474 599 m** (`system_soi_r_m 150 × cot(θ_min/2) 76.389 830 658`), the ring is
  **12 031.398 329 m** (× the 1.05 slack), the asleep-at-departure margin is **572.923 730 m**, and
  the tear-down radius is **11 483.474 599 m** (the band's whole width IS the velocity lead:
  `|v_rel| · dt · (K_SAFETY + extra)` = 25.0 m at the shipped speed). The galaxy shell moved to
  **12 483.456 992 m** with the two-level clearance re-solve; it is not a term of either handover
  budget.
- **G-HANDOVER is SYMMETRIC (judge fix, closes A-3/B-4)**: the wake footprint budget (spin-up
  geometry + boot p99 + one look cadence) AND the departure footprint budget (teardown band +
  roster-loss TTL) are both derived bounds the pixel gate asserts; ε from |v_rel| × frame-dt.
  Asserts per frame at S's projected position: exactly one of {marker, body} drawn — never zero,
  never both; centroid path continuous; footprint monotone through the swap.

## 2.9 Shard-side emission and the deletion map (`crates/sim/src/stub.rs`)

Per tick, per open window (ONE code path, every realm kind — HR3/HR4; ships excluded + counted
per D-SHIP-1):
1. Serialize authored rows once (unchanged machinery; counter bump stays observer-independent).
2. Per `Child(c)` window: build `HopRow{c, inv}` (invert once at the author) and ship
   `WindowFrame{hop, rows}`.
3. Per `Occupants` window: ship `WindowFrame{hop: None, rows}` (subsumes today's direct emit).
4. On change / on open: ship `WindowBody` self-look + one `Marker` per direct child (luma cached
   from the boot roster).
5. On the AoI cadence: `WindowMembership` from the existing `aoi_decide` fold (per-dot for
   `Occupants`, the existing `child_visible` proxy fold for `Child` scopes). Hysteresis, bands,
   grace, demand path untouched (owner items 4/6).

DELETED (in Slice C2, after the client swap makes them producer-less): the observed-interior fan
+ TTL machinery; the up-observation ship + `on_realm_observation` + `on_realm_shape_observation`
+ `lift_shapes_from_child_frame`; the cascade (`restate_rows_in_child_frame` /
`restate_rows_in_frame` / `push_cascade` / `on_realm_cascade`); the down-reflect block +
`FromAboveScene` / `ChildSceneSent` / `on_child_scene_set` / `restate_shapes_in_child_frame` /
`shape_hop_to` / `hop_to_child` / `SHAPE_LANE_TICK` / `child_shape`'s placement read; **the SL1
self-placement filter — at stub.rs:8384 (line corrected per judge verification; the whole §2.9
deletion map is RE-WALKED against the tree as the first task of Slice C2, since drift was
proven)**. The filter dies WITH its hazard: no restated center ever ships toward a realm again —
strictly stronger protection than the filter; explicitly put before the owner (Open Question 3),
because the owner previously said the filter stays. The per-dot scene fold's shape-pushing
shrinks to the ids-only membership emit. The entity lane is untouched — SL2 unchanged.

## 2.10 `RealmShapeObservation` evolution (R7 / D-LANE-4)

This slice: the inter-shard arm is tombstoned; its CONTENT becomes the shard→gateway self-look
(`WindowBody::SelfLook`). A realm authors its OWN look; the parent's per-realm-lane message
shrinks to a placement and nothing else (SL3 reached; the parent is removed from the look path
entirely — the purest form any design offered). The marker datum ships to the connection plane,
not into any realm (tension T2, §3). Later (registered in DEFERRED): look bags grow tags
(surface, detail, self-chosen meshes — P4+); display TLV migrates into the P9 SignalBag (same
codec — a re-tag, not a rewrite); census-scale marker roster bound (owed DERIVATION, never a
literal); P10 rotated-hop composition; ships as window subjects (P8).

## 2.11 Boot-file removal + picture-gate re-base (R5 / D-LANE-6)

Deleted: `--realm-boxes` / `debug_scene_arg` / `from_regions_json` / `from_boxes_json` /
`load_scene` (client), `write_world_regions` / `emit-world-scene` (bins lib + devcluster), the
emit+pass in `scripts/visual-run.sh` / `client.sh`. One world, one source — the stream (SL5).
D-LANE-6 flips 🟩. Scenarios are re-based, never deleted:
- **G-RENDER-BOXES-SMOKE (re-based)**: real single-node cluster, real login; drawn set == the
  gateway-emitted level set (anti-vacuity vs the run manifest, gated on the demand loop's own
  settle signal — never a sleep literal); origin marker == home realm; rim probes + zero-magenta
  unchanged; HR6 manifest.
- **G-RENDER-CROSSING (re-based)**: J1 becomes the origin-marker assert (home draws at origin on
  both sides; epoch bumps exactly once per crossing) + the crossing no-flicker gate (no capture
  frame across the swap where a persisting body's screen delta exceeds the one-tick motion bound;
  no frame with the scene absent).
- Camera reconstruction / harness verdicts read streamed extents (`DevRealmBox` gains
  `extent_m`); `devproto DevState` gains `origin: (RealmId, u64)` and per-box
  `{body_kind: marker|look, newest_tick}` — the diagnosis surface every gate reads.
- **Dot detection is DevState-driven (judge fix)**: gates assert from DevState projected-position
  + body-kind plus LOCAL pixel probes at the projected location — never a full-frame pixel
  search at ~1 px scale. `DevRealmBox` also carries the marker's `luma` datum since Slice D, so a
  gate sizes a point sprite's rectangle through the SAME Tier-A pair the renderer scales it by.
- **THE PILOT VIEW (Slice D, `client --capture-pilot`)**: the scene-fitting capture framing above
  is the DEFAULT and stays so — but it fits the union of everything drawn, so on a flight between
  two FIXED points of the star ring it barely moves and a system you fly toward cannot grow on
  screen. The warp acceptance is a statement about what the PILOT sees, so a capture run may
  instead place the camera at the avatar's eye along its DELIVERED facing, through the one Tier-A
  expression (`pilot_capture_camera`) the gate reconstructs it by. No local view state is involved:
  a headless run has no mouse, so the server-authored orientation is the only honest facing, and an
  injected `LookAt` turns the avatar and the agent's eyes together.

## 2.12 Named invariants and pins (new with this design)

| invariant | pin |
|---|---|
| INV-BODY-AT-ORIGIN: a realm's body sits at its own frame origin | unit pin on THE world (Slice A) |
| Exact-cadence: all chain shards stamp levels at identical universe ticks | boot/CI invariant, loud failure |
| Rotated cross-cell hop inversion cannot occur in THE world (pre-P10) | measurement pin, fails when untrue |
| Live-sibling one-level interior is inside the two-level fence's guarantee | named test pinned in Slice A (contingent Q1) |
| Zero sessions ⇒ zero window state | structural teardown test |
| Deliberate mixed-tick compose FAILS the shear gate | anti-vacuity test |
| Hop-vs-child-row f64 agreement under rotation inversion | measured bound in Slice-B parity gate |

---

## 2.13 Scalability model (owner-requested 2026-08-15; the structural argument + its gates)

The scaling claim is structural: no component holds global state; every cost is bounded by
something local. Axis by axis, mechanism → bound → where it is proven:

- **World size:** only live realms run; a sleeping child costs ONE marker row authored by its
  parent (no server, no tick). Cost ∝ occupied realms, not world size. Proven: the demand loop
  is already in gate.
- **Players:** gateways hold only their own sessions' chains; ZERO gateway↔gateway traffic; per
  player per tick the composer folds 4–6 strata (small frame folds). Linear capacity by adding
  gateways. Gate: G-COMPOSE-LOAD (compose + ingest decode).
- **Depth:** all hops ship in parallel, direct to the gateway — depth adds rows, never serialized
  hops. Picture latency = max(one hop) + same-tick alignment (≤ one cadence), NOT a sum over
  depth. This is the structural win over the deleted bucket chain.
- **Popular realm:** egress ∝ interested GATEWAYS (each aggregates all its sessions), never per
  player. Statements are identical for every subscriber and attested/self-contained, so a dumb
  repeater/fan tier can be inserted later WITHOUT touching authorship — ledgered as the named
  headroom move, taken only when a realm measures hot.
- **Wire volume:** placements ride the existing per-tick pose lane (measured: p99 0.7 ms under
  1.8 GB bulk, SPIKE-3a); looks + markers are send-on-change with TTL, never per-tick; the bit
  beats 2/s. Per-realm rows/tick = O(direct children) — bounded by the lattice at every scale.
- **Glass-to-glass:** author → gateway (1 hop) → compose (in-process) → client (1 hop); the
  client's 100–150 ms interpolation buffer remains the budget owner, unchanged. Gate: the slice-B
  latency measurement rides the parity harness.
- **Elasticity:** re-home ⇒ the gateway re-attests via the directory head (≤ one cadence blind,
  the stratum HOLDS meanwhile — never tears); gateway loss ⇒ every bit of window state is
  re-derivable from the streams (nothing durable on the picture path). Gate: slice-A TTL-expiry
  chaos.
- **Named limits (honest):** (a) single-realm interior DENSITY (a thousand players in one city)
  is the ledgered D-9/P6 interior wall — orthogonal to this lane; (b) at true galactic scale the
  per-author marker list stays bounded because the galaxy becomes the P10 cell-realm lattice —
  "all direct children in the sky" never means millions of rows from one author.

## 2.14 Shared composition + the LOD tier seam (owner Q&A 2026-08-15, recorded)

- **Shared stacks.** The composed level depends on (origin realm, chain, tick) only — never on the
  player. The composer therefore memoizes per (origin, tick): ONE fold per occupied realm per
  gateway per tick, then a thin per-session band filter + socket write. Compose cost ∝ distinct
  occupied realms per gateway; fan cost ∝ sessions. The slice-B parity gate's dedup-agreement
  measurement is the exactness proof of this sharing.
- **Gateway failure posture (owner-raised).** No new failure class: every client byte already
  flows through its gateway; window state is fully re-derivable from the streams (nothing durable
  on the picture path); keep-alive TTLs expire a dead gateway's subscriptions shard-side. Crash
  cost = that gateway's sessions reconnect (handshake + admission + window re-open + first frames
  ≤ one beat) — seconds-class, world state untouched, other gateways unaffected.
- **The LOD tier seam (registered for P4+; NOT this slice's work).** The model already carries
  the ladder's top: tier 0 = parent-authored marker (~30 B), tier 1 = self-authored look/outline
  (~100 B). Detail tiers become MORE TAGS in the same look bag, authored by the realm (SL3: a
  realm chooses its own detail; fractal terrain coarsens free by dropping octaves). SELECTION
  lives where knowledge lawfully lives: the gateway knows every observer's angular size (it
  composes the chain) — it forwards only the tier a band needs and requests a higher tier via a
  plain tier number on the window subscription ("serve tier 2" — nothing about who watches or
  from where). A dot never costs a mesh. Collision/entities never coarsen (standing law,
  untouched — this lane is render-only).

# 3. LAW-COMPLIANCE TABLE

| Law | How satisfied — STRUCTURALLY |
|---|---|
| SL1 only-parent-knows | Hop rows pre-inverted BY the parent; every minus happens in the lawful parent; the own-body-in-child-frame row exists only as a shard→gateway type — no InterShardFlow arm exists for it, so the type system keeps it out of every realm. The restated-center leak vector leaves the wire with `center`. Boundary recorded (H13): SL1 seals shards, not eyes — the child's OCCUPANTS' clients lawfully see the parent's body (R1's explicit wording). |
| SL2 no occupant pose crosses | Entity lane untouched; window subscriptions carry no account/pose; `Occupants`/`Child` scopes are occupancy-bit-equivalent; the `Observed` scope is honestly declared as MORE and gated on the owner (Q2). |
| SL3 a realm draws itself | Self-look ships shard→gateway directly — the parent is removed from the look path entirely; the shape lane dies; the parent's per-child realm-lane message is now literally a placement and nothing else. Marker luma (dormant children only) is the one owner-ruled bend (R4), named verbatim in the ask, superseded by data presence the instant the child speaks. |
| SL4 physics/re-home separate | Emission consumes placements from the placement book; no motion symbol on the path; the gateway CANNOT hold motion (dependency gate) — it never asks how anything moves, and never interpolates/extrapolates. |
| SL5 one world | All gates on THE seed universe; boot file deleted; no fixture world; gate preconditions are THE world's solved numbers. |
| SL6 ask-first | §1.1 ask precedes Slice 0; the one genuinely new information flow (Observed scope) is its OWN item with the honest sentence "a realm learns a gateway observes it from outside" — the under-declaration hole (H3) closed. |
| SL7 parent decides AoI; nothing central | Membership = the PARENT's band/hysteresis verdict shipped as ids; the gateway never re-derives AoI (module dependency rule). Windows are per-chain, refcounted, zero-state-at-zero-sessions proven. No global assembly: chain derivation is session/stream-only, no seed table (H10 closed). |
| HR1 sealed shards | Net REMOVAL of four inter-shard arms; ZERO added; every new row is realm→connection-plane (R8's exact scope). Sealing strictly increases. |
| HR2/HR3 one machinery | One emission path for every realm kind; marker-vs-look is a data-presence/type branch, never a realm-kind branch; tier decision is representability. |
| HR4 features once | The window fixture passes identically on ≥2 shard profiles (G-IDENTICAL, Slice A). |
| HR5 100 % coverage | Composer as branchless generic shims + monomorphic helpers; the failure-mode table is the both-arms counter checklist; maximal reuse of ALREADY-covered `transfer_frame` core minimizes new branchy surface. |
| HR6 agent-operable | Every gate emits HR6 manifests; DevState carries origin/body-kind/newest-tick. |
| Server-composes / client-passthrough | The GATEWAY (server) composes ready-to-draw rows at one tick; the client applies zero transforms; mixed-tick composition refused by `InstantMismatch` at the type level. |
| Agnostic client | Pose + skip-unknown TLV bag; no realm-kind branch anywhere; marker/look is a data shape. |
| No prediction | Gateway composes only at authored common stamps (no interpolation — CF's rejected); client buffer unchanged, freeze-never-coast. |
| No magic numbers | Every bound in `WindowTuning`, derived (2 cadences + 1 pattern; ε from velocity × frame-dt; budgets from geometry). |
| Fences / attestation | Every row fenced; roster-head attestation fail-closed + counted on every lane. |
| Determinism seam | All stamps off the universe clock; Category-A math stays at its authors; compose at equal stamps only. |
| Never-assume | Every equivalence is a measurement: Slice-B parity gate, dedup f64 bound, inversion-inertness pin, exact-cadence pin, anti-vacuity shear failure. |
| Load tests with the subsystem | G-COMPOSE-LOAD covers compose AND ingest sides. |
| Never delete a scenario | All picture gates re-based; the failing `frame_conversion_e2e` violator flips green as a designed outcome and stays as the regression pin. |
| Postcard discipline | Mesh appends (minor 15); ONE owner-authorized flag day (minor 16, floor move, cited ledger); tombstone discipline preserved. |
| Economy | Untouched. |

---

# 4. SLICE PLAN (each lands green under `just gate`; HR5 Tier-A 100 %)

**Slice 0 — the ask, the rulings, the skeleton** (mesh minor 15).
Owner approves §1.1 and answers Q1–Q3. Wire types land: `WindowOpen/Close` + `WindowScope`,
`WindowFrame`/`WindowBody`(typed `BodyStmt`)/`WindowMembership`; attestation predicates;
positional/roundtrip pins; ledger entry with citation. Per-system taxonomy draw added to
`generate_system_forest` (pinned f(seed) values).
*Owner-visible outcome*: the signed ask + the wire contract, nothing moves yet.
*Gates*: wire pins, additive-decode, generator determinism pins, 100 % touched crates.

**Slice A — shard emits the window** (old lanes still running; no client change).
Window registry + TTLs, hop-row inversion at the author, look/marker/membership emits.
*Owner-visible outcome*: every realm publishes its window statements; the old picture unchanged.
*Gates*: unit 100 %; G-IDENTICAL (same fixture, ≥2 shard profiles); attestation fail-closed
(forged sender dropped + counted); TTL-expiry chaos (kill a gateway, fan dies by TTL);
INV-BODY-AT-ORIGIN pin; inversion-inertness pin; two-level fence live-sibling pin (per Q1).

**Slice B — gateway composition engine, SHADOW mode.**
`window.rs` complete: rings, `transfer_frame` fold, per-stratum hold + dead-hop exit, origin +
epoch, session-only chain derivation, dependency gate replacing the doc scan. Output compared —
never shipped.
*Owner-visible outcome*: a measured proof the new picture equals the old one.
*Gates*: **parity MEASUREMENT** (composed output vs live old-lane client feed, per realm per
tick, zero unexplained mismatches — could fail); G-SHEAR unit half incl. the deliberate
mixed-tick FAILURE; exact-cadence boot pin; dedup f64 measured bound; zero-state teardown test;
G-COMPOSE-LOAD (compose + ingest).

**Slice C1 — THE FLAG DAY** (client minor 16, floor → 16; the cut as small as the owner-mandated
flag day can be — the shard deletion does NOT ride this slice, per the feasibility verdict).
Client consumes the composed lane; `RealmShape.center` leaves; origin/epoch scene swap replaces
`forget_space`; `pin`/`render_pin` deleted; boot file + emitter + scripts deleted; picture gates
re-based.
*Owner-visible outcome*: the game runs on the composed picture; boxes come only from the stream.
*Gates*: full re-based e2e suite; G-RENDER-BOXES-SMOKE; G-RENDER-CROSSING (origin-marker J1 +
crossing no-flicker); G-SHEAR pixel half; `frame_conversion_e2e` violator flips green; ledger
citation gate; coverage 100 %.

**Slice C2 — the deletion + tombstones** (mesh minor; producer-less machinery removed).
FIRST: re-walk the §2.9 deletion map against the tree (line drift proven — SL1 filter at :8384).
Then: the four inter-shard arms tombstoned; the SL1 self-filter deleted with its lane (per Q3);
cascade/down-reflect/shape-hop/observed-interior machinery removed.
*Owner-visible outcome*: four world-to-world scenery lanes are gone forever; nothing changes on
screen (proven by the suite staying green).
*Gates*: intershard_closed pins (producer-less golden set +4); full suite green; coverage 100 %.

**Slice D — THE WARP ACCEPTANCE (capstone, R6). 🟩 LANDED 2026-08-16.**
Marker point-sprite rendering (luma-driven), roster-driven look pruning, symmetric handover
budgets — all three landed, plus the PILOT VIEW the acceptance turned out to require.

*Owner-visible outcome, delivered*: **the warp experience in pixels, on THE world.** Fly system
A → B: B is its parent's point of light at departure (3.00 px, 35 pixels painted at a local
probe), its footprint never shrinks while you approach (2018 samples, 3.00 px → 90.21 px, worst
sample-to-sample shrink 0.000 px), its own look takes over flicker-free, and it fills the view on
arrival (92.78 px, 25 150 pixels). A hands back to its parent's marker behind you and is drawn at
the shared point-of-light floor. Same-tick composition throughout; every drawn row's provenance
attested in the run manifest's state dumps.

*What landed, beyond the plan.* **The pilot view** (`client --capture-pilot`,
`vd_client_harness::camera::pilot_capture_camera`): the default capture framing fits the whole
drawn scene, and on a flight between two FIXED points of the star ring that frustum barely moves,
so a system you fly toward could not grow on screen at all. The warp is a statement about what the
PILOT sees, so the agent's eyes now look through the pilot's eyes — at the avatar's own eye, along
its DELIVERED facing, through the ONE Tier-A expression the gates reconstruct the camera by. The
scene-fitting framing stays the default; every existing box gate is byte-identical.
**★ And a REAL DEFECT the gate exposed**, root-caused by measurement: `nav::look_at` had no brake
against the delivered-pose feedback lag (the one `walk_to` has had since Stage B4). A 90° `LookAt`
reported ALIGNED at 0.0030 rad and then rotated another **0.6030 rad** as the in-flight deltas
landed — a 425 m miss over the ring, which is how the first flight sailed past its destination.
Cured with `LOOK_FEEDBACK_STEPS = 4`, the same measured lag the walk brake is sized to.

*Gates, with their measured numbers (THE world, seed 0, 500 m/s, 50 Hz, 10.0 m per tick).*
Budgets are counted in TICKS off the landed cadences and reported in metres by the occupant's own
travel — which is what makes a handover honest whether the ship is flying or parked when it lands.
- **G-WARP-PIXELS** (`just warp-pixels`) — the acceptance bar, DevState-driven detection + LOCAL
  pixel probes at the projected position, never a full-frame search; every wait on the demand
  loop's own signal.
- **G-HANDOVER**, both directions, symmetric derived budgets. WAKE = 2×25 AoI cadence + 1
  reconcile + the cluster's OWN measured boot + 2 Q2 relay + 1 compose + 1 draw: **measured 14
  ticks / 140 m vs 61 ticks / 610 m**. DEPARTURE = 51 grace-hold + 25 cadence + 1 hop + 1 compose
  + 1 draw: **measured 47 ticks / 470 m vs 83 ticks / 830 m**. **THE Q2 RELAY HOP, asserted APART
  per the owner's ruling: 13 ticks / 130 m vs 60 ticks / 600 m — it does NOT break the wake
  budget, so D-WINDOW-2 stays closed and the direct window remains a fresh owner ask.** A
  companion test closes the derivation before any process runs, so a world-numbers change fails in
  milliseconds rather than only in a four-minute flight.
- **G-TWO-SHIPS** (`just two-ships`, owner-ordered) — two real capture clients, two realms of
  different depth: (a) the two chains agree on the star-to-planet separation to 0.0612 m against
  the planet's own 1.4646 m of travel over the sampling gap, each hull drawn by its OWN look and
  pixel-probed at 50 px; (b) the inner observer draws the Planet's own body around itself AT THE
  ORIGIN, in pixels; (c) 2139 relayed statements ingested, 0 undecodable, attested in both
  manifests; (d) one watched crossing — the crossing client's epoch bumps exactly once, the
  watcher's does not move, and its picture is sampled continuously through the commit (412 samples,
  3288 continuity checks, worst 1.088 m against its own 10.814 m allowance); (e) occupant figures
  ABSENT from the window lane and ABSENT out of a realm. NOTE on "ship X / ship Y": `RealmId::Ship`
  realms are structurally impossible before P8 (D-SHIP-1), so the two hulls are the two REALMS the
  players stand in — the star System and one of its Planets — which is exactly what each assertion
  tests. THE WORLD IS UNCHANGED (SL5).
- **G-SHEAR full**: the pixel half joins the Slice-B unit half — every captured frame's drawn rows
  are inside the composer's declared retention (measured spreads 0–2 ticks).

DEFERRED flips: D-LANE-4 🟩 (its owed client-side half — the point sprites and the pruning — is
what landed here), D-LANE-6 🟩, D-PLACE-3 🟩; **D-WINDOW-1 flips 🟩, the ladder complete**.
D-PLACE-2's residual SHRINKS but STAYS OPEN — Slice D touched neither of its two live readers, and
saying otherwise would be a claim rather than a measurement. New owed rows registered: D-WINDOW-3
(census-scale photometric derivation + the HDR exposure model the point sprite stands in for),
D-WINDOW-4 (P10 rotated-hop composition), D-WINDOW-5 (the §2.14 LOD tier seam for P4+), D-WINDOW-6
(two things this slice MEASURED and did not assert). Global progression report to the owner.

---

# 4.5 COMPONENT-BY-COMPONENT OWNER APPROVAL (2026-08-15/16, the five-topic walk)

The owner walked the design component by component and approved all five:
- **Topic 1 — the wire contract** (five realm→gateway statement kinds + window control;
  zero realm→realm scenery): APPROVED. Q&A recorded in §2.14: gateway failure posture
  (nothing durable, re-derivable, TTL-expired fans), shared per-(origin,tick) composition,
  and the LOD tier seam.
- **Topic 2 — the shards** (witness, never courier; serialize once; three foreign stores +
  relay transforms + four audited receive paths deleted): APPROVED.
- **Topic 3 — the gateway composer** (fold once per occupied realm per tick; ~2% of one core
  per 100 occupied realms; no physics linked, structurally unable to author): APPROVED.
  AAA refinement adopted: the load gate asserts p99 compose+fan < one tick, not the mean;
  sessions-per-occupied-realm ratio on the diagnosis surface.
- **Topic 4 — the client** (one feed, one frame, epoch-swap replaces inference; boot file +
  guessing code deleted; one-space/echo guards die only in the commit that removes their
  cause, post-C1): APPROVED.
- **Topic 5 — the proof ladder** (shadow parity before any client cut; mismatch CLASSES
  reported; a soak added to slice B's exit; pixel gates DevState-driven; old lanes deleted
  never disabled): APPROVED.

# 5. OPEN QUESTIONS FOR THE OWNER

## RULINGS (owner, 2026-08-16 — the questions below are ANSWERED; kept for the record)

- **Q1 = YES, and GENERIC**: one level into ANY live realm you are next to — never a
  planet-specific case. A live station shows its areas' outlines; a live ship shows its rooms'
  outlines; identical code path, pinned on ≥2 realm kinds (G-IDENTICAL discipline).
- **Q2 = PARENT RELAY**: the parent forwards its live children's self-authored statements
  VERBATIM (child's fence + attestation intact; no store, no merge, no re-state, no read) to
  interested gateways. `WindowScope::Observed` never ships. The no-flicker gate (G-HANDOVER)
  explicitly measures the relay hop at the wake moment; the direct window is ledgered as the
  named per-realm upgrade taken only if that measurement fails. Rationale: zero new edges
  (reuses child→parent + parent→gateway, both TTL-guarded/attested/chaos-tested); re-home
  followed automatically by the parent's existing holder tracking; "am I observed from
  outside" stays UNREPRESENTABLE in every realm.
- **Q3 = APPROVED (owner, 2026-08-16)**: the SL1 self-placement filter is DELETED in Slice C2
  together with the lane it guards — the behavioral guard is replaced by the structural one
  (no realm-inbound message type carries a placement field: compile-time pin + the absence
  assertion re-based onto the composed stream). This AMENDS the owner's decision-1 wording
  ("the filter stays") by the owner's own ruling; the C2 commit MUST cite this ruling in the
  ledger so the audit trail shows retirement by amendment, not erosion.
- **THE SL6 FORMAL ASK (§1.1) = APPROVED through the five-topic walk (2026-08-15/16)**: Topic 1
  approved the five statement kinds + window control with the full SL6 form (what data, from
  which to which, why the receiver cannot compute it, cost of doing without). Item 5b
  (`Observed`) is MOOT under Q2 = relay and never ships.
- **G-TWO-SHIPS (owner-ordered named pixel gate, Slice D)**: ship X (player 1) in the System,
  ship Y (player 2) on the Planet, hulls mutually visible across the Planet boundary. Asserts,
  in pixels on THE world: (a) each observer draws the OTHER hull at its same-tick composed
  position (mixed-age strata forbidden — the shear law across two chains of different depth);
  (b) player 2 draws the Planet's own body around them (the hop row — the decision-1 hole
  closed, in pixels); (c) each hull's look is the REALM'S OWN statement, delivered via the Q2
  relay, provenance attested in the manifest; (d) one crossing while both watch: X crosses
  into the Planet; both observers' pictures stay continuous (screen delta ≤ one tick of true
  motion; epoch bumps exactly once for X's own client; the WATCHING client's picture never
  jumps as X's position author flips at the commit). (e) Occupant figures through windows are
  asserted ABSENT (the D-RLM-18 remote-figure lane is a future owner-gated ask; it must not
  sneak in). Exercises Q1-generic, Q2-relay, the hop row, and the crossing swap in one scene.

**Q1 — Are a live sibling system's planets visible from OUTSIDE the system?** (Judges found the
three designs read the two-level visibility bound oppositely; the boot fence is real but this
edge is exactly at its boundary.) If YES: the live-child interior lane exists and G-WARP asserts
"planets appear sub-threshold" before the crossing. If NO: the lane is deleted everywhere and
G-WARP's planet clause moves to post-crossing. One ruling swings a whole lane and one gate
assertion; nothing lands until it is answered.

**Q2 — If Q1 = YES: direct window or parent relay?** Default proposal: the gateway subscribes
DIRECTLY to the live sibling (fastest look at wake, no relay hop) — but this tells that realm
"some gateway observes me from outside", which is genuinely new information into a realm-adjacent
process, so it is its own ask item (§1.1 item 5b), default NO until you say yes. Fallback if
refused: the parent relays its live children's self-statements one hop (lane-legal, one relay
hop of look latency, budgeted in G-HANDOVER).

**Q3 — The self-placement filter.** You said the filter stays. This design deletes it in Slice
C2 — TOGETHER with the lane it guarded, because the hazard (a restated center on the wire) ceases
to exist; that is strictly stronger protection. Confirm the deletion, or the filter stays as
dead code with a tombstone note.

---

# 6. JUDGE HOLES REGISTER (every named hole → resolution)

| Hole | Resolution |
|---|---|
| H1 (CF, CRITICAL: no subscription mechanism) | Moot — CF's push model not adopted. WM's `WindowOpen/Close` kept, upgraded to typed `WindowScope` (§2.3). |
| H2 / B-5 (CF: gateway interpolation breaks R6) | NOT grafted, explicitly rejected. Compose at exact authored common stamps only; `InstantMismatch` refuses everything else (§2.6.3). |
| H3 (WM: live-sibling window under-declared in SL6) | The `Observed` scope is its OWN honest ask item ("a realm learns a gateway observes it from outside"), owner-gated as Q2; fallback = parent relay (§1.1/5b, §2.3). |
| H4 (CC: parent's visibility verdict missing on render path) | WM's `WindowMembership` kept — the parent's SL7 verdict gates bodies/interiors (§2.2). |
| H5 / C-2 (CC: two-feed author contradiction) | Everything rides the ONE composed feed via the own-level `Occupants` window (hop: None); single author, single counter, no seam (§2.2/2.4). |
| H6 (CF: looks on an unreliable lane) | `WindowBody` is ReDriven/reliable (§2.2). |
| H7 (WM: WindowOpen child:None overload) | Typed `WindowScope` with per-scope stated semantics (§2.3). |
| H8 (CF: arm 34 vs "no realm receives anything new") | Moot — no new InterShardFlow arm anywhere; the up-lanes are tombstoned, looks go shard→gateway (§2.5, §2.10). |
| H9 (WM: body-at-frame-origin assumed) | INV-BODY-AT-ORIGIN named invariant + pin (§2.12). |
| H10 (WM: f(seed) topology table at gateway) | Deleted — session/stream-only chain derivation from login lineage + attested hop rows + roster (§2.6.2). |
| H11 (all: marker luma bends SL3) | Named verbatim in the ask (item 3b) and in the law table; owner-ruled by R4; superseded by presence (§1.1, §3). |
| H12 (CC: parent keeps child-interior up-lanes) | Not retained — all four scenery up-lanes tombstoned (§2.5). |
| H13 (all: occupants' clients see parent body in child frame) | Recorded in the law table: SL1 seals shards, not eyes — matches R1's wording (§3). |
| A-1 (WM: whole-sky freeze; no dead-hop exit) | Per-stratum hold: sky above the fault holds at last composed poses, local continues; DERIVED dead-hop exit → reliable removal; counted + alarmed (§2.6.4). |
| A-2 (WM: markers membership-filtered) | Full direct-child roster always ships; membership gates bodies/interiors only — stars always visible by construction (§2.2). |
| A-3 / B-4 (departure swap unbudgeted/asymmetric) | G-HANDOVER symmetric derived budgets, both directions (§2.8). |
| B-1 (CF: no look path to a warp-transit observer) | Moot — the observer's own-level + chain windows carry looks for every chain level; the waking system's look arrives via its window (or Q2's chosen path). |
| B-2 (CF: tree-shaped truncation) | Rejected as first response; hold-then-fenced-removal instead (§2.6.4). |
| B-3 (CF: crossing pose gap) | WM's flip kept: old scene held until the new level (which CARRIES poses in SceneRows) lands; early new-epoch datagrams held one beat (§2.4, §2.7). |
| C-1 (CC: designed crossing blink) | SceneRows carry poses — drawable on arrival; no undrawn window exists (§2.4). |
| C-3 (CC: thin wake margin) | Budgeted in G-HANDOVER; direct window (Q2 default) removes the relay hop that thinned it (§2.8). |
| ALL-1 (two-level bound read oppositely) | Promoted to Open Question 1 + a named fence pin in Slice A (§5, §2.12). |
| ALL-2 (census-scale marker bound unproven) | Owed DERIVATION registered in DEFERRED at Slice D; full-roster markers make it a wire-volume bound, not a visibility bound (§2.10). |
| Exact-common-tick unpinned (both WM+CC) | Boot/CI invariant, loud failure (§2.6.3, §2.12). |
| SL1-filter line drift :8282-8294 → :8384 | Corrected; full deletion-map re-walk is Slice C2's first task (§2.9). |
| Nested realm_snapshot_bytes double-serialization | `WindowFrame.rows` is typed `Vec<RealmSnap>` — no serialize-inside-serialize (§2.2). |
| Dedup f64 agreement argued not measured | Measured bound inside the Slice-B parity gate (§2.6.5, §2.12). |
| "InstantMismatch structurally impossible" was an argument | Now the enforcement: composition runs through `transfer_frame` itself (§2.6.3). |
| Ingest-side cost unmodeled | G-COMPOSE-LOAD gates the decode side too (§2.6.7). |
| G-WARP dot detection flaky on raw pixels | DevState projected-position/body-kind + local pixel probes (§2.11). |
| WM Slice C blast radius (bundled cut) | Split C1 (flag day only) / C2 (deletions + tombstones) (§4). |
| FINE span ±0.952 ly slip | Corrected to ±0.476 ly (§2.8). |
| SL1-filter deletion vs owner's "stays" | Promoted to Open Question 3, explicit in the slice plan (§5). |
| T11/owner-eyes practice | Kept: every owner-instruction deviation is named in the plan (§4, §5). |
