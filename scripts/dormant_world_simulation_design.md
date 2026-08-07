# FINAL DESIGN — THE WORLDLINE: dormant-world simulation as a main-game substrate

## 🟥 REVERSAL NOTE (2026-07-27) — "the currency is a material" is REVERSED; money is a real currency again

**Read `scripts/money_and_markets_design.md`. It is the current word on money AND on markets.** On
2026-07-27 the user reversed the "currency is a material" choice and asked for a real transferable
currency, handled locally (station, city, star system) and globally (galaxy, universe), with banks
possible. This is a pointer only; nothing below has been rewritten.

⚠ **THE SUBSTRATE IN THIS DOCUMENT STILL SURVIVES, EXACTLY AS THE 2026-07-26 POINTER SAYS.** §4.1–§4.5 and
both of that pointer's corrections are untouched by the reversal. What changes is only how money uses it.

**REINSTATED, but resolved DIFFERENTLY — and this is the substantive change:**

- **§6.7's "money balances as zero-rate worldline subjects" is right in spirit and wrong as an
  address.** Balances genuinely are zero-rate stocks that do not advance while dormant, are loss-budget
  zero, and rehydrate through an adopt push. But money **cannot be realm-keyed**, and the code settles it:
  `to_realm_id` collapses Universe and Galaxy to fixed singletons (`crates/core/src/realm_path.rs:25-27`,
  `:74-92`), so **every galaxy is currently the same realm**, both stand-ins collide with any star system
  whose seed is 0 or 1, and the comment at `:77-78` says a later slice will *change* those identifiers,
  orphaning anything keyed by them. Two identity schemes are also in play and disagree under a re-parent.
  So account balances live at a dedicated always-on service keyed by the account principal, which embeds
  nothing — **not** as worldline subjects. Anything genuinely place-bound (a venue's takings box, a
  seller's proceeds, a mint's backing vault) still inherits this document's custody discipline, and is
  blocked behind per-place durable storage keyed by realm identity.
- **The woken-equals-never-slept determinism gate IS reinstated for monetary values**, in a corrected
  form: **money-family-scoped, not whole-store.** A whole-store byte-identity assertion fails on tick one,
  because `group_commit` writes the clock record and calls `commit()` unconditionally every tick
  (`crates/node/src/saga_runtime.rs:2507-2511`) — 50 fsyncs a second on an idle coordinator. Assert
  instead that the money family's keys and bytes are unchanged **and** that zero money-family deltas were
  staged. The second assertion is the one that actually proves nothing is clock-driven.
- **The dormant catch-up tier for balances stays UNNECESSARY.** A balance is a stock; nothing accrues.
  There is no interest, no upkeep, no account fee, no storage charge anywhere in the new design.
- **§7.3's two extra cross-server arms stay unnecessary.** The new design adds two *different* arms — a
  side-effecting payment request and a fire-and-forget balance read — for a different purpose, not to seed
  or journal a monetary substrate. The no-phantom-money-inflow gate and the dormant-cross-realm-zero-flow
  constraint remain absent by construction.
- **D-81 (addressing for cross-realm NON-SPATIAL state) becomes directly load-bearing again** — it names
  currencies explicitly — and the new design answers it **for money only**: a dedicated node kind, keyed
  by the account principal, deployed statically like the gateway. The precedent it says is missing does
  exist: `NodeKind::GalaxyRelay` is already a non-place node kind in the closed enum
  (`crates/sim/src/capability.rs:24-31`). Factions, wars, reputation and alliances are still unanswered.
- **D-83 (fault and liveness isolation for the economy port) is strengthened into a requirement**: the
  ledger gets a named per-tick operation budget with overflow queued to the next tick, plus a latency gate
  asserting that flooding payments does not move the crossing-saga latency percentile.

---

## 🔶 SUPERSEDING POINTER (2026-07-26) — the user has proposed a PURELY PLAYER-DRIVEN economy

**Read `scripts/player_driven_economy_comparison.md` before implementing from this design.** After this
document was written the user proposed removing economic **simulation** entirely: no computer-controlled
traders, no formula-driven prices, no background market evolution — only plumbing (companies,
player-built shops, stock, owner-set prices, buying, selling, ownership, taxes) plus telemetry, with all
behaviour from real players. That document evaluates the proposal, **recommends adopting it**, and is the
later decision where the two disagree. This is a pointer only; nothing below has been rewritten.

⚠ **THE SUBSTRATE IN THIS DOCUMENT SURVIVES.** §4.1–§4.5 (the lazy-field kernel, the production and
depletion integral with its piecewise breakpoints, the seasonal driver, the cohort integral, the sparse
absolute-rebase event log and compaction) ships **regardless**, because resource regrowth, structure
condition wear and creature/people life are main-game infrastructure. **Only the ECONOMY's use of it
disappears** — and §2 row 27 already made that split correctly ("structures wear as normal; nobody bills
you"). Anybody reading the proposal as "the worldline is cancelled" has misread it: the deletion is the
top layer, never the foundation.

**UNNECESSARY if the proposal is adopted:** §6.7 money balances as zero-rate worldline subjects
(`:2104-2138`); the woken-equals-never-slept determinism gate applied to **monetary** values; the
dormant catch-up tier for balances; §7.3's two extra cross-server arms that exist only to seed and
journal the **monetary** substrate (`:2252-2313`) plus the no-phantom-money-inflow gate and the
dormant-cross-realm-zero-flow constraint they protect; the money half of §6.6's live-invariant halt flag;
and eleven **monetary** rows of the §2 layer table (4, 14, 15, 16, 17, 18, 19, 20, 21, 24, 26 and the
monetary halves of 27/28) — absent by construction rather than absent by flag. §2 row 4's NPC
**trading-policy** port also goes; the existence/needs/pose/routine split above it **survives**, with its
price-free needs-driven default becoming the only variant.

⚠ **TWO CORRECTIONS THAT CHANGE A CONCLUSION IN THIS DOCUMENT:**
1. **§3.4 / `G-EMPTY-SESSION-ONLY` / the `:2742` pre-terrain row are MIS-SCOPED.** They schedule
   `is_session_occupant` "before the first NPC entity". **The first durable non-session entity in this
   game will be a SHOP, not a creature** — commerce lands at P6, creature life is later and behind an
   unbuilt port. Verified: `aoi_decide`'s occupant fold takes every simulated owned dot
   (`crates/sim/src/stub.rs:4397-4410`) and a shop is a `NamedConstruction` (`Durable`,
   `GhostPolicy::Always`; `crates/core/src/entity_kind.rs:216-223`). One shop makes every
   commerce-bearing realm permanently un-reapable, **silently**. Re-label the prerequisite: *before the
   first durable non-session entity*.
2. **§2 row 27's "a structure can decommission on its own schedule" while dormant COLLIDES with the
   flagship byte-identity claim** and would destroy player property held inside it. New binding rule
   (**RULE PDE-1**, §3.2 of the new document): while dormant, a condition stock may **drain** but may
   never **cross a terminal threshold** — decommission, collapse and spoilage (`W4`'s retained spoil arm
   included) are authored only by a **live** shard on wake, with per-item typed loss reasons. Clamp every
   dormant closed form one unit above its terminal value.

**WEAKENED:** §3.2's `EconomyPort` collapses to **one** method returning `()` — but `EconCommand` must be
**kept unexercised**, not deleted, because `:2790` correctly names it the one thing that cannot be added
behind an emit-only port later. And `G-ECON-ABSENT` (`:596-604`) becomes **vacuous** — it would stay green
with the port wired to nothing — so it needs an anti-vacuity control (assert the economy-on arm emits
every declared fact kind and the economy-off arm loses a **named** capability).

**UNCHANGED and still binding:** LAW-E1 and its four machine checks; LAW-E2 for the **physical** world;
LAW-WL-1, LAW-WL-2 (the fence-dominant fold + custodian stale-reject) and LAW-WL-7; `RULE WL-ITEM`,
`RULE WL-CONSERVED-FACT`, `RULE WL-LIEN`, `RULE WL-AGGREGATE`, `RULE WL-LIVENESS`; §4.8's separate
read-only analytics handle; every §11.1 pre-P4 seam (integer kernel, no-seconds rule, integer generator
boundary, `WorldEpoch`, closed-form ore distribution, and the **minted realm identity** as primary
durable key — which `grep` confirms does not exist at all today); **D-78** (decouple the addressable-place
count from the 64-slot membership word), which becomes *more* important because world size is now the
liveliness lever; and every physical row of the §2 table.

---

> Four designs were produced and judged on three lenses (LAW-E1 decoupling; LAW-E2 believability at
> scale; hard-rules/determinism/cost). Scores: **A 8.5 / 8 / 9**, **C 9 / 6.5 / 8**, **B 6.5 / 5.5 / 6**,
> **D 4.5 / 7.5 / 4**. This document is **Design A's core** (`vd-core::worldline`) with **Design C's
> record discipline grafted wholesale**, **Design B's three cheap written laws**, and **Design D's two
> salvaged artefacts plus its coarse agent tier as an explicitly optional later layer**. Every fatal flaw
> the judges named is repaired in-line and the repair is called out. Alternatives are recorded in §12 so
> the user can overrule.
>
> **Every repo claim below was read from the code at HEAD `4a1f7e1`.** Where this document contradicts
> `scripts/economy_research_20260726.md`, `scripts/realm_lifecycle_design.md`, or the design prose in
> `docs/design/`, **THE CODE WINS** and the contradiction is named. Claims that could not be verified are
> marked ⚠ **[U]**.
>
> **Revision 2 (2026-07-26, post-adversarial-review).** Four adversarial vetters reviewed revision 1;
> §14 "Review record" lists the verdicts, every sustained finding, and the three findings rejected on the
> merits. **Eleven corrections changed a CONCLUSION, not a number**, and they are the reason this revision
> exists:
> 1. The fold had **no fence term**, so a force-reaped-but-still-running incarnation could revert a
>    successor's rebase and **mint material**. LAW-WL-2 now keys on `(realm_fence, tick, seq)` with the
>    fence dominant, plus a custodian stale-reject (§4.4).
> 2. The compaction reader rule was keyed on **tick alone** while the fold key is `(tick, seq)`, so a
>    same-tick deviation was silently dropped — again minting material. Fixed to a `(tick, seq)` high-water
>    with an exact-key-set delete (§4.5).
> 3. `EconomyPort::weights()` fed an economy value into NPC decisions, and NPC decisions author durable
>    deviations — so **the economy determined physical world state** and the flagship gate was
>    unsatisfiable with the economy on. `weights()` is **DELETED**; the economy→game direction is now an
>    explicit journaled **command** channel, which is also the answer to the report's §7.18 (§3.2, §2 row 26).
> 4. Materialised NPCs entered `aoi_decide`'s occupant fold, making an NPC-bearing realm **permanently
>    unreapable** and `G-WL-LIFECYCLE-BLIND` unsatisfiable. Fixed by a real code mechanism
>    (`is_session_occupant`), not a gate (§3.4, §6.3).
> 5. **Every closed form was monotone-to-a-cap**, so each realm has a computable *believability horizon* —
>    **23.15 days in this document's own worked example** — after which `evaluate(t)` is constant forever.
>    A seed-derived **integer seasonal driver** is added (§4.3a) and `G-WL-DORMANT-MOVES` becomes
>    multi-horizon (§10).
> 6. `RealmKey` was frozen at **6 levels with no depth invariant**, so a 7-level path silently ALIASES two
>    realms' durable rows; and because the key *is* the lineage, a **re-parenting (mobile/ship) realm
>    orphans its own state**. Durable rows are now keyed on a stable minted `RealmUid`, with a
>    variable-depth path key demoted to a secondary subtree index (§11.1).
> 7. **W2(a) had no third option** and its compaction sweep needed an unbounded family prefix scan inside
>    a **512 MiB GUARANTEED-QoS** orchestrator. Compaction becomes **lazy-on-adopt only**, and a dedicated
>    custodian StatefulSet is added as option (c) (§4.5, §5.4, W2).
> 8. Cross-realm dormant supply is **algebraically incompatible** with the substrate, not merely deferred;
>    v1's only legal dormant flow is **zero**, enforced by a registry validator (§7.2 #7).
> 9. The worldline's seed-derived **INPUTS** (`K`, `S₀`, extents) come from the **f64/libm** generator, so
>    the design's "no floats ⇒ bit-equal today" claim was false at its own boundary. An **integer generator
>    boundary** rule plus a cross-binary gate replaces it (§9.1, §9.5).
> 10. `promoted_count` was an **increment** — the one non-idempotent operation LAW-WL-2 exists to forbid.
>     It is now DERIVED from the row set (§6.3).
> 11. `RULE WL-ACK` put the orchestrator on the **critical path of every player mutation**, with the
>     adopt-lost case unspecified. It is re-cast as a durability barrier with locally-enforced budgets and
>     a normative adopt-lost rule (§4.6, §5.5).

---

## 0. One-line thesis, and the five corrections that make it true

**The dormant world is not simulated slowly, cached, or shipped — it is EVALUATED.** A realm's physical
state at any `universe_tick` is `worldline(realm, t) = evaluate(baseline, deviations_before(t), t)`: a pure
integer function in `vd-core`, O(1) in elapsed ticks, with **no process, no timer, no storage, and no
cross-shard byte** for a dormant realm. Believability comes from the fact that the only inputs are (a) the
seed and (b) the sparse, absolute, durable record of what somebody actually *did*.

Five corrections separate this from the naive version. The first three were in revision 1; the last two
were forced by the adversarial review and are what make the thesis survive contact with LAW-E2 and LAW-E3.

1. **TWO RECORDS, NOT ONE.** The **worldline state** (bounded, authoritative, required: ~630 B–5.3 kB per
   *touched* realm, **0 bytes** for an untouched one) is a different object from the **fact journal**
   (unbounded, historical, optional: 2.3–4.1 TB/yr). Design C proved the journal cannot live on a shard's
   256 Mi PVC (`deploy/k3d/50-shard.yaml:94-96`); Design A proved the state can live anywhere. Conflating
   them is what made C expensive and what left A without a history. Separating them means **the world
   advances with the archive absent**, and the archive's availability is never a gameplay precondition.
2. **THE CLOSED FORM IS THE MASTER (LAW-WL-1).** A *live* shard's per-tick delta is `F(t) − F(t−1)`, never
   an independently accumulated counter. Live == dormant becomes true **by construction** rather than by
   reconciliation — which is the hardest property in any dormancy design, and here it falls out for free.
   It is also a permanent design tax on P6/P8/P11 (§12, weakness W-1) and the user must accept it knowingly.
3. **HISTORY IS AN OPTIONAL LAYER, NOT A PROPERTY OF THE SUBSTRATE.** A closed form advances state but
   cannot *generate events* — no wreck to find, no station that changed hands with a date. Design D's
   coarse agent tier is the only mechanism in the field that produces autonomous history, and it is
   admitted here as **an optional layer over the sparse installed-capacity subset**, on Design D's pin,
   with its flow-driven `KeepAlive` deleted (§5.6, §11 phase W-6). Design D's own fallback *is* the closed
   form, so this is a superset relationship, not a choice.
4. **A CLOSED FORM THAT SATURATES IS A FROZEN WORLD (LAW-WL-6).** Every monotone-to-a-bound law has a
   *believability horizon*: production stops at the hopper cap, depletion stops at zero, population stops at
   `K`. In this document's own worked example that horizon is **day 23** — after which a 6-month dormancy and
   a 5-year dormancy return **bit-identical** state. Revision 1 computed the horizon (§5.4) and did not draw
   the conclusion. The substrate therefore carries a **mandatory non-saturating driver**: a seed-derived
   **integer seasonal rate/capacity table** indexed by `(t / season_quantum_ticks) mod W` (§4.3a), whose
   integral is still exact, integer and O(1) in elapsed ticks. Without it, LAW-E2 expires after roughly one
   month and the coarse agent tier (W5) is not optional but load-bearing.
5. **THE ECONOMY MUST BE ABLE TO CHANGE THE WORLD, AND EXACTLY ONE SHAPE OF THAT IS LEGAL.** Revision 1's
   port was `observe(facts)` + `weights(subject)`. `observe` is right. `weights` was wrong twice over: it fed
   an unfenced, unjournaled, off-tick-derived economy value into NPC decisions, and NPC decisions author
   durable deviations — so the *physical* layer became a function of monetary state (breaking the flagship
   gate and §8.4's replay obligation), while the economy still could not *deliberately* move an item, pay an
   insurance hull, or deliver a contract (report §7.18's exact complaint). Both are fixed by one inversion:
   the economy is an **ACTOR, never a callee**. It enqueues an ordinary fenced, refusable, idempotent
   **`EconCommand`** on the same path a player's action takes, and the game journals the resulting deviation
   as its own. Absence of the economy then means *"no such command was ever issued"* — never *"the game
   blocked"* and never *"the world state is unexplainable without the economy's history"* (§3.2).

---

## 1. The two laws, restated as engineering constraints

### 1.1 LAW-E1 (decoupling) → a one-way dependency rule with four machine checks

> *"If the whole economy is down or turned off, the main game can still run."*

**As a property, in two halves — and revision 1 stated only the first.**

**(P1) INERT-ABSENCE BYTE-IDENTITY.** For every scenario `S` in the accumulated `vd-tests` suite, the
observable result of `S` is **byte-identical** across three builds — (i) `vd-econ` not compiled,
(ii) `vd-econ` compiled with the port injected as `None`, (iii) `vd-econ` compiled with a **present-but-idle**
`MemEcon`. Byte-identity, not green-vs-green: the idiom already exists as
`shard_profile_swap_is_capability_inert_for_every_profile_kind` (`crates/sim/src/stub.rs:4762`).

⚠ **Two mechanical corrections revision 1 got wrong here.**
- **"(i) `--no-default-features` on `vd-bins`" was a no-op that could never change what the SUITE compiles.**
  Verified: `vd-bins` already has `default = []` (`crates/bins/Cargo.toml:58`), and `vd-tests` depends on
  `{core, wire, sim, node, harness, connection-plane}` and **NOT on `vd-bins`** (`tests/Cargo.toml:10-17`).
  A `vd-bins` feature therefore cannot alter the suite's build at all, so arms (i) and (ii) were the
  **identical build** and the three-arm gate silently degenerated to two. **Arm (i) means "`vd-tests` built
  WITHOUT its own non-default `economy` feature"** — so the feature is needed on **both** `vd-bins` (for the
  shipped binaries) and `vd-tests` (where the real injection happens), and both go in `just lint-combos`.
- **`NullEcon` and `MemEcon` live in `crates/sim/src/io/mem.rs`**, next to `MemStore`/`MemSpawner` — the twin
  pattern this design cites everywhere else. That makes arms (ii) and (iii) ONE build and lets the
  byte-identity comparison be an in-process digest compare instead of two suite runs (which is also the
  cheaper mechanism §13 open question 7 was groping for).

**(P2) ATTRIBUTABLE-PRESENCE CAUSALITY.** Byte-identity is only assertable while the economy is *idle*, and
saying only that would hide the real production question: a **live** economy issues `EconCommand`s
(§3.2), and those change the world on purpose. So the production-side property is: **with the economy
present and active, every physical divergence from the economy-absent run is attributable to a specific
`EconCommand` recorded in the GAME's own journal.** Nothing else may differ. That is strictly stronger than
"the suite is still green", it is what keeps §8.4's replay obligation ("the projection is a pure function of
the game's journal") true in the direction that matters, and it is gated by **G-ECON-ATTRIBUTABLE** (§10)
with a RED unattributed-divergence control. Revision 1 had neither the property nor the gate, and with
`weights()` in the port it could not have had them.

Four checks, of which three are compile- or build-time:

| # | Mechanism | Cost | What it makes impossible |
|---|---|---|---|
| **E1-a** | The **crate graph**, asserted by the gate that already exists. `tests/tests/crate_isolation.rs:15-37` parses `cargo metadata --no-deps` into per-package direct normal+build dep sets and asserts named forbidden edges with HR-labelled panics. | ~15 lines | `vd-econ` appearing anywhere in `deps(core\|wire\|sim\|node\|connection-plane\|harness\|client\|client-harness\|devproto)`. |
| **E1-b** | The **seam inversion**: an object-safe trait DEFINED in `vd-sim`, IMPLEMENTED in `vd-econ`, injected as `Option<Box<dyn … + Send + Sync>>` from `vd-bins`. Edge direction is **econ → sim**. | ~120 lines | A `vd-sim → vd-econ` edge existing at all. |
| **E1-c** | **A RETURN-TYPE-SHAPED port.** Both econ-facing methods return `()`; the econ→game direction is a queued `EconCommand` the game may refuse, never a value a game decision reads. | ~30 lines | The economy *answering a question a game code path is waiting on*. There is no such method to call. |
| **E1-d** | **The IMPLEMENTATION absent as a compile fact**: `economy = ["dep:vd-econ"]`, NON-default, on **`vd-bins`** *and* on **`vd-tests`**. The doctrine is written verbatim in that manifest: `dev-control` is "NON-default and cfg-gated, so a release `cargo build -p vd-bins` contains NO listener code at all — absence is a compile fact, not a runtime flag" (`crates/bins/Cargo.toml:53-57`). | 2 lines + 2 `lint-combos` cells | Economy **decision logic** existing in a release binary. |

⚠ **E1-c's mechanism was MIS-STATED in revision 1 and the correction matters, because the wrong version
invites a reviewer to expect a guarantee the compiler does not give.** Revision 1 claimed a *sealed*
`WorldFact` type makes "a game system receiving an authority-gating answer from the economy" untypable. It
does not, for two verified reasons. (a) The seal idiom (`crates/sim/src/coupling.rs:16-24`) is a private
`sealed::Sealed` **trait** in **`vd-sim`**; it constrains which payloads may be *admitted into* the marked
set — i.e. it constrains the **GAME→ECON** direction (who may add fact arms) — and a **`vd-core` enum cannot
implement a `vd-sim`-private trait at all**, so "sealed `WorldFact` enum in `vd-core`" (§11.2, and §6.5's
self-contradictory "sealed but OPEN-ARMED enum") is a category error. A `pub enum` is *already* closed to
outside variants; the seal buys nothing there. (b) The direction that needed constraining was `weights() ->
PolicyWeights`: a plain unsealed value type nothing stopped a game system branching on. **The actual
enforcement is the RETURN TYPE (`-> ()`) plus E1-a's crate graph** — and with `weights()` deleted (§3.2)
there is no econ→game value type left to seal.

⚠ **E1-d was OVERSTATED and is restated honestly.** The feature removes only `vd-econ`'s **implementation and
decision logic**. `EconomyPort`, `EconCommand`, `NpcStrategy`, `NeedsOnlyStrategy` and `WorldlineTuning` stay
in `vd-sim`; the `WorldFact` taxonomy stays in `vd-core`; the emit sites are deliberately **unconditional**
(§3.2). So a release binary always contains the economy's **seam**, its fact taxonomy, its null sink and its
price-free default policy — all Tier-A code owing 100 % region+branch. That is the *right* design (it is what
makes byte-identity meaningful rather than coincidental), but the reviewable claim is "nothing behind the
feature is reachable from a game code path", which is exactly what E1-a proves — **not** "no economy code
exists in a release binary".

⚠ **The gate has THREE verified bypasses that MUST close in the same diff** or the new rule ships with known
holes. Revision 1 listed two and missed the widest one:
- **DEV-DEPENDENCIES ARE SKIPPED BY DESIGN.** `tests/tests/crate_isolation.rs:29-31` is literally
  `if matches!(kind, Some("dev")) { continue; }` with the comment "dev-deps are test-only and never ship".
  So `[dev-dependencies] vd-econ` in `vd-sim` **passes forever** — which is enough to let economy types into
  `vd-sim`'s own fixtures and ratchet the API, and it instantiates econ code in a **Tier-A test binary**,
  which is precisely the HR5 cost §12.1 uses to reject Design C's 5th trait. The rule must additionally
  assert `vd-econ` is not a **dev**-dependency of any Tier-A crate.
- **THE CHECK IS DIRECT-EDGE ONLY, WITH NO TRANSITIVE CLOSURE.** The graph is built from per-package
  `dependencies` arrays under `--no-deps` (`:17`, `:24-34`), so **`vd-sim → vd-life → vd-econ` is green** —
  and this design itself proposes a `vd-life` crate at D-71. The assertion must be over the **transitive
  normal+build closure**.
- **IT IS A NAME BLACKLIST, NOT AN ALLOWLIST.** `!deps.contains(forbidden)` over a hardcoded list, with
  `internal()` filtering on `starts_with("vd-")` (`:109`). Any new game crate (`vd-life`,
  `vd-worldline-archive` per D-74, a future `vd-ai`) is **unguarded by default**, and a differently-named econ
  crate evades the exact-name check entirely. **Invert it: assert `vd-econ` is unreachable from ANY crate
  outside `{vd-bins, vd-tests, vd-econ}`, and assert the converse (only those three may name it).**
- `TIER_A` at `tests/tests/crate_isolation.rs:53-60` lists **six** crates (`vd-core`, `vd-wire`, `vd-sim`,
  `vd-node`, `vd-connection-plane`, `vd-harness`) while `justfile:12` lists **nine** (adds `vd-devproto`,
  `vd-client`, `vd-client-harness`). Those three are coverage-Tier-A but **isolation-ungated**.
- `the_dependency_law_holds_bins_to_node_to_sim_to_wire_to_core`
  (`tests/tests/crate_isolation.rs:102-128`) asserts only that `vd-core` has no internal deps, `vd-wire ==
  {vd-core}`, and `vd-sim ⊇ {core, wire}` while `⊅ {node, connection-plane}`. It asserts **nothing** about
  `vd-node`, `vd-harness`, `vd-connection-plane` or `vd-bins` — so a **`vd-node → vd-econ` edge passes CI
  today**, and that is the single most tempting placement (the directory CAS, the saga runtime and the only
  durable `Store` all live there).

⚠ **This OVERRULES `scripts/economy_research_20260726.md` §7.1** (`:1411`, `:1414`, `:1426`), which places
`vd-econ` **below** `vd-sim` with `vd-sim` driving it. That requires a normal `sim → econ` edge, links the
economy into every shard binary, and reduces LAW-E1 to a runtime flag. H1 and §7.1 cannot both stand; §7.1
is wrong and the seam inversion is the resolution. Precedent, twice in-tree: `Store`
(`crates/sim/src/io/mod.rs:417-434`, whose doc records object-safety as an *HR5 requirement* — "OBJECT-SAFE
by construction …, so there is NO per-monomorphization region gotcha") and `RealmSpawner`
(`crates/sim/src/io/mod.rs:436-479`), held as `Box<dyn Store + Send + Sync>`
(`crates/node/src/saga_runtime.rs:225`), injected at `crates/node/src/orchestrator.rs:152`, mirrored by a
deterministic `mem` twin.

**The critical placement consequence, and it is free:** the *physical* substrate lands in **`vd-core`**,
which `tests/tests/crate_isolation.rs:108-113` **already** asserts depends on no workspace crate. So "the
dormant world advances with zero economy code linked" is a **landed** build break, with no new enforcement
owed. No other candidate placement has that property.

**⚠ The residual nobody's design solved, stated as a design fact rather than discovered later:** the crate
graph is **not a fault boundary**. Every candidate runs economy-adjacent code inside a shard's `step_tick`,
so a panic there still kills the tick and takes pose authoring with it. LAW-E1 therefore carries a fifth,
non-structural clause: **RULE WL-PANICFREE — every worldline/economy code path is panic-free by
construction** (integer-only; all `expect`/`assert` confined to boot-time `validate()`; no `unwrap` — the
workspace already sets `unwrap_used = "warn"` at `Cargo.toml:96-99`), and its budget/refusal paths return
typed errors counted as faults, never panic. Gate: G-WL-NOPANIC (§10).

**⚠ And a SIXTH clause revision 1 omitted, because a panic is not the only way to fate-share a tick.**
`observe(&mut self, facts)` runs inside `step_tick` and takes `&mut self`, i.e. exclusive access on the
tick's critical path. WL-PANICFREE bounds *crashes*; nothing bounded **time or allocation**, so a slow,
looping or memory-hungry `observe` stalls the tick and takes pose authoring with it just as surely as a panic
— a hazard the stated rule did not cover:

> **RULE WL-LIVENESS.** The in-tick economy ingress is **`&self`, bounded, and does no user work.** The game
> owns a **bounded double-buffered queue**; `observe` is an O(n) copy into it with **drop-and-count on
> overflow** (a `worldline_econ_ingress_dropped` FAULT counter), and the economy **drains it off-tick**.
> Symmetrically, `EconCommand`s are drained from a bounded queue by the game at its own cadence. The in-tick
> cost is therefore `O(facts_this_tick)` with a named `econ_ingress_budget_bytes_per_tick` in
> `WorldlineTuning`, asserted **release-only** against a named latency budget with
> `vd_harness::latency::percentile_unstable` (`crates/harness/src/latency.rs:22`) exactly as
> G-WL-WAKE-LATENCY does — including a **hostile-econ cell** (an `observe`/`drain` that sleeps and allocates)
> proving the tick budget still holds. Gate: **G-WL-ECON-LIVENESS** (§10).

### 1.2 LAW-E2 (dormant believability) → four measurable properties

> *"Most parts of the galaxy most of the time are off… the economy should not be frozen because of that,
> so we need a believable simulation, based on the data we have."*

"Believable" is decomposed into properties a gate can check:

1. **NOT FROZEN, AT EVERY HORIZON** — for any dormant realm and any of `Δ ∈ {30 d, 180 d, 2 yr}`,
   `evaluate(·, Δ) ≠ evaluate(·, Δ − 30 d)`. ⚠ **Revision 1 stated this unconditionally at 30 days only, and
   the design AS WRITTEN FAILED IT at longer horizons** — every law was monotone toward a bound, so
   `evaluate` became constant after the believability horizon (day 23 in §5.4's own example) and a 30-day
   gate could not tell 30 days from 3,000. Correction 4 (§0) supplies the non-saturating driver (§4.3a) and
   the gate becomes **multi-horizon**: G-WL-DORMANT-MOVES.
2. **CAUSAL** — every difference is attributable to a seed fact, a recorded deviation, or a journaled
   `EconCommand`. There is no fourth input. Gates: G-WL-ZERO-BYTES (an untouched realm stores nothing, so its
   state is *provably* pure seed) and G-ECON-ATTRIBUTABLE (§1.1 P2).
3. **CONSISTENT WITH A LIVE-BUT-UNOCCUPIED INTERVAL** — the state a returning player finds is bit-identical
   to what the *same realm running continuously with no autonomous authoring in that window* would have
   produced. Gate: G-WL-CLOSEDFORM-EQ-ACCUM (**exact**, not within tolerance — LAW-WL-1 makes exactness
   achievable; see §10).
   ⚠ **Revision 1 claimed bit-identity against "a continuously live realm", full stop, and that is FALSE in
   the only interesting case and UNRUNNABLE as a gate.** Unrunnable: in the reaped arm there is no shard
   during `[k, T]`, so scripted inputs in that window cannot be delivered to anything — and if the log has no
   inputs there, the LIVE arm is occupant-empty, self-reports `DemandVerb::Empty`
   (`crates/sim/src/stub.rs:4410-4421`) and would itself have been reaped. False: a genuinely *occupied* live
   realm would have had NPCs acting, haulers draining the hopper, promotions and demotions firing — none of
   which happens dormant, by design. So the real property is the one stated above (it still proves LAW-WL-1,
   which is a statement about the closed form versus an accumulator, not about NPCs), and any live/occupied
   comparison is a **separate, weaker, attribution** gate (G-WL-ATTRIBUTABLE-DIVERGENCE, §10) — never a
   byte-identity one.
4. **CONTINUOUS (no pop)** — materialisation of a dormant field into entities happens strictly outside the
   observable extent, and a materialised body appears at its own closed-form pose **and velocity**, already
   moving correctly. Gates: G-WL-SEAMLESS-NPC (with the negative cell) **and G-WL-SEAMLESS-FIELD** — the
   second is new, because §5.6's no-pop argument covered only NPC pose/velocity and said nothing about the
   FIELD: on adopt, `evaluate` materialises machines and constructions all at once, and if the player is
   already inside the render extent when the realm boots (which is *why* it booted), those meshes pop in at
   close range.

⚠ **What LAW-E2 does NOT get from the substrate alone.** Three separate admissions, and revision 1 made only
the first:
- **No autonomous HISTORY.** Closed-form attrition is a rate, so populations fall believably but no
  individual battle happened and no wreck exists to find. That is §5.6's optional layer and **W15** in §12.
- **No POSITIVE dynamics at all.** Composing the mechanisms honestly: `K` rises only on a deviation, deposits
  only deplete, hoppers only fill, populations only relax toward a `K` that is static while dormant, and
  §6.5's future "structure decay/upkeep" is also monotone down. Nothing dormant *adds* capacity, discovers a
  deposit, completes a construction or repairs anything — those live exclusively in W5. So the emergent rule
  before §4.3a's driver is **"the parts of the galaxy you invested in decay; the parts you ignored are
  frozen"** — the inverse of a living economy. §4.3a's seasonal table is deliberately **bidirectional** (a
  rate that rises *and* falls) and `RegenPolicy::RespawnAfter` becomes the recommended **default** rather
  than an option, specifically so that "better" and "worse" both occur; anything richer needs W5.
- **A bounded PERCEPTIBLE RANGE.** Everything the substrate buys is delivered between roughly **one day and
  one month** of dormancy. Below it: a 5-minute round trip still pays a full wake (a realm is reapable within
  about a minute of the last occupant leaving, since `min_dwell = spinup_cooldown + launch_ttl`,
  `crates/sim/src/rlm.rs:129`) for a state delta of 6,000 ticks — in §5.4's example 150 input consumed
  against a 500,000-unit hopper, a **0.015 % change**, imperceptible. Above it: the horizon. §5.6 states the
  band as a design property and attacks both ends (a `min_dormant_ticks` time hysteresis at the short end —
  legal under LAW-WL-7 because it is a lifecycle-internal window, not an economy value — and §4.3a's driver
  at the long end).

**⚠ The anchor against overselling** (Design D's contribution, with revision 1's unsourced external figure
REPLACED). Revision 1 rested this on "X4: Foundations reads as alive at roughly 150–300 stations and
2,000–4,000 ships over ~50 sectors" ⚠ **[U — never re-sourced]**, and then derived "~60 fine-grained agents
per observed sector" from it. A load-bearing derivation must not rest on an unsourced figure, and it does not
need to: the argument is **geometric, not empirical** — perceived life scales with the region a player can
actually *see*, i.e. with the render/interaction extent and the content-AoI band this design already computes
(§6.3). So state it in-house: **the fine tier (real entities, real NPCs, real books) must saturate the
content-AoI band; the coarse tier buys consistency and causality OUTSIDE it.** The in-house target is
therefore "enough materialised subjects to fill one AoI band", derived from `max_materialized_npcs` and the
band radius at W-3, and measured against our own frame budget — not against another game's ship count.
Nothing in this design should ever be described as "the galaxy is visibly alive everywhere".

### 1.3 LAW-E3 (co-design) → this is a MAIN-GAME substrate; the economy is consumer #2

> *"The system design goes together with decisions we make for the main game."*

Consequence, and it is the reason the design is shaped as it is: NPC life, material stocks, production by
installed constructions, and destruction are **main-game subsystems**. They therefore live on the game side
of E1-a, and the substrate that serves them must be **ONE tooling** (HR3) serving:

| Consumer | Crate | Reads | Writes |
|---|---|---|---|
| the live realm shard (refine the field, materialise NPCs, author deltas) | `vd-sim` | `evaluate` | deviations |
| the orchestrator (compaction, spin-up initial condition) | `vd-node` | `evaluate`, `compact` | baselines |
| NPC life (population, routines, needs) | `vd-sim` + `vd-core` | `evaluate` | deviations |
| Tier-B dashboard / `vdctl` (evaluate-on-read) | `vd-bins` + `crates/io-prod/src/admin.rs` | `evaluate` | — |
| the harness oracle + counterfactual twin | `vd-harness` | `evaluate` | synthetic |
| **the economy (optional)** | `vd-econ` | `evaluate` | its own `Posting` facts only |

Because the kernel is in `vd-core` (zero internal deps, verified), every consumer reaches the *same* code
with no new crate and no new graph edge. HR3 is satisfied structurally, not by convention. **Corollary: a
new `vd-life`/`vd-econ` crate is NOT required for the substrate** — which also means `justfile:12`'s
nine-crate `tier_a` list needs **no edit** for the substrate itself (it does for `vd-econ`, see §11).

---

## 2. THE LAYER SPLIT — the table that makes LAW-E1 real

**Layer definitions.** **PHYSICAL (P)** = state that must advance with the economy absent; lives in
`vd-core::worldline` + `vd-sim`; authored by the realm's own shard while live, by the closed form while
dormant. **MONETARY (M)** = an optional projection over the physical layer; lives in `vd-econ`; absent
means absent, never stale.

**Verdict legend.** **DECOUPLED** = byte-identical with the economy absent. **EMIT-only** = the game action
is unchanged; only the accounting posting is missing. **COUPLED/fail-open** = the action *wants* an economy
answer and MUST proceed without one. **M-internal** = economy-only; absence is the declared degradation.

⚠ **THE AVAILABILITY COLUMN, which revision 1 omitted and which changes how the table must be read.**
"DECOUPLED" answers *one* question — "is this byte-identical with the ECONOMY absent?" — and revision 1 then
let the reader infer the stronger claim "this is unconditionally available". It is not: under RULE WL-ACK
(§4.6) with W2(a) custody, several physical rows are **decoupled from the economy but coupled to the
CUSTODIAN**. §12.2 W2(a) admitted this in prose ("'the economy is off' can never mean 'the orchestrator is
off'") and never propagated it into the verdicts, which is exactly the kind of quiet asymmetry LAW-E1 exists
to forbid. The **Custodian** column is therefore mandatory, and §4.6's reshaped WL-ACK is what keeps every
entry in it either "no" or "durability only, never authority":

| # | Subsystem | Layer | Authority (single writer) | Advances with economy OFF? | Needs the CUSTODIAN? | What the player sees if OFF | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **NPC existence, population, cohorts** | P | live realm shard; closed form while dormant | **YES — identical** | no | identical | DECOUPLED |
| 2 | **NPC pose / routine / activity mix** | P | containing realm's shard (standing frame-authority law) | **YES — identical** | no | identical | DECOUPLED |
| 3 | **NPC needs (physical stock draws)** | P | live realm shard | **YES — identical** | durability only | identical | DECOUPLED |
| 4 | **NPC trading POLICY** (what to quote, at what price) | M | injected `NpcStrategy` behind the port | falls back to `NeedsOnlyStrategy` (price-free, needs-driven) | no | NPCs still hunt, haul, work, flee, fight — they choose by physical scarcity, not price | **COUPLED, declared default** |
| 5 | **Materials: deposit stock + depletion** | P | live realm shard; `evaluate` while dormant | **YES — identical** | durability only | identical | DECOUPLED |
| 6 | **Material regeneration / respawn** | P | closed form (seed) + deviations | **YES — identical** | no | identical | DECOUPLED |
| 7 | **Production by installed constructions** (refinery/port/factory) | P | live realm shard; `advance` while dormant | **YES — identical**; this is precisely what LAW-E2 demands | durability only | the refinery kept refining | DECOUPLED |
| 8 | **Construction install / completion / decommission** | P | live realm shard (a deviation) | **YES — identical** | durability only | identical | DECOUPLED |
| 9 | **Destruction / wreck / salvage yield** | P | the destroying shard | **YES — identical physics**; only the `Sink::Destruction` posting is missing | durability only | blocks/ships break normally; loot drops per its per-entity `loot_drop_ratio` | EMIT-only |
| 10 | **Mining a voxel** | P | the mining shard; yield is `f(seed, realm_path, voxel_pos)` — a **content fact**, never an economy answer | **YES — identical**; only `Faucet::Extraction` is missing | durability only (per §4.6's optimistic-ack rule the voxel edit itself never waits) | identical | EMIT-only |
| 11 | **Crafting / refining recipes** | P | the **GAME** registry in `vd-core`, next to `KindDef::is_coherent` (`crates/core/src/entity_kind.rs:175-191`), with its DAG-acyclicity validator **there** | **YES** | no | identical | EMIT-only ⚠ *conditional: put the recipe registry in `vd-econ` and crafting stops when the economy stops* |
| 12 | **Inventory / items (positions)** | **P** | the container's owning shard; the GAME's store is the uniqueness authority | **YES — identical** | durability only (P6/W-6: local once Store B lands) | identical; no-dupe still provable (§8) | DECOUPLED |
| 13 | **Docking / undocking** | P | the transfer saga (a containment re-home: `crates/sim/src/stub.rs:9315-9326`, `:11282-11307`) | **YES — docking was never an economy interaction** | **no — never** (a movement primitive may not wait on any remote answer) | identical | DECOUPLED |
| 14 | **Docking FEE** (if ever added) | M | `vd-econ` | no fee charged | no | you dock; the fee becomes a post-hoc receivable | **COUPLED, FAIL-OPEN — never a precondition** |
| 15 | **Prices (ambient + book)** | M | `vd-econ` | no | no | "no market data" (fail-loud in UI, never a stale zero) | M-internal |
| 16 | **Money / wallets / balances / M(0)** | M | `vd-econ` ledger; **its durable position is a zero-rate worldline subject** (§6.7) | no | durability only | wallet UI unavailable | M-internal |
| 17 | **Markets / order books / escrow** | M | `vd-econ` | no | no | no venue | M-internal ⚠ *and see row 28: an economy object may never be the container of record for goods* |
| 18 | **Taxes** | M | `vd-econ` | no | no | untaxed | M-internal |
| 19 | **Contracts / collateral / courier adjudication** | M | `vd-econ` | no | no | cannot post contracts; physical delivery still works | M-internal |
| 20 | **Equity / corporations / dividends** | M | `vd-econ` | no | no | unavailable | M-internal |
| 21 | **Insurance PAYOUT** | M | `vd-econ` | no payout now, **retro-payable later ONLY for the fact classes §4.6 puts in the STATE record** | no | pays out when the economy returns | M-internal, **retro-recoverable — conditionally, see below** |
| 22 | **The destruction/delivery FACT that insurance and tax depend on** | **P** | the GAME's journal | **YES** | — | — | **DECOUPLED — and this single placement carries most of LAW-E1's practical weight** |
| 23 | **Dashboard / analytics: physical half** | P (Tier-B read) | `io-prod::admin` running the same `evaluate` over a **separate read-only store handle** (§4.8) | **YES** | read-only, budgeted, never the ECS `StoreRes` | physical panels live | DECOUPLED |
| 24 | **Dashboard: monetary half** | M | Tier-B | no | no | absent panels | M-internal |
| 25 | **Realm lifecycle (spin-up / spin-down)** | P | the RLM reconciler, **sole kill authority** | **YES, and BIT-IDENTICALLY** (LAW-WL-7, §3.4) | no | identical | DECOUPLED, **gated** |
| **26** ⚠ *new* | **A monetary event CAUSING a physical mutation** (a filled trade moves goods; insurance yields a hull; a wage/contract delivers; a bought station exists) | **P outcome, M trigger** | the GAME — the economy is an ACTOR that enqueues a refusable `EconCommand` (§3.2), never a callee the game waits on | **YES — the world is unchanged; the command was simply never issued** | no | your goods are not delivered by the *economy*; every physical primitive still works and every obligation becomes a receivable | **COUPLED, FAIL-OPEN, ONE-DIRECTIONAL** |
| **27** ⚠ *new* | **Recurring asset sinks (storage rent, structure upkeep)** — report **D21**, decided YES there and missing from revision 1's table | **split** | the **PHYSICAL** half is a worldline `Construction` **condition** stock draining at a per-entity integer rate (a normal closed form, §4.3); the **MONETARY** rent is a `vd-econ` receivable | **YES** — condition keeps draining and a structure can physically decay/decommission on its own schedule | durability only | structures wear as normal; nobody bills you | DECOUPLED (physical) + M-internal (rent) |
| **28** ⚠ *new* | **Goods held in escrow / posted as collateral** | **P** | a **GAME** container (the venue realm's own physical container) with an economy **LIEN** referencing it — never an econ-owned container | **YES** | durability only | the goods are exactly where they physically are; the lien is unenforceable | DECOUPLED, **by the rule below** |

**Row 22 is the load-bearing one.** If the fact log lived in `vd-econ`, an economy outage would be
*permanent history loss* — retro-payouts, retro-taxes and reconciliation would be unreplayable. Because it
is game-owned, an outage is a gap in the *projection*, never in the *truth*.

⚠ **Row 21's "retro-recoverable" was FALSE AS WRITTEN, and the fix is a placement rule, not a caveat.**
Revision 1 made the destruction FACT game-owned (row 22) and then put physical facts in the **sheddable**
journal (§4.6: physical facts "may shed LOUD"), with the archive itself **optional** (W10(a) offered "history
is best-effort" as a legitimate answer). So the exact facts insurance depends on were the sheddable, optional
ones: with the economy off for longer than `local_segments_max`, the claims are simply gone and row 21 is a
lie. Worse for §8.4's proof, a shed is recorded as `JournalGap{from_lsn, to_lsn, count}` — a **count**, with
no per-item attribution — and a count cannot repair a per-`ItemId` identity, so in production the
conservation identity was unverifiable whenever anything shed, while `G-WL-ITEM-CONSERVATION` stayed green
because it runs over in-memory `InspectReport`s with the journal entirely out of the loop. That is precisely
the anti-theater failure §8.4 demands RED controls against.

> **RULE WL-CONSERVED-FACT.** A fact that changes a **conserved total** or founds a **claim** is not history —
> it is STATE. Item **mints and burns**, declared **losses**, and **destruction/delivery** facts are rows in
> the **WORLDLINE STATE** record (`LossBudget::ZERO`, never shed, `Sink`-attributed), *not* entries in the
> sheddable journal. Per-block placement history, telemetry and analytics detail stay in the journal and stay
> sheddable. This is affordable precisely because mints/burns/destructions are **sparse** (§5.4 measures
> ~0.3 destructions/s against 833 block edits/s) and because §4.3b's aggregate-before-journal rule keeps a
> player-built installation ONE subject rather than thousands. Consequences: nothing in the journal is ever
> `LossBudget::ZERO` (which also removes the shared-ring back-pressure hazard §4.6 now fixes), and if any
> conservation-bearing fact is ever left in the journal then `JournalGap` **must** carry per-kind
> per-quantity attribution and §8.4's identity must be restated as an inequality with a **declared**
> unverifiable window. Gate: G-WL-ITEM-CONSERVATION gains a **mid-scenario shed cell** with a RED control.

> **RULE WL-LIEN (row 28).** **An economy object may NEVER be the container of record for an item position.**
> Escrow or collateral over *goods* is a **lien** — a claim in `vd-econ` referencing a stack whose position
> stays in a GAME container — so "the economy is absent" means "the lien is unenforceable", never "the goods
> are nowhere". Revision 1 left this open, and the hole was real: rows 17/19 put escrow and collateral wholly
> inside `vd-econ`, and commodity escrow holds **cargo**, not only money. An econ-owned container is invisible
> to the harness `InspectReport` capture the oracles audit (`crates/harness/src/oracle.rs:1-5`), so with the
> economy absent those items are either a phantom (positions < mints) or, on economy restart against a stale
> escrow view, a duplicate — and row 19's degraded note ("physical delivery still works") tacitly assumed
> goods never sit in escrow, which is not how commodity markets work. Gate cell: open an escrow over goods,
> kill the economy, assert the identity still holds and the goods are still in a game container. **This also
> settles the physical half of report D15 independently of the monetary half.**

**Row 4 corrects a latent violation in the existing report** that nobody flagged:
`scripts/economy_research_20260726.md:1411` and §7.7 (`:1620`) place **"NPC agent strategies" inside
`vd-econ`**. As written, switching the economy off deletes the NPCs — directly against LAW-E2's own words
("so players can interact with NPCs"). The fix is the existence/needs (game) vs trading-policy (injected)
split, with a *named* price-free default that is not a stub: needs-driven behaviour is fully expressible
from the physical field alone. **Prices make NPCs smarter, never alive.**

⚠ **Row 4 also carried the design's deepest defect in revision 1, and it is worth stating why the repair is
structural rather than a caveat.** Row 4 said behaviour *differs* with the economy on ("they choose by
physical scarcity, not price"), and §3.2's `weights()` was how. But NPC decisions draw physical stocks (row 3)
and, under W-5, choose haul destinations — so **a durable worldline deviation became a function of monetary
state**. Three consequences revision 1 never analysed: (1) LAW-WL-1 says a live tick's delta is
`F(t) − F(t−1)` of a closed form that has **no weights input**, so with the economy on, live ≠ dormant and the
flagship gate was **unsatisfiable** — it passed only in the economy-idle arm, i.e. the design's headline
property was coincidental, not structural; (2) it inverted §8.4's obligation, since the game's journal became
a function of the economy's history, making the physical world unreplayable from the game's own truth while
that history was declared optional and sheddable; (3) it re-opened the exact monetary→topology chain §3.4
congratulates itself for repairing in Design D, through a different arm. **The repair is correction 5 (§0):
`weights()` is deleted and the only economy influence on the world is a journaled `EconCommand`, so any
economy-caused physical change is authored by the game's own authority at a known tick and replays from the
game's journal alone.** The recorded alternative, if the user actively *wants* prices to steer NPC hauling
without a command: keep an advisory value but (i) tick-aligned and fence-stamped, (ii) **recorded in the
game's journal at the tick it was applied**, and (iii) carried by a reviewed arm with a stated
`effect_class`. That is strictly more machinery than the command channel, for less capability — which is why
it is the alternative and not the recommendation (**W17**).

**Row 14's rule generalises: never gate a movement primitive on an economy answer.** The in-repo precedent
is that `DockState.clamped` was **deleted** as a coupling precisely because a discrete authority-gating
value must not ride a lossy latest-wins carrier (`docs/design/sealed_shards.md:362`); it was re-routed
through the `Transfer` arm and the saga.

---

## 3. THE DECOUPLING MECHANISM

### 3.1 Crate topology (exact positions)

Verified current graph (`cargo metadata --no-deps`, normal + build edges): `vd-core → {}`;
`vd-devproto → {}`; `vd-wire → {core}`; `vd-sim → {core, wire}`; `vd-node → {core, sim, wire}`;
`vd-connection-plane → {core, sim, wire}`; `vd-harness → {core, wire, sim, node, client}`;
`vd-client → {core, devproto, sim, wire}`; `vd-io-prod → {core, node, sim, wire}`; `vd-bins → {…all…}`;
`vd-tests → {connection-plane, core, harness, node, sim, wire}`. Internal **dev**-only edges exist in
exactly two places: `vd-connection-plane → vd-harness` and `vd-io-prod → vd-harness`.

Target after this design:

```
vd-core            worldline kernel (Fx, TickRate, muldiv, ipow, RealmKey, Baseline, Deviation,
                   evaluate, compact, WorldEpoch, ItemId, sealed WorldFact, recipe registry)
   ↑
vd-wire            + WorldlineAdopt (and WorldlineDeviation only if not folded into BlockEdit)
   ↑
vd-sim             projections, NPC materialisation/promotion, EconomyPort (trait), NpcStrategy
                   (trait) + NeedsOnlyStrategy (impl), WorldlineTuning, run-conditions
   ↑
vd-node            custodian: StoreKey families, deviation ingest, compaction sweep, adopt push
   ↑
vd-bins            wires Option<Box<dyn EconomyPort>>; `economy` non-default feature
   ↑                                            ↖
vd-econ (NEW, optional, NON-default) ────────────┘   deps: {vd-core, vd-wire, vd-sim}
```

⚠ **`vd-io-prod` is NOT in `[workspace.dependencies]`** (`Cargo.toml:22-31`); `vd-bins` path-deps it
(`crates/bins/Cargo.toml:79`). Add `vd-econ` to `[workspace.dependencies]` rather than copying that
inconsistency.
⚠ **`crates/io-prod/Cargo.toml:18-21` asserts a graph fact that is FALSE**: it claims "vd-node depends on
vd-io-prod only as a DEV-dependency (tests)". `cargo metadata` shows `vd-node` has **zero** dev-deps on any
workspace crate. The real edge is one-way with no reverse edge of any kind — *stronger* than the comment
claims. Fix the comment in the same diff; do not reason about the graph from manifest prose.

**The `vd-econ` crate's own rule set** (clippy config is **not** inherited — eight per-crate `clippy.toml`
files exist, e.g. `crates/sim/clippy.toml:1-19`, and there is **no** workspace-root one):
- `crates/econ/clippy.toml` as a near-copy of `sim`'s (bans `SystemTime::now`/`Instant::now`/
  `thread::sleep`, default-hasher `HashMap`/`HashSet`, `UdpSocket`/`TcpStream`/`TcpListener`).
- ⚠ **`#![deny(clippy::float_arithmetic)]` as a crate-level inner attribute** — the in-tree precedent is
  `crates/io-prod/src/lib.rs:36` (`#![warn(clippy::await_holding_lock)]`). A `clippy.toml`
  `disallowed-types` entry for `f64` **silently does nothing** — `f64` is a primitive, not a path. Written
  that way it is a rule that appears enforced and is not, which is worse than no rule.
- ⚠ **BUT NOT crate-level for the WORLDLINE.** The worldline kernel lives in **`vd-core`**, which is
  float-heavy by design (`celestial.rs`, `geometry.rs`, `taxonomy.rs`). A crate-level `#![deny]` there does
  not compile. §10's G-WL-NO-SECONDS row said "plus `#![deny(clippy::float_arithmetic)]`" without noticing
  this; the correct form is a **module-level** `#[deny(clippy::float_arithmetic)]` on the
  `crates/core/src/worldline` module item, paired with the source test that no `f32`/`f64` token appears
  under that directory. Crate-level `#![deny]` applies to `vd-econ` only.

### 3.2 The port: EMIT-ONLY, plus an ACTOR channel — and why `weights()` is deleted

```rust
// crates/sim/src/io/econ.rs — object-safe, mirroring Store/RealmSpawner exactly
pub trait EconomyPort {
    /// GAME → ECONOMY. Returns (). The `-> ()` IS the LAW-E1 enforcement: there is no
    /// method whose result a game code path can wait on or branch upon.
    /// RULE WL-LIVENESS (§1.1): `&self`, bounded, an O(n) copy into a game-owned
    /// double-buffered queue with drop-and-count on overflow. The economy drains
    /// OFF-TICK. No user work runs inside `step_tick`.
    fn observe(&self, facts: &[WorldFact]);

    /// ECONOMY → GAME, the ONLY direction of influence. The economy ENQUEUES commands;
    /// it never answers a question. Also `-> ()`: the economy learns whether a command
    /// succeeded only by OBSERVING the resulting `WorldFact`, exactly as any other
    /// consumer does. Drained by the game at its own cadence from a bounded queue.
    fn drain_commands(&self, out: &mut EconCommandBuf);
}
```

**`weights(SubjectId) -> PolicyWeights` IS DELETED.** Revision 1 chose it over
`quote_hint(..) -> Option<PriceHint>` for a good reason (an `Option` invites `if let Some(p) = … else
return`, which is the branch that becomes a block) and then shipped a worse problem: a *total* value is still
a value a game decision reads, and NPC decisions author **durable deviations**, so it made the physical
worldline a function of monetary state (§2 row 4). It was also **unreplayable** (the fact stream is teed
off-tick and physical facts may shed, so `weights()` returns a value whose history depends on I/O timing) and
it had **no possible data source** (a price needs cross-shard economy state; §7.2 #11 concedes monetary
delivery has no carrier, so `weights()` could only ever read shard-local facts, which is not a price). Its
only defence was a written review rule — exactly the status E1-c claimed to have eliminated.

**`EconCommand` is the replacement, and it is strictly more capable.** It is an ordinary game command on the
same path a player's action takes:

```rust
/// vd-core: a REQUEST, never an instruction. Fenced, refusable, idempotent.
pub struct EconCommand {
    pub id: EconCommandId,       // minted from a durable monotone high-water (never reused)
    pub subject: SubjectId,
    pub verb: EconVerb,          // MoveStack | GrantHull | DeliverGoods | SetLien | ClearLien | …
    pub issued_epoch: WorldEpoch,
}
```
Five properties, each load-bearing:
1. **The game may REFUSE it**, with a typed reason counted as a fault, and the refusal is itself a
   `WorldFact` the economy observes. No command is ever a precondition of anything.
2. **It is applied by the GAME's own authority** at a known tick, so its effect is an ordinary deviation in
   the game's own journal — which is what keeps §8.4's "the projection is a pure function of the game's
   journal" true, and what makes §1.1's P2 attribution property checkable.
3. **It is idempotent** (keyed on `EconCommandId`) and re-drivable, so the economy may re-issue freely.
4. **Absence of the economy means the command was never issued** — never that the game blocked, and never
   that world state is unexplainable.
5. **It answers report §7.18** ("name at least one designed consumer … 'the economy changes nothing in the
   world' must be a DECISION, not an omission"). Revision 1 answered NO *by accident*: with `observe` +
   `weights` only, every monetary row 15–21 was **terminal**, so a filled trade could never move goods, an
   insurance payout could never yield a hull, and a courier's forfeited collateral could never move cargo.
   That is the accounting-fiction-bolted-beside-the-game failure §7.18 exists to prevent.

**Review rule, unchanged in spirit and now actually true: LAW-E1 is checkable by reading this trait alone** —
both methods return `()`, so no method's absence can prevent a game action. If a third method ever returns a
value, the trait is wrong.

Emit sites are **UNCONDITIONAL** and absorbed by a null sink. The game's instruction path is therefore
identical with and without an economy, which is what makes §3.5's byte-identity assertion *meaningful*
rather than coincidental.

**Where NPC POLICY sits, restated under the deletion.** NPC existence, needs and pose are game state (§2
rows 1–3). Trading *policy* is an injected `NpcStrategy` whose **default is `NeedsOnlyStrategy`** — price-free,
driven by the physical field alone. With the economy present, a smarter strategy may be injected, but it may
influence the world **only** by issuing `EconCommand`s, never by biasing a score the deviation-authoring path
reads. That keeps §2 row 4's declared default honest and keeps G-WL-CLOSEDFORM-EQ-ACCUM satisfiable.

### 3.3 The switch: BOTH, and the feature goes on `vd-bins`

- **Runtime**: `Option<Box<dyn EconomyPort + Send + Sync>>`; `None` is the absent path and the default.
  `NullEcon` and `MemEcon` live in **`crates/sim/src/io/mem.rs`** next to `MemStore`/`MemSpawner`, so the
  present-but-idle arm needs no extra crate and no extra build (§1.1).
- **Link time**: `economy = ["dep:vd-econ"]`, NON-default, on **`vd-bins`** (the shipped binaries) **AND on
  `vd-tests`** (where the real injection into the accumulated suite happens) — the `dev-control` doctrine
  verbatim (`crates/bins/Cargo.toml:53-57`), plus **two** cells in `just lint-combos` (`justfile:63-75`, which
  exists *because* audit FG-1 found a violation reachable only in a non-default combo: "clippy needs no
  GPU, it only type-checks"). ⚠ **A `vd-bins`-only feature could never change what the SUITE compiles**
  (`vd-tests` does not depend on `vd-bins`, `tests/Cargo.toml:10-17`; and `vd-bins` already has
  `default = []`, `crates/bins/Cargo.toml:58`), which silently collapsed §3.5's three arms into two.
- ⚠ **NOT a feature on `vd-sim` or `vd-core`.** `#[cfg]` inside a Tier-A crate multiplies the compiled
  surface, and **HR5 counts regions per monomorphization AND per test binary** — the feature-off build
  would owe its own full coverage pass. Non-obvious, with real cost.
- **Operationally undeclarable-by-drift**: `enforce_cloud_preflight` is a no-op in `Profile::DevTest` and
  fails LOUD in `Profile::Cloud` (`crates/io-prod/src/boot.rs:410-470`, with an always-compiled test
  covering *both* the reject and the accept arm, `crates/bins/tests/boot_guard.rs:1`). It gains one
  assertion: a Cloud deployment must set `VD_ECONOMY=on|off` **explicitly**.

### 3.4 LAW-WL-7 — realm topology is economy-blind and life-blind (Design B's graft, and its repair)

**THE RULE — re-scoped, because revision 1's phrasing let the worst violation through.** Revision 1 wrote:
*no economy value, no life-tier value, and no economy-derived boolean may EVER enter `desired_alive`
(`crates/sim/src/rlm.rs:455-463`) or `teardown_ready` (`crates/sim/src/rlm.rs:471-488`).* That is enforceable
by inspecting one pure signature — the reconciler takes
`(ledger, dir, liveness_dead, launch_live, tuning, now, quiesced_until)` and has no store or `World` access
(`crates/sim/src/rlm.rs:585`) — and it is **insufficient**, because a life value does not have to reach those
two predicates *directly*. It can arrive as an **input to them**:

> **LAW-WL-7 (corrected).** No economy value and no life-tier value may enter `desired_alive`,
> `teardown_ready`, **the `aoi_decide` OCCUPANT SET, or the `AoiMembership` map**. Lifecycle sees SESSIONS and
> geometry; it never sees NPCs, stocks, machines, prices, or obligations.

`desired_alive` has exactly two arms, verified:
```rust
demanded_recently(cell, now, tuning.demand_ttl_ticks)
  | (running_live & !empty_confirmed(cell, now, tuning.empty_grace_ticks))
```

⚠ **THE VIOLATION THE OLD PHRASING MISSED, AND IT IS A DESIGN-KILLER, NOT A NIT.** §6.3 makes materialised
NPCs "ordinary occupants". Verified: `aoi_decide` builds
```rust
let occupants: Vec<(DVec3, DVec3)> = dots.0.values().filter(|d| d.authority.simulates())…
    .chain(owned.0.values().filter(|t| t.status.is_held())…).collect();
if occupants.is_empty() { push_demand(… DemandVerb::Empty …); return; }
```
(`crates/sim/src/stub.rs:4397-4421`) and then feeds that same set to `aoi_min_dist` for every direct child,
emitting `SpinUp`/`KeepAlive` (`:4425-4440`). Both NPC classes land in that fold — a durable named NPC is an
owned simulated dot, an ambient crowd NPC is the `.chain(...)` held-transient arm. So **two** things happen,
neither through arm A:
1. **One materialised NPC makes `empty_confirmed` permanently FALSE** (`rlm.rs:439-448`) ⇒ `desired_alive`
   arm B (`running_live & !empty_confirmed`) holds forever ⇒ `teardown_ready` never fires (`:471-488`) ⇒
   **THE REALM IS NEVER REAPED.** LAW-E2's premise ("most of the galaxy off") collapses in exactly the realms
   that have content, and `G-WL-LIFECYCLE-BLIND` is unsatisfiable by construction.
2. **An NPC's POSITION emits realm-lifecycle demands**, so the life tier causes *processes to launch* — the
   same monetary/life → process-topology chain this section disqualifies Design D for, arriving through
   `aoi_decide` instead of `pin_realms`.

**THE MECHANISM, which revision 1 costed as "`G-EMPTY-SESSION-ONLY` + the doc line, ~20 lines" and which is
actually a real code edit with its own coverage:**

> **ONE predicate, used in BOTH places.** Add `fn is_session_occupant(&self) -> bool` to the dot/held-transient
> surface — true for a player session's own entities and anything a session carries, **false for every
> NPC-kind entity**. `aoi_decide`'s occupant fold filters on it, so (a) an NPC-only realm still self-reports
> `DemandVerb::Empty` and IS reaped, and (b) no NPC position ever emits a `SpinUp`/`KeepAlive` demand. NPCs
> remain full occupants for **physics, containment, collision and the observer feed** — the split is between
> "occupant for simulation" and "occupant for LIFECYCLE", and it exists in exactly one predicate so it cannot
> drift.

It **must land before the first NPC entity** (§11.2), with `G-EMPTY-SESSION-ONLY` gaining a RED control (make
one NPC a session occupant, assert the realm stops being reaped) and **`G-WL-LIFECYCLE-BLIND` moving from W-5
to W-3**, because that is the slice where the violation would otherwise land.

⚠ **This OVERRULES `scripts/economy_research_20260726.md` §6.4 requirement 2**, which proposes making
"has resting orders / holds escrow" an AoI/hysteresis input to keep a market realm warm. That would make
**which realms are alive** a function of economic state — the deepest available LAW-E1 violation, and the
report does not flag it. It is a **defect**. The rule-respecting alternative: *the state must be correct
across an arbitrary reap*, which replay-from-base already gives for free.

⚠ **THE FATAL FLAW IN DESIGN D, REPAIRED HERE.** Design D's placement answer is `pin_realms` folding a
synthetic `KeepAlive` into the ledger — i.e. **a write into `desired_alive` arm A**. Worse, D's own
cross-shard item 3 makes it dynamic ("the CAT keeping a realm warm because a materialised flow is about to
enter it → existing `RealmDemand` with `KeepAlive`… the life tier can now cause processes to launch"),
which chains to `CAT-M biases lane scoring → which flows are dispatched → which realms are warmed` — an
indirect **monetary → process-topology** path that D never flags. Repairs, both mandatory if the pin is
adopted (**W6**):
1. **The pin is a static, economy-independent DEPLOYMENT CONSTANT** — a `pinned: BTreeSet<RealmPath>`
   config field validated at boot, never a function of any runtime value. It is charged to the GAME.
2. **The flow-driven `KeepAlive` is DELETED.** A coarse flow that needs a realm warm must instead be
   scheduled so that it does not (its arrival is evaluated by the pinned host, and materialisation happens
   only where a realm is already live for player reasons).

Both are proved by **G-WL-LIFECYCLE-BLIND** (§10): toggle the economy and the life tier on/off mid-scenario
and assert the RLM `LifecycleAction` trace is **BIT-IDENTICAL**. This is the only machine proof in the
field that process topology is economy-blind, and it is the gate that would have caught the report's §6.4
proposal.

**Also rejected (Design B's other flaw):** an **ack-gated Kill**. Design B has `LifecycleAction::Checkpoint`
inside the sole kill authority's action set and "the reconciler must not kill until the ack lands or
`handback_deadline_ticks` expires" — so a wedged derived subsystem delays teardown. Rule-respecting
alternative, adopted: **no handback ack, ever.** An un-reported gap means *re-adopt from the custodian's
last durable value*, with the bounded loss posted as a **declared** physical loss (§9.4). The reconciler's
existing two-phase `drive_drain` (`crates/sim/src/rlm.rs:555-572`) is used **only** as the window inside
which the shard *voluntarily* emits its final deviation batch; nothing gates the kill on it.

### 3.5 The degraded-mode contract, and the CI gate

Every genuinely coupled edge (rows 4, 14, 21 of §2 — that is **three** edges in the whole design) carries a
**declared `DegradedMode`** with its own gate cell, copying `G-COUPLING-DEGRADED`'s structure
(`docs/design/sealed_shards.md:350`): kill source, kill sink, partition the link, drive the normal path, and
assert the declared mode in each cell.

**THE CI GATE — `G-ECON-ABSENT`** (name it once, use it everywhere): run the **entire** accumulated
`vd-tests` suite three ways — economy **not compiled**, port **`None`**, port **present-but-idle** — and
assert **byte-identical** scenario results. The worldline is PRESENT in all three arms; that is the point.
Plus one `lint-combos` cell.

⚠ **Price it before approving it.** `just gate` is already **16 steps** (`justfile:215`: `fmt-check lint
lint-combos test client-load orch-crash spike2a spike3a rlm-soak render-smoke render-boxes-smoke
render-crossing-smoke node-per-realm-walk rlm-proc-spawn rlm-kill9 coverage`). A second full suite run
roughly **doubles the `test` step's wall clock**. Options: (a) run all three arms in `gate` (highest
confidence, highest cost); (b) run *two* arms in `gate` (not-compiled vs present-but-idle) and the third in
a release-only recipe; (c) run the byte-identity arm on a **named subset** of scenarios that touch every
emit site, with an anti-vacuity assertion that the subset covers all `WorldFactKind`s. **Recommend (c)**,
with the full three-arm run as a release-only `just econ-absent release` recipe wired into a pre-release
gate. This is **W7**.

---

## 4. THE LAZY-FIELD SUBSTRATE — one mechanism, three hosts

### 4.1 The kernel (HR3: one tooling, stated as types)

One module tree, `crates/core/src/worldline/`, in `vd-core`.

```rust
// ── the ONLY entry point ──────────────────────────────────────────────────────
pub fn evaluate(
    base: &Baseline,            // scanned by the caller (exact-key scan == pseudo-get)
    devs: &[DeviationRecord],   // scanned by the caller, ALREADY ascending
    at:   UniverseTick,
    tuning: &WorldlineTuning,
) -> Evaluated;                 // subjects + cohorts + digest

pub fn compact(base: &Baseline, devs: &[DeviationRecord], cut: UniverseTick) -> Baseline;
```

Shaped exactly like `reconcile` (`crates/sim/src/rlm.rs:585`): pure, `World`-free, I/O-free, immutable-in,
run-twice-deterministic, caller supplies already-scanned rows.

**HR5 shape, and it is why this kernel is cheap at 100% region+branch:** **no generics anywhere in the
arithmetic core.** Every function is monomorphic over `i128`/`u64`/`u128`, so the per-monomorphization
region hazard is absent by construction. The only real branches are `match`es on the two policy enums, each
in ONE monomorphic helper covered once — the `place_child` idiom (`crates/sim/src/stub.rs:701-710`, whose
doc says exactly this: "the `match` is covered once here …, NOT per generic monomorphization (HR5)").
Everything else is `min`/`max`/`div_ceil`/`saturating_*` — branchless integer intrinsics, free at 100%
branch coverage.

### 4.2 Arithmetic primitives

```rust
/// Q32 fixed-point minor units. STATE is exact in Q32; only OBSERVATION floors, and the
/// floored remainder STAYS IN THE STATE.
pub struct Fx(pub i128);            // value = Fx.0 / 2^32
pub const FX_ONE: i128 = 1 << 32;

/// A per-TICK rational rate: `num` minor units per `den` TICKS. NEVER per second.
pub struct TickRate { pub num: u64, pub den: u64 }

/// ⚠ NARROWED FROM `(u128, u128, u128)`. With u128 arguments `a*b` needs 256 bits, so the
/// function as revision 1 declared it was UNIMPLEMENTABLE as specified: it would need a
/// multi-word 256-bit intermediate (5–10x the cost §5.4 assumed) or a `checked_mul` that
/// REFUSES legitimate inputs — and the design said neither, leaving the API contract
/// undefined. These signatures make the intermediate provably fit u128 and the divide a
/// single 128/64.
fn muldiv_floor(a: u128, b: u64, c: u64) -> u128;   // floor(a·b/c) — a producer never OVER-produces
fn muldiv_ceil (a: u128, b: u64, c: u64) -> u128;   // ceil        — a THRESHOLD is never reached early
/// Signed variants with an EXPLICIT sign contract (§4.5 passed an i128 difference into a
/// u128 parameter — a silent contract violation).
fn muldiv_floor_i(a: i128, b: u64, c: u64) -> i128;
fn muldiv_ceil_i (a: i128, b: u64, c: u64) -> i128;
fn ipow_muldiv(gap: i128, num: u64, den: u64, m: u64) -> i128;  // gap·(num/den)^m, O(log m)
```

The **floor/ceil asymmetry is deliberate and asserted**: outputs round down, thresholds round up, so no
combination of roundings can manufacture material.

**THE OVERFLOW CAPS ARE DERIVED, NOT NAMED.** ⚠ Revision 1's §9.2 said only "every product is `checked_mul`
under `WorldlineTuning` caps (`max_rate_num`, `max_stock_minor`) validated fail-loud at registry load" and
gave **no values and no derivation** — i.e. magic numbers by omission, against the standing rule. The caps
follow from the u128 headroom and are written out here so `validate()` can name the inequality each one
enforces (the `AoiConfig::for_velocity_safe` discipline of deriving a bound rather than asserting one):

| Cap | Inequality it enforces | Why |
|---|---|---|
| `max_rate_num` | `max_rate_num · 2³² · u64::MAX ≤ u128::MAX` ⇒ `max_rate_num ≤ 2³²` | §4.3 step 1 computes `muldiv_floor(rn << 32, Δ, rd)`; with `Δ` up to `u64::MAX` this is the binding constraint |
| `max_stock_minor` | `max_stock_minor · max_den ≤ u128::MAX` with `max_den ≤ 2³²` ⇒ `max_stock_minor ≤ 2⁹⁶` (Fx), i.e. `2⁶⁴` whole units | §4.3 step 2's `S.0 · rd` and step 5's `q_full · rd` |
| `max_den` | `max_den ≤ 2³²` | both of the above; also keeps `ipow_muldiv`'s Q64 intermediate inside i128 |
| `max_season_entries` (`W`) | `W ≤ 16` | §4.3a's per-period prefix sum is a fixed-size loop, so `W` is what keeps it O(1) |

G-WL-OVERFLOW asserts the **derived** caps, and its `validate()` test names the inequality per cap. Where
saturation is reached it is the **semantically correct** answer (it dried out long ago), commented as such and
proptested at `Δ = u64::MAX`.

**`ipow_muldiv` replaces every transcendental.** A logistic `P(t) = K/(1+((K−P₀)/P₀)e^{−rΔ})` needs `exp`,
a libm call whose **cross-host bit-equality is UNPROVEN in this repo** (`crates/core/src/celestial.rs`
documents the SPIKE-6a owe). Instead the integer geometric relaxation IS the law:
`P(Δ) = K − ipow_muldiv(K − P₀, gn, gd, Δ / growth_quantum_ticks)`. The in-tree precedent for pinning a
deterministic integer procedure instead of an analytic transcendental is `KEPLER_FIXED_ITERS` as a
compile-time constant so every host runs identical steps (`crates/core/src/celestial.rs:104-110`).

⚠ **A real finding on the existing RNG**: `SplitMix64::next_f64` (`crates/core/src/rng.rs:31-37`) and
`chance` (`:39-42`) are **float**-based, and `crates/core/src/taxonomy.rs:11` says its samplers are too. The
worldline may use **only** `next_u64` and `range_u64` (`crates/core/src/rng.rs:44-50`), and needs
**integer-threshold twins** of the categorical samplers (compare a `u64` draw against `u64` cumulative
thresholds). ~40 lines, and it must exist before the first field draw.

### 4.3 THE PRODUCTION / DEPLETION INTEGRAL — the actual math with its piecewise breakpoints

A machine subject at base tick `t₀`, `Δ = t − t₀`; input rate `r = (rn, rd)` minor units per tick; yield
`y = (yn, yd)` output per input; input stock `S: Fx`; hopper level `h: Fx`; hopper cap `B: Fx`.

**Step 1 — uncapped cumulative input demand (Fx).** One `u128` muldiv:
```
D(Δ) = muldiv_floor(rn << 32, Δ, rd)
```

**Step 2 — the DEPLETION breakpoint.** `D(Δ) ≥ S.0` ⟺ `(rn·2³²·Δ)/rd ≥ S.0` ⟺
```
Δ_dry = (S.0 as u128 · rd as u128).div_ceil((rn as u128) << 32)
```
Exact integer, one `div_ceil`. **This is the piecewise breakpoint, in closed form.**

**Step 3 — the SATURATION (hopper-full) breakpoint.**
```
q_full = muldiv_ceil(B.0 − h.0, yd, yn)                    // input units needed to fill the hopper
Δ_full = (q_full · rd as u128).div_ceil((rn as u128) << 32)
```

**Step 4 — the stop tick.** Under `HopperPolicy::Stall` (a full refinery stops eating ore — the physical
answer):
```
Δ_stop = min(Δ_dry, Δ_full);   Δ_eff = min(Δ, Δ_stop)
```
**Why using the UNCAPPED curve for `Δ_full` is EXACT and not an approximation:** uncapped output ≥ capped
output, so `Δ_full_uncapped ≤ Δ_full_true`. If `Δ_dry < Δ_full_uncapped`, the deposit dries first and
`Δ_dry` wins (correct). If `Δ_full_uncapped ≤ Δ_dry`, then no capping occurred anywhere on `[0, Δ_full]`,
so uncapped == true on that interval. Hence `min(Δ_dry, Δ_full_uncapped)` is exact. ⚠ **This step is
subtle and a wrong `min` produces a plausible-but-wrong economy silently — it is the single strongest
argument for the mandatory differential reference (§10, G-WL-DIFFERENTIAL).**

**Step 5 — the results.**
```
C  = Fx(min(D(Δ_eff), S.0))          // consumed input
S' = Fx(S.0 − C.0)                   // remaining stock
G  = muldiv_floor(C.0, yn, yd)       // gross output
h' = Fx(min(h.0 + G, B.0))           // new hopper
```
Under `HopperPolicy::Spoil(SinkId)` instead: `Δ_stop = Δ_dry`, and `spoil = (h.0 + G) − h'.0` is posted to
a **declared** sink so conservation stays exact.

**Total: 4 muldivs, 3 `div_ceil`s, 3 `min`s, O(1) IN ELAPSED TICKS, fully integer, two explicit breakpoints.**

⚠ **THE COMPLEXITY CLAIM, STATED PRECISELY, because revision 1 wrote "O(1)" in two places where it is not
true.** Each deviation **ends an interval**, so the honest rule is: **`evaluate` is `O(#deviations +
#subjects)` closed-form segments, and O(1) in ELAPSED TICKS *within* a segment.** §0's headline ("O(1) in
elapsed ticks") is precise; §4.3's "still O(1)" for `RespawnAfter` was not, and neither was §4.8's implicit
claim about the Tier-B path. This matters because the composition is where a wrong breakpoint hides: under
`HopperPolicy::Stall` a machine stops consuming when the hopper fills, so a *later hopper drain* (a deviation)
restarts consumption and **shifts `Δ_dry`**; combined with `RespawnAfter`, the breakpoint set is a function of
the deviation *sequence*, not of `Δ` alone. That is perfectly consistent with LAW-WL-3's segment structure —
but it must be stated, and G-WL-DIFFERENTIAL's cross-product must be **policy × deviation-sequence**, not
policy alone (§10).

**Regeneration policy** (this is **W3**, and it sharpens report **D16**):
- `Finite` — never returns. Exact, one breakpoint. Free.
- `RespawnAfter(ticks)` — a **step**: stock returns to `S_full` at `t_deplete + respawn_ticks`, where
  `t_deplete = t₀ + Δ_dry`. ⚠ **Revision 1 stated the evaluation as "`n_cycles = Δ / cycle_len` plus a
  remainder term" UNCONDITIONALLY, and that is wrong in the regime of its own worked example.** Two regimes,
  and they need an explicit case split:
  - **DRY-FIRST (`Δ_dry < Δ_full`)**: the deposit exhausts, `t_deplete` occurs, and the cycle is real.
    `cycle_len = Δ_dry + respawn_ticks`; `n_cycles = Δ / cycle_len` plus a remainder. Exact, O(1) in Δ.
  - **FULL-FIRST (`Δ_full ≤ Δ_dry`)**, which is §5.4's example (`Δ_full = 40 M < Δ_dry = 160 M`): under
    `HopperPolicy::Stall` the machine stalls, the deposit **never** depletes, `t_deplete` never occurs, and
    **`cycle_len` is UNDEFINED — no respawn cycle happens at all.** Revision 1's formula would divide by a
    meaningless quantity.
  - **MIXED** (a player drains the hopper every few weeks): the cycle *phase* depends on the drain ticks, so
    the composed evaluation is `O(#deviations)` segments — see the complexity rule above.
  `RegenPolicy::RespawnAfter` is also the recommended **DEFAULT** rather than an option (§1.2's
  "no positive dynamics" admission), because it is the only mechanism in the substrate that restores
  something without a player or an agent.
- `FieldReplenish(rate)` — continuous regeneration RACING continuous production. **There is no general
  closed form** (the stock solves a coupled recurrence). Honest bound: restrict regeneration slower than
  the coarsest cadence and evaluate on a common quantum grid, making it `O(min(Δ/regen_quantum,
  max_quanta))` — bounded, but **no longer O(1)**, with a `max_quanta` cliff.

### 4.3a THE SEASONAL DRIVER — the mandatory non-saturating term (LAW-WL-6)

⚠ **This subsection did not exist in revision 1, and without it LAW-E2 expires after roughly a month.**
Revision 1's laws are all monotone toward a bound, so each realm has a computable **believability horizon**
after which `evaluate(t)` is **constant forever** — including in §5.4's own worked example:
`Δ_stop = min(Δ_dry, Δ_full) = 40 000 000` ticks = **day 23.15**. A 6-month dormancy and a 5-year dormancy
therefore returned *bit-identical* state: the refinery filled its hopper on day 23 and has been idle ever
since, the deposit frozen at 75 %, the population sitting at exactly `K`. The 30-day
`G-WL-DORMANT-MOVES` gate passed identically at 30, 300 and 3,000 days, so **the gate written to prove "not
frozen" was passed by a world that freezes on day 23.** The machine also faced a hard trade with no third
option: a small hopper gives one interesting 23-day story then freezes, and pushing the horizon to a year
needs a ~16× larger hopper, which deletes the "the refinery filled and stopped" story §5.6 offers as the
believability proof.

**THE MECHANISM: a seed-derived integer SEASON TABLE, and nothing else.**

```rust
/// W ≤ max_season_entries (16). Seed-derived per subject at generation; a DEVIATION may
/// rebase the table, exactly like any other field.
pub struct Season { pub w: u8, pub num: [u32; MAX_SEASON_ENTRIES] }   // multipliers in bp
```
The effective rate in season `j` is `r_j = r · num[j] / 10_000`, with `j = (t / season_quantum_ticks) mod W`.
`season_quantum_ticks` is a named `WorldlineTuning` field derived from the cadence quantum (§4.7).

**Why this is still exact, integer, and O(1) in Δ:**
- **Cumulative input demand** over `[0, Δ]` = `(Δ / period) · Σⱼ r_j·q + prefix(Δ mod period)`, where
  `period = W · season_quantum_ticks` and `prefix` is a **fixed W-step** loop. Two muldivs plus ≤W adds.
- **The depletion breakpoint** `Δ_dry`: integer-divide `S` by the per-period integral to get whole periods,
  then a fixed W-step scan of the prefix inside the final period. Exact; O(W) = O(1).
- **The saturation breakpoint** `Δ_full`: identical shape.
- **NO FLOATS, NO SECONDS, NO RNG STREAM.** The table is data; the index is an integer divide and modulo.
  ⚠ **REJECTED alternative, and the rejection is on the merits:** driving the period from **orbital phase**
  looks attractive (the celestial layer already has closed-form phase) but that phase is **f64/libm** whose
  cross-host bit-equality is ungated (SPIKE-6a, `crates/core/src/celestial.rs:11-18`), so it would violate
  §9.1's integer-generator-boundary rule at the exact seam where durable authority is decided. A seed-derived
  integer table buys the same non-monotone dynamics with none of that exposure.

**Why it fixes the freeze rather than papering over it:** the rate is non-monotone, so a machine that stalled
in a low season **resumes** in a high one; a deposit keeps depleting until it genuinely dries; and the
composed `evaluate(t)` keeps changing for as long as any subject has stock. The table is deliberately
**bidirectional** (multipliers both above and below 10 000 bp) so that dormant regions can get *better* as
well as worse — §1.2's "no positive dynamics" admission.

**For the COHORT the target itself moves**, which is what removes the fixed point (§6.2): `K(t) = K_base ·
num_K[(t / season_quantum_ticks) mod W] / 10_000`. Relaxation toward a **periodic piecewise-constant** target
still has a closed form — the per-quantum map `P_{k+1} = K_k + r(P_k − K_k)` is affine, so over one full
period of W quanta `P_{k+W} = r^W·P_k + c` with `c = Σⱼ r^{W−1−j}(1−r)K_j`, and after `n` full periods
```
P = ipow_muldiv(P₀, rⁿᵂ) + muldiv(c, 1 − rⁿᵂ, 1 − r^W)
```
— two `ipow_muldiv` calls, one muldiv with a precomputed `(1 − r^W)` denominator, then a fixed W-step partial
period. **O(1) in Δ, integer, exact to the declared fixed-point width.** The fixed point is now a periodic
**orbit** rather than a point, so `P(t)` keeps moving forever and the "every region with the same `K` sits at
exactly `K`" degeneracy (§6.2) is gone.

**Gates:** `G-WL-DORMANT-MOVES` becomes **multi-horizon** (assert a nonzero delta in the LAST interval at
30 d, 180 d **and 2 years**), and `G-WL-DIFFERENTIAL`'s cross-product gains the seasonal arm with an
anti-vacuity assertion that the fixture crossed at least one season boundary, one saturation breakpoint and
one dry breakpoint in the same run. **The design as revision 1 wrote it FAILS the multi-horizon gate, which
is exactly why that is the gate that matters.**

### 4.3b THE COHORT-CONSUMPTION INTEGRAL, and the AGGREGATE-BEFORE-JOURNAL rule

⚠ **Two gaps revision 1 left in the one place LAW-E2 needs most.**

**(a) The integral the design never derived.** §4.3 derives only the **constant-rate** machine, but §2 row 3
makes "NPC needs (physical stock draws)" a first-class physical subsystem and §6.4 makes population the
demand field — and population is a **geometric relaxation**. Consumption proportional to `P(Δ)` is therefore a
**time-varying** rate: its cumulative draw is a geometric sum, not `rn·Δ/rd`, and its depletion breakpoint is
not a `div_ceil`. "A town ate its granary while you were away" — the most believable dormant consequence there
is — had no arithmetic anywhere in the document. Two admissible answers, and the design picks the first:

1. **RECOMMENDED — exact geometric sum with a searched breakpoint.** Per growth quantum `k`, the draw is
   `d_k = muldiv_floor(P_k, cn, cd)`. Over `m` quanta the cumulative draw is a geometric sum in `r = gn/gd`
   plus a linear term in `K`, both closed-form:
   `D(m) = muldiv(K·m, cn, cd) − muldiv(gap₀ · (1 − rᵐ)/(1 − r), cn, cd)`, i.e. **two `ipow_muldiv` calls and
   two muldivs, O(1) in Δ**. The depletion breakpoint (`D(m) ≥ S`) is monotone in `m`, so it is a **binary
   search over `m ≤ Δ/growth_quantum`** — `O(log Δ)`, integer, exact, and bounded by the same
   `max_catchup_rounds`-shaped cap. State the `O(log Δ)` honestly: this is the one closed form in the design
   that is not O(1) in Δ.
2. Recorded alternative: hold consumption **piecewise-constant per growth quantum** (draw at the
   quantum-start population), which is O(1) with a declared per-quantum error bound of `|ΔP| · cn/cd`.

Either way it enters **G-WL-DIFFERENTIAL** as its own cross-product arm — it is the case most likely to be
silently wrong, and it composes with §4.3a's moving `K`.

**(b) AGGREGATE BEFORE JOURNAL — the registry's missing second criterion.** §11.2's criterion (LAW-WL-5':
*seed-derivable behaviour is never journaled*) applied honestly **forces the explosion it is meant to
prevent**: a player-placed refinery, conveyor, thruster, port module or reactor is by definition **not**
seed-derivable, so each becomes a `SubjectState`. The standing end-goal is stations and ships built FROM
BLOCKS, so a mature P8 player station plausibly carries 10²–10⁴ functional blocks — against a wake budget that
allows a few hundred subjects, a wire budget bounded by `PLAYER_DEF.max_state_bytes = 4096`
(`crates/core/src/entity_kind.rs:204`) / `SHIP_DEF` and `NAMED_CONSTRUCTION_DEF` at 8192
(`:213`, `:222`), and a baseline row size that drives the custodian's memory bill (§5.4). The declared
behaviour on exceeding it is a typed refusal — i.e. **"you cannot place another machine in this realm"**, a
hard gameplay wall arriving at P8 in a game whose premise is building.

> **RULE WL-AGGREGATE.** A player-built installation enters the worldline as **ONE**
> `SubjectKind::Construction` (or `Machine`) whose capacity, rate and condition are the **integer SUM over its
> constituent functional blocks**, recomputed by the live shard from the block graph. The **block graph stays
> Category-C shard state the worldline never sees.** Subjects are therefore proportional to
> **INSTALLATIONS** (tens), never to **BLOCKS** (thousands).

This is also what makes §4.3's "a machine draws from a STOCK" world rule survive P6 conveyors, and it is the
rule that keeps `max_subjects_per_realm` derivable from the measured wake and wire budgets rather than from
hope. Forward-check gate: build a station with 10⁴ functional blocks and assert the realm's subject count and
baseline bytes stay inside budget (§10, G-WL-AGGREGATE-BOUND).

**(c) The player-built machine's dormant RATE — the question §4.3's static `TickRate` could not answer.**
A refinery's throughput is emergent from layout, power and (at P9) signals; report §7.12 names this the
biggest drift risk against the full end-goal. Revision 1 had no rule, leaving two bad implicit answers (every
block edit is a rate-changing deviation, contradicting §11.2; or a dormant machine runs at an undefined stale
rate).

> **RULE WL-SETTLED-RATE.** A machine's dormant rate is its **LAST SETTLED rate** as of its last deviation.
> Any signal- or layout-driven variability is **FROZEN at the last settle**, and the settle is emitted at the
> **installation's** commit site (per RULE WL-AGGREGATE), never per block. This is a **declared
> approximation**, stated here rather than discovered at P9.
>
> **Dormant failure semantics**, which also needed deciding: a machine whose input is itself a worldline stock
> **stalls at its own `Δ_dry`** (the algebra already supports it); a machine whose input is a **live signal**
> (a shadowed power block, a thruster-driven mobile refinery) is **frozen at its last settled rate** and is
> named as such — it does not silently fail and it does not silently run forever.

**Multi-stage chains** (refinery → factory). Composing two piecewise-linear integer curves is a
breakpoint-set merge: depth `d` gives `O(Σ Kᵢ)` breakpoints, evaluated by binary search + one muldiv,
implementable as a fixed-capacity `PwlCurve { segs: [(Δ_start, slope_num, slope_den, v_start); MAX_SEGS] }`.
**v1 does not need it, via a WORLD RULE rather than an approximation:** **a machine draws from a STOCK,
never directly from another machine's live output.** An intermediate stock is always materialised, and a
stock changes only at event boundaries — so every machine is independent given its input stock and
composition collapses entirely. This is not a shortcut: it makes **hauling mandatory**, and hauling is the
arbitrage engine (report §4.11 action #6). Direct machine-to-machine piping becomes possible when P6
conveyor blocks land; at that point `PwlCurve` composition is the designed answer, ledgered (D-70) rather
than discovered.

### 4.4 The sparse event log — absolute rebase, O(N), no ordering hazard

**LAW-WL-2 (absolute rebase, fence-ordered).** *Every durable deviation is an ABSOLUTE re-basing of one
subject's initial condition, never a delta; and no durable counter is ever incremented — every aggregate is
DERIVED from the absolute row set.*

```rust
pub struct DeviationRecord {
    pub realm_fence: Fence,   // ⚠ NEW in revision 2 — see the dupe below
    pub tick: UniverseTick,
    pub seq: u64,             // minted by the CUSTODIAN, per (realm, fence)
    pub epoch: WorldEpoch,
    pub dev: Deviation,
}

pub enum Deviation {
    RebaseSubject(SubjectState),         // "this deposit's stock IS S at tick t"
    RemoveSubject(SubjectId),
    PromoteNpc { id: SubjectId, state: SubjectState },
    SetCohort { cohort: u16, k: Fx, p0: Fx },
}
```
The fold is **last-wins per `(subject, field)` keyed by `(realm_fence, tick, seq)` max, FENCE DOMINANT** — a
commutative, idempotent monoid. An arbitrarily reordered, duplicated, or lost-then-redelivered stream yields
**byte-identical** state. This is `record_demand`'s proven max/latch discipline
(`crates/sim/src/rlm.rs:266-282`, whose doc states the property: "keying on `max`-tracked ticks, so the result
is ORDER-INDEPENDENT"). It is not academic: the RLM crash-replay proptest caught a **real** order-sensitivity
bug (the `empty_confirmed` streak) that unit tests masked.

⚠ **THE FENCE TERM IS NEW, AND ITS ABSENCE WAS A MATERIAL-MINTING DUPE.** Revision 1's key was `(tick, seq)`
with **no fence or incarnation term**, so a stale shard incarnation whose deviations carry a LOWER
`realm_fence` but a LATER tick **WON** over its live successor. That is not exotic — it is the default
behaviour of the durability class §7.2 assigns to the arm (`ReDriven` = "re-asserted every tick it holds",
`crates/wire/src/intershard.rs:296-306`). The concrete sequence:
1. `t=1000`, shard **A** (fence `f1`) rebases deposit `D` to 900 after a player mines 100 ore. Durable, acked,
   the ore is in the player's blob.
2. `t=1005`, a transient partition makes the orchestrator's liveness latch mark A's node dead ⇒
   `zombie(head, dead)` (`crates/sim/src/rlm.rs:404-410`) ⇒ **ForceReap** (`:611-622`). **A's PROCESS is still
   alive and still re-asserting.**
3. `t=1010`, `SpinUp` mints incarnation **B** (fence `f2 > f1`), which adopts `D = 900`.
4. `t=1015`, B's player mines 100 more ⇒ `RebaseSubject(D, 800) @ (1015, 7)`.
5. `t=1020`, A's periodic field refresh re-asserts `RebaseSubject(D, 900) @ (1020, 8)` — **later tick, stale
   fence**. `max(tick, seq)` ⇒ `D` reverts to 900 while both players hold 200 ore between them. **100 units
   minted from nothing, and the deposit is re-mineable indefinitely.**

Nothing in revision 1's §10 caught it: `G-WL-ORDER`'s "independent reference" implemented the **same monoid**
so it agreed with the bug; `G-WL-CONSERVE-ACROSS-LIFECYCLE` kill-9s a shard (a kill-9'd shard emits nothing
further, so the cell never reaches the two-writer state); `G-WL-LIVE-EQ-DORMANT` has a single writer by
construction. It also made the worldline **the one durable write path in the repo that did not inherit fence
discipline** — `crates/wire/src/seams/directory.rs:7` says outright that the fence "is what makes a stale owner
harmless (fence rule 5)". And §7.2's phrase "keyed on the subject fence" was the **idempotency** key, not a
staleness reject; no section stated a reject rule.

**The four repairs, all mandatory:**
1. `DeviationRecord` carries `realm_fence`; the fold's ordering key is `(realm_fence, tick, seq)` with the
   **fence dominant**.
2. **The custodian REJECTS**, with a typed `WlRefused::StaleFence` counted as a **FAULT** (never benign), any
   deviation whose `realm_fence` is below the directory head's current fence for that realm — the same
   stale-reject the directory CAS already applies.
3. **`seq` is minted by the CUSTODIAN on ingest**, per `(realm, fence)`, from its durable monotone
   high-water — **not shard-locally**, because two incarnations minting from their own counters collide. The
   shard-side counter is only a client-side sequence within one fence and never reaches the fold key.
4. **A mandatory new gate cell** in both `G-WL-ORDER` and `G-WL-CONSERVE-ACROSS-LIFECYCLE`: *a
   force-reaped-but-still-running incarnation re-asserts deviations CONCURRENTLY with its successor*, with a
   **RED control** (remove the fence term and assert the gate FAILS). **The independent reference must be
   written from the FENCE rule, not from the same monoid**, or it will keep agreeing with the bug.

Absolute rebasing **costs nothing at the author site**: the authoring shard evaluated `f` to learn the
stock before mutating it, so it already *has* the absolute value. Deltas, by contrast, are neither
idempotent nor commutative and would make correctness depend on exactly-once delivery and a total order
that **no code path guarantees today** (`AppliedSteps` is an in-memory `BTreeSet<(TransferId, u32)>`,
`crates/sim/src/stub.rs:1087`; the durable table is owed as D-22).

⚠ **THE NO-STORED-COUNTER CLAUSE, and it is not decoration.** Revision 1's §6.3 said a promotion makes "the
cohort's `promoted_count` INCREMENT so the field never double-counts it" — **an increment, i.e. the one
non-idempotent operation this law exists to forbid**, on a `ReDriven` channel that re-asserts until acked, in
the design's own NPC section. The `Deviation` enum has no absolute carrier for it (`SetCohort` has no promoted
field), so a redelivered or replayed promotion increments twice, `P(t) − promoted_count` double-subtracts, and
anonymous NPCs **vanish**. §10's `G-WL-PROMOTION-BOUND` meanwhile asserted "promotion applied twice is a
no-op", so the **gate contradicted the design text** and revision 1's central idempotence claim (§6.3
"Recommend idempotent-by-construction") was false as written. **The fix is free: DERIVE it.**
`promoted_count(c) = |{ promoted subjects in the row set whose cohort == c }|` — idempotent, commutative,
crash-safe, and it makes **demotion** correct for the same reason (a `RemoveSubject` on a promoted NPC returns
it to the cohort automatically). Gate: a `G-WL-ORDER` cell that duplicates a `PromoteNpc` N times and asserts
byte-identity.

The custodian's `seq` high-water uses the `WaterMark` pattern
(`crates/node/src/rlm_spawn.rs:172-180, 292-355, 458-489`): persisted BEFORE each mint, **never** re-derived
from `max(survivors)`, so a `seq` retired by compaction can never be re-minted.

### 4.5 LAW-WL-3 (rebase-as-an-event) — the compaction/re-baseline rule, and Design A's repair

**THE RULE (Design B's framing, adopted over Design A's mitigation):** *the advance is always computed FROM
THE LAST EVENT, never from an intermediate checkpoint; a checkpoint is legalised by BEING an event
(`Deviation::RebaseSubject`).* Then **no flow/semigroup property is ever required**: `evaluate(base, devs, t)`
is a pure function of durable data.

⚠ **BUT A COMPACTED BASELINE IS *AUTHORITATIVE*, NOT A "DISCARDABLE DERIVED CACHE" — revision 1's wording was
false and it was load-bearing.** The stated mechanics **DELETE** the folded deviations, so after the delete the
pre-compaction inputs are **gone** and the baseline is the only record. A 1-minor-unit error in it is
**permanent**, and it is re-incurred every sweep. Revision 1 leaned on the "cache" framing to call the residual
"a cache-precision question, not a correctness one"; that is not available. Three consequences:
- **The residual needs a CUMULATIVE bound, not only a per-compaction one.** At
  `worldline_compact_interval_ticks = 72 000` that is ~8 760 compactions/year, i.e. an honest bound of
  ~8 760 minor units/subject/year (≈2×10⁻⁶ whole units at Q32 — small, but the design asserted *exactness* and
  had no cumulative bound at all).
- **The clean fix, and it is available: make compaction BIT-EXACT by construction.** Restrict a cut to ticks at
  which every subject's segment is closed by an actual `RebaseSubject`. A rebase carries the **absolute**
  value, so **no arithmetic is redone** and there is no residual to bound. Under lazy-on-adopt compaction
  (below) that is trivially arrangeable: the adopting shard emits a settling `RebaseSubject` per subject as
  part of adopt, then compacts at that tick. **RECOMMEND this**, and then `G-WL-RECOMPACT` asserts
  **bit-equality** rather than "≤1 minor unit".
- **`G-WL-RECOMPACT` AS WRITTEN WAS SELF-CONTRADICTORY** and must be re-specified either way: it asserted
  agreement "≤1 minor unit per subject **AND** the DIGEST agrees exactly". If the digest covers the `Fx` state
  those clauses cannot both hold; if it covers only the floored observation then the digest is a **lossy
  projection**, and the flagship gate's "byte-identical digest" was byte-identity of a lossy projection, which
  silently weakened it. **Decision, stated once: the digest covers the EXACT `Fx` state**, and compaction is
  bit-exact per the bullet above.
- The residual, if a non-rebase cut is ever permitted, is posted as a **declared physical loss**
  (`PhysicalLossChannel::CompactionResidual`, §9.4), **never** as a monetary event. That much of revision 1's
  repair of Design A's `Sink::Rounding` (which made a physical maintenance operation mint and burn *money*)
  stands unchanged.
- **The invariant `G-WL-RECOMPACT` should really pin, and did not:** `compact` is **IDEMPOTENT and
  ORDER-INDEPENDENT in the cut sequence** —
  `compact(compact(b, d, t₁), d', t₂) == compact(b, d ++ d', t₂)` **exactly**.

**Compaction mechanics** — write-ahead-then-effect, the `spawn_realm` pattern verbatim
(`crates/node/src/rlm_spawn.rs:438-539` with the `Store::flush` contract at
`crates/sim/src/io/mod.rs:426-433`):
```
put(Baseline@(cut_tick, cut_seq)) → commit() → flush() → delete(THE EXACT SCANNED KEY SET) → commit()
```
A crash in the window leaves BOTH the new baseline and the folded deviations. **The reader rule makes that
harmless and idempotent: prefer the highest `(base_tick, base_seq)` baseline and IGNORE every deviation whose
`(tick, seq) ≤ (base_tick, base_seq)`.** No double-apply is possible.

⚠ **THE READER RULE WAS KEYED ON TICK ALONE, AND THAT SILENTLY DROPPED DEVIATIONS — MINTING MATERIAL, WITH NO
CRASH REQUIRED.** §4.4 makes `(tick, seq)` the fold key, i.e. **multiple deviations at one tick are explicitly
expected**. Revision 1's reader rule was "prefer the highest `base_tick` baseline and IGNORE every deviation
with `tick ≤ base_tick`" and its delete was a `≤ cut` **range**. Concrete loss: two deviations exist at
`tick == cut`, `seq 3` (deposit 900) and `seq 4` (the player's second mine, deposit 800). The sweep scans,
folds `seq 3`, writes `Baseline@cut(D=900)`, then deletes `dev keys ≤ cut` — **deleting `seq 4` unfolded**; and
even had the delete not run, the reader rule would ignore `seq 4` **forever**, because its tick is not
`> base_tick`. The player keeps 200 ore; the deposit reads 900. This needs **no crash at all**: compaction and
deviation ingest are both custodian-side, `Store::scan` returns only COMMITTED rows
(`crates/sim/src/io/mod.rs:408-410` clause 3) and `commit` is block-on-**prior**
(`:426-433`), which *widens* the window. `G-WL-RECOMPACT` could not see it: it tested random cut points against
`evaluate` **over the same rows**, so both sides dropped the same row and agreed.

**The four repairs:**
1. `Baseline` carries **`(base_tick, base_seq)`** — a `(tick, seq)` **high-water**, not a tick — and the reader
   rule compares the pair.
2. The delete step deletes **the exact key set that was scanned and folded**, never a `≤ cut` range.
3. The cut is chosen as `min(now, last_fully_ingested_seq_watermark)` so **a tick is never split**.
4. `G-WL-RECOMPACT` gains a cell that **INTERLEAVES ingest with the sweep** and a cell with **≥2 deviations at
   the cut tick**, each with a RED control asserting the tick-only rule FAILS.

⚠ **DESIGN A'S ILLEGAL SENTENCE, STRUCK.** A said "the realm's live shard opportunistically" compacts via a
directory `Fence::cas_next`. **A shard cannot CAS a directory key**: the commit point is called directly
from the thread that owns the directory (`crates/node/src/saga_runtime.rs:997` — "THE single commit point,
called DIRECTLY (this thread owns the directory)", and again at `:1142`), and under the recommended storage
option the shard has **no `Store` at all**. As written that is a **SECOND COMMIT AUTHORITY**. Repair:
**compaction is CUSTODIAN-ONLY.**

⚠ **AND IT IS LAZY-ON-ADOPT ONLY — the periodic SWEEP is DELETED.** Revision 1 scheduled an orchestrator sweep
at `worldline_compact_interval_ticks` and **never said how it ENUMERATES the realms to compact.** There is no
worldline index anywhere in the design and the store families are keyed by realm, so enumeration **IS a family
prefix scan** — and `RedbStore::scan` materialises the entire prefix run into a
`Vec<(Vec<u8>, Bytes)>` with two heap copies per row (`crates/io-prod/src/store.rs:751-776`), inside a
**512 MiB `requests == limits` GUARANTEED-QoS** orchestrator (`deploy/k3d/30-orch.yaml:65-67`) whose liveness
probe has `failureThreshold: 6`. So growth in worldline rows OOMKills the one process that holds all durable
state, it restarts, re-attempts the same scan, and **crashloops** — while the RAM-only demand ledger
(`crates/sim/src/rlm.rs:227-236`, D-RLM-2) drops all realm-lifecycle intent on every restart. The alternative
(a RAM dirty set) is worse: it is lost on restart and, unlike the demand ledger, **does not self-heal**
(nothing re-asserts "realm X has pending deviations" every tick), so untouched realms are never compacted,
their logs are never bounded, and the terminal state is `WlRefused::LogFull` — **players in that realm can no
longer mine or build.** A restart-induced silent leak with a gameplay-visible end state.

> **COMPACTION IS LAZY-ON-ADOPT, AND THAT IS THE ONLY MODE.** A realm's log is folded exactly when the realm is
> next spun up and adopts — which needs **no index, no sweep, no dirty set and no durable cursor**, is bounded
> by `max_pending_devs` by construction, and is **free**, because the adopt path already performs that exact
> read (§5.5 step 4). For the pathological case (a realm continuously live for weeks), the **live shard**
> settles at `field_refresh_period_ticks` by shipping a normal `RebaseSubject` per subject — legal under
> LAW-WL-2/WL-3, needing no second commit authority. `worldline_compact_interval_ticks` is retired from
> `WorldlineTuning`; `field_refresh_period_ticks` does that job.
>
> **No code path may prefix-scan a worldline family.** If a background sweep is ever wanted anyway it needs a
> **durable cursor** in its own `StoreKey` family plus a bounded page size — and that requires a paged
> `scan_from(prefix, after_key, limit)` on the `Store` seam, which is the **real** forcing case for D-73 (a
> paged scan, not the point read revision 1 ledgered). Both are ledgered, not assumed.

**The pending-log bound, corrected.** `max_pending_devs` per realm; exceeded ⇒ new deviations are **REFUSED**
with a typed `WlRefused::LogFull`. ⚠ **Revision 1 classified that refusal BENIGN in §10 while making the
compaction residual a FAULT — exactly backwards by player impact**, since under WL-ACK a refused deviation
means the player **cannot mine or build** because a maintenance sweep fell behind. And the refusal path should
be **unreachable by construction**: LAW-WL-2's fold is last-wins per `(subject, field)`, so the pending log can
**always be COALESCED on ingest** (fold at write time, which absolute rebase makes free and exact). Its steady
state is then `O(subjects × fields)`, never `O(actions)`, so `max_pending_devs` can never legitimately exceed
`max_subjects_per_realm × fields`. **Repairs: coalesce on ingest; `WlRefused::LogFull` is a FAULT, never
benign; and `G-WL-PROMOTION-BOUND` gains a cell asserting that N actions on ONE subject produce ONE pending
row, plus a cell asserting a realm driven at the maximum player mutation rate never reaches the refusal.** In
practice `N` is bounded by **player attention**, not by world size or elapsed time (§5.5).

### 4.6 THE TWO RECORDS (the synthesis's central structural decision)

| | **WORLDLINE STATE** | **FACT JOURNAL** |
|---|---|---|
| Purpose | what a wake-up reads; what `evaluate` advances | history, audit, analytics, retro-payout replay |
| Authority | **required** — the game's truth | **derived** — a projection input |
| Volume | ~630 B–5.3 kB per *touched* realm; **0 B** untouched; **~5.3 MB** per 100 k-realm universe at 1 % visitation | 2.3–4.1 TB/yr (§5.4) |
| Shape | `Baseline` (overwritten) + bounded `Deviation` log | append-only **sealed content-addressed segment files** |
| Loss policy | **`LossBudget::ZERO`** — never shed; refuse-and-fault instead. Carries every **conserved-total** and **claim-founding** fact per RULE WL-CONSERVED-FACT (§2) | **ALWAYS SHEDDABLE, NEVER BLOCKING** — drop-and-count with a declared `JournalGap{from_lsn, to_lsn, count}`. **Nothing here is ever `LossBudget::ZERO`** |
| Home (v1) | the **worldline custodian**'s own redb file, new `StoreKey` families (§5.4 W2) | Tier-B segment files + Warehouse (**optional**, §5.4) |
| Absence | impossible (it is the state) | **the world advances byte-identically; only history has an auditable hole** |

**This is what resolves Judge-1's and Judge-3's objections to Design C in one move.** C's cost problem was
that its *one* record had to be both authoritative and 2.3 TB/yr; C's third-byte-channel and
required-Warehouse problems follow from that. Splitting them keeps **every** piece of C's discipline while
making the archive optional. **A declared gap is auditable; a silent one is not.**

**ONE EMIT, TWO CARRIERS, AND THEY MUST NOT SHARE BACK-PRESSURE.** A shard emits exactly one thing — a
`WorldFact` — and the bin layer **tees** it: (a) the economically-significant subset becomes a
`WorldlineDeviation` on the **reviewed `InterShardFlow` arm** (§7.2 #3), `LossBudget::ZERO`, with its own
back-pressure; (b) the whole stream feeds a **separate, always-sheddable, never-blocking** segment sink. ⚠ **This
deliberately avoids adding a 5th `sim::io` trait**, which Judge 3 named as Design C's largest structural cost
("a `FactSink` goes into `build_app`, so `MemFactSink` is instantiated in EVERY integration test binary, where
HR5 rule (c) is all-or-nothing"). Zero new Tier-A trait, zero new mem twin in every test binary.

⚠ **REVISION 1 SHARED ONE CARRIER BETWEEN TWO SINKS WITH ASYMMETRIC LOSS POLICIES, WHICH LETS THE ECONOMY
BACK-PRESSURE THE GAME.** It mandated "ONE EMIT, ONE CARRIER, TWO SINKS" and then gave the two sinks *different*
loss policies on that shared carrier: physical facts may shed LOUD, but "money-class facts are
`LossBudget::ZERO` and the ECONOMY stalls instead". On a shared bounded ring (`local_segments_retained` 8 /
`local_segments_max` 128, §5.4) a non-sheddable fact **cannot be dropped**, so the ring cannot advance, so the
tee cannot accept, so **the SHARED EMIT SITE BLOCKS** — i.e. "the economy stalls" becomes "the tick stalls",
which is the precise LAW-E1 failure this document exists to prevent. Revision 1 applied exactly this critique
to Design C's `G-JOURNAL-FAITHFUL` (§12.1: "cannot hold across the physical shed C itself specifies") and did
not apply it to its own carrier.

⚠ **AND IT NAMED THE CARRIER TWO INCOMPATIBLE WAYS.** §4.6 said the fact rides "the existing egress" teed by
`io-prod`, and §7.2 #10 said the journal leg is "NOT an `InterShardFlow` arm — io-prod-local, below the seam,
OPAQUE bytes", while §7.2 #2/#3 said deviations ride `BlockEdit`/`WorldlineDeviation` **`InterShardFlow`** arms.
Those are different byte channels with different review status, and the io-prod-local path is precisely the
"third unreviewed byte channel" §7.3 rejects Design C for. **Resolved, one carrier per record, named:**

| Record | Carrier | Reviewed? | Loss |
|---|---|---|---|
| **`WorldlineDeviation`** (the STATE record, incl. every RULE WL-CONSERVED-FACT row) | the reviewed `InterShardFlow` arm §7.2 #3, `MsgClass::Saga`, its own back-pressure | **YES — HR1-reviewed, `effect_class` + `durability_class` compiler-forced** | `LossBudget::ZERO`; refuse-and-FAULT, never drop |
| **The fact JOURNAL** | an **in-process bin-layer drain in the shard binary** (`vd-bins → vd-io-prod` already exists and io-prod is not Tier-A), handed off through one resource the sim writes and the bin reads | not a wire arm at all — it never crosses a process boundary from the shard's perspective | **always sheddable, drop-and-count, NEVER blocks the emit** |

The journal's sim→bin handoff resource is a real, coverable artefact and is named as such in §11.4 (it was
implicit in revision 1). And because RULE WL-CONSERVED-FACT (§2) moves every conserved-total fact into the
STATE record, **no journal fact needs to be `LossBudget::ZERO`**, which is what removes the shared-ring
block entirely rather than merely bounding it.

**Atomicity, honestly.** Design C's best graft is "stage the fact in the SAME `Store::commit()` batch as
the game mutation, so a fact exists iff its mutation exists" (the transactional-outbox shape already named
at `crates/io-prod/src/outbox.rs:1-16`). **That is not available on a shard today**, because the mutation
lives in the World (RAM) and there is **no shard-side `Store`** (`StoreRes` is inserted only by
`register_orchestrator_with_store`, `crates/node/src/orchestrator.rs:152`; grep finds no store in
`crates/bins/src/bin/shard.rs`). The saving grace is that this is *consistent*: if the shard dies, the
mutation dies with it — which is exactly what the shard's own SIGTERM drain asserts ("the shard holds no
un-fsynced durable state", `crates/bins/src/bin/shard.rs:294-302`). So:

> **RULE WL-ACK — a DURABILITY BARRIER, never an AUTHORITY GATE (rewritten in revision 2).**
>
> 1. **The live shard is the AUTHORITY OF RECORD for its own realm's worldline.** The custodian is a durable
>    replica plus the dormant evaluator. This is the standing frame-authority law, and revision 1 inverted it
>    by making a remote `WlRefused` able to reject a mutation the shard had already applied to its `World` —
>    with **no rollback path anywhere in the design**, so the two states would simply diverge.
> 2. **Every budget is enforced LOCALLY, BEFORE the local apply**, from the adopted counts:
>    `max_subjects_per_realm`, `max_pending_devs` (post-coalescing, §4.5) and the per-blob budget are checked at
>    **MUTATION time** with a typed refusal — the same mutation-time-budget rule §7.3 already states for TLV
>    blobs. A refusal is therefore synchronous, local and explicable; it is never an async surprise.
> 3. **The ack waits on `flush`, not `commit`.** `Store::commit` is block-on-**prior** depth-1 and returns
>    BEFORE its own batch fsyncs (`crates/sim/src/io/mod.rs:426-433`; `crates/io-prod/src/store.rs:783-790`),
>    so acking after `commit` can lose a mutation the player was already told succeeded — the exact class the
>    write-ahead-then-flush-then-effect pattern exists to prevent (`crates/node/src/rlm_spawn.rs:438-539`).
>    Revision 1 said "durable" without saying which, and the fsync-rate ceiling that follows is a **measured**
>    number owed by W-(−1) (§5.4) and gated by G-WL-CUSTODIAN-THROUGHPUT (§10).
> 4. **OPTIMISTIC ACK IS LEGAL FOR PURE ABSOLUTE REBASES, and that is most of gameplay.** A deviation that is
>    a pure `RebaseSubject` is idempotent and replayable, so the shard may ack **immediately** and mark the row
>    provisional; the durability barrier applies only to **irreversible mints** (an item minted, a hull
>    granted, a claim founded — i.e. the RULE WL-CONSERVED-FACT classes). Mining a rock and installing a
>    machine are rebases; they never wait. This is what keeps §2 rows 5/7/8/10/12 honest.
> 5. **The provisional window is bounded and visible**: a `worldline_provisional_devs` gauge plus a
>    `worldline_ack_barrier_ms` percentile, both DevState-pollable, with a fail-loud counter when the barrier
>    cannot be met — never a rejected mutation and never a silent stall.
>
> When shard-side Store B lands (P6/P7), Design C's same-commit-batch graft applies verbatim and even the
> irreversible-mint barrier becomes local — the design is forward-compatible, and this is **W16**.

> **RULE WL-ADOPT-REFUSE — what a player may do BEFORE `worldline_seeded` (revision 1 left this undefined,
> and both plausible answers were bad).** §5.5 step 3 gates every worldline system on
> `.run_if(worldline_seeded)`, and `WorldlineAdopt` is pushed by the orchestrator *after* launch, so there is a
> window in which a realm is live and a player is in it with no baseline. Revision 1 said nothing, leaving
> either "refuse the mine" (an unexplained stall) or "act against a seed-only baseline" (**a mined-out deposit
> reads full again — material duplication, a fresh split-brain**). Normatively:
> **the realm is admitted for MOVEMENT and PHYSICS immediately; worldline MUTATIONS are REFUSED with a typed,
> player-visible, non-fatal `WlRefused::NotSeeded` counted as a FAULT, and there is NEVER a silent
> seed-baseline fallback.** Gate: a `G-WL-CONSERVE-ACROSS-LIFECYCLE` cell that delays/drops the adopt N times
> and asserts no material is created, plus an `unseeded_live_realms` DevState counter so the window is
> observable rather than mysterious (§7.2 #4).

### 4.7 Cadence hierarchy

One struct, `WorldlineTuning`, copying `RlmTuning` verbatim (`crates/sim/src/rlm.rs:35-179`): every window
derived from the cadence quantum (never inline literals), an **all-zero `Default`** that is provably
byte-identical (`eval_interval_ticks == 0` ⇒ the system early-returns), and a fail-loud `validate()` at
boot next to the existing `RlmTuning::validate` call (`crates/bins/src/bin/orchestrator.rs:366-368`).

⚠ **THE CADENCES ARE IN TICKS, DERIVED FROM ONE QUANTUM — revision 1's table derived them from SECONDS at a
hardcoded 20 Hz, which contradicts its own RULE WL-4.** §9.2 asserts "no worldline quantity may derive from a
`secs_since_epoch` result or from the local `tick_dt_s`" and concludes the design does not depend on
D-Finding-3. The table then wrote `field_refresh_period_ticks = 600 s × 20`,
`growth_quantum_ticks = 3600 × 20`, `max_catchup_rounds = 720 (30 days)`, and §5.4's whole worked example
converts ticks to human time at a rate that **is not a universe constant**: `ClockSync` carries only
`{universe_tick, epoch}` (`crates/wire/src/seams/directory.rs:120-125`) and `tick_hz` is computed per-shard as
`1.0 / config.tick_dt_s` (`crates/sim/src/stub.rs:4391`). So WL-4 bought bit-equality while every rate literal
silently fixed a rate, and a content author writing "1 unit per 40 ticks" had hardcoded 20 Hz into the world's
economy.

**Decision: the TICK is the sole unit, everywhere, including content authoring.** Every window is an integer
multiple of ONE named `cadence_quantum_ticks` field in `WorldlineTuning`; the seconds column below is
**human-facing annotation only, at an illustrative 20 Hz**, and no formula in this document may read it. The
design therefore genuinely does not depend on D-Finding-3 — but §11.1 records the honest consequence: if the
user ever wants *content* to be portable across differing `VD_TICK_DT` deployments, or wants believability
targets expressed in human time, then a canonical universe seconds-per-tick on `UniverseConfig`/`ClockSync`
(D-Finding-3) becomes a **content** prerequisite, though never a bit-equality one.

| Cadence | Field | Multiple of the quantum | ≈ at an illustrative 20 Hz | Who runs it |
|---|---|---|---|---|
| per tick | NPC materialisation band check | 1 tick | every tick | live shard |
| **the quantum** | `cadence_quantum_ticks` | 1 | 12 000 (≈10 min) | — |
| field refresh + settle | `field_refresh_period_ticks` | 1 × quantum | 12 000 | live shard |
| growth quantum | `growth_quantum_ticks` | 6 × quantum | 72 000 (≈1 universe-hour) | the `ipow` exponent unit |
| **season quantum** ⚠ *new* | `season_quantum_ticks` | 6 × quantum | 72 000 | §4.3a's table index unit |
| journal seal | `journal_seal_period_ticks` | 1 × quantum | 12 000 | the bin-layer drain (off-tick) |
| ~~compaction sweep~~ | ~~`worldline_compact_interval_ticks`~~ | **RETIRED (§4.5)** — compaction is lazy-on-adopt; the live-shard settle at `field_refresh_period_ticks` does the rest | — | — |
| coarse agent round (opt.) | `site_round_period_ticks` | 6 × quantum | 72 000 | pinned host (§5.6) |
| faction round (opt.) | `faction_round_multiple` | 24 rounds | 24 × 72 000 | pinned host |
| **fallback only** | `max_catchup_rounds` | 720 rounds | 51.84 M ticks (≈30 days) | exceeded ⇒ adopt the pure closed form |

⚠ **The naive dormant catch-up tier is DELETED as a primary mechanism.**
`scripts/economy_research_20260726.md` §6.4 item 4 computes that 30 days at 20 Hz is 51.8 M ticks = ~52 s
of blocking CPU on wake even at an optimistic 1 M ticks/s, and its own declared fallback when
`max_catchup_ticks` is exceeded is *"adopt the pure closed-form field rather than replay"* — i.e. it needs
the closed form **regardless**. Keeping only the fallback deletes two tuning fields and a cliff behaviour.
`max_catchup_rounds` survives **solely** for the optional coarse-agent tier's abnormal gaps (§5.6).

Full field list: `cadence_quantum_ticks`, `eval_interval_ticks`, `field_refresh_period_ticks`,
`growth_quantum_ticks`, **`season_quantum_ticks`**, **`max_season_entries`**, `journal_seal_period_ticks`,
`max_subjects_per_realm`, `max_pending_devs`, `max_promoted_subjects`, `max_materialized_npcs`,
`max_rate_num`, `max_stock_minor`, **`max_den`**, `local_segments_retained`, `local_segments_max`,
`journal_bytes_per_realm_budget`, `raw_fact_retention_days`, `max_catchup_rounds`,
**`econ_ingress_budget_bytes_per_tick`**, **`min_dormant_ticks`**, **`content_epoch`**.
⚠ **`promotion_ttl_ticks` is DELETED** (§6.3: the TTL demotion branch guaranteed an unbounded teleport);
⚠ **`worldline_compact_interval_ticks` is RETIRED** (§4.5).

### 4.8 The same code in all three hosts (HR3, literally)

Because `evaluate`/`compact` are in `vd-core` with no internal deps:
- **the sim** calls `evaluate` at `field_refresh_period_ticks` and on adopt;
- **the wake-up path** calls the identical `evaluate` on the adopted baseline;
- **the Tier-B dashboard** calls the identical `evaluate` over the same durable rows — H4's
  evaluate-on-read, and it never wakes the realm;
- **the harness** calls the identical `evaluate` over a synthetic stream for gates.

The **only** difference is where the rows come from, and that lives behind the caller, never inside the
kernel.

⚠ **THE TIER-B READ PATH DOES NOT EXIST, AND REVISION 1 BILLED IT AT "NONE — off the taxonomy".** §7.2 #6 said
"`crates/io-prod/src/admin.rs` reads the store and calls the same `evaluate`". It cannot: the durable store is
**moved into the ECS `World`** as `StoreRes(Box<dyn Store + Send + Sync>)` (`crates/node/src/saga_runtime.rs:225`,
injected `crates/node/src/orchestrator.rs:106`, `:152`) and used as `&mut dyn Store`, while `admin.rs` serves
over axum/tokio behind a `SnapshotSource: Send + Sync + 'static` returning a pre-built `AdminSnapshot`
(`crates/io-prod/src/admin.rs:24-36`) whose own doc requires values be "cheap + non-blocking" (`:39-41`). There
is **no handle by which admin can `scan`**. So H4 — one of four headline properties — was an **unbuilt slice
priced at zero**; and built the naive way it would run an unbounded analytics query inside the process that
owns the sole directory CAS commit point (`crates/node/src/saga_runtime.rs:997`) and the sole RLM kill
authority, over a `Store` with no point read, so **a monetary dashboard could stall transfers** — an economy
consumer stalling the game.

> **THE READ PATH, specified and costed as a real W-1 deliverable.** A `WorldlineRead` port defined next to
> `Store` — **scan-only and `&self`** — implemented in io-prod over its **own redb read transaction** obtained
> at boot and held by the admin task, **never** the ECS `StoreRes`, with a `MemWorldlineRead` twin. redb MVCC
> read txns are safe alongside the writer, and a torn read across a commit is impossible by construction (a
> read txn sees one snapshot). Contract: **per-request row caps** (`admin_max_rows_per_request`), a **named
> latency budget**, and an assertion that the read path can never take the ECS `World` or block the tick.
> Gate: **G-WL-READPATH-ISOLATION** (§10) — hammer the evaluate-on-read endpoint and assert the custodian's
> tick budget and the RLM `LifecycleAction` trace are unaffected.

⚠ **H4 is demonstrable only at toy scale today, and this is not our bug but it is our dependency.**
`generate_system_forest` hardcodes **one** Universe, **one** Galaxy, one `SYSTEM_A` plus `n_planets`
(`crates/core/src/worldgen.rs:383-421`); `direct_child_levels` (`:290-300`) and `container_coord_at`
(`:471-500`) each **regenerate the whole forest per call**, and `container_coord_at`'s own doc says so
("unifying the two under one seed-lazy generator is the P4 owe"). Until the lazy per-subtree generator
lands, "a never-visited realm costs zero bytes because its content is seed-derived" is provable on a
single-galaxy forest only.

---

## 5. DORMANCY END-TO-END — a worked 30-day example

### 5.1 Setup

Realm `Planet(7)/Area(3)`, a mining outpost. Seed-derived content: 4 ore deposits, 6 construction slots.
Player history: one refinery installed at tick 1 000 000; one deposit half-mined at tick 1 002 500; one
named foreman NPC (born promoted). At **T₀ = 1 010 000** the last player leaves.

### 5.2 T₀: the realm is REAPED, and nothing needs flushing

Every deviation was already durable at the custodian before its player ack (RULE WL-ACK), so the shard
holds no un-fsynced worldline state — exactly what `crates/bins/src/bin/shard.rs:294-302` already asserts
about itself. **This design needs NO spin-down checkpoint hook**, which dissolves the hardest blocker in
the fact base: `kill_realm` deletes the launch intent, commits, then SIGTERM/SIGKILLs the process group
(`crates/node/src/rlm_spawn.rs:541-555`; `crates/bins/src/proc_launch.rs:214-242`) with no drain-then-flush
anywhere. **This is only affordable because deviations are SPARSE** — player actions, not per-tick state.

⚠ **`Empty` means zero OCCUPANTS, not zero activity** — verified: `aoi_decide` builds the occupant set as
owned simulating dots **chained with held transients** and self-reports `DemandVerb::Empty` iff that set is
empty (`crates/sim/src/stub.rs:4396-4421`). So a realm full of NPCs and refineries **is** reapable. Under
LAW-E2 that reads backwards — and for this design it is **exactly right**: we WANT it reaped, because
nothing needs to run.

### 5.3 T₀ → T₀+30 d: ABSOLUTELY NOTHING HAPPENS

No process, no tick, no timer, no aggregate shipped by anyone, no cross-shard byte. Zero CPU. Zero bytes
written. The orchestrator's compaction sweep touches this realm and finds nothing to do (2 pending
deviations, far under `max_pending_devs`).

There is no "dormant tier that ticks a little", and there cannot be: **there is no `Dormant` state in the
code.** The actual state space is a four-way partition of `absent` / `launching` / `running_live` /
`zombie` (`crates/sim/src/rlm.rs:394-415`, pinned by a test literally named
`running_live_zombie_launching_partition_the_actual_states` at `:1015`). Dormant means **no process and no
storage**. And `D-RLM-4` mandates that any warm pool be profile-**agnostic** blank shards with **no mode
branch**, so "a realm that ticks economy only" is precisely the per-kind mode fork that decision forbids.

### 5.4 The numbers

**Time.** 30 days at 20 Hz = `30 × 86 400 × 20 = 51 840 000` ticks. `Δ = 51 840 000`.

**Refinery** (rn/rd = 1 unit per 40 ticks; S = 4 000 000 units; hopper B = 500 000 output units; yield
1 output per 2 input, so yn/yd = 1/2):
```
Δ_dry  = 4 000 000 × 40                     = 160 000 000 ticks   (the deposit does NOT dry in 30 days)
q_full = 500 000 × 2 / 1                    =   1 000 000 input units
Δ_full = 1 000 000 × 40                     =  40 000 000 ticks
Δ_stop = min(160 000 000, 40 000 000)       =  40 000 000  <  51 840 000  ⇒ the hopper FILLED
fill day = 40 000 000 / (20 × 86 400)       =  40 000 000 / 1 728 000 = 23.15 days
idle     = (51 840 000 − 40 000 000) / 1 728 000                     =  6.85 days
C = 1 000 000   S' = 3 000 000   h' = B = 500 000
```
⇒ **the refinery filled its hopper on day 23 and has been idle for ~7 days.** Believable, correct,
computed in **4 muldivs + 3 `div_ceil`s**.

⚠ **AND — the conclusion revision 1 computed and did not draw — WITHOUT §4.3a'S SEASONAL DRIVER, `evaluate(t)`
IS CONSTANT FOR EVERY `t > t₀ + 40 000 000`.** A 6-month dormancy (315 532 800 ticks) and a 5-year dormancy
return **bit-identical** state to the 23-day case. `Δ_stop` **is** the believability horizon
(`Δ_stop = min(Δ_dry, Δ_full)`), and with the driver the machine resumes in the next high season instead. This
is LAW-WL-6 (§0 correction 4) in the design's own numbers.

**NPC cohort.** `m = Δ / growth_quantum_ticks = 51 840 000 / 72 000 = 720`; `log₂ 720 = 9.49` ⇒
**10 squaring steps** in `ipow_muldiv`. Population relaxed from P₀ = 180 toward K = 240 — the outpost grew,
*because the player built the refinery* (an installed construction raises K).

⚠ **AND THE COHORT FREEZES HARDER THAN THE MACHINE, because Q32 UNDERFLOWS.** The minimum representable `Fx`
magnitude is `2⁻³²`, so the gap term floors to **exactly zero** once `((K − P₀) << 32) · rᵐ < 1`, i.e. at
`m* = ln((K − P₀)·2³²) / −ln r`. For this example (`K − P₀ = 60`, so `ln(60·2³²) = 26.27`):

| `r` per growth quantum | `m*` (quanta) | horizon | observed `P` at 30 days |
|---|---|---|---|
| 0.90 | 249 | **10.4 days** | exactly 240 = K |
| 0.95 | 512 | **21.3 days** | exactly 240 = K |
| 0.99 | 2 614 | 108.9 days | **already exactly 240** (observation floors at `m ≈ 407`, day 17) |
| 0.999 | 26 269 | ~3.0 years | 210.8 — the only band with both a long horizon and a perceptible 30-day change |

So for any plausible relaxation rate **every dormant region converges to precisely its seed-derived `K` and
stays there, bit-exactly, and every region sharing a `K` is numerically identical** — which is the mechanism
behind §1.2's "the same crowd, phase-advanced" admission, and it is worse than that admission implies. §4.3a's
**moving `K`** removes the fixed point entirely (the attractor becomes a periodic orbit), and the `m*` formula
above belongs next to the population law in §6.2 so anyone tuning `gn/gd` can see the horizon they are
choosing. `ipow_muldiv` should also **early-exit** once the gap underflows — the `O(log m)` cost is moot past
`m*`.

**Foreman NPC** (promoted): unchanged pose ephemeris, phase-advanced by Δ. One integer modulo.

**Storage.** ⚠ **REVISION 1'S `SubjectState ≈ 96 B` WAS A `sizeof` CALCULATION, NOT THE ENCODING.** It read
`3×Fx(i128)` as a fixed 48 bytes, but the workspace mandates **postcard v1** (`Cargo.toml:39`), which
LEB128-varints every integer wider than 8 bits and zigzag-varints signed ones. Measured against postcard v1 in
this workspace:

| Value | Encoded |
|---|---|
| `Fx(0)` | 1 B |
| `Fx(1 << 32)` (= 1.0) | 5 B |
| `Fx(4_000_000 << 32)` | 8 B |
| `i128::MAX / 2` | **19 B** |
| `u64` = 0 / 10³ / 10⁶ / 5.18×10⁷ / `u64::MAX` | 1 / 2 / 3 / 4 / **10** B |

So `SubjectState` is **data-dependent over roughly 35 B (a `Cohort`) to ~170 B (every field at its cap)**, with
a *typical* `Machine` including the `WorldEpoch` that LAW-WL-5 mandates on every durable record landing around
**55 B**. **A 4–5× data-dependent range cannot be served by one constant**, and the two budgets it feeds pull in
opposite directions:

> **THE ENFORCEMENT, which replaces the estimate.** Declare **`MAX_SUBJECT_BYTES`** as a named const asserted
> by a test that **postcard-encodes every `SubjectKind` arm with every field at its MAXIMUM** and compares
> against it (the fail-loud-at-registry-load pattern), plus a proptest that no reachable subject encodes above
> it. Then derive **`max_subjects_per_realm` from the WORST case** —
> `MAX_SUBJECT_BYTES × n ≤ a named fraction of MAX_FIELD_BYTES` — and quote the **typical** figure only for the
> storage bill. Under-estimating the wire budget is the dangerous direction: an oversize *required* TLV tag is
> `MissingRequiredTag`, documented "refuse the transfer, never decode to Default"
> (`crates/core/src/tlv.rs:58`), so a 1.5× overshoot converts "this realm got rich" into "**this realm can
> never be adopted**" (§7.3).

With that, and marking the endpoints ⚠ **owed to W-(−1)** per the report's own E-(−1) discipline:
```
Baseline = key + base_tick + base_fence + base_seq + digest + Σ subject encodings
  3 subjects  :  typical ≈  0.2 kB      worst ≈  0.6 kB
 55 subjects  :  typical ≈  3.0 kB      worst ≈  9.4 kB
DeviationRecord: typical ≈ 60–90 B     worst ≈ 200 B  (it now also carries realm_fence + epoch)
This realm    :  typical ≈  0.4 kB
```
- **A never-visited realm stores ZERO BYTES** — absence of a baseline *means* pure seed, and that is
  gate-asserted (G-WL-ZERO-BYTES).
- 100 000 realms at 1 % ever-visited: **≈3.0 MB typical, ≈9.4 MB worst** (revision 1 said 5.36 MB, which sits
  inside the band and was a fixed-width coincidence rather than a derivation).
- ⚠ **ARITHMETIC CORRECTION to Design A**, which claimed this is "0.002 % of the 256 Mi PVC". Against
  `256 Mi = 268 435 456 B` the **per-realm** figure is ~10⁻⁵, but the **whole universe** is ~1–4 %, not
  0.002 % — A conflated the two by 1000×. The conclusion survives comfortably; the number must be stated
  correctly.
- Against the report's own `journal_bytes_per_realm_budget` of 192 MiB (§2.3a) the worldline **state** is
  four orders of magnitude smaller. **That is what dissolves the report's §6.4 storage-topology blocker
  for the physical layer** — it stays true for an order book, which lives only where a live venue exists.
- ⚠ **AND THE 1 %-EVER-VISITED ASSUMPTION IS UNJUSTIFIED OVER YEARS.** `G-WL-ZERO-BYTES` only asserts that a
  **never**-visited realm is empty; a realm visited **once** keeps its baseline forever, so the touched set is
  **monotone and never shrinks**. A retention rule is therefore owed and is ledgered (**D-79**): a realm whose
  baseline has re-converged to its pure-seed evaluation within the declared fixed-point width may be
  **tombstoned back to zero bytes**, which is lossless by construction — the same argument the report's §7.7
  makes for evicting a decayed *deviation*, and unavailable for stock, which is value.

**Wake compute — TWO bands, warm and cold, because revision 1 quoted only the warm one.** A general `u128`
division on x86-64 is **not an instruction**: it lowers to a compiler-rt `__udivti3` software routine, ~40–120
cycles (≈13–40 ns at 3 GHz), not the ~20–40 cycles / ~10 ns revision 1 assumed.
```
WARM  : 55 subjects × ~10 muldiv/div ops = 550 ops × ~30 ns   ≈ 17 µs
        + exact-key Store scan, redb page cache warm          ≈  1–5 µs
        + postcard decode of ≤9.4 kB                          ≈  2–5 µs
        TOTAL                                                 ≈ 20–30 µs
COLD  : the same, but the first read is 2–3 page reads from a `local-path` PVC
        ⇒ ~0.3 ms on NVMe, up to tens of ms on slower backing storage
        TOTAL                                                 ≈ 0.3–30 ms
```
⚠ **This is why revision 1's `WORLDLINE_WAKE_BUDGET_US` (~300 µs, "~15× observed") would PASS IN CI AND FAIL
IN CLOUD**: CI is warm and effectively tmpfs-backed; a single cold read on non-NVMe storage exceeds it. **Two
named budgets, gated separately** (§10): a WARM compute+decode budget, and a COLD first-read budget measured
against a **dropped page cache on the real PVC class**. Both are ⚠ **[U] until W-(−1) measures them**, and
G-WL-WAKE-LATENCY must carry the delivery/count floor `percentile_unstable`'s own doc demands (it returns
`Duration::ZERO` on an empty sample set, `crates/harness/src/latency.rs:16-21`, so a gate with no samples
passes deceptively).

⚠ **The "≈6 orders of magnitude cheaper than the naive catch-up tier" headline is DELETED as a strawman.** It
compares `evaluate` against a tier this design itself removes. The honest statement is that **`evaluate` is
negligible against PROCESS BOOT** — which is an argument for the closed form *and* for D-RLM-4's warm pool, and
which points the other way by five orders of magnitude (§5.5).

**Ongoing dormant cost: exactly zero** — no process, no timer, no ticking, no aggregate on any wire.

**⚠ THE CONCURRENCY CEILING, STATED ONCE, because revision 1 priced two different worlds.** Every wake **forks
a real process** (`ProcLaunchBackend`), each shard requests 256 Mi / limits 384 Mi
(`deploy/k3d/50-shard.yaml:66-68`), and `RlmTuning::cloud_with_boot`'s own doc records "a slow real fork (a ~3 s
CI boot)" and floors `launch_ttl = max(hz·3, boot_ticks_p99 + settle)` (`crates/sim/src/rlm.rs:92-108`).
Inverting on a 1 TiB cluster (~4 000 live shards at 256 Mi) and allowing ~10 % simultaneously booting (~400),
the sustained fork rate is ~133/s ⇒ ~27 chain-wakes/s ⇒ at one region transition per player per 60 s,
**≈1.6×10³ concurrent players.** Revision 1's journal arithmetic below (2 300 facts/s, 833 block-edits/s) is a
**10⁴-player** figure. **Decision: v1 is scoped to the ~10³ figure**, every number in this document is derived
at that scale, and **D-RLM-4's warm-spare pool is named a hard prerequisite of the 10⁴ story** rather than an
unrelated RLM deferral — a pre-booted profile-agnostic blank shard is the only thing that turns a ~3 s wake
into a ~100 ms adopt, and it is also what makes §5.5's AoI-radius derivation tractable for small realms. Gate:
a release-only **G-WL-WAKE-RATE** soak (N realms cycling reap/re-spin at a target rate, asserting no fork
backlog and bounded aggregate RSS).

**The journal's separate bill** (§4.6), from Design C's arithmetic, re-derived at the 10³-player scope:
```
facts/s  = 833 (block place/break) + 1 000 (item mint/move) + 0.3 (destruction)
         + 0.0006 (NPC births/deaths) + 300–600 (postings, economy ON)  ≈ 2 300/s  [at 10⁴ players]
facts/yr = 2 300 × 3.156×10⁷ = 7.26×10¹⁰
bytes/yr = 7.26×10¹⁰ × ~48 B = 3.48×10¹² = ~3.5 TB/yr
mean realm (10⁴ live) = 0.23 facts/s ⇒ 7.26×10⁶ facts/yr × 48 B ≈ 348 MB/yr
                       ⇒ ~9 months to fill a 256 Mi PVC
hot realm (10 % of facts in 1 % of realms) ≈ 3.5 GB/yr ≈ 13× the PVC per year
```
⚠ **`32 B/fact` was too small and the direction matters.** A fact that is actually **replayable** (§10
G-WL-SCHEMA-FLOOR; W-4's "reproduces the state offline from the journal alone") must carry at minimum a kind
tag, a realm reference, a subject/`ItemId`, a tick, the mandated `WorldEpoch`, a quantity, a typed reason and an
LSN — measured at **~48 B** under postcard with realistic values (a `u128` `ItemId` alone is 18 B; a year-scale
tick is 5 B). So the archive bill is **~3.5 TB/yr**, not 2.32, and the mean realm fills its PVC in **~9
months**, not 14. Two consequences:
- **Never key a fact by the full path key**; that would add ~44 B/fact (≈+3 TB/yr). Carry the realm identity
  **once per SEGMENT header** and a compact lowered reference per fact.
- **Take the free 36 % reduction, or decide not to.** 833 of the 2 300 facts/s are block place/break, which
  §11.2 already excludes from the worldline **state**. Excluding them from the **archive** too (or coarsening
  them to the `BulkPlaced`/`BulkDestroyed` summaries §11.3 already proposes) drops the bill from ~3.5 to
  **~2.2 TB/yr**. ⚠ **State the decision: is per-block history worth ~1.3 TB/yr?** (**W18**.)

**The ring sizing survives the correction** (`128 × 20 000 × 48 B ≈ 123 MB ≤ journal_bytes_per_realm_budget =
192 MiB`), **but the ring's HOME does not.**

⚠ **THE SHARD-LOCAL JOURNAL RING IS LOST ON EVERY REAP, WHICH MANUFACTURES A PERMANENT NONZERO FAULT COUNTER.**
Revision 1 put the staging ring on shard-local disk, while §12.2 W2(b) establishes that the k3d PVC is
`ReadWriteOnce` + `local-path` + bound to the StatefulSet **ordinal**, not to a realm
(`deploy/k3d/50-shard.yaml:92-96`) — so the next incarnation of the same realm cannot reach the previous
incarnation's ring. Meanwhile §3.4 explicitly rejects any handback ack, `kill_realm` SIGTERMs then SIGKILLs with
no drain-then-flush (`crates/node/src/rlm_spawn.rs:541-555`; `crates/bins/src/proc_launch.rs:214-242`), and the
shard's own SIGTERM path asserts it "holds no un-fsynced durable state"
(`crates/bins/src/bin/shard.rs:294-302`). Therefore **every** spin-down drops up to one
`journal_seal_period_ticks` of unsealed facts, forever — at even 10³ reaps/day that is 10³ FAULT records per day
as **normal operation**, a gate that requires `faults == 0` is permanently red, and the inevitable response is
to reclassify the fault as benign, which is the §8.6 Rule-0 gate-decay this design cites everywhere else.

> **THE FIX: the ring lives at the CUSTODIAN, not on shard-local disk.** Facts are teed to the custodian
> alongside the deviation write (they already share that path under RULE WL-ACK, so it costs no extra round
> trip), and the **custodian owns the segment ring** — history durability then inherits the custodian's
> availability instead of a doomed ordinal-bound local disk. If a local ring is ever kept anyway, it must
> **seal-on-drain** inside the reconciler's existing two-phase `drive_drain` window
> (`crates/sim/src/rlm.rs:555-572`) — the same window §3.4 already uses for the voluntary final deviation
> batch — and the residual must be a **separately-named EXPECTED counter** (`journal_unsealed_on_reap`) that is
> **not in the fault total**, with a bounded-magnitude gate rather than a zero gate.

**The whole archive is optional (W10)**: the world advances without it — but per RULE WL-CONSERVED-FACT (§2)
nothing conservation-bearing is in it, which is what makes that optionality safe.

### 5.5 THE WAKE-UP

A player's AoI brings the realm into demand; the reconciler emits `SpinUp`, **ancestor-first**
(`crates/sim/src/rlm.rs:627-636`, ordered at `:678-684`).

1. `spawn_realm(&RealmCoord, at_tick) -> NodeId` — takes only a coord and a tick and returns only a
   `NodeId` (`crates/sim/src/io/mod.rs:436-479`). **There is no place to hand a realm an initial
   condition, and this design does NOT change that frozen sealed port.**
2. Immediately after launch, the **orchestrator PUSHES** `WorldlineAdopt{ realm_fence, baseline, devs }`
   to the fresh shard. ⚠ **Orchestrator-RELAYED, never parent→child direct** (Design B's routing graft): a
   parent does not know its children's `NodeId`s (the spawner mints them,
   `crates/node/src/rlm_spawn.rs:438-539`) and the **ancestor peer book is FIXED at the child's boot**
   (`crates/node/src/rlm_spawn.rs:596-602`, D-RLM-6 🟥), so a child cannot reach an ancestor that restarted
   under a new incarnation.
3. The shard's worldline systems run behind `.run_if(worldline_seeded)` — the `has_synced` run-condition
   idiom verbatim (`crates/sim/src/stub.rs:4318`, scheduled `.after(observe_clock_syncs)`) — so a pre-adopt
   shard **AUTHORS NOTHING**. Wrong state is structurally impossible, not merely unlikely.
4. On adopt the shard calls `evaluate(&baseline, &devs, now)` — the **same** `vd-core` function the
   dashboard calls — and materialises: subjects as ECS entities; NPCs inside the materialisation AoI band
   at their exact closed-form pose **and velocity**.
5. Thereafter **LAW-WL-1** governs: the live shard's per-tick delta is `F(t) − F(t−1)`. The hopper stays at
   B (idle) until the player drains it; the drain is a deviation, durable before ack, and the subject
   re-bases.

**Latency against RLM `boot_ticks` — and revision 1 GATED THE WRONG QUANTITY BY FIVE ORDERS OF MAGNITUDE.**
`WORLDLINE_WAKE_BUDGET_US` (~300 µs) defends `evaluate` + scan + decode, which is a fine **component** check.
But the wake path a player experiences is dominated by **process boot**, and the repo states the figure itself:
`RlmTuning::cloud_with_boot`'s doc records "a slow real fork (a ~3 s CI boot)" and floors
`launch_ttl = max(hz·3, boot_ticks_p99 + settle)` (`crates/sim/src/rlm.rs:92-108`). End to end:
```
demand detect (≤1 reconcile_interval)
+ write-ahead fsync (~25 ms for a 5-deep chain, crates/node/src/rlm_spawn.rs:438-539)
+ child boot ~3 s (parallel across the chain — `launch()` forks and returns)
+ adopt push RTT + first authored frame
+ the mandated 100–150 ms client interpolation buffer
≈ 3.2 s
```
**Two budgets, not one** (§10): the µs-scale component budget survives, and a new **G-WL-WAKE-E2E** measures
**demand → first authored frame p99** through the real `ProcLaunchBackend` over a 5-deep chain. That second
number is the one SEAMLESS depends on.

**⚠ AND THE AoI LEAD DISTANCE WAS NEVER DERIVED — for small realms it is geometrically insufficient.** SEAMLESS
requires the realm to be live before the player can see it, i.e.
`spin_up_r_m ≥ render_extent + v_rel_max · T_wake_p99`. At `T_wake_p99 ≈ 3.2 s`:

| Player speed | Required lead |
|---|---|
| 1.4 m/s walking | 4.5 m |
| 300 m/s atmospheric flight | **960 m** |
| 1 km/s in-system cruise | **3.2 km** |
| 10⁷ m/s quantum | 3.2×10⁷ m |

For large realms this is free (`planet_soi(1.496e11, 5.972e24, 1.989e30) ≈ 9.2×10⁸ m`,
`crates/core/src/geometry.rs:1452`). For **small** realms it is not: a 200 m Area realm needs 960 m of lead at
300 m/s — **larger than its own extent** — and `spin_up_r_m` is derived from the child's extent by the
generator's `spin_up_factor` (`crates/core/src/worldgen.rs:186-193`, `:862-876`). So for Areas and Stations at
flight or combat speed the band cannot cover boot, and "raise `spin_up_factor`" only multiplies the wake rate.

> **DERIVE THE RADIUS, DO NOT TUNE IT.** The generator must **require**
> `spin_up_r_m ≥ render_extent + v_rel_max · T_wake_p99`, feeding the **measured** `T_wake_p99` into
> `AoiConfig::for_velocity_safe`'s `dt` (`crates/core/src/geometry.rs:783-819`) so the existing type invariant
> does the work, and **fail LOUD at generator build** when a realm's extent-derived radius cannot satisfy it.
> That surfaces "this realm is too small to wake seamlessly at this speed" at **config time**, not in play —
> and it is the concrete reason D-RLM-4's warm pool is a prerequisite of the small-realm case (§5.4).

`WORLDLINE_WAKE_BUDGET_US` is asserted **release-only** at `max_subjects_per_realm` + `max_pending_devs` using
`vd_harness::latency::percentile_unstable` (`crates/harness/src/latency.rs:22`, which is TOTAL and returns
`ZERO` on an empty set — so **pair it with a delivery/count floor**, exactly as its own doc warns). It must
fit inside RLM's `boot_ticks_p99`, which floors `launch_ttl` and therefore `min_dwell_ticks`
(`crates/sim/src/rlm.rs:129`, `:477`). ⚠ **`boot_ticks_p99` is TWO different knobs sharing one env var and
BOTH default to 0**: in `RlmTuning::cloud_with_boot` it floors `launch_ttl`/`min_dwell`
(`crates/sim/src/rlm.rs:99-132`), and in `StubConfig` it is the predictive-AoI look-ahead horizon
(`crates/sim/src/stub.rs:4392` reads `config.boot_ticks_p99` into `horizon_s`). **Both protections are off
by default**, so every latency and no-pop gate must set them explicitly or it measures nothing — and
**`boot_ticks_p99` needs a non-zero DERIVED default read by BOTH consumers from ONE named source**, since a
zero default silently disables the launch-TTL floor *and* the predictive look-ahead at the same time.

### 5.6 THE SEAMLESSNESS ARGUMENT, and the parent-authored aggregate answer

**Why the returning player sees a coherent changed system with no pop.**

*Coherent and changed:* a refinery that filled its hopper and stopped (23 days of output waiting, then 7
days idle — a genuine consequence of a cap they chose); a deposit down by exactly the ore the refinery ate;
a population that GREW because they built the refinery; the same foreman where his routine says he should
be. Every one of those is a causal consequence of a decision the player made, computed **exactly**.

*No pop, for two independent structural reasons:*
1. The closed-form pose is **continuous in `t`**, so a materialised NPC appears at exactly the position AND
   velocity its own closed form dictates, **already moving correctly** — it does not spawn at a spawn point.
2. The materialisation band is an `AoiConfig`, whose only live constructor `for_velocity_safe` widens the
   dead zone by `|v_rel| · dt · (K_SAFETY + extra)` and is **fallible LOUD** unless `0 < spin_up <
   tear_down` (`crates/core/src/geometry.rs:783-819`), so hysteresis is a **type invariant** rather than a
   runtime check.

⚠ **BUT THE FIELD POPS, AND REVISION 1'S ARGUMENT COVERED ONLY NPCs.** On adopt, `evaluate` materialises
machines and constructions as ECS entities **all at once** — and if the player is already inside the render
extent when the realm boots (which, per the AoI derivation above, is exactly the small-realm case), those meshes
appear at close range. **G-WL-SEAMLESS-FIELD** (§10) therefore asserts that **no materialised subject's
first-visible frame occurs inside the render extent**, with the negative cell at look-ahead 0. The mechanism is
the same derived radius: the realm must be live, adopted and authoring before its contents enter the render
extent, which is precisely what the fail-loud generator requirement guarantees.

**⚠ THE PERCEPTIBLE-RANGE BAND, stated as a design property because it determines whether the substrate is
worth its cost.** Everything this design buys lands between roughly **one day and one month** of dormancy
(§1.2). Both ends are attacked here rather than left implicit:
- **Short end (a 5-minute round trip).** A realm is reapable within about a minute of the last occupant leaving
  (`min_dwell = spinup_cooldown + launch_ttl`, `crates/sim/src/rlm.rs:129`, plus `empty_grace` and
  `teardown_drain`), so a 5-minute trip pays the full ~3.2 s wake for a 6 000-tick delta — 150 input consumed
  against a 500 000-unit hopper, **0.015 %**, imperceptible. The AoI distance hysteresis
  (`tear_down_r > spin_up_r`) does not help a player who leaves the system and returns. **Fix: a per-realm
  `min_dormant_ticks` — a hysteresis in TIME, not distance — below which an empty realm is not reaped.** It is
  legal under LAW-WL-7 (a lifecycle-internal window, not an economy or life value), it is a one-field addition
  to `RlmTuning`, and D-RLM-4's warm pool is the alternative (make the wake cheap instead of preventing the
  reap). W-2's HR6 line gains a `vdctl` fixture for the **5-minute** case, not only the 30-day one, so the
  cheap-but-useless wake is visible in testing.
- **Long end:** §4.3a's seasonal driver plus `RespawnAfter`-by-default; beyond that, W5.

⚠ **THE RADIUS CONVENTION IN CODE IS THE OPPOSITE OF THE DESIGN DOC, VERIFIED.** In code
`spin_up_r_m` is the **INNER** (smaller) radius and `tear_down_r_m > spin_up_r_m` is the **OUTER** release
radius (`crates/core/src/geometry.rs:767-780`, fields private, `for_velocity_safe` rejecting
`!(0 < spin_up < tear_down)`). `scripts/realm_lifecycle_design.md:176-179` says the reverse
("`spin_up_r_m: f64,  // create (outer) edge`"). **Anyone sizing a materialisation radius from the doc
inverts the hysteresis and causes flap.** Fix the doc.

⚠ **RLM IS INERT AT WALK AND CANONICAL SCALE.** `InterestConfig::inert()` (spin-up factor 0 ⇒
`AoiConfig::inert` ⇒ nothing ever in range ⇒ no demand) is used by **both** `walk_scale` and `canonical`
(`crates/core/src/worldgen.rs:968`, `:1026`); only `visual_scale` is live (`:1048-1055`). And the
orchestrator only arms the reconciler under `VD_DEMAND` (`crates/bins/src/bin/orchestrator.rs:357-368`),
which is mutually exclusive with `VD_STATIC_FOREST` and is **not set in k3d**
(`deploy/k3d/30-orch.yaml:50-56`; the shard is a StatefulSet with `replicas: 1` and a hardcoded
`VD_REALM_SEED: "7"`, `deploy/k3d/50-shard.yaml:20-98`). **Every dormancy gate must run the Visual preset
with `VD_DEMAND` set, and there is currently NO cloud-tier evidence for any dormancy claim.**

### 5.7 The parent-authored aggregate (H3): NOT kept, and why

**H3 is rejected on two independent grounds, both verified.**

1. **There is NO shard→shard realm-state carrier to reuse.** `RealmSnap` carries only
   `{realm: RealmId, pose: StampedPose}` (`crates/wire/src/channels.rs:200-204`) on
   `MsgClass::RealmSnapshot`, which is **UNRELIABLE latest-wins** (`crates/sim/src/io/mod.rs:542-546`),
   and the pipeline is shard → **gateway → CLIENT** only, fanned to every Active subscriber with **no
   fence gate**, no per-sub re-tag, `Durability::Ephemeral`
   (`crates/sim/src/stub.rs:4247-4310`; `crates/connection-plane/src/gateway.rs:2001-2035`). **It is a
   RENDER feed, not a coupling.** Widening a frozen row to carry monetary/inventory state on a lossy
   channel is precisely the `DockState.clamped` mistake this architecture already made and deleted
   (`docs/design/sealed_shards.md:362`).
2. **It is REDUNDANT.** Both a parent and a child compute the *identical* `f` from the shared seed plus
   their own durable rows. Shipping an aggregate spends bytes to transmit something the receiver can
   derive — the exact argument `crates/core/src/worldgen.rs:1-5` already makes for geometry, and the exact
   standing frame-authority law (two shards at a shared boundary use different frames and **neither**
   re-derives the other's value).

Also: `emit_realm_frames` is gated on **four** conditions including ≥1 *emitting observer dot*
(`crates/sim/src/stub.rs:4257-4279`) plus `.run_if(has_synced)`, so a realm with no observer ships
**nothing** — exactly the case dormancy is about.

**FAN-OUT ARITHMETIC, so the rejection is quantified rather than asserted.** A live parent already computes
closed-form state for every direct child every tick via `child_placements`
(`crates/sim/src/stub.rs:670-691`), which makes it superficially the perfect hook. But: it is **POSE-only**,
`O(direct children)` per tick, and hard-capped by `MAX_REGIONS = 64` — a fixed-width `u64` membership
bitset (`crates/sim/src/stub.rs:494-505`, `:552`), boot-guarded at `crates/bins/src/bin/shard.rs:229`. A
galaxy of 10⁴ sibling systems is therefore **literally unbootable** through that path. Even a separate
uncapped roster costs, per Design B's own numbers: `40 children × 500 ns = 20 µs/pass`, amortised over a
12 000-tick period `= 33 ns` per 50 ms tick (0.00007 %); `10⁴ children = 5 ms/pass`, time-sliced to
`10⁴/12 000 ≈ 0.83 children/tick ⇒ 500 ns` worst tick; but `10⁶ children = 500 ms/pass and 171 MB
resident` — **infeasible**, and the `RealmPath`-prefix bucketing that would fix it is, in Design B's own
words, "a design, not a landed pattern".

⚠ **AND `MAX_REGIONS = 64` BINDS *THIS* DESIGN, NOT ONLY H3 — revision 1 cited it as someone else's
constraint.** Verified: `child_placements` selects direct children by filtering `self.regions`
(`crates/sim/src/stub.rs:684-689`) — the **same `Vec`** that `guard_regions_nest(regions, MAX_REGIONS)` rejects
at >64 (`crates/core/src/geometry.rs:1042-1058`, boot-guarded `crates/bins/src/bin/shard.rs:229`) — and
`aoi_decide` consumes `child_placements` (`:4425`), so **RLM demand can only ever be emitted for children inside
that ≤64 set.** With a 6-level lineage that is own(1) + ancestors-incl-root(5) = 6, leaving **≤58 direct
children**. Consequence, computed:

> Universe→Galaxy ≤58 and Galaxy→System ≤58 gives a **MAXIMUM of 3 364 systems in the entire universe.**

§12.1 quotes Design D's density figures at "10⁴ sites / 10⁵ systems". **10⁵ systems per galaxy is short by
~1 724× at ONE level.** So the whole dormancy story — whose premise is "most of the galaxy is off because
nobody visits it" — was being designed for a universe the demand mechanism **cannot address**. The two fixes are
mutually exclusive with other pre-P4 commitments, which is why this must be resolved **before W-0**:
1. **RECOMMENDED — decouple the AoI/demand child ROSTER from the containment REGION set.** The region bitset is
   a *membership* structure for containment; the demand roster is an *addressing* structure. §5.7's own numbers
   above price the uncapped roster (feasible to 10⁴ time-sliced; infeasible at 10⁶ without the `RealmPath`-prefix
   bucketing, which must then be landed rather than cited).
2. Accept an intermediate Sector/Cluster level — but that needs a **7th `RealmKindTag`** and therefore
   overflows any fixed-depth key, which is one of the two reasons §11.1's key is now variable-depth with a
   validated `MAX_REALM_DEPTH`.

Independently: **a boot-time assertion that the generated forest's max branching factor is
`≤ MAX_REGIONS − lineage_depth − 1`, failing LOUD with both numbers named**, so a future generator cannot
silently produce an unbootable galaxy. And the same cap bounds **any** per-direct-child worldline fan-out a
future slice might want (the pinned host's `SiteAgent` roster at D-71, a `child_aggregates`-shaped extension,
per-child materialisation bands): **any per-direct-child worldline iteration must use its own uncapped roster,
never the region bitset** — and W-5's pinned Galaxy host is precisely the realm with the most children, so its
child count must be bounded and gated.

**LAW-E1 answer for H3, had we kept it:** it would have been physical (pop/stock/capacity) and therefore
game-side, with prices as a forward-skipped TLV tag. We keep that *discipline* (§7.4) and drop the
mechanism.

**Design B's genuinely marginal value, and why it is not built:** when nobody is online anywhere,
`ancestor_close` of an empty desired set **is empty** (`crates/sim/src/rlm.rs:498-512`) — so even the
galaxy is reapable and there is no author at any level. Design B says so itself. Its ~6 100 LOC + 2 new
reviewed HR1 arms + a new `LifecycleAction` variant + a new drain phase are therefore **entirely marginal**
over this design, purchasing only sibling coupling and shock propagation for the thin root→player path —
i.e. the player's own neighbourhood, which AoI spins up anyway. **Not built. Its three cheap rules are
grafted (LAW-WL-2/3, LAW-WL-7, the Jacobi discipline) and its mechanism is recorded in §12.**

---

## 6. NPC LIFE AS THE FIRST PEER CONSUMER

NPC life is a **PEER** subsystem on the same substrate, not a client of the economy. Adding it is adding two
`SubjectKind` arms plus their closed forms — not a subsystem.

### 6.1 The shared substrate, exactly

A deposit is `SubjectKind::Deposit`; a refinery `SubjectKind::Machine`; an installed structure
`SubjectKind::Construction`; an anonymous NPC crowd `SubjectKind::Cohort`; a promoted NPC
`SubjectKind::PromotedNpc`. **All five are rows in the same `Baseline`**, advanced by the same `evaluate`,
folded by the same LAW-WL-2 log, compacted by the same orchestrator sweep, and materialised by the same
AoI band. ONE machinery (HR3).

### 6.2 What it publishes

- **Field (zero bytes):** carrying capacity `K` seed-derived from the realm's content (habitable extent,
  construction slots — ⚠ **quantised to an integer at the generator boundary**, §9.1) plus event deviations (a
  station built raises K; destroyed lowers it), and **seasonally modulated** per §4.3a so the target itself
  moves; population `P(Δ) = K − ipow_muldiv(K − P₀, gn, gd, Δ / growth_quantum_ticks)` within one season, and
  the affine-periodic form of §4.3a across seasons; net hazard attrition folded into `gn/gd`; activity mix as an
  **integer** categorical inverse-CDF over a cumulative `u64` threshold table keyed
  `f(realm_seed, cohort, tick / activity_period_ticks)` — the `sample_galaxy_type` shape with `u64`
  thresholds instead of `f64`.

  ⚠ **THE UNDERFLOW HORIZON BELONGS NEXT TO THE LAW**, because it is what anyone tuning `gn/gd` is actually
  choosing: the Q32 gap term floors to **exactly zero** at
  ```
  m* = ln((K − P₀) · 2³²) / −ln(gn/gd)          [quanta]
  ```
  and the *observed* (floored) population reaches exactly `K` earlier still, at `ln(K − P₀)/−ln r`. §5.4 tables
  the values: `r = 0.95` freezes at **21 days**. Without §4.3a's moving `K` this makes every dormant region with
  the same `K` numerically identical, forever. It is part of `G-WL-DORMANT-MOVES`' multi-horizon assertion.
- **Individuals ARE the field's quantiles — no storage.** `NpcId = child_seed(realm_seed, NPC_SALT, index)`
  for `index ∈ 0..P(t)` (`crates/core/src/rng.rs` child-seed helper). The roster is a pure function of the
  field.
- **An NPC's pose is an EPHEMERIS.** `g(NpcId, tick)`: a seed-derived waypoint ring of W points drawn from
  the realm's own seed-derived construction slots, a seed-derived speed and phase; leg index
  `= (tick·speed_num/leg_len + phase) mod (W·leg_ticks)`; position = an integer lerp between two waypoints
  in the `LatticePos` domain; velocity = leg direction × speed. All integer.
- **`unrest_bp` (Design D's graft):** unmet needs raise unrest; unrest biases the optional faction tier's
  structural decisions and later spawns hostile activity. This is the cleanest mechanism for making a
  dormant-region *aggregate* produce a player-facing *consequence*, and it composes with promotion (an
  unrest-driven raid promotes its participants).

**THE UNIFICATION THAT MAKES NPCs COST NOTHING: an NPC is a moving realm occupant with an ephemeris,
exactly like a planet.** The standing frame-authority law applies UNCHANGED (the containing realm's shard
authors it in its own frame and ships it; the client renders shipped state only; a passive body is the same
law with zero signals), and the existing `child_placements`/`place_child` shape
(`crates/sim/src/stub.rs:663-710`) is the code idiom, **not a new one**. An NPC flying into a planet's SOI
is an ordinary containment re-home through the existing saga.

### 6.3 How GAMEPLAY consumes it (players interacting with individual NPCs)

**Materialisation by AoI, not by realm activation.** NPCs whose closed-form pose is inside the observer's
content-AoI band become real ECS entities with rapier colliders; outside, they are not entities but they
are still in the field and still closed-form. The materialiser is `children_within`-shaped
(`crates/core/src/worldgen.rs:239-256`, the P6/D-9 spatial-index seam with a linear impl over an
already-bounded slice).

Materialised NPCs are **ordinary occupants FOR SIMULATION** — physics, containment, collision, the observer
feed — and they cross shards on the ONE `TransferableKind` registry + saga (HR2). Durable named NPCs are
`Durable`/`LossBudget::ZERO`; ambient crowd NPCs are `Transient` with a budget, exactly as
`DROPPED_BLOCK_DEF` (`LossBudget(2)`), `DEBRIS_DEF` (4) and `PROJECTILE_DEF` (8) are
(`crates/core/src/entity_kind.rs:225-259`, with `is_coherent` at `:183-189` enforcing Durable ⇒ ZERO).
⚠ **They are NOT occupants for LIFECYCLE** — `is_session_occupant()` is false for every NPC kind, which is the
code mechanism LAW-WL-7 needs and which §3.4 specifies in full. Without it one materialised NPC makes its realm
permanently unreapable and NPC positions emit `SpinUp` demands.
⚠ Note the behaviour half of `TransferableKind` **does not exist** (an explicit D-31 INTERIM banner at
`crates/core/src/entity_kind.rs:10-18`; the real producer emits `state: vec![]` at
`crates/node/src/saga_runtime.rs:828`) — NPCs would be its **second** consumer, not its first.

**PROMOTION — the mechanism that makes a pure function survive contact with players.** The instant a player
interacts with an NPC (talks, trades, hires, shoots), that NPC's future stops being closed-form. So the
interaction is a **deviation that PROMOTES the NPC out of the field into the log**: it becomes an
individually-logged `SubjectState`, and the cohort's promoted population is **DERIVED** from the row set —
`promoted_count(c) = |{promoted subjects whose cohort == c}|` — so the field never double-counts it.
⚠ **Revision 1 had `promoted_count` INCREMENT, which is the one non-idempotent operation LAW-WL-2 forbids and
which its own gate contradicted; §4.4 records the defect and the derivation that replaces it.** Named NPCs
(station owners, quest-givers) are **born promoted** — seeded as log subjects at generation time — so the field
carries only anonymous population.

> **PROMOTION IS TRIGGERED ONLY BY AN EXPLICIT PLAYER INTERACTION — never by the materialisation band.**
> ⚠ Revision 1 left this ambiguous, and the ambiguity was a determinism defect: the band decision is entirely
> **float and pacer-dependent** (`AoiConfig`'s fields are `f64`, `crates/core/src/geometry.rs:776-780`; the
> distance is over `DVec3` poses; and the predictive horizon is
> `horizon_s = f64::from(config.boot_ticks_p99) * config.tick_dt_s`, `crates/sim/src/stub.rs:4392`). So a
> shard's `VD_TICK_DT` and float rounding would change **which NPCs get promoted**, i.e. which durable
> deviations exist — defeating both the no-float rule and RULE WL-4 at the exact boundary where the pure field
> becomes authoritative durable state. Separating the two decisions costs nothing: **materialisation** is a
> rendering/interaction affordance and may stay `f64`; **promotion** is durable authority and is triggered by a
> discrete player action only. Gate cell in G-WL-NO-SECONDS. If band-triggered promotion is ever wanted, its
> predicate must be an **integer** distance test in the `LatticePos` domain.

> **DEMOTION: ONLY ON RE-CONVERGENCE, AND ONLY WHILE UNOBSERVED. `promotion_ttl_ticks` IS DELETED.**
> ⚠ Revision 1 demoted "when the logged state re-converges to the closed form, **OR** on
> `promotion_ttl_ticks`" while §10's `G-WL-SEAMLESS-NPC` asserted continuity across demote
> (`|Δpos| ≤ one tick of its own closed-form velocity`, `|Δvel| == 0`). The TTL branch makes that
> **unachievable, not merely risky**: between promotion at `t₁` and TTL demotion at `t₂` the promoted NPC moved
> under live AI/physics to position `X` while its closed-form ephemeris says `Y`, and `X ≠ Y` with an
> **unbounded** gap — a player who walked an NPC across a station for an hour produces an arbitrarily large
> snap. So revision 1 shipped a gate its own policy guaranteed would fail (and §13 Q9 admitted the policy was
> undecided).
>
> The three requirements form a real triangle: {seamless demotion} wants re-convergence-only; {bounded promoted
> state} wants a time-based reaper because re-convergence may never happen; {no hard refusal} rules out
> `WlRefused` as the answer, because refusing a promotion means refusing a player's interaction with an NPC.
> **Pick two, explicitly: DELETE the TTL, and bound promoted state by a NON-TELEPORTING mechanism** — a
> per-realm `max_promoted_subjects` enforced **at promotion time** with **graceful degradation** (promote as a
> `Cohort`-attributed named individual with **no independent trajectory**, so it keeps its identity and its
> dialogue while its motion stays closed-form) rather than a refusal or a snap. Then `G-WL-SEAMLESS-NPC` becomes
> passable and `G-WL-PROMOTION-BOUND` becomes a **budget** test instead of a reaper test — plus a negative cell
> asserting that a TTL-forced demotion of a diverged NPC **FAILS** the seamless gate, so the policy cannot
> silently return.

**This is what bounds `N` forever: promotions are proportional to PLAYER-HOURS, not to world size or
elapsed time, and player-hours are bounded by concurrency.** A galaxy nobody visits for a year accumulates
zero log rows and costs zero bytes.

⚠ **Promotion must be exactly-once**, and there is **no durable exactly-once journal on any shard** today. Two
options: become D-22's first forcing consumer, or be **idempotent by construction** — which LAW-WL-2's absolute
rebase already achieves, at the cost of every promotion message carrying the full absolute state. **Recommend
idempotent-by-construction**; a promoted-NPC state is a few tens of bytes (§5.4), so the message cost is trivial.

⚠ **PROMOTION CREATED TWO AUTHORITIES OVER ONE OBJECT, WITH NO RECONCILIATION AND NO POPULATION
CONSERVATION.** Revision 1 made a promoted NPC simultaneously (a) a realm-keyed `SubjectState` row at the
custodian and (b) a materialised ECS entity that crosses shards on the transfer saga — and **nothing moved the
subject row when the entity re-homed.** Three consequences:
1. After the entity re-homes to realm B, realm A's baseline **still contains the promoted subject**, so a later
   adopt of A **re-materialises it** — duplication with no fence relationship and no CAS.
   `verify_authority_unique` (`crates/harness/src/oracle.rs:121-146`) audits entity **holders** only, so the
   worldline copy is invisible to it and no landed oracle can catch this.
2. `promoted_count` is per-realm, so an NPC walking from A to B must decrement A's and increment B's — a
   **value-bearing population transfer across a sealed boundary with no arm, no fence and no conservation
   identity**. §9.4's declared loss channels do not include population, and
   `G-WL-CONSERVE-ACROSS-LIFECYCLE` is per-realm, so a phantom would read as legitimate.
3. A materialised NPC's `EntityId` is minted shard-locally (`EntityId::pack{kind, mint_shard, seq, rand}`,
   `crates/core/src/ids.rs:151-208`), so the **same** conceptual NPC would get a different `EntityId` on every
   materialisation, and the design gave `NpcId = child_seed(...)` without ever relating the two (D-72 covered
   only the narrative half).

**The three repairs:**
- **v1 RESTRICTION, stated rather than discovered: a PROMOTED subject may not cross a realm boundary.** Its
  materialised entity is confined to its own realm (it is an NPC with a routine inside one realm, which is what
  §6.2's waypoint ring already models). Cross-realm promoted NPCs land only when the subject row's realm
  attribution is moved by **the same transfer saga** that moves the entity, so there is **one commit point** —
  ledgered as **D-80**, gated before it lands.
- **`verify_population_conservation` in `vd-harness`**, a sibling of `verify_authority_unique`:
  `Σ(cohort P + promoted rows + materialised entities)` conserved per scenario across every
  materialise / promote / demote / re-home, with a **RED 1-NPC-imbalance control**.
- **`NpcId ↔ EntityId` is a DETERMINISTIC DERIVATION, not a fresh mint**: the entity's identity is derived from
  the stable `NpcId` so successive materialisations of one NPC are the same entity to every oracle and to the
  client. (Ambient crowd NPCs that are Transient-only and can never be promoted are exempt, and that exemption
  is what keeps the transient loss budgets meaningful.)

### 6.4 How the ECONOMY consumes it (population/activity as a demand driver)

- Population and activity mix **are** the labour supply and the demand field. The ambient price for a
  commodity in a dormant region is `f(own-realm stock, own-realm production rate, own-realm population,
  SEED-DERIVED distance)` — a pure read of the physical field, needing no state and no book. That realises the
  report's §7.13 regional-divergence mechanism for free, and means **every dormant region has a meaningful price
  with nothing running**.
- Conversely an installed construction raises `K`, so a refinery literally grows the town. Already a
  deviation, so the coupling needs no new mechanism.

> ⚠ **RULE WL-READ (THE HR1 READ BOUNDARY) — revision 1 asserted HR1 in §7.3 and then relied on its negation
> here.** Revision 1's ambient price was `f(stock, production rate, population, distance)`, and `distance`
> implies **neighbour** regions — whose `stock` is free only for the *pure-seed* part. The **deviation** part
> (a player mined that deposit out) lives in the neighbour realm's durable rows at the custodian, and **there is
> no cross-shard QUERY in this architecture**. §6.5 point 2 repeated the assumption ("any subsystem can be a
> reader of the whole physical history"). §7.2 #7 handles the sibling case honestly for routes and recommends
> letting realms disagree; §6.4 and §6.5 quietly did not. Normatively:
>
> **A live shard may read ONLY (a) its own realm's baseline + deviations, and (b) the PURE-SEED content of any
> realm. Any neighbour quantity that depends on a DEVIATION must arrive via a reviewed arm (the ledgered D-70
> sibling fan) or be read at Tier-B — it may never be assumed derivable.**
>
> So the ambient price is **own-realm-and-seed only**, and the resulting boundary divergence is the *same
> accepted cost* §7.2 #7 already declares, stated once rather than in two contradictory places. Gate: the
> G-WL-NO-SECONDS source assertion is extended to a **read-boundary source assertion** — no worldline symbol
> reaches another realm's deviation rows.

### 6.5 Explicitly usable by OTHER future systems

The user named "maybe some other systems of the game". ⚠ **Revision 1 claimed genericity over a list that
conflates THREE different shapes, only one of which this substrate serves — and a future designer following that
list would go down the wrong road.** Checked against the shape actually specified (a per-realm baseline of
per-subject scalars advanced by an integer rate):

| Shape | Examples | Does this substrate serve it? |
|---|---|---|
| **(i) Realm-local scalar stocks / cohorts** | wildlife, structure decay + upkeep condition, contraband heat, terrain overgrowth, debris density, traffic density, ambient-ship counts | **YES, natively** — a new `SubjectKind` arm plus its closed form; zero new machinery, zero new store family, zero new gate infrastructure |
| **(ii) SPATIAL FIELDS** | **weather** | **NO — and it needs nothing.** Weather is `f(seed, tick, position)` with **no state**, and it must be continuous **across** realm boundaries. `SubjectState` has no spatial dimension. Weather is a stateless closed form that needs no substrate; revision 1's claim here is **withdrawn** |
| **(iii) CROSS-REALM NON-SPATIAL entities** | factions, wars, politics, **reputation**, corporations, currencies, alliances, multi-realm quest chains, trade routes | **NO — and there is no addressing scheme for it anywhere today.** Every subject is realm-keyed; report D2 already established there is no way to spawn or protect a non-spatial node. Revision 1's only gesture at this was the optional pinned `FactionAgent` |

**Shape (iii) is a real gap and it is stated as one, not implied away.** The candidate answers, none of them
free: an **owning-realm-of-record + fence rule** (a faction's authoritative row lives in one named realm, others
read it only at Tier-B or via a reviewed arm); the report §6.4 option C **ancestor-escalation** shape (state
lives at the LCA of its members' realms, which under LF-1 is at-least-Dormant); or the **static pin** (W6),
which is the only mechanism that exists today. **Ledgered as D-81 and BLOCKING for factions, corporations and
reputation** — i.e. the systems the user named — with the decision owed before any of them is designed.

The two properties that *are* generic, corrected:
1. **`SubjectKind` is an open registry of closed forms** for shape (i) above, not an economy taxonomy.
2. **`WorldFact` is a plain `pub enum` in `vd-core` — closed to outside variants because that is what a Rust
   enum is, not because of any seal** (⚠ revision 1 called it "a sealed but OPEN-ARMED enum", which is a
   contradiction *and* a category error: the `sealed::Sealed` token is a **private module in `vd-sim`**, so a
   `vd-core` type cannot implement it, §1.1 E1-c). Any subsystem may be a **reader** of the fact stream without
   any producer knowing about it — subject to RULE WL-READ (§6.4): a reader in a live shard sees its own realm;
   a reader that wants the whole history is a Tier-B reader.
3. **Aggregate → individuals is one pattern** (field → AoI-materialised entity → promotion on interaction),
   reusable by any "many things, few observed" system in shape (i).

⚠ **What this must NOT be built on today.** `Signal` is **not** a landed `InterShardFlow` arm: the enum has
**23** arms and the header lists `BlockEdit` (P6), `Coupling` (P8) and `Signal` (P9) as **RESERVED —
prose-only** (`crates/wire/src/intershard.rs:9-33`). And `signal_relay` is carried by `profiles::galaxy()`
and `profiles::station()` **only** (`crates/sim/src/capability.rs:176`, `:230`; the tests assert
`!ship.signal_relay()` at `:294` and "a System coord's profile has no signal_relay" at `:443`). So NPC
life reaching other systems via P9 is a **dependency**, not a foundation: this design requires **nothing**
from Signal and works with zero Signal support.

### 6.6 WHAT THE PLAYER ACTUALLY DOES (new in revision 2)

⚠ **Revision 1 was thin here in exactly the four cases the design is judged on**, and W-3's HR6 criterion
("walk up to an NPC") was not a buildable spec without it. Each row below becomes an HR6 criterion (§11.4) and a
`vdctl` verb (§11.6).

| Subject | Interaction verbs | Read surface | Write surface | Promotion / durability |
|---|---|---|---|---|
| **`Cohort` NPC (anonymous)** | none directly — they are ambient; approaching one **materialises** it | shipped pose only | — | interacting promotes it (below) |
| **`PromotedNpc` (named or promoted)** | `talk` (dialogue), `trade` (opens a venue view if the realm has one), `hire`, `attack` — at a per-kind `interaction_range_m` | name, role, disposition, `unrest_bp`-derived mood; all **server-authored, client renders shipped state only** | dialogue choice; a trade/hire is an ordinary game action that emits a `WorldFact` | promoted on the FIRST such verb; **never demoted while observed** (§6.3) |
| **`Machine` (refinery/factory)** | `inspect`, `drain` (take the hopper), `set_recipe` (from the GAME recipe registry, §2 row 11), `power_on/off` | current stock, hopper level, **effective rate** (the last settled rate, RULE WL-SETTLED-RATE), the season multiplier in force, and `Δ_dry`/`Δ_full` as human-facing "runs dry in ~" / "full in ~" | `drain` is a deviation (absolute rebase, optimistically acked); `set_recipe` is a **rate settle** (LAW-WL-5) | the installation is ONE subject (RULE WL-AGGREGATE) |
| **`Deposit`** | `mine` (the ordinary voxel action) | the four visible states below | voxel edits; the scalar rebase is DERIVED from them (§7.6) | — |
| **`Construction`** | `inspect`, `repair`, `decommission` | condition (a draining stock, §2 row 27), capacity contribution to `K` | repair/decommission are deviations | — |

**A deposit's four player-visible states, which §7.6's aggregation rule makes coherent:**

| Rock present? | Scalar stock | What the player sees | Legal? |
|---|---|---|---|
| yes | > 0 | ore in the wall, and mining yields | **the normal state** |
| no | 0 | mined-out void; the vein is visibly exhausted | **the normal end state** |
| yes | 0 | ⚠ **must be impossible** — the geometry is the authority, so an exhausted scalar with intact rock is a reconciliation FAULT (§7.6) | no |
| no | > 0 | ⚠ **must be impossible** for the same reason | no |

**Explicitly stated for v1, so it is a decision and not a discovery: WRECKS AND SALVAGE EXIST ONLY INSIDE AoI.**
`wreck_persistence_ticks` and the `BulkDestroyed` summary (§11.3) are the **live** case. A ship destroyed in a
dormant region leaves no findable wreck, because the substrate advances *state* and not *events* (§1.2). That
removes salvage-in-dormant-space, which is a named end-goal loop — it is one of the three reasons W5 is
load-bearing rather than optional (**W15**).

### 6.7 THE MONETARY LAYER'S OWN DORMANCY (report D3 / D-53) — new in revision 2

⚠ **Revision 1 solved the PHYSICAL layer's dormancy and, for the monetary layer, DELETED three of the report's
mechanisms without replacing them** — S6's `DirectoryKey::Account` (rejected, W13), the KeepAlive-for-open-
obligations lever (forbidden, W14), the dormant catch-up tier (deleted, §4.7) — while never answering report
**D3/D-53**: where does a book, an escrow, a wallet, a dormant realm's **treasury**, an accrued upkeep debit or
an insurance premium live **across a reap**? §2 rows 16–17's answer ("wallet UI unavailable / no venue") is the
economy-**OFF** case, not the realm-**DORMANT** case, and revision 1 silently conflated the two. "Accounts get
single-writer discipline from whichever realm hosts them" (§8.4) is precisely the unanswered question when that
realm is reaped.

**The substrate answers it almost for free, and that is the recommendation:**

> **A wallet, a treasury and a settled escrow POSITION are ZERO-RATE worldline subjects.** A balance is a
> stock with `TickRate { num: 0, den: 1 }`: it does not advance while dormant (which is correct — money does not
> grow by itself, and any interest is a *deviation* posted by `vd-econ` per report D23(b)), it is
> `LossBudget::ZERO`, it is custodied identically, it rehydrates on adopt through the same `WorldlineAdopt`, it
> inherits fence discipline and the stale-reject, and it is covered by the same conservation oracle. `SubjectKind`
> gains one arm (`Balance`) and **nothing else changes**.
>
> This is what makes report **D3 answerable without option (A)'s blocked storage prerequisite**: the account's
> durable position rides the custodian, not a realm's ordinal-bound PVC, so a reaped realm strands nothing.
> Live-book state (resting orders, in-flight matches) is still `vd-econ`'s own and still needs D-53's
> byte-identical rehydration — the substrate does **not** claim to solve that, and the split is exactly RULE
> WL-LIEN's (§2 row 28): **positions are physical and game-custodied; obligations are monetary and econ-owned.**

**What remains OPEN, stated rather than implied:** live order-book rehydration at `max_orders_per_book`, and
`expiry_universe_tick` honoured across a down period, are still report D3/D-53's problem. What this design
supplies to them is (a) a custodied position that cannot be stranded, (b) the fence-ordered absolute-rebase log
they can copy, and (c) `min_dormant_ticks` (§5.6) as the *legal* way to keep a busy venue warm — a time
hysteresis charged to the GAME's lifecycle tuning, never a KeepAlive lever charged to an obligation, which is
what W14 forbids and why the distinction matters.

---

## 7. CROSS-SHARD

### 7.1 First, the taxonomy correction — the prompt and the design prose are both wrong

**Verified:** `InterShardFlow` has **23** arms (`crates/wire/src/intershard.rs:116-258`; the header's own
"LANDED (16 arms)" note at `:9` is **STALE**), and **none** of them is `Signal`, `Coupling` or `BlockEdit`
— those three are prose reservations (`:28-30`). `EffectFree` is real but lives in **`vd-sim`**
(`crates/sim/src/coupling.rs:16-42`), not `vd-wire`, and `CouplingPort` does not exist yet. **There is no
reserved slot to piggyback on.** A designer planning to "ride the reserved Signal arm" is planning on
nothing.

Adding an arm is a genuinely gated event: `effect_class()` and `durability_class()` are **exhaustive
matches** (`crates/wire/src/intershard.rs:305-330` and its sibling), so an unclassified arm **does not
compile** (the `g_sealed_effect_classes` conformance test). The FIRE-AND-FORGET contract explicitly forbids
carrying "a transfer trigger or authority-gating discrete state" (`:31-33`).

### 7.2 Every interaction, assigned

| # | Interaction | Arm | Classification |
|---|---|---|---|
| 1 | **Advance a dormant child's worldline** | **NONE — zero bytes** | — |
| 2 | Player mines / installs / destroys ⇒ a durable deviation | fold into **`BlockEdit`** (P6, reserved) as its durable-journal half | `SideEffecting{TransferId, step_id}`, `ReDriven` |
| 3 | Non-block deviation (NPC promotion, cohort rebase, balance rebase) | new **`WorldlineDeviation`** arm | `SideEffecting{TransferStep{transfer, step_id}}` where `step_id` is the custodian-minted `seq` — ⚠ **NOT a "subject fence", see below**. `durability_class`: **`ProducerLessReliable`** unless a named re-driver exists; `LossBudget::ZERO` |
| 4 | Orchestrator hands a spun-up shard its initial condition | new **`WorldlineAdopt`** arm (orch→shard) | `SideEffecting{FencedKey{realm_fence}}`, Reliable `MsgClass::Saga`. **`ReDriven` only because of the level-triggered `WorldlineSeeded` report below** — as revision 1 specified it, it was a one-shot mislabelled `ReDriven` |
| 5 | Compaction | NONE — custodian-local, **lazy-on-adopt** (§4.5) | — |
| 6 | Dashboard / `vdctl` reads a dormant realm | a **new scan-only `WorldlineRead` port** implemented in io-prod over its own redb read txn (§4.8) — ⚠ **NOT "none": revision 1 billed an unbuilt slice at zero** | Tier-B, off the wire taxonomy, but a real seam with a row budget and a latency budget |
| 7 | Cross-realm NPC supply route | **NONE, and dormant cross-realm flow is ZERO for v1** — see below | — |
| 8 | Ambient price digest to a neighbour (monetary, later) | a **TLV tag inside an existing blob**, never a new field on the frozen `TransferEnvelope` | inherits the enclosing arm's class |
| 9 | Realm spin-up/down demand | existing `RealmDemand`, **unchanged** | `SideEffecting{FencedKey{parent_fence}}`, `ReDriven` |
| 10 | Journal publication (optional) | **NOT an `InterShardFlow` arm** — an in-process **bin-layer drain in the shard binary** handed to io-prod, then custodian-owned segments (§4.6); never a wire arm from the shard's perspective | Tier-B, always sheddable |
| 11 | Client delivery of wallet/market state (monetary) | **has NO carrier today; this design does not invent one** | — |
| 12 | Shard reports it is seeded (so adopt is level-triggered) ⚠ *new* | new **`WorldlineSeeded{realm, base_tick, base_seq}`** (shard→orch), or an added field on an existing shard→orch report | `EffectFree`-shaped level-triggered fact, `ReDriven` **by construction** — this is what makes #4 legitimately `ReDriven` |
| 13 | Economy issues a world-changing command ⚠ *new* | **NONE cross-shard** — `EconCommand` is drained in-process by the shard the economy is co-resident with, and its EFFECT is an ordinary deviation on #2/#3 | see §3.2; the command never crosses a shard boundary itself |

**#1 is the headline: the dormant advance needs ZERO cross-shard bytes.** Both parties compute the
identical `f` from the shared seed plus their own durable rows.

⚠ **#3's IDEMPOTENCY KEY DID NOT EXIST, AND THE ONLY AVAILABLE FALLBACK SILENTLY DROPS DEVIATIONS.** Revision 1
classified `WorldlineDeviation` as "`SideEffecting` keyed on the **subject fence**". **There is no subject
fence.** Fences are authority tokens over **directory-keyed** subjects, and `DirectoryKey` is a closed 4-arm
enum (Session/Entity/Realm/Ship, `crates/wire/src/seams/directory.rs:17-26`) consumed by an exhaustive
wildcard-free match — and §8's W13 correctly **refuses** to add an economy-adjacent arm. A deposit, cohort,
machine or balance therefore has no fence of its own. The only fence available is the **realm** fence, and
`IdempotencyKey::FencedKey` has dedupe semantics that make that fatal: it is documented "idempotent by fence
comparison (lease grants/revokes)" (`crates/wire/src/intershard.rs:282-283`), and `RealmDemand`'s own
classification comment states the consequence outright — "a redelivered demand for the same fenced parent is a
**covered no-op**" (`:441-445`). So **N distinct deviations emitted under one unchanged realm fence would
collapse to ONE applied deviation and the rest would be silently no-oped** — a value-destroying bug
`G-WL-ORDER` (which tests the fold, not the arm's dedupe) would not catch.
**Fix: key the arm on `IdempotencyKey::TransferStep{ transfer, step_id }` with `step_id` = the custodian-minted
`seq` (§4.4)** — or add a reviewed `IdempotencyKey` kind in the same diff as the arm. State the key in the
design, not "the subject fence", and add the `intershard_closed.rs` marker-test cell for it.

⚠ **BOTH NEW ARMS' DURABILITY CLASS WAS UNSTATED, AND `WorldlineAdopt` HAD NO RE-DRIVER — GIVING A
SPAWN→REAP→RESPAWN LIVELOCK, NOT A SELF-HEAL.** `crates/wire/src/intershard.rs:296-306` defines exactly the trap:
`ReDriven` means a producer re-asserts level-triggered every tick; `ProducerLessReliable` is a one-shot whose
push site **MUST** carry `Durability::Retained` or it is silently lost on a source crash, and the classifier
exists to force that decision. Revision 1 labelled both arms `ReDriven` and named **no producer**. For
`WorldlineAdopt` (§5.5 step 2: "immediately after launch, the orchestrator PUSHES") the orchestrator has no ack
and no view of the shard's `worldline_seeded` flag, so it **cannot re-derive** "this shard still needs its
baseline" — it is a one-shot in fact. A crash in that window leaves a live shard **permanently unseeded**;
because every worldline system is behind `.run_if(worldline_seeded)` it authors **nothing**; `aoi_decide` then
self-reports `DemandVerb::Empty` (`crates/sim/src/stub.rs:4410-4421`) so the reconciler reaps it and demand
respawns it — **a loop**.
**Fix, using the idiom already proven for `RealmDemand`: make adopt LEVEL-TRIGGERED off a shard-emitted report
(#12).** The shard reports `WorldlineSeeded{realm, base_tick, base_seq}` (or its absence) every tick, the
orchestrator re-pushes adopt while unseeded, and the ledger fold is a max/latch so it is order-independent
(`crates/sim/src/rlm.rs:266-282`). For `WorldlineDeviation`: if it stays a one-shot it is
**`ProducerLessReliable`** and its push site needs `Durability::Retained` plus the marker test. Add an
**`unseeded_live_realms` DevState counter** so the livelock is observable rather than mysterious.

⚠ **#7 WAS PRESENTED AS ELEGANT AND IS ALGEBRAICALLY IMPOSSIBLE — the honest version follows.** Revision 1 said
a refinery in realm A fed by a mine in realm B costs "NONE for v1" because "an NPC hauler's route is
seed-derived, so both realms independently evaluate the SAME route function and agree on the flow with zero
bytes", and scoped the owed work (D-70) as a **wire** problem ("fan arm #3 to the affected sibling"). Both are
wrong, and the second is the dangerous one, because the user would price a wire arm and receive an intractable
math problem. Working the algebra:
- A **constant** seed-derived unidirectional flow *does* compose in closed form: `S_A(t) = S_A0 + (i − r)·t`
  stays piecewise-linear.
- It **stops composing** the instant the flow's rate changes at a breakpoint owned by **another realm's**
  deviation log — e.g. B's deposit dries at a `Δ_dry` that depends on B's own player history. Then A's closed
  form requires B's **breakpoint set**, HR1 forbids the query (RULE WL-READ, §6.4), and the composed system has
  `O(K_A + K_B)` breakpoints requiring exactly the `PwlCurve` merge deferred as D-69 — **re-merged on every
  deviation on either side**.
- **Bidirectional** coupling (A supplies B *and* B supplies A) is a mutual recurrence with **no closed form at
  all** — the same class as `FieldReplenish`, which §4.3 **rejects** on precisely these grounds. So revision 1
  rejected intra-realm coupled dynamics as intractable and then assumed inter-realm coupled dynamics was free.
- The concrete unsafety: if A evaluates an **assumed inflow** while B's source is exhausted, **A MANUFACTURES
  MATERIAL FROM NOTHING** — and `G-WL-CONSERVE-ACROSS-LIFECYCLE` cannot see it, because it is per-realm and the
  phantom appears in A's own conserved sum as a legitimate inflow.

**The v1 rule, stated as a gameplay constraint the user must see rather than a fact discovered at P8:**

> **A DORMANT MACHINE DRAWS EXCLUSIVELY FROM STOCKS INSIDE ITS OWN REALM. The only legal dormant cross-realm
> flow is ZERO**, enforced by a **registry validator**: no `SubjectState` may name an out-of-realm source.
> Combined with §4.3's world rule ("a machine draws from a STOCK, never another machine's output; an
> intermediate stock is always materialised, so hauling is mandatory") and the fact that no hauling occurs while
> dormant, the honest consequence is: **every dormant multi-realm supply chain halts at the first realm
> boundary, and all dormant production is strictly local.** LAW-E2 names "Ports, refineries, space stations" —
> **ports are exactly the cross-realm objects, and they cannot function dormant in v1.**

**Gate: G-WL-NO-PHANTOM-INFLOW** — a two-realm fixture where B's source is exhausted while both are dormant,
asserting A's evaluated stock did **not** increase, with a RED cell in which a deliberately-assumed inflow
FAILS. **D-70 is re-scoped honestly**: not "fan an arm to a sibling" but *"solve or bound a coupled
piecewise-linear system across sealed authority domains"*, inheriting W3(c)'s rejected shape (a bounded quantum
loop with a `max_quanta` cliff). And this is the **second independent reason W5 is required rather than
optional** (§12.2): the coarse agent tier is the only mechanism in the field that can move material between
dormant realms at all. Note also that the boundary flow is an **unconserved estimate** in v1 — it may not feed
any conservation identity or gate, and the seed-derived route function must be the single source both realms
evaluate, so the disagreement is bounded to player-caused deltas only. That is a **declared approximation with a
bound**, not (as revision 1 framed it) an application of the frame-authority law: that law permits disagreement
about a *representation*, never about a *conserved quantity*.

### 7.3 What genuinely needs a NEW arm, and why nothing else does

**`WorldlineAdopt` is unavoidable.** Checked one by one: `spawn_realm` has no payload room
(`crates/sim/src/io/mod.rs:436-479`); `RealmDemand` is a frozen positional-postcard 4-field record with the
parent coord deliberately *not* on it (`crates/wire/src/intershard.rs:517-545`); `ReHomeState`'s only arm
today is `PoseOnly` with `Snapshot(Vec<u8>)` deferred to P7; and there is **no cross-shard QUERY** in this
architecture by design. Three ways in, one acceptable:
- env payload on the child (the `VD_PEERS` precedent, `crates/bins/src/proc_launch.rs:122-136`) —
  **rejected**: an unbounded ~5 kB blob in an env var;
- widen `spawn_realm` — **rejected**: it is the frozen sealed port, object-safe and F2-monotone;
- **push after launch, behind a `worldline_seeded` run-condition** — **RECOMMENDED**. It reuses the proven
  `has_synced` gating idiom so a pre-adopt shard authors nothing, is naturally fence-stamped, is
  re-drivable, and **generalises**: P4 terrain will want the same boot-payload shape, so shaping it ONCE
  here is HR3-positive.

**`WorldlineDeviation` should be MERGED into `BlockEdit` where it can be.** Most deviations *are* block
edits (a mined voxel, a placed refinery, a destroyed station), and `BlockEdit` is already the reserved P6
arm — so this design's dominant cross-shard need is **already scheduled**, and folding the durable-journal
half into `BlockEdit`'s design is strictly better than a 24th arm. A dedicated arm is needed only for
non-block deviations. ⚠ `Signal` is **not** a candidate carrier: a deviation is authority-gating durable
state, which the fire-and-forget contract forbids (`crates/wire/src/intershard.rs:31-33`).

**Net cost: 1–2 new reviewed arms** (versus 2 for Design B, ≥3 for Design D once its unspecified
"how does the CAT learn a child went live" arm is counted, and 0 for Design C at the price of a third
unreviewed byte channel plus a required Warehouse).

⚠ **THREE hard budget rules on any blob** (revision 1 had two, and missed the one that fires on a routine
deploy).

**(a) Enforce budgets at MUTATION time as a typed refusal, never discover them during a crossing.** An oversize
*required* TLV tag is `MissingRequiredTag`, documented "refuse the transfer, never decode to Default"
(`crates/core/src/tlv.rs:58`), and the version floor refuses at PREPARE — so exceeding a budget silently
converts "this realm got rich" into "this realm can never be adopted", or "a rich player" into "a player who
can never re-home". ⚠ **And the BINDING cap is far tighter than the frame cap revision 1 quoted.**
`MAX_FIELD_BYTES = 1 << 20` (`crates/core/src/tlv.rs:43`) matches `MAX_STREAM_FRAME_BYTES = 1 << 20`
(`crates/wire/src/framing.rs:17`), but the **per-kind** hard cap is `KindDef::max_state_bytes: u16`
(`crates/core/src/entity_kind.rs:172`) and the landed values are **`PLAYER_DEF = 4096`** (`:204`),
`SHIP_DEF = 8192` (`:213`), `NAMED_CONSTRUCTION_DEF = 8192` (`:222`). So a player's **entire** TLV blob — pose
plus inventory plus everything else — is **4 KiB today**, which at §5.4's measured ~55 B/row bounds §8.3(c)'s
in-blob inventory to a few **tens** of stacks, not hundreds. `max_state_bytes` is the number W8(c) must be
enforced against, named here so nobody sizes an inventory from the 1 MiB frame cap.

**(b) `WorldEpoch` NEVER GOES ON A TRANSFER BLOB'S REQUIRED TAGS — otherwise an accounting-config change
becomes a MOVEMENT OUTAGE.** ⚠ This composition was invisible in revision 1 and it is worse than the size
hazard, because it needs no rich player and fires on a routine rebalance. §8.3(c) (RECOMMENDED) puts item stacks
inside the container's Durable TLV blob. LAW-WL-5 (§9.2) stamps `WorldEpoch: u32` on **every** durable record and
mandates that "a reader that cannot represent an epoch returns `Refused`", explicitly mirroring
`MissingRequiredTag`. Compose the two: during a **rolling deploy**, or after **any rate/capacity rebalance** that
bumps `WorldEpoch`, a player whose blob carries a newer-epoch inventory tag **cannot be adopted by an
older-epoch dest shard** — the transfer is REFUSED at PREPARE, and re-home **IS** the containment/docking
primitive (`crates/sim/src/stub.rs:9315-9326`, `:11282-11307`). **The player cannot move between realms, cannot
dock, cannot undock** — the accounting layer's versioning routed straight into the physics/movement path, which
§2 row 13 forbids in every other form. Normatively:

> `WorldEpoch` belongs on **WORLDLINE records only**. Items ride an **OPTIONAL, forward-skippable** TLV tag (the
> framing already forward-skips unknown tags by length), and an epoch mismatch **degrades to "carry the item
> bytes verbatim, opaque"** — **never** `Refused`, because refusing blocks movement. Gate:
> **G-WL-SCHEMA-FLOOR cell 5** — a **MIXED-EPOCH cluster must still complete a re-home in both directions.**

**(c) Never append an `EffectFree` price/supply field to `TransferEnvelope`**: **effect class is a property of the
ARM, not of a field**, and `TransferEnvelope` is the frozen contract every transfer carries.

### 7.4 Signal-design inputs (P9), stated as requirements on that design

1. **`signal_relay` is Galaxy + Station ONLY today** (`crates/sim/src/capability.rs:176`, `:230`), and
   `RealmCoord::profile_kind()` → `profile_kind_of` maps **both Universe and Galaxy** to
   `ProfileKind::Galaxy` (`crates/core/src/realm_coord.rs:93-106`), so a Universe root also relays. System,
   Planet, Ship, Asteroid and Area (which maps to `profiles::ship()`) are **false**. Whether
   System/Planet realms need a relay or processing capability at all is an **open requirement on the P9
   design**, not an assumption this design may make.
2. **`ShardProfile` is a PURE function of realm KIND** (`profile_for` is a total, wildcard-free match, so a
   new `ProfileKind` is a hard compile error until mapped). A per-**instance** capability has no home
   today. Consequence for the economy: **S5a alone makes ALL stations venues** — S5b is a real RLM slice,
   confirmed.
3. **The economy's cross-shard needs, fed in once:** price/supply digests are `EffectFree`-shaped and must
   ride a **TLV tag inside a blob**, never a new envelope field; Signals must **never** carry
   authority-gating economic state nor the credit half of an applied debit; a worldline **deviation** is
   never a Signal.
4. **NPCs are the natural first non-block Signal emitters/receivers**, and the sealed `WorldFact` enum
   should be shaped now with that projection in mind (an NPC hauler broadcasting a distress signal is a
   `WorldFact` → Signal projection).

⚠ **A blocking prerequisite this design inherits and cannot fix.** **A demand-spawned shard never receives
`ClockSync`**: `ClockPeers` is a static boot `Vec` exposed only as `Res`, never `ResMut`
(`crates/node/src/orchestrator.rs:60`, `:149`, `:184-213`); the child env sets `VD_PEERS` but nothing
registers clock membership (`crates/bins/src/proc_launch.rs:122-136`); and every authoring system is
`.run_if(has_synced)` (`crates/sim/src/stub.rs:4318`, schedule `:1242-1262`). The process gate says so out
loud — children boot "without ever syncing" (`crates/bins/tests/rlm_proc_spawn_smoke.rs:116-121`). Since
this design's state is `f(t)`, a wrong or absent `t` is **fatal** — more fatal than for a design that
accumulates, because there is no accumulated state to fall back on. **It blocks RLM Step 6/7 equally, so it
is a shared dependency rather than an economy cost — but it must be named, or the first wake-up gate will
fail for reasons nobody expects.** Options for the fix (a user/RLM decision, not ours): a runtime
`ClockPeers` mutation on spawn, a broadcast to `spawner.live_nodes()`, or a pull-based clock request.

### 7.5 The rebalance / operator-tweak LOOP, as an operation rather than a rule

⚠ **Revision 1's only tweak machinery was LAW-WL-5's settle-on-change plus one gate cell, which is a RULE, not a
design** — and the end goal explicitly includes "dashboards with operator tweaks". Three things were undesigned:

1. **The MECHANICS and COST of a global rebalance.** Settle-on-change over every *touched* baseline is a **bulk
   write across the entire custodian**. Specified: a **fenced, resumable, idempotent custodian sweep** with a
   **bounded rate** (`rebalance_rows_per_tick`), a **durable cursor** in its own `StoreKey` family (the one place
   a cursor is legitimately needed, since this genuinely is a whole-family operation), a progress counter, and
   the rule that a realm **live at that moment** settles locally on its next `field_refresh_period_ticks` rather
   than being written under it. A player mid-sweep sees the **pre-rebalance** interval, because settle-on-change
   means the prior interval was closed with settled integers and is never re-integrated.
2. **Authz and audit.** The physical layer's rates are now operator-tweakable durable state, so the tweak surface
   **reuses `enforce_cloud_preflight`'s profile discipline** (`crates/io-prod/src/boot.rs:410-470`) and report
   **D-13's admin authn** / **§9.5's authorization of economic commands** — it does **not** invent one. Every
   rebalance is an audited, attributable event with the operator identity and the epoch transition recorded.
3. **A CODE change to a closed FORM across a rolling deploy** — the hazard `WorldEpoch` does **not** cover,
   because it versions the config and not the math. Every closed form carries a **`FORM_VERSION`** participating
   in the same settle-on-change and epoch-floor refusal (§9.2), so two binaries can never integrate one interval
   with two different closed forms.

### 7.6 VOXEL TERRAIN ↔ A `Deposit`'s SCALAR STOCK — the aggregation law (new in revision 2)

⚠ **This was a hole, not an omission, and it sits on the interface to the NEXT TWO PHASES.** The production
integral's `S₀` is `f(seed, realm_path, voxel_pos)` per S12, but the authoritative stock is a **per-realm scalar**
`SubjectKind::Deposit` row, while mining is **thousands of voxel edits per player-minute** — and §11.2 rules that
"a placed BLOCK must NOT be a deviation" (else the custodian is the bottleneck). Revision 1 reconciled the two
nowhere. Unreconciled, either a player mines out a mountain and the scalar never falls (**the refinery keeps
eating ore that is physically gone** — farmable), or the scalar hits zero while intact rock remains (a
seamlessness/believability break); and any *batched flush* of accumulated voxel yield breaks LAW-WL-1's
`F(t) − F(t−1)` exactness, which is the one property the flagship gate proves. P4 fixes the geometry irreversibly,
so this is decided **now**.

> **LAW-WL-8 (GEOMETRY IS THE AUTHORITY; THE SCALAR IS DERIVED).**
> 1. **The VOXEL EDIT LOG is the authority for geometry**, and it is Category-C shard state the worldline never
>    sees (the same placement as the block graph under RULE WL-AGGREGATE).
> 2. **The live shard keeps a per-deposit ACCUMULATOR** whose **only durable expression is an absolute
>    `RebaseSubject`** — never a delta, never a batched increment.
> 3. **It is emitted at DECLARED BREAKPOINTS ONLY**: deposit exhaustion, a `field_refresh_period_ticks`
>    boundary, and the spin-down drain window. Between breakpoints the scalar's value **is** `evaluate`'s, so
>    LAW-WL-1 holds exactly: the rebase does not *add* to the closed form, it **replaces** its base — which is
>    precisely what absolute rebase is for.
> 4. **The rebase is DERIVED FROM the voxel log, never the reverse.** A scalar may not authorise the existence of
>    rock, and rock may not be re-created by a scalar rebase.
>
> **Reconciliation identity, gated (`G-WL-SUBJECT-ARTEFACT`, §10):**
> ```
> Σ voxel yield since base == S₀ − S_current + Σ declared physical loss
> ```
> and the two illegal combinations of (rock present/absent) × (stock zero/nonzero) are **FAULTS with a declared
> repair action**, not silent states (§6.6's four-state table).

**Consequence for slice order:** W-2's in-game criterion ("see the changed refinery") and W-3's ("walk up to an
NPC") both presuppose a durable **artefact** to re-materialise from, i.e. P6 block persistence / Store B at W-6.
§11.4 re-orders accordingly rather than leaving an unachievable criterion in the plan.

---

## 8. THE S11 RESOLUTION — item conservation and no-dupe under LAW-E1

### 8.1 The decision

**S11 is resolved as DISCIPLINE, NOT DEPENDENCY, and the answer is YES: no-dupe is provable with the ledger
entirely out of the loop.**

Items are **PHYSICAL** (§2 row 12). The append-only-POSITION discipline and the `ItemId` minting shape land
in the **GAME's** `vd-core`/store; the economy is a **READER/projection**. The report's S11 as literally
written ("the ledger, never the blob, is the uniqueness authority",
`scripts/economy_research_20260726.md:2539`) is **inverted coupling** under LAW-E1: if items *are* ledger
entries, an economy outage stops a player mining a rock.

### 8.2 THE RULE that must constrain the FIRST line of P6/P7 inventory code

> **RULE WL-ITEM.** A stack is an **APPEND-ONLY POSITION**, never a mutable count. Every stack carries an
> `ItemId` minted exactly like `EntityId::pack{kind:8, mint_shard:32, seq:64, rand:24}`
> (`crates/core/src/ids.rs:164-171`) — **wait-free shard-locally, NEVER wall-clock-derived, never reused**,
> with `rand` masked (`& 0x00FF_FFFF`, `:170`) so overflow cannot corrupt `seq`, and backed by a **durable
> monotone high-water** persisted before each mint and never re-derived from `max(survivors)` (the
> `WaterMark` pattern, `crates/node/src/rlm_spawn.rs:172-180`). Split and merge are **zero-sum entry
> sets**. Every disappearance carries a **typed reason**. **Items inherit their CONTAINER's durability.**
> **And the container of record is ALWAYS a GAME container — never an economy object (RULE WL-LIEN, §2 row 28).**

The R7 rationale for "never time-derived" is already written into that module header
(`crates/core/src/ids.rs:10-11`) — the old SipHash-of-`SystemTime::now` session token was forgeable and
enabled three bug classes. Nothing new needs justifying.

**Cost split, stated honestly** (this is where the report's "~0 lines" is half-true):
- **~0 lines / NOW**: the written rule, plus `ItemId` (~40 lines mirroring `EntityId` + its pack/unpack
  bijection proptest). Pure `vd-core`, zero economy dependency, **landable today**.
- **A REAL SLICE / later**: the durable substrate it presupposes. There is **no shard-side `Store`**
  (`crates/node/src/orchestrator.rs:152` is the only injector) and **no durable exactly-once journal**
  (`AppliedSteps` is a RAM `BTreeSet`, `crates/sim/src/stub.rs:1087`; D-22 owed, unbounded retention).

### 8.3 The item-durability decision (W8) — the third option the report omits

⚠ **Item conservation can NEVER be strict `Σ in == Σ out`.** Verified: `DROPPED_BLOCK_DEF` is
`Transient`/`LossBudget(2)`, `DEBRIS_DEF` (4), `PROJECTILE_DEF` (8), `ROCKET_DEF` (2)
(`crates/core/src/entity_kind.rs:225-259`), and `InterShardFlow::TransientAbandon` exists **specifically**
to abandon a batch as an accounted loss-within-budget when the dest dies, policed by
`verify_transient_loss_budget` (`crates/harness/src/oracle.rs:440-457`). **The game is already licensed to
destroy in-flight items.**

| Option | Consequence |
|---|---|
| (a) every stack becomes `Durable`/`LossBudget::ZERO` | **destroys the batched-transient amortisation**: one go-token per BATCH becomes one saga per stack |
| (b) stay `Transient`, loss posted to a declared `Sink::TransientLoss` (the report's §4.9.1) | correct, but every loose stack is lossy including one in a player's hands |
| **(c) items inherit their CONTAINER's durability** — a stack inside a `Durable` player/ship TLV blob is Durable-by-containment and zero-loss; only **LOOSE** world drops are Transient/lossy | **RECOMMENDED.** Preserves the amortisation, matches player intuition, and confines loss to exactly the objects the loss budgets were written for. ⚠ **Its binding constraint is `KindDef::max_state_bytes`, not the 1 MiB frame cap**: `PLAYER_DEF = 4096` B (`crates/core/src/entity_kind.rs:204`), `SHIP_DEF`/`NAMED_CONSTRUCTION_DEF = 8192` B — so at §5.4's measured ~55 B/row the in-blob inventory is a few **tens** of stacks per container, enforced at **mutation** time against that per-kind number (§7.3(a)). Raising it is a `KindDef` decision with a wire-sizing consequence, not a free parameter |

### 8.4 THE PROOF, re-established with the ledger out of the loop

`verify_item_conservation` is a **SIBLING of three LANDED oracles**, not new science. All four live in
`vd-harness` and audit **captured `InspectReport`s** — never a node's claim about another
(`crates/harness/src/oracle.rs:1-5`).

| Property | Shape to copy | Existing sibling |
|---|---|---|
| **No duplication (per tick)** | an `ItemId` counted in **`>1`** container is duplication, always failure — **`>1`, not `!=1`**, so the structural mid-flight zero-holder gap stays legal | `verify_transient_conservation_tick` (`crates/harness/src/oracle.rs:380-419`) |
| **Exactly one owner** | **`len == 1` exactly**, never `<= 1`, so orphans are not hidden | `verify_authority_unique` (`:121-146`) |
| **Declared loss judged** | reads the handover-attributable counter, never the gross drop count | `verify_transient_loss_budget` (`:440-457`) |
| **Exactly-once application, typed discards, phantom detection** | `AppliedTwice` / `Phantom` / `Unaccounted` typed violations | `verify_input_conservation` (`:586-645`) |
| **Dead-aware** | `_excluding(dead)` variants so a killed source's corpse `Held` claim is not a phantom second holder | all of the above |

**THE IDENTITY:**
```
Σ mint − Σ burn − Σ DECLARED_loss == Σ positions          (per ItemId kind, per scenario)
∀ tick: |{container : container holds ItemId}| ≤ 1        (>1 ⇒ FAILURE)
```
**RED anti-theater control is MANDATORY** (the discipline at `crates/harness/src/oracle.rs:1044-1068`): a
deliberate 1-unit imbalance must be **DETECTED**, and a deliberately non-dead-aware variant must fail.
Without it the gate is decorative — the exact failure the D-6 no-op-stub-`Store` guard was written to
prevent.

⚠ **AND THE IDENTITY WAS NOT REPAIRABLE UNDER REVISION 1'S OWN SHED POLICY — the fix is RULE
WL-CONSERVED-FACT (§2), not a caveat here.** `Σ DECLARED_loss` presupposes per-`ItemId` attribution, but a shed
was recorded as `JournalGap{from_lsn, to_lsn, count}` — a **count**. A count cannot repair a per-`ItemId`
identity, so in production the identity was **unverifiable whenever anything shed**, while this gate stayed green
because it runs over in-memory `InspectReport`s with the journal entirely out of the loop, i.e. **green precisely
in the arm where the hazard is absent.** Because §2 now puts every mint, burn, declared loss and
destruction/delivery fact in the **STATE** record (`LossBudget::ZERO`, never shed), the identity is repairable by
construction. Two obligations follow and both are gate rows:
- **A mid-scenario SHED cell**: shed the journal mid-run and assert the identity **still holds**, with a RED
  control (move one mint back into the journal, shed it, and assert the gate **FAILS**).
- If any conservation-bearing fact is ever left in the journal, `JournalGap` **must** carry per-kind
  per-quantity attribution, **or** the identity must be restated as an inequality with an **explicitly
  declared** unverifiable window. Silence is not an option.

⚠ **A POPULATION conservation identity is owed too, and §9.4's three loss channels did not cover it** — an NPC
crossing a realm boundary created or destroyed population with no arm, no fence and no declared loss (§6.3):
```
∀ scenario: Σ(cohort P + promoted rows + materialised entities) conserved
            across every materialise / promote / demote / re-home
```
`verify_population_conservation` is a sibling of `verify_authority_unique` with a **RED 1-NPC-imbalance
control**.

**Three things this proof gains by living in the game's harness:**
1. It is **provable BEFORE the economy exists** — which is the direct answer to the report's own S8
   anti-theater worry.
2. It runs where every existing conservation proof runs (over captured reports), so it is **re-run by the
   accumulated crash/chaos suite** (`tests/tests/p3_crash_matrix.rs`, `tests/tests/p3_transient.rs`; and
   `tests/Cargo.toml:6` records that nothing is ever deleted).
3. It inherits the dead-aware discipline a fresh econ oracle would have to rediscover.

**The ONE new obligation it creates, stated up front:** with the ledger out of the loop,
`Δ money_supply == Σ faucets − Σ sinks` becomes a **JOIN over two independently-durable logs**, so a
reconciliation invariant is owed — **the econ projection must be provably a pure function of the game's
journal**. That reframes `G-ECON-REPLAY` from "the twin reproduces the economy" to "**the economy never
diverges from the game's own truth**", which is the stronger property anyway.

⚠ **REJECTED: report S6 (`DirectoryKey::Account(AccountId)` reserved inert).** `DirectoryKey` is a 4-arm
closed enum (Session/Entity/Realm/Ship) consumed by an exhaustive wildcard-free match
(`crates/wire/src/seams/directory.rs:17-26`, `:200-207`), and **the directory CAS is the only commit
point**. Putting an economy concept there makes an economy fact an **authority input** — exactly the
coupling LAW-E1 forbids. Accounts get single-writer discipline from whichever realm hosts them. This is
**W13**.

---

## 9. DETERMINISM

### 9.1 Why closed-form + event replay is bit-identical

Four properties, each **structurally guaranteed** rather than tested-for.

1. **LAW-WL-1 — no per-tick accumulation, live or dormant.** The cumulative closed form IS the state; a
   live tick's delta is `F(t) − F(t−1)`. Naturally lumpy (a "drip"), which is correct and observable rather
   than smooth-and-wrong. **This makes live == dormant BY CONSTRUCTION.** In the repo's vocabulary the
   worldline is **Category A** (analytic `f(seed, universe_tick)`, zero per-tick accumulation,
   `docs/design/PLAN.md:122`), deliberately avoiding Category C (checkpoint-carried rapier state) — which
   is why it needs no realm checkpoints, and realm checkpoints do not exist.
2. **NO FLOATS in the authoritative path — AND AN EXPLICIT INTEGER BOUNDARY AT THE GENERATOR, which revision 1
   omitted and without which the claim was FALSE.** `i128`/`u64`/`u128` only inside
   `crates/core/src/worldline/`; every rate an integer rational over ticks; every relaxation `ipow_muldiv`;
   every categorical draw a `u64` threshold compare.

   ⚠ **Revision 1 firewalled floats INSIDE the kernel and then read every seed-derived INPUT from the float
   layer**, which is what made "the worldline is bit-equal on any target today, unlike the layer it sits beside"
   untrue at its own boundary. The whole premise is **multi-host independent evaluation of the same `f`** (§4.8
   "the same code in all three hosts"; §7.2 #1 "both parties compute the identical `f` from the shared seed";
   H4's Tier-B evaluate-on-read) — and: carrying capacity `K` is "seed-derived from the realm's content
   (habitable extent, construction slots)" (§6.2); the NPC waypoint ring is drawn from seed-derived construction
   slots (§6.2); deposit `S₀` is the S12 generator (§11.1); and `Baseline::seed(key, seed)` in G-WL-ZERO-BYTES
   **is by definition the generator's output**. That layer is f64 + libm with **ungated** cross-host
   bit-equality: `crates/core/src/celestial.rs:11-18` says the determinism test "only re-runs the SAME binary
   twice on one host, which cannot catch cross-build FMA-contraction/vectorization divergence", and
   `crates/core/src/taxonomy.rs:8-25` says every sampler is "exactly one `SplitMix64::next_f64`" with the
   transcendentals in "the SAME libm class" and the gate owed to **SPIKE-6a**. `celestial.rs:16-18` states the
   standing containment verbatim: *"AUTHORITY never depends on this bit-equality regardless … the hole is
   determinism-hygiene for terrain, not authority."* **This design INVERTED that** — it made authority-of-record
   physical state a value three different binaries independently recompute from the seed, promoting SPIKE-6a from
   hygiene to **authority-load-bearing**. `G-WL-ZERO-BYTES` cannot catch it: it compares in ONE process.

   > **RULE WL-INTQ (the integer generator boundary).** **Every value crossing from the content generator into a
   > worldline quantity is QUANTISED to an integer by ONE named function at generation time**, and
   > `Baseline::seed` is integer-only over those quantised values. This is the same physics→control quantisation
   > discipline the repo already mandates. Concretely it **broadens report S12** from "ore distribution" to
   > **ALL worldline inputs**: `K`, `S₀`, extents, construction-slot positions, waypoint coordinates and every
   > rate numerator. Gate: **G-WL-NO-SECONDS is extended** to assert that no worldline quantity reads a
   > float-valued generator output un-quantised, plus a **cross-BINARY determinism gate** (two builds / two
   > `target-cpu` settings, digest-diffed) as a **W-0 exit criterion** — not a P4 owe, because the worldline is
   > what promotes the hole to authority. ⚠ The recorded alternative (materialise every baseline once, at first
   > touch, and never re-derive) only helps for *touched* realms — a never-touched realm is still re-derived by
   > whoever reads it — so it does not remove the exposure and is **not** the recommendation.

   > **AND A `content_epoch`, because P4 WILL CHANGE THE GENERATOR AFTER W-0/W-1/W-2 LAND.** `evaluate`'s inputs
   > include the seed-derived subject **roster**, and the design leans on "absence of a baseline MEANS pure seed"
   > (G-WL-ZERO-BYTES). A P4 terrain change silently re-bases every untouched realm and mismatches every existing
   > baseline's roster. `WorldEpoch` covers rate/config, not content — so **`content_epoch` is stamped on every
   > baseline** and participates in the same **settle-on-change** rule (LAW-WL-5), so a generator change closes out
   > the prior interval with settled integers instead of rewriting history. Decided **before P4**.
3. **LAW-WL-4 — NO SECONDS.** Rates are per-**TICK**, so `universe_tick` alone is the time base and the
   local pacer never enters.
4. **LAW-WL-2/WL-3 — no ordering hazard.** Commutative idempotent last-wins fold; big-endian `(tick, seq)`
   in the store key makes `scan`'s **contractual** ascending order (`crates/sim/src/io/mod.rs:408-410`,
   clause 3) *be* the sort. Determinism is a property of the key encoding and the monoid, not of code
   anyone can get wrong.

### 9.2 The hazards, with the rule for each

| Hazard | Verified evidence | The rule |
|---|---|---|
| **Clock propagation / `universe_tick` is NOT a portable time base** | `ClockSync` carries only `{universe_tick, epoch}` (`crates/wire/src/seams/directory.rs:120-125`); `secs_since_epoch(tick, tick_hz)` is fed the shard's OWN `1.0 / config.tick_dt_s` (`crates/sim/src/stub.rs:4381`, `:4262`), and `crates/core/src/celestial.rs:271-273` documents `tick_hz` as "a per-shard knob passed in". Two shards with different `VD_TICK_DT` convert tick 1000 to **different seconds**. D-Finding-3 is unimplemented (`scripts/realm_lifecycle_design.md:151`). | **RULE WL-4 (no seconds):** no worldline quantity may derive from a `secs_since_epoch` result or from the local `tick_dt_s`, and **every cadence and every content rate is authored in TICKS** as a multiple of one named quantum (§4.7 — revision 1's table derived them from seconds at a hardcoded 20 Hz, which violated this rule in the design's own text). **BIT-EQUALITY therefore does NOT depend on D-Finding-3** — a real de-risking — but ⚠ **content PORTABILITY does**: if the user ever wants content authored once to mean the same thing across deployments with differing `VD_TICK_DT`, or believability targets expressed in human time, a canonical universe seconds-per-tick on `UniverseConfig`/`ClockSync` becomes a **content** prerequisite. Gate: G-WL-NO-SECONDS, both arms tested. |
| **RULE WL-4 IS BROKEN AT THE MATERIALISATION SEAM** ⚠ *new* | a materialised NPC is an ordinary ECS occupant (§6.3), and occupant motion integrates as `move_speed_mps * tick_dt_s * time_multiplier` (`crates/sim/src/stub.rs:1853-1858`) — **seconds, from the shard's own local pacer**, the very thing this row calls non-portable — while its closed-form ephemeris (§6.2) is expressed in **TICKS**. So `G-WL-SEAMLESS-NPC`'s `\|Δvel\| == 0` across the band was achievable only via exactly the canonical seconds-per-tick constant WL-4 claims not to need, and two shards with different `VD_TICK_DT` move the same NPC at **different speeds** and disagree at a shared boundary. Revision 1's "does NOT depend on D-Finding-3" was false at the one seam where continuity is asserted. | **A worldline occupant carries a TICK-NATIVE speed field** (`units per tick`, an integer rational) and integrates in ticks, so the closed form and the live path share ONE unit and WL-4 holds end-to-end. Recommended over the alternative (admit the D-Finding-3 dependency), because it is cheaper and keeps the de-risking real. Gate: **G-WL-NO-SECONDS gains a cell that runs the SAME materialisation fixture on two shards with different `VD_TICK_DT` and asserts identical NPC pose AND velocity** — a cell that fails today. |
| **Iteration order** | default-hasher `HashMap`/`HashSet` are banned in sim/node by per-crate `clippy.toml` (`crates/sim/clippy.toml:1-19`), and config is **not inherited** | `BTreeMap`/`BTreeSet` keyed by `(RealmPath, id)` throughout; `DetHashMap` exists but ordered iteration is what a deterministic fold needs. Any new crate adds its own `clippy.toml`. |
| **Integer breakpoints** (a wrong `min`, or `div_floor` where `div_ceil` belongs) | the §4.3 exactness argument is correct but **not obviously** correct | **MANDATORY**: differential-test every closed form against a slow `O(Δ)` per-tick reference in the test module — the ARM-A/ARM-B divergence-oracle idiom (`scripts/rlm_step4_persistence_spec.md:70-144`), which in RLM found a real bug unit tests masked. This is the **largest single test cost in the design** and it is not optional. |
| **Overflow on huge Δ** | `Δ` up to `u64::MAX`; `S.0 · rd` can exceed `u128`; and `muldiv(u128, u128, u128)` needs a **256-bit** intermediate | ⚠ **Revision 1 named the caps and never DERIVED them, i.e. magic numbers by omission**, and declared a `muldiv` signature that was **unimplementable as specified**. §4.2 now narrows the signatures to `(u128, u64, u64)` so the intermediate provably fits, adds signed variants with an explicit sign contract, and **derives** `max_rate_num ≤ 2³²`, `max_den ≤ 2³²`, `max_stock_minor ≤ 2⁹⁶` (Fx) from the u128 headroom with the inequality written out per cap. Every product is `checked_mul`; `validate()` is fail-loud at registry load and **names the inequality each cap enforces** — the `ShardProfile::build` / `AoiConfig::for_velocity_safe` pattern. G-WL-OVERFLOW asserts the **derived** caps. Where saturation is reached it is the **semantically correct** answer (it dried out long ago), commented as such and proptested at `Δ = u64::MAX`. |
| **Two `seq` minters, and a STALE INCARNATION** | verified: `zombie(head, dead)` ⇒ ForceReap while the process still runs (`crates/sim/src/rlm.rs:404-410`, `:611-622`); the arm is `ReDriven` | ONE durable monotone high-water **at the CUSTODIAN**, per `(realm, fence)`, persisted before each mint, **never** re-derived from `max(survivors)` (`crates/node/src/rlm_spawn.rs:172-180`, `:292-355`). ⚠ **AND the fold key is `(realm_fence, tick, seq)` with the FENCE DOMINANT, plus a custodian stale-reject counted as a FAULT** — without the fence term a force-reaped-but-running incarnation reverts its successor's rebase and **mints material** (§4.4 has the full sequence). Re-deriving after compaction deleted rows would re-mint a retired `seq` and silently reorder history. |
| **Compaction non-associativity** | `ipow_muldiv` floors per squaring step | ⚠ **NOT "a cache-precision question": compaction DELETES the folded deviations, so a compacted baseline is AUTHORITATIVE and any error in it is permanent** (§4.5). The answer is to make compaction **bit-exact by construction** — cut only at ticks where every subject's segment is closed by an actual `RebaseSubject` (which carries the absolute value, so no arithmetic is redone) — and then **G-WL-RECOMPACT asserts BIT-EQUALITY**, plus `compact`'s idempotence and cut-order independence. If a non-rebase cut is ever permitted, the residual needs a **CUMULATIVE** bound (~8 760 compactions/yr at the old sweep cadence), and it is posted as a **declared physical loss**, never a monetary event. |
| **A same-TICK deviation dropped by compaction** | the fold key is `(tick, seq)` while revision 1's reader rule was `tick ≤ base_tick` | `Baseline` carries a `(base_tick, base_seq)` **high-water**; the delete removes **the exact scanned key set**; the cut never splits a tick (§4.5). This one minted material with **no crash required**. |
| **A journaled record with no SCHEMA version** | `Baseline`/`SubjectState`/`DeviationRecord` are postcard-**positional**; CLAUDE.md mandates TLV framing with decode-to-Default **BANNED** for Durable kinds | ⚠ **Revision 1's only versioning was the SEMANTIC `WorldEpoch` (plus `SEGMENT_FORMAT_VERSION` for optional segments), so adding one field to `SubjectState` would silently mis-decode every historical record.** **TLV-frame the durable worldline records**, reusing `crates/core/src/tlv.rs` verbatim (it is already the sanctioned Durable-blob encoder and the canonical HR5 branchless-shim reference): a declared `field_count` catches field-boundary truncation, `required_max_tag` is the **schema** version floor, and an unrepresentable record is a typed refusal, never a partial decode. **`WorldEpoch` is a CONFIG version and the TLV floor is a SCHEMA version — different axes.** Gate: G-WL-SCHEMA-FLOOR cell 4 (a truncated record and an unknown-required-tag record each produce a typed refusal). |
| **A CODE change to a closed FORM across a rolling deploy** | — | ⚠ Also missing from revision 1: `WorldEpoch` versions the **config**, not the **math**. Two binaries integrating one interval with different closed forms diverge silently. **Every closed form carries a `FORM_VERSION` that participates in the same settle-on-change and epoch-floor refusal**, so a rolling deploy settles the prior interval instead of re-integrating it. |
| **Schema / rules evolution** (Design C's graft; Design A had NOTHING here) | — | **LAW-WL-5 — `WorldEpoch: u32` on EVERY durable record + SETTLE-ON-CHANGE.** Any capacity/rate/config change **CLOSES OUT** the prior interval with a settled integer (`RebaseSubject` carrying `produced_since`), so `advance` only ever integrates the CURRENT config and a rebalance **never rewrites history**. A reader that cannot represent an epoch returns `Refused` and its output is stamped `Partial{from_epoch}` — never a number it cannot justify, mirroring `MissingRequiredTag`'s "refuse …, never decode to Default" (`crates/core/src/tlv.rs:58`). ⚠ Silent when violated ⇒ its own gate cell (G-WL-SCHEMA-FLOOR cell 3: a mid-scenario rebalance). |
| **Agent order** (only if the optional coarse tier is built) | — | **JACOBI, never Gauss-Seidel** (Design B's graft, kept as a standing rule): a coupling/decision sweep reads only the PREVIOUS pass's values, so it is permutation-invariant. Plus phase decimation by `hash(seed, agent_id) % round_period_ticks`, and `DetRng` seeded `f(seed, world_epoch, round_index, agent_id)` — **never** from a tick or wall clock. |
| **Dormant/live divergence** | — | LAW-WL-1 + **G-WL-CLOSEDFORM-EQ-ACCUM** with a RED control (introduce a live per-tick accumulator and assert the gate FAILS), scoped to a window with **no autonomous authoring** — plus **G-WL-ATTRIBUTABLE-DIVERGENCE** for the occupied case, since a live realm with NPCs, haulers and promotions is not expected to be byte-identical to a dormant one and never was (§1.2 property 3). |
| **`DetRng` does not exist** | prose only — `crates/sim/src/lib.rs` and `crates/sim/src/io/mod.rs` mention it; the trait was never landed | any randomness must be a **pure hash** of `(seed, ids, epoch)`, never an RNG *stream* — a stream makes fold order load-bearing and silently breaks replay. |
| **Subjective vs objective time** | `time_multiplier` applies ONLY to occupant movement integration (`move_speed_mps * tick_dt_s * time_multiplier`, `crates/sim/src/stub.rs:82-89`, applied `:1853-1858`); celestial/child placement gets **no** multiplier | **RULE: production, depletion and population are OBJECTIVE (universe time, no multiplier); only occupant-carried actions are subjective.** Nothing in code decides this and the answer changes every rate formula, so it must be written down — **W9**. |
| **Seed content is a toy today** | `generate_system_forest` hardcodes one Universe/Galaxy/system (`crates/core/src/worldgen.rs:383-421`); `container_coord_at` regenerates the whole forest per call and documents the lazy generator as the P4 owe (`:468-500`) | the zero-bytes and evaluate-on-read claims are **demonstrable at toy scale only** until the lazy per-subtree generator lands. Stated as a prerequisite, not a risk. |

### 9.3 Category placement, stated once

- **Worldline state = Category A.** Analytic, no accumulation, no checkpoint needed. ✅ sanctioned pattern.
- **Promoted subjects = Category A too**, because a promotion carries the full absolute state (LAW-WL-2).
- **The optional coarse agent tier = Category C** (path-dependent, checkpoint-carried, never re-simulated
  on another host). Two same-seed universes **diverge** after its first round; analytics must replay its
  log, not the seed; and H4 evaluate-on-read is **forbidden for that layer**. This is the honest price of
  autonomous history — and per §12.2 it is why W5 is a **scope** decision with three named consequences, not a
  free "optional" (revision 1's framing).
- ⚠ **The coarse tier ALSO needs a per-INSTANCE capability that has no home, and it cannot satisfy G-IDENTICAL.**
  W-5 runs only on a `pinned` host, i.e. behaviour that exists on one realm instance and not its siblings. But
  `ShardProfile` is a **pure function of realm KIND** (`RealmCoord::profile_kind()` → `profile_kind_of`,
  `crates/core/src/realm_coord.rs:61-64`, `:93-106`; `profile_for` is a total wildcard-free match), so "this
  instance runs the agent tier" has no capability home and becomes a **runtime instance fork inside a feature** —
  and HR4 requires the identical fixture to pass on ≥2 shard kinds, which a Galaxy-pinned tier structurally
  cannot. §13 Q6 noticed the capability problem while §11.4 still scheduled the tier without resolving it.
  **If W-5 is adopted: the tier goes behind an INJECTED object-safe port whose only implementation is wired in
  `vd-bins` from the pin config** (the `Store`/`RealmSpawner` idiom, `crates/sim/src/io/mod.rs:415-479`), so
  `vd-sim` contains no instance fork and no kind match, and **G-IDENTICAL is declared as "the identical fixture
  passes on ≥2 kinds with the port ABSENT"** — honest and checkable. Otherwise the tier defers until report
  **S5b** (per-instance capability) lands, and D-71's dependency column says so.

### 9.4 Loss accounting so no gate is ever relaxed

Declared loss channels, each posted at the site that causes it, so **every** conservation identity
stays an *equality* rather than being weakened (the report's §8.6 Rule-0 gate-decay hazard).

⚠ **FIRST, THE REGISTRY SPLIT — revision 1 had ONE `Sink` enum spanning physical loss channels AND monetary
sinks, so a physical durable record referenced the monetary taxonomy.** §4.3's `HopperPolicy::Spoil(SinkId)` is a
field of a **physical** subject; this section named `Sink::CompactionResidual` / `TransientLoss` /
`UnreportedGap` as physical loss channels; §11.3 put `Sink::Destruction` in "a `vd-core` sink registry" — while
`Faucet`/`Sink` is the **monetary** faucet/sink taxonomy inherited from the report's S7. If they share one id
space then (a) a monetary rebalance renumbers `SinkId` and mis-interprets durable **physical** records, (b) a
monetary registry change forces a `WorldEpoch` bump which under settle-on-change rewrites **physical** intervals,
and (c) "is this economy code?" becomes unanswerable because the enum is both. That is the same
layer-split mis-assignment as §2 row 4's, hiding in a type name.

> **TWO REGISTRIES.** **`PhysicalLossChannel`** in `vd-core` (`Spoil`, `CompactionResidual`, `TransientLoss`,
> `DestructionAttrition`, `Decay`) is the **only** thing a worldline durable record may name. **`MonetarySink`**
> lives in `vd-econ` and may **project from** a physical loss channel but is **never referenced by a physical
> durable record**. Enforced by an exhaustive coherence test next to `KindDef::is_coherent`
> (`crates/core/src/entity_kind.rs:175-191`) asserting no worldline field names a monetary id.

| Channel | Bound | Classification |
|---|---|---|
| `PhysicalLossChannel::CompactionResidual` | **zero** under §4.5's bit-exact rebase-only cuts; otherwise ≤1 minor unit/subject *per compaction* with a **cumulative** annual bound | FAULT if nonzero under the bit-exact rule |
| `PhysicalLossChannel::TransientLoss` | within the per-kind declared `LossBudget`, posted where the per-kind `transient_loss` counter increments | judged by `verify_transient_loss_budget` |
| `PhysicalLossChannel::Spoil` | the `HopperPolicy::Spoil` overflow, exact by construction | accounted, not a fault |
| ~~`Sink::UnreportedGap`~~ | **DELETED — see below** | — |

⚠ **`Sink::UnreportedGap` IS DELETED, because it contradicted two of this design's own laws and legalised the
silent loss of player-caused change.** Revision 1 declared it as "a shard died without emitting its final
deviation batch — bounded by one `field_refresh_period_ticks`, counted as a FAULT". Both halves are wrong:
- Under **LAW-WL-1** a closed-form quantity loses **nothing** on a crash — it is re-derived from base+devs at
  any `t` — so there is no time-proportional amount to lose.
- Under **RULE WL-ACK** every irreversible player-visible mutation is already durable before its ack, so the
  only thing a dying shard can lose is an **un-acked discrete** deviation — which is not "bounded by one
  `field_refresh_period_ticks`" (12 000 ticks ≈ 10 minutes) but bounded by whatever is un-acked.
- As written it therefore **booked up to ten minutes of player-caused change as a DECLARED loss**, meaning
  `G-WL-CONSERVE-ACROSS-LIFECYCLE` would stay green while a player's work vanished — the exact gate-decay
  pattern §12.1 criticises in Design C's `G-JOURNAL-FAITHFUL`.

> **Replacement:** un-pushed **player-caused** deviations are impossible by construction (no batching of
> player-caused deviations; durable-or-provisional per mutation, §4.6), and any residual gap is a **HARD FAULT**
> with its own counter — never a declared loss. NPC/machine-caused batching is legal only where the quantity is
> re-derivable from the closed form, in which case there is **no loss to declare**.

⚠ **AND A COARSE-TIER FLOW IN TRANSIT WHEN ITS HOST DIES HAD NO RECOVERABLE ACCOUNTING.** W-5's `Flow` was
specified as "depart/arrive deadlines, **debit-on-depart / credit-on-arrive**", Category C on a pinned host, with
rounds at `site_round_period_ticks` (≈1 universe-hour) and a conservation oracle
`Σ departed == Σ arrived + Σ declared_loss`. But the only channel that could declare the loss was
`UnreportedGap`, bounded by one *field refresh* and requiring **a posthumous post by the dead host** — the very
process that died. A kill-9 mid-round therefore lost **hour-scale in-transit goods with no declared loss**, and
because the departure debit already happened durably the failure direction is **silent DESTRUCTION of material
with an unbalanced ledger**. **Fix: a flow is an ABSOLUTE-REBASE PAIR, not a debit/credit.** Departure writes an
absolute in-transit subject `RebaseSubject(InTransit{from, to, qty, arrive_tick})`; arrival rebases all three
rows in one fold. A dead host then loses nothing — replay reconstructs the in-transit row — and an overdue
arrival is resolved by the closed form rather than by a posthumous sink post. W-5's gate gains a
**kill-9-mid-flow** cell with a RED control.

---

## 10. GATES + INVARIANTS (harness-shaped)

| Gate | Shape | What it catches |
|---|---|---|
| **G-WL-CLOSEDFORM-EQ-ACCUM** *(flagship — RENAMED and RESCOPED from `G-WL-LIVE-EQ-DORMANT`)* | Run a realm LIVE for T ticks **with NO autonomous authoring inside the sliced window** (no observer, no materialised NPCs, economy idle, agent tier absent) against a scripted input log; capture `worldline_digest` (over the **exact `Fx` state**, §4.5) from its `InspectReport`. Re-run identically with the realm REAPED at tick k and re-spun at T (MemSpawner + the RLM mem-twin, the 5f-3a in-process shape). Assert the digest is **BYTE-IDENTICAL** — exactly. Third arm: slice the window at 7 arbitrary wake/sleep boundaries and assert bit-equality with the single-slice run. **RED**: introduce a live per-tick accumulator and assert the gate FAILS. ⚠ **Revision 1 specified this against "a continuously live realm" with no carve-out and it was UNRUNNABLE** (no shard exists in `[k, T]` to deliver the scripted inputs to; and with no inputs there, the live arm is occupant-empty and would itself have been reaped) **and FALSE** in the occupied case (NPCs, haulers, promotions). §1.2 property 3 is restated to match. | The only proof of LAW-WL-1: the live path's per-tick delta really is `F(t) − F(t−1)`. |
| **G-WL-ATTRIBUTABLE-DIVERGENCE** ⚠ *new — the honest replacement for the occupied case* | Run the same fixture with NPCs live and (separately) with the economy live. Assert every physical divergence from the quiet arm is **attributable to a specific recorded deviation or journaled `EconCommand`**, with a **RED unattributed-divergence control**. Never a byte-identity assertion. | The property that actually matters in production, which revision 1 implied by overselling the flagship gate. |
| **G-WL-DORMANT-MOVES** *(MULTI-HORIZON)* | At **each** of `Δ ∈ {30 d, 180 d, 2 yr}` assert `evaluate(Δ) ≠ evaluate(Δ − 30 d)` — a nonzero delta in the **LAST** interval — plus `Σ\|Δstock\| > 0`, ≥1 depletion/saturation/season breakpoint crossed, and a nonzero population delta. ⚠ **Revision 1 asserted only the 30-day form, which is satisfied identically at 30, 300 and 3 000 days — so the gate written to prove "not frozen" was passed by a world that freezes on day 23.** The design as revision 1 wrote it **FAILS** the multi-horizon form, which is why it is the gate that matters (§4.3a). | LAW-E2 property 1 — the world is not frozen **at any horizon**. Also guards a future refactor that makes `evaluate` a no-op for dormant realms. |
| **G-WL-RECOMPACT** | Proptest (1024 fixed seeds, the landed `crash_replay_proptest` idiom): for random cut points, `evaluate(base₀, devs, T)` vs `evaluate(compact(base₀, devs, T_cut), devs_after, T)` agree **BIT-EXACTLY** (achievable because cuts are restricted to rebase-closed ticks, §4.5); **plus** `compact` is idempotent and cut-order independent: `compact(compact(b,d,t₁), d', t₂) == compact(b, d ++ d', t₂)` exactly. Plus the crash window (baseline committed, deviations not yet deleted) is idempotent under the reader rule. **NEW CELLS**: (i) ≥2 deviations at the cut tick; (ii) ingest **INTERLEAVED** with the sweep; each with a **RED control** asserting the tick-only reader rule FAILS. ⚠ Revision 1 asserted "≤1 minor unit **and** the DIGEST agrees exactly", which cannot both hold. | Compaction silently changing the future; a double-apply after a mid-compaction crash; **the same-tick deviation drop that minted material with no crash at all**. |
| **G-WL-ORDER** | Op-DSL proptest (Deviate/Compact/Evaluate/Crash/Reboot/ReorderDup) against a hand-maintained independent reference **written from the FENCE rule, not from the same monoid**, with an anti-vacuity `Cell<[bool; N]>` coverage floor and named witnesses. **NEW CELLS**: (i) a **force-reaped-but-still-running incarnation re-asserts deviations CONCURRENTLY with its successor** — RED: remove the fence term and assert the gate FAILS; (ii) duplicate one `PromoteNpc` N times and assert byte-identity (the derived-`promoted_count` proof). | Order sensitivity — RLM's version of this gate found a **real** bug unit tests masked — **and the stale-incarnation dupe §4.4 documents**. |
| **G-WL-DIFFERENTIAL** *(mandatory)* | Every closed form vs a slow `O(Δ)` per-tick reference, over the **policy × DEVIATION-SEQUENCE** cross-product (revision 1 said policy alone): `Stall × RespawnAfter × Seasonal × ≥3 interleaved deviations`, both `RespawnAfter` **regimes** (dry-first and full-first, §4.3), and the **cohort-consumption** integral (§4.3b). Anti-vacuity assertion that the fixture actually crossed a saturation breakpoint, a dry breakpoint, a respawn boundary **and a season boundary** in one run. `Δ ≤ 10⁴` in the fast suite (~1024 cases ⇒ ~10⁷ reference iterations ≈ 1–2 s), `Δ ≤ 10⁸` in a release-only `just worldline-soak`. | A wrong `min`, a `div_floor`/`div_ceil` swap, or a wrong regime boundary — silent, and produces a plausible-but-wrong economy forever. |
| **G-WL-OVERFLOW** | Proptest at `Δ = u64::MAX`, `S = i128::MAX/2`, `rn` at the cap, every policy combination: no panic, no wrap, every saturation semantically correct. Plus a fail-loud `validate()` test **per DERIVED cap (§4.2) that NAMES the inequality it enforces**, with the offending field named. | Overflow, and a mis-tuned or **underived** cap shipping silently. |
| **G-WL-WAKE-LATENCY** *(TWO budgets)* | Release-only named budgets (SPIKE-3a pattern + `crates/harness/src/latency.rs:22`): build a `max_subjects_per_realm` baseline with `max_pending_devs` rows; assert p99(restore + fold + advance) against **`WORLDLINE_WAKE_WARM_BUDGET_US`** and, on a **dropped page cache over the real PVC class**, against **`WORLDLINE_WAKE_COLD_BUDGET_MS`** — ⚠ revision 1's single ~300 µs budget was a WARM number that a single cold read exceeds, so it would pass in CI and fail in cloud (§5.4). Both budgets derived from W-(−1) measurement, both inside RLM's `boot_ticks_p99`. With the ratio-floor / honesty / competitor-real guards **and the delivery/count floor** `percentile_unstable`'s own doc demands, `just worldline-wake release`, wired into `just gate`. ⚠ Must set `boot_ticks_p99` explicitly (it defaults to 0 in the bins). | Wake latency regression; a boot budget silently exceeded; a warm-only budget shipping as if it covered cold. |
| **G-WL-WAKE-E2E** ⚠ *new* | Measure **demand → first authored frame p99** through the real `ProcLaunchBackend` over a 5-deep chain, against a named budget (~3.2 s derived, §5.5). | The number SEAMLESS actually depends on — revision 1 gated a µs-scale component and left the 5-orders-larger real path unmeasured. |
| **G-WL-WAKE-RATE** ⚠ *new, release-only soak* | N realms cycling reap/re-spin at a target rate; assert no fork backlog and bounded aggregate RSS. Derives and pins the concurrency ceiling §5.4 states (~10³ players in v1). | The unstated node-per-realm concurrency wall, and a future slice quietly assuming 10⁴. |
| **G-WL-CUSTODY-BOUND** ⚠ *new, release-only* | Build N = 10⁴ baselines at `max_subjects_per_realm`; assert **custodian RSS and every read's duration inside the measured limit and inside `ProbeTuning::DEFAULT.stall_deadline_ticks`** (`crates/node/src/health.rs:50`), and assert **no code path prefix-scans a worldline family** (a source assertion). RED cell at 10× asserting it FAILS. | The verified OOM/crashloop path: a family `scan` materialises the whole prefix (`crates/io-prod/src/store.rs:751-776`) inside a **512 MiB `requests == limits`** orchestrator (`deploy/k3d/30-orch.yaml:65-67`) whose restart drops all RAM-only lifecycle intent. |
| **G-WL-CUSTODIAN-THROUGHPUT** ⚠ *new, release-only* | The SPIKE-3a shape at concurrency: sustained deviations/s at the target player count with the **ack-barrier percentile**; a **rehydrate** budget asserted inside RLM's `boot_ticks_p99`; and **transfer-saga p99 unchanged with the worldline write path saturated** (the competitor-real guard). | The one structural scaling risk this design introduces, and the standing "load tests land with the perf-bearing subsystem" law. Revision 1 had only a per-op spike and a per-realm wake budget. |
| **G-WL-READPATH-ISOLATION** ⚠ *new* | Hammer the Tier-B evaluate-on-read endpoint; assert the custodian's tick budget and the RLM `LifecycleAction` trace are unaffected, that the read path never takes the ECS `World`, and that `admin_max_rows_per_request` is enforced. | H4 was billed at "NONE — off the taxonomy" while having **no read path at all** (§4.8); built naively it lets a dashboard stall transfers. |
| **G-WL-AGGREGATE-BOUND** ⚠ *new (P6/P8 forward-check)* | Build an installation with 10⁴ functional blocks; assert the realm's **subject count** and **baseline bytes** stay inside `max_subjects_per_realm` / `MAX_SUBJECT_BYTES × n`, i.e. that RULE WL-AGGREGATE held and the block graph never entered the worldline. | The "stations and ships built FROM BLOCKS" collision that LAW-WL-5' as written would have forced (§4.3b). |
| **G-WL-NO-PHANTOM-INFLOW** ⚠ *new* | Two-realm fixture, both dormant, B's source exhausted; assert A's evaluated stock did **NOT** increase. RED cell: a deliberately-assumed cross-realm inflow FAILS. | §7.2 #7's material-from-nothing, which per-realm conservation structurally cannot see. |
| **G-WL-ECON-LIVENESS** ⚠ *new, release-only* | A **hostile** `MemEcon` whose `observe`/`drain_commands` sleeps and allocates; assert the tick latency budget still holds, that ingress overflow is **dropped and counted** (never blocking), and that `econ_ingress_budget_bytes_per_tick` is enforced. | RULE WL-LIVENESS (§1.1): WL-PANICFREE bounded crashes and nothing bounded time or allocation on the tick's critical path. |
| **G-WL-ZERO-BYTES** | For a realm nobody ever visited: `scan(&[TAG_WL_BASE] ++ RealmKey)` is EMPTY, `scan(&[TAG_WL_DEV] ++ …)` is EMPTY, and `evaluate(&Baseline::seed(key, seed), &[], t)` equals the live shard's materialised state. | This is what keeps "a dormant realm costs zero bytes" TRUE as features accrete. Without it, some future slice writes a heartbeat row per realm and nobody notices until 100 k realms. |
| **G-WL-SEAMLESS-NPC** | An NPC crossing the materialisation band: pose and velocity continuous across promote AND demote (`\|Δpos\| ≤ one tick of its own closed-form velocity`, `\|Δvel\| == 0`); materialisation strictly OUTSIDE the render/interaction extent; a full in-out-in traversal produces no duplicate entity and no gap. **NEGATIVE CELL (mandatory)**: set the AoI look-ahead to 0 and assert the gate **FAILS** — otherwise it proves nothing, because `boot_ticks_p99` defaults to 0 and vanish is the DEFAULT behaviour. **NEW NEGATIVE CELL**: a **TTL-forced demotion of a diverged NPC must FAIL** the gate, so the deleted `promotion_ttl_ticks` policy (§6.3) cannot silently return. | The seamless hard rule; the pop/vanish class. |
| **G-WL-SEAMLESS-FIELD** ⚠ *new* | On adopt, assert **no materialised subject's first-visible frame occurs inside the render extent**; negative cell at look-ahead 0. | §5.6's no-pop argument covered only NPC pose/velocity; `evaluate` materialises machines and constructions **all at once** on adopt, which pops at close range in exactly the small-realm case §5.5 identifies. |
| **G-WL-PROMOTION-BOUND** *(a BUDGET test, not a reaper test)* | promoted subjects/realm ≤ `max_promoted_subjects`, enforced **at promotion time with graceful degradation** (a cohort-attributed named individual), never a refusal and never a snap; a soak asserts the log does NOT grow with zero player presence (the attention-bound claim). **`WlRefused::LogFull` is a FAULT, never BENIGN** (⚠ revision 1 had it backwards by player impact: under WL-ACK it means the player cannot mine or build), and it must be **unreachable in normal play** — a cell asserts N actions on ONE subject produce **ONE** pending row (coalesce-on-ingest, §4.5) and a cell drives a realm at the maximum player mutation rate and asserts the refusal never fires. `worldline_recompact_residual_minor` is a **FAULT** that must read 0. Promotion applied twice is a no-op (idempotence, now by derivation not by increment). | Unbounded state growth; the promotion exactly-once hazard; **a bookkeeping backlog silently becoming a gameplay wall.** |
| **G-WL-SCHEMA-FLOOR** | Cell 1: write at `WorldEpoch = N`, read with a reader that knows only `N−1` ⇒ `Refused` + `Partial{from_epoch}`, never a Default-decoded value. Cell 2: a bumped `SEGMENT_FORMAT_VERSION` is **QUARANTINED**, not mis-framed (the `OUTBOX_FORMAT_VERSION` discipline, `crates/io-prod/src/outbox.rs`). Cell 3: a config **rebalance** mid-scenario, asserting settle-on-change kept the pre-rebalance interval exactly reproducible. **Cell 4** ⚠ *new*: a **truncated** worldline record and an **unknown-required-tag** record each produce a **typed refusal**, never a Default-decoded subject (the TLV framing of §9.2, which revision 1 omitted entirely for durable rows). **Cell 5** ⚠ *new*: a **MIXED-EPOCH cluster still completes a re-home in BOTH directions** — the movement-outage hazard of §7.3(b). **Cell 6** ⚠ *new*: a `FORM_VERSION` change settles the prior interval instead of re-integrating it. | Multi-year schema evolution; a forgotten settle-on-change; **a positional-postcard field addition silently mis-decoding history**; **an accounting version change bricking movement**. |
| **G-WL-CONSERVE-ACROSS-LIFECYCLE** | Kill-9 the orchestrator mid-compaction; kill-9 a shard mid-deviation; reap and re-spin a realm 20× in one scenario. Assert `Σ subject stock + Σ declared physical loss` is conserved across every spin-down/up, and that no deviation is double-credited. RED: drop one deviation without a declared gap and assert failure. **NEW CELLS**: (i) a **concurrent stale incarnation** (§4.4) — RED without the fence term; (ii) **delay/drop `WorldlineAdopt` N times** and assert no material is created and the shard refuses mutations with `WlRefused::NotSeeded` rather than falling back to a seed baseline (RULE WL-ADOPT-REFUSE, §4.6); (iii) kill-9 **mid-flow** for the coarse tier's in-transit rows (§9.4). | Conservation across spin-down/up; double-credit; a silent gap; **the adopt-window duplication and the stale-incarnation revert.** |
| **G-WL-ITEM-CONSERVATION** | §8.4's identity in `vd-harness` over captured `InspectReport`s, with the `>1`-not-`!=1` duplication shape, the `len == 1` ownership shape, `_excluding(dead)`, and the **RED 1-unit-imbalance control**. **NEW CELLS**: (i) **shed the journal mid-scenario** and assert the identity still holds, RED by moving one mint back into the sheddable journal (RULE WL-CONSERVED-FACT, §2); (ii) **open an escrow over goods, kill the economy**, assert the identity holds and the goods are still in a **game** container (RULE WL-LIEN). | No-dupe/no-loss, **provable before the economy exists** — and provable in the arm where the hazard actually lives. |
| **G-WL-POPULATION-CONSERVATION** ⚠ *new* | `Σ(cohort P + promoted rows + materialised entities)` conserved per scenario across every materialise / promote / demote / re-home, with a **RED 1-NPC-imbalance control**. | §6.3's second authority over a promoted NPC — invisible to `verify_authority_unique`, which audits entity holders only. |
| **G-WL-NO-SECONDS** | Source assertion (crate_isolation-shaped) that no worldline symbol reaches `secs_since_epoch` or `tick_dt_s`; a **module-level** `#[deny(clippy::float_arithmetic)]` on `crates/core/src/worldline` (⚠ **NOT crate-level — `vd-core` is float-heavy and would not compile**, §3.1); and a source test that no `f32`/`f64` token appears under that directory. **Both arms tested** (a deliberate f64 and a deliberate seconds derivation must each FAIL). **NEW CELLS**: (i) **no worldline quantity reads an un-quantised float generator output** (RULE WL-INTQ, §9.1); (ii) **the same materialisation fixture on two shards with different `VD_TICK_DT` yields identical NPC pose AND velocity** — a cell that fails today; (iii) **no worldline symbol reads another realm's deviation rows** (RULE WL-READ, §6.4); (iv) **promotion is never triggered by the float AoI band** (§6.3). | Cross-shard integral divergence under different `VD_TICK_DT`; a float creeping in; **a float-derived seed input deciding durable authority**; **an HR1 read-boundary violation**. |
| **G-WL-XBINARY-DETERMINISM** ⚠ *new — a W-0 EXIT CRITERION* | Build the worldline kernel + `Baseline::seed` under **two builds / two `target-cpu` settings** and diff the digest over a fixed seed corpus. | The worldline promotes SPIKE-6a from determinism-hygiene to **authority**; every existing determinism test re-runs the same binary on one host (`crates/core/src/celestial.rs:11-18`). |
| **G-WL-NOPANIC** | Fuzz every worldline entry point with adversarial `Baseline`/`Deviation` bytes (truncated, duplicate-tag, out-of-range, `Δ = u64::MAX`) and assert **no panic** — a typed error every time. Plus a source assertion of zero `unwrap`/`expect` outside `validate()`. | The residual fate-sharing §1.1 names: the crate graph is not a fault boundary. |
| **G-WL-LIFECYCLE-BLIND** | Toggle the economy AND the optional life tier on/off mid-scenario; assert the RLM `LifecycleAction` trace (`crates/sim/src/rlm.rs:344-360`) is **BIT-IDENTICAL**. | The only machine proof that no economy/life value reached `desired_alive`/`teardown_ready` — and the gate that would have caught the report's §6.4 proposal and Design D's flow-driven `KeepAlive`. |
| **G-EMPTY-SESSION-ONLY** | A realm populated **only** by NPCs still self-reports `DemandVerb::Empty` and IS reaped, **and no NPC position ever emits a `SpinUp`/`KeepAlive` demand**. **RED control (mandatory)**: make one NPC a session occupant and assert the realm **stops** being reaped. ⚠ **Revision 1 costed this at "~20 lines" — a gate and a doc line — with no change to the occupant fold anywhere. It needs the real `is_session_occupant` code mechanism (§3.4) with its own coverage.** | The verified RLM-killer: `aoi_decide`'s occupant fold chains **held transients** (`crates/sim/src/stub.rs:4397-4421`), so **either** NPC class makes `empty_confirmed` permanently false ⇒ `teardown_ready` never fires ⇒ demand-driven lifecycle is **DEAD** — and the same set feeds the per-child demand loop (`:4425-4440`), so NPC positions launch processes. Must land **before the first NPC entity**. |
| **G-WL-SUBJECT-ARTEFACT** ⚠ *new* | Every `Machine`/`Construction` subject has **exactly one** physical artefact and every artefact has **at most one** subject; the declared repair action (subject wins / artefact wins) is exercised, not discovered. Same shape for a `Deposit` subject versus the voxel edit log (§7.6's reconciliation identity). | The subject↔artefact drift revision 1 never bounded: a `Machine` subject describing a refinery the block world no longer contains, or the reverse. |
| **G-ECON-ABSENT** | §3.5 — the whole suite, three arms, **byte-identical**. Arm (i) means **`vd-tests` built WITHOUT its own non-default `economy` feature** (⚠ **not** "`--no-default-features` on `vd-bins`", which could never change what the suite compiles, §1.1). Arms (ii)/(iii) are ONE build because `NullEcon`/`MemEcon` live in `crates/sim/src/io/mem.rs`, so the comparison is an in-process digest compare. Plus **two** `lint-combos` cells. | LAW-E1, behavioural half. |
| **G-ECON-ATTRIBUTABLE** ⚠ *new* | With the economy **present and active**, assert every physical divergence from the absent arm is attributable to a journaled `EconCommand`, with a RED unattributed-divergence control, and an anti-vacuity assertion covering every `WorldFactKind` **and every `EconVerb`**. | §1.1's P2 — the production-side half of LAW-E1 that byte-identity structurally cannot express. |
| **G-ECON-ISOLATION** | `the_game_never_depends_on_the_economy` in `tests/tests/crate_isolation.rs`, as an **ALLOWLIST over the TRANSITIVE normal+build closure**: `vd-econ` unreachable from any crate outside `{vd-bins, vd-tests, vd-econ}`, plus the converse. **Plus all THREE verified holes closed in the same diff** (§1.1): dev-dependency edges asserted too; the closure computed rather than direct edges only; `TIER_A` 6→9; and explicit per-crate assertions for `vd-node`/`vd-harness`/`vd-connection-plane`/`vd-bins`. | LAW-E1, structural half — the `vd-node → vd-econ` edge that passes CI today, **plus the `[dev-dependencies]` and `vd-sim → vd-life → vd-econ` bypasses revision 1 did not list.** |
| **G-WL-IDENTICAL** *(HR4)* | The identical worldline fixture (adopt → refresh → deviate → reap → re-adopt) passes on **≥2 realm kinds** (a Planet realm and a Station realm) with byte-identical digests modulo seed-derived content. | A "the worldline only runs on X" shortcut, i.e. matching on shard kind (`profile_for` is a total wildcard-free match fed by `RealmCoord::profile_kind()`, so this is an HR3 defect if it appears). |

⚠ **Test-cost honesty**: the `O(Δ)` differential reference is the expensive item. Cap it at `Δ ≤ 10⁴` per
proptest case in the fast suite and push the `10⁸`-tick sweep to a release-only soak (the `just rlm-soak`
shape). Every proptest carries an explicit `PROPTEST_CASES` and a committed `proptest-regressions/` corpus.

---

## 11. WHAT LANDS WHEN

### 11.1 Pre-P4 seams (real deadlines — P4 terrain fixes the material geometry irreversibly)

| Seam | Where | Cost | Why it cannot wait |
|---|---|---|---|
| **`RealmUid` (PRIMARY durable key) + a variable-depth `RealmKey` (SECONDARY subtree index) + `MAX_REALM_DEPTH`** | `vd-core` | ~240 prod / ~200 test | ⚠ **REVISION 1'S FIXED-6-LEVEL PATH KEY WAS WRONG TWICE OVER, and it is the one artefact this document calls unmigratable.** See the two defects and the resolution immediately below. |
| **The integer arithmetic kernel** — `Fx`, `TickRate` (per-TICK), `muldiv_floor`/`muldiv_ceil`/signed variants with **narrowed `(u128, u64, u64)` signatures** and declared opposite rounding, `div_ceil`, `ipow_muldiv` (with the underflow early-exit), the **derived** overflow caps + their `validate()` | `vd-core` | ~360 / ~480 | Subsumes report **S1**'s `muldiv` seam and is the prerequisite for **S12**. Retrofitting integer arithmetic into a shipped float economy is not a refactor. ⚠ Revision 1's `(u128, u128, u128)` signature was unimplementable and its caps underived (§4.2). |
| **RULE WL-4 (no seconds)** + `cadence_quantum_ticks` + its gate | design + 1 source test | ~0 | `ClockSync` carries only `{universe_tick, epoch}`; `secs_since_epoch` is fed the local pacer. Writing this down now means **bit-equality** never depends on D-Finding-3 — ⚠ though **content portability** does (§4.7), which must be said out loud rather than implied away. |
| **RULE WL-INTQ (the integer generator boundary) + `content_epoch`** ⚠ *new* | `vd-core` | ~60 + the quantisation of each generator output | The worldline's seed-derived INPUTS (`K`, `S₀`, extents, slots) are **f64/libm** today (`crates/core/src/taxonomy.rs:8-25`), which promotes SPIKE-6a from hygiene to **authority** (§9.1). And P4 changes the generator **after** W-0/W-1/W-2, so `content_epoch` must exist before the first baseline or every untouched realm silently re-bases. |
| **RULE WL-8 (geometry is the authority)** ⚠ *new* | design + the accumulator seam | ~0 rule / ~80 seam | §7.6 — P4 fixes the material geometry irreversibly, and without the law the scalar stock and the voxel world drift into a farmable inconsistency. |
| **`is_session_occupant` + `G-EMPTY-SESSION-ONLY`** ⚠ *moved EARLIER from §11.2* | `vd-sim` | ~60 prod / ~120 test | §3.4 — it is a real edit to `aoi_decide`'s occupant fold with its own coverage, not the "~20 lines" of gate revision 1 costed, and it must precede **any** NPC entity. |

**⚠ THE `RealmKey` RESOLUTION, because revision 1 froze the one thing it called unmigratable and got it wrong in
two independent ways.** Revision 1: `[(kind u8, seed u64 BE) × 6]`, 54 B, absent = `(0xFF, 0)`, justified as "an
ancestor's `k·9`-byte prefix selects EXACTLY its subtree" and flagged as "A FROZEN ON-DISK ENCODING … landing it
after the first durable row is a migration of every historical record".

**Defect 1 — DEPTH ALIASING.** `6` was chosen from `RealmKindTag::ALL`'s cardinality
(`crates/core/src/realm_path.rs:39-58`), **not from any depth invariant, and there is none.** Verified:
`RealmPath` is an unbounded `Vec<RealmLevel>` and `RealmCoord::child(level)` does `levels.to_vec(); levels.push(level)`
with **zero validation** of depth or of kind non-repetition (`crates/core/src/realm_coord.rs:83-90`); a grep for
`MAX_REALM_DEPTH` finds nothing. Area-in-Area, Station-in-Station and any inserted Sector/Cluster level are all
constructible **today** — and §5.7's fan-out ceiling means inserting a level is one of only two ways past ~58
children per parent. A 7-level path truncated into a 6-level key means **two DISTINCT realms map to ONE
`RealmKey`**, so realm X's rows are read as realm Y's: silent cross-realm state corruption, in the unmigratable
artefact, invisible to every gate (`G-WL-ZERO-BYTES` only checks that a *never-visited* realm is empty, and
per-realm conservation reads phantom stock arriving via an aliased key as legitimate).

**Defect 2 — MOBILITY.** The key **IS** the lineage, and re-parenting is a landed, generic mechanism (containment
re-home; moving Station / reparent ledgered to P4/P5/P8). The end goal is **player-built ships you fly and walk
inside**, and the standing realm-unification law makes everything spatial a realm — but `RealmKindTag` has
**no Ship tag** (six tags: Universe, Galaxy, System, Planet, Station, Area), so a ship realm needs a 7th tag
*and* nests Area-inside-Ship-inside-Station, overflowing any fixed depth. Worse, a ship carrying a refinery
**changes its path on every flight**, so its durable baseline key changes and its worldline state is **orphaned —
silently**, because "absence of a baseline MEANS pure seed" reads an orphaned realm as untouched.

> **THE RESOLUTION (both defects, one shape).**
> - **PRIMARY durable key: `RealmUid(u64)`** — minted from a **durable monotone high-water, never reused** (the
>   `WaterMark` pattern, `crates/node/src/rlm_spawn.rs:172-180`), stable across **re-parenting** and across kind
>   changes. All `StoreKey` worldline families key on it. A mobile or ship realm keeps its state forever.
> - **SECONDARY subtree index: a VARIABLE-DEPTH prefix-order-safe path key** — a **1-byte depth header** followed
>   by `depth × (kind u8, seed u64 BE)`. An ancestor's `1 + k·9`-byte prefix still selects **exactly** its subtree
>   (the only property revision 1 says it uses), with **no ceiling**. Rebuilt on re-parent, and it is a **derived
>   index**, so rebuilding it is not a history migration.
> - **`MAX_REALM_DEPTH` as a named const with a FAIL-LOUD `RealmCoord::child`** (`RealmKeyError::TooDeep{found,
>   max}`), depth headroom well above 6 (§5.7's own scale story needs an inserted level), plus a **proptest that
>   the encode is INJECTIVE over paths of every depth up to `max + 1`, with the `max + 1` case returning `Err`** —
>   refusal, never truncation, exactly the `guard_regions_nest` / `AoiConfig::for_velocity_safe` discipline.
> - Because compaction is now **lazy-on-adopt** (§4.5), the *only* remaining consumer of the subtree prefix is
>   **Tier-B reads**, which is why demoting the path key to a secondary index costs nothing.
> - Revision 1's own correction still holds for the secondary index: with absent = `0xFF`, byte order is **not** a
>   pre-order DFS (a parent sorts *after* its subtree, since `0xFF` > every real tag `0..=5`); the subtree-prefix
>   property is all we use.
> - It also still gives RLM's deferred `StoreKey::Rlm(RealmPath)` snapshot a correct key shape.
| **RULE WL-1 / WL-2 / WL-3** written down | design | ~0 | WL-1 must precede P6 or every P6 machine ships a per-tick accumulator (each one a separate retrofit **and** a permanent live/dormant divergence). WL-2 buys crash/reorder/duplicate safety free. WL-3 dissolves the compaction-associativity anxiety. |
| **`WorldEpoch` + SETTLE-ON-CHANGE** (report **S4**, broadened from `econ_epoch` to a world-wide epoch in `vd-core`) | `vd-core` | ~30 | Every analysis must be joinable to the config **and cadence** that produced it; adding either later orphans all prior history. Settle-on-change is what makes closed-form production replayable across a rebalance. |
| **Ore/resource distribution as closed-form `f(seed, realm_path, voxel_pos)` in INTEGER minor units** (report **S12**) | `vd-core` generator | ~120 | **This design REQUIRES it, not merely benefits**: the deposit initial stock `S₀` in the production integral **IS** this function, and it is what makes a never-visited realm cost zero bytes. P4 fixes the faucet geometry irreversibly. |
| **Integer-threshold twins of the categorical samplers** | `vd-core` | ~40 / ~60 | `next_f64`/`chance`/`taxonomy` samplers are float-based (`crates/core/src/rng.rs:31-42`); must exist before the first field draw or the no-float rule is violated on day one. |
| **`WorldlineTuning`** (one struct, inert all-zero default, fail-loud `validate()`) | `vd-sim` | ~180 / ~250 | Simultaneously the no-magic-numbers answer, the off switch (provably byte-identical), and the mis-tuning tripwire. Diffuse retrofit once literals are sprinkled through emit sites. |
| **G-ECON-ISOLATION + closing both gate holes** | `vd-tests` | ~40 | A rule landed **before** the tempting crate exists is a rule; after, a refactor. And a `vd-node → vd-econ` edge passes CI **today**. |
| **`EconomyPort` (emit-only) + `EconCommand`/`EconVerb` + `NullEcon`/`MemEcon` in `sim::io::mem`** (signatures only) | `vd-sim` + `vd-core` | ~160 | Fixes the **direction of the arrow** before anything is built on the wrong side of it. Reversing this later is a rewrite, not a refactor. ⚠ **`EconCommand` in particular must land NOW**: it is the one thing that **cannot** be added behind an emit-only port later (§3.2), and it is the answer to report §7.18. `PolicyWeights` is **deleted**. |
| **RULE WL-LIVENESS + `econ_ingress_budget_bytes_per_tick`** ⚠ *new* | `vd-sim` | ~40 | The port runs inside `step_tick`; WL-PANICFREE bounded crashes and nothing bounded time or allocation (§1.1). Cheap now, a re-plumb of every emit site later. |
| **RULE WL-LIEN + RULE WL-CONSERVED-FACT** ⚠ *new* | design | ~0 | §2 rows 21/28 — both are placement rules that must precede the first escrow and the first mint, and both are free today. |

### 11.2 Pre-P6 seams

| Seam | Where | Cost | Why |
|---|---|---|---|
| **RULE WL-ITEM + `ItemId`** (report **S11**, resolved) | design + `vd-core` | ~40 + proptest | The largest avoidable retrofit in the plan. ⚠ Only the id shape and the rule are ~0 lines; the **durable substrate is a real slice**. |
| **The sealed `WorldFact` enum** (the `EffectFree` token idiom) | `vd-core` | ~60 | Converts LAW-E1's most important clause from a review checklist into a **type-level impossibility**. Must precede any block/inventory code that would emit. |
| **The ECONOMIC-SIGNIFICANCE registry field** — which entity/block kinds produce a worldline **deviation**, plus the **out-of-realm-source validator** (§7.2 #7) | `vd-core`, next to `KindDef::is_coherent` | ~90 | ⚠ **The single highest-risk data-model decision.** A placed BLOCK must **NOT** be a deviation (thousands per player-minute at P6 would make the custodian the bottleneck); a machine, a construction completion, a deposit rebase, a balance rebase and an NPC promotion must. Validated exhaustively over `EntityKind::ALL` like `is_coherent` (`crates/core/src/entity_kind.rs:175-191`). **The criterion is BOTH rules below** — and the validator additionally refuses any `SubjectState` naming an out-of-realm source, plus asserts no worldline field names a `MonetarySink` id (§9.4). |
| **LAW-WL-5' — SEED-DERIVABLE BEHAVIOUR IS NEVER JOURNALED** (Design C's volume rule) | design | ~0 | Arithmetic: 5 000 agents deciding every 400 ticks = 250 facts/s = 7.9×10⁹/yr = **250–440 GB/yr of pure redundancy**; per-unit production output is worse. **Break this rule and every storage number in this document fails by 10–100×.** Design A named the registry as its top risk but supplied no criterion; this is half of it. |
| **RULE WL-AGGREGATE — AGGREGATE BEFORE JOURNAL** ⚠ *the criterion's missing SECOND half* | design + `vd-core` | ~0 rule / ~60 sum | ⚠ **LAW-WL-5' applied alone FORCES the explosion it is meant to prevent**: a player-placed machine is by definition not seed-derivable, so at 10²–10⁴ functional blocks per station every one becomes a subject — against a wake budget of a few hundred and `max_state_bytes` of 4–8 KiB, with a **typed refusal ("you cannot place another machine here") arriving at P8 in a building game**. One installation = ONE subject whose capacity is the integer sum over its blocks; the block graph stays Category-C (§4.3b). |
| **RULE WL-SETTLED-RATE** ⚠ *new* | design | ~0 | §4.3b(c) — a player-built, signal-driven machine has no representable static `TickRate`; the dormant rate is the last settled one, and the dormant failure semantics are declared rather than discovered at P9. |
| **TLV-framing the durable worldline records** ⚠ *new* | `vd-core` | ~80 | §9.2 — the records are postcard-**positional** with no schema version, so adding one field silently mis-decodes all history, and CLAUDE.md bans decode-to-Default for Durable kinds. Reuses `crates/core/src/tlv.rs` verbatim. |
| ~~`G-EMPTY-SESSION-ONLY`~~ | — | — | **MOVED to §11.1** (pre-P4): it is a real code mechanism, not a gate + doc line, and it must precede any NPC entity. |
| **`verify_item_conservation`** | `vd-harness` | ~250 / ~200 | Provable before the economy exists — the answer to S8's anti-theater worry. |
| **Shard-side Store B** (boot open, rehydrate, commit barrier, `WL_*` families) | `vd-node` + `vd-bins` | ~400 / ~600 | ⚠ **"Store B, per-realm redb" is DESIGN PROSE, not code.** A **real slice**. Not blocking for the recommended v1 custody (W2 option A), but it is what makes RULE WL-ACK local and unblocks item/inventory durability. |

### 11.3 Pre-P11

| Seam | Where | Cost | Why |
|---|---|---|---|
| **`Sink::Destruction` emitted at the destruction COMMIT site**, with `loot_drop_ratio`/`salvage_yield_bp`/`wreck_persistence_ticks` as **per-entity** fields (report **S13**) | `vd-sim` + `vd-core` sink registry | ~40 at the emit site | Two independent reasons it cannot wait: the primary sink is otherwise unmeasurable and no material balance is assertable; and **insurance retro-payouts after an economy outage are only replayable from a log the GAME owns**. |
| **`WorldFactKind::BulkDestroyed{aabb, count, salvage_total}`** — a coarse settled summary | `vd-core` | ~30 | The burst case is the least-designed part: `facts_per_tick_budget` handles ordinary play, but a station destroyed in one tick is exactly the fact you least want shed. Design C flagged this and left it undesigned; land the schema before P11. |

### 11.4 The real slices

| Slice | Content | Gate | HR6 "works in-game" |
|---|---|---|---|
| **W-(−1)** *(do FIRST)* | A ~250-line in-engine measurement spike: `evaluate` over `max_subjects_per_realm` inside a real `step_tick`, measuring (a) ns per muldiv-chain **including the `__udivti3` software-divide cost**, (b) **WARM and COLD** wake µs/ms vs our own `RedbStore` on the real PVC class (not a published benchmark), (c) deviation commit **and `flush`** latency, (d) rehydrate ms per 10⁴ baselines, (e) **custodian RSS at 10⁴ baselines**, (f) **the postcard-measured `MAX_SUBJECT_BYTES`**, (g) **demand → first authored frame p99 through the real `ProcLaunchBackend`**. | The seven measured numbers land in `WorldlineTuning` as budgets; **no figure in §5.4 enters an implementation plan until this runs**, and every one is marked ⚠ **[U]** until it does. | `vdctl state` shows the stub counters advancing. |
| **W-0** *(pre-P4, cheap)* | Everything in §11.1. All inert; byte-identity preserved. | `just gate` green; 100 % region+branch on the new Tier-A code; G-WL-NO-SECONDS (all four new cells) + G-WL-OVERFLOW (derived caps) + G-WL-NOPANIC + **G-WL-XBINARY-DETERMINISM as an EXIT CRITERION** + the `RealmKey` depth-injectivity proptest; the isolation gate with **all three** holes closed; **G-EMPTY-SESSION-ONLY with its RED control**. | `vdctl worldline-eval <realm>` prints a seed-only evaluation. |
| **W-0b** ⚠ *new, gates W-2* | **The fan-out and custody decisions §5.7 and §5.4 force**: the AoI/demand child roster decoupled from the `MAX_REGIONS` bitset (or the intermediate level accepted and the key sized for it), the branching-factor boot assertion, and the **custody option chosen (W2)**. | the boot assertion's both arms; G-WL-CUSTODY-BOUND at the chosen custody. | `vdctl` boots a forest with > 58 siblings and fails LOUD with both numbers named. |
| **W-1** | `Baseline`/`DeviationRecord` (with `realm_fence`, `epoch`, `content_epoch`)/the fence-ordered fold/`compact`, the `WL_*` `StoreKey` families in the **custodian's own redb file**, coalescing deviation ingest with the **custodian-minted `seq`** and the **stale-fence reject**, **lazy-on-adopt compaction**, `WorldlineAdopt` + `WorldlineSeeded` + the `worldline_seeded` run-condition, **the `WorldlineRead` port** (§4.8). | G-WL-ORDER (incl. the concurrent-stale-incarnation cell), G-WL-RECOMPACT (bit-exact + the two new cells), G-WL-DIFFERENTIAL, G-WL-ZERO-BYTES, G-WL-WAKE-LATENCY (both budgets), G-WL-WAKE-E2E, **G-WL-CUSTODIAN-THROUGHPUT**, **G-WL-READPATH-ISOLATION**. | a realm reaped and re-spun shows the same stocks via `vdctl`; `vdctl worldline-read <realm>` answers **without waking it**. |
| **W-2** | The production/depletion/respawn closed forms + `HopperPolicy` + `RegenPolicy` (**`RespawnAfter` default**, both regimes) + **the §4.3a seasonal driver** + **§4.3b's cohort-consumption integral**; the live shard's `F(t) − F(t−1)` delta path; **RULE WL-8's deposit accumulator**. | **G-WL-CLOSEDFORM-EQ-ACCUM** (with its RED control), **G-WL-DORMANT-MOVES at 30 d / 180 d / 2 yr**, G-WL-CONSERVE-ACROSS-LIFECYCLE (incl. the adopt-lost cell), G-WL-NO-PHANTOM-INFLOW, G-WL-SUBJECT-ARTEFACT, G-WL-IDENTICAL. | ⚠ **REORDERED**: the artefact-free demo lands here — a **Deposit-only** realm: fly out, fast-forward 30 universe-days, fly back, see the depleted deposit and the moved population; **and the 5-MINUTE round trip** (§5.6). The **refinery** demo moves to **after W-6**, because a `Machine` subject presupposes a durable block artefact to re-materialise from (§7.6). |
| **W-3** | NPC life: cohorts, population relaxation, the **tick-native** pose ephemeris, AoI materialisation, **interaction-triggered promotion**, re-convergence-only demotion, `NpcStrategy` + `NeedsOnlyStrategy`, §6.6's interaction verbs. | G-WL-SEAMLESS-NPC (**with both negative cells**), **G-WL-SEAMLESS-FIELD**, G-WL-PROMOTION-BOUND (as a budget test), **G-WL-POPULATION-CONSERVATION**, **G-WL-LIFECYCLE-BLIND** *(moved here from W-5 — this is the slice where the violation would land)*. | walk up to an NPC in a realm that spun up 10 s ago; it is where its routine says; talk to it; it does **not** keep its realm alive. |
| **W-4** | The journal tee: the sim→bin handoff resource, **custodian-owned** sealed content-addressed segments, per-`WorldFactKind` shed with `JournalGap`, the `WorldEpoch`/`FORM_VERSION`/settle-on-change reader, §7.5's rebalance sweep. | G-WL-SCHEMA-FLOOR (all six cells), a journal-shed chaos cell (game continues byte-identically vs an unpartitioned control), **the mid-scenario-shed cell of G-WL-ITEM-CONSERVATION**, segment idempotence across a kill-9 between `rename` and ack. | `vdctl worldline-history <realm>` reproduces the state offline from the journal alone; `vdctl worldline-rebalance` is resumable and audited. |
| **W-5** *(REQUIRED for the LAW-E2 claim past the horizon — user decision W5)* | The **coarse agent tier** over the sparse installed-capacity subset, on the **static** pin, **behind an injected port wired only in `vd-bins`** (§9.3): `SiteAgent`, `Flow` **as an absolute-rebase pair** (§9.4), `FactionAgent` structural change, `unrest_bp`. **Flow-driven `KeepAlive` DELETED.** | a flow-conservation oracle (`Σ departed == Σ arrived + Σ declared_loss`) **with a kill-9-mid-flow cell** and a RED control, a pin gate (soak with zero players; un-pinning reaps normally; kill-9 rehydration byte-identical), G-IDENTICAL declared as "passes on ≥2 kinds with the port ABSENT", a release-only round-latency budget, and the pinned host's **bounded child count** (§5.7). | return after a month and find a station that a faction **built**, and a convoy you can follow. |
| **W-6** *(later)* | Shard-side Store B; RULE WL-ACK's irreversible-mint barrier becomes local; item positions durable; the `BlockEdit` fold. | the durable half of G-WL-ITEM-CONSERVATION; **G-WL-AGGREGATE-BOUND**. | inventory survives a shard kill-9; **and the refinery demo from W-2 now runs end-to-end**. |

### 11.6 THE HR6 SURFACE, PER SLICE — and the fast-forward facility every dormancy gate needs

⚠ **Revision 1 gave each slice a "works in-game" column (right) and enumerated NONE of what HR6 means in this
repo (wrong).** HR6 is DevState counters + `WaitField` predicates + `runs/` manifests, and `WaitField` is a
**closed 6-arm CLIENT-side enum** (`crates/devproto/src/predicate.rs:14-32`, `ALL` at `:51`, with
`is_boolean` and `ALL` both requiring updates per arm) — so "walk up to an NPC and it is where its routine says"
and "the same stocks after a reap" need **new arms and an assertion path that does not exist**. Revision 1
mentioned exactly two ad-hoc counters (`worldline_digest`, `worldline_recompact_residual_minor`).

| Slice | New DevState counters | New `WaitField` arms | `vdctl` verbs | `runs/` manifest fields |
|---|---|---|---|---|
| W-0 | — | — | `worldline-eval <realm>` | the seed digest |
| W-1 | `worldline_digest`, `worldline_seeded`, `unseeded_live_realms`, `worldline_pending_devs`, `worldline_stale_fence_rejects`, `worldline_ack_barrier_ms`, `worldline_provisional_devs` | `WorldlineSeeded`, `WorldlineDigest`, `WorldlinePendingDevs` | `worldline-read`, `worldline-devs` | custody choice, digest, pending count |
| W-2 | `worldline_recompact_residual_minor`, `worldline_breakpoints_crossed`, `worldline_season_index` | `WorldlineBreakpointsCrossed` | `worldline-fastforward`, `worldline-subject <id>` | the horizon set actually exercised |
| W-3 | `npcs_materialized`, `npcs_promoted`, `npc_pose_discontinuities`, `realm_reaped_with_npcs` | `NpcsMaterialized`, `NpcsPromoted` | `npc-list`, `npc-interact` | per-NPC pose/velocity at the band crossing |
| W-4 | `journal_segments_sealed`, `journal_gap_facts`, `journal_unsealed_on_reap` | `JournalSegmentsSealed` | `worldline-history`, `worldline-rebalance` | the gap ledger |
| W-5 | `flows_departed`, `flows_arrived`, `flows_in_transit` | `FlowsArrived` | `flow-list` | the flow conservation sum |

⚠ **AND THE 30-UNIVERSE-DAY CRITERION HAD NO MECHANISM.** `CeilingClock::advance()` hands out **exactly ONE
tick per call** under a durable ceiling check (`crates/node/src/universe_clock.rs:122-141`), and `recover` is the
only forward **jump** (and it is crash-shaped). So 51.84 M advances was the only path to this design's flagship
demo — unpriced and unspecified. **Owed as a real, small facility:** a **harness/dev-profile universe-tick
seed** (or a bounded, audited ceiling jump), **fail-loud in `Profile::Cloud`** via `enforce_cloud_preflight`'s
existing discipline, exposed as `vdctl worldline-fastforward <ticks>`, with its wall-clock budget stated. Without
it **none** of `G-WL-DORMANT-MOVES`, `G-WL-CLOSEDFORM-EQ-ACCUM` or W-2's in-game criterion is runnable.

### 11.5 Proposed `DEFERRED.md` ledger entries (continuing from **D-67**)

Highest existing id in `docs/design/DEFERRED.md` is **D-46** (plus the `D-RLM-2/4/5/6` family), and
`scripts/economy_research_20260726.md` §10.4 proposes **D-47…D-66**, so this block starts at **D-67** and
does not collide. Format follows the file's convention: WHAT / WHERE / WHEN / DEPENDENCY / PIN.

| Id | WHAT is missing | WHERE | WHEN | Dependency | Pin (exists-to-be-flipped) |
|---|---|---|---|---|---|
| **D-67** 🟥 | **The dormant-world substrate design + its `docs/design/` home.** This document is a `scripts/` plan; `docs/design/` has no worldline row and `PLAN.md` has no dormancy row. | `docs/design/worldline.md` (absent) | Design before P4; W-0 with it | this document + §12's decisions | this entry; `PLAN.md` gains a worldline row |
| **D-68** 🟥 | **`FieldReplenish` (continuous regeneration racing production).** Rejected for v1 on closed-form-algebra grounds: no O(1) form, needs a bounded quantum loop with a `max_quanta` cliff. | `vd-core::worldline` `RegenPolicy` | if/when gameplay demands continuous belts | W3's decision; the quantum-grid design | `RegenPolicy` has exactly two arms and a test asserting the enum is closed |
| **D-69** 🟥 | **`PwlCurve` multi-stage chain composition** (a machine drawing directly from another machine's live output). v1's world rule ("a machine draws from a STOCK") makes it unnecessary. | `vd-core::worldline` | with P6 conveyor blocks | the breakpoint-merge design + `MAX_SEGS` validation | a doc-comment on `advance` naming conveyors as the owed consumer |
| **D-70** 🟥 | **Cross-realm player-changed logistics routes.** v1 lets two realms disagree at a boundary (the frame-authority law's own precedent); a player blockade/buyout needs the deviation fanned to the affected sibling. | `vd-wire` (fan arm #3) + `vd-sim` | P8+ | D-67, the sibling-fan arm | the seed-derived route function exists and both realms agree on it |
| **D-71** 🟥 | **The coarse agent tier** (autonomous dormant HISTORY: flows, faction install/decommission, provenance). v1 advances state but generates no events. Category C. | `vd-life` or `vd-sim` + the pinned host | optional, after W-4 | W5 + W6 decisions; the static pin; a storage-topology answer (3.4 GB/yr vs a 256 Mi ordinal-bound RWO PVC ⇒ 29 days to full, so compaction is day-one) | `G-WL-LIFECYCLE-BLIND` passes with the tier absent; the pin config field exists and is empty |
| **D-72** 🟥 | **Per-NPC durable identity across dormancy.** A promoted NPC has history; an anonymous one has only its seed-derived ordinal, so "the NPC I befriended" has no continuous life story unless promoted. ⚠ Revision 2 supplies the missing half of the identity question — the **`NpcId ↔ EntityId` binding is a deterministic derivation, not a fresh mint** (§6.3) — so what remains deferred is the *narrative* continuity, not the identity. | `vd-core::worldline` NPC arms | with the first NPC-relationship feature | D-67; the demote-on-re-convergence-only policy (§6.3, replacing the deleted `promotion_ttl_ticks`) | `max_promoted_subjects` exists, is enforced at promotion time with graceful degradation, and the promotion path is idempotent |
| **D-73** 🟥 ⚠ *re-aimed* | **A PAGED `Store` read — `scan_from(prefix, after_key, limit)` — and, separately, a point `get`.** ⚠ Revision 1 ledgered only the point read; the **real** forcing need is the paged cursor, because `RedbStore::scan` materialises an entire prefix run with two heap copies per row (`crates/io-prod/src/store.rs:751-776`) and any family-scale operation (a background compaction sweep, §7.5's rebalance, Tier-B at scale) is unimplementable inside the custodian's memory limit without it. A frozen-seam edit touching `MemStore`, `RedbStore` and the fail-loud adapter contract (a read fault must PANIC, not return empty). | `crates/sim/src/io/mod.rs` | the paged scan **with the first whole-family operation** (§7.5's rebalance); the point read if/when a fold-at-boot proves insufficient | W12 | the exact-key-`scan`-as-pseudo-get idiom is used and documented (`crates/node/src/rlm_spawn.rs`); a source assertion that no code path prefix-scans a worldline family (G-WL-CUSTODY-BOUND) |
| **D-74** 🟥 | **The journal ARCHIVE tier (Warehouse).** 2.3–4.1 TB/yr cannot live on a shard PVC (232 MB/yr mean realm ⇒ ~14 months; 2.3 GB/yr hot ⇒ 9×/yr). v1 sheds physical facts LOUD with a declared `JournalGap`. | `vd-worldline-archive` (absent) + io-prod segment writer | W10; when there is history worth keeping | D-67, D-52 (the econ log shares it) | `JournalGap` is emitted, counted as a FAULT, and pollable via `vdctl`; the ring's `local_segments_max` is enforced |
| **D-75** 🟥 | **Fault isolation for the worldline path.** The crate graph is not a panic boundary; a panic in worldline code still kills the tick and takes pose authoring with it. v1's answer is RULE WL-PANICFREE + G-WL-NOPANIC (discipline, not isolation). | `vd-sim` schedule / a future sandbox | if a real isolation mechanism is ever wanted | D-67 | G-WL-NOPANIC exists and the zero-`unwrap` source assertion passes |
| **D-76** 🟥 | **`AoiConfig` radius-convention doc fix + the design-doc corrections this document names**: `scripts/realm_lifecycle_design.md:176-179` inverts inner/outer; `crates/wire/src/intershard.rs:9` says "16 arms" (there are 23); `crates/io-prod/Cargo.toml:18-21` asserts a dev-dep that does not exist. | the three files | with W-0 | — | each file's line is corrected and a test/comment pins it |
| **D-77** 🟥 ⚠ *new* | **A universe-tick FAST-FORWARD facility.** `CeilingClock::advance()` yields ONE tick per call (`crates/node/src/universe_clock.rs:122-141`), so every 30-universe-day gate and W-2's in-game criterion has **no runnable mechanism** today. Owed: a harness/dev-profile tick seed or a bounded audited ceiling jump, fail-loud in `Profile::Cloud`. | `vd-harness` + `vd-node` + `vdctl` | **with W-2** — it BLOCKS the flagship demo | `enforce_cloud_preflight`'s profile discipline | `vdctl worldline-fastforward` exists and refuses in Cloud |
| **D-78** 🟥 ⚠ *new* | **The AoI/demand child ROSTER decoupled from the `MAX_REGIONS = 64` containment bitset.** Verified: `child_placements` filters the same `Vec` that `guard_regions_nest` caps at 64, and `aoi_decide` consumes it — so **≤58 direct children ⇒ a maximum of 3 364 systems in the entire universe** (§5.7). The `RealmPath`-prefix bucketing that fixes it is, in Design B's own words, "a design, not a landed pattern". | `vd-sim` + `vd-core` | **W-0b — it GATES W-2 and it constrains the key** | the branching-factor boot assertion; D-79's depth headroom | a boot assertion fails LOUD when max branching > `MAX_REGIONS − lineage_depth − 1`, naming both numbers |
| **D-79** 🟥 ⚠ *new* | **Retention / GC of touched-then-abandoned realms.** `G-WL-ZERO-BYTES` only asserts a **never**-visited realm is empty; the touched set is **monotone and never shrinks**, so the "1 % ever visited" assumption that carries every storage number is unjustified over years of play. Owed: tombstone-back-to-seed when a baseline has re-converged to its pure-seed evaluation within the declared width. | `vd-core` + the custodian | after W-4, before real longevity | RULE WL-INTQ (the seed evaluation must be reproducible) | a `worldline_touched_realms` gauge exists and a tombstone path is tested |
| **D-80** 🟥 ⚠ *new* | **A PROMOTED subject crossing a realm boundary.** v1 forbids it (§6.3), because the subject row's realm attribution and the entity's authority would be **two settlement paths with no reconciliation** — a later adopt of the origin realm re-materialises the NPC (duplication), and the per-realm promoted population transfers across a sealed boundary with no arm and no conservation identity. Owed: move the attribution on the **same transfer saga** that moves the entity, so there is one commit point. | `vd-core::worldline` + the saga | with the first cross-realm NPC feature (P8+) | D-67; `verify_population_conservation` | the v1 restriction is asserted (a promoted subject cannot re-home) and `G-WL-POPULATION-CONSERVATION` passes |
| **D-81** 🟥 ⚠ *new, BLOCKING for factions/corporations/reputation* | **An addressing scheme for CROSS-REALM NON-SPATIAL state**: factions, wars, politics, reputation, corporations, currencies, alliances, multi-realm quest chains, trade routes. Every worldline subject is **realm-keyed**, and report D2 already established there is no way to spawn or protect a non-spatial node. Candidates: owning-realm-of-record + fence; ancestor-escalation to the members' LCA; the static pin (W6). ⚠ Revision 1's §6.5 claimed genericity over this class without an addressing answer. | `vd-core` + `vd-node` + the pin | **before any of those systems is designed** | D-67, W6, report D2 | §6.5's three-shape table names it; no `SubjectKind` arm exists for shape (iii) |
| **D-82** 🟥 ⚠ *new* | **`PwlCurve`-free cross-realm COUPLED dynamics** (the re-scoped D-70's algebra). Not a wire slice: "solve or bound a coupled piecewise-linear system across sealed authority domains", inheriting W3(c)'s rejected shape (a bounded quantum loop with a `max_quanta` cliff). v1's answer is that dormant cross-realm flow is **ZERO**, validator-enforced. | `vd-core::worldline` | P8+, with D-70 | D-69's breakpoint-merge design | `G-WL-NO-PHANTOM-INFLOW` passes and the out-of-realm-source validator refuses |
| **D-83** 🟥 ⚠ *new* | **Fault- and liveness-isolation for the economy port.** RULE WL-LIVENESS bounds time and allocation by discipline + a gate, not by isolation; the bounded double-buffered queue is the interim. A real isolation mechanism (a separate thread with a hard deadline, or a sandbox) is owed if the port ever hosts third-party logic. | `vd-sim` schedule | if/when the port hosts anything untrusted | D-75 | `G-WL-ECON-LIVENESS` exists with its hostile-econ cell |

---

## 12. ALTERNATIVES + DECISIONS FOR THE USER

### 12.1 The designs not chosen, and why

**Design B — PAHA (parent-authored hierarchical aggregate). NOT BUILT; three rules grafted.**
Its mechanism is **inactive in exactly the regime LAW-E2 tests**, by its own admission: `ancestor_close` of
an empty desired set is empty (`crates/sim/src/rlm.rs:498-512`), so at 99 %-off even the galaxy is reapable
and there is **no author at any level** ⇒ B degenerates to this design. Even when a parent IS alive it
authors only DIRECT children, so the enlivened set is the player's own neighbourhood — which AoI spins up
anyway. It also has an **unaddressed fence defect**: single-writer is decided by `running_live`, a fact
resolved in the **orchestrator's** ledger that a parent shard cannot legally obtain (no cross-shard query;
the directory is orchestrator-owned), so inferring liveness from push-absence gives **split-brain under
partition** — a partitioned live child stops pushing, the parent resumes authoring, both write. And its
coupling makes state Category C, forfeiting same-seed reproducibility permanently. **Grafted anyway**:
LAW-WL-2/WL-3, LAW-WL-7 + G-WL-LIFECYCLE-BLIND, the orchestrator-relayed handoff, the Jacobi discipline,
the mutation-time budget rule, and prices-as-a-forward-skipped-TLV-tag. **Rejected specifically**: the
ack-gated Kill and `LifecycleAction::Checkpoint` inside the sole kill authority.

**Design C — The World Journal. NOT the dormancy mechanism; its record discipline grafted wholesale.**
Its dormant *advance* is this design's monotone integral plus a 720-step population map, so a player flying
into a 30-day-dormant system sees **exactly the same thing** — while C additionally requires an always-on
Tier-B Warehouse and a 2.3–4.1 TB/yr archive, has **no** no-pop mechanism at all (no NPC ephemeris, no
materialisation band, no continuity gate), and pays the largest HR5 tax in the field (a 5th `sim::io` trait
instantiated in every integration test binary). It buys **HISTORY**, which §4.6 keeps as a separate,
optional record. **Grafted**: sealed content-addressed segment files (with the verified `flock` argument),
the same-`commit()`-batch atomicity target, `WorldEpoch` + settle-on-change + `Refused`/`Partial`,
per-`WorldFactKind` shedding with declared gaps, `kind_bitset` staleness without reading facts, the
never-journal-seed-derivable-behaviour volume rule, the versioned checkpoint/restore discipline,
`verify_item_conservation` as an oracle sibling, and the rejection of `DirectoryKey::Account`. **Also
rejected here, and worth recording**: C's `G-JOURNAL-FAITHFUL` as written ("journal replay == the live World
byte-identically") **cannot hold** across the physical shed C itself specifies — the gate must be
`JournalGap`-aware or it will be quietly relaxed, which is the gate-decay hazard.

**Design D — CAT (coarse agent tier on a pinned galaxy spine). NOT the substrate; admitted as an optional
LAYER, with its fatal flaw repaired.** Disqualifying **as specified**, because `pin_realms` writes into
`desired_alive` arm A and its flow-driven `KeepAlive` forms an unflagged **monetary → process-topology**
chain (§3.4). Its cost also inverts the premise: an always-on 2-process spine per galaxy charged to the
GAME, ~101 processes and 6–12 GB always-on at 100 galaxies for ~0.06 % of a core; density is `10⁴ sites /
10⁵ systems = 0.1 sites/system`, so ~90 % of the galaxy has no economic content (deep but **sparse** — the
mirror of this design's dense-but-shallow); storage is 3.4 GB/yr against a 256 Mi RWO node-local
ordinal-bound PVC = **29 days to full** (`268 435 456 / 9 216 000 ≈ 29`); and it must implement this
design's closed form **anyway** as its `max_catchup_rounds` fallback, so it is a strict **superset**.
**Grafted**: the ~90-line static PIN (the only legal way to host any always-on tier — verified:
`record_demand` is source-agnostic and `TearDown` is a no-op, `crates/sim/src/rlm.rs:266-282`), the
`G-EMPTY-SESSION-ONLY` rule, ~~`PolicyWeights` as the economy→game return shape~~ (⚠ **this graft is WITHDRAWN in
revision 2** — a *total advisory value* is still a value a deviation-authoring path reads, so it made the physical
worldline a function of monetary state; the economy→game direction is a journaled `EconCommand` instead, §3.2),
`unrest_bp`, the resource-
dimension honesty, the 20× CPU correction, the negative-cell discipline for no-pop gates, and the X4
anti-overselling anchor. **Also recorded**: D's rejection of a *synthetic* economy realm, with its precise
defect — a synthetic per-galaxy realm at System level lowers to `RealmId::System(seed)`, `lowered()` is
documented LOSSY with System seeds colliding across galaxies, and the reconciler keys the directory on
exactly `lowered()` (`crates/sim/src/rlm.rs:596`), so **two galaxies' econ realms COLLIDE in the
directory**. Keep this so nobody re-proposes it.

**Other rejected shapes, recorded so they are not re-proposed:** one global journal with a single writer (a
new arm carrying ~100× gameplay traffic through the ONE reviewed HR1 file, or a centralised sequencer that
re-creates the hotspot node-per-realm exists to shard away); per-realm redb tailed by an out-of-process
sidecar (⚠ **downgraded from "technically impossible" to "NEEDS VERIFICATION"** — the claim rests on redb taking
`flock(LOCK_EX|LOCK_NB)` at open, which is ⚠ **[U]** in the report and was **not re-verified here**, so it may not
carry the weight of rejecting a whole alternative; it is rejected anyway on the same custody grounds as W2(b), and
that argument does not depend on the lock); a
Dormant tier that ticks cheaply (no `Dormant` state exists, and D-RLM-4 forbids the mode fork); floating-
point closed forms with a tolerance (cross-host libm bit-equality is unproven; "within epsilon" has no
meaning for an integer stock); a relative-delta event log (neither idempotent nor commutative); a cargo
feature on `vd-sim` (Tier-A `#[cfg]` doubles the HR5 surface); banning floats via `clippy.toml`
`disallowed-types` (silently does nothing).

### 12.2 The decisions the user must make

**W1. Crate topology / port direction.** (a) report §7.1: `vd-econ` **below** `vd-sim`, `vd-sim` drives it
— requires a `sim → econ` edge; LAW-E1 becomes a runtime flag only. (b) `vd-econ` strictly **above**
`vd-node` — zero edge, but projection-only; it can never clear a market inside a tick. (c) **the seam
inversion**: object-safe port DEFINED in `vd-sim`, impl in `vd-econ`, edge **econ → sim**, injected
`Option<Box<dyn …>>` from `vd-bins`. **RECOMMEND (c)** — the only shape where LAW-E1 is structural AND a
shard can still do authoritative economic work on a tick, and it mirrors `Store`/`RealmSpawner` exactly.
**⚠ CHANGES report §7.1/§7.1.1.**

**W2. Where the worldline's durable state lives.** ⚠ **Revision 1 offered only two options and BOTH FAIL; the
one that actually works was missing.**
- **(a) the orchestrator's `Store` A with new family tags** — buildable today, one durable always-live
  crash-recovered process, no realm↔PVC identity problem. **BUT the orchestrator is `requests == limits`
  512 MiB GUARANTEED QoS** (`deploy/k3d/30-orch.yaml:65-67`) with zero burst headroom, its 1 Gi `data` PVC
  (`:95`) also holds the directory, the saga WAL and the clock ceiling, `RedbStore::scan` materialises an entire
  prefix run with two heap copies per row (`crates/io-prod/src/store.rs:751-776`), and a restart drops the
  **RAM-only** demand ledger (`crates/sim/src/rlm.rs:227-236`, D-RLM-2) — so an OOMKill is a **crashloop that
  takes the whole universe's realm lifecycle with it**, and a large first read after a restart trips
  `ProbeTuning::DEFAULT.stall_deadline_ticks` (`crates/node/src/health.rs:50`) on the one process the manifest
  itself calls "the ONLY node holding un-fsynced durable state". Revision 1's "~5.4 MB for 100 k realms makes the
  chokepoint objection evaporate" is safe **at its own toy number** and fails ~one order of magnitude up.
- **(b) per-realm Store B** — **blocked**: the k3d PVC is `ReadWriteOnce` + `local-path` + bound to the
  StatefulSet **ordinal**, not a `RealmId` (`deploy/k3d/50-shard.yaml:92-96`), so a respawned realm cannot reach
  its own store.
- **(c) hybrid** — two code paths for one property, the exact HR3 smell.
- **(d) ⚠ NEW — a dedicated `worldline-custodian` StatefulSet**, sharded by the secondary path key's prefix
  range, with **its own PVC and its own memory budget**. The orchestrator stays the clock/directory/CAS authority
  and **never holds worldline bytes** (which also removes the LAW-E1-in-the-availability-direction hazard: with
  (a), worldline write load or redb growth degrades **every transfer saga in the universe**, since the directory
  is the only commit point and `commit` is block-on-prior — the `launch.redb` precedent at
  `crates/bins/src/bin/orchestrator.rs:234` is the in-tree pattern for exactly this separation, and
  `StoreKey`'s own doc notes the Directory family was given a distinct prefix so it can be split out "without a
  cross-file atomic transaction", `crates/node/src/saga_runtime.rs:82-88`, the D-32 seam).

> **RECOMMEND (d)**, with **(a) permitted for dev/single-node only**. In **every** option three things are
> mandatory: compaction is **lazy-on-adopt** so no code path prefix-scans a family (§4.5); a **paged
> `scan_from`** exists before any whole-family operation (D-73); and the worldline families live in **their own
> redb file with their own writer**, never sharing a commit path with the directory or the saga WAL. And put
> explicit numbers in the design — an OOM ceiling in rows-per-read derived from the chosen memory limit, a stall
> budget derived from `stall_deadline_ticks`, and a PVC budget that **includes redb B-tree/free-space overhead**
> (~1.5×) — all measured by W-(−1) and gated by **G-WL-CUSTODY-BOUND** + **G-WL-CUSTODIAN-THROUGHPUT**.
> **⚠ CHANGES report D3.**

**W3. Regeneration policy algebra.** (a) `Finite` only. (b) `Finite` + `RespawnAfter(ticks)` (a **step**).
(c) + `FieldReplenish(rate)` (continuous). **RECOMMEND (b)** — exact, O(1), one extra breakpoint, and it
still delivers renewable belts. Rejecting (c) is a **determinism-algebra** call, not a gameplay one: it has
no O(1) form and forces a bounded loop with a `max_quanta` cliff. **⚠ SHARPENS report D16**: the report's
recommended (c) "bulk ores respawn, rare deposits are finite" is **accepted**, provided "respawn" means
`RespawnAfter` (a step), **never** continuous replenishment.

**W4. Hopper-full policy default.** (a) `Stall` (a full refinery stops eating ore). (b) `Spoil(SinkId)`
(keeps consuming, posts the overflow to a declared sink). **RECOMMEND (a)** — no sink, no accounting,
physically the more believable behaviour; keep (b) for perishables. Same cost either way.

**W5. Is the coarse agent tier built at all, and when?** ⚠ **Revision 1 framed (a) as "a legitimate answer the
user should choose knowingly" and described its cost as "no wreck to find". THREE INDEPENDENT RESULTS SAY THE
COST IS MUCH LARGER, and the user must choose against the real description.** Choosing (a) means choosing all
three of:
1. **Dormant regions are FROZEN after roughly a month** — 23.15 days in §5.4's own machine example, 10–110 days
   for a cohort depending on `gn/gd` (§4.3a). §4.3a's seasonal driver pushes this out and is now mandatory, but a
   *driver* produces evolving state, not **events**.
2. **ZERO dormant inter-realm flow** — cross-realm supply is algebraically incompatible with the substrate
   (§7.2 #7), and W-5 is the **only** mechanism in the design that can move material between dormant realms at
   all. LAW-E2 names "Ports, refineries, space stations"; **ports cannot function dormant without it.**
3. **MONOTONE DECAY ONLY** — nothing dormant adds capacity, discovers a deposit, completes a construction or
   repairs anything (§1.2). The parts of the galaxy a player invested in only get worse; the parts they ignored
   are static.

Options: (a) never; (b) after W-4, over the sparse installed-capacity subset, on the static pin, **behind an
injected port** (§9.3) — D-71; (c) as part of v1. **RECOMMEND (b), and D-71 is re-labelled a named PREREQUISITE
of the LAW-E2 claim rather than an optional layer.** (a) remains a defensible **scope** choice — "my base is
exactly as I left it, minus wear, and dormant space is quiet" is a coherent game — but it must be chosen with the
three consequences above written down, which is why §1.2 property 1 is now scoped to the believability horizon
rather than stated unconditionally.

**W6. Is the PIN adopted, and what is pinned?** (a) no pin — nothing is always-on; the galaxy is reapable.
(b) pin the Universe + Galaxy realms as a **static, economy-independent deployment constant**, with the
flow-driven `KeepAlive` deleted and `G-WL-LIFECYCLE-BLIND` proving topology blindness. **RECOMMEND (b)
only if W5 is (b)/(c)**; land the ~90-line mechanism regardless, because it is also the honest way to host
the report's D2 option B (a pinned **spatial** realm rather than an unspawnable non-spatial node). ⚠ State
the always-on cost in the deployment budget rather than discovering it there.

**W7. The economy-absent gate's scope.** (a) all three arms in `just gate` (highest confidence; roughly
doubles the `test` step of an already-16-step gate). (b) two arms in `gate`, the third release-only.
(c) a **named scenario subset** covering every emit site, with an anti-vacuity assertion, plus a release-only
full three-arm run. **RECOMMEND (c).**

**W8. Item durability.** (a) all stacks `Durable`/`LossBudget::ZERO` — kills the batched-transient
amortisation. (b) stay `Transient` with loss posted to a declared sink. (c) **items inherit their
CONTAINER's durability**. **RECOMMEND (c)** — a gameplay-policy call with real machinery consequences; it
is the option the report omits.

**W9. Is production subjective or objective time?** (a) OBJECTIVE (orbit-like; no `time_multiplier`).
(b) SUBJECTIVE (occupant-like; multiplier applies). **RECOMMEND (a)** — the standing law says the
multiplier must never touch a parent-authored/objective quantity, and nothing in code applies it for you.
Changes every rate formula, so decide before W-2.

**W10. Is the journal archive built, and when?** (a) never — physical facts shed LOUD with a declared
`JournalGap` and history is best-effort. (b) at W-4, as sealed segment files + a Warehouse (D-74).
(c) at v1. **RECOMMEND (b)** — and note the world advances byte-identically in all three, which is the
point of the two-record split.

**W11. `raw_fact_retention_days`.** The report's D17 recommends per-event audit granularity; retention sets
the moderation/appeal window and the storage bill (90 days ⇒ 0.57–1.0 TB retained hot). **RECOMMEND 90
days** as a named `WorldlineTuning` field, calibrated from our own dashboard.

**W12. Does `Store` gain a point read (`get`)?** (a) no — exact-key `scan` as the pseudo-get plus
in-memory folds rehydrated at boot (this design's assumption). (b) yes — a frozen-seam edit touching
`MemStore`, `RedbStore` and the fail-loud adapter contract. **RECOMMEND (a)**, ledgered as D-73.

**W13. Reserve `DirectoryKey::Account(AccountId)`?** (a) reserve it inert (~10 lines, report S6). (b)
**reject it**. **RECOMMEND (b)** — the directory CAS is the only commit point, so an economy key there
makes an economy fact an authority input. **⚠ CHANGES report S6.**

**W14. Does the economy ever get a `KeepAlive` lever?** (a) yes (the report's §6.4 requirement 2: keep a
market realm warm). (b) **no, ever** (LAW-WL-7 + G-WL-LIFECYCLE-BLIND). **RECOMMEND (b)** — (a) makes
*which realms are alive* a function of economic state. **⚠ CHANGES report §6.4 requirement 2, which is a
defect.**

**W15. Dormant history depth.** (a) statistical rates only — with §4.3a's seasonal driver, evolving but
**event-free**: populations rise and fall believably, no individual battle happened, no wreck exists, **and no
salvage exists outside AoI** (§6.6). (b) agent-generated events (W5(b)). **This is the same decision as W5 seen
from the believability side** and it inherits W5's three corrected consequences; it is listed separately because
it is a **gameplay-policy** choice the user should make consciously rather than inherit from an arithmetic
constraint.

**W16. Who acks a player's economically-significant mutation?** ⚠ **Restated, because revision 1's (a) put a
single central writer on the critical path of the PHYSICAL layer — a coupling on the availability axis strictly
worse than the crate-graph coupling LAW-E1 forbids, and absent from §2's verdict table.** Options:
(a) the custodian, before **every** player ack; (b) **the reshaped WL-ACK of §4.6** — the shard is
authority-of-record, budgets are enforced **locally before the apply**, **optimistic ack for pure absolute
rebases** (mining, installing, draining — most of gameplay), and the durability barrier on `flush` applies
**only to irreversible mints**; (c) fully local, once shard-side Store B exists.
**RECOMMEND (b) for v1, (c) at P6/P7.** (a) is rejected: it makes "the orchestrator is down" mean "**no player
anywhere can mine, build, install or destroy anything**", it serialises every action behind one process's fsync
at depth 1, and revision 1 never said whether the ack waited on `commit` (which can lose an acked mutation) or
`flush` (which caps global throughput). §2 gains a **Custodian** column so no row reads as unconditionally
available.

**W17.** ⚠ *new* — **Does the economy ever get an advisory value the game READS?** (a) no — the port is
**emit-only + `EconCommand`** (§3.2). (b) yes, but only tick-aligned, fence-stamped, **recorded in the game's
journal at the tick it was applied**, and carried by a reviewed arm with a stated `effect_class`.
**RECOMMEND (a)** — (b) is strictly more machinery for less capability, and revision 1's unfenced, unjournaled
`weights()` made the physical worldline a function of monetary state, breaking the flagship gate and §8.4's
replay obligation.

**W18.** ⚠ *new* — **Is per-block history worth ~1.3 TB/yr?** 833 of the ~2 300 facts/s are block place/break
(§5.4). (a) journal them per-block (~3.5 TB/yr). (b) coarsen to `BulkPlaced`/`BulkDestroyed` summaries
(~2.2 TB/yr). **RECOMMEND (b)** — §11.3 already proposes the schema, and the state record never needed them.

**W19.** ⚠ *new* — **How is the fan-out ceiling resolved (D-78)?** (a) decouple the AoI/demand child roster from
the `MAX_REGIONS` bitset and land the `RealmPath`-prefix bucketing. (b) accept an intermediate Sector/Cluster
level (a 7th `RealmKindTag`) and size the key for it. **RECOMMEND (a)**, because (b) also forces a wider key and
a new tag; but **it must be decided before W-0** since ≤58 direct children caps the universe at **3 364
systems** (§5.7) and the answer constrains §11.1's key.

**W20.** ⚠ *new* — **Where does cross-realm non-spatial state live (D-81)?** (a) owning-realm-of-record + fence.
(b) ancestor-escalation to the members' LCA (report §6.4 option C). (c) the static pin (W6). **No
recommendation** — it needs the same options-and-tradeoffs treatment the report gave D2, and it is **BLOCKING for
factions, corporations and reputation**, which are systems the user named.

**W21.** ⚠ *new* — **Is `min_dormant_ticks` adopted (the short-end fix, §5.6)?** (a) no — a 5-minute round trip
pays a full ~3.2 s wake for a 0.015 % state change. (b) yes — a per-realm **time** hysteresis below which an
empty realm is not reaped (legal under LAW-WL-7: a lifecycle-internal window, not a life or economy value).
(c) D-RLM-4's warm pool instead. **RECOMMEND (b) now and (c) later** — (b) is one `RlmTuning` field.

### 12.3 Corrections to `scripts/economy_research_20260726.md` that are not decisions

1. **§2.3c — the concentrated-agent CPU figure is overstated by 10–20×.** The report says 5 000 noise
   agents × 10³ markets = 5×10⁶ evaluations/round = "1.0–2.0 core-seconds", then concludes ">100 % of a
   core". But its own cadence table sets `agent_decision_period_ticks = 400` = **20 s** at 20 Hz. So
   `1.0–2.0 core-s / 20 s = 5–10 % of ONE core`, not >100 %. The report's distributed row
   (`1.5 / 10³ / 20 = 0.0075 %`) is internally consistent, so the error is confined to the concentrated
   row: it reads core-**seconds-per-round** as **cores**. **The EGRESS branch survives and still rejects
   the split topology**: `5×10⁶ msg / 20 s = 250 000 msg/s ≈ 25 MB/s` at ~100 B/msg. This changes the
   arithmetic under **D2** and **D8**; correct it before anything inherits it.
2. **§6.4's dormant catch-up tier is deleted as a primary mechanism** (§4.7): its own fallback is the
   closed form, so `dormant_catchup_tick_period` and `max_catchup_ticks` shrink to a fallback-only knob for
   the optional agent tier. This also removes the 52-second wake problem rather than bounding it.
3. **§8.1's "the event log must BE the ledger's journal, not telemetry" is SHARPENED into two records**
   (§4.6): the worldline **STATE** is not the journal; the journal is the **history**. Both are game-owned.
   `LossBudget::ZERO` applies to the state and to money-class facts; physical facts may shed with a
   **declared** gap.
4. **§7.7 / `:1411` — NPC agent strategies inside `vd-econ` is a latent LAW-E1 violation** and is
   overruled (§2 row 4, §6).
5. **S4 is broadened**: `econ_epoch` becomes a world-wide `WorldEpoch` in `vd-core`, stamped on every
   durable record, because the physical layer needs it first.
6. **S12 is promoted** from "an input to the faucet" to a **hard prerequisite of the production integral**
   — `S₀` in §4.3 *is* that function.
7. **S9's second half is superseded**: "escrow is held by the ACCOUNT owner, never the market host" remains
   a live D15 question, but the claim that it is one of "the two rules that make RLM teardown safe" no
   longer holds for the physical layer, which needs no teardown-safety machinery at all.
8. **D-54's premise weakens for the physical layer**: `rehydrate` is O(one epoch), not O(history), because
   compaction is a first-class law (LAW-WL-3) rather than a capacity optimisation. It stands unchanged for
   the monetary ledger.

⚠ **FIVE MORE, WHICH REVISION 1 DID NOT DECLARE.** Two design documents in one repo that silently contradict
each other is a defect, and revision 1 declared eight of thirteen. The report has been edited in place at each of
these points with a cross-reference note (its own "Revision 3" header line lists them).

9. **⚠ CHANGES report D2 (`:2647-2681`), not only S6.** D2's recommendation is to key **`Account(AccountId)`** and
   `Market(RealmId, CommodityId)` as directory key families **"from day one"**. W13 **rejects the `Account`
   arm** — the directory CAS is the only commit point, so an economy key there makes an economy fact an authority
   input — which removes **half of D2's mechanism**. Revision 1 said it changed only S6. The replacement for the
   removed half is §6.7: **an account's durable POSITION is a zero-rate worldline subject** at the custodian, so
   single-writer-per-account is a property of the custodian's fence-ordered log rather than of a directory arm.
   `Market(RealmId, CommodityId)` is untouched.
10. **⚠ CHANGES report D8 / §7.7 on the AGENT MECHANISM, not only on placement.** D8 explicitly **rejected trait
    objects** for agent strategies on HR5 monomorphisation grounds and prescribed a `TraderStrategy` **enum
    registry**. §3.2 introduces `NpcStrategy` as an **injected trait** with `NeedsOnlyStrategy` — deliberately,
    because LAW-E1 requires the *implementation* to be absent-able and the enum registry would put economy
    decision logic in Tier-A. The HR5 cost is paid by making the trait **object-safe** (the `Store`/`RealmSpawner`
    idiom, so there is no per-monomorphisation region multiplication) — but it **is** a change to D8's decision
    and it is declared here rather than inherited silently.
11. **⚠ CHANGES report D21 (recurring asset sinks), which had no row in revision 1's layer table.** D21 decides
    storage rent and structure upkeep **YES**. As an unqualified monetary sink, "structure decommissioned because
    rent went unpaid" is a **monetary → physical** coupling LAW-E1 forbids. §2 **row 27** splits it: the
    **PHYSICAL** half is a `Construction`'s **condition** stock draining at a per-entity integer rate (an ordinary
    closed form, so a structure wears and can physically decommission on its own schedule with the economy off);
    the **MONETARY** rent is a `vd-econ` **receivable** that may never itself destroy a structure.
12. **⚠ CHANGES report D22 (`season_id` as a day-one partition key).** D22 says "free now, a migration later" —
    and the **day-one artefacts are exactly the ones this design freezes**: `RealmUid`, the `StoreKey` worldline
    families and the durable record schema. `season_id` is **absent** from all three. **Decide D22 before W-0**:
    if seasons are ever possible, `season_id` joins the key/record now at zero cost; if "never wipe" is decided,
    say so, because it makes retention (D-79, report D-54), escheatment and inequality policy load-bearing
    forever.
13. **⚠ ANSWERS report §7.18 (`:1948-1961`), which revision 1 answered NO by accident.** §7.18 requires that "the
    economy changes nothing in the world" be a **decision, not an omission**. Revision 1's port was
    `observe` + `weights`, so every monetary row was **terminal** and no monetary event could move an item, yield
    an insured hull, pay a wage or deliver a contract. §3.2's **`EconCommand`** is the designed consumer, §2 row
    26 is its verdict row, and the degraded mode is declared (an undelivered obligation is a receivable, exactly
    as a docking fee is).

---

## 13. OPEN QUESTIONS

1. **How does a demand-spawned shard ever receive `ClockSync`?** `ClockPeers` is a static boot `Vec`
   exposed only as `Res` and nothing registers a minted `NodeId`; every authoring system is
   `.run_if(has_synced)`; the process gate says children boot "without ever syncing". **Nothing in this
   design can fix it, and every dormancy claim is undemonstrable until it lands** — it blocks RLM Step 6/7
   equally. Runtime `ClockPeers` mutation, a broadcast to `spawner.live_nodes()`, or a pull-based clock
   request?
2. **When does the lazy per-subtree seed generator land (P4-owed)?** Until then "a never-visited realm
   costs zero bytes" and "the analytics tier runs the same pure `f`" are provable on a single hardcoded
   galaxy only.
3. **Is the economic-significance registry's line drawn correctly at P6?** LAW-WL-5' is the criterion, but
   the concrete per-kind assignment is a judgement made once and expensive to move. Getting it wrong makes
   the custodian the universe's bottleneck.
4. **What is the second occupancy/desire input, if the user ever wants one?** Today `desired_alive` has
   exactly two arms and this design deliberately **accepts being reaped**. If "this station stays warm
   because it has open obligations" is ever wanted, that is a **third input** that does not exist and that
   LAW-WL-7 currently forbids. ⚠ Note the **legal** neighbour of that wish, which revision 1 did not offer:
   `min_dormant_ticks` (§5.6, W21) is a *time* hysteresis charged to the GAME's lifecycle tuning and is
   economy-blind, so it keeps a busy realm warm without letting an obligation vote.
5. **Does `Dormant`-as-a-cheaper-capability ever exist?** D-RLM-4 mandates warm spares be
   profile-**agnostic** blank shards with **no mode branch**, so a "realm that ticks only the worldline" is
   exactly the forbidden fork. This design needs no such tier — but if one appears, does its existence
   change the compaction custody answer?
6. **Can `ShardProfile` ever be per-INSTANCE?** It is a pure function of realm KIND today, so an
   "economy-capable" or "worldline-heavy" realm has no home without forking on kind (HR3 pressure). This is
   report S5b, and it is a real RLM slice.
7. **Which harness tier proves the byte-identity of a *three-arm* run without tripling the gate's wall
   clock?** W7(c) is a compromise; a cheaper mechanism (a deterministic scenario digest compared across
   arms in one process, rather than three suite runs) may exist and has not been designed.
8. **What is the right storage topology if W2(b) is ever chosen?** RWX/networked storage with RealmId-keyed
   volume identity, an explicit handoff step in the RLM spin-down/up saga, or node-pinning — all three cost
   real cloud money and none is designed. Same blocker the report flags for its own option A.
9. ~~**Is `promotion_ttl_ticks` demotion player-visible?**~~ **RESOLVED in revision 2 and no longer open.** The
   TTL branch guaranteed an **unbounded** teleport (§6.3), so `promotion_ttl_ticks` is **deleted**: demote only on
   re-convergence **and** only while unobserved, and bound promoted state at **promotion** time by graceful
   degradation to a cohort-attributed named individual. `G-WL-SEAMLESS-NPC` gains a negative cell so the policy
   cannot silently return.
10. **What bounds the optional faction tier?** There is **no equilibrium proof**: over 10⁴ rounds a
    `FactionAgent` can monotonically strip or monopolise a galaxy. Install/decommission budgets and
    decommission floors must be seed- or rate-derived rather than magic numbers, and this needs
    **playtesting rather than gates**.
11. **Does the burst case need more than `BulkDestroyed`?** A station destroyed in one tick is exactly the
    fact you least want shed, and the right shed *policy* for a burst (as opposed to the mechanism) is
    undesigned.
12. **Is `WorldFact`'s sealed enum the right shape for the P9 Signal projection?** §6.5 claims NPC state
    projects onto Signals cleanly, but P9's arm does not exist and its design has not consumed this input
    yet. Shaping `WorldFact` before that conversation risks a second re-shape.
13. ⚠ *new* — **What is the right shape for cross-realm non-spatial state (D-81)?** §6.5's shape (iii) —
    factions, wars, reputation, corporations, currencies, alliances, multi-realm quests, trade routes — has **no
    addressing scheme anywhere in the architecture today**. This is W20 and it BLOCKS systems the user named.
14. ⚠ *new* — **Is `season_id` (report D22) in the frozen key and record, or is "never wipe" decided?** Free now,
    a migration after the first durable row (§12.3-12).
15. ⚠ *new* — **What is the equilibrium behaviour of the seasonal driver over decades?** §4.3a makes the
    attractor a periodic orbit rather than a fixed point, which fixes the freeze — but nothing proves a
    long-horizon bound on the *composed* system (a seasonal machine feeding a seasonal cohort feeding seasonal
    consumption). `G-WL-DORMANT-MOVES`' 2-year arm is evidence, not a proof, and this needs playtesting rather
    than gates (the same status as open question 10).

---

## 14. REVIEW RECORD (revision 2)

Four adversarial vetters reviewed revision 1 independently, each with its own lens. Verdicts:

| Vetter | Lens | Verdict |
|---|---|---|
| **V1** | LAW-E1 coupling, the fold/compaction algebra, the item/conservation proof | **SOUND_WITH_FIXES** |
| **V2** | LAW-E2 believability at scale, storage/custody/concurrency arithmetic, the gates' anti-vacuity | **MAJOR_REWORK** |
| **V3** | determinism, HR1/HR2/HR3 conformance, wire classification, cornering | **SOUND_WITH_FIXES** |
| **V4** | the full end-goal (economy→world feedback, blocks/ships, player-facing spec), cross-document coherence | **SOUND_WITH_FIXES** |

**Every sustained finding is fixed in place with real content — no TODOs.** The eleven that changed a
CONCLUSION are listed in the header block. The remainder are fixed at their own sections and each carries a
⚠ marker naming what revision 1 said and why it was wrong, so nobody re-imports the old version:

- **§1.1** — E1-c's mechanism corrected (the seal is aimed at the game→econ direction and a `vd-core` enum cannot
  implement a `vd-sim`-private seal; the enforcement is `-> ()` + the crate graph); E1-d restated honestly (the
  IMPLEMENTATION is absent, not the seam); the isolation gate's **third** bypass added (dev-deps, transitive
  closure, allowlist); the `vd-bins`-only feature shown to be invisible to `vd-tests`; `NullEcon`/`MemEcon`
  re-homed to `sim::io::mem`; **RULE WL-LIVENESS** added; the byte-identity property split into P1 (inert
  absence) and **P2 (attributable presence)**.
- **§1.2** — property 1 scoped to the believability horizon and the gate made multi-horizon; property 3 restated
  as live-**but-unoccupied**; the "no positive dynamics" and "perceptible range" admissions added; the unsourced
  X4 anchor replaced with an in-house AoI-derived target.
- **§2** — a **Custodian** availability column; rows 26 (economy→world), 27 (D21 upkeep) and 28 (goods escrow)
  added; row 21's retro-payout claim repaired by **RULE WL-CONSERVED-FACT**; **RULE WL-LIEN** added.
- **§3.1/§3.2/§3.3/§3.4** — the float `deny` corrected to module-level for the worldline; the port rewritten
  (emit-only + `EconCommand`); the feature placed on `vd-tests` too; **LAW-WL-7 re-scoped to the occupant set and
  the AoI membership map, with the real `is_session_occupant` mechanism** and `G-WL-LIFECYCLE-BLIND` moved to W-3.
- **§4** — `muldiv` signatures narrowed and the overflow caps **derived**; `RespawnAfter`'s two regimes;
  **§4.3a the seasonal driver** and **§4.3b the cohort-consumption integral + RULE WL-AGGREGATE + RULE
  WL-SETTLED-RATE**; LAW-WL-2 given the **fence term**, the custodian-minted `seq`, the stale-reject and the
  **no-stored-counter** clause; compaction given `(base_tick, base_seq)`, an exact-key delete, **lazy-on-adopt as
  the only mode**, coalesce-on-ingest, and `LogFull` reclassified as a FAULT; the carrier **split** and WL-ACK
  rewritten as a durability barrier plus **RULE WL-ADOPT-REFUSE**; cadences moved to ticks-from-one-quantum; the
  **Tier-B read path** specified as a real deliverable.
- **§5** — the size arithmetic replaced with **measured postcard** figures and an enforced `MAX_SUBJECT_BYTES`;
  the journal bill corrected to ~48 B/fact; **warm and cold** wake bands; the **concurrency ceiling** stated once;
  the shard-local journal ring moved to the custodian; **G-WL-WAKE-E2E** and the **derived AoI lead radius**; the
  perceptible-range band; **`MAX_REGIONS` shown to bind THIS design** with the 3 364-system consequence.
- **§6** — the underflow-horizon formula; interaction-only promotion; TTL demotion deleted; the two-authorities /
  population-conservation / `NpcId↔EntityId` repairs; **RULE WL-READ**; §6.5 split into three shapes with (ii)
  withdrawn and (iii) ledgered as blocking; **new §6.6 (what the player does)** and **§6.7 (monetary dormancy)**.
- **§7** — the deviation arm's **idempotency key** corrected off "the subject fence"; both arms' **durability
  class** stated and the adopt **livelock** fixed with a level-triggered `WorldlineSeeded`; #7 rewritten as an
  algebraic impossibility with a zero-flow v1 rule and a phantom-inflow gate; the **`WorldEpoch`-on-a-blob
  movement outage** and the real `max_state_bytes` cap; **new §7.5 (the rebalance operation)** and **§7.6 (LAW-WL-8,
  geometry is the authority)**.
- **§8/§9** — the identity's shed-repairability; population conservation; the **integer generator boundary** and
  `content_epoch`; TLV-framing the durable records; `FORM_VERSION`; the **registry split** into
  `PhysicalLossChannel` vs `MonetarySink`; `Sink::UnreportedGap` **deleted**; flows as absolute-rebase pairs; the
  coarse tier's per-instance-capability answer.
- **§10** — the flagship gate renamed/rescoped, `G-WL-RECOMPACT` de-contradicted, `G-WL-DORMANT-MOVES` made
  multi-horizon, and **thirteen new gates** added (G-WL-ATTRIBUTABLE-DIVERGENCE, G-WL-SEAMLESS-FIELD,
  G-WL-WAKE-E2E, G-WL-WAKE-RATE, G-WL-CUSTODY-BOUND, G-WL-CUSTODIAN-THROUGHPUT, G-WL-READPATH-ISOLATION,
  G-WL-AGGREGATE-BOUND, G-WL-NO-PHANTOM-INFLOW, G-WL-ECON-LIVENESS, G-WL-POPULATION-CONSERVATION,
  G-WL-XBINARY-DETERMINISM, G-WL-SUBJECT-ARTEFACT), plus new cells and RED controls on nine existing ones.
- **§11** — `RealmKey` replaced by `RealmUid` + a variable-depth secondary index + `MAX_REALM_DEPTH`; new pre-P4
  seams; W-2 **re-ordered** to an artefact-free demo with the refinery moved after W-6; **new W-0b**;
  **new §11.6 (the HR6 surface per slice + the fast-forward facility)**; **D-77…D-83** added and D-73 re-aimed at
  a paged scan.
- **§12** — W2 given a **third option** and (a) demoted to dev-only; W5/W15 re-framed with their three real
  consequences and D-71 re-labelled a prerequisite; W16 rewritten; **W17–W21** added; the `flock` claim
  downgraded to NEEDS-VERIFICATION; the six-orders-of-magnitude headline deleted as a strawman; **five undeclared
  report contradictions declared** (D2, D8, D21, D22, §7.18).
- **§13** — Q9 resolved; Q13–Q15 added.

### Findings REJECTED or PARTIALLY ACCEPTED on the merits

| # | Finding / proposed fix | Judgment |
|---|---|---|
| 1 | **V2's proposed non-saturating driver (i): "a PERIODIC rate driven by ORBITAL PHASE — the celestial layer already gives closed-form phase".** | **MECHANISM REJECTED; the finding it serves is ACCEPTED.** The freeze is real and §4.3a fixes it — but orbital phase is **f64/libm** whose cross-host bit-equality is ungated (SPIKE-6a, `crates/core/src/celestial.rs:11-18`), so driving a **durable** worldline rate from it would violate RULE WL-INTQ (§9.1) at the exact seam where authority is decided — the very defect V3's CRITICAL #1 raises. A seed-derived **integer season table** indexed by `(t / season_quantum_ticks) mod W` buys identical non-monotone dynamics with none of that exposure, and its integral is exact. Recorded in §4.3a as a rejected alternative. |
| 2 | **V2's CRITICAL #1 row-count thresholds** ("the scan alone OOMKills at 118 000 light rows / 39 000 worst-case / 12 200 built-realm rows / 4 420 worst-case built"), derived from an assumed "~300 MiB of realistic scan headroom". | **FINDING ACCEPTED; THE NUMBERS NOT IMPORTED.** The constraint is verified and load-bearing (`requests == limits: 512Mi` GUARANTEED QoS, `deploy/k3d/30-orch.yaml:65-67`; `RedbStore::scan` materialising a whole prefix with two heap copies, `crates/io-prod/src/store.rs:751-776`; a restart dropping the RAM-only demand ledger) and it drove both the third custody option and lazy-on-adopt compaction. But the headroom figure is an estimate, not a measurement, and this document's own E-(−1) discipline forbids an unmeasured number entering an implementation plan. §12.2 W2 therefore states the **mechanism** and requires the **measured** ceiling from W-(−1) + `G-WL-CUSTODY-BOUND`, rather than printing thresholds we did not measure. |
| 3 | **V2's claim that `KindDef::max_state_bytes` bounds a container blob at "≤64 KiB" (the `u16` type max).** | **CORRECTED IN THE OTHER DIRECTION — the vetter was too generous by 8–16×.** The landed values are `PLAYER_DEF = 4096`, `SHIP_DEF = 8192`, `NAMED_CONSTRUCTION_DEF = 8192` (`crates/core/src/entity_kind.rs:204`, `:213`, `:222`). The finding (name the binding cap and enforce W8(c) against it at mutation time) is **accepted and strengthened**: at §5.4's measured ~55 B/row the in-blob inventory is a few **tens** of stacks, not "a few hundred". |
| 4 | **V3's alternative to the integer generator boundary: "make a baseline ALWAYS materialised by exactly one host at first touch and never re-derived — which forfeits `G-WL-ZERO-BYTES`".** | **PARTIALLY REJECTED as stated.** It does **not** forfeit `G-WL-ZERO-BYTES` (a never-touched realm still stores nothing), and more importantly it does not remove the exposure: a never-touched realm is still **re-derived by whoever reads it**, so a second host still recomputes float-derived inputs. It is recorded in §9.1 accurately as a partial mitigation, and **RULE WL-INTQ is the answer**. |
| 5 | **V2's framing that `G-WL-LIVE-EQ-DORMANT` "cannot be run as written" *and* that what survives is "much narrower"; V3's framing that it is "unachievable".** | **BOTH ACCEPTED, and merged into one resolution** rather than two. The gate is renamed `G-WL-CLOSEDFORM-EQ-ACCUM` with an explicit no-autonomous-authoring carve-out (which still proves LAW-WL-1), §1.2 property 3 is restated to match, and the occupied case gets a **separate attribution** gate (`G-WL-ATTRIBUTABLE-DIVERGENCE`) instead of a weakened byte-identity one. Noted because the two vetters proposed slightly different scopes and only one gate is built. |
| 6 | **V1's CRITICAL #3 alternative: "if the user WANTS prices to steer NPC hauling, then G-WL-LIVE-EQ-DORMANT becomes economy-off-only and the physical journal is no longer self-sufficient".** | **ACCEPTED AS AN ALTERNATIVE, NOT AS THE RESOLUTION.** It is recorded verbatim as **W17(b)** with its three conditions (tick-aligned, journaled at the applied tick, carried by a reviewed arm). The recommendation is W17(a) — delete `weights()` and use `EconCommand` — because it delivers *more* capability (the economy can actually move goods, report §7.18) for *less* machinery, and it keeps the physical journal self-sufficient. |

**One structural note on the review itself.** V2 returned MAJOR_REWORK and was right to: three of its five
CRITICALs (the believability horizon, the fan-out ceiling, the custody wall) are properties of the design's own
arithmetic that revision 1 computed and then did not conclude from, and two of them (§4.3a's driver, D-78's
roster) change what W-0/W-2 must contain. The other three vetters' CRITICALs were all **local defects with local
fixes** (the fold's missing fence term, the compaction reader rule, the port's direction, the occupant fold, the
float generator boundary, the promotion authorities, the voxel bridge, the machine rate) — serious, but repairable
in place, which is why their verdicts stand at SOUND_WITH_FIXES and this revision is a revision rather than a
fifth design.
