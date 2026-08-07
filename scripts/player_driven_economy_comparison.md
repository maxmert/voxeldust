# Would a purely player-driven economy be better? — decision document (2026-07-26)

> **What this is.** The user proposed dropping economic simulation entirely: no computer-controlled
> traders, no formula-driven prices, no background market evolution. The server provides plumbing
> (companies, player-built shops, stock, owner-set prices, buying, selling, ownership, taxes) and
> telemetry. All behaviour comes from real players. This document answers whether that is better.
>
> **Status.** Decision-grade. Two full designs were produced and three adversarial vetters reviewed
> them; every sustained finding is folded into the substance below, and the four findings I did not
> sustain are named in §11. **Where this document contradicts
> `scripts/economy_research_20260726.md` or `scripts/dormant_world_simulation_design.md`, this
> document is the later decision** — see the dated pointers now at the head of both of those files.
> **Where it contradicts the code, the code wins**, and every such contradiction is called out.
>
> **Nothing here is built.** `grep -rn "ItemId\|CompanyId\|Wallet\|Faucet\|EconomyPort\|CurrencyId\|worldline" crates/`
> returns zero hits. That is the good news: every decision below is free today and expensive later.

---

## 🟥 REVERSAL NOTE (2026-07-27) — this document's "currency is a material" recommendation is REVERSED

**Read `scripts/money_and_markets_design.md`. It is the later decision on money and on markets, and it
supersedes the abandoned consignment-market design.** On 2026-07-27 the user reversed the goods-as-money
choice: *"having money as items is not a very scalable idea — we will not be able to transfer money
anywhere, we will not be able to have banks or similar. We need to have currency and some ways to deal
with the money locally (stations/cities/star systems) and globally (galaxy and universe levels)."* This is
a pointer only; nothing below has been rewritten.

**REVERSED — one recommendation, and one material choice.**

1. **Money is no longer a commodity you haul.** It is a real transferable currency in two forms: coins
   (objects with weight, spendable with no communications, losable) and account balances (rows at one
   always-on service that no region teardown can destroy).
2. **The backing material must be USELESS for everything else** — dense, rare, no industrial use. This
   document recommended the universal fuel and repair input, and that is exactly wrong for a *minted*
   currency: every coin locks its backing in a vault, and at the planning population the supply requires
   about twenty months of community mining effort. Sterilising the most useful material in the game for
   twenty months is something players will simply refuse to do. **This must be settled before terrain
   generation.**

**WEAKENED — the two problems this document claimed the material answer removed outright are BACK**, and
they are solved differently rather than avoided:

- *Carrying money across a region being switched off.* Solved by **refusing to put accounts in the thing
  that gets switched off**. Confirmed as the only honest option available: world shards' durable outbox is
  inert (no manifest sets its path, and the demand-spawn allow-list at `crates/bins/src/lib.rs:2031-2054`
  carries no storage key at all), teardown is an outright kill with no flush step, and the volumes that
  exist bind to pod ordinals rather than to places.
- *Proving a region woken after five years holds the same values.* Solved by **on-touch evaluation
  everywhere** — no interest, no upkeep, no fees, no decay, no timers — plus a money-family-scoped
  dormancy gate (a whole-store byte-identity assertion would fail on tick one, because the clock record is
  written unconditionally every tick).

**UNTOUCHED and still adopted in full:** purely player-driven, with no computer traders, no formula-driven
prices and no background market evolution; the consignment market shape with a per-sale venue fee; never
charging by elapsed time; an economy object is never the container of record for goods; items are
append-only positions with never-reused identities in the game's own store; the economy is optional and
the game never waits on it.

**LOAD-BEARING and used verbatim in the new design:** §4.1's arithmetic killing the fixed-supply option
(5,000 × 1,000 = 5,000,000 supply against a 250,000/month drain = 20 months to zero; the measured 61.0%
velocity fall implying prices halving every 7.0 years); §4.2's "endow goods, never money" cold-start
finding and its wash-trading arithmetic (which becomes the reason the burn share must never be zero);
§4.5's survey result that no shipped game has ever run a fixed-supply tradeable currency, which is why
minting from a material is presented to the user as an experiment with a stated escape hatch.

---

## 1. The answer, in plain language

**Yes — a purely player-driven economy is the better choice.** Adopt it. But adopt it for a different
reason than the obvious one, at a smaller saving than it appears to offer, and only after answering two
questions it does not answer by itself.

**What it genuinely wins.** It deletes roughly half of the economy layer we designed — every price
formula, every computer trader, the order book, the tax algebra, shares and dividends, contracts and
insurance. It removes the two hardest unsolved problems in that design outright: carrying money
balances across a region being switched off, and proving that a region woken after five years holds
the same values as one that never slept. It shortens the path to a first working shop by a slice or
two. And it makes the rule that the game never waits on the economy nearly free to enforce instead of
expensive: with no prices and no markets, the economy never needs to cause anything to happen in the
world, so an elaborate two-way seam collapses to a one-way reporting channel anybody can verify by
reading one page.

**Where the headline claim is overstated.** The biggest advertised win — that this dissolves the
sleeping-world problem — is real but partial, and being precise about it matters. The machinery for
evolving a switched-off region does not get deleted. It ships anyway, because resource regrowth,
structures wearing out, and people and creature life are all main-game systems already declared. What
this proposal removes is the economy as a *user* of that machinery, and it happens to remove the single
hardest user: money that has to be caught up after a long absence. That is a genuine and valuable
saving. It is not the removal of a subsystem.

**Where the usual reasoning is simply wrong, and this is the most useful thing in the document.** The
fear that a player-only economy makes the galaxy feel like a ghost town is misdiagnosed, and measured
data reverses it. In the largest shipped player economy, fighting happens in ninety-five percent of all
star systems and mining and building are spread nearly as widely — but trade collapses into one place,
with a single region carrying sixty-nine percent of all trading. That game seeds computer-run sellers
in essentially every station, and trade concentrates anyway. So computer traders demonstrably do *not*
spread commerce out, and their absence is not what makes a world feel empty. Concentrated trade with
dispersed activity is the natural shape of a big world, not a failure. Design for it: expect a handful
of busy market places and a galaxy where people are digging, building and fighting near you. Judge
liveliness by the second thing, never by whether the shop next door has customers.

**The one thing it does not solve, and cannot.** Finding a shop. There is deliberately no way for one
server to ask another server a question, and most of the galaxy is switched off, so a shop nobody has
visited is a shop nobody can learn exists. This is *harder* under a player-only model than under a
simulated one, because a simulated market can publish a plausible price for an unvisited place straight
from a formula for free. Every game that shipped player shops ended up building a world-wide search
index, and a world-wide index is exactly what this architecture forbids. The honest answer is a
searchable catalogue that lives outside the game, built from the telemetry this proposal already wants,
allowed to be out of date, that the game itself never reads and never waits on. If we do not build it,
players will — in one shipped economy they run a program that watches their own network traffic and
uploads prices to a public website.

**The correction that matters most to the money question.** Do not invent an abstract currency and do
not hand new players a purse. Let the medium of exchange be a real material — the universal repair and
fuel input — that players dig out of the ground and that building, flying, repairing and dying consume.
A consumable has its source and its sink in the same object, which is exactly the pair that money
lacks. Then express an asking price as a small bundle of goods, because a bundle containing one kind of
thing is precisely a price in that thing, which keeps the whole money decision reversible for free.
And give the world exactly one published constant: a poor, universally available way to convert what
you dug into the material everybody wants. That single number is the one price nobody ever has to look
up, it works in a region nobody has ever visited with the website switched off, it guarantees a
newcomer an income on their first hour, and it destroys material rather than creating it — so it can
never inflate anything. It is not a trader. It is a recipe.

**Two things must be decided with this, not after it.** First, how unevenly materials are spread across
places. If a player can dig everything they need within reach, no shop will ever open no matter how
well built it is, and that decision freezes permanently when terrain lands. Second, how many places may
be open at once. The busy player-driven economy runs about forty people per place; the famously empty
one runs under three. At four hundred players — the realistic planning number for a game of this shape,
based on the nearest live example — matching the busy figure means roughly ten places open at a time,
not thousands.

**And one rule above all others: never charge a shop rent by the hour or by the day.** Both of the
closest shipped precedents did exactly that, and both shut down and empty an unpaid shop. At the rate
one of them charged, a shop in a region switched off for five years wakes owing six hundred and
fifty-seven thousand. Charge a small share of each sale instead. That is the entire difference between
a shop that changes only when a customer touches it and a shop that needs a clock — and the clock is
the thing this proposal exists to avoid.

**Bottom line: adopt it.** Keep everything the existing designs built for the physical world. Treat
the money question, the finding-a-shop question, the terrain question and the how-many-places question
as part of this decision. Build nothing yet — five things a shop is made of do not exist, and four of
them are mid-roadmap.

---

## 2. The comparison

Three candidates. **S** = the simulated model already designed (order books, computer traders, price
formulas). **P** = purely player-driven, exactly as proposed, with no non-player economic behaviour of
any kind. **P+** = player-first with one thin non-player edge (the recommended answer; §5 defines it).

| Criterion | S — simulated | P — pure player | P+ — player-first + one constant | Winner |
|---|---|---|---|---|
| **Machinery** (production lines of economy code) | ~6,000–8,500 Tier-A + an analytics tier + 5 late-added mechanisms (~2,000–2,500) | **~2,500 prod / ~4,000 test ≈ 6,500 total** | ~2,500 prod + **~120 prod** for one recipe table and its validator | **P** by a nose, **P+** effectively tied |
| **Risk** (delivery) | Engineering risk, large and measurable: the flagship gate was unrunnable as written; every closed form saturated (dormant regions froze after ~23 days in its own worked example); cross-region dormant flow is algebraically impossible so v1's only legal value is zero; the recommended storage custody needs a new stateful component | Social risk, unfalsifiable pre-launch: trade may never start; hubs eat the frontier; a shop has near-zero expected customers at low population | Same social risk, minus the two worst engineering items | **P / P+** — engineering risk is what kills projects |
| **Time to something playable** | Blocks + items + per-region saving + action channel + event channel, **plus** a money type, a ledger and a price | The same five prerequisites and nothing more | The same five, plus one content table | **P**, by the last two slices — not by the first ten |
| **Liveliness** | Buys a real but narrow win: everything it delivers lands between about a day and a month of absence, and its own document forbids ever claiming the galaxy is visibly alive everywhere. Before its mandatory seasonal driver its emergent rule is *the parts you invested in decay; the parts you ignored are frozen* | Economic state changes only when a customer touches it, which is honest rather than fake. Liveliness comes from mining, building and fighting near you — which measurably decentralises on its own | Same, plus a guaranteed counterparty in a cold place, which is what a first arrival actually needs | **P+** on the evidence; **S** only on the letter of "the refinery kept refining" |
| **Analyzability** | Buys a closed creation-and-destruction vocabulary, a supply identity assertable in CI, a replay twin, a counterfactual simulator | **Worse, not better.** No comparison case; blind to the switched-off majority of the galaxy; still contains cheating and real-money trading; with no formula holding an expectation, a duplicated item produces no anomaly at all | Same as P, plus one published constant to measure against | **S**, clearly — the parent assessment is wrong here |
| **Law one — the game never waits on the economy** | Satisfied expensively: inverted seam, four machine checks, a refusable command channel back into the game, a three-way byte-identity gate. Its own count is three genuinely coupled edges, each needing its own argument | Satisfied almost for free: the observation port becomes one method returning nothing. **But** it makes finding a shop harder, which is the one place a hidden dependency can form | Same as P; the constant sits on the **game** side as content, so nothing on the sale path reaches across the seam | **P / P+** decisively on code shape |
| **Law two — sleeping regions must not freeze the world** | Satisfied by the substrate, at the cost of its hardest consumer (money as a zero-rate dormant subject, the woken-equals-never-slept proof over monetary values, the catch-up tier) | Takes the economy **out of scope** rather than satisfying the law — legitimate only because the substrate survives for the physical world. Backed by shipped kernel code: a region holding a player is structurally un-killable | Identical | **Tie** on the law; **P / P+** on the cost of meeting it |

**Read the table this way.** P and P+ differ by about a hundred and twenty lines of content and one
paragraph of policy. S differs from both by most of an arc. The decision is P-versus-S; the choice
between P and P+ is a game-feel decision about whether a first arrival in an empty place deserves a
guaranteed counterparty, and §4 recommends that it does.

**Code corrections to the framing used in earlier documents.** The closed cross-server set is **23
arms**, not six: `crates/wire/src/intershard.rs:116-257` (counted by parsing the enum). `BlockEdit`,
`Coupling` and `Signal` are prose reservations at `:28-29`, scheduled P6/P8/P9 — they do not exist as
variants, so **there is no gameplay message bus to hang a shop bulletin on.** The module header's own
"LANDED (16 arms)" at `:9` is stale and the enum wins. Earlier judgement text saying 25 arms is also
wrong.

---

## 3. What the proposal deletes — the classified ledger

**Counts: 16 survive · 12 survive and become more important · 21 become unnecessary · 17 newly needed.**

### 3.1 ⚠ THE WARNING THAT MUST BE READ FIRST

**The lazy-evaluation substrate does NOT get deleted.** `scripts/dormant_world_simulation_design.md`
§4.1–§4.5 (`:710-1214`) — the production and depletion integral with its piecewise breakpoints, the
seasonal driver, the cohort integral, the sparse absolute-rebase event log, compaction, the fence-ordered
fold — **ships anyway**, because resource regrowth, structure condition wear and creature/people life are
main-game infrastructure the user has already declared. Row 27 of that document's own layer table
(`:382`) already made the split correctly: the physical half of a recurring asset sink is a condition
stock draining at a per-entity integer rate while dormant, and **the monetary half is the part we are
deleting** ("structures wear as normal; nobody bills you"). Anybody who reads this proposal as "the
dormant-world design is cancelled" has misread it. **The deletion is the top layer, never the
foundation.**

### 3.2 ⚠ AND THE COLLISION THAT WARNING CREATES — a new binding rule

Row 27 also says a structure "can physically decay/decommission on its own schedule" with the economy
off. A decommission destroys the structure hosting a counter, therefore its containers, therefore its
stock — **while dormant, with no customer present.** That is precisely the time-driven economic state
change this proposal claims to have eliminated, and it makes the flagship byte-identity gate
unsatisfiable the moment a shop stands on a wearing structure. Three aggravators: `RULE WL-ITEM`
(`:2417-2437`) requires a typed loss reason per never-reused item identity, which a scalar closed form
cannot produce; `:2098-2100` says wrecks and salvage exist only inside area-of-interest, so a structure
decommissioned while dormant leaves nothing findable; and `W4` at `:3004-3006` retains a spoil arm for
perishables, which is the same hazard arriving through food.

> **RULE PDE-1 (binding).** While a region is dormant, a condition stock may **drain** but may never
> **cross a terminal threshold**. Decommission, structural collapse and spoilage are authored only by a
> **live** shard on wake, as ordinary deviations, with per-item typed loss reasons written to the
> durable state record. Clamp every dormant closed form at one unit above its terminal value, so the
> woken value is a pure function of elapsed ticks and the destruction is a single live event. This is
> the same *a comparison performs no action* idiom the designs already adopt for offer expiry,
> generalised to the place where it actually destroys player property.

### 3.3 Survives unchanged (16)

| # | Item | Where | Note |
|---|---|---|---|
| 1 | The one-way dependency law and its crate-graph, transitive-closure and dev-dependency checks | dormant `:107-238` | Survives with a much smaller surface — see §3.6 for the vacuity hazard it now has |
| 2 | The requirement that sleeping regions keep evolving, applied to the **physical** world | dormant `LAW-E2` `:239-310` | In full, for regrowth, wear and life |
| 3 | The entire lazy-field substrate | dormant `:710-1214` | See §3.1 |
| 4 | Fence-ordered fold with a dominant fence term + custodian stale-reject | dormant `LAW-WL-2` §4.4 | Crash/reorder/duplicate safety for physical state |
| 5 | The closed form is master; a live tick's delta is a difference of it | dormant `LAW-WL-1` | Permanent design tax on blocks, ships, combat |
| 6 | Integer arithmetic kernel, no-seconds rule, integer generator boundary, integer twins of the float samplers | dormant §11.1 (~360/~480) | Determinism spine; the money-arithmetic seam is subsumed by it |
| 7 | World epoch with settle-on-change | dormant §11.1 | Now also guards offer stamps — see §8 |
| 8 | One tuning struct, all-zero inert default, fail-loud validator | dormant §11.1; pattern at `crates/sim/src/rlm.rs:60-76` | Verified fully inert |
| 9 | Recipe-graph acyclicity validated in the **game**, not the economy | econ `S3` `:2641`; placement dormant `:382` row 11 | Crafting must not stop when the economy stops. Now load-bearing: the recommended constant **is** a recipe |
| 10 | Analytics as the same evaluation code over a **separate read-only** store handle | dormant §4.8 | Survives with a new and heavier job |
| 11–16 | Every physical row of the 28-row layer table: rows 1,2,3,5,6,7,8,9,10,11,12,13,23,25 and the physical half of 27 | dormant `:356-456` | Row 22 (destruction is game-owned) is the load-bearing one and is untouched |

### 3.4 Survives and becomes MORE important (12)

| # | Item | Where | Why it gets heavier |
|---|---|---|---|
| 1 | **Only real player sessions count as lifecycle occupants** | dormant §3.4 + `G-EMPTY-SESSION-ONLY` `:2718`, moved pre-terrain at `:2742` (~60 prod/~120 test) | **⚠ SCOPING CORRECTION — this is the single most dangerous item in the document.** Both designs and the dormant design schedule this "before the first NPC entity". **Wrong: the first durable non-session entity in this game will be a SHOP.** Verified: `aoi_decide` folds occupants as `dots.0.values().filter(\|d\| d.authority.simulates())` chained with held transients (`crates/sim/src/stub.rs:4397-4410`); an empty set self-reports `DemandVerb::Empty` (`:4412-4421`); that is the sole input to `empty_confirmed` (`crates/sim/src/rlm.rs:439`), which gates `desired_alive` (`:459-462`) and `teardown_ready` (`:475-489`). A shop is a `NamedConstruction` — `Durable`, `GhostPolicy::Always`, `LossBudget::ZERO` (`crates/core/src/entity_kind.rs:216-223`) — i.e. an owned, simulated dot. It lands in that fold. Commerce ships at P6; creature life is later and behind an unbuilt port. `grep -rn "SessionOccupant\|is_session_occupant" crates/` → **zero hits.** The moment one shop exists, every commerce-bearing region becomes permanently un-reapable, **silently**, because `teardown_ready` simply never fires and nothing logs a refusal. **Re-label the prerequisite: before the first durable non-session entity, and that entity is the venue.** |
| 2 | Stock as append-only positions with never-reused identities and zero-sum splits | dormant `RULE WL-ITEM` `:2417-2437`; econ `S11` `:2650` | With no formula holding an expectation about any price, a duplicated stack produces **no market-wide anomaly**. The accounting identity is the *sole* fraud detector. One shipped case saw a single item duplicated over two million times with price effects that persist today |
| 3 | Item conservation as a harness oracle with deliberately failing controls | econ `S8` `:2647`; `verify_item_conservation` `:2802` (~250/~200) | Same reason: the anti-vacuity controls now matter more than the check |
| 4 | A conserved fact is durable **state**, never a sheddable log line | dormant `RULE WL-CONSERVED-FACT` | A shed records a count, and a count cannot repair a per-item identity. There is no second source of truth |
| 5 | **A durable duplicate-suppression record on each region's server** | D-22; `crates/sim/src/stub.rs:1087` | Verified RAM-only `BTreeSet` with no retention bound, on shards with **no durable store at all** (`crates/node/src/orchestrator.rs:152` is the only `StoreRes` injection in the workspace). The only channel by which value can appear from nothing. See §8 for the fsync window that makes it worse than it looks |
| 6 | Resource distribution as a closed-form integer function of seed and position | econ `S12` `:2651`; promoted to prerequisite dormant `:2733` | Becomes the economy's entire supply tap — a content fact that exists with the economy deleted — and it fixes how unevenly materials are spread, which decides whether trade happens at all |
| 7 | A stated destruction reason per destroyed block, from its first commit | econ `S13` `:2652` | Becomes the primary demand drain. Shipped calibration: **40.79%** of production value destroyed over 37 consecutive months, band 33.0–54.0%. If not declared from day one it is unmeasurable and no target is assertable |
| 8 | An expiry stamp on every offer-like record | econ `S9` `:2648` | Was teardown safety; becomes the cure for the abandoned-shop museum **and** the only dormancy-safe form of upkeep, because it is a comparison made when somebody looks, not a clock ticking in an empty room |
| 9 | Roll events up into per-material, per-region, per-window aggregates before journalling | dormant `RULE WL-AGGREGATE` `:938` | Because the drain is physical, the economic event stream **is** the block-edit stream: 150 events/second at 1,000 concurrent players ≈ 0.60 TB/year raw, against a provisioned **256 MiB** per-region volume (`deploy/k3d/50-shard.yaml:96`) — about 185× one month of one region's entire disk. Hourly roll-up: 1.66 GB/year, a 362× reduction. **"Only collect the data" is the expensive half** |
| 10 | A minted place identity as the primary durable key; the lineage path demoted to a rebuildable index | dormant §11.1 `:2733-2795` | **⚠ CORRECTION to earlier judgement text, which implied this partly exists. It exists nowhere:** `grep -rn "RealmUid\|realm_uid" crates/` → zero hits, and the current key **is** the lineage path — `RealmCoord { level, path }` with `path` documented as "the root→leaf lineage (the globally-unique key)" (`crates/core/src/realm_coord.rs:19-22, :43`), extended by `child` at `:83-89`. A shop inside a ship changes its lineage on every flight, so a path-keyed record is silently orphaned, and an orphaned record reads as *never touched*. This is a from-scratch prerequisite, not a hardening |
| 11 | **Teardown safety and its storage-topology prerequisite** | econ `D-53` `:2693` | **⚠ RECLASSIFIED from "unnecessary".** Only book rehydration disappears. Verified worse than the report knew: shard volumes are `volumeClaimTemplates` — `ReadWriteOnce`, `local-path`, 256Mi/64Mi, bound to the pod **ordinal** (`deploy/k3d/50-shard.yaml:91-101`); and the per-realm child environment carries node id, bind, probe, coord, cookie, realm kind, realm seed and peers and **no store path at all** (`crates/bins/src/proc_launch.rs:123-137`). A realm re-spawned on a different host cannot find its own state. Stakes rise: a lost order book is an inconvenience, a lost shop stock is **confiscated player property** |
| 12 | Decoupling how many places the galaxy can address from the fixed-width membership word | dormant `D-78` `:2884` | `MAX_REGIONS = 64` (`crates/sim/src/stub.rs:552`), boot-guarded (`crates/bins/src/bin/shard.rs:229`), six hierarchy levels (`crates/core/src/realm_path.rs:51-58`), and `child_placements` filters the same capped vector `aoi_decide` consumes ⇒ **58 × 58 = 3,364 addressable systems.** Under this proposal world size becomes *the* liveliness lever, so this stops being a scaling chore. Note the code contradicts itself: the comment at `:550-551` claims 64 is "generous headroom, not a ceiling" because children resolve through the directory — that is not what the demand path does today, and the demand path wins |

### 3.5 Becomes unnecessary (21)

1. **The money type** — currency identifiers, minor exponents, multi-currency exchange, the numeraire
   question, the opening supply, issuance bands (econ `S1`/`D4`/`D9`, ~300 lines plus the exchange
   layer). `D4` was flagged *nearly irreversible once the wire format freezes*, so this removes an
   irreversible choice.
2. **The double-entry ledger kernel** with pending/post/void and a timeout (`D-48` `:2688`,
   ~1,000–1,500 lines). Replaced by stock entries summing to zero per item kind. What survives, and
   matters more, is the closed vocabulary of creation and destruction reasons.
3. **Order books, matching, clearing**, escrow as an economy-owned holding, book partitioning, the hot-market
   throughput analysis (`D1` `:148`, slice `E-7` `:2728`) — and the whole batch-versus-continuous
   determinism argument.
4. **Every computer-controlled economic actor** and everything feeding one: trading strategies, agent tick
   placement, market makers, the adaptive bid, the cold-start liquidity mechanism, the equilibrium
   solver, production-chain solving, the ambient price field (`D8`, `D13`, `E-8`, econ §4 `:495-594`,
   §7.7 `:1670-1755`). ⚠ **What does NOT go:** the split between a creature's existence and its trading
   policy survives, and its price-free needs-driven default becomes the only variant.
5. **The hierarchical tax algebra** — rate composition up the ancestor chain, ceilings, floors, input
   credits, tariffs decomposed at the common ancestor, remittance, rate-change front-running (`S-tax`,
   `D10`/`D11`, `E-4`, ~400 lines plus five algebraic proptests). Reduced to one commission charged by
   the owner of the built place you are standing in. Integer largest-remainder rounding with a proptest
   is the only residue.
6. **Corporations as securities** — shares, dividends, registries, delisting, liquidation, pro-rata treasury
   distribution, N-party atomic distribution (`D12`, `D-50` `:2690`, `D-63` `:2703`, §7.11c).
7. **Contracts, collateral, courier adjudication, insurance as a funded pool** (`D-59` `:2699`, `D-66`
   `:2706`) — slices the report placed *before* order books.
8. **Player-issued currency and every yield-bearing instrument**, plus the issuance registry, reserve
   disclosure and redemption escrow (`D23`, §5.8). Supported by the record: every instrument that
   promised yield or lent out deposits collapsed (largest verified virtual theft ≈ 790 billion units),
   and one game permitted player-minted currencies for two decades with **zero** adoption. Only custody
   without lending survives — and a till and a vault already *are* custody.
9. **The rule that every account must always be homed**, the never-dormant per-account home region, and
   escheatment of balances (econ §7.10 `:1821-1863`). Dissolves entirely under the till decision.
10. **Money balances as zero-rate subjects inside the dormancy substrate** (dormant §6.7 `:2104-2138`) —
    an elegant repair to a problem the proposal removes.
11. **The two extra cross-server arms** that existed only to seed and journal the dormant *monetary*
    substrate, plus the no-phantom-money-inflow gate (dormant §7.3 `:2252-2313`). A direct saving
    against the closed reviewed list, the most expensive place in this architecture to add anything.
12. **The counterfactual economy simulator** and the verified-matcher differential oracle (`D-56`
    `:2696`, `D-57` `:2697`). The replay twin's mechanical half survives as content tuning.
13. **The money half of the live-invariant halt flag** with per-invariant blast radius (`D-61` `:2701`).
    The item-conservation half survives and gets heavier.
14. **The market-mechanism decision and the money-representation decision** (`D1`, `D4` `:148,:151`) —
    two of the report's four highest-stakes decisions stop existing.
15–21. **Eleven monetary rows of the layer table** (4, 14, 15, 16, 17, 18, 19, 20, 21, 24, 26 and the
    monetary halves of 27/28): prices, order books, wallets, taxes as an algebra, contracts, equity,
    insurance payouts, docking fees, the monetary dashboard, and *a monetary event causing a physical
    mutation* — absent by construction rather than absent by flag.

**⚠ ONE ITEM RECLASSIFIED FROM "DELETED" TO "KEPT, UNEXERCISED".** `EconCommand`, the channel from the
economy back into the game. The dormant design names it explicitly: *"`EconCommand` in particular must
land NOW: it is the one thing that cannot be added behind an emit-only port later"* (`:2790`), with five
load-bearing properties at §3.2 `:544-566`, the fifth being that without it every monetary row is
terminal. The till design does dissolve the terminality problem — the game's own shard moves the goods
and no economy actor ever needs to — so deleting it is defensible in substance. It is not free: the
moment anything must settle physically after the fact (a back-dated commission, a retro payout, a
computer operator that *initiates* rather than reacts) the channel is owed, and that means a drain point
in the game, an identifier journal, and the attribution gate at `:106-118`. **Keep the signature and its
identifier minting, leave it unexercised, with a test asserting the drain queue is always empty in v1.**
A few dozen lines against a named irreversible.

### 3.6 Newly needed (17)

1. **A venue record** with a stated-kind operator, pricing rule as data, and upkeep rule with the
   elapsed-time variant deliberately absent. Rides the existing `NAMED_CONSTRUCTION_DEF`
   (`crates/core/src/entity_kind.rs:216-223`) rather than adding a new durability/continuity/ghosting
   triple, which the registry doc itself calls a crash-matrix multiplication (`:7-8`).
2. **A fence on the venue and on any pot-shaped counter.** Every authoritative record in this system
   carries its stamp and receivers reject stale ones — `OwnerRecord { authority, fence, lease_expires,
   in_transfer }` (`crates/wire/src/seams/directory.rs:63-83`, where the fence "IS the linearizability
   primitive"). Fold it through the same order-independent max-tracking pattern `empty_confirmed` uses
   (`crates/sim/src/rlm.rs:439-451`).
3. **Permissions that travel WITH the acting player** as a signed, expiring, fenced grant — no new arm,
   no pending-delivery table, no acknowledgement gate, works in a region whose whole ancestor chain is
   cold. Same idiom as the remote-access grants already designed
   (`docs/design/sealed_shards.md:140,151`). **⚠ Use Ed25519, not HMAC.** HMAC is symmetric: any shard
   that can verify a grant can mint one. The codebase already made exactly this split for exactly this
   reason — `LoginTicket` is Ed25519 verified against the auth service's public key, while HMAC-SHA256
   is confined to a gateway-cluster-internal resume ticket under a rotating key ring
   (`crates/connection-plane/src/tickets.rs:8-14, :38-44`). Both primitives are already workspace
   dependencies (`Cargo.toml:77-79`), so **no new library**. Ship only the verifying key to shards — one
   non-secret value, so pod provisioning stays trivial; today only the gateway gets key material
   (`deploy/k3d/40-gateway.yaml:58`).
4. **A company registry** as a new durable key family in the coordinator's existing store, tag 8 (1..7
   taken at `crates/node/src/saga_runtime.rs:109-119`; the tag space is documented append-only).
   Deliberately **not** a fifth directory arm: that enum is four write-authority leases over a *node*,
   and its expiry sweep is O(all rows) at ~5 sweeps/second (`crates/sim/src/directory.rs:41-42, :174`)
   with the partition blocker already ledgered (`D-32`).
5. **Takings accumulate physically inside the shop; the owner or an employee collects them.** The largest
   single performance decision. Verified alternative cost: settling into a remote company account makes
   every sale a seven-phase acknowledged saga (`crates/sim/src/saga.rs:386-446`) at ≈6–10 ticks
   (`:86-89`) = **120–200 ms** at 50 Hz (`deploy/k3d/10-configmap.yaml:15`), through one central commit
   point, with a durable write per phase.
6. **Prices as bundles of goods.** Up to a handful of *this much of that thing*. A bundle of one kind is
   exactly a scalar price in that thing. This is the single cheapest reversibility purchase available.
7. **Listings that travel as notes carried by players, plus public boards.** Zero new arms — notes ride
   inside the entity-state payload that already crosses (today literally `state: vec![]` at
   `crates/node/src/saga_runtime.rs:828/3166/6150`).
8. **A raised and ENFORCED carried-state budget.** ⚠ Verified: `max_state_bytes` is enforced **nowhere** —
   `grep -rn max_state_bytes crates/` returns two hits and both are doc comments
   (`crates/core/src/tlv.rs:40`, `crates/wire/src/intershard.rs:866`). So "raise the player budget"
   changes a field with no consumer. The real owed work is a checked serializer that refuses an
   oversize blob at construction, with one expected-error test per refusal arm. Player 4096 B, ship and
   named construction 8192 B (`crates/core/src/entity_kind.rs:204,213,222`); at the designs' measured
   ~55 B per stock row that is ~74 rows, so a wallet rides free and a warehouse cannot. The framing cap
   that *does* exist is 1 MiB per field (`crates/core/src/tlv.rs:40-42`).
9. **A test asserting no shop, offer, company or pot record contains an elapsed-time rate field.** The
   tripwire that keeps the dormancy win.
10. **A reliable way to say "buy this" and a reliable way to answer.** ⚠ Verified absent: the client's
    entire vocabulary is five housekeeping messages plus an unreliable movement stream
    (`crates/wire/src/channels.rs:44-62`), and `EventMsg` (`:178-182`) has no transport class to arrive
    on — `MsgClass` has eight arms and none is an event or bulk arm (`crates/sim/src/io/mod.rs:51-81`,
    ledgered `D-4`). **Cost credit:** the reliable client-to-shard discrete-action arm is already
    ledgered as `D-39.1` with two named consumers (P6 block-edit forward, P11 fire registration), and
    `BulkKind::Catalog` already exists (`crates/wire/src/channels.rs:170-176`). **Shape requirement
    commerce adds:** a per-request correlation identifier and a **typed response**, because a block edit
    and a fire registration are fire-and-apply while a purchase must be refusable. Settle that with the
    block-edit slice, not after it. **Cost NOT credited:** `MsgClass` is a frozen append-only wire
    discriminant with a golden pin, and `io-prod` carries an *independent* durable key mapping for the
    same enum with retained on-disk rows (`crates/io-prod/src/outbox.rs:67-78`) — two mappings, two
    pins, one disk format.
11. **A closed refusal taxonomy and a staleness display.** Client prediction is banned and stock arrives
    on a 100–150 ms buffer, so a purchase must be refusable: out of stock, price changed, offer expired,
    insufficient goods, no permission, container full, per-tick cap. Each a counted fault with a
    distinct player-facing message. The interface must show the stamp at which the stock and price it
    displays were observed — the layer table already forbids the shortcut (row 15 `:373`: fail loud,
    "never a stale zero").
12. **A frozen purchase idempotency key.** ⚠ This is the concrete proof that the computer-operator door is
    not free. A client-shaped key `(shop, offer, request_nonce)` has no nonce for a computer-initiated
    sale, and the workspace's closed vocabulary is `IdempotencyKey::{TransferStep, FencedKey,
    FencedCas}` (`crates/wire/src/intershard.rs:271-289`), all requiring a transfer id or a fence, with
    the durable table keyed on `(transfer, step_id)` — **an on-disk format.** Fix the shape now:
    `(venue_uid, actor_principal, seq)` with `seq` from a durable monotone per-venue high-water (the
    `WaterMark` pattern already in tree at `crates/node/src/rlm_spawn.rs:172-180`), so a player supplies
    a nonce that maps into `seq` and a computer actor mints its own **with no schema change.** Give it
    its own proptest: never reused, monotone across a kill mid-transaction. Add a retention bound (the
    open `D-22` gap) on a record that would otherwise grow once per sale forever.
13. **A custody-handoff rule for item identities.** Neither design adds a second *authority* commit point —
    credit both — but both create a second *custody* for value: a stack lives either inside an entity's
    transfer blob (moved by the saga, committed by the single directory compare-and-set) or inside a
    region's checkpoint (moved by a local batch with no compare-and-set). A purchase moves a stack
    across that boundary and a re-home moves the player. **Rule: an item identity is in exactly one
    custody per fence, and the handoff is the same fence-ordered fold as the pose, never an independent
    write.** Aggravated by `D-31` (the per-kind serialize/spawn/rebind seam is unbuilt) and `D-33` (the
    atomic N+1-key compare-and-set for a ship carrying a shop plus passengers is **missing**).
14. **A refusal when the buyer has a crossing in flight.** The existing freeze is a freeze of *client
    input at the gateway* (`crates/wire/src/seams/transfer_control.rs:33`, compensator at `:55`), not of
    shard-side world mutation, and the pose flush is a separate later step. A wallet or inventory
    mutation landing after the flush is not in the flushed blob and is destroyed. The directory record
    already carries the signal — `in_transfer: Option<TransferId>`
    (`crates/wire/src/seams/directory.rs:83`) — and neither design consults it.
15. **A hoard-side drain expressed as physical wear.** See §4. Every other drain scales with trading, and
    hoarders do not trade.
16. **An out-of-world searchable catalogue, plus a gate proving the game works with it off, stale, and
    serving wrong data.**
17. **A world-size decision and a cap on how many places may host commerce**, taken jointly with this one.

### 3.7 The vacuity hazard the deletion creates

Both designs move every value-bearing mechanism into the game to satisfy sealed shards and dormancy.
What remains behind the optional port is the exporter and the dashboards. Therefore `G-ECON-ABSENT`
(dormant `:596-604`), which runs the accumulated suite three ways and asserts byte-identical results,
becomes **trivially true and would stay green with the port wired to nothing** — the exact gate theatre
that document demands red controls against (`:400-410`, `:2663`).

> **Fix.** Keep the crate-graph allowlist and the transitive-closure and dev-dependency checks — they
> are cheap and they guard the exporter — but drop byte-identity to a named subset arm and **add an
> anti-vacuity control**: assert the economy-on arm emits at least one instance of every declared fact
> kind, and the economy-off arm loses at least one **named** observable capability.
>
> **And say in one plain sentence what "the economy is switched off" means:** the dashboards and the
> searchable catalogue disappear; every purchase, price, till and item keeps working. That is a good
> answer, it should be the headline of the optionality story, and it is also why the elaborate one-way
> apparatus becomes cheap.

### 3.8 Where the new economy code lives — and a gap that must be closed in the same commit

⚠ **A new crate escapes BOTH enforcement mechanisms, because both are per-crate opt-in.** The
determinism bans live in per-crate `clippy.toml` files — verified present for `client`,
`connection-plane`, `core`, `devproto`, `harness`, `node`, `sim`, `wire`, each banning `SystemTime::now`,
`Instant::now`, `thread::sleep`, `HashMap`, `HashSet` and raw sockets. And Tier-A membership is an
explicit list: `justfile:12`. **A new crate with no `clippy.toml` may use a default-hasher map and read
the wall clock, in the code that decides money, with no coverage obligation at all.**

Decide the tier in the design: the sale path, the stock arithmetic and the constant table are
authoritative and belong in **Tier-A** (add to `justfile:12` and ship a `clippy.toml` copied from
`crates/core/clippy.toml` in the same commit as the first line of code). Only the exporter and the
catalogue may be Tier-B, on the `io-prod` ratcheted-floor precedent (`justfile:51`). Add a tripwire
test asserting every Tier-A member has a `clippy.toml` with the ban set — nothing enforces that today.

---

## 4. The money and demand question

### 4.1 The arithmetic that eliminates two of the three usual options

**A fixed supply issued at character creation cannot coexist with any recurring fee.** Stability
requires money destroyed per player per period ≤ starting purse × population growth rate. At a
1,000-unit purse and 1% monthly growth that is **10 units per player per month**; at 0% growth — the
steady state every game reaches — it is **exactly zero**. The one shipped shop fee available for
comparison is 1,800/month per shop: **180× the budget.** Concretely: 5,000 players × 1,000 each =
5,000,000 supply; 50 destroyed per player per month × 5,000 = 250,000/month; **gone in 20 months.**

**And a starting grant is not monetary policy at any size.** In the reference economy the equivalent
grant is **31 parts per million** of all money creation; the entire month's grants across the whole game
are worth about **1,872 player-hours** of ordinary income in a month that logs 17.65 million. You would
need ~31,771× more account creations to fund that economy's faucets from grants. Any grant large enough
to matter is a bounty on creating accounts.

**A fixed supply also deflates, and this is measured rather than theoretical.** Money supply grew
+11.68%/year over nine and a half years while the price index rose only +3.59%/year — two thirds of all
new money went into hoards. Worse: from January 2023 to June 2026 the **price index FELL 5.2%** while
the money supply was growing at 2.16% *per month*. Circulation speed fell **61.0%** over the window,
−9.43%/year compounding. Set issuance to zero and keep that decay: prices fall 9.4%/year at flat
output (halving every 7.0 years), 11.2%/year at 2% output growth (5.8 years), 13.7% at 5% (4.7 years),
17.7% at 10% (3.6 years). Deflation at that rate makes holding money strictly better than holding
goods, which lowers circulation further. **With no issuance, a newcomer's income is arithmetically
identical to some veteran's spending, so every unit a veteran parks is a newcomer's lost wage.**

### 4.2 Why the "one thin non-player buyer" as designed does not survive contact

The obvious third option — a floor-price buyer funded by a pot of destroyed fees — is the mechanism one
live game ships (a 2% trade fee funds a buyer that deletes player items; its developers state the buyer
is blocked from spending more than the fee has removed, and that well over half the collected fees are
never returned, so the revealed-safe return share is under half). Its wash-trade property is genuinely
sound and worth recording: paying a 3% fee to credit a pot and recovering at most the return share
yields −2.25% of turnover at 25%, −1.50% at 50%, −0.75% at 75%, −0.03% at 99%, and exactly break-even at
100% — **so washing trades through your own accounts loses money by arithmetic for every setting
strictly below the whole fee.** That is a correctness property, not a detection problem, and the
100% setting must be **unconstructible at startup**, not merely discouraged.

But the per-place version of it fails on four independent counts, and three of them are fatal:

1. **It dies exactly where it was introduced to work.** At the designed constants (25,000 purse, 50%
   return share, 3% fee) the spendable purse per place is 12,500. One automated miner extracting 200
   units/hour at a floor of 10 earns 2,000/hour and **empties a place's entire purse in 6.25 hours.** A
   twenty-miner fleet clears 40,000/hour and would drain the galaxy's whole genesis money (252.3M
   spendable across ~20,184 places) in 6,308 fleet-hours = **263 days of continuous operation.** After
   that, a quiet frontier place refills only from fees collected in that same quiet place — which is
   approximately nothing. **The guarantee is dead precisely in the places it exists for, and alive only
   where players already trade.**
2. **The advertised supply bound is wrong by a factor of two.** Applying the pot invariant to the genesis
   endowment strands half of it: 12,500 per place can never enter circulation, so the headline
   504.6M is really **252.3M**. The bound was the design's main selling point.
3. **The per-place mint is unbounded, non-deterministic, and keyed on a type that does not exist.** Two
   vetters found this independently and the code confirms it. `RealmUid` does not exist
   (`grep` → zero hits). Realm identity **is** the lineage path (`crates/core/src/realm_coord.rs:43`),
   which `RealmCoord::child` extends (`:83-89`), so **a re-parented station or ship gets a different
   globally-unique key and mints a second endowment** — exactly the P8 ship case. And if players ever
   create realms, each one mints money: `ProfileKind::Ship` already exists
   (`crates/core/src/taxonomy.rs:544`) while `RealmKindTag::ALL` has only six seed-derived kinds
   (`crates/core/src/realm_path.rs:51-58`), so player-built realms are anticipated and unbuilt — and the
   stated end goal is ships and stations built from blocks. Build N stations, mint N × 25,000.
   Separately, a **once-ever flag cannot live in the checkpoint of the very thing it endows**, because
   that is the thing which gets destroyed, epoch-refused or wiped.
4. **The lookup is the shape the dormant design explicitly deleted.** A synchronous
   `quote(class) -> Option<Price>` read on the authoritative sale path, behind the optional economy
   port, is `weights()` restored. That document is unambiguous: *"the `-> ()` IS the LAW-E1
   enforcement: there is no method whose result a game code path can wait on or branch upon"*
   (`:509-520`), and `weights()` was deleted precisely because *"a total value is still a value a game
   decision reads"* (`:528-535`). It also leaves the price **un-versioned across servers**: two shards
   on different builds during a rolling deploy quote different prices and the supply diverges with no
   detector. The codebase's precedent for that class of agreement is the universe epoch carried on the
   envelope with refuse-on-mismatch (`crates/wire/src/intershard.rs:602-607, :814`).

**Both vetters' cure is the same and it is the right one: express the endowment as a seed-derived
DEPOSIT in the terrain rather than as a minted balance.** Then nothing mints, the once-ever bookkeeping
disappears, a re-parented or player-created realm cannot re-mint because the deposit is a content fact,
and the money question stops depending on an identity type that does not exist. And the closest shipped
precedent agrees emphatically: the game nearest this proposal runs its cold start ~135 times — once per
war, every 10 to 70 days — and **has never once started from nothing.** Every base is pre-seeded with
**goods**: 500 rifles, 1,500 rounds, 500 supplies, 3,000 basic materials, 25 radios, 25 wrenches, 50
diesel, 25 grenades, 100 shovels, 300 sandbags, 300 barbed wire; depots and seaports start with trucks,
cranes and construction vehicles explicitly to begin early logistics. **Endow goods, not money.**

### 4.3 THE RECOMMENDATION

**Four parts. The first is the important one and it makes the money question mostly disappear.**

**ONE — the currency is a MATERIAL, and it is the one everybody burns.** Do not invent an abstract
money. Let the medium of exchange be the universal repair, fuel and upkeep input: a real, divisible,
stackable material that players mine and that building, flying, repairing and dying consume. The reason
is structural: a consumable has its source and its sink in the **same object**, which is exactly the
pair money lacks. It cannot run out, because people keep mining it; it cannot pile up without limit,
because people keep burning it; and hoarding it has a real opportunity cost — every unit you sit on is a
repair you did not do — which abstract money never has. It is also mildly inflationary by construction,
which is the **only** clean answer to the measured circulation decay in §4.1.

Shipped proof is strong. One game ran eleven years with no gold at all, trading in crafting consumables
whose intrinsic uses provided their own sinks. Another has **no currency whatsoever**, running a full
production economy on pure material barter at ~2,000 concurrent players. In a game whose premise is that
everything is built from blocks, refined material is the obvious medium of exchange.

**TWO — an asking price is a BUNDLE of goods, not a number.** Up to a handful of *this much of that
thing*. A bundle containing exactly one kind of thing **is** a scalar price in that thing. So barter and
money are one mechanism instead of two; the dashboards keep a unit of account (whichever material
becomes money, **discovered by counting** rather than decided in advance); and if a conventional
currency is ever wanted or unwanted, it is one more item kind and a bundle of size one. This is the
single place where starting pure is provably not a one-way door, and it costs nothing to build in.

**THREE — the world's ONE published constant is a lossy conversion RECIPE, not a buyer.** The world will
accept raw material and yield the currency material at a fixed published integer ratio, **destroying
more than it yields**. Hand-performable everywhere at a poor ratio; better with a player-built machine.
This is strictly better than the floor-price buyer on every count that killed it:

- **It creates nothing.** Net material change is strictly negative, so it cannot inflate anything, and
  it is a **sink** — which matters, because we start about two-fifths short of the reference economy's
  sink capacity (its computer-run item stores are 38.8% of all its money destruction and we are not
  building them; trading fees are 34.4% and we get those free).
- **It needs no pot, no purse, no endowment, no once-ever flag and no minted realm identity.** The
  entire family of defects in §4.2 — drained in 6.25 hours, stranded half, re-minting on re-parent,
  minting per player-built station — **does not exist.** So does the wash-trade proof: there is nothing
  to wash.
- **It cannot be farmed into money**, because there is no money to farm. Feeding it converts raw into
  refined at a loss, which is just mining at a poor yield — the intended newcomer floor.
- **It is content, not economy.** A conversion ratio is a recipe, and recipe validation is already
  required to live in the **game** (survives-item 9). So the sale path never reaches across the economy
  port, the `weights()` objection dissolves completely, and the awkward consequence of the buyer design
  — that the money tap would be a permanent game mechanic *not* covered by "the economy is optional" —
  becomes unremarkable: it is a crafting recipe, and nobody expects crafting to be part of the economy.
- **It still buys the discovery win, which is the real reason to keep any constant.** One published
  ratio, identical everywhere forever, needs no message, no index, no lookup and no live region to be
  known. You can always turn what you dug into the material everybody wants, at a rate you knew before
  you undocked, in a place nobody has ever visited, with the website off and nothing else awake. That
  turns discovery from a **requirement** into **upside**: a player's shop is where you beat the ratio,
  not where you get your only offer.
- **It is event-driven.** It happens only when a player performs it. Nothing ticks while a region sleeps.

⚠ **Keep the ratio poor** — a salvage rate far below what refining properly yields, so it binds only
where there is no player market: a cold place, an off-peak hour, a newcomer with nobody to sell to.
**Measure what share of conversions use it**; if that share is large, the ratio is too generous and the
constant is doing work the player market should do. The response is to lower a number, **never** to make
it adaptive. A conversion that drifted with stock or with time would be clock- or observation-driven and
would destroy the dormancy win — so the dormancy requirement is itself the filter that selects which
non-player mechanisms are admissible.

**FOUR — make the drain real, measure it from day one, and add a HOARD-side drain.** Under a
near-fixed supply the tap and the drain are one problem, and the drain is the half to build. The shipped
benchmark is **40.79% of production value destroyed, sustained for 37 consecutive months**, band
33.0–54.0%, with 30.8% of the value in a loss surviving as recoverable salvage. But it is dangerously
concentrated: the top 10% of destruction events do **73.0%** of the destroying and the top 1% do
**29.4%** — a handful of large organised battles. **Do not rely on combat.** Engineer the non-combat
drains: material permanently consumed into placed blocks, fuel burned in flight, ammunition, repair
inputs, and structure condition wearing down.

And add the one drain that falls on a **hoard** rather than on a **trade**, because every other drain
scales with trading and hoarders do not trade: **physical wear on the containers and structures that
hold wealth.** Things that sit still degrade and need real material to maintain. Size it so the cost of
parking wealth exceeds the deflation gain from parking it. It must be expressed as the *physical* half
of row 27 — condition, not a rent bill — and it is subject to **RULE PDE-1**: it may drain while dormant
but may never cross a terminal threshold unattended.

**One free sink worth copying:** reward content in a non-transferable, place-scoped point that must be
spent alongside goods. In the reference game the mission loop is a **net destroyer of ~14.555 trillion a
month** for exactly this reason, and because such a point never leaves the place that issued it, it needs
no cross-server machinery at all — the cheapest reward currency this architecture can have.

### 4.4 Four refusals, each with the condition for reconsidering it

| Refused | Why | Reconsider if |
|---|---|---|
| A starting **money** grant | 31 ppm of money creation in the reference economy; any grant large enough to matter prices account registration | Never as monetary policy. Onboarding liquidity is answered by the conversion recipe |
| A starting **kit containing anything the world will take** | ⚠ **The exploit the two designs created independently.** A free kit plus a published conversion is a money grant in costume: if its convertible contents are worth 500, then 10 alts mint 5,000, 1,000 alts mint 500,000. **RULE: anything the game gives away must be provably non-convertible and non-sellable to the world**, with a test asserting no starter-kit item appears in the conversion table | Never. Give tools and a shelter, not inputs |
| Selling currency for real money | Makes supply track revenue and imports financial regulation | Only ever the neutral shape: sell an **item** players resell for value that already exists — which doubles as an anti-hoarding pump, since a hoarder buying it bids their hoard back into circulation |
| Combat bounties | Need creatures that do not exist; reward one narrow activity | If the physical drains measurably fail to reach the 33–54% band across a stated window — then add them clock-free and observation-free, same signature |

### 4.5 The honest caveats

A mineable currency has its inflation rate set by mining throughput, which means **automated mining sets
it** — that trades a design problem for a permanent enforcement cost, and enforcement is an operating
expense forever, not a shipped feature. Its exchange value drifts with mining yields and building
activity, so it is a mediocre yardstick: compute price indices against a **declared basket**, never
against it. Pure barter's cost is arithmetic — k item kinds need k prices with a numeraire and
k(k−1)/2 pairwise rates without one, which is **12,497,500 rates at 5,000 kinds** — and the bundle
representation is what buys us out of that, because a de facto currency emerges and every bundle
collapses to a price in it. Company **shares are deferred**, not refused: valuing one needs a numeraire,
and once the market has picked its currency by revealed behaviour, shares become expressible without
any new decision.

And the load-bearing honesty: **no shipped game has ever run a fixed-supply tradeable currency.** Every
one verified has a computer-controlled source, and the three that avoided the question removed money
rather than bounding it. The recommendation is on the precedented side of that line — the currency is a
material with a real player-driven source — but treat it as an experiment, and note that the escape
hatch is part two.

---

## 5. The plumbing — the recommended design in implementable detail

**Name it plainly: player-only commerce, prices in goods, one published conversion.** It is the
no-money design's spine, the one useful idea from the thin-edge design re-expressed as content, and
every sustained correction folded in.

### 5.1 Principals and companies

A **principal** is an `AccountId` — already documented as "the durable principal"
(`crates/core/src/ids.rs:3,:66`). A login account, a company account and (if ever wanted) a
computer-run operator are the **same type**; a company simply has no credential and has members. So
*an actor trades at a venue* is the shape on day one at zero cost.

⚠ **One audit before the first non-session principal exists.** `AccountId` sits inside a three-way
identity triple where the directory maps session ↔ account ↔ entity (`crates/core/src/ids.rs:1-11`), and
the directory's key set is four write-authority leases over a *node*
(`crates/wire/src/seams/directory.rs:17-26,:47-51`). Grep every consumer and assert none assumes a
session or a credential — the per-account spawn-pose stand-in read at boot
(`crates/bins/src/bin/shard.rs:89-92`) is the obvious one.

**Company record:** never-reused id from a durable monotone high-water, display name, founder, bounded
member roster, revocation generation. Stored as key family **tag 8** in the coordinator's existing store.
No balance, no wallet, no treasury key — which deletes the entire offline-account-authority problem.

### 5.2 Rights, without a cross-server query

A right is a **carried signed grant**: `{company, grantee, rights_bits, expiry_universe_tick,
generation, fence, signature}`, ~80–110 B, **Ed25519-signed** by the coordinator (§3.6 item 3),
riding in the grantee's durable entity blob and verified **locally** by whichever shard hosts the
counter. No query, no new arm, works offline, works in a region with no reachable ancestor. Rights bits:
`Restock | SetPrice | EmptyTill | PlaceCounter | SetCommission | Hire`. The grant carries a real
`Fence` and is refused by the same stale-reject rule as every other authoritative record — **not** a
bespoke generation comparison. Revocation is eventually consistent, bounded by one named
grant-lifetime field: **a dismissed employee can act until their grant expires.** That is the price of
needing no arm and no pending table, and it should be stated rather than hidden.

### 5.3 The shop as a player-built structure

**Not a new entity kind and not a realm.** A **counter block bound to N container blocks** on one
player-built structure — pure block machinery with a role, riding
`NAMED_CONSTRUCTION_DEF` (`Durable`, `GhostPolicy::Always`, `LossBudget::ZERO`, 8192-byte state). It
requires `functional_blocks`, which the lattice already forces to require `voxel`
(`crates/sim/src/capability.rs:56-67, :88-93`). Authority is the containing realm's shard, which under
node-per-realm owns both the counter and its customers (`crates/sim/src/io/mod.rs:463-479`). **No branch
on shard kind anywhere.** Under the aggregation rule one shop is **one** durable subject, not thousands
of blocks.

```
Venue { venue_uid, realm_uid, local_pose, operator: Principal,
        offers: [Offer; ≤N], stock: [ContainerRef], till: ContainerRef,
        price_policy: PricePolicy,          // closed enum, ONE variant: FixedBundles
        upkeep_policy: UpkeepPolicy,        // closed enum, ONE variant: CommissionBp
        commission_bp_seen, revocation_generation_seen, last_restock_tick,
        fence }                             // ⚠ REQUIRED — §3.6 item 2

Offer { offer_id, side: Ask|Bid, goods: (ItemKind, u32),
        want: PriceBundle,                  // ≤4 (ItemKind, u32) pairs — a bundle, never a scalar
        remaining: u32, expiry_universe_tick, listed_tick }
```

`UpkeepPolicy` has **no** elapsed-time variant, and its absence is the design (§3.6 item 9). Break-even
between the two shipped models is **1,142.86 units of sales per day** — below which commission is
cheaper, which is every quiet shop in a mostly-switched-off galaxy.

### 5.4 Stock

Append-only **positions**, never mutable counts. Identities minted like `EntityId`
(`crates/core/src/ids.rs:164-171`) from a durable monotone high-water, never wall-clock-derived, never
reused. Split and merge are zero-sum entry sets. Every disappearance carries a typed reason. Items
inherit their container's durability, so a stack inside a player or ship is zero-loss while loose world
drops are lossy. The container of record is **always a game container**; any economic claim over it is a
reference to that container, never possession of it.

### 5.5 The purchase transaction

Customer and counter are in **one sealed world**, so:

1. Reliable idempotent request from the client, keyed `(venue_uid, actor_principal, seq)` (§3.6 item 12).
2. Shard validates — offer exists, not expired, enough remaining, buyer holds the bundle, per-tick and
   per-offer caps, grant valid if the actor is not the operator, **and `in_transfer` is not set on the
   buyer** (§3.6 item 14).
3. **One atomic posting**: a set of position entries summing to exactly zero per item kind, moving the
   payment bundle into the till and the goods to the buyer, with the commission split out in the same set.
4. Journal the key in the **durable** applied-steps table **before** the effect.
5. Reliable confirmation, or a **typed refusal** from the closed set (§3.6 item 11).

**Cost: 0 shard-to-shard messages, 0 sagas, 0 directory operations, 0 uses of the central commit point.**
Validation is O(offers on this counter) with a configured cap. One dedup row (~24 B) plus two postings
(~64 B each) = **~152 B** into that region's group-commit batch. The rejected alternative — takings
settling instantly in a remote company account — is one full seven-phase acknowledged saga per sale,
120–200 ms at 50 Hz, through one central writer. **The till deletes it.**

⚠ **Arithmetic:** use `checked_mul` / `checked_add` with a typed refusal, **never saturating**.
`pay = qty × price` with an attacker-chosen quantity is exactly where a 64-bit integer overflows, and
saturating arithmetic on value is a silent value change that also hides the overflow branch from the
coverage counter.

⚠ **No floating point on a money path.** The tax authority is a containment result, and containment is
f64 throughout (`crates/core/src/geometry.rs:24-51, :60-144`, and `child_placements` takes `tick_hz: f64`
at `crates/sim/src/stub.rs:670-690`). If a sale re-derived *which realm am I in* from a distance, two
servers could disagree about who taxes a trade, and the disagreement would be a float comparison.
**Read the already-committed integer state** — the hysteretic membership bit
(`crates/sim/src/stub.rs:493-520`) or simply the hosting shard's own `config.realm` — with a test that
the sale path takes no floating-point input.

### 5.6 Money movement

There is no abstract money, so value moves three ways only: **hand to hand across a counter** in one
region; **carried inside a person or ship** when they change region (free — it rides state that already
crosses); or **sitting still in a container**. No cross-region value transfer exists, so nothing on the
value path can ever touch a saga, the directory, or the one central commit point.

### 5.7 Taxes

**One local commission, in kind.** The owner of a player-**built** realm may set `commission_bp` and name
a treasury container in that realm; a share is destroyed (a sink) and the rest lands in the treasury.
The **deepest containing built realm's owner** is the only taxing authority — **no composition up the
tree.** Natural realms (galaxy, system, planet) charge nothing in v1. ⚠ **Open:** the resolution rule for
a shop inside a ship inside a station, where the parent changes on every flight. It is answerable only
once the minted realm identity exists (§3.4 item 10), which is why that identity is a prerequisite and
not a nicety. Rounding is integer largest-remainder over the bundle with a named `min_taxable_qty` below
which the cut is zero, plus a proptest that the split sums exactly and creates or destroys no unit. Rate
changes carry an `effective_tick` delay so an owner cannot front-run their own change.

### 5.8 Telemetry

```rust
trait EconomyObserver { fn observe(&self, facts: &[WorldFact]); }   // ONE method, returns ()
```

Law one is then checkable by reading one trait. Keep `EconCommand`'s signature unexercised (§3.5).
Sealed fact enum: `Sale`, `Extraction`, `Consumption`, `Destruction`, `Conversion`, `ShopPlaced/Removed`,
`OfferSet`, `TillEmptied`, `CompanyChanged`. Conserved facts are durable **state**; everything else is
**aggregate-before-journal** — per (material, region, window) roll-ups with only exceptional events
journalled individually. In-tick ingress is `&self`, bounded, drop-and-count on overflow, drained
off-tick. Analytics run the same evaluation code over a **separate read-only** store handle. ⚠ Today's
only observability is loopback-only and unauthenticated (`crates/io-prod/src/admin.rs:109-110`, contract
at `crates/wire/src/admin.rs:10-14`) with **no event export anywhere**, so the exporter is greenfield and
needs its own volume and retention policy.

### 5.9 Discovery

**Primary — knowledge is a carried good, and knowing things is a trade.** Visiting a counter yields an
immutable `ListingNote { venue_uid, realm_uid, ≤4 (kind, qty, bundle) rows, observed_tick, fence }`
(~120 B) that rides your durable blob and crosses regions inside state that already crosses. Public
board blocks are bounded containers of notes; reading is free. Effect-free, re-derivable, no authority,
never gates a sale — and **zero new arms**, which is decisive because adding to the closed list is the
most expensive act available here. Information propagates at the speed of ships and a merchant's route
knowledge becomes a real asset, which is the in-genre answer.

⚠ **Byte collision, and it is real.** ~10–15 notes fit alongside inventory in the 4,096-byte player
budget, and at ~55 B per stock row a full inventory alone is ~74 rows = ~4,070 B. **They collide.** The
resolution is a deliberate reviewed budget raise (16 KiB = 1.6% of the 1 MiB frame cap) with the note
and inventory sections separately framed so neither mis-decodes the other — **plus** building the
enforcement that does not exist (§3.6 item 8). Scarcity of carrying capacity is a **feature**: it is what
makes a broker's memory worth paying for.

**Secondary — the out-of-world catalogue, which is what the telemetry pillar is actually for.** A
read-only projection over a separate read-only store handle, materialised outside the sealed world, with
its staleness bound displayed. Advisory, allowed to be stale, never read by the game, never gating a
sale. **This is the synergy the original assessment missed: the proposal pays for its own discovery
mechanism.** And if we do not ship it, players will.

⚠ **Do NOT design the catalogue on the future signal plane.** Verified: a counter is a functional block
and the lattice derives `signal_graph = signal_graph || functional_blocks`
(`crates/sim/src/capability.rs:88`), but `signal_relay` is carried by only **galaxy and station**
profiles (`:176, :230`) and the tests pin the exclusions — a ship has none (`:294`), a system coord's
profile has none (`:443`), and "a planet lacks `signal_relay` ⇒ not satisfied" (`:514`). The plane would
be unavailable on exactly the planet and ship realms where most player shops will stand, and the
cross-shard signal arm does not exist at all. If a bulletin is ever wanted there, that is a reviewed
profile change **plus** a reviewed arm — and if such an arm is ever added, it must be pinned
fire-and-forget/effect-free in **both** exhaustive matches (`effect_class` `:313`, `durability_class`
`:458`) with an explicit assertion that no authority decision reads a digest, because the module's own
contract forbids such an arm from carrying authority-gating state (`:33-35`).

⚠ **The trap, stated because it is the real risk to law one.** If the catalogue becomes the only
practical way to find anything, the game depends on the economy layer even though no code edge shows it.
The line: walking into a shop, reading a board and selling across a counter must all work with the
catalogue and the whole telemetry pipeline **off**, with a gate cell proving it and a red control where a
design that gates a sale on the catalogue **fails**. Discovery may be a convenience whose absence is an
inconvenience; never one whose absence is a broken game.

⚠ **And the search cost is dominated by process boots, not by flying.** Checking three systems with
three candidate places each means waking up to **nine cold regions ≈ ≥27 seconds of pure waiting** on
top of flight time, and the answer may still be no. §7 addresses this.

### 5.10 Offline owners, and regions switching off

**Offline owner: a non-event.** Takings pile up physically in the till; nobody needs to be online for a
shop to sell or for a till to fill. There is no balance to home, no escheatment, no adjudication against
an offline player's escrow, no never-dormant per-account home region.

**At switch-off:** the region writes one economy section into its checkpoint — every stock position with
its identity, every owner-set price, the till, listing stamps, the recorded operator — then flushes,
then its lease is revoked, then the process dies. Nothing is accruing, so nothing needs settling.

⚠ **Precondition, currently unmet.** `exec_kill` calls `kill_realm` then revokes the head with **no
flush of any kind** (`crates/node/src/rlm_runtime.rs:258-275`), and teardown is SIGTERM-to-the-group
then SIGKILL (`crates/bins/src/proc_launch.rs:220-254`). The correct ordering is specified and
**unlanded**: *release honored → checkpoint written → fsync durable → lease revoke honored → kill*
(`scripts/realm_lifecycle_design.md:279-281, :319-326`), with an epoch-tagged header failing loud on
mismatch. A SIGTERM drain path exists (`crates/bins/tests/sigterm_drain.rs`) with nothing hooked into
it — that window is the natural home for the flush.

**While off: literally nothing.** No timer, no interest, no rent, no decay of the economy section, no
queued deliveries, no pending rights pushes. A shop off five years and one off five minutes hold
identical bytes — **subject to RULE PDE-1** (§3.2), without which a wearing host structure breaks this.

**The one time-shaped thing is a comparison, not an action.** Expiry is `(now − listed_tick) > ttl`,
evaluated when somebody looks. An expired offer is not purchasable and displays as stale; **the stock
never moves**; the owner reclaims by visiting. If expiry moved stock or charged anything, a region asleep
five years would diverge from one awake and untouched five years, and the byte-identity gate would fail.
⚠ **Couple it to the epoch explicitly:** every offer stamp is meaningful only against the world epoch
stored in the same checkpoint, because the universe clock is a durable write-ahead ceiling
(`crates/node/src/saga_runtime.rs:112`, injected `crates/node/src/orchestrator.rs:147`) and a genesis
reset would flip every expiry at once. A checkpoint whose epoch does not match must **refuse to load**.

**⚠ And the trap that would silently destroy all of it: a shop must never count as a lifecycle
occupant.** See §3.4 item 1. This is a hard prerequisite, not a follow-up.

### 5.11 Config, with no magic numbers

One struct, all-zero **inert** default so an unconfigured build is byte-identical, a `validate()` that
fails loud on every ordering the design depends on, a `cloud(tick_hz)` derivation for **every field
expressed in ticks**, and a **compile-time assertion that the derived set validates at 10, 20 and 50 Hz**
— exactly the shape already in tree (`crates/sim/src/rlm.rs:60-76` inert; `:100-130` derived from hz;
`crates/sim/src/directory.rs:174-183` for the compile-time proof, where "a mis-derivation fails the
BUILD"). Fields: max offers per counter, max containers bound, bundle arity, per-tick sales cap per
shard, offer max lifetime, grant max lifetime, `min_taxable_qty`, max commission basis points, extraction
aggregation window, telemetry ring bytes per tick. **A tick-denominated field with a bare default
silently retunes when the tick rate changes** — that is the specific failure this shape prevents.

Game-design numbers (conversion ratios, deposit yields) are **seed- or content-derived content facts**,
not tuning. ⚠ **They must be versioned across servers:** hash every value-bearing content table into a
digest, carry it in the per-region checkpoint header beside the world epoch, and refuse a mismatched
load loud — mirroring the refuse-on-mismatch idiom at `crates/wire/src/intershard.rs:602-607`. **Two
nodes with different digests must refuse to transact, not silently disagree.**

---

## 6. Does it corner us?

**No — provided nine things are written down now and one decision is made deliberately rather than by
default. But "at zero cost" needs correcting in two directions, and the second correction is the most
useful finding here.**

### 6.1 The abstraction is not speculative — it already shipped three times

In both of the games closest to this proposal, a player's shop literally **is** a computer-controlled
shopkeeper with a player owner: you buy an employment contract from a computer-run character and place
that character in your building. In a third game the identical shop mechanism is operated by the
computer, buying player-made goods and putting them back into circulation. *An actor trades at a venue,
and a player is one kind of actor* is a description of three shipped systems, not a guess.

### 6.2 What the venue record must carry — the complete list

1. **A venue identity minted once and never reused**, plus its place and its position within that place.
2. **An operator that is a REFERENCE with a stated kind** — player, company, or a reserved non-player
   kind — never a bare player field. Making a principal an `AccountId` (§5.1) means the tagged shape
   exists on day one at literally zero cost.
3. **Stock as added entries with conserved lineage**, so *any* operator's holdings audit through the
   identical invariant and the identical oracle. If stock were a mutable count, a computer operator's
   books would need their own audit.
4. **The pricing rule as DATA** in a closed set with exactly one member. This is the hinge.
5. **The upkeep rule as DATA** with exactly one member and the elapsed-time member deliberately absent,
   plus a test asserting its absence.
6. **Every state change fence-stamped** and journaled at an idempotency key before the effect, so any
   operator is crash-safe by the same mechanism and none is a special case in the crash matrix.
7. **A listing publishable outward regardless of operator**, so discovery does not care who runs the counter.
8. **Operator and pricing rule resolvable ENTIRELY from the region's own checkpoint**, with no remote
   lookup — miss this and a computer operator could never run in a region that has just switched on.
9. **Every collected fact names the operator by that same stated-kind reference**, so year-three data
   still joins to year-one data across the introduction of a new operator kind.

### 6.3 Correction one — "zero cost" is wrong by one slice

Planting the nine items is very nearly free. Walking through the door later is not: a new member of
either closed set owes complete region-and-branch coverage in a fully-covered crate, counted separately
per instantiated type **and** per test binary, plus a pass of the identical fixture on two different
kinds of server. **Budget it as a small slice, not a field somebody adds on an afternoon.**

### 6.4 Correction two — there are TWO doors, with opposite costs

**The cheap door, which stays wide open:** a non-player operator that acts **only when a player
interacts with it**. It changes nothing about the sale path, nothing about saving state, nothing about
switching places off.

**The expensive door, which this proposal deliberately closes:** a non-player operator that acts on a
**clock** — one that quotes both sides and drifts its prices between visits, or anything charging rent
over elapsed time. That needs the whole lazy substrate wired to value, a woken-equals-never-slept proof
over economic state, and it destroys the property this proposal exists for.

**So the dormancy requirement is itself the filter that decides which non-player mechanisms are
admissible.** The door is open to anything that reacts to a customer's touch and shut to anything that
reacts to the passage of time. Encode it as a **type**: the world's price is asked exactly one question
— *which kind of material is this* — and is never handed the clock or the state of anything. An adaptive
bid, a drifting quote, a shortage-responsive price and a demand-following price all need at least one of
those two forbidden inputs, so none of them can be **written down** without first widening that
signature in a reviewed file. That is a compile-time property rather than discipline, and it is the
strongest version of the defence available.

### 6.5 ⚠ Correction three — the cheap door's price differs by design, and earlier text conflated them

The "one variant plus coverage" figure holds for a **currency-bearing** design that already has a scalar
and a pot. It does **not** hold for the recommended goods-only design: a fee-funded floor buyer would
need a scalar price and a scalar pot, i.e. **introducing a unit of account into live player holdings** —
which is the one genuinely irreversible choice. The mitigation that *does* survive is the bundle
representation (a bundle of one kind is exactly a scalar price in that kind), which makes the
**representation** reversible. It does **not** make the fee-and-pot arithmetic, the closed-supply
identity, or the wealth metrics free.

### 6.6 What is genuinely irreversible

Seven things, and only the first is caused by this proposal.

1. **The money representation.** Mitigated to near-zero by prices-as-bundles (§4.3 part two).
2. **How stock is represented.** Editable counts → added entries is a migration of live player property,
   which both existing designs insist must never happen.
3. **The durable key a region's saved state hangs on.** Must be a minted identity surviving re-parenting
   — verified to not exist at all (§3.4 item 10).
4. **The on-the-wire shape of a value and of a stack.** Freezes with first use.
5. **The material geometry of the terrain.** Fixes how unevenly materials are spread, which decides
   whether there is any reason to trade. Freezes at the terrain phase.
6. **The accounting vocabulary** — every reason a quantity may be created or destroyed. Retrofitting
   means re-deriving history rather than reading it.
7. **Whether the world ever resets.** Never resetting makes abandonment, inheritance and inequality
   policy load-bearing forever.

**NOT irreversible, and not worth agonising over now:** the operator shape, the pricing-rule shape, the
upkeep shape, till versus remote settlement, a floor mechanism, the discovery mechanism, combat rewards.

### 6.7 The residual risk no design can engineer away

If the galaxy feels empty and the chosen answer is computer-run shops, the operator slot makes that a
configuration — and at that moment somebody will observe that a fixed constant in a place with no
players is either too high or too low, and will want it to respond. **That request is legitimate and
cannot be pre-refused on merit.** What the design buys is **visibility**: responding requires widening
one signature in one reviewed file, the same class of event as adding to the cross-server list.
**Nobody should read "we will never simulate" as promised.**

---

## 7. Liveliness and cold start, honestly

### 7.1 ⚠ The emptiness fear is misdiagnosed, and measured data reverses it

Recomputed from the cached June 2026 monthly economic report of the largest player-driven economy
(771,903 kill rows, 8,490 systems, 69 regions), the four activity types have **completely different
spatial shapes** and nobody had separated them before:

| Activity | Concentration | Gini |
|---|---|---|
| **Destruction** | **8,089 of 8,490 systems (95.3%)** saw ≥1 kill; 88.5% saw ≥5; 72.2% saw ≥10; the top 500 systems (6.4% of the map) account for only 52.0% | **0.761** |
| **Mining** | top region 9.7% of value; top five 32.1% | — |
| **Production** | top region 11.0%; top five 38.3% | — |
| **Trade** | **top region 68.8%; top three 80.2%; top five 85.1%** | **0.916** |

**Extraction, building and fighting decentralise on their own. Trade collapses to a point.** And that
game seeds computer-run market orders in essentially every station — **trade concentrates anyway.** So
the claim that a large computer-driven economy "fills the gaps" is measurably **false for the only gap
that matters**, and the felt emptiness of that world is not caused by trade concentration, because 95%
of its systems have combat in them.

**Design consequences.** Size the shop machinery for roughly **three to thirty busy market places**, not
one per place. Do not expect computer traders to decentralise trade — the largest shipped case does
exactly that and it does not work. **Judge liveliness by whether people are mining, building and
fighting near you.** ⚠ **Partial dissent recorded (§11):** this does *not* license skipping "per-place
commerce features", because a frontier counter and a hub counter are the **same code** — branching on
place would violate the one-tooling and features-once rules. The saving is in content and operations
(how many places host commerce), not in code.

### 7.2 The population arithmetic

Verified addressable ceiling: **58 × 58 = 3,364 systems ≈ 20,184 places** (§3.4 item 12). The
galaxy-size knob exists and has **zero readers** (`crates/core/src/worldgen.rs:713-716` declares
`CANONICAL_SYSTEM_COUNT_LO/HI = 1/8` and `WALK_SYSTEM_COUNT = 2`; written at `:929-930, :987-988`, read
nowhere), so today's shipped world is a hand-authored forest and the range is an inert seam. **Nothing
is foreclosed.**

Density benchmarks — the busy fully-player-driven economy runs **40.79 players per place** (2,161.84
average concurrent over 53 regions, no computer traders, no currency at all); the player-crafted one
**17.88 per zone** (7,687.61 over 430); the famously empty one **2.72 per system** (21,845 over ~8,033,
peak 4.07) **with** a large computer economy. That is a **14×** density gap in favour of the game with a
tenth of the population.

**Places we may have open at once, to match each benchmark:**

| Concurrent players | To match 40.79/place (busy) | To match 2.91/place (empty) |
|---|---|---|
| 100 | **2.5** | 34 |
| 400 | **9.8** | 137 |
| 1,000 | **24.5** | 344 |
| 10,000 | **245** | 3,436 |

⚠ **And the realistic planning number is 400–2,000, not 10,000.** The nearest live example of a
player-only economy — a full-loot single-continent game with minimal computer vendors — ran monthly
averages of 427.72 (Feb 2026), 539.60, 552.23, 476.03 and 378.21 (Jun 2026), with a recent 30-day surge
to 1,382.58 and an all-time peak of 9,618. Its five-month trend before the surge was **downward**. Check
every liveliness claim at the low end. (A land-area figure was unobtainable, so no per-square-kilometre
comparison is asserted.)

At the addressable ceiling with 100 players, **98.8%** of non-hub systems hold nobody at any instant and
an ordinary shop sees one customer every **26 days** with only a **68%** chance of any customer in a
month; at 1,000 players 88.8% empty and one every 2.6 days; at 10,000, 30.3% empty and 3.8 a day. **The
total demand budget is fine at every population** — ~960, ~9,600 and ~96,000 shop interactions a day.
**So this is purely a spreading problem, and the lever is how many places are open, not how many people
are logged in.** The busy precedent literally switches map regions off to fit expected population.

### 7.3 The first week, honestly

**Hours 1–8.** Everybody mines by hand because no shop exists. The game must be worth playing as a
building game with **no economy at all**. The conversion recipe gives a first-hour income floor: what you
dug becomes the material everybody wants, at a poor published ratio, anywhere, with nothing else awake.

**Days 1–3.** Seeded goods at the canonical starting places (per §4.2 — goods, not money) let the first
production chains start without waiting for somebody to build a refinery. The first counters appear at
the busiest built place.

**Days 3–7.** ⚠ **This is where it either starts or does not, and the deciding input is terrain, not
plumbing.** A shop opens only if somebody has a surplus of a thing somebody else **cannot dig**. If
material geometry is even, no shop ever opens, and every line of commerce code is dead. **Script this
week as a scenario before building any commerce machinery.**

**Weeks 2–8.** Two or three built places become hubs, because traffic and commission compound and the
measured penalty for being off the hub is **6× to 84×** revenue per hour for identical stock. The
catalogue becomes how most people find things. Couriers and knowledge-brokers become real professions
because listings travel only with people.

### 7.4 ⚠ The wake-latency hazard that is specific to commerce

A shop is a destination you **walk** to. Everywhere else, the predictive spin-up term hides the boot
delay because it is `velocity × boot_horizon` — but the spin-up radius itself scales with realm
**extent** (`VISUAL_AOI_SPIN_UP_FACTOR = 2.5`, `crates/core/src/worldgen.rs:829`, through the
velocity-safe constructor at `crates/core/src/geometry.rs:783-800`) while boot time scales with realm
**content**, and the launch floor is `max(hz×3, boot_p99 + settle)` (`crates/sim/src/rlm.rs:105-108`). A
3-metre kiosk area gets 7.5 metres of lead ≈ 5.4 seconds at walking pace: adequate against a 3-second
floor, **marginal** once a block-bearing region boots by loading a block store. **So the term that hides
latency everywhere else is at its smallest exactly where shopping happens.**

> **Fix.** Floor the spin-up radius on boot time as well as extent —
> `spin_up = max(extent × factor, walk_speed × boot_p99 × safety)` — reusing the `need` term the
> velocity-safe constructor already computes rather than adding a mechanism, and widen the tear-down gap
> to keep the existing no-flap invariant true by construction. Add a gate cell that **walks** a player
> into a small cold venue region and asserts the region is live before the counter is in interaction
> range. And pre-warm the places a player is heading toward, or shopping becomes the worst-feeling
> activity in the game.

### 7.5 ⚠ People and creature life — put them in the wilderness

The usual counter-example does not transfer, and the reason matters: the reference game's
computer-controlled characters are **ships in space** and mission agents behind a menu, not shopkeepers
standing in towns. Nobody expects a hostile frigate to sell them anything. In a game where players walk
around inside built settlements, **a visible person in a town who can be talked to and fought but cannot
trade reads as unfinished**, because every game that put people in towns put vendors in towns.

> **Recommendation.** Put creature and people life in the **wilderness, the space between places, and
> the ruins**; keep towns and stations player-built and player-staffed. Then the absence of computer-run
> shops is legible: there are no shopkeepers because there are no computer-run towns, and the only
> settlements are ones players made. That is a coherent world rather than a missing feature, and it costs
> nothing to decide now.

And the honest limit either way: **living characters exist only where a player already is**, so they make
the place you are standing feel inhabited and do nothing for the majority of the world nobody is in.
They are **not** the answer to the emptiness risk. Whatever is decided, the only-real-sessions fix must
land first (§3.4 item 1).

---

## 8. Correctness — the invariants and the tests

### 8.1 ⚠ The crash story is three gaps deep, and none of them is closed today

A purchase can be applied twice or silently lost, and three independent verified gaps guarantee it:

1. **The dedup journal is RAM-only** (`crates/sim/src/stub.rs:1087`, D-22), on shards with **no durable
   store at all** — `crates/node/src/orchestrator.rs:152` is the only `StoreRes` injection in the
   workspace, and the spawned realm's environment carries **no store path**
   (`crates/bins/src/proc_launch.rs:123-137`). Every switch-off erases all dedup memory.
2. **`commit` is block-on-PRIOR.** Read the contract verbatim: *"`commit` is block-on-PRIOR (it stages
   the current batch to an off-tick writer and returns before THAT batch's own fsync)"*
   (`crates/sim/src/io/mod.rs:425-433`). So a purchase "committed" in tick N is **not on disk when the
   tick ends**. RLM Step 5e calls commit-then-flush precisely because of this. **Neither design's gate
   list mentions the fsync window.**
3. **No client action channel with its own seq and ack** (`crates/wire/src/channels.rs:44-62`), so no
   retry can be made idempotent.

**Net effect: acknowledge a sale, lose the fsync window, and the buyer has goods the ledger never
recorded — or the client retries and is charged twice.**

> **Required, as prerequisites and not follow-ups.** (a) A **client-minted** idempotency key stable
> across retries, journaled consult-before-effect / record-after-effect exactly as
> `crates/wire/src/intershard.rs:273-289` already specifies, in the frozen shape of §3.6 item 12.
> (b) The journal in the **region's own durable store** — which makes the per-region persistence slice a
> hard blocker: **no venue before it.** (c) A stated durability rule: **either** gate the client
> confirmation on `flush()` (a real per-sale fsync cost that must be **measured**, not assumed) **or**
> declare the confirmation advisory with a bounded loss window and prove re-drive with the same nonce.

### 8.2 The invariants

| # | Invariant | Note |
|---|---|---|
| I1 | **Value is never created or destroyed except by a named reason.** Every posting set sums to exactly zero per item kind; total = genesis + Σ named creations − Σ named destructions, **with no reconciliation plug** | The reference game's own published flow table over-predicts its money growth by **82%** after nineteen years (1,899T claimed vs 1,043T actual, a 856T gap papered with a plug). If ours does that, every supply claim becomes unverifiable and the invariant gets switched off rather than fixed |
| I2 | **A purchase is applied at most once**, across a crash, a restart, a retry, and a region switching off | The only channel by which value can appear from nothing |
| I3 | **Every item identity traces to exactly one mint**, is never reused, and splits and merges sum to zero | With no formula holding an expectation, this is the sole fraud detector |
| I4 | **An item identity is in exactly one custody per fence** (§3.6 item 13) | The concrete duplication path across two commit points |
| I5 | **A dormant region writes zero bytes and its economy section is byte-identical on wake** | Subject to RULE PDE-1 |
| I6 | **No shop, offer, company or pot record contains an elapsed-time rate field** | Tripwire |
| I7 | **No money path takes a floating-point input** | §5.5 |
| I8 | **Two nodes with mismatched content digests refuse to transact** | §5.11 |
| I9 | **Anything the game gives away is non-convertible and non-sellable to the world** | §4.4 |
| I10 | **A venue is never a lifecycle occupant** | §3.4 item 1 |

### 8.3 The gates

1. **`G-EMPTY-SESSION-ONLY`** — a region holding a stocked venue, a full till and a live offer, with **no
   session present**, self-reports empty and is torn down inside the normal window. RED controls: make
   one venue a session occupant and assert the region stops being reaped; and one real session present
   must never be torn down. **Ship in the same diff as the venue.**
2. **Zero-sum posting proptest** — over arbitrary bundles, quantities and commission rates, the entry set
   sums to exactly zero per kind and no identity is reused. Plus **explicit expected-error tests per
   refusal arm**, because a proptest reaching a branch is not a guarantee across runs.
3. **In-kind commission proptest** — largest-remainder split sums exactly, creates or destroys no unit, is
   zero below the minimum taxable quantity, monotone in the rate.
4. **Dormancy round-trip** — stock, sell, switch off, advance the universe clock by five equivalent
   years, switch on: byte-identical, **zero** bytes written while dormant. Must include an offer that
   crosses its expiry threshold during the horizon.
5. **RULE PDE-1 cells** — a stocked venue on a structure whose condition would have expired mid-dormancy:
   assert **exactly one** destruction event, per-item attribution, and I1 closing. Same for a perishable
   stack.
6. **⚠ Kill INSIDE the fsync window** — after `commit`, before `flush`. The test store is explicitly built
   with a staged/committed two-tier model to crash across exactly that window
   (`crates/sim/src/io/mod.rs:395-405`), so the cell is cheap and is **absent from both designs' gate
   lists**.
7. **Kill mid-purchase, then replay** — exactly one terminal outcome, no double charge, no free goods,
   dedup record survives the restart.
8. **Purchase in every saga window** — fire a purchase in each phase of the transfer chain
   (`crates/sim/src/saga.rs:386-446`) with a kill in the window; exactly one terminal outcome, totals
   conserved, and a buyer with `in_transfer` set is refused.
9. **Two clients race the last unit** — exactly one succeeds, the other gets a typed refusal, and neither
   client ever displayed a predicted success.
10. **Epoch refusal** — reload a venue checkpoint under a bumped epoch: **loud refusal**, never a silently
    re-dated offer table. Same for a mismatched content digest.
11. **`G-IDENTICAL-SHOP` on two kinds that actually differ** — ⚠ pin **planet (Spherical) versus ship or
    station (Cartesian)** (`crates/sim/src/capability.rs:281-291`) and say why. **Do not accept station
    versus area:** `ProfileKind::Area => profiles::ship()` (`:259-262`), so that pair is the *same*
    capability set and proves nothing. Add a cell where the hosting realm **re-anchors** while a sale is
    in flight — the specific way an identical fixture will fail.
12. **Economy-absent, three arms, WITH the anti-vacuity control** of §3.7. Plus the inverse: with the
    catalogue off, stale, and serving wrong data, every purchase path succeeds — and a red control where
    a design that gates a sale on the catalogue **fails**.
13. **Item conservation oracle** over captured inspection reports, with a mid-scenario shed and a
    deliberately failing control.
14. **Rights** — a signed grant is accepted by a cold region with no reachable ancestor and no query; a
    grant below the known revocation generation is refused; an expired grant is refused; a grant signed
    by anything but the coordinator's key is refused.
15. **Volume gate under the existing density fixture** — hundreds of players trading in one place: per-tick
    sales cap holds, aggregate rows per second stay inside the configured window budget, no snapshot
    latency regression against the existing sub-millisecond budget.
16. **Walk-to-cold-venue latency cell** (§7.4) and a load test landing **with** the feature: N counters, M
    concurrent buyers, plus cold-arrival wait time.
17. **Headless agent drive** — walk to a counter, buy, read a board, collect a till, convert at the
    published ratio: injected input, screenshot evidence, run manifest, no human in the loop.
18. **Operating gate, not pre-merge** — the share of conversions using the published ratio is measured and
    reported. If large, the constant is doing work the player market should do; the response is to lower
    a number, never to make it adaptive.

### 8.4 Coverage feasibility

Reachable, and both designs got the hardest part right: an object-safe `dyn` port avoids the
per-monomorphization multiplication, exactly as the codebase states for its two existing seams
(`crates/sim/src/io/mod.rs:413-415, :461-464`). Three traps: (a) `Option<Box<dyn …>>` puts a
Some/None branch at every call site and regions are counted **per test binary**, so `vd-sim`,
`vd-node`, `vd-harness` and the workspace scenario suite must each exercise both states; (b) proptests
do not replace expected-error tests; (c) **saturating arithmetic on value hides the overflow branch from
the counter entirely** — use checked arithmetic with a typed refusal (§5.5). Generic functions stay
branchless shims with all branching in monomorphic helpers, following `crates/core/src/tlv.rs`.

### 8.5 ⚠ Honest cost, re-baselined

Measured on this repository at the `#[cfg(test)]` boundary: `vd-core` 0.95, `vd-wire` 1.07,
`vd-harness` 1.14, `vd-sim` 1.48, `vd-node` 1.52 — **24,559 production against 31,819 test = 1.30**,
rising to 1.59 attributing the 7,124-line workspace scenario suite. So the widely quoted "2.5 to 3 test
lines per production line" **overstates test cost by roughly 2×**, and one design's own table was
internally inconsistent (columns summing to 11,700 against a stated 13,700 and prose saying 14,300).

**Corrected figures.** Normalised for what each design counts, the two **agree** on commerce-specific
production code at roughly **2,500 lines**, i.e. ~4,000 test and **~6,500 total** — a two-to-three
slice arc, not a phase. The one published constant is **~120 production plus ~200 test**, about **4%** of
the economy work — so the conclusion that the constant is cheap gets **stronger**, not weaker. Against
the simulated model's 6,000–8,500 Tier-A production lines plus an analytics tier, the real saving is
**~3,500–6,000 production lines of economy code**. State plainly that this is a small fraction of a
working shop, which is dominated by blocks, items and per-region persistence that **neither design
changes**.

**Storage.** Per shop: an offer table plus positions and till metadata ≈ 2–4 KB; 10,000 shops galaxy-wide
= 20–40 MB spread across their own regions, so a region holding 1,000 shops uses ~4 MB = 1.5% of its
256 MiB volume. **Shops are not a storage problem.** The telemetry stream is: at ~1.66 GB/year aggregated
hourly (versus 0.60 TB/year raw) it must **not** land on a region's volume, so the exporter needs its own
volume and retention policy as a real deployment line item.

---

## 9. What lands when

### 9.1 Free right now, before terrain — and unmigratable later

None of these is a feature; all are decisions, and every one costs approximately nothing today.

1. The nine venue-record items (§6.2), written down.
2. Stock as added entries, never edited numbers (§5.4).
3. A **minted realm identity** as the primary durable key, surviving re-parenting — **verified to not
   exist at all**, so from-scratch (§3.4 item 10).
4. Deposit yield as a whole-number formula of seed and position.
5. A declared destruction reason at the destruction site.
6. A world epoch on every durable record, and a content digest beside it (§5.11).
7. The closed vocabulary of every reason a quantity may be created or destroyed.
8. The no-rent rule and its tripwire test.
9. **Only real sessions count as lifecycle occupants** (§3.4 item 1) — re-scoped to *before the first
   durable non-session entity*.
10. Decouple the addressable-place count from the fixed-width membership word (§3.4 item 12).
11. **Prices as bundles of goods** (§4.3 part two).
12. The frozen purchase idempotency key shape (§3.6 item 12).
13. **RULE PDE-1** (§3.2).
14. The integer arithmetic kernel and the integer generator boundary.
15. The free-items rule (§4.4, I9).

### 9.2 Decide jointly with terrain, because terrain freezes it

- **How unevenly materials are spread across places.** THE input; if even, the whole commerce layer is
  dead code.
- **How big the world is, and how many places may be open at once** (§7.2).
- **Whether the world ever resets** (§10 decision 4).

### 9.3 Build order

1. **Per-region durable saving**, with a version-tagged checkpoint refusing a mismatched epoch and a
   write-then-flush-then-release shutdown — designed and unbuilt, and the **one hard blocker**
   (§5.10). Add the missing plumbing: a store path in the spawned region's environment, plus the
   three-way storage-topology decision carried forward from `D-53` (networked storage with realm-keyed
   volume identity, **or** an explicit fence-stamped idempotent handoff in spin-down/spin-up, **or**
   economic state only on node-pinned realms).
2. **Blocks.**
3. **Items and containers**, in their final shape.
4. **The reliable action channel and the reliable event class**, shared with block editing, with the
   response shape settled *in* that slice (§3.6 item 10).
5. **The durable duplicate-suppression journal with a retention bound.**
6. **Counters, tills, offers, companies, carried notes, boards.**
7. **The aggregating exporter.**
8. **The published conversion ratio** — content, one table, one validator.
9. **The out-of-world catalogue, last**, with the economy-off gate.

### 9.4 What must NOT be built yet

- Any shop, before per-region saving lands. Stock would evaporate seconds after the street empties: at
  50 Hz the drain window is **25 ticks (0.5 s)**, the cooldown **50 (1 s)** and the demand window **200
  (4 s)** (`crates/sim/src/rlm.rs:101-116`).
- Any venue, before the only-real-sessions fix.
- Any transferable or ship-hosted venue, before `D-31` (the per-kind serialize/spawn/rebind seam,
  unbuilt) and `D-33` (the atomic N+1-key compare-and-set, **missing**). A shop inside a player ship also
  needs a **new frozen realm kind** — `ProfileKind::Ship` exists with no `RealmKindTag`.
- Company shares, until the market has revealed its currency.
- Any bulletin over the future signal plane (§5.9).
- Any rights model built independently of the P9 grants table — the same mechanism, so building rights
  first risks building them twice (`docs/design/sealed_shards.md:140,151`).
- ⚠ **And note what nobody has ever run:** the reconciler that kills regions ships **fully inert** by
  default — every field zero, verified at `crates/sim/src/rlm.rs:60-76` — so **no one has yet operated
  this system with regions genuinely coming and going while holding state**, which is the exact regime a
  shop depends on. The crash proofs to date cover launch bookkeeping and transfers, not world state. And
  a deep region's view of its ancestors is fixed at the moment it starts (`D-RLM-6`), with the cure
  decided and unbuilt.

---

## 10. Decisions for the user

Ten decisions. Each has options, real consequences, and a recommendation. Numbers 1, 2 and 3 are the
ones that cannot wait.

**1. What is money?**
- *(a)* No money at all; every price is a bundle of goods.
- *(b)* **The medium of exchange is a real material — the universal repair and fuel input — that players
  dig up and that building, flying, repairing and dying consume. Prices are still bundles, so a bundle
  of one thing is a price in that thing.**
- *(c)* An abstract currency with some source that puts new units into the world.
- **Consequences.** Option (a) is the purest and the cheapest, and it costs you the common yardstick that
  wealth figures, inequality figures, price indices, taxes and company shares all need. Option (c)
  reintroduces the whole question of where new units come from, and the arithmetic in this document
  shows it cannot coexist with charging shops anything. Option (b) has a real source and a real sink in
  the same object, so it can neither run out nor pile up without limit, and hoarding it costs you the
  repairs you did not do — but its supply rate is set by how fast people can mine, which means
  automated mining sets it, which is a permanent enforcement cost rather than a shipped feature.
- **RECOMMEND (b).** It is the only option with a shipped precedent for both ends, and choosing bundles
  as the representation means (a) and (c) both remain reachable later at almost no cost.

**2. How big is the world, and how many places may be open at once?**
- *(a)* Set it once, large, and grow into it.
- *(b)* **Cap how many places may be open at once as a function of how many people are logged in, and
  raise the cap as the population grows.**
- **Consequences.** The busy fully-player-driven economy runs about forty people per place; the famously
  empty one runs under three and has computer traders filling gaps. At four hundred players — the
  realistic planning figure — matching the busy number means roughly ten places open at a time. Option
  (a) at any generous size gives a shop one customer every few weeks and makes the world feel like a
  mausoleum, with or without simulation. The closest shipped precedent literally switches parts of its
  map off to fit expected numbers.
- **RECOMMEND (b),** and separately cap how many places may host a shop, which the switch-off machinery
  does not currently bound. Also lift the internal limit that presently caps addressable places, because
  "we will make the galaxy bigger later" is not currently true.

**3. How unevenly are materials spread across places?**
- *(a)* Broadly even, so everyone can be self-sufficient.
- *(b)* **Deliberately uneven, so the thing you need is reliably somewhere else.**
- **Consequences.** This is the single most important input to the entire proposal and it is not a
  plumbing decision. If a player can dig everything they need within reach, no shop will ever open no
  matter how well the machinery is built, and every line of commerce code is dead. It freezes
  permanently when terrain lands.
- **RECOMMEND (b),** decided by people thinking about how the game feels, and validated by scripting the
  first week as a scenario **before** any commerce machinery is built.

**4. Does the world ever reset?**
- *(a)* Never. One persistent universe.
- *(b)* Seasonal resets every few weeks, with materials re-scattered.
- **Consequences.** The game closest to this proposal sustains a fully player-built economy on about two
  thousand players by resetting roughly every ten to seventy days, and it has run that cold start about
  a hundred and thirty-five times. Resetting permanently solves both the money-supply question and the
  problem of joining a mature world. It also contradicts a persistent universe, which is the promise.
  Never resetting makes abandonment, inheritance and inequality policy load-bearing forever.
- **RECOMMEND (a),** decided **now and deliberately** rather than discovered in year three, with those
  three consequences accepted in writing.

**5. Where do living people and creatures live?**
- *(a)* Everywhere, including towns.
- *(b)* **Wilderness, the space between places, and ruins. Towns and stations stay player-built and
  player-staffed.**
- **Consequences.** A visible person standing in a town who can be talked to and fought but cannot trade
  reads as unfinished, because every game that put people in towns put shopkeepers in towns. Under (b)
  the absence of computer-run shops is legible: there are no shopkeepers because there are no
  computer-run towns.
- **RECOMMEND (b).** It costs nothing to decide now and it removes the pressure that would otherwise
  bring price simulation back through the side door.

**6. Do we charge a hoard, and how?**
- *(a)* No. Fees on trades only.
- *(b)* **Yes, as physical wear: things that sit still degrade and need real material to maintain.**
- **Consequences.** Every other drain scales with trading, and the measured data shows hoarders do not
  trade — circulation speed fell sixty-one percent over nine and a half years in the reference economy.
  Without (b) a near-fixed supply seizes up. A monetary charge over elapsed time would be the obvious
  fix and it would destroy the entire switch-places-off property, so it must be physical wear.
- **RECOMMEND (b),** sized so parking wealth costs more than it gains, and subject to the rule that a
  dormant thing may degrade but may never be destroyed while nobody is present.

**7. Are tills robbable?**
- *(a)* No — protected by a rule.
- *(b)* **Yes, in places where full loss already applies.**
- **Consequences.** A strongbox in a breakable building is robbable unless something says otherwise. Under
  (b) that is excellent content and a reason to fortify; under (a) it is a special case that has to be
  justified in a world where everything else is breakable. Either way it changes the security model of
  every shop and must be settled before the shop record is frozen.
- **RECOMMEND (b)** where full loss applies, with a safe-place option available to shop owners who want it.

**8. Do we ship the searchable catalogue ourselves?**
- *(a)* No; let players find shops in the world.
- *(b)* **Yes, deliberately, with its staleness shown honestly, outside the game, never read by the game.**
- **Consequences.** Every shipped player-shop economy ended up with an index, and where none was
  provided, players built one by watching their own network traffic and publishing it. Option (a) means
  somebody else owns the data and the quality bar. Option (b) requires a test proving the game works
  perfectly with it switched off, or the game has quietly started depending on the economy.
- **RECOMMEND (b),** with the off-switch test as a hard gate.

**9. What may a free gift contain?**
- *(a)* Anything useful.
- *(b)* **Nothing the world itself will take in exchange for anything.**
- **Consequences.** A free starting kit plus one published conversion rate is a money faucet in costume:
  if the kit's convertible contents are worth five hundred, then a thousand alternate accounts mint five
  hundred thousand from nothing, repeatably. This exploit exists purely because the free-gift decision
  and the published-rate decision were made in different documents.
- **RECOMMEND (b)** as a standing rule with an automated check, and give tools and shelter rather than
  raw inputs.

**10. Company shares now, or later?**
- *(a)* Now.
- *(b)* **Later.** A company is a name, a member list and permission slips.
- **Consequences.** Valuing or trading a share needs a common yardstick, and under the recommended answer
  the yardstick is discovered by watching what players actually trade in rather than declared up front.
  Building shares first means guessing it.
- **RECOMMEND (b),** and note it is a deferral rather than a refusal: once the market has revealed its
  currency, shares become expressible with no new decision.

---

## 11. Open questions, and what could not be verified

### 11.1 Genuinely open — no answer in either design

1. **Who taxes a shop inside a ship inside a station**, when the parent changes on every flight. Answerable
   only once a minted realm identity exists (§5.7).
2. **What happens when a player creates a realm.** `ProfileKind::Ship` exists with no corresponding
   `RealmKindTag`; the stated end goal is ships and stations built from blocks. Every per-realm money
   mechanism breaks on this, which is a further reason the recommendation mints nothing.
3. **The durability rule for a purchase confirmation** — gate on a real fsync (with the per-sale cost
   measured) or declare it advisory with a bounded loss window (§8.1).
4. **How the per-sale fsync cost actually measures.** Nobody has measured it, and it decides (3).
5. **Secret distribution for grant verification keys.** Today only the gateway receives key material.
   Ed25519 makes this trivial (a public value), but the provisioning path does not exist.
6. **A retention bound for the purchase dedup journal**, on a record that would grow once per sale forever.
7. **The block-edit rate**, which sets the telemetry volume. The 0.15 and 1.0 events-per-second figures
   are **assumptions** and must be replaced by a measurement from the first live world.
8. **Whether ambient building, fuel and repair actually reach the 33–54% destruction band** without
   large organised combat. Unknowable until those systems ship, and the whole demand side rests on it.

### 11.2 Could not be verified

- **Several long-run player-economy cases.** The web-search budget was exhausted before reaching Life is
  Feudal (a large-world player-economy MMO that shut down), Wurm Online's actual economic figures, or
  any wealth-distribution **measurement** for a player-priced economy. ⚠ **The wealth-concentration
  claim rests on one designer's retrospective statement, not on data, and should be re-sourced before
  it is relied on.**
- **Games not reached at all**, and which must not be treated as researched: Mortal Online (the first),
  Rust vending machines, ARK, Space Engineers, and the bazaar fee, listing-cap and duration figures of
  the closest historical precedent. Several reference wikis actively block automated fetching.
- **The commonly cited counterexample of a long-lived third-party escrow service** in the reference
  economy. Treat "escrow survived where banks failed" as **unverified**; the structural claim that stands
  is narrower: every instrument that promised **yield** or performed **maturity transformation**
  collapsed, and nothing in the record indicts pure custody.
- **A land-area figure** for the nearest live player-only economy, so no per-square-kilometre density
  comparison is asserted (§7.2).
- **The unexplained 1.36× gap** between the reference economy's aggregate destruction series and its
  event-level dump. Treat them as different aggregates and never mix them in one derivation.
- **The decomposition of that economy's 856-trillion accounting gap.** The gap is arithmetic; the causes
  are not verified.

### 11.3 Findings I did not sustain

Recorded so they are not silently dropped.

1. **"Do not spend engineering on per-place commerce features that will be used almost nowhere."**
   *Partially rejected.* The measurement behind it (trade concentrates even with computer sellers
   everywhere) is sound and is adopted in §7.1. The engineering conclusion is not: a frontier counter and
   a hub counter are the **same code**, and branching on place would violate the one-tooling and
   features-once-run-anywhere rules. The saving is in content and operations — how many places host
   commerce — not in code.
2. **"Ten open places should be the starting world size."** *Reframed rather than adopted.* Two distinct
   levers were conflated. The **addressable** ceiling is 3,364 systems and is a code limit; **how many
   are open at once** is a lifecycle policy. Ten is a good open-at-once target at four hundred players
   (§7.2) and is not a world size. Setting a world size today would also change nothing, because the
   generator's range has **zero readers**.
3. **The reserved catalogue bulk kind as a cost credit.** *Partially rejected.* `BulkKind::Catalog` does
   exist (`crates/wire/src/channels.rs:170-176`), but `BulkMsg` has no transport class either — the same
   gap as the event message — so the credit saves an enum variant, not the arm. And in the recommended
   design the catalogue lives **outside** the world as a website, so that variant is not needed for v1 at
   all.
4. **"Clamp the dormant closed form and you are done."** *Adopted but insufficient as stated.* The clamp
   is necessary and is in RULE PDE-1, but it is not sufficient on its own: the woken shard must also
   **author** the destruction as a live deviation with per-item typed loss reasons, or the conservation
   identity cannot close over a structure that was going to expire while nobody was watching.

### 11.4 What both designs got right, and should not be re-litigated

No branch on shard kind anywhere. No second **authority** commit point — the directory compare-and-set
stays the only one. The till genuinely removes the seven-phase saga from the sale path. Object-safe
`dyn` for the port, matching the codebase's own stated reason. The switch-off argument is backed by
shipped kernel code: teardown requires **all three** of out-of-closure, the child's own affirmative empty
report, and surviving the drain veto (`crates/sim/src/rlm.rs:465-482, :551-566`, with the child as
occupancy authority at `crates/wire/src/intershard.rs:517-532`). Expiry as a comparison that performs no
action. The no-time-based-rent rule with a tripwire test — the single highest-value rule in either
design. The company table as a new store key family rather than a fifth directory arm, correct on both
counts. And no new third-party library anywhere: `hmac`, `sha2` and `ed25519-dalek` are already
workspace dependencies (`Cargo.toml:77-79`), and the two proptest frameworks are already dev-dependencies.
