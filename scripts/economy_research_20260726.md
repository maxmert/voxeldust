# Economy research — decision-grade report (2026-07-26)

Produced by 9 research agents + 1 library verifier + 3 adversarial critics. This document is (a) the
artefact the user reads and decides from, and (b) the design input for a future economy arc (same
register as `scripts/realm_lifecycle_design.md` / `scripts/visual_universe_arc_plan.md`).

---

## 🟥 REVERSAL NOTE (2026-07-27) — "the currency is a material" is REVERSED; money is a real currency again

**Read `scripts/money_and_markets_design.md`. It is the current word on money AND on markets, and it
supersedes the abandoned consignment-market design.** On 2026-07-27 the user reversed yesterday's
"currency is a material" choice: *"having money as items is not a very scalable idea — we will not be
able to transfer money anywhere, we will not be able to have banks or similar. We need to have currency
and some ways to deal with the money locally (stations/cities/star systems) and globally (galaxy and
universe levels)."* This is a pointer only; nothing below has been rewritten.

**What that reverses in THIS report** — some of the material listed as UNNECESSARY below comes back, in a
much smaller form than originally designed:

- **The money type, minor units and currency ids (S1, D4, D9) are REINSTATED** — but as *one* monomorphic
  integer-minor-unit amount type carrying a runtime currency tag, never a type parameter (coverage counts
  regions per monomorphization). One currency at launch; the tag exists from the first line so more remain
  possible. The numeraire question is answered by the backing material, not by a market.
- **The double-entry ledger kernel (D-48) is REINSTATED**, at roughly 600 lines: one account row, credit,
  debit, transfer, hold, release, and a single conservation assertion with exactly three named exceptions.
- **Escrow is REINSTATED** as full escrow at listing and at reservation — but the goods themselves stay in
  a GAME container, so the standing rule that an economy object is never the container of record holds.
- **The account-authority-for-value lifecycle is REINSTATED and resolved differently.** Not always-homed
  accounts, and not a per-realm home: one always-on account service that is *not a place at all*, because
  the reconciler can only be asked for a realm and there is no always-alive root.

**What stays UNNECESSARY, unchanged:** every computer-controlled economic actor and everything feeding one
(D8, D13, slice E-8, the §4 catalogue, §7.7); the equilibrium solver, production-chain solving and the
ambient price field; order books, matching and clearing (the market is consignment, not an order book);
the hierarchical tax algebra with its ceilings, tariffs-at-the-common-ancestor and remittance (S-tax, D10,
D11, slice E-4); corporations as full securities (D12, D-50, D-63) — only a minimal share register
survives, as the honest replacement for deposit banks; contracts, courier adjudication and insurance as a
funded pool (D-59, D-66) — insurance especially, because a "safety net" priced by feel is issuance;
player-issued scrip and every yield-bearing instrument (D23, §5.8) — promised yield on deposits is banned.

**What is LOAD-BEARING and used verbatim in the new design:** §2.1's measured figures. The realised
transaction take (34.45 T on 870.89 T = 3.96%) is the calibration point for the venue fee; the faucet/sink
totals (196.98 − 134.06 = 62.92 T on 2,908.38 T = +29.3%/year) bracket the burn lever's required range;
the insurance net-faucet figure (+2.797 T/month) is the argument against any compensation mechanism; the
seven-division corporate wallet is the shape for treasuries; and the 100%-escrow rule is adopted as law.

---

## 🔶 SUPERSEDING POINTER (2026-07-26) — the user has proposed a PURELY PLAYER-DRIVEN economy

**Read `scripts/player_driven_economy_comparison.md` before designing from this report.** After this
report was written the user proposed removing economic **simulation** entirely: no computer-controlled
traders, no formula-driven prices, no background market evolution — only plumbing (companies,
player-built shops, stock, owner-set prices, buying, selling, ownership, taxes) plus telemetry, with all
behaviour coming from real players. That document evaluates the proposal, **recommends adopting it**, and
is the later decision where the two disagree. This is a pointer only; nothing below has been rewritten.

**If the proposal is adopted, these parts of THIS report become UNNECESSARY** (details and counts in §3
of the new document): the money type / minor units / currency ids / multi-currency exchange and the
numeraire question (**S1**, **D4**, **D9**); the double-entry ledger kernel (**D-48**); order books,
matching, clearing, escrow-as-an-economy-holding and book partitioning (**D1**, slice **E-7**); every
computer-controlled economic actor and everything feeding one — trading strategies, agent tick placement,
market makers, the adaptive bid, the cold-start liquidity mechanism, the equilibrium solver,
production-chain solving, the ambient price field (**D8**, **D13**, slice **E-8**, §4 catalogue, §7.7);
the hierarchical tax algebra with its ceilings, floors, tariffs-at-the-common-ancestor, remittance and
rate-change front-running (**S-tax**, **D10**, **D11**, slice **E-4**); corporations as securities —
shares, dividends, registries, delisting, liquidation, pro-rata distribution (**D12**, **D-50**,
**D-63**); contracts, collateral, courier adjudication and insurance as a funded pool (**D-59**,
**D-66**); player-issued scrip and every yield-bearing instrument (**D23**, §5.8); the
account-authority-for-value lifecycle including always-homed accounts, never-dormant per-account home
shards and escheatment of balances (§7.10); the counterfactual economy simulator and the
verified-matcher differential oracle (**D-56**, **D-57**); the money half of the live-invariant halt flag
(**D-61**); and the two highest-stakes decisions **D1** (market mechanism) and **D4** (money
representation) simply stop existing.

**WEAKENED or RE-CLASSIFIED — do not act on these as written:**
- **D-53** (teardown safety) was classified "unnecessary" in an earlier pass. That is **wrong**: only
  book rehydration disappears. Its **storage-topology prerequisite survives and becomes MORE
  load-bearing** — a lost order book is an inconvenience; a lost shop stock is confiscated player
  property. Carry its three-way decision forward verbatim as a blocking prerequisite.
- **§7.6 / S11 items-as-ledger-positions** and **S8 item conservation** get *heavier*, not lighter: with
  no formula holding an expectation about any price, a duplicated stack produces **no** market-wide
  anomaly, so the accounting identity becomes the **sole** fraud detector.
- **§2.1/§2.2 growth figure**: "+2.16%/month" is one month's snapshot, not the rate. The 37-month
  geometric rate is **+1.208%/month** (doubling in 58 months, not 32). The **OSRS tax is now 2%** (raised
  2025-05-29), not 1%, still capped at 5M/item — and that cap makes it regressive at the top.
- **§7.11c company equity** is *deferred*, not refused: valuing a share needs a unit of account, which
  the recommendation lets the market reveal rather than declaring up front.

**UNCHANGED and still binding:** the closed creation/destruction taxonomy and the supply identity with
**no reconciliation plug**; integer money arithmetic and the no-seconds rule; items as append-only
positions with never-reused ids and zero-sum splits (**S11**); item conservation as an oracle with
deliberately failing controls (**S8**); the expiry stamp on every order-like record (**S9**); the
closed-form integer resource distribution (**S12**); the per-destroyed-block destruction reason
(**S13**); recipe-graph acyclicity validated in the **game** (**S3**); the read-only Tier-B analytics
seam; the harness oracles; and every §12.3 re-sourcing obligation.

---

⚠ **Provenance caveat — read before trusting a [V] tag.** Three pooled verifier passes were planned;
**two died on a session limit and never ran**: the pooled *numbers* pass and the pooled *repo-fit* pass.
What did run: the **library** verifier (all 47 crates, licenses, maintenance — the hallucinated-crate
sweep is complete), and three adversarial critics who independently re-verified in their own lanes —
critic 2 rebuilt §2.1/§2.2/§2.3 from primary sources (live ESI + the MER ZIP + redb's own benchmark
constants) and critic 1 re-checked the in-repo claims to `file:line`. **The residual gap is therefore
quantitative claims OUTSIDE §2 and §12.3** (chiefly the per-game figures in §5 and the tooling
versions in §3.6–§3.8): those carry their original research-agent confidence tag and were never
independently pooled-checked. §12.3 lists every figure already known to need re-sourcing; treat an
unlisted `[L]` outside §2 as "one agent's single source", not as cross-checked.

**Evidence convention** — every non-obvious claim is tagged:

| Tag | Meaning |
|---|---|
| **[V]** | HARD-VERIFIED: a verifier or I fetched the page / queried the API / read the source / parsed the file, on 2026-07-25 or 2026-07-26. |
| **[R]** | IN-REPO: read from this worktree (path cited). |
| **[L]** | LIKELY: single credible source, not independently cross-checked. |
| **[U]** | UNVERIFIED: recollection, a blocked URL, or a figure the verifier could not source. **Must be re-sourced before it enters a design doc as fact.** |

Where a research agent and a verifier disagreed, **the verifier's corrected version is what appears
below**, and the correction is called out so nobody re-imports the wrong version from an agent log.

**Revision 2 (2026-07-26, post-adversarial-review).** Three critics reviewed revision 1. Every
sustained finding is fixed in place; §13 "Review record" lists the verdicts, what changed, and the one
finding rejected on the merits. The corrections that changed a *conclusion* rather than a number:
the recommended event-log carriage (§8.1) was technically impossible and is retracted; the D2 authority
recommendation is inverted (§6.3/§11); the durable ledger volume was understated ~10–50× (§2.2/§8.3);
the ledger-ops ceiling was a non-derivation (§2.2); D3 durable-dormant is blocked on a storage-topology
prerequisite (§6.4); and eight whole subsystems the ask requires were missing and are now designed
(§4.10, §4.11, §6.6, §7.8–§7.18, §8.8, §8.9, §9.4, §9.5) — see §13 for the full list.

**Revision 3 (2026-07-26) — CROSS-REFERENCE ONLY, no re-analysis.** `scripts/dormant_world_simulation_design.md`
(THE WORLDLINE, revision 2) is the approved dormant-world / main-game-substrate design and it **changes,
invalidates or constrains** a number of this report's seams and decisions. Rather than rewrite this document,
each affected point carries an inline **⚠ WORLDLINE (rev 2)** note pointing at the section that supersedes it.
The points touched: **§7.1** (crate topology — overruled by the seam inversion), **§6.4** (requirement 2's
KeepAlive lever is a LAW-E1 defect; the dormant catch-up tier is deleted), **§7.7** (NPC agent strategies inside
`vd-econ` is a latent LAW-E1 violation; the closed-form tier is now a *game* substrate; the state budget is
answered), **§7.18** (answered — the economy CAN change the world, via a journaled command), **§8.1** (split into
two records), **S1**, **S2**, **S3**, **S4**, **S6** (rejected), **S7** (split into two registries), **S8**,
**S9**, **S11** (inverted — the GAME's store is the uniqueness authority), **S12** (broadened and promoted to a
hard prerequisite), **S13**, **D2** (the `Account` directory arm is removed and replaced), **D3**, **D8** (agents
become an injected object-safe port, not an enum registry), **D15** (its physical half is settled), **D16**
(sharpened, and `RespawnAfter` becomes the default), **D21** (split physical/monetary), **D22** (`season_id` must
join the worldline's day-one artefacts). Nothing else in this document changed.

---

## 1. Executive summary

### 1.1 What exists to reuse

**Nothing on crates.io can be ADOPTED into Tier-A for the market/money core, and that is a
defensible finding rather than a gap.** The survey covered 47 crates, all of which exist at the
stated names [V] (no hallucinated crates — the single naming slip is that `bourse` is a repo, the
crate is `bourse-book` [V]). Every mature Rust order book is built for HFT *concurrency*, which is
the exact inverse of what a deterministic sealed shard needs:

- `orderbook-rs` 0.12.1 (2026-07-23, MIT) has **tokio as a mandatory dependency**, plus `dashmap`
  (random-seeded hasher), `crossbeam-skiplist`, `uuid` v4, `sha2` [V]. Each is independently banned
  by our own rules (`clippy.toml` disallowed-types on default-hasher `HashMap`; "no I/O outside the
  seam") [R].
- `nautilus-model` 0.60.0 is **LGPL-3.0-or-later** [V] — a hard stop for a statically linked
  proprietary binary, before technical fit matters.
- `rustledger-core` 0.21.0 is **GPL-3.0-only** [V] (the research called this "license varies, verify";
  it is a hard stop).
- Every Rust double-entry ledger crate requires Postgres + sqlx + tokio (`cala-ledger` 0.18.5) [V].
- Every Rust CQRS/event-sourcing crate either is dead (`eventually`, last release 2020-10-04 [V]),
  unmaintained-by-declaration (`thalo` [V]), or owns the transaction/commit boundary — which
  collides head-on with "the directory CAS is the only commit point" [R].

What **is** reusable is design, not code, and there is a lot of it:

| Role | Best reuse | Legal basis |
|---|---|---|
| Order-book data structure | `limitbook` (565 SLoC, BTreeMap + VecDeque + HashMap) | Apache-2.0, portable [V] |
| Matcher state-machine shape | `matchcore` ("single-threaded, deterministic, in-memory state machine", 3 deps) | MIT/Apache-2.0 on crates.io, **Apache-2.0 on the repo — resolve before porting** [V] |
| Integer tick/lot + golden-replay + 100%-branch policy | `lanpishu6300/match-rust` | Apache-2.0 [V] |
| Ledger schema + two-phase escrow semantics | TigerBeetle docs (128-byte Account; `debits_pending`/`credits_posted`; `pending` → `post_pending_transfer`/`void_pending_transfer`/`timeout`; `ledger:u32` partitions who may transact; FX = 2 linked same-ledger transfers) | Apache-2.0 docs; reference-only (Zig service) [V] |
| Correctness proof-by-differential | Coq verified double auctions (arXiv:2410.18751, arXiv:2412.08624) — O(n log n), Ω(n log n) lower bound, 10M orders in "a couple of minutes", extracted OCaml/Haskell built explicitly for "automatic detection of violations in exchange systems by comparing their output to that of a verified program" | **NO LICENSE on either repo** → oracle-only, never vendored [V] |
| NPC trader algorithms | Bristol Stock Exchange (`BSE.py` contains ZIC, ZIP, GVWY, SHVR, SNPR, PRZI, PRSH, PRDE; **AA is a separate file, GDX is absent**) | MIT (LICENSE.md verbatim, © 2012 Dave Cliff) [V] |
| Agent population ratios | ABIDES RMSC03/RMSC04 configs (50:1 noise:value; ~1% market makers) | BSD-3-Clause; **repo ARCHIVED 2024-07-22** [V] |
| Production-side agents (price beliefs → production → professions) | Doran & Parberry, IJIGS **Vol. 7, No. 3**, 2012 + `bazaarBot` | Paper open; bazaarBot MIT, dormant since 2020-06-04 [V] |
| Closed-form regional pricing | Victoria 3 (`price = base·[1 + 0.75·clamp((BUY−SELL)/min(BUY,SELL), ±1)]`, 25–175% bounds, `local = MAPI·market + (1−MAPI)·state`) — formulas verbatim | Algorithm, not code [V] |
| Cheapest per-region price process | Endless Sky (`KEEP=.89`, `VOLUME=2000`, `LIMIT=20000`, `price = base + (int)(−100·erf(supply/LIMIT))`) — constants verbatim from `source/System.cpp` | Code is GPL-3.0 → **re-derive, never copy** [V] |
| Cross-shard price discovery without a query | Veloren: the price+supply vector rides as a payload **on the goods delivery itself** | Code is GPL-3.0 → re-derive only [V] |
| Taxes/laws as data over the action stream | Eco (trigger → source-filter → condition → action graphs; nested individual→town→nation→global government tiers; Treasury; pro-rated multi-owner wealth tax) | Design; EcoModKit has **NO license** [V] |
| Hierarchical tax composition | US combined state+local sales tax (additive on ONE base); EVE planetary customs (NPC 10% **+** player POCO %, capped at 100% total) | Design precedent [V] |
| Anti-haven lever | OECD Pillar Two: 15% floor, jurisdictional blending, QDMTT → IIR → UTPR rule order | Design precedent [V] |
| Dashboard specification | EVE's Monthly Economic Report (72 named ISK flows; 4 price indices monthly since 2003-11; daily money supply + velocity since 2017) | Read-only reference; **do not vendor CCP's data** [V] |

### 1.2 What must be built

Roughly 6,000–8,500 lines of pure Tier-A Rust, none of it exotic, plus a Tier-B analytics tier.
(Revision 1 said 4,000–6,000; the review surfaced five whole mechanisms it had not counted — direct
trade, contracts/collateral, the corporation organisation layer, the material faucet/sink pair, and the
live-invariant/reversal machinery — worth ~2,000–2,500 lines.)

1. **Money type** — `Money { amount: i128, currency: CurrencyId }` over integer minor units, with a
   per-currency `minor_exponent`, checked arithmetic, ONE named rounding policy, and largest-remainder
   apportionment. ~300 lines.
2. **Double-entry ledger kernel** — accounts (four monotone `u128`/`i128` counters, balance *computed*),
   append-only entries, `Σdebits == Σcredits` **per currency**, two-phase pending/post/void with a
   universe-tick timeout, a closed `Faucet`/`Sink` taxonomy. ~1,000–1,500 lines.
3. **Order book + clearing** — `BTreeMap<Price, Slab/VecDeque>` book, one clearing function, the
   order-type × time-in-force × self-trade matrix as a table-driven test. ~600–1,200 lines.
4. **Tax composition** — situs resolution (deepest containing realm), additive rate summation over
   the ancestor chain, VAT input credit, ceiling/floor/top-up clamps, LCA-edge tariffs. ~400 lines,
   pinned by 5 algebraic proptests.
5. **NPC agents** — ZI-C + PRZI + the trivial baselines, each its own monomorphic module behind an
   enum registry. ~500 lines.
6. **Conservation oracles + chaos cells** — siblings of the existing
   `verify_transient_conservation_tick` / `verify_input_conservation` / `verify_authority_unique`
   family [R].
7. **The event log + analytics projection** — the one genuinely new *architectural* piece, because
   no existing machinery answers "economy status across galaxies". It needs a NEW `sim::io` seam
   (`EventSink`) — see §8.1; the revision-1 "zero new seam" option was impossible.
8. **Item/inventory as ledger positions** rather than mutable counts — the change that makes
   no-dupe provable with the same oracle as money. **Must constrain the FIRST line of P6/P7 inventory
   code** (S11), not be retrofitted after it.
9. **Direct player-to-player trade** — a two-sided atomic swap; the most common economic interaction,
   the classic dupe vector, and the cheapest possible first feature (§7.8). ~400 lines.
10. **Contracts + collateral + courier adjudication** — under sealed shards these are *more*
    load-bearing than order books, because they are what makes a distributed market work without a
    global one (§7.9). ~600 lines.
11. **The material faucet/sink pair** — extraction yield per voxel and the destruction/loot ratio.
    These, not the fee schedule, decide whether the economy works, and P4/P5/P6 fix their geometry
    irreversibly (§4.10).
12. **Live-invariant behaviour + the reversal primitive** — what a running shard does when I1/I3/I10
    fails with players connected, and how a cohort is made whole without minting (§6.6). ~350 lines.
13. **The corporation organisation layer** — permissions, membership, member-held assets, role-change
    front-run delay, corp-action audit (§7.11). ~500 lines.

### 1.3 The decisions the user must make

Full options and tradeoffs in §11 — **twenty-four** numbered choices (revision 1 had fourteen). The
**eight** that gate everything else:

| # | Decision | Why it gates |
|---|---|---|
| **D1** | **Market mechanism**: frequent batch (uniform-price) auction vs continuous double auction vs two-tier. | Batch clearing is a pure function of an unordered order *set* ⇒ replayable, latency-fair (we have NO client prediction and a 100–150 ms buffer [R]). CDA outcomes depend on arrival order ⇒ exactly one authority, no cross-check. This decides the whole determinism story. (The Coq cross-check is weaker than revision 1 claimed — see §4.1.) |
| **D2** | **Authority placement**: accounts-by-owner + markets-by-(realm,commodity) (A) vs one never-dormant economy-capability shard (B) vs per-station hierarchy (C). | Decides directory key-space growth, RLM dormancy exposure, whether the economy is a hotspot, **and where NPC agents may tick** — the constraint that inverted this recommendation in revision 2. |
| **D3** | **RLM dormancy answer** for books/wallets/escrow. | A market book with long-lived resting orders cannot live in a realm shard that RLM may kill. **Durable-dormant is blocked on a storage-topology prerequisite the k3d manifests do not satisfy** (§6.4). |
| **D4** | **Money representation**: `i128` minor-units newtype vs `rust_decimal` vs `fixed`. | Nearly irreversible once the wire/TLV format freezes. |
| **D5** | **Analytics stack**: minimal (offline Parquet + `datafusion-cli`) vs standard (+ ClickHouse + Grafana) vs heavy. | Every option here is NEW dependencies; the user decides all deps. |
| **D15** | **Where escrow lives**: co-located with the BOOK, or held by the ACCOUNT owner. | Revision 1 asserted both. They are mutually exclusive, and the choice decides whether a fill is a local mutation or a saga — i.e. the entire throughput model and the E-7 slice shape (§6.4/§7.2). |
| **D16** | **Does mined terrain regenerate?** Finite-per-planet vs respawning. | A P4 decision, not a P9 one: it fixes the primary material faucet's geometry irreversibly, and a finite economy and a regenerating one are different games (§4.10). |
| **D17** | **Audit granularity**: per-event provenance, or aggregated micro-faucets. | Revision 1 claimed both. Per-event auditability (I12, "prove this account earned that ISK") and per-minute/per-session faucet aggregation cannot both hold; the choice sets the ledger volume, the storage bill and what a moderation appeal can prove (§8.3). |

### 1.4 Recommended shape (one paragraph)

Make the economy a **policy fan-out on machinery that already exists**, not a new subsystem. Value is
a registered `TransferableKind`; a settlement is the existing transfer saga (reserve → directory CAS →
post); escrow is TigerBeetle's two-phase pending transfer, whose "posting cannot fail" property makes
the saga's compensation (`void`) trivially safe — unlike compensating a posted credit, which is the
classic un-payable saga. Add a **venue/economy** capability to the existing `ShardProfile` lattice
(`crates/sim/src/capability.rs:71`, which already has 8 fields and a `build()` that rejects incoherent
requests [R]) **plus a per-realm capability override on the RLM spawn path** — because `ShardProfile`
today is a pure function of realm KIND (`RealmCoord::profile_kind()` → `profile_kind_of(RealmKindTag)`
[R]), so a bare field would make *every* station a venue (§7.5). Carry economic *gameplay* data on the
reserved `Signal` arm (P9). **Key markets `Market(RealmId, CommodityId)` and accounts
`Account(AccountId)` from day one** (D2 option A): books must be co-located with the NPC agents that
quote into them, and concentrating them on a single never-dormant economy shard is arithmetically
impossible (§2.3). One shard owning many market keys is then a *deployment configuration* of the same
code, and it is reversible; the other direction is not. Markets are realm-local (HR1 forces it; EVE's
own domain model is already region-partitioned [V]) so cross-region price differences are arbitraged by
physically hauling goods — which turns min-cost-flow spatial equilibrium into gameplay instead of
decoration, **provided regional divergence is engineered deliberately** (§7.13 — it is the documented
failure mode of the price family we recommend). Prices are a clamped closed form over two accumulated
integer scalars per (good, realm) for ambient markets, with real order books only where a venue
capability is enabled. The whole
monetary layer is *queries over one append-only double-entry journal*, so faucet/sink reports, money
supply, velocity, Gini/Theil and price indices are projections, not features — and the accounting
identity `Δ money_supply == Σ faucets − Σ sinks` becomes a CI gate rather than a forensic question,
**backed by per-arm faucet budgets** (I16) so a correctly-*declared* mint can never leak forever
unnoticed. Ship the aggregate layer first and standalone; agents are strictly additive on top (Star
Citizen has been publicly designing exactly this agent economy for ~8 years and it is still an unshipped
roadmap deliverable [V]).

⚠ **One premise revision 1 over-claimed, corrected here because five later sections leaned on it.**
LF-1 guarantees every realm on a leaf→LCA up-path is at-least-Dormant [R] — it does **not** guarantee
those realms can relay or process a Signal. `signal_relay` is carried by `profiles::galaxy()` and
`profiles::station()` only; `system()`, `planet()`, `ship()` and `area()` all have it false, and a unit
test exists asserting exactly that (`a_galaxy_coord_selects_a_signal_relay_profile_a_system_does_not`,
`crates/sim/src/capability.rs:417`) [R]. What a *Dormant* realm can process is also undefined
(Dormant-as-a-cheaper-capability is itself deferred — "v1 runs a full server per realm even when
unoccupied" [R]). So "the jurisdiction chain of an active market is always live" holds only in the weak
liveness sense; **do not build tax remittance routing on relay capability that only Galaxy and Station
have today.** S10 must feed this into the P9 Signal design, and possibly into
`profiles::system()`/`planet()`.

---

## 2. What "EVE level" actually means

### 2.1 The verified numbers

All figures below were measured or parsed on 2026-07-25/26 against live ESI and the actual
`EVEOnline_MER_202606.zip` (content-length 65,072,093 B [V]).

| Metric | Value | Source | Conf. |
|---|---|---|---|
| Solar systems | **8,490** | ESI `/universe/systems/` → 8,490 ids (exact) | **[V]** |
| Region ids | **114** | ESI `/universe/regions/` | **[V]** |
| Regions with *economic activity* | **69** | MER `1_key_economic_figures_by_region.csv` = 69 rows | **[V]** ⚠ corrected |
| Item types with a published mean value | **50,301** | MER `static_type_values.csv` | **[V]** |
| Active (region, type) market pairs | **≈353,000** | X-Pages census of `/markets/{region}/types/` | **[V]** |
| **Live resting orders, universe-wide** | **1.55–1.67 M** (ΣX-Pages = 1,665–1,666 pages × ≤1000); an independent full snapshot count gives **1,609,638** | live ESI census; EVE Ref snapshot 2026-07-24T23:45Z (220,264,568 B raw CSV / 20,660,158 B bz2) | **[V]** |
| Resting orders in The Forge (Jita) | ≈409,000 = **24.6%** of orders by count | ESI `/markets/10000002/orders/` X-Pages 409 | **[V]** |
| Order size on the wire | 237 B/order as ESI JSON ⇒ ~50–60 B packed ⇒ **~80–100 MB for the entire New Eden book** | content-length 237,297 for 1,000 rows | **[V]** |
| Matched trades/day, universe-wide | **≈0.5–1.5 M/day ⇒ 6–17 trades/s mean**; an independent sum of EVE Ref daily market-history gives **575,369 trades on 2026-07-22** | sampled ESI history extrapolation; EVE Ref archive | **[V]** (sampling **[L]**) |
| Money supply (2026-06-30) | **2,908.38 T ISK** = 2,330.19 T character + 578.19 T corp (exact to the digit: 2,908,377,752,883,101.5) | MER `money_supply.csv`, last row | **[V]** |
| ISK velocity | **0.2792** (0.1787 excl. accessories) | same | **[V]** |
| Money-supply series | 3,449 **daily** rows, 2017-01-01 → 2026-06-30 | same | **[V]** |
| Faucets / sinks, June 2026 | **196.98 T / 134.06 T ⇒ net +62.92 T = +2.16% of M in one month** | MER `sinks_and_faucets_history.csv` | **[V]** |
| Named ISK flow categories | **72** distinct `entry_name` (and 72 `entry_id`), daily, 2023-06-01 → 2026-06-30 (69,853 rows) | same | **[V]** |
| Top 5 sinks (Jun 2026) | Transaction Tax 34.45 T · LP Store 18.68 T · Skill Purchase 14.96 T · Manufacturing 12.13 T · Broker's Fee 11.64 T | same | **[V]** |
| Top 5 faucets (Jun 2026) | Bounty Prizes 58.47 T · Commodity (Market) 52.90 T · ESS Main Bank Autopayment 31.31 T · Sansha Incursions 16.92 T · Triglavian Invasions 10.97 T | same | **[V]** |
| **Insurance is a NET FAUCET** | +2.80 T/month (payouts 4.495 T vs premiums 1.698 T) | same | **[V]** |
| Market trade value | **870.89 T ISK/month ≈ 29.0 T/day** | Σ `trade_value` over the 69 regions | **[V]** |
| Jita concentration | The Forge = 599.10 T of 870.89 T = **68.8% of value on 24.6% of orders** | regional CSV + live census | **[V]** |
| ⚠ *corrected* concentration ratio | share ratio 68.8/24.6 = **2.80×**; value/order 599.10 T/409 k = **1,465 M ISK/order** vs rest-of-universe 271.79 T/1.201 M = **226 M** ⇒ **6.5×** | derived from the two rows above | **[V]** ⚠ corrected (revision 1 said "~9×", which follows from neither) |
| Effective transaction-tax rate (audit identity) | 34.45 / 870.89 = **3.96%** (consistent with 7.5% base → 3.3–3.6% at Accounting V) | derived from two MER CSVs | **[V]** |
| Destruction events, June 2026 | **771,903 rows** (≈25.7 k/day), 81,270,286 B event-level CSV | MER `kill_dump.csv`, parsed | **[V]** ⚠ corrected (research said 762,959) |
| Price indices | **4** (Consumer, Mineral, Primary Producer, Secondary Producer) × sub-index basket, **monthly since 2003-11-01**, 16,238 rows | MER `economy_indices_details.csv` | **[V]** |
| Faucet/sink fact table (3 yr, all categories) | 69,695 rows / **4,067,080 bytes** | R6 measurement | **[V]** |
| Max active orders per character | **305** = 5 + 5×(4+8+16+32) | EVE Uni wiki | **[V]** |
| Max order duration | **90 days**; expires at `issued + duration` | live ESI + spec | **[V]** |
| Price tick size | **4 significant figures** (since 2020-03-10) | CCP "Broker Relations" | **[V]** |
| Buy-order escrow | **100%** (Margin Trading partial escrow removed 2020) | same | **[V]** |
| Corporation wallet divisions | **exactly 7** (`division` min 1 max 7, array maxItems 7) | ESI swagger | **[V]** |
| ESI cache TTLs | region orders 300 s · `/markets/prices/` 3600 s · public contracts 1800 s · **`/universe/structures/` 3600 s** · corp wallets 300 s · history `maxItems: 500` | swagger v1.36 (860,667 B, 180 paths) | **[V]** ⚠ corrected (research said structures 300 s) |
| Tranquility node model | 1 node = **1 single-threaded process on 1 core**; ~90–100 blades × 2 nodes ≈ 204 nodes; **four nodes serve ALL market regions**; a dedicated node for Jita and a second for The Forge market | CCP dev blogs | **[V]** |
| DB tier | **one SQL Server primary/standby pair**; 250 M txn/day (~2.9 k/s avg); hosts reaching 32–128 cores and 2–4 TB RAM; CCP: DB "starting to become our main bottleneck" | CCP + highscalability | **[V]** / **[L]** |
| Time dilation floor | **10%** of real speed | CCP | **[V]** |
| Botting enforcement | **3,165 accounts** banned for macro use in June 2026 alone | CCP Monthly Ban Report | **[V]** |
| Concurrency | ACU **24,518** (Feb 2026); all-time peak 65,303 (2013-05-05) | third-party trackers | **[L]** |

⚠ **Two EVE ISK-flow figures in the research are mutually inconsistent and must not be used as
design targets**: "≈524 T ISK/day traded (Sept 2022)" vs "777.3 T ISK/month spent (Feb 2026)" differ
~20× in implied daily terms [V that they conflict]. They measure different aggregates. The figures in
the table above are the ones re-derived from the primary MER ZIP; use only those.

### 2.2 The derived budget for us

⚠ **Revision 2 rebuilt this table.** Four lines were wrong: the ledger-ops ceiling was a
non-derivation, the ledger volume was understated 10–50× *and* carried a straight 2× arithmetic error,
the concentration ratio did not follow, and the client read fan-out — named as a binding cost — was
never numbered. **Everything below is derived from ONE consistent chain**: events/s → postings/event →
postings/s → bytes/yr, with the fan-out written down.

**The posting fan-out per event** (this was the missing link):

| Event | Postings | Legs |
|---|---|---|
| A fill | **5–8** | buyer debit, seller credit, transaction tax, broker fee, dust sink, (± escrow release, ± FX pair) |
| An order placement (100% escrow) | **2** | account → `Escrow(OrderId)` pending |
| A cancel / expiry | **2** | void back |
| A micro-faucet credit (bounty, salvage) | **2** | `Faucet::X` → account |
| A crafting job | **4 + inputs** | input burns (declared `Sink`), output mint (declared `Faucet`), job fee, index surcharge |
| A tax remittance | **3–4** | clearing debit, beneficiary credit, dust |

| Budget line | Value | Derivation |
|---|---|---|
| Order flow, 100 k CCU at 1 order/player/10 min | **~167 orders/s avg** (100,000/600) | arithmetic. ⚠ The 1-order/10-min *rate* is an assumption; cross-check against EVE's own data, which the report already holds: 1.61 M live orders with a ≤90-day duration and a ~7-day mean lifetime implies a replacement rate of only **~2.7 orders/s** at 24.5 k ACU ⇒ ~11/s scaled to 100 k CCU, or ~10–100/s including modifications. **167/s is a conservative budget, not a measurement.** |
| Order flow, 1 M CCU (40× EVE) | **~1,700 orders/s avg**, maybe 10× peak | arithmetic |
| Single-core matching headroom | `limitbook` measures 204 ns non-crossing / 290 ns matching / ~31 ns market+cancel on a **hot 2,000-order** book ⇒ 3.4–4.9 M orders/s [V]; `matchcore` 61–142 ns/submit (vendor-reported, **not reproduced**) [V]. ⚠ **Cold-cache correction**: across 10⁵–10⁶ sparse books expect 2–4 cache misses + a B-tree descent ⇒ **1–5 µs/order ⇒ 0.2–1 M orders/s** | **≥3 orders of magnitude of headroom** (revision 1 said 4; it survives the cold-cache correction either way). Matching throughput is a NON-PROBLEM. |
| Per-tick matching budget at 20 Hz | the economy gets a **SLICE** of the tick, not all of it: at 10–20% of a 50 ms tick ⇒ **15 k–60 k warm** or **1 k–10 k cold** orders/tick | derived. ⚠ Revision 1's "150,000–300,000/tick" assumed 100% of the tick, and its top end needed 6 M orders/s — above its own stated range. |
| **Clearing-sweep cost** (the cost that scales with cardinality, not flow) | a batch auction must visit every book per batch period: at 10⁶ books and a 5 s period that is **200,000 book visits/s ⇒ ~0.20 cores of clearing mostly-empty books.** ⇒ **the dirty-set rule (§4.1) is mandatory, not an optimisation**: clear only books with ≥1 new/modified/expired order since the last batch ⇒ O(dirty), not O(books) | derived; absent from revision 1 |
| Market cardinality (the real wall) | ⚠ **restated at OUR venue granularity.** EVE's ~353 k active (region,type) pairs is a **REGION**-scoped model over 114 regions × 50,301 types = 5.73 M possible; ours is **realm**-scoped and §7.5 puts venues at station/area. EVE has ~5×10³ NPC stations + ~10⁵ player structures ⇒ station-granular × 5×10⁴ item kinds = **2.5×10⁸–5×10⁹ POSSIBLE books**, i.e. 100–1,000× revision 1's budgeted 10⁶ | ⇒ the instantiated set must be bounded by CONFIG, not by possibility: `max_venues`, `max_books_per_venue`, `max_orders_per_book`, plus the venue-capability gate. This is an explicit RLM capacity-planning input. (Revision 1's "~1.7 M valid pairs" conflated the live-**order** count with a pair count.) |
| Empty-book footprint | ⚠ **~200 B**, not 1 KB: two `BTreeMap`s (24 B each empty), a `Slab`, an id index ⇒ 10⁶ books ≈ **150–250 MB** resident before a single order | derived. Revision 1's "1 GB" was 4–7× pessimistic. The *conclusion* (lazy + empty-cheap) is unchanged and correct. |
| Order-book state | ~80–100 MB packed for the whole universe; a 100 k-order realm ≈ 5 MB; **a hub book at `max_orders_per_actor`×5,000 actors = 1.5 M orders ≈ 150–300 MB resident** | derived from 50–60 B/order ⇒ `max_orders_per_book` is the cap that actually bounds it |
| **Durable ledger volume** ⚠ **rebuilt** | Revision 1's "~10⁶ economic txns/day" was EVE's **trade count alone** (575,369 trades on 2026-07-22 [V]) and omitted every faucet, fee, job, insurance payout, contract and bounty — which dominate. Rebuilt from EVE's own verified faucet: 1.95 T ISK/day Bounty Prizes ÷ 200–500 k ISK/NPC-kill = **4–10 M credit events/day** = 45–113/s ⇒ 90–226 postings/s; trades 575 k/day × 5–8 postings = **33–53 postings/s**; plus fees, jobs, contracts, insurance ⇒ **300–600 postings/s sustained** ⇒ **0.95–1.9×10¹⁰ postings/yr** ⇒ at 56–128 B/record **≈0.53–2.4 TB/yr** | ⚠ Revision 1 said "41–47 GB/yr", which was wrong twice: the upper bound was computed against 3.65×10⁸ instead of 7.3×10⁸ (7.3e8 × 128 B = **93.4 GB**, not 47), and the input count was 10–50× low. **This is 3–4 orders of magnitude above the provisioned volumes** (see the PVC row) and it makes D-54 retention a GATING decision, not a footnote. |
| **PVC / disk sizing** (missing entirely from revision 1) | `deploy/k3d/50-shard.yaml` requests **256 Mi** per shard (+64 Mi) and `30-orch.yaml` **1 Gi** (+64 Mi), both `accessModes: [ReadWriteOnce]`, `storageClassName: local-path` [R] | ⇒ irreconcilable with 0.5–2.4 TB/yr **and** with a single hub book's 150–300 MB. Shard PVC size must become `f(max_orders_per_book, journal_retention_ticks)` and the manifests must be re-derived from it. Also the direct blocker on D3 (§6.4). |
| Analytics write rate | **~10–40 M econ rows/day ⇒ ~116–460 rows/s avg, 2–5 k/s peak** (rebuilt from the corrected posting rate) | R6 projection, re-derived |
| Analytics storage | **~123–492 MB/day, 45–180 GB/yr** in Parquet+zstd at the measured 12.29 B/row; 5-year retention **0.2–0.9 TB** | measured columnar win on real EVE order data: row CSV 136.8 B/row → naive per-column zstd-10 = **12.29 B/row** [V]. (Revision 1 said 36–51 GB/yr from the low event count.) |
| **Client read fan-out at a hub** (named as binding in revision 1, never numbered) | 2,000 concurrent clients × 10 watched items × top-10 levels × ~8 B/level = **1.60 MB/s (13 Mbit/s) at a 1 Hz digest** — fine. At the existing **20 Hz** snapshot cadence it is **32 MB/s (256 Mbit/s) from ONE shard**, larger than the rest of that shard's entire egress. A full book page for 50 items ≈ 80 KB ⇒ 2,000 clients opening market UIs is a **~160 MB burst** | derived ⇒ `market_digest_hz` (default 0.2–1 Hz), `digest_levels`, `max_watched_items_per_session` are `EconomyTuning` fields, book pages ride a paged bulk-class carrier with per-session rate limiting, and **market data is seconds-stale by design** — which is also EVE's answer (a **300 s** cache TTL on region orders [V]) and what makes the Veloren gossip option coherent. |
| **Durable commit budget** ⚠ **rebuilt from redb, not TigerBeetle** | Revision 1 multiplied 20 fsyncs/s by TigerBeetle's **8,189 events/request** — a *wire-protocol* constant of a system we are not using (its ~1 MB message frame ÷ 128 B record; exceeding it returns `ERR_TOO_MUCH_DATA`). It bounds neither redb's commit size nor ours. **Re-derived from redb 2.6.3's own published benchmark** (constants read locally: `BULK_ELEMENTS` 1,000,000, `INDIVIDUAL_WRITES` 1,000, `BATCH_WRITES` 100, `BATCH_SIZE` 1,000, `KEY_SIZE` 24, `VALUE_SIZE` 150 ⇒ 174 B records; README timings on a Ryzen 5900X + Samsung 980 PRO NVMe) [V]: individual committed writes 1,000/226 ms = **4,400 commits/s** (0.23 ms/commit); batch 100,000 puts/2,522 ms = **39,700 puts/s** (25.2 ms per 1,000-put commit); bulk 1,000,000/2,689 ms = **372,000 puts/s** in one transaction. A commit that must complete inside a 50 ms tick window therefore holds **~2,000 puts ⇒ ~40,000 durable postings/s** — and that is the **WHOLE shard's** durable write budget, shared with the directory, the saga WAL, `applied_steps`, block edits and checkpoints. **Economy realistically gets ⅓ ⇒ ~13,000 postings/s.** On k3d `local-path` (a container-mediated volume, 1–10 ms fsync) another 2–5× off ⇒ **~3,000–7,000/s** | ⇒ the real ceiling is **~12–50× BELOW** revision 1's 1.6×10⁵. Against EVE's ~2.9 k/s DB average that is **4–14×**, not ~50× — and even that comparison is apples-to-oranges, because one EVE SQL transaction may contain many postings. ⚠ These figures are redb's *published* benchmark on *their* hardware; the **only** number that should ever enter a design doc is a measurement of our own `RedbStore` (E-(−1), §10.5). |

**The four conclusions that shape the design:**

1. **Do not optimise matching.** A single deterministic single-threaded matcher has ≥3 orders of
   magnitude of headroom over EVE-scale flow even after the cold-cache correction. Every argument for a
   lock-free book, a flat tick ladder, or a concurrent engine is buying performance we provably do not
   need at the cost of determinism.
2. **Optimise for cardinality, for the write path, and for the read fan-out.** The binding costs are
   the *number* of markets, the **clearing sweep** (which scales with cardinality, not flow — hence the
   dirty set), the per-order-command saga/idempotency cost, order-**modification** churn, the
   read-path fan-out to clients (**1.6 MB/s per hub at 1 Hz; 32 MB/s at 20 Hz**), and a **0.5–2.4 TB/yr**
   durable ledger.
3. **Analytics at genuine EVE scale is still a SINGLE-NODE problem — and this conclusion is robust to
   the 10–50× volume correction.** ClickHouse single-node ingest is verified in the wild at ~4 M rows/s
   (65 billion rows / ~14 TiB on one node) [L], so 116–460 rows/s sits ~4 orders of magnitude below one
   node. Do **not** build a distributed warehouse or a Kafka tier. ⚠ **But revision 1's "no rollup
   pyramid" does NOT survive**: a full-year scan over 180 GB of Parquet at 1–2 GB/s effective is
   **90–180 s**, which is not an interactive galaxy dashboard. Build **exactly ONE materialized daily
   fact table, and treat it as MANDATORY rather than optional** (§8.3).
4. **Every ceiling above must be restated per shard before it is actionable.** A global orders/s or
   GB/yr figure is the one framing HR1 makes meaningless. §2.3 does that.

### 2.3 Per-shard budgets and the economy's cadence hierarchy

Two things revision 1 omitted entirely, each of which invalidated several of its own numbers.

**(a) The per-shard budget.** Under node-per-realm the only actionable targets are per shard, with a
stated fan-out factor. Proposed `EconomyTuning` budget fields (all named, none inline):

| Per-shard budget | Proposed default | Bound by |
|---|---|---|
| `postings_per_tick_budget` | **150** (≈3,000/s) | the corrected commit budget ÷ 3 ÷ fan-out; the whole-shard ceiling is ~2,000 puts/commit |
| `orders_per_tick_budget` | **500** | matching is free; this bounds journal + escrow postings, not CPU |
| `max_books_per_venue` | **2,000** | resident memory (~200 B empty, ~60 B/order) |
| `max_orders_per_book` | **50,000** | ⚠ **missing from revision 1**, which capped only `max_orders_per_actor` (EVE: 305). 305 × 5,000 actors on ONE commodity is a **1.5 M-order single-writer book** (150–300 MB resident, 1.5–3 s to rehydrate). This is the cap the RLM capacity formula needs. |
| `agent_eval_budget_per_tick` | **2,000** | see (c) below; the phase-decimation factor is **derived from this**, not asserted |
| `market_digest_hz` / `digest_levels` / `max_watched_items_per_session` | 0.5 Hz / 10 / 20 | the client read fan-out row |
| `journal_bytes_per_realm_budget` | **192 MiB** | must be ≤ the shard PVC minus the directory/WAL/checkpoint share |
| Clients per hub realm | **2,000** | egress: 1.6 MB/s digest + page bursts |

**Fan-out factor:** at 10⁴ live realms and 300–600 postings/s universe-wide, the *mean* shard sees
**0.03–0.06 postings/s** — three to four orders of magnitude under its own budget. The budget exists for
the **hub**, which by EVE's measured concentration carries ~2.8× its order share and ~6.5× the
per-order value of the rest of the universe. Hub postings: ~200–260 k trades/day in the hottest region
⇒ **2.3–3.0 trades/s mean ⇒ 12–24 postings/s**, and ~2.5 M postings/day ⇒ **29/s**. Even at 100× peak
that is ~300 trades/s — **three orders of magnitude below the corrected per-shard write ceiling.**

**(b) The cadence hierarchy.** Revision 1 silently ran the economy at the sim's 20 Hz everywhere
("per-tick matching budget at 20 Hz", "20 fsyncs/s", "51.8 M ticks for 30 days", "catch up by N cheap
ticks on wake"). **Nothing in an economy needs 50 ms resolution** — Veloren's own economy ticks at 90
simulated *days* [V] — and this single omission is what made four other numbers wrong (the clearing
sweep, the ledger volume, the dormancy catch-up, and the counterfactual twin). The cadence is also
baked into the meaning of every event record, so it belongs in the S-list, not in a later slice:

| `EconomyTuning` field | Proposed default | What runs at it |
|---|---|---|
| `settlement_period_ticks` | **1** (20 Hz) | ledger postings only — money must be immediate |
| `clearing_period_ticks` | **100** (5 s) | the batch auction, over the dirty set only |
| `agent_decision_period_ticks` | **400** (20 s) | NPC quotes, phase-spread by `f(seed, agent_id)` |
| `ambient_reprice_period_ticks` | **12,000** (10 min) | closed-form ambient prices + the depletion index |
| `dormant_catchup_tick_period` | **72,000** (1 universe-hour) | the coarse tier; 30 days = **720 ticks**, i.e. microseconds |
| `max_catchup_ticks` | **2,000** | hard bound with a defined fallback: exceed it and the realm adopts the pure closed-form field rather than replaying |

**(c) The agent load, carried into the placement decision.** §4.2's population figure is 5,000 noise
agents × 10³ active markets = **5×10⁶ evaluations per decision round**. At ~200–400 ns per evaluation
(one `DetRng` draw + bounds + a `BTreeMap` insert) that is **1.0–2.0 CORE-SECONDS per round**:

- **Distributed one-market-per-realm-shard**: 1.5 ms per 20 s round per shard = **0.0075% of a core** — free.
- **Concentrated on one never-dormant economy shard** (revision 1's D2 recommendation): **1–2 full
  cores of agent work on a single-threaded sequencer** that must also sequence every ledger op
  universe-wide ⇒ **>100% of one core before any player traffic.**
- **The only alternative under that topology** — agents tick in realm shards, books live on the economy
  shard — makes every agent quote a cross-shard side-effecting idempotency-keyed journaled command:
  5×10⁶ msg/round ≈ **250 k msg/s ⇒ ~25 MB/s of `InterShardFlow` egress** at a 20 s round, and 500 MB/s
  at a 1 s round. That is the same objection §8.1 uses to keep the event log *off* the reviewed
  taxonomy.

**Both branches of the concentrated topology are infeasible. Therefore: books MUST be co-located with
the agents that quote into them** — which is D2 option **A**, not B. This is the constraint that
inverted the D2 recommendation in revision 2 (§6.3/§11).

---

## 3. The reuse verdict

Verdicts: **ADOPT** = take as a dependency (all still require the user's explicit sign-off — no dep is
adopted unilaterally). **PORT** = re-derive in our own Tier-A code from a written spec, so we own 100%
region+branch coverage. **REF** = read for design only. **REJECT** = do not use, reason recorded so
the search is not repeated.

### 3.1 Money type

| Name | Kind | License | Maintained? | Verdict | One-line why |
|---|---|---|---|---|---|
| `Money(i128, CurrencyId)` newtype (ours) | pattern | n/a | n/a | **ADOPT** | ⚠ **CORRECTION**: revision 1 said "fixed 16-byte primitive ⇒ canonical postcard bytes". postcard v1 encodes `i128` as a **zigzag varint**, not 16 fixed bytes, so the premise is wrong — but the *conclusion* holds and is still the right reason to prefer the newtype over `rust_decimal`: **a single canonical varint encoding per value, with no trailing-zero / scale degrees of freedom.** Pin it with a golden byte test for `Money` in the wire conformance suite. Plus exact integer arithmetic, one monomorphic overflow/rounding site, zero deps, near-zero HR5 cost. `i128` at 2 dp gives ~1.7×10³⁶ headroom vs EVE's ~7.8×10¹⁴/month; `i64` at 2 dp caps at 9.2×10¹⁶ and is already marginal for galaxy aggregates [V]. |
| `rust_decimal` 1.42.1 | crate | MIT | Very active (2026-06-12, 122.9 M lifetime dl) [V] | **REF** (Tier-B display only) | Genuinely deterministic (96-bit integer mantissa + scale, no float), but it **preserves trailing zeros**, so `1.0` and `1.00` are equal yet encode to different bytes — which makes our byte-identity gates (G-IDENTICAL) representation-sensitive and would embed a third-party serde impl inside the FROZEN `vd-wire` contract. Its `checked_*` API also multiplies error arms at every call site under per-monomorphisation region counting. |
| `fixed` 1.31.0 | crate | MIT/Apache-2.0 | **Active and STABLE — 1.31.0 published 2026-03-20, 12.89 M lifetime / 2.38 M recent dl** [V] | **REF** (rates only, if needed) | ⚠ **CORRECTION**: the research variously called this "a 2024 alpha, ~2 years old" (R3), "still pre-1.0" (R2), and "2.0.0-alpha.28.0 (2026-03-20)" (R1) — **all three are wrong**. 2.0.0-alpha.28.0 is a parallel alpha line from 2024-07-25; the release line is 1.x and current. The maintenance half of the reject rationale is void. The surviving argument is radix: it provides **binary** fixed-point and explicitly does not provide decimal fixed-point, so base-2 cannot represent 0.01 — **wrong for money**, acceptable for *rates* if confined to a few monomorphic newtypes. And it is needed at all only if integer basis-points/ppm in `i128` proves insufficient. Deps (`az`, `bytemuck`, `half`, `typenum`) are genuinely `no_std`-shaped [V]. |
| `rusty-money` 0.5.0 | crate | MIT | Active (2026-01-14, 839 k lifetime / 123.5 k recent dl) [V] | **PORT** (`allocate` only) | Its largest-remainder `allocate` is the one algorithm every ledger needs and `rust_decimal` does not provide: splitting 100 minor units 3 ways MUST be 33/33/34 or the double-entry invariant fails at commit. Also telling: it ships both a decimal `Money` and a 64-bit-integer `FastMoney` — independently reaching the integer-minor-unit design for the fast path. API details **[U]** (not opened). |
| `iso_currency` 0.5.3 | crate | MIT | Stable/complete (2025-03-16, 27 KB) [V] | **REF** | Copy the per-currency subunit-**exponent** model verbatim; reject the crate (we do not want real ISO-4217 currencies, and it pulls `iso_country`). |
| `bigdecimal` 0.4.10 | crate | MIT/Apache-2.0 | Active (2025-12-27) [V] | **REJECT** | Heap-allocates per value with unbounded scale — allocation in a sim tick plus a non-canonical representation. Possible Tier-B offline use only. |

### 3.2 Ledger

| Name | Kind | License | Maintained? | Verdict | One-line why |
|---|---|---|---|---|---|
| TigerBeetle **model** (schema + two-phase semantics) | oss-repo/docs | Apache-2.0 | Very active [V] | **REF — highest-value item in the whole survey** | 128-byte `Account` with `debits_pending`/`debits_posted`/`credits_pending`/`credits_posted` and **no stored signed balance** (balance is computed); `flags.pending` reserves, resolution is a NEW immutable transfer carrying `pending_id` with `post_pending_transfer` (partial allowed, remainder returned) or `void_pending_transfer`, plus an auto-voiding `timeout`; validation is **pessimistic** at reserve time so **posting can never fail**; `ledger:u32` partitions which accounts may transact and cross-ledger transfers are prohibited, so FX is two atomically **linked** same-ledger transfers through liquidity accounts with the spread as an explicit third entry [V]. This maps 1:1 onto our saga (reserve ≈ `Frozen`/`FlushSource`, post ≈ the directory CAS, void ≈ `ThawSource`, timeout ≈ `scan_deadlines`, "resolves at most once" ≈ `IdempotencyKey::TransferStep` in `applied_steps`) [R]. |
| `tigerbeetle-client` 0.16.68 | crate | Apache-2.0 | 2026-04-16 — but **only 2 published versions ever, 41 total downloads, 6.28 MB crate** [V] | **OPTION WITH HARD BLOCKERS** (re-graded down from the research's "adopt-candidate") | ⚠ **CORRECTIONS**: the research said "26 MB crate" (it is 6.28 MB) and graded it adopt-candidate without stating the blockers. Real blockers: an external replicated Zig cluster (3–6 replicas) to operate; "Linux >= 5.6 is the only production environment we support"; tokio-async FFI; and architecturally a **second commit authority**, which contradicts "the directory CAS is the only commit point" unless exactly ONE economy-authority shard owns the client. Batch limit is **8,189 events/request** per the docs' request table (the performance page says 8,190) [V]. Jepsen: strong serializability met from 0.16.26, but a single-node failure raised minimum latency from sub-ms to **≥10 s** until 0.16.43, and clients retry indefinitely (issue #206 unresolved) [V]. **The "42,000 TPS measured / 1 M TPS design target / 15,000 for batched PostgreSQL" figures appear nowhere in current TigerBeetle docs or README [V] — treat as [U] and do not put them in a design doc.** |
| Our own ledger kernel over redb + the saga WAL | pattern | n/a | n/a | **ADOPT** | We already own the hard parts: redb 2, an append-only saga WAL, `(TransferId, step_id)` idempotency journaled in `applied_steps` with consult-before-effect/record-after-effect, a durable outbox with reconnect replay, and the directory fence CAS as the sole commit point [R]. Missing is only accounts + entries + the invariant: ~1,000–1,500 lines of pure Tier-A code. |
| `cala-ledger` 0.18.5 | crate | Apache-2.0 | Very active (2026-07-25, 144 k dl) [V] | **REJECT** (read it though) | Hard-requires sqlx 0.8.3 + postgres + tokio ^1.52 + rust_decimal ⇒ a second durability authority and a second crash matrix beside redb. **Worth reading**: it carries a **CEL expression interpreter** (`cala-cel-interpreter`) [V] — direct precedent for Eco-style "laws as data". |
| `sqlx-ledger` | crate | **[U]** (not fetched) | **[U]** | **REJECT** | Same SQL-bound lineage as cala-ledger. Listed for landscape completeness only. |
| `accounting-core` 0.1.1 | crate | MIT/Apache-2.0 | One release, 2025-08-27 [V] | **REJECT** | 2.5 k SLoC, bigdecimal money, async-trait storage in the domain trait, GST-specific tax logic. Only the 5-class chart of accounts is reusable, and that is textbook. |
| `beankeeper` 0.2.0 | crate | MIT/Apache-2.0 | 2026-03-16, **42 downloads** [V] | **REJECT** | Beancount-style personal accounting; far too immature for money authority. |
| `rustledger-core` 0.21.0 | crate | **GPL-3.0-only** [V] | Active (2026-07-11) | **REJECT — LICENSE HARD STOP** | ⚠ **CORRECTION**: the research said "license varies, verify before use". It is GPL-3.0-only, and the repo is GPL-3.0. For a proprietary game binary that ends the conversation. |
| `ousia-ledger` 2.0.2 | crate | MIT | 2026-07-25, created 2026-02-17, **470 downloads**, already a 1.x→2.x break [V] | **REF** | Advertises exactly the right shape (two-phase execution + value-object fragmentation) — worth *reading* for the API. Five months old with a breaking change already; not a foundation for player money, and a third-party crate in Tier-A would have to reach 100% region+branch on someone else's generics. |
| Event-sourcing / CQRS crates: `eventually` (0.4.0, **2020-10-04, dead**), `thalo` (**"Thalo is currently unmaintained"** verbatim), `cqrs-es` 0.5.0 (2025-12-30), `disintegrate` 4.0.0 (2026-02-02, PG-only, ~130 k SLoC), `evento` 2.0.0-alpha.25 (2026-07-13, **990 recent dl**, alpha at 96 releases) | crates | MIT/Apache-2.0 | as noted [V] | **REJECT all** | Each (a) needs async + an external SQL store, (b) **owns the transaction/commit boundary** — a direct collision with "the directory CAS is the only commit point", and (c) puts I/O inside the domain trait. We already implement the pattern (WAL + `applied_steps` + outbox + replay). Read `disintegrate`'s event-first "decision model" for the one shape where per-entity aggregates genuinely break down: a cross-shard corporation-wide credit limit. |

### 3.3 Order book / matching

| Name | Kind | License | Maintained? | Verdict | One-line why |
|---|---|---|---|---|---|
| `matchcore` 0.4.0 | crate | crates.io says MIT/Apache-2.0; **repo detects Apache-2.0 only — resolve before porting code** [V] | 2026-04-05 (created 2026-03-17); **158 downloads, 2 GitHub stars, 1 fork** [V] | **PORT (design)** | Self-describes verbatim as "a **single-threaded, deterministic, in-memory state machine**" citing LMAX; `BTreeMap<Price, LevelId>` + `Slab<PriceLevel>` + ring-buffer command/outcome flow; deps are **only** `rustc-hash`, `slab`, optional `serde` — no tokio, no time, no rng [V]. Architecturally the closest thing on crates.io to our house style. But 15 k SLoC with 2 stars and no coverage/soundness claims cannot be the authority over player money (a matching bug is directly exploitable for currency creation, i.e. a security issue). Port the design at ~600–1,200 SLoC. |
| `limitbook` 0.1.0 | crate | Apache-2.0 | One release 2025-08-08, 22 KB crate, 781 dl [V] | **PORT (structure)** | 565 SLoC (**[U]** — not measured, but consistent with 22 KB): BTreeMap price levels + VecDeque FIFO per level + HashMap id index. Its benchmarks are the numbers that settle the throughput debate. Substitutions needed: default-hasher `HashMap` → `DetHashMap`/BTreeMap, `rust_decimal` → integer minor units, `eyre` → `thiserror` [V on all three deps]. |
| `lanpishu6300/match-rust` | oss-repo | Apache-2.0 | 100 stars, last commit 2026-07-22 [V] | **REF — best external corroboration in the report** | README verbatim: "Golden NDJSON replay via `match-replay`", "Fixed-point `price_tick` / `qty_lot`", "Optional ART-style radix index (`--features art`)", "SPSC worker, cache-line padded ring", "100% branch gate (protocol/core/hp)" [V]. Someone shipping a production exchange independently arrived at our HR5 + determinism doctrine. Three transplantable ideas: integer tick/lot as the representation; golden-NDJSON replay as a `vd-tests` pattern; the radix ladder as an **opt-in** fast path, never the default. |
| `lobster` 0.7.0 | crate | ISC | **Last release 2020-10-26, last commit 2023-01-29**; 176 stars, 22.5 k dl; **zero normal dependencies** [V] | **REF / PORT** | The only credible Rust LOB with literally no deps, small enough (313 KB crate incl. tests) to read end-to-end. Note: it implements a continuous CLOB, which is *not* EVE's model. "~1.5 k lines, two f64 uses" **[U]**. |
| `orderbook-rs` 0.12.1 | crate | MIT | Very active (2026-07-23, 495 stars) [V] | **REJECT** | ⚠ Understated by the research: **tokio is a MANDATORY normal dep**, as are `serde_json`, `sha2`, `uuid` (v4/v5), `tracing`, `dashmap`, `crossbeam-skiplist` [V]. Lock-free by design ⇒ the matching *outcome* depends on thread interleaving ⇒ no byte-identical replay, no verified-oracle diff, no chaos-replay gate. Mine its order-type list as a requirements checklist. |
| `pricelevel` 0.9.1 | crate | MIT | Active (2026-07-14) [V] | **REJECT** | orderbook-rs's engine layer; same disqualifiers (crossbeam-skiplist, dashmap, sha2, ulid/uuid). **Best reference catalogue** for order types and TIF (limit/iceberg/post-only/trailing-stop/pegged/market-to-limit/reserve; GTC/IOC/FOK/GTD/Day) plus self-trade-prevention-by-identity and checksummed snapshots. |
| `nautilus-model` 0.60.0 | crate | **LGPL-3.0-or-later** [V] | Very active (2026-06-29, 24,987 stars) | **REJECT — LICENSE HARD STOP** | By far the most professionally engineered Rust trading domain model; static linking into a proprietary binary triggers the relinking obligation. Reading it for `Price`/`Quantity`/`Money` newtype discipline is fine. |
| `fluidex/dingir-exchange` | oss-repo | **NO LICENSE** [V] | 269 stars, 2026-06-12 | **REF (read-only)** | Its *persistence* strategy is the most directly relevant reference found: append operation-log + Redis-style fork-and-save snapshot — i.e. WAL + checkpoint applied to an order book, which is the RLM spin-down answer. No license ⇒ read only, never translate. |
| `dgtony/orderbook-rs` | oss-repo | **MIT** [V] | 453 stars, **dead since 2018-04-13**, 23 KB | **REF** | ⚠ **CORRECTION**: the research said "license not verified"; it is MIT, so it is legally re-usable. Value is pedagogical: the simplest complete price-time-priority implementation. |
| `llc-993/matching-core` (157★, **no license**), `petr-tik/dark_rusty_pool` (73★, **no license**), `crypto-zero/apex-engine` (19★, NOASSERTION), `me-imfhd/velocity` (68★, MIT, **ARCHIVED**), `gocronx/matcher` (83★ MIT), `philipgreat/lighting-match-engine-core` (110★ MIT), `amankrx/matching-engine-rs` (67★ MIT), `jmcph4/ironlobe` (17★ MIT), `mental32/exchange-orderbook` (29★ MIT), `fran0x/matchina` (21★ MIT) | oss-repos | as noted [V] | mixed | **REJECT / REF** | Recorded so the survey is provably exhaustive. Two unlicensed ⇒ read-only. `lighting-match-engine-core`'s "**8-nanosecond per Order Execution**" is its own README marketing with **no independent benchmark [U]**. `matching-engine-rs` targets the ITCH protocol (mild reference for a compact market-data feed). Collective value: the price-level design space is exactly three points — BTreeMap of levels (ordered, empty-cheap, snapshot-friendly), flat/radix tick ladder (O(1), memory-hungry per book), intrusive FIFO within levels. Given 10⁵–10⁶ sparse books, **BTreeMap+Slab is correct and the ladder is an opt-in exception**. |
| `ganitsutra/DoubleAuctions` + `suneel-sarswat/auction` (Coq) | oss-repos | **NO LICENSE on either** [V] | 2020-07-26 / 2025-08-08, 1 star each | **REF (oracle only)** | Machine-checked uniform-price and dynamic-price double auctions with **uniqueness** theorems, individual rationality, fairness, maximality (`produce_MM`/`produce_UM`), extracted to executable OCaml/Haskell explicitly to diff a real exchange against a verified program; the continuous follow-up is O(n log n) with a proven Ω(n log n) lower bound and clears 10 M orders in "a couple of minutes" (was "a few days") [V]. No license ⇒ never vendored. |

### 3.4 Agent simulation

| Name | Kind | License | Maintained? | Verdict | One-line why |
|---|---|---|---|---|---|
| Bristol Stock Exchange (`davecliff/BristolStockExchange`) | oss-repo | **MIT** (LICENSE.md verbatim, © 2012 Dave Cliff — GitHub's detector says NOASSERTION) [V] | 340 stars, 2025-04-13 | **PORT** | ⚠ **CORRECTION**: not "the entire zoo in one codebase". `BSE.py` contains **ZIC, ZIP, GVWY, SHVR, SNPR, PRZI, PRSH, PRDE**; **AA lives in a separate `Trader_AA.py`** and **GDX does not appear in `BSE.py` at all** [V]. Port order: ZIC → PRZI → GVWY/SHVR/SNPR → ZIP → GD. Porting also imports the published parameter values and the experimental protocol, which is how we evidence "believable market" as a regression test rather than an opinion. |
| ABIDES (`jpmorganchase/abides-jpmc-public`) | oss-repo | **BSD 3-Clause** (LICENSE verbatim, © 2021 J.P. Morgan Chase) [V] | **ARCHIVED, last push 2024-07-22** [V] | **PORT (config, not code)** | Supplies the agent taxonomy (exchange / noise / value / momentum / market-maker / execution) and calibrated mixes: RMSC03 = 1 exchange + 1 POV MM + 100 value + 25 momentum + **5,000 noise**; RMSC04 = 1 exchange + 2 MMs + 102 value + 12 momentum + 1,000 noise ⇒ **50:1 noise:value, ~1% market makers** [V]. These become fields in a `MarketPopulationTuning` struct, seed-derived per realm. ABIDES-Economist (arXiv:2402.09563) extends the same core to households/firms/central bank/government — the right structural reference for the policy layer. |
| Doran & Parberry, "Emergent Economies for Role Playing Games", **IJIGS Vol. 7, No. 3, 2012** | paper | open (PDF live, 1,977,252 B, first page verified verbatim) [V] | static | **PORT** | ⚠ **CORRECTION**: R1 cited "vol 7 no 1"; it is **No. 3**. Page range (35–47 vs 35–48) **[U]**. The only implementation-grade agent algorithm in the space: per-commodity price **belief intervals**, uniform draw inside the interval as the quote, size = ideal × favorability, shuffle-then-sort clearing matching top bid vs top ask, explicit narrow/widen/translate belief updates, and **bankrupt agents replaced by a copy of a currently-profitable role** (which IS the profession allocator). Two determinism hazards to design for, not discover: the mandatory book **shuffle** and the uniform sample both need a seam-provided `DetRng`, and the authors note chaotic sensitivity to initial conditions ⇒ replay tests must pin seed AND agent order. |
| `bazaarBot` | oss-repo | MIT | Haxe, 390 stars, dormant since 2020-06-04, 16 MB [V] | **REF (executable spec)** | The reference implementation of the paper; use it as an oracle to diff a fresh Rust port against on the same scenario, then delete. Ports exist in JS (`economia`) and Java. |
| ZI-C (Gode & Sunder, JPE 1993), ZIP (Cliff, HPL-97-91), GD (Gjerstad–Dickhaut, GEB 1998), GDX (Tesauro–Bredin 2002), AA (Vytelingum 2006/2008), PRZI/PRSH/PRDE (Cliff, arXiv:2103.11341) | papers | open | static | **PORT the cheap ones; REF the expensive ones** | The actionable finding is the *reversal*: Snashall & Cliff (arXiv:1910.09947) and Rollins & Cliff (arXiv:2009.06905) show AA's dominance does not survive realistically dynamic or asynchronous markets, and Cliff & Rollins (arXiv:2011.14346, "Methods Matter") find a **sub**-zero-intelligence strategy is *more* profitable than the published AI/ML traders once you run millions rather than thousands of sessions. So ship a cheap MIX (ZI-C + PRZI bulk, GVWY/SHVR/SNPR baselines, a minority of ZIP), gate GD on observed volume (O(H) per quote), and **reject GDX** (O(H·T) DP inside a tick). PRZI is the standout: **one** real control variable `s ∈ [−1,+1]` spans the whole strategy space (s=0 is exactly ZI-C), so per-NPC personality is `s = f(seed, npc_id)` — no magic numbers by construction. |
| `krABMaga` 0.6.2 | crate | MIT | Active (2026-06-19, 218 stars, 66 MB repo — exact) [V] | **REJECT** | ⚠ Worse than described: it depends on **rayon, ahash (randomly seeded by default), getrandom, lazy_static, num_cpus, plotters, chrono, crossterm** [V] — ambient entropy plus global mutable state plus parallel scheduling, with no documented determinism guarantee, inside a shard that already runs `bevy_ecs` 0.18 on a deterministic tick. Its **reproducibility macro** (run twice from one seed, assert identical) and its parameter-sweep/model-exploration tooling are good design references for a Tier-B calibration harness. |
| `big-brain` 0.22.0 | crate | Apache-2.0 | **Repository is ARCHIVED (read-only), final push 2025-10-07**; pinned to **bevy ^0.15** [V] | **REJECT** | ⚠ **CORRECTION**: the research said "effectively stalled"; it is archived upstream. Two majors behind our bevy 0.18, and adopting it would gate engine upgrades. The payload is ~200 lines anyway (Scorer/Action/Thinker utility scoring over our own `bevy_ecs`). |
| `dogoap` / `bevy_dogoap` 0.5.0 | crates | MIT | 2025-05-21; core depends on **bevy_reflect ^0.16** + `pathfinding` [V] | **REJECT** | Not framework-free despite appearances (same bevy version-skew problem), and GOAP is the wrong algorithm class: economic agents want price-response/reservation-price rules, not symbolic multi-step planners. Keep in reserve for station/logistics NPCs that genuinely need plans. |
| `bourse-book` / `bourse-de` 0.4.0 | crates | MIT | 2024-03-28, 13,819 B crate (exact), 10.4 k lifetime / 69 recent dl [V] | **PORT (subset) / REF** | A *simulated* LOB + discrete-event market simulator built for ABM and RL — biased toward determinism and reproducibility rather than wall-clock trading, which is the right bias. ⚠ **CORRECTION**: normal deps are `serde` + `serde_json` **only**; `rand` and `rand_xoshiro` are **dev-deps** [V] (the research listed `serde_with`, which is not a dependency). Its purity/`no_std` status is **[U]** — verify by reading before deciding. |
| MMOAgent (Xu et al., KDD 2025) | paper | code at Zenodo, license **[U]** | 2025 | **REJECT (for the sim)** | LLM-driven agents over a NetEase MMO: 6 resources × 5 activity types, 10-agent and 30-agent runs over 200 steps [V]. LLM agents are non-deterministic, unbounded-latency and network-dependent — incompatible with every determinism/io-seam/coverage rule. Two useful things: the resource/activity **taxonomy** as a completeness check on our own economic action set, and the demonstration that generative ABM works at ~3 orders of magnitude below live-game scale, which is itself the argument for cheap ZI/PRZI populations. |

### 3.5 Solver / equilibrium

| Name | Kind | License | Maintained? | Verdict | One-line why |
|---|---|---|---|---|---|
| Leontief over an **acyclic** BOM DAG (reverse-topological sweep) | algorithm | n/a | n/a | **ADOPT** | The highest-leverage simplification available. If the recipe graph is validated acyclic at registry load (the same shape as the existing `ShardProfile::build` capability-lattice validation [R]), then `A` is nilpotent under topological order and the whole `(I−A)⁻¹d` requirement/cost computation is one integer sweep in O(edges) — **no matrix inversion, no float, no dependency, Tier-A and fully coverable**. It is also where the VAT input-credit base is computed, so acyclicity buys the tax layer its stage-count invariance too. Cyclic recipes (fuel to make fuel) must be an explicitly ledgered deferral, not an accident. |
| Spatial price equilibrium as min-cost flow (Samuelson 1952; Takayama–Judge, AJAE 46(1):67–93, 1964) | algorithm | n/a | n/a | **PORT** | Goods flow i→j only when `p_j − p_i ≥ c_ij`; in equilibrium no price gap exceeds transport cost. **With integer costs, successive-shortest-path with Johnson potentials / cost-scaling are exactly deterministic**, so unlike tâtonnement this one CAN live in the deterministic tier. In our HR1 world it is *gameplay*: there is no global book, so the equilibrium is enforced by players flying cargo, and the min-cost-flow solution is simultaneously the NPC hauler plan and the dashboard's ground truth for "where is the arbitrage". Tariffs enter as an additive integer edge cost on `c_ij`. ⚠ **CORRECTION**: revision 1 said "~250 lines over petgraph". **`petgraph` is NOT a dependency of this workspace** — it appears in no `Cargo.toml` [V, grepped]; the "already approved" belief is a recollection from the *old* project's library evaluation, and the standing no-unilateral-dep law applies. Route min-cost flow to **hand-written integer SSP-with-potentials inside `vd-econ`** (~250 lines), which the report already argues is strictly better for determinism and avoids putting a generic-heavy graph crate inside a Tier-A 100%-region+branch crate. Keep `petgraph` strictly in the **Tier-B analytics** crate if the user adopts it at all. |
| Proportional response dynamics (Zhang, TCS 2010) | algorithm | n/a | n/a | **ADOPT (Tier-B advisory)** | Each buyer splits its budget in proportion to the utility each good delivered last period; converges to Fisher-market equilibrium for CES utilities with **no agent solving any optimisation problem**, and provably robust to asynchronous/adversarial update order (arXiv:2307.04108) — exactly our situation (realms tick independently, some are killed). Per iteration = one sparse pass. Float ⇒ advisory only: emit a reference price vector, quantise to integer ticks, inject as NPC quote centres. |
| Tâtonnement with modern rates (Codenotti et al. 2005; Cole–Fleischer 2008; entropic ≡ gradient descent under KL) | papers | n/a | n/a | **REF** | The fallback when provable rates over a Leontief production structure are needed (O(1/ε) for Leontief Fisher markets). Float; must have an iteration cap and a documented non-convergence behaviour (hold last prices, flag the market) — never an unbounded loop in a tick. |
| `good_lp` 1.15.2 | crate | MIT | Active (2026-05-31) [V] | **REF, and only with `default-features = false`** | ⚠ **TWO CORRECTIONS.** (a) The default-feature trap is REAL and verified: `features.default = ["coin_cbc", "singlethread-cbc"]`, and the README says cbc "requires to have the cbc C library headers available on the build machine" [V]. (b) The **license attribution in the research is wrong**: `coin_cbc` the crate is **MIT** and COIN-OR CBC itself is **EPL**, not LGPL; the actually-LGPL backend is **lpsolve** (good_lp README: "lp_solve is a free (LGPL) linear (integer) programming solver"), and SCIP is reached via `russcip` which is **Apache-2.0** (SCIP relicensed at v8) [V]. R4 graded this adopt-candidate *without* the `default-features = false` condition — unsafe as written. |
| `microlp` 0.5.0 | crate | Apache-2.0 | Active (2026-07-17) [V] | **REF (Tier-B offline, pinned)** | The most defensible LP backend: pure Rust, MILP via branch & bound, warm starts. Determinism hazard is explicit rather than incidental — it depends on **`web-time`** and advertises *time limits with resumable solving*, so output can depend on wall clock [V]. Also: it has an **optional `highs` dep** (so "pure Rust" holds only with that feature off) and it pulls **`sprs` transitively**; `good_lp` 1.15.2 still pins `microlp ^0.4.0` while microlp is at 0.5.0 [V]. |
| `clarabel` 0.11.1 | crate | Apache-2.0 | 2025-06-11 [V] | **REF (offline analysis)** | Interior-point conic solver, the only good_lp backend exposing constraint **duals** — shadow prices are exactly the "analysis" output an operator dashboard wants. But optional MKL/OpenBLAS/Netlib backends mean different SIMD kernels per CPU/build, no determinism statement anywhere, and interior-point returns a tolerance-dependent answer by construction. |
| `highs` 2.4.0 / `highs-sys` | crates | MIT (wrapper and HiGHS) [V] | Active (2026-07-09) | **REJECT** | HiGHS "fully leverage[s] all available processor cores" ⇒ results depend on thread count and parallel reduction order, so two runs on the **same** machine can differ — the worst possible determinism property, and non-obvious. Plus a C++ toolchain in the build (a real cost given this repo's documented 15-minute-build history). If a heavyweight MILP is ever needed, run it out-of-process with a pinned thread count. |
| `argmin` 0.11.0 | crate | MIT/Apache-2.0 | Active (2025-09-28) [V] | **REF** | Right tool for offline **calibration** (fitting production coefficients, demand elasticities, faucet/sink rates to target statistics). Pure Rust, no toolchain cost. Deterministic only with a seeded rng and rayon off; Nelder-Mead/PSO/SA are stochastic by nature. Never in a tick. |
| `nalgebra-sparse` 0.12.0 | crate | Apache-2.0 | Active (2026-05-24, dimforge — same org as rapier) [V] | **REF** | COO/CSR/CSC + mat-vec, but **no direct sparse solver** (triangular only), and self-describes as "early but usable". Probably unnecessary: an acyclic BOM makes the Leontief solve a topological sweep. |
| `sprs` 0.11.4 | crate | MIT/Apache-2.0 **except the Cholesky feature** | 2025-11-04, 628 stars [V] | **REJECT (and assert the feature stays off)** | LICENSE TRAP verified verbatim in the README: "simple sparse Cholesky decomposition (**requires opting into an LGPL license**)" [V] — i.e. the one feature you would reach for to solve a Leontief system is LGPL-gated, and sprs arrives transitively with microlp. ⚠ **CORRECTION**: the research said it "depends on ndarray + rayon (+ alga)"; only **ndarray is mandatory** — `rayon`, `alga`, `num_cpus`, `approx`, `serde` are all optional [V], so sprs-via-microlp does not by itself bring rayon or alga. |
| `pathfinding` 4.15.0 | crate | MIT/Apache-2.0 | Very active (2026-03-10) [V] | **REF (possibly ADOPT for integer graphs)** | ⚠ **CORRECTION to its own lib.rs summary, confirmed**: there is **NO min-cost-flow module** — the directed module list is astar, bfs, count_paths, cycle_detection, dfs, dijkstra, **edmonds_karp (MAX flow)**, fringe, idastar, iddfs, SCC, topological_sort, yen [V]. So min-cost flow must be ported. What IS available and useful: Edmonds-Karp (trade-route capacity), Kuhn-Munkres (hauler↔contract assignment), Yen (alternative routes). Deps (`indexmap`, `num-traits`, `rustc-hash`, `deprecate-until`, `integer-sqrt`, `thiserror`) are all pure — the one solver-adjacent crate that might be legal in-tick on integer weights. |
| `mcmf` 2.0.0 | crate | MIT | **2018-08-30, Rust 2015, bundles LEMON C++ via a `gcc 0.3` build dep** [V] | **REJECT** | Unacceptable in this workspace on every axis. Hand-write SSP-with-potentials instead — strictly better for us because integer min-cost flow is exactly deterministic. |
| VCG / general combinatorial winner determination | algorithm | n/a | n/a | **REJECT (for shipped gameplay)** | NP-hard and inapproximable in general, tractable only on bounded-treewidth item graphs, and VCG multiplies the solve count. Incompatible with a bounded-time deterministic tick. Single-item **Vickrey** for contracts is O(n) and fine. |

### 3.6 Analytics stack

| Name | Kind | License | Maintained? | Verdict | One-line why |
|---|---|---|---|---|---|
| `arrow` 59.1.0 / `parquet` | crates | Apache-2.0 | Very active [V] | **ADOPT (Tier-B, offline tool only)** | The archive-of-record format. Note both a 59.1.0 (2026-07-07) and a 58.4.0 (2026-07-22) line exist [V] — easy to misread as a bad version. |
| `datafusion` 54.1.0 | crate | Apache-2.0 | Very active (2026-07-21, 23.35 M dl) [V] | **ADOPT (Tier-B, separate binary)** | Embeddable SQL/DataFrame over Arrow. **47 direct deps including tokio, object_store, parquet** — must not be a workspace member of the server build graph given this repo's build-perf history. |
| `polars` 0.54.4 / `duckdb` 1.10505.0 | crates | MIT / MIT | Active [V] | **ADOPT alternative** | polars = DataFrame-first, smaller; duckdb = richest SQL and cheapest to operate, at the cost of a C++ dep in a Tier-B binary. Pick ONE. |
| ClickHouse + `clickhouse` 0.15.1 Rust crate | service + crate | Apache-2.0 / MIT-Apache | Active [L] | **ADOPT (standard option)** | Single-node ingest is 3–4 orders of magnitude above our ~120 events/s [L]. Pure-Rust client (`Inserter`, RowBinary, zstd, rustls — no C++). |
| Grafana 13.0.0 + `grafana/clickhouse-datasource` | service | **AGPLv3** / Apache-2.0 | 2026-04-14 [L] | **ADOPT (standard option)** | Galaxy dashboards with zero bespoke UI work. AGPL is fine as a separate process; a fork or embed would be a licensing event. |
| Prometheus 3.11.2 / VictoriaMetrics | services | Apache-2.0 | Active [L] | **ADOPT (operational series)** | Scrape the `/metrics` we already serve; `econ_*` names are additive. |
| GreptimeDB | service | Apache-2.0 core (+ separate enterprise license) | v1.0 GA [L] | **ADOPT alternative** | Rust, SQL **and** PromQL, built on Arrow/DataFusion/Parquet/object-store — one engine for metrics *and* events. Architecturally elegant, considerably less battle-tested than ClickHouse at this workload. |
| `deltalake` 0.32.4 / `iceberg` 0.10.0 / `object_store` 0.14.1 | crates | Apache-2.0 | Active [L] | **REF (heavy option)** | Time-travel/ACID on the lake. Only worth it if counterfactual runs need reproducible as-of snapshots of the archive. |
| `tdigest` 1.0.0 / `sketches-ddsketch` 0.4.0 / `hyperloglogplus` 0.4.1 | crates | **[U]** | **[U]** | **ADOPT candidates (streaming stats)** | Percentiles (time-to-fill, spread) and distinct counts (active traders) without storing everything. Versions/licenses **[U]** — verify before deciding. |
| `augurs` 0.10.2 (Grafana's Rust time-series toolkit) / `linfa` 0.8.1 | crates | **[U]** | **[U]** | **REF (heavy option)** | Series outlier scoring and clustering, strictly advisory and human-gated. |
| EVE Monthly Economic Report raw dataset | service/data | CCP-published, **no explicit license** | Monthly since ~2016; June 2026 edition 65,072,093 B [V] | **REF (calibrate + specify; do NOT vendor)** | The requirements document for the dashboard half of the ask, already written by a company that has done it for 19 years, plus a real 19-year series to calibrate against. |
| ESI (EVE Swagger Interface) | service | public read API | Live, spec v1.36 (860,667 B, 180 paths) [V] | **REF** | A battle-tested API design for exposing an economy to external analysis: region-scoped resources, explicit per-resource cache TTLs, X-Pages pagination, an error-limit header contract, a closed 12-value `context_id_type` provenance enum. **Its one big design mistake is publishing snapshots instead of events** — copy the shape, invert that decision. |
| Albion Data Project (`ao-data/albiondata-client`) | oss-repo | MIT, Go, 182 stars, 2026-07-01 [V] | Active | **REF** | The proven community-ingest shape: a client extracts market data and ships it to a central server with a public API. ⚠ The "NATS bus" detail is a client-repo implementation detail not on the homepage — **[U]**. Lesson: ship the feed ourselves or third parties will sniff the wire. |

### 3.7 Dashboard / front-end / charting

| Name | License | Status | Verdict | Why |
|---|---|---|---|---|
| Existing `axum` + `serde_json` admin shell + `/metrics` registry | n/a | in-repo [R] | **ADOPT** | An additive `EconSnapshot` endpoint and `econ_*` metric names cost nothing and are immediately agent-operable (HR6). ⚠ But the `/admin/*` router is documented read-only **by construction**, loopback-only, with authentication still owed (D-13) [R] — the *mutating* tweak surface must not be bolted onto it. |
| `leptos` 0.8.20 / `dioxus` 0.7.9 | MIT/Apache-2.0 | Active [L] | **REF (heavy option)** | Only if a bespoke Rust front-end is wanted over Grafana. |
| `egui` 0.35 + `egui_plot` 0.36 | MIT/Apache-2.0 | Active [L] | **ADOPT candidate (desktop ops tool)** | Strongest Rust desktop story. ⚠ `bevy_egui` 0.39 in our tree pins `egui ^0.33` [L] — so an egui ops tool must be its **own binary**, not a shared version. |
| `charming` 0.6.0 (ECharts JSON builder) | **[U]** | **no release since 2025-06-17** [L] | **REF** | Thin and appealing, but stale. |
| `plotters` 0.3.7 | MIT | **no release since 2024-09-08** [L] | **REF** | Stale. |
| `plotly` 0.14.1 | MIT | 2026-02 [L] | **ADOPT candidate** | Current; the most viable Rust charting option if not using Grafana. |
| `perspective` 4.5.2 | Apache-2.0 | current, but the Rust crate has **~1.1 k recent downloads** [L] | **REJECT** | Adoption too thin for an ops-critical dependency. |

### 3.8 Detection / integrity

| Name | Kind | Verdict | Why |
|---|---|---|---|
| Deterministic invariants in-shard (conservation, non-negative balance, escrow closure, fence monotonicity) | pattern | **ADOPT — highest-value control by a wide margin** | Catches duping **the tick it happens**, not by a market analyst three weeks later. Siblings of the existing oracle family [R]. |
| Benford's law on trade amounts; wash-trade A→B→A within a window at off-band prices; self-trade; place/cancel ratio + time-to-cancel distribution (spoofing); robust-z/MAD price outliers vs regional median; HHI counterparty concentration; new-account inflow percentile (the RMT-buyer signature) | algorithms | **PORT (Tier-1 statistical batch)** | All cheap, all integer-expressible, all runnable over the event log offline. |
| `petgraph` — farmer→broker→buyer bipartite motifs, wash rings as SCCs, low-diversity high-throughput nodes, assortativity, community detection | crate + algorithms | **ADOPT CANDIDATE (Tier-2 graph, Tier-B ONLY)** | ⚠ **CORRECTION**: revision 1 tagged this "already an approved dep". **It is not a dep of this workspace at all** [V, grepped every `Cargo.toml`] — the belief comes from the old project's evaluation. It remains a **user decision** per the no-unilateral-dep law, and it may live only in the Tier-B analytics crate (a generic-heavy graph crate inside Tier-A is an HR5 cost we would pay per monomorphisation). The published gold-farming literature is explicit that the winning features come from the **trading graph** plus connection patterns; we can additionally add session/gateway co-occurrence edges that most games cannot. |
| extended-isolation-forest; `augurs`; `linfa` | crates | **PORT/REF (Tier-3, advisory only)** | Human-gated, and **every model score written back into the log as an event** so enforcement decisions are themselves auditable. |
| CME Self-Match Prevention (SMP ID; cancel aggressor or resting on match) | pattern | **ADOPT** | Not optional in a game: without it an alt-pair or a corporation trivially wash-trades to manufacture a price history that moves NPC/AMM anchors, farm volume rewards, and launder value between accounts while looking like normal trade. Key it on the owning **legal entity** (character → corporation → alliance), resolvable from our realm/ownership registry. |
| Hogan-Hennessy, Xenopoulos & Silva, "Market Interventions in a Large-Scale Virtual Economy" (arXiv:2210.07970) | paper | **REF — the most important caution in the report** | Causal study of two real OSRS interventions: the transaction tax had **minimal** impact on trading at the taxed price points, and the item sink **paradoxically raised** luxury prices without cutting volume. Taxes and sinks are **weaker levers than designers assume** ⇒ the dashboard must be able to *measure* an intervention, not just apply it. |

---

## 4. The algorithm catalogue

Columns: **Cost** = asymptotic + practical; **Det.** = may it live in the deterministic authoritative
tick? (**YES** = integer/pure; **NO** = float/thread/clock-dependent ⇒ Tier-B advisory only);
**Params** = what goes in the ONE `EconomyTuning` config struct (no inline literals).

### 4.1 Matching

| Algorithm | Cost | Det. | Params | Port note |
|---|---|---|---|---|
| **Price-time-priority CDA** — ordered price-level container + intrusive FIFO per level + id→handle map. Add at a NEW level O(log M), add at an existing level O(1), cancel O(1), execute O(1) at the level front. | 200–300 ns/order measured; 3–5 M/s single-core | **YES** with `i64` ticks/qty | `tick_size`, `lot_size`, `max_levels`, `max_orders_per_account`, `order_ttl_ticks` | The *outcome* depends on arrival order ⇒ exactly ONE authority per book, no replication, no cross-check against a verified oracle. `BTreeMap` is what our conventions already force (default-hasher `HashMap` is clippy-banned in sim/node [R]). |
| **Frequent batch / uniform-price call auction** — accumulate over N ticks, clear at the single price maximising executed volume, **under the FULL total order in the note** | O(n log n) per clearing (or O(levels) if the book is already ordered), **over the DIRTY SET only** | **YES**, and *better*: the outcome is a pure function of an unordered order **SET** — but only once the tie-break is total (see the ⚠ below) | `clearing_period_ticks`, `end_time_jitter_ticks` (seed-derived per (realm,item,tick), à la Xetra's randomised auction end), `price_band_bp`, `interruption_extension_ticks` | Budish–Cramton–Shim (QJE 130(4):1547–1621, 2015) show the CLOB's serial processing creates mechanical arbitrage rents from *symmetrically observed* public information, driving a wasteful speed race; the fix is uniform-price auctions at frequent discrete intervals. For us: latency-fair (we have NO client prediction and a 100–150 ms buffer [R]) and replayable. ⚠ **Two corrections to revision 1, both load-bearing on the D1 recommendation.** (1) **The Coq claim is downgraded.** The verified results give uniqueness of the *maximum matching / traded volume*, **not of the clearing PRICE** — a volume-maximising uniform price is in general a non-degenerate *interval*. So "safe to cross-check across replicas" is true for volume and for our own fills under a fixed rule, but the oracle does not hand us price uniqueness. (2) **Name the FULL total order or the set-purity claim is false**, because revision 1 stopped at "minimise imbalance, then proximity to a reference price" with no tie-break below that: <br>• **clearing price = the LOWEST price in the volume-maximising interval** (after the imbalance and reference-proximity filters); <br>• **marginal allocation = pro-rata with largest-remainder** (§4.5's Hamilton routine, one rounding point); <br>• **remainder ties broken by `(price, OrderId)` ascending** — never by insertion order, never by a hash, never by arrival time (time priority would re-introduce arrival order and destroy the whole determinism argument). <br>**Pin with a proptest that permutes the input order set and asserts byte-identical fills.** <br>⚠ **(3) The dirty set is mandatory.** A batch auction visits every book every period, so the cost scales with **cardinality**, not order flow: 10⁶ books at a 5 s period is 200,000 visits/s ≈ 0.20 cores of clearing empty books (§2.2). Maintain a per-realm intrusive dirty set of books with ≥1 new/modified/expired order since the last batch and clear only those; **a book absent from the dirty set must be provably unchanged** (an invariant, not a comment). Lazy instantiation fixes memory; only the dirty set fixes the sweep. One clearing pass per N ticks over the dirty set is what makes 10⁵–10⁶ markets affordable. |
| **EVE-style event-driven matching with arrival precedence** — the broker attempts a match only at order *creation or modification*; among reachable counterparties the best price wins, and **whoever arrived first gets the better side of the spread** (sell-first ⇒ trade at the MAX satisfying price; buy-first ⇒ the MIN). | O(1) amortised per command | **YES** | as CDA, plus the precedence rule | Much cheaper than a CLOB: no resting-order crossing daemon, no continuous loop, no per-tick book walk. `apply_order(book, cmd) -> (book', Vec<Fill>)` is a pure Tier-A fn. Unusual semantics may confuse players who expect exchange behaviour. |
| **Order types + TIF** — limit/market/post-only; GTC/IOC/FOK/GTD; **iceberg** (exchange-managed displayed slice + hidden reserve; on refresh the displayed quantity is reinstated **at the BACK of the queue** at that price); **self-match prevention** (SMP ID on each order; on aggressor/resting match, cancel one side rather than crossing). | O(1) each | **YES** | `iceberg_refresh_to_back: bool`, `min_display_qty`, `stp_policy` | CME publishes exact specifications [V]. `pricelevel`'s taxonomy is the best available requirements checklist; a game probably wants GTC + IOC + expiry and can skip pegged/trailing. Iceberg makes large industrial players interesting rather than instantly front-run; STP is the cheap structural fix for wash trading. |
| **Single-item Vickrey (sealed-bid second-price)** for contracts | O(n) per contract, O(1) state | **YES** | `min_bid_increment`, `bid_window_ticks` | Dominant-strategy incentive-compatible, and it removes the "refresh the contract list every 200 ms" reflex skill — good for a WAN game. Right for hauling/mining/escort/manufacturing contracts (one-off, heterogeneous, thin, private-value). Do **not** extend to combinatorial bundles (NP-hard). |
| **Order expiry** — `expiry_universe_tick` on every resting order, evaluated as a deterministic tick predicate | O(expiring) per tick | **YES** | `max_order_duration_ticks` (EVE: 90 days) | Makes the book self-garbage-collecting with no ambient clock, and it is what stops a never-respawned realm holding escrowed value forever. |
| **Per-actor capacity cap** (EVE: hard 305 orders/character) **AND a per-book cap** | O(1) | **YES** | `max_orders_per_actor`, **`max_orders_per_book`** | Strictly better than a rate limit for us: it bounds **state**, is deterministic, and needs no clock ⇒ enforceable as a pure Tier-A invariant. Treat rate limiting as transport-layer only. ⚠ **`max_orders_per_book` was missing from revision 1** and it is the cap that actually bounds a single-writer book's memory and its rehydration latency: 305 × 5,000 admitted actors on ONE commodity is a 1.5 M-order book (150–300 MB resident, 1.5–3 s to rehydrate). The RLM capacity-planning formula is `per-shard resident = Σ_venues Σ_books min(max_orders_per_book, Σ actors × max_orders_per_actor) × ~60 B`, and the G4/I11 rehydrate gate must be sized at `max_orders_per_book`, not at an arbitrary 40 k. |
| **Order-modification fee + tick size** (the anti-churn pair) | O(1) | **YES** | `relist_fee_bp`, `relist_fee_min`, `price_significant_figures` | CCP's own account: the pre-2020 optimal strategy was "always create your orders at 0.01 ISK above/below the best, and always update ASAP" — a write-amplification DoS that they fixed **economically**, not technically: prices limited to 4 significant figures (so there is no 0.01 increment to undercut by) plus a relist charge `max(0, BR·(P2−P1)) + (1−RD)·BR·P2`, min 100 ISK [V]. Tick size is a *quantisation* rule we must follow anyway ⇒ we get the anti-spam property for free. Together with the order cap this gives three composable, deterministic, clock-free abuse controls: **cap the state, quantise the price lattice, price the mutation.** |

### 4.2 Trading agents

| Algorithm | Cost | Det. | Params | Port note |
|---|---|---|---|---|
| **ZI-C** (Gode & Sunder, JPE 1993) — uniform-random quote inside a budget constraint (never sell below cost, never buy above value) | O(1), one RNG draw | **YES** if the draw comes from `sim::io`'s seeded rng | only the value/cost distribution, which should be **seed-derived from the realm's resource profile** | Reaches ~100% allocative efficiency purely from market structure. The cheapest believable-liquidity agent ⇒ the **bulk** population. Cliff showed the headline result is partly artefactual, so ZI-C alone will not produce realistic price *dynamics*. |
| **PRZI / PRSH / PRDE** (Cliff, arXiv:2103.11341) — one real control variable `s ∈ [−1,+1]` parameterises the quote-price PMF; `s=0` is exactly ZI-C uniform, `\|s\|→1` collapses mass onto one end of the range; PRSH adapts `s` by stochastic hill-climbing, PRDE by differential evolution | O(1) per quote | **YES** with care: the inverse-CDF sample must be integer/rational or a fixed-point table, **never `powf`** | `s = f(seed, npc_id)` — a single seed-derived value per NPC | **Best value in the catalogue.** ONE parameter spans the entire strategy space, so per-NPC personality satisfies no-magic-numbers by construction and the whole family is one monomorphic function. Adaptive populations show stability over hundreds of thousands of interactions punctuated by regime change — exactly the "market feels alive while players sleep" property. |
| **GVWY / SHVR / SNPR** (giveaway / shaver / sniper) | O(1) | **YES** | none of substance | 5-line strategies that "Methods Matter" shows are competitive with the published AI/ML traders. Ship them as cheap bulk. |
| **ZIP** (Cliff, HPL-97-91, 1997) — Widrow-Hoff momentum learner over a profit margin, raised/lowered on observed trades | O(1) per quote | **YES** in fixed-point bp | ~6 params: learning rate, momentum, margin bounds, jitter → a named `ZipTuning` struct | A minority adaptive population that makes prices converge and respond to shocks. **Coverage warning**: a branch-dense adaptive loop ⇒ its own monomorphic module, never a generic. |
| **GD** (Gjerstad & Dickhaut, GEB 22:1–29, 1998) — belief function P(quote accepted at p) from frequencies of recent bids/asks/accepts; quote the expected-surplus maximiser | **O(H)** per quote in history window H | **YES** | `history_window`, `min_volume_to_enable` | Only worth it for DEEP books where the frequency estimates mean anything ⇒ **gate on observed volume**. |
| **GDX** (Tesauro & Bredin, AAMAS 2002) — dynamic programming over the remaining horizon | **O(H·T)** | **YES** but unbounded work | — | **REJECT**: an O(H·T) DP inside a tick for marginal benefit, and the later literature shows the sophisticated agents do not dominate under realistic dynamics. |
| **AA — Adaptive-Aggressive** (Vytelingum 2006; AIJ 2008) — an "aggressiveness" variable trading profit against fill probability relative to a believed competitive equilibrium | O(1)–O(H) | **YES** | aggressiveness bounds, learning rates | Beat ZIP/GD/GDX head-to-head and beat humans at IJCAI-2011 — **but** Snashall & Cliff (arXiv:1910.09947) and Rollins & Cliff (arXiv:2009.06905) show the dominance evaporates under realistically dynamic and asynchronous markets, and Cliff & Rollins (arXiv:2011.14346) find a **sub**-ZI strategy is more profitable once you run millions rather than thousands of sessions. Port only if a specific NPC archetype (a professional market-maker faction) must visibly outcompete players. **Methodological warning for us: any claim about our market's behaviour needs millions of seeded runs** — a load/perf-shaped task, pairing with the existing latency-gate pattern in `crates/harness/src/latency.rs` [R]. |
| **Doran–Parberry price-belief agents** — per-commodity belief interval `[low, high]`, uniform draw inside it as the quote, `quantity = ideal × favorability` (favorability = where the historical mean sits inside the observed trading range); shuffle-then-sort clearing matching top bid vs top ask; on ≥50% fill contract the interval inward by 1/10 of the upper limit, on rejection expand it by 1/10 and translate toward the previous round's average; bankrupt agents replaced by a copy of a currently-profitable role | O(agents log agents) per round | **YES** in integer bp — with **two named hazards**: the mandatory book **shuffle** and the interval sample both need `DetRng`, and the authors note chaotic sensitivity to initial conditions ⇒ replay tests pin seed AND agent order | `belief_narrow_bp`, `belief_widen_bp`, `belief_translate_bp`, `bankruptcy_threshold`, `favorability_curve`, `profitability_window` (8–15 rounds works well), `idle_fine` | The only widely-used game-economy algorithm that closes the loop from prices back to **production** and to the **distribution of professions** — bankruptcy-and-replacement IS the profession allocator, so NPC producers need no hand-authored spawn tables. It is a *periodic batch* clearing, structurally the same family as the frequent-batch auction ⇒ **one clearing engine serves both the player market and the NPC production economy (HR3)**. Deliberate deviation when porting: clear at the volume-maximising **uniform** price rather than the bid/ask midpoint, so we inherit the Coq uniqueness theorem. Belief-update arms are ~12 across two functions ⇒ apply the branchless-shim discipline from the first commit or this one feature dominates the Tier-A coverage budget. |
| **Agent population mix** (ABIDES) | — | — | `noise_agents_per_market`, `value_agents_per_market`, `momentum_share_bp`, `market_maker_count`, all seed-derived per realm | 50:1 noise:value and ~1% market makers is the empirically-tuned answer to "what mix produces a realistic price series cheaply" [V]. **Compute warning**: 5,000 noise agents × 10³ active markets = 5×10⁶ agent evaluations per decision round ⇒ decimate the decision cadence (each agent acts on a phase derived from `(seed, agent_id)`, not every tick) and AoI-scale the population, or the economy becomes the dominant tick cost. |
| **Producer-exit hysteresis** (the one transferable idea from Star Citizen) | O(1) | **YES** | `exit_margin_bp`, `exit_lag_ticks`, `subsidy_budget` | Zurovec's stated objection to naive rules: when price drops below profitability "in a simplistic rules based game engine, the production immediately ceases. In reality, that takes time, some people will continue to subsidise loss making operations" [V]. A producer must not shut down the instant margin goes negative. This is the **same anti-thrash hysteresis RLM already needed for spin-up/down** [R], and it is precisely the failure mode a naive Veloren-style labour reallocation exhibits (which Veloren patches with 0.8 smoothing and a per-industry floor — **smoothing constants [U]**, not located in `mod.rs`). |

### 4.3 Market makers / NPC liquidity

| Algorithm | Cost | Det. | Params | Port note |
|---|---|---|---|---|
| **Avellaneda–Stoikov** (Quantitative Finance 8(3):217–224, 2008) — CARA-utility HJB solution giving a reservation price shifted from mid by (inventory × risk aversion γ) plus an optimal spread; quotes lean against inventory | O(1) | **NO as written** (uses exp/ln) ⇒ either fixed-point log/exp with a documented rational approximation and property tests, or compute Tier-B and quantise the quotes to integer ticks | one free `gamma` per NPC (seed-derived per faction/station), `spread_floor_bp`, `max_inventory` | **Best model for an NPC trader**, precisely because inventory skew means the NPC's price MOVES as players drain it — it cannot be farmed indefinitely, unlike a fixed-price NPC order. Strongly preferred over LMSR for commodity markets. |
| **Constant-product AMM** (`x·y=k`; Angeris et al., arXiv:1911.03380 — price tracks a reference closely under no-arbitrage, with the bound **loosening as the fee rises**; stable across a wide range of conditions in large-scale agent simulation) | O(1), two integers of state | **YES** with `u128` muldiv and **rounding always in the pool's favour** | `fee_bp`, initial reserves (seed-derived) | Best fit as a **bootstrap market** for a freshly spun-up realm and for tiny/remote markets where a real book would be empty: minimal state, always quotes, and its slippage curve IS the price-impact mechanism. **Integer bug that becomes a faucet**: if rounding ever favours the trader the pool leaks value on every trade. **REJECT for equity** — a constant-product pool over a fixed-supply share class is a permanent free option for anyone who can move the underlying (the sandwich / pump-and-dump vector). |
| **LMSR / LS-LMSR** (Hanson; Othman, Pennock, Reeves & Sandholm, EC'10) — always-available liquidity with a proven bounded worst-case market-maker loss of `C(q⁰)`; LS-LMSR makes the liquidity parameter a function of outstanding shares so the spread responds to volume | O(assets) | **NO** (exp/ln) | `liquidity_b`, `max_subsidy_per_market` | **Prediction/insurance markets only** (bounty odds, sovereignty futures, cargo insurance) — never commodities: the bounded loss is a subsidy *we* pay, and LMSR's liquidity-**insensitivity** lets a large player move the implied probability cheaply, which is a direct exploit if anything in the game reads the implied price. Prefer LS-LMSR if used at all. |
| **Albion-style adaptive NPC bid** — an NPC that buys player-crafted goods and **raises its own bid when it lacks stock** | O(1) | **YES** | `bid_ramp_bp_per_period`, `bid_ceiling`, `budget_per_period` | The cold-start and realm-wake answer: guarantees every realm a counterparty and a price signal on day one, before any player trades exist. Fee rates for Albion's marketplace (2.5% setup on create *and* edit, 8%/4% sale tax) are **[U]** — the wiki is bot-blocked. |
| **Fixed-price infinite NPC order** | O(1) | YES | — | **REJECT — this is the single most dangerous pattern in the catalogue.** Any NPC *buy* price ≥ any NPC *sell* price across a craftable/refinable/repackable chain is an **unbounded money printer**, and it can appear from one mispriced item, one rounding direction, or one AMM invariant that rounds against the pool. If price anchors are wanted they must be **finite quantity with a config replenishment budget** — i.e. a bounded faucet/sink, never an infinite one. |

### 4.4 Equilibrium + production chains

| Algorithm | Cost | Det. | Params | Port note |
|---|---|---|---|---|
| **Victoria 3 clamped closed form** — `price = base·[1 + 0.75·clamp((BUY−SELL)/min(BUY,SELL), ±1)]`, hard-bounded 25–175% of base; `local = MAPI·market + (1−MAPI)·state`; shortage triggers at BUY ≥ 2×SELL (−5% throughput then −1%/day to a −75% floor, recovering 1%/day); **no stockpiles** — each tick is a snapshot wiped clean | O(goods × realms), one branch | **YES** in fixed-point/`i128` | `base_price` per good, the `0.75` amplitude, clamp bounds, `mapi_weight` | Paradox ships this at ~700 regions × up to 100 k population groups [V]. Ideal Tier-A: no allocation, no I/O, one branch with both arms trivially reachable ⇒ near-free 100% region+branch. `min(BUY,SELL)==0` must be a defined, tested arm. The MAPI blend is literally a parent-market/local-market interpolation ⇒ maps onto the realm tree with LCA propagation. **Zero cross-tick state** ⇒ nothing to restore when a realm wakes. |
| **X4 storage lerp** — `price = lerp(max_price, min_price, storage_fill_ratio)`; band ≈ ±15–30% around average per ware **[L]** | O(1) | **YES** | `min_price`, `max_price` per ware | Best possible RLM fit: **zero economy-specific state** (inventory is the only truth), so nothing extra to replicate, checkpoint, reconcile or restore on wake — a spun-down realm's prices are recoverable *exactly* from persisted inventory. Weakness: with no memory it cannot express expectation, hoarding or speculation, and prices snap immediately ⇒ pair with smoothing/hysteresis. |
| **Endless Sky AR(1) supply through a saturating curve** — `supply *= 0.89; supply += Normal()·2000; price = base + (int)(−100·erf(supply/20000))`; player/event impact enters through exactly one hook (`SetSupply`) | O(1), two floats of state per (commodity, realm) | **NO as written** (Normal + erf + f64) ⇒ (a) the deviate must come from the io-seam rng **or** be replaced by closed-form `f(seed, universe_tick)` per our celestial-math rule, and (b) erf must become a **quantised integer/fixed-point monotone curve** (lookup over an integer supply grid) | `keep`, `export`, `volume_sigma`, `limit` | Constants verified verbatim from `source/System.cpp` [V]. Naturally mean-reverting and bounded (erf saturates) ⇒ it cannot run away, which is the failure mode that kills naive supply/demand loops. **GPL-3.0 ⇒ re-derive the curve shape and constants independently.** |
| **Leontief input-output over an ACYCLIC BOM DAG** — `x = (I−A)⁻¹d`; if the recipe graph is acyclic, `A` is nilpotent under topological order ⇒ one reverse-topological sweep | **O(edges)**, integer, no matrix inversion | **YES** | the technical-coefficient matrix `A` = seed-derived world data, never literals | The single biggest simplification available (see §3.5). Also the site of the VAT input-credit computation. |
| **Proportional response dynamics** (Zhang, TCS 2010) | one sparse pass per iteration; no solver, no line search | **NO** (float) ⇒ Tier-B advisory: emit a reference price vector, quantise, inject as NPC quote centres | `iteration_budget_per_rollup`, `convergence_epsilon_bp`, `ces_rho` per commodity class | **Build this first** among equilibrium methods: no agent solves any optimisation problem, and it is provably robust to asynchronous/adversarial update order — exactly our situation. |
| **Entropic / multiplicative tâtonnement** | O(1/ε) iterations for Leontief Fisher markets; O(log 1/ε) for complementary-CES | **NO** | `max_iterations`, `epsilon`, `on_nonconvergence` policy | The fallback when provable rates over a Leontief production structure are needed. Must have an iteration cap and a documented hold-last-prices-and-flag behaviour — never an unbounded loop. |
| **Gauss-Newton least-squares labour allocation** (`cpetig/econsim`) | data-dependent iteration count, dense float matrices | **NO** | — | **REJECT** (and the repo has **no license** [V]). Documents the road not taken and *why*: variable per-tick cost, float-sensitive convergence, and an untestable branch surface. Useful as the answer to "why not just solve it properly": determinism and bounded per-tick cost, not laziness. |
| **Veloren iterative site economy** — `~10 GoodMap<f32>` + a `LaborMap<f32>` over professions per site, 15 abstract Goods, Leontief min-limiting productivity, economy-of-scale `absolute_scale = (1.0 + total_workers/100.0).min(3.0)` (**verbatim from `mod.rs`** [V]); `TICK_PERIOD` 90 days × `HISTORY_DAYS` 500 years ⇒ ~2,000 rayon-parallel ticks **run once at worldgen**, gameplay reading only a normalised price vector | a few hundred bytes/site; no allocation in the hot loop | **NO as written** (f32 + `hashbrown::HashMap` + `lazy_static` + rayon, all verified present [V]) | 15-goods taxonomy, smoothing factors, industry floor | Proves an entire multi-site production/trade economy fits in a few hundred bytes per site AND that **the sim need not be live** — a dormant realm can be caught up by N cheap ticks on wake, or run entirely offline at generation time. Its own source admits prices end up nearly identical in every town ⇒ **regional divergence must be engineered deliberately.** GPL-3.0 ⇒ re-derive. The smoothing constants (0.8, `2^(1−surplus/demand)`, `sum/1000` floor) are **[U]** — not located in `mod.rs`; they likely live in `context.rs`/`cache.rs`. |
| **Abstraction ratio: ~15–30 abstract goods + per-item bill-of-materials expansion** | O(recipe edges) per item price | **YES** | goods taxonomy, item→goods BOM, per-kind and per-quality merchant sell-discount | Keeps per-realm sim state at O(30) floats **regardless of how many block/item types P6 adds** — i.e. sim cost independent of content growth, which matters enormously for a voxel game. Veloren's quality-scaled sell-discount (0.9 for Tools/Armor/Wood/Stone, 1.0 for Food/Potions/Coin, × quality 1.0→0.0) is the built-in spread and primary currency sink. **This is a one-way design decision** and must be vetted against the full end-goal (signal-heavy cross-shard blocks, ships and stations built from blocks) before it is fixed. |
| **Single mid price per good per realm** (Offworld Trading Company) | O(1), one number to compute/persist/replicate/render | **YES** | — | Soren Johnson shipped two prices (buy = 2× sell, specifically to block self-arbitrage) then collapsed them into ONE and reported the mechanic became "so much more powerful" — ⚠ the cited interview URL **404s [V]**, so the quantitative detail is **[U]** and needs re-sourcing (Johnson's Designer Notes / GDC talks). The *decision* still stands on its merits: a single price is immune to intra-realm self-arbitrage, needs no per-order state (nothing to reconcile on wake), and pushes the spread to the edge (merchant discount or transaction fee). Supports a **two-tier design**: single-price ambient markets everywhere, real order books only at venue-capable realms. |
| **Per-realm congestion/depletion index** (EVE's system cost index generalised) — `job_cost = EIV × (system_cost_index − structure_bonus + facility_tax + 4% SCC)`, index per solar system and rising with local activity | O(1) read, O(activity) update | **YES** | `index_growth_rate`, `index_decay_rate`, `facility_tax_bp`, `authority_surcharge_bp` | The best precedent for our realm model: an economic parameter spatially scoped to exactly one authority domain, derived from activity inside it, consumed by every job there ⇒ **no cross-shard reads in the common case**, egress only as a periodic `EffectFree` publication. CCP named the *absence* of a place-scoped negative feedback loop on extraction ("the more you mine, the more you get") as the root cause of their worst economic crisis [V] ⇒ **build this from day one**; it is cheap, local, and it decentralises activity without global coordination. |

### 4.5 Money, indices, inequality

| Algorithm | Cost | Det. | Params | Port note |
|---|---|---|---|---|
| **Double-entry posting** — every committed entry-set sums to zero **per currency**, ≥2 entries; a 2-entry cross-currency transaction is a **bug** | O(entries) | **YES** | — | Modern Treasury is explicit that FX needs ≥4 entries across two per-currency pools, because rates fluctuate (historical verification becomes impossible) and there is no universally agreed rate [V]. Directly binding on the multi-currency ask. |
| **Two-phase pending/post/void with timeout** (TigerBeetle) | O(1) | **YES** | `escrow_timeout_ticks` (must be **derived from** the saga abort deadlines, not chosen independently) | Pessimistic validation at reserve time ⇒ **posting can never fail** ⇒ the saga's compensation is always `void` on unspent reservation, never a debit of posted funds. |
| **Closed faucet/sink taxonomy** — money is created/destroyed only through a named `Faucet`/`Sink` arm; every wallet mutation cites one, plus a closed `(ref_type, context_id, context_id_type)` provenance triple | O(1) | **YES** | per-arm budgets/caps | CCP publishes 72 named flows and still needs an "Active ISK Delta" **plug** because their accounting is not closed [V]. If ours is closed, `M(t) = M(t−1) + Σfaucets − Σsinks` is an exactly assertable identity ⇒ "is the economy leaking?" becomes a CI gate, not a forensic question. Copy ESI's `context_id_type` closed 12-value enum shape verbatim [V]. Include `Sink::AdminConfiscation` / `Faucet::AdminGrant` as **explicit named arms** so the identity never needs a plug. |
| **Largest-remainder (Hamilton) apportionment, ONE rounding point** | O(k log k) for k levies | **YES** | tie-break = `(RealmId, level)` ascending, never a hash or float compare | ~25 lines; **do not take `largest-remainder-method` as a dependency**. Rounding each level independently is the classic penny-allocation bug and in a game it is literally a faucet or a leak. Pin with a proptest: for random bases and rate vectors, `Σshares == total` and every share ≥ `floor(quota)`. |
| **Fisher / Laspeyres / Paasche / Törnqvist indices with chaining** | O(basket) per period | **YES** as `i128` rationals; the geometric mean's sqrt belongs in Tier-B or uses an integer sqrt | `index_baskets: [{name, items, weights}]`, `rebase_cadence` | **Compute FISHER as the headline, not Laspeyres**: Laspeyres is upward-biased exactly when players substitute away from an expensive good, which is the normal reaction to a supply shock ⇒ a Laspeyres CPI systematically overstates inflation and would mislead every operator tweak. EVE's Mineral/Ship/Module/Consumer split is the proven decomposition to start from [V]. |
| **Gini / Lorenz / Theil / Atkinson** | O(n log n), dominated by the sort | **YES** as `i128` rationals | — | **Expose THEIL per realm subtree, not Gini**, as the primary dashboard metric: Theil is additively decomposable into within-realm and between-realm components, which answers the actual operator question ("is this a rich-station problem or a rich-player problem?"). Gini is not decomposable. |
| **MV = PQ monitoring identity** | O(1) over window aggregates | **YES** | window length | With a closed ledger this needs **no new instrumentation**: M = money supply, V = Σ trade value / M over the window, PQ = Σ fill value. |
| **Money representation** — `i128` minor units + per-currency `minor_exponent`; all rates in integer basis points / ppm; every division an explicit `muldiv` with a declared rounding direction; rounding dust to a **declared sink account** | O(1) | **YES** | `minor_exponent` per currency, `rounding_policy` | The dust rule is load-bearing: if dust is discarded rather than posted, `Σdebits == Σcredits` fails **intermittently and under load only** — the hardest possible ledger bug to find, and the one most likely to get the invariant gate disabled. |

### 4.6 Multi-currency FX

| Algorithm | Cost | Det. | Params | Port note |
|---|---|---|---|---|
| **`ledger`-partitioned currencies + linked-transfer FX** (TigerBeetle) — `ledger:u32` partitions which accounts may transact; cross-ledger transfers are **prohibited**; an exchange is two atomically **linked** transfers (source→source-liquidity in ledger 1, dest-liquidity→dest in ledger 2) with the spread as an explicit third linked transfer | O(1) per leg | **YES** | `spread_bp` per pair, liquidity-account owners | Resolves multi-currency cleanly and cheaply with **no new mechanism**: `CurrencyId` is a partition key, cross-currency is never a single entry, and every FX spread becomes an explicit ledger entry rather than an implicit float conversion — which is exactly what "fully analyzable" requires. Type-forbid adding two different `CurrencyId`s at compile time. |
| **Numeraire star topology + N−1 books** | O(1) cross-rate derivation | **YES** | `numeraire: CurrencyId` | With N currencies a full mesh is N(N−1)/2 books whose cardinality explodes and whose triangular inconsistencies are permanent bot income; a **star** has N−1 books, makes cross-rates *derived* rather than quoted, and makes triangular arbitrage **structurally impossible in the base case**. |
| **Bellman-Ford negative-cycle detection on −log(rate)** | O(V·E); a real-time crypto implementation reports ~0.002 ms average detection latency **[L]** | **YES** with integer log-rates in scaled `i64` (never f64 logs) | `alert_threshold_bp` | Use as a **monitoring assertion, not a feature**: with an enforced star topology, "a negative cycle exists" means a mechanism is leaking (a mispriced NPC anchor, an AMM rounding bug, a stale peg) ⇒ fail a CI invariant and light up the dashboard. |
| **Currency SCOPE taxonomy** (from EVE's four currencies) — `{fungible + transferable + global}` \| `{issuer-scoped + non-transferable + spendable only at the issuer}` \| `{commodity-ised, i.e. currency wrapped in a tradeable item}` \| `{account-scoped, out-of-band-issued}` | — | **YES** | a registry entry per currency | The decisive insight: **each scope rule changes what cross-shard machinery is needed.** Issuer-scoped (EVE's Loyalty Points, 18.68 T/month sink) needs NO transfer machinery at all — it never leaves the issuer's authority. Commodity-ised (Skill Extractors: 14.96 T + 9.71 T/month) needs only existing *item* transfer. Only ISK-like currency needs a cross-shard `Funds` saga. Getting this taxonomy right up front makes "add a currency" a registry entry (HR2-style), not a feature. |
| **Peg vs float** | — | — | — | Precedents: Entropia **hard peg** 10 PED = US$1 with real withdrawals (**this is the only Entropia figure the citation actually supports** [V]); Second Life LindeX **floating** resident-to-resident market (~L$247/USD historically); EVE **dual currency** (PLEX bought for real money, freely traded against ISK). A hard peg means the issuer takes a full reserve obligation and the peg *becomes* the game's monetary policy. |
| **Player/corporation-issued scrip** — an on-ledger issuance registry (total issued is a queryable balance, not a claim), a declared redemption asset held in a **separate** escrow that cannot be the issued currency, mandatory public reserve disclosure, and **no protocol-level acceptance guarantee** | O(1) | **YES** | issuance caps, disclosure cadence | What makes a player currency collapse is structural, not algorithmic: no redemption commitment, reflexive collateral (reserves that are themselves the currency), no registry authority for who issued how much, and a bank run once confidence breaks. Eco's dual model (infinite personal fiat vs mint-backed by permanently destroying a material, ≤25,000 units per action) is the shipped precedent **[L]** — it makes currency an **issuable object with a per-currency issuance policy** rather than a hardcoded enum of coin types. Under HR1 the mint must be a single authoritative shard per currency, or supply becomes forgeable across shards. |

### 4.7 Corporations + stocks

| Algorithm / mechanism | Cost | Det. | Params | Port note |
|---|---|---|---|---|
| **Fixed wallet divisions** (EVE: exactly 7, `division` min 1 max 7, array `maxItems` 7) | O(1) | **YES** | `wallet_divisions` (small constant) | Bounded state, no dynamic allocation, trivially serialisable, and role access becomes a fixed-width bitmask. Copy the count-is-small decision. |
| **Orders "on behalf of" an org via a FLAG** (EVE's `is_corporation`) | O(1) | **YES** | — | ONE machinery with policy fan-out on an actor field — literally HR2's Durable-vs-Transient pattern applied to actor identity, rather than a separate corp-order type. |
| **Share registry as a first-class ledger instrument** — fixed issued supply; total issued is *derived*, never asserted | O(1) | **YES** | `max_supply`, `par_minor` | Entropia's Calypso Land Deeds are the only shipped virtual equity with real dividends — ⚠ but the **60,000 deeds / 25% of net income / weekly payout figures are [U]**: the cited Wikipedia page supports only the 10:1 PED redemption [V]. Re-source from MindArk before any of it becomes a design premise. |
| **Issuance / dilution gated by a shareholder vote** | O(holders) | **YES** | `quorum_bp`, `threshold_bp`, `record_tick_offset` | Weight votes by the registry balance **snapshotted at a named record tick**, otherwise flash-loan vote-buying is trivial. |
| **M-of-N officer signatures above a threshold** | O(N) verify | **YES** | `withdrawal_threshold`, `m`, `n` | Corporate assets must live in accounts officers cannot unilaterally drain. We already have `ed25519-dalek` and `hmac` [R]. |
| **Dividends / buybacks as batched multi-party distribution** | O(holders), drained over several ticks | **YES** | `batch_size`, `max_ticks_per_distribution` | ⚠ **This is the one genuinely NEW requirement**: our saga is 2-party (source/dest), and a dividend to 5,000 holders, a share split, or a merger is N-party. The cheap correct answer is the **existing batched go-token pattern**: ONE `TransferId`, ONE CAS, N credit legs each idempotent by `(TransferId, holder_step_id)`, drained under bounded backpressure — with `LossBudget::ZERO` and the `Durability::Retained` outbox marker instead of a loss budget [R]. Credits are monotone/I-confluent so they need only at-least-once + dedup, making this the **cheapest possible large operation**. **Do NOT** implement it as a multi-key directory CAS — that would collide with D-32's range-partition-by-region blocker [R]. Integer division with an explicit remainder policy is mandatory. |
| **Disclosure as a mechanic** — treasury balance and share count publicly queryable | O(1) | **YES** | disclosure cadence | Converts "fraud" from an information asymmetry into visible risk-taking. Players can still be fooled about the **future** (good gameplay) but not about the **present balance** (that is just a missing feature). |
| **Ban on unbacked yield-bearing instruments** | — | — | — | Second Life's Ginko Financial collapsed with **L$55 M of withdrawal requests against L$180 M of deposits** [V], triggered by an unrelated gambling-policy change. ⚠ The often-quoted **69.7%/yr rate, the ~US$750 k destroyed, and the "no interest without a real banking charter" rule are all [U]** — the cited page supports none of them (that rule comes from Linden Lab's 2008 banking policy, which needs its own citation). EVE's record: EIB/Currin ~700 bn ISK **[L]**, Ebank ~200 bn ISK embezzled **[L]**; Second Life's World Stock Exchange died after an insider breach **[L]**. **The structural lesson is solid regardless**: every yield-bearing instrument must be **server-ledgered with solvency enforced by the simulation** (a dividend is a scheduled, funded, fence-stamped transfer with a balance check), never a promise held by a player. |
| **Equity trades on the same clearing engine as goods** | — | — | — | HR3 (one tooling). And explicitly **no AMM for equity** (see §4.3). |

### 4.8 Hierarchical taxes — the structural answer

Eight rules. This is the most load-bearing design content in the report because it is where a naive
implementation silently breaks under RLM (which changes tree depth dynamically), and because every
rule is expressible as an algebraic proptest rather than a scenario.

| # | Rule | Precedent | Why |
|---|---|---|---|
| **T1** | **ONE taxable event, ONE base, ONE situs.** Every economic act emits exactly one `TaxableEvent { kind, base_minor, situs: RealmId, event_fence }` whose situs is the **deepest realm containing the event** — precisely the containment result the realm layer already computes [R]. **Never** "every realm the goods passed through." | — | Unambiguous incidence; no double counting. |
| **T2** | **Rate composition = SUM over the ancestor chain, applied ONCE to the same base.** `chain = path_index(situs)`; `total_bp = Σ rate_bp[level]`; `tax = base·total_bp/10_000`. **FORBIDDEN**: multiplicative/cascading composition (each level taxing the level below's post-tax amount). | NY combines 4% state + 3–4.75% local + 0.375% surcharge into ONE combined rate (~8.45% avg) on one base [V]. EVE planetary customs is **additive**: NPC base 10% (5% with Customs Code Expertise V) **PLUS** the player POCO owner's tax, "at a maximum the total tax rate could be 100%" [V]. | Cascading = **tax pyramiding**, which makes the effective rate a function of **tree depth** — catastrophic for us specifically because RLM changes depth dynamically. `path_index` is already O(depth·log L) since RLM 5e-3a [R]. **Proptest: inserting a realm with rate 0 anywhere in the chain must not change any tax by one minor unit.** |
| **T3** | **Value-added base for production.** A manufacture/refine event's base is `output_value − Σ credited_input_value`, implemented as a `tax_credit_minor` TLV field carried on each item stack. | VAT input credit; the ideal sales-tax base explicitly exempts intermediate goods to avoid taxing the same item repeatedly [V]. | Without it a 5-stage BOM taxes the same ore five times and **adding a stage silently raises prices**, so recipe refactors move the economy. Combined with an acyclic recipe DAG this yields: total tax on a finished ship is **independent of how many intermediate stages its BOM has**. ⚠ Requires planning the TLV schema **now** — decode-to-Default is banned for Durable kinds [R], so adding the field later is a versioned migration. |
| **T4** | **Ceilings + subsidiarity with a DECLARED clamp order.** `TaxTuning { per_level_cap_bp, chain_cap_bp, clamp_policy }`; a parent stores a `child_cap_bp` so caps themselves compose top-down; clamp each level to its cap, then if `Σ > chain_cap_bp` scale down by a declared policy (root-claims-first \| leaf-claims-first). | EVE's 100% total ceiling on planetary customs [V]. | Never an undeclared order — that is a silent nondeterminism. |
| **T5** | **Floor + nearest-ancestor top-up (the anti-haven lever).** If a child's effective rate < `floor_bp`, the nearest ancestor with `top_up=true` levies exactly the difference. | **OECD Pillar Two verbatim**: a 15% minimum with jurisdictional blending and a hierarchical rule order — QDMTT gives the local jurisdiction first claim, the parent's IIR tops up, UTPR is the backstop [V]. | Kills "tax-haven station" **without banning low taxes**. Pure function of the ancestor chain + config ⇒ evaluable at the situs shard with no cross-shard call. One clamp routine implements both T4 and T5. |
| **T6** | **ONE rounding point + largest-remainder split.** Floor once on the total, then apportion among levies by Hamilton so `Σshares == total` **exactly**; ties broken by `(RealmId, level)` ascending. | §4.5 | Rounding each level independently is a faucet or a leak. |
| **T7** | **Tariffs are EDGE taxes at the LCA.** A flow a→b has situs = the *edge*; decompose into an export chain `a→LCA(a,b)` **exclusive** of LCA, an import chain `LCA→b` exclusive, plus an optional transit levy by the LCA itself. | — | Charging every realm on the traversed path **double-counts every shared ancestor** (a 2-hop intra-system move would pay the galaxy rate twice). LCA is already the Signal system's routing primitive ⇒ reuses `path_index`/`closure_peers` with zero new machinery [R]. **Proptests**: `a==b ⇒ tariff ≡ 0`; siblings ⇒ only the parent levies; inserting a 0-rate realm on the path changes nothing; symmetric under swapping export/import roles when rates are symmetric. Feeds the min-cost-flow transport-cost matrix as an additive integer edge cost, so tariffs automatically reshape trade routes — which is the gameplay point. |
| **T8** ⚠ **REWRITTEN in revision 2** | **Idempotency via Fence + PUSH-WITH-RETENTION.** The assessment is a **pure function** of (event, ancestor chain, config), keyed by the originating action's `Fence`/`(TransferId, step_id)`. At assessment the situs shard debits the payer **locally** and credits an **`InTransit`/treasury-clearing account** (N7) — so the money is *located* the instant the debit lands. The remittance to each beneficiary is then a **side-effecting flow keyed `(TransferId, step_id)`**, classified `FlowDurabilityClass::ProducerLessReliable` with `Durability::Retained` (the R-6d outbox), exactly as `TransientBatch` and `GhostFlow::Despawn` already are [R]. | — | ⚠ **Revision 1 had this exactly backwards** and it broke I3. It said the situs holds the accrual and the beneficiary **PULLS** when it next runs. Under HR1 a shard cannot query another shard, and under RLM the **situs** realm is the one that gets torn down (LF-1: killable on zero occupants AND zero live descendants AND no live KeepAliver AND declared-parent-live [R]). So the durable half of the debit sat at a shard that may be dead or unreachable when the beneficiary reconciled, the credit half had **no re-driver**, and the carrier was additionally classified "`EffectFree` for the notice / idempotently re-derivable" — but re-derivation only works while the situs is alive. Net effect: a debit with no matching credit ⇒ **I3 fails, or the accrual is silently written off.** The push-with-retention form puts the retry obligation on the side that holds the money and has an outbox. **Alternative worth presenting:** hold accruals in the orchestrator's durable saga store, which is never dormant. **Either way, state the rule explicitly: a `Signal` notice may NEVER be the only carrier of the credit half of an applied debit.** (The Signal notice remains useful as an *advisory* wake-up; it is not the mechanism.) And if the beneficiary realm is never respawned at all, the escheatment rule in §7.10 applies — an expiry to the nearest live ancestor treasury as a declared `Transfer`, never a silent drop. |

**Tax incidence and avoidance are MEASURED, not modelled.** With supply elasticity εs and demand
elasticity εd the buyer's share of an ad-valorem tax is εs/(εs+εd) — we do not implement this, we
observe it on the dashboard by comparing pre- and post-tax equilibrium prices. Likewise Laffer-style
avoidance: Saez, Slemrod & Giertz (JEL 50(1):3–50, 2012) put the best real-world elasticity-of-taxable-income
estimates at **0.12–0.40** [V], but in a game with cheap mobility the effective elasticity is far
higher because players simply move their trade to a lower-tax station — so **per-region rates MUST be
paired with T5 or high-tax regions self-empty.**

**EVE's calibrated fee schedule** (the initial value set for `EconomyTuning` — real rates from a
20-year-live economy beat invented ones) [V]:

| Fee | Rate |
|---|---|
| Sales/transaction tax | 7.5% base → 3.3–3.6% at Accounting V; paid to NPCs = a **SINK** |
| Broker's fee (NPC stations), charged at **order creation**, non-refundable | `3% − 0.3%·BrokerRelations − 0.03%·faction_standing − 0.02%·corp_standing`, floor ~1%; in player structures a fixed 0.5% SCC surcharge **+ an owner-set %** = a **TRANSFER**, not a sink |
| Upwell-structure minimum broker fee | 0% → 1% in 2020, with **half remitted to an NPC (the SCC)** |
| Relist fee | `max(0, BR·(P2−P1)) + (1−RD)·BR·P2`, min 100 ISK; RD 50% → 75–80% with Advanced Broker Relations |
| Industry job cost | `EIV × (system_cost_index − structure_role_bonus + facility_tax + 4% SCC)`; NPC facility tax 0.25% of EIV |
| Corporation tax | 0–11% (NPC corps 11%), applying **only** to bounty / Project Discovery / AIR daily / mission payouts **over 100,000 ISK** |
| Reprocessing | 5% → 0% at 6.67 standings |
| Planetary import/export | NPC 10% high-sec (5% at Customs Code Expertise V) **+ player POCO tax**, total capped at 100% |
| Factional-warfare maintenance | 0–75% on donated LP |
| Hypernet Relay lottery | 5% |

Two structural lessons beyond the rates: (a) charging on **order creation** as well as on fill is a
deliberate anti-spam/anti-reprice mechanism and belongs in our design from day one, not retrofitted;
(b) the recurring pattern is **a player-configurable rate with an NPC floor, plus a mandatory
remittance split to a system authority** — which *is* hierarchical tax remittance up the realm tree,
and it is what makes the economy analyzable, because every unit is attributable to a
`(flow_kind, realm)` pair.

Other shipped fee schedules for calibration: **Guild Wars 2** 5% non-refundable listing + 10% exchange
= 15% total, explicitly framed as a gold sink (cite `wiki.guildwars2.com/wiki/Trading_Post`, **not**
help.guildwars2.com which 403s) [V]. **RuneScape (RS3)**: "Since 9 January 2023, **2%** of the sales
price of items is withheld by the Grand Exchange in the form of a tax", rounded down per item, **no
cap** — ⚠ **CORRECTION**: the research said "1% capped at 5 M gp"; that is *Old School RuneScape*, a
different game [V]. Also verified from the same page: buying "is restricted to a certain quantity
every 4 hours", the ±5% price band was **removed on 2011-02-01** in favour of quantity limits, and
Jagex states "We will only intervene as a last resort, and only if we think price manipulation is
going on" [V]. **Albion** 2.5% setup on create *and* edit + 8%/4% sale tax and **Dual Universe**'s
flat + 1% + 0.02/day storage taxes with a 5-minute order-edit cooldown are **[U]** (both sources are
bot-blocked/unreachable).

### 4.9 Anti-degeneracy

| Exploit / failure | Structural defence | Det. | Note |
|---|---|---|---|
| NPC price seam (any NPC buy ≥ any NPC sell over a craftable chain) | Finite-quantity NPC anchors with a config replenishment **budget**; no infinite-depth fixed-price orders | YES | The single most dangerous pattern; see §4.3. |
| Zero-margin bot arbitrage | A non-zero **per-order cost** (listing/setup fee) so zero-margin round-tripping is unprofitable by construction | YES | Cheapest bot deterrent there is; EVE runs 7.5% sales + 1–3% broker [V]. |
| Order-book spam / relist churn | `max_orders_per_actor` cap + price tick size + modification fee | YES | The composable trio: cap the state, quantise the lattice, price the mutation. |
| Wash trading / self-trade | Self-match prevention keyed on the owning **legal entity** (character → corp → alliance) | YES | Without it, alt-pairs manufacture price history, farm volume rewards and launder value. |
| Sub-tick undercutting loops | Minimum tick + minimum order size | YES | Falls out of quantisation. |
| Fat-finger repricing an index | Price bands / volatility interruptions that extend the call phase when the book would clear outside a band (Xetra) | YES | Xetra also publishes only the *indicative* clearing price/volume and **randomises the auction end time** [V] — both worth copying. |
| Duping | Items as **ledger positions** with a conserved lineage, not mutable counts; the ledger (never the blob) as the uniqueness authority; the double-entry invariant as a HARD assertion | YES | Makes a dupe a **failing test the tick it happens**, not an economics observation. Historical scale: the 2003-11-07 RuneScape Purple Party Hat was duplicated **well over 2,000,000 times** and the price effects persist today [V]. |
| Unfunded compensation (insurance, rebates) | Actuarially closed by construction: the payout pool is a real balance funded by premiums; an overdraw **fails loud** rather than minting | YES | EVE's insurance has been a **~+2.8 T ISK/month net faucet for two decades** [V] — a money printer run deliberately because it is load-bearing for ship-loss psychology. The clearest cautionary tale in the dataset. |
| Extraction with no negative feedback | Per-realm depletion/congestion index (§4.4) | YES | CCP named this absence as the root cause of the Scarcity crisis [V]. |
| Player financial fraud | Registry authority as authoritative game state + solvency enforcement + present-balance disclosure | YES | See §4.7. |
| RMT / botting | Named `Sink::AdminConfiscation` / `Faucet::AdminGrant` arms + graph-based detection (§3.8) | YES | 3,165 macro bans in June 2026 alone [V]; historical exploits of ~2.5–3 T ISK over **four years** [L] survived because nobody was diffing production against expectation ⇒ **analyzability is a security control, not a nice-to-have.** |
| Operator tweaks as an unmeasurable weapon | An **intervention log** designed at the same time as the tweak API | YES | The OSRS causal study found the tax barely moved trading at taxed price points and the item sink **raised** luxury prices [V] ⇒ sinks and taxes are weaker levers than designers assume, and item sinks can be price-**raising** for speculative goods. |
| Global efficiency as an implicit goal | Deliberate friction: per-realm books, physical haulage, presence requirements, order-edit cooldowns | YES | Path of Exile has repeatedly declined an in-game auction house to protect "economic integrity" [V]. Under sealed shards friction is nearly free and global efficiency is expensive ⇒ the incentives align. **Treat every request for a galaxy-wide market as a request to centralise load, and price it as such.** |
| **Ledgered value destroyed by a LEGITIMATE transient loss** ⚠ *new in revision 2* | Any kind that may carry a ledger position is **either** `Durable`/`LossBudget::ZERO`, **or** its loss path posts to a named `Sink::TransientLoss` | YES | See §4.9.1 — this is the hole that would have silently broken I3/I5/I10. |
| **A correctly-DECLARED faucet that leaks forever** ⚠ *new in revision 2* | Per-arm, per-epoch, per-realm-subtree **budgets** enforced at the mint site (I16), plus an external cross-check of each shard's self-reported partials against its own emitted journal (I17) | YES | I5 only requires that a mint *cites* a registry arm. A shard-local bug (or an exploited mechanism) that mints and correctly declares the mint passes I3 **and** I5 forever — the sum reconciles *because* the faucet is declared. That is precisely EVE's four-year undetected-exploit failure mode reproduced inside our own invariant suite, and under HR1 the reconciliation sweep can only ever sum what each sealed shard self-reports. |

#### 4.9.1 Ledger positions vs the landed transient loss machinery

Revision 1's central anti-dupe claim (I10 "every `ItemId` is in exactly one container at every committed
instant", I3 conservation, crafting as a declared faucet+sink pair) was stated as if all item movement
is lossless. **It is not, and the machinery that legitimately destroys in-flight entities is already
landed:**

- `crates/core/src/entity_kind.rs:225–260` registers `DEBRIS_DEF` **`LossBudget(4)`**,
  `DROPPED_BLOCK_DEF` **`LossBudget(2)`**, `PROJECTILE_DEF` **`LossBudget(8)`**, `ROCKET_DEF`
  **`LossBudget(2)`** [R].
- `InterShardFlow::TransientAbandon` (`crates/wire/src/intershard.rs:180`) exists precisely to abandon a
  batch as *an accounted loss-within-budget* when the dest dies [R], policed by
  `verify_transient_loss_budget` (`crates/harness/src/oracle.rs:440`) [R].

A `DroppedBlock` is an economically valuable item. If a dropped stack carries a ledger position, then
**every abandoned batch destroys ledgered value outside a declared `Sink`** ⇒ I3, I5 and I10 all fail,
and the overwhelmingly likely outcome is that the conservation gate gets relaxed — the exact failure
mode §8.6 Rule 0 warns about.

**The rule (design it, do not discover it):** an entity kind may carry a ledger position **only if**

1. it is `Durable` with `LossBudget::ZERO`, **or**
2. its loss path **posts to a named `Sink::TransientLoss`** at the moment `TransientAbandon` /
   over-budget loss is accounted, so the loss is an *explainable non-conservation* rather than drift.

`verify_transient_loss_budget` is the coupling point: the same accounting that counts the lost items
must emit the sink postings. **Pin it with a G6 sibling (G6b):** kill the dest of a batch of
value-bearing dropped stacks and assert `Δ money_supply == Σ faucets − Σ sinks` still closes, with the
abandoned value showing up under `Sink::TransientLoss` and nowhere else.

### 4.10 The material economy — extraction, destruction, and the issuance targets

⚠ **This section did not exist in revision 1**, which built an elaborate *accounting* frame (the closed
`Faucet`/`Sink` enum S7, I3/I5, the per-realm depletion index) while leaving the two flows that actually
decide whether a **voxel** economy works as an open question (OQ #6, "which sinks replace EVE's?").
That is inverted: the fee schedule is a tuning detail, the material balance is the design.

**Why it cannot wait for P9.** P4 terrain, P5 physics and P6 blocks land **before** the economy arc, and
each fixes the faucet geometry irreversibly. A regenerating terrain is a different economy from a
finite one — different scarcity, different hauling, different sinks, different endgame — and it cannot
be changed later without a world migration.

**(a) The primary faucet: extraction.**

| Parameter | Shape | Note |
|---|---|---|
| `yield_per_voxel` | per-material, integer minor units of the resulting `ItemKind` | The extraction faucet's unit. A per-entity/per-material field, never a constant. |
| Ore distribution | **closed-form `f(seed, realm_path, voxel_pos)`** | ⚠ **A pre-P4 seam (S12).** Same discipline as celestial math (Category A). It makes extraction deterministic **and** makes the faucet *computable ahead of the economy existing* — the generator can answer "how much iron is reachable in this system" before a single ledger row exists, which is what lets the issuance target be set rather than discovered. |
| `regeneration_policy` | **`Finite` \| `RespawnAfter(ticks)` \| `FieldReplenish(rate)`** per material class | **This is D16, a user decision.** Finite-per-planet makes depletion permanent and hauling distance monotonically increasing; respawning belts make extraction a renewable flow and turn the depletion index into the only scarcity lever. |
| `extraction_rate_cap` | per-tool/per-machine, bounded by config | With the per-realm depletion index (§4.4) this is the place-scoped negative feedback CCP named as the absence that caused their worst economic crisis [V]. |
| Target | **faucet rate per player-hour**, stated per material class | The number that makes the economy tunable at all. |

**(b) The primary sink: destruction.**

| Parameter | Shape | Note |
|---|---|---|
| `loot_drop_ratio` | integer bp of a destroyed assembly's blocks that survive as loot | The rest is the sink. **This is the single biggest currency/material sink in a block-built game** — Albion's full-loot PvP is the shipped precedent [L], and EVE destroys 771,903 things/month [V]. |
| `wreck_persistence_ticks` | how long a wreck is salvageable | Bounds the loot-drop faucet's tail and stops wreck fields becoming permanent state. |
| `salvage_yield_bp` | fraction recovered by salvaging | The second-order faucet; must be < 100% or destruction is not a sink. |
| Emission | **a declared `Sink::Destruction` posting per destroyed block, from its FIRST commit** | ⚠ **A pre-P11 seam (S13).** If the destruction path does not emit a declared sink from day one, the primary sink is **unmeasurable** and no issuance target is assertable. Retrofitting it means re-deriving history. |

**(c) The issuance targets** (this is what revision 1 left as an open question):

- **Net currency issuance ≤ X %/month of money supply**, stated as a band, with the *validation* rule
  from §8.6 rejecting any config whose modelled issuance exceeds it. EVE runs **+2.16%/month** [V] —
  reported here as an *observation*, and a defensible starting band is **−1% to +1.5%/month**, tighter
  than EVE because we intend destruction sinks EVE would envy.
- **Net material issuance ≈ destruction + decay**, i.e. `Σ extraction ≈ Σ (destroyed − salvaged) + Σ
  storage decay`, per material class, over a stated window. This is a dashboard series **and** a config
  validation, not a hope.
- **The two knobs are `yield_per_voxel` and `loot_drop_ratio`.** Everything else (fees, taxes, sinks) is
  second-order — the OSRS causal study is explicit that taxes and item sinks are **weaker levers than
  designers assume** [V].

**Coupling to P11.** `loot_drop_ratio` and the full-loot decision are the same decision viewed from two
sides. Decide them together, or the economy's primary sink is set by a combat-design accident.

### 4.11 The economic ACTION taxonomy (from which S7's closed enum is derived)

⚠ **Also absent from revision 1**, which presented S7 (the closed `Faucet`/`Sink` enum) as a cheap
~80-line early seam whose "only cheap moment" is before the first faucet exists — while never producing
the set of player actions the enum is a function of. A closed, **wire-visible** enum authored from an
incomplete action set is exactly the versioned migration S2 warns about. This is also why the user's
"driven by REAL USER INTERACTIONS" half was the thinnest part of revision 1.

**The action set** (the completeness check MMOAgent's "6 resources × 5 activity types" [V] was standing
in for):

| # | Action | Faucet arms | Sink arms | Notes |
|---|---|---|---|---|
| 1 | **Extract** (mine/harvest a voxel) | `Faucet::Extraction` | — | §4.10(a); gated by the depletion index |
| 2 | **Refine** | `Faucet::RefineOutput` | `Sink::RefineInput`, `Sink::RefineLoss` | VAT input credit base (T3) |
| 3 | **Craft / assemble** | `Faucet::CraftOutput` | `Sink::CraftInput`, `Sink::JobFee` | recipe registry, not player build (§7.12) |
| 4 | **Build / place a block** | — | `Sink::Construction` | material leaves circulation into a structure |
| 5 | **Deconstruct** | `Faucet::Deconstruct` (partial) | `Sink::DeconstructLoss` | Eco's multi-clause-law lesson: charge for *both* directions or build/destroy loops farm it [V] |
| 6 | **Haul** | — | `Sink::FuelBurn` | the arbitrage engine's cost term |
| 7 | **Trade P2P** (§7.8) | — | `Sink::TradeTax` | two-sided atomic swap |
| 8 | **List / modify / fill an order** | — | `Sink::BrokerFee`, `Sink::RelistFee`, `Sink::TransactionTax`; `Transfer::VenueOwnerFee` | fee on *creation* as well as fill (§4.8) |
| 9 | **Issue / accept / complete a contract** (§7.9) | — | `Sink::ContractFee`; `Transfer::CollateralForfeit` | collateral is a transfer, never a sink, unless forfeited to an NPC |
| 10 | **Insure / claim** | `Faucet::InsurancePayout` **capped by the pool** | `Sink::Premium` | §7.14 — actuarially closed by construction |
| 11 | **Salvage** | `Faucet::Salvage` | — | bounded by `salvage_yield_bp` |
| 12 | **Rent / store** | — | `Sink::StorageRent` | one of the few sinks that scales with *hoarding* (§7.10) |
| 13 | **Pay wage / dividend / tax** | — | `Sink::TaxToNpc`; `Transfer::*` for player beneficiaries | a dividend is N-party (D-50) |
| 14 | **Destroy** (combat, collision) | — | `Sink::Destruction` | §4.10(b); the primary sink |
| 15 | **Die / drop loot** | `Faucet::LootDrop` (bounded by `loot_drop_ratio`) | `Sink::DeathLoss` | the two halves must sum to the destroyed value |
| 16 | **Transient loss in flight** | — | `Sink::TransientLoss` | §4.9.1 |
| 17 | **Admin grant / confiscate** | `Faucet::AdminGrant` | `Sink::AdminConfiscation` | operator identity + case id (§9.2) |
| 18 | **Genesis** | `Faucet::Genesis` | — | M(0), §7.15 |
| 19 | **Escheat** | — | `Transfer::EscheatToAncestor` | never a silent drop (§7.10) |

**The extension rule** (so arm #20 is a planned migration, not a wire break): the `event_kind` and
`Faucet`/`Sink` discriminants are **`u16` with a reserved range per category** (0x0000–0x0FFF core,
0x1000–0x1FFF gameplay, 0xF000+ operator), decoded under the existing per-kind **version-floor
writer-N+1/reader-N handshake** [R]; an unknown discriminant in the *analytics* stream is retained
verbatim as `Unknown(u16)` and reported, while an unknown discriminant on an **authoritative** posting is
a hard fail (decode-to-Default is banned for Durable kinds). EVE needed **72** flow categories after 19
years [V]; 19 actions × ~2 arms is the right order of magnitude for day one.

---

## 5. Case studies, mined for lessons

Short and load-bearing only. Each: copy / avoid / why.

### 5.1 EVE Online — the only shipped economy at the target scale

**COPY (mechanisms).** The region-partitioned market model: it is **already sealed-shard-shaped** —
ESI is region-keyed, there is no global `/markets/orders/` endpoint [V], so a per-realm order book is
not a departure from EVE, it is EVE's *logical* model with the central DB removed. Event-driven
matching at order create/modify. 100% buy-order escrow (it converts a distributed-credit problem into
a local-funds problem, which is what lets EVE's market be asynchronous). Bounded order duration (self-GC).
Tick size. Per-actor order caps. The complete fee lattice (§4.8). The `(ref_type, context_id,
context_id_type)` closed provenance triple on every currency movement. Exactly 7 corp wallet divisions.
Orders "on behalf of" an org via a flag. **Contracts** — item exchange / courier (reward + hauler
**collateral** in escrow + a days-to-complete deadline) / auction, scoped Public \| Private \| Corp \|
Alliance, with a 30-day retention window [V]: courier contracts are *the* mechanism that makes a
distributed market work without a global market, and on our architecture a courier contract is almost
exactly a transfer saga with a collateral escrow and a deadline ⇒ **contracts are not a separate
subsystem, they are the transfer machinery with an economic policy on top.** The per-system industry
cost index. The **ESS** — a per-system *contested escrow*: currency held by a PLACE, not an actor, with
a contested release condition (a publicly-visible ~6-minute drain countdown), paying out via batched
multi-recipient distribution; 31.31 T/month, a top-3 faucet [V]. That pattern is what makes economic
state interesting rather than merely accounted, and it gives a *gameplay* reason for economic state to
be per-realm — which is what HR1 wants anyway.

**COPY (governance).** The MER itself is the dashboard spec: 72 named flows at daily granularity, four
price indices with basket decomposition since 2003-11, daily money supply + velocity, per-region key
figures, and an **event-level** kill dump [V].

**AVOID (engineering).** Everything reconverges on **one ACID SQL Server**, and every mechanism
(escrow, wallet, order, contract, industry job) is a row in it. A node is one single-threaded process
pinned to one core; a solar system cannot be subdivided, which is *why* time dilation exists at all
[V]. That is precisely the assumption HR1 removes: we must replace "one transaction" with "one saga +
one directory CAS" for every value movement that crosses a shard, and **our answer to an overloaded
market realm must be RLM partitioning, never slowing the clock.**

**AVOID (analytics).** CCP publishes the market as **snapshots**, so trades must be *inferred*: a
fully-consumed order is indistinguishable from a cancelled one, and a naive reconstruction recovers
only **~30% of the true trade count** even when it captures the removed volume [V]. Third-party
archives pay ~90 GB/yr for this [V]. **This is the strongest single argument for making the economic
event log a first-class, load-bearing output**: we already must write `applied_steps` before every
side-effect, so if every fill/fee/mint/burn is an append-only fence-stamped record in that same
journal, the dashboard is a projection over a log we write for *correctness* — zero extra durability
cost, and 100% (not 30%) of trades observable.

**AVOID (concentration by default).** The Forge is **68.8% of trade value on 24.6% of orders** [V], and
the causes are all mechanical and all avoidable: sell orders cannot reach out (asymmetric range), fees
are location-independent so there is no cost to centralising, information is free within a region so
the deepest book always wins — and CCP then **ratified it in hardware** (a dedicated node for Jita,
another for The Forge market). Note the asymmetry precisely: a **sell order is station-locked** while a
**buy order carries a jump range** (`station|solarsystem|1,2,3,4,5,10,20,30,40|region`, skill-gated) [V].
Making sell range symmetric, or making range **cost money**, is the lever EVE never pulled. On our
architecture range is naturally a **realm-subtree predicate** rather than a jump distance — cheaper,
and it composes with our path index.

**AVOID (violent supply interventions).** From 2020 CCP deliberately contracted mineral supply
(Scarcity), driving the Mineral Price Index to all-time highs; reported second-order effects were that
wars stopped, capital ships docked up, players hoarded, and activity slumped [V]/[L]. The named root
cause was the *absence of a place-scoped negative feedback loop* on mining. ⇒ **build the per-realm
depletion index from day one so the operator never has to make a game-wide supply intervention**, and
give the tweak surface simulate/backtest-before-apply.

### 5.2 Eco — the closest shipped game to the literal ask

**COPY.** Taxes/regulation as a **rule graph over the game's own action stream**, not an economy
subsystem: a law is authored by picking a trigger (an ordinary game action — "Pollute Air", chop a
tree, construct a road item, or a periodic "Citizen Timer"), narrowing the source, optionally adding a
condition, and running an action; actions include **currency transfer in both directions** with the
amount computed by an arithmetic expression over a selectable game value, "Change Property Owner", and
authorisation gates [V]. Funds land in a visible **Treasury**. Governments are **nested tiers** —
individual (Home Claim) → town (Town Hall) → nation (National Constitution) → global (Global Charter) —
each a "Government Source" projecting an influence radius sized by its building's housing value [V].
Two exploit-class details worth copying verbatim: multi-clause laws (pay for building a road, **charge**
for deconstructing it, "so you can't just build and destroy the same piece over and over") and the
wealth tax that "looks at non-government accounts, and for multiply owned accounts it takes a
**pro-rated** amount. So, no stashing those funds in overseas accounts" [V].

**Why it matters for us.** This is a direct blueprint for a tax layer that respects HR3 and HR4: taxes
are **data** interpreted against the action/Signal stream every feature already emits, so no feature
ever needs to know about taxation and "features once, run anywhere" holds. Our realm tree already *is*
the nested hierarchy with a live path index, so a tax law is a rule attached to a realm node applying
to all descendants, and a player's applicable law set is the path to the root — computable with the
`path_index`/`closure_peers` we already built [R].

**AVOID.** Eco's competing same-tier **influence radii** (overlapping jurisdictions). Our strict tree
gives unambiguous, cheap, deterministic membership; overlapping jurisdictions make tax incidence
ambiguous for a single transaction. Also: the law interpreter must be a **total, terminating** evaluator
(no user-supplied loops) to stay deterministic and coverable — **do not reach for `mlua` here.**

⚠ Sourcing note: `wiki.play.eco` and `docs.play.eco` are Cloudflare-blocked, so this is reconstructed
from Strange Loop's own Steam dev blogs (strong) plus community guides (weaker) [V for the blogs].
The precise current 9.x–11 law/contract/store API surface is **[U]**.

### 5.3 Star Citizen (Quanta / StarSim) — the closest published intent, and the clearest schedule warning

**COPY (two engineering constraints).** (a) Agents must be stripped of pathfinding, animation and
geometry to reach 10⁵ scale — "hundreds of thousands of lightweight NPCs called Quanta … without the
overhead of things like pathfinding, animations, or even geometry" [V]. ⇒ **quanta are records in flat
arrays, never ECS entities with colliders.** (b) Dev tooling is a **first-class part of the system**:
a product-node graph, PvP heatmaps used to deploy security forces, and tunable variables like refinery
time and factory wages [V]. That is a strong hint for our dashboard's primary view: the **BOM DAG and
the min-cost-flow trade graph annotated with prices and tax incidence**, not a wall of candlesticks.
(c) The one non-obvious mechanism: **hysteresis on producer exit** (§4.2).

**AVOID.** Treating the agent layer as a prerequisite. The wiki page is tagged `{{Outdated}}` and its
live citation is a **roadmap deliverable** ("1169-StarSim", accessed 2025-11-12); the game crossed $1 B
crowdfunding over 14 years with no 1.0 date [V]. After ~8 years of public design the agent-based
background economy **has not shipped**. There is **no published price-formation algorithm** — the
Zurovec interview is philosophy with zero equations [V]. ⇒ **the aggregate closed-form layer must be
independently shippable and independently valuable; agents are strictly additive.** Also note the
reported "two orders of magnitude" gain from moving Quantum to C# [V] implies their bottleneck was
allocation/indirection — a Rust struct-of-arrays implementation starts from a much better place, which
is our genuine advantage.

### 5.4 Albion Online — the shipped proof that localized markets work

**COPY.** Per-city marketplaces not shared globally, so prices diverge and transporting goods between
them *is* the arbitrage gameplay (and the risk) **[L]**. Full-loot PvP destruction as a permanent item
sink **[L]**. The **Black Market** NPC: it buys player-crafted gear, mobs then *drop* those
player-crafted items as PvE loot, and if it lacks an item it posts a buy order and **gradually raises
the price** until someone fills it **[L]** — an adaptive market-maker that bootstraps a thin economy
and answers our cold-start / realm-wake problem. The **setup fee charged on every price EDIT**,
deliberately so that "players who babysit their orders all day" are not the only viable sellers **[L]**.

**AVOID.** Nothing structural, but note the documented complaint of record: regionally fragmented
markets **suppress trade volume** **[L]** — which is the exact tension our per-realm books must tune,
and it warns that per-realm books need a deliberate cross-realm arbitrage affordance.

⚠ All Albion fee figures (2.5% / 8% / 4%) are **[U]** — `wiki.albiononline.com` is bot-blocked [V].

### 5.5 RuneScape Grand Exchange — the anti-manipulation toolkit

**COPY.** Three decisions, all verified from `runescape.wiki` [V]: (a) **rate-limit by QUANTITY per
time per item per account** rather than clamping prices — the ±5% band was **removed on 2011-02-01**
in favour of per-item 4-hour buy limits, because a band creates a guaranteed arbitrage ladder while a
quantity limit does not; (b) make the tax a percentage (RS3: **2% since 2023-01-09**, rounded down, no
cap; OSRS is the capped variant) so it is a sink at the low end; (c) **publish an official price/volume
feed** and reserve the right to intervene manually — "We will only intervene as a last resort, and only
if we think price manipulation is going on."

**AVOID.** Manual overrides as untracked pokes. Operator overrides must be **first-class, fence-stamped,
audited economy events** so the dashboard's "apply tweak" is replayable and attributable.

### 5.6 Veloren — real Rust, real multi-site economy, and the one structural gem

**COPY (the gem).** **Price discovery is gossiped as a payload on the goods shipment itself.** The
three-phase protocol: `plan_trade_for_site` buys from the cheapest *remembered* neighbour price, paying
in barter goods, budgeted by a `Transportation` good that is itself produced by a merchant profession →
the supplier rations by `order_stock_ratio` and ships a `TradeDelivery` **carrying its own price and
supply vector** → the buyer `mem::swap`s that into `NeighborInformation.last_values/last_supplies`. So
**no shard ever queries another shard's market**, information is always one round stale *by design*
(which creates arbitrage gameplay), and it is exactly an `EffectFree` piggybacked field ⇒ **HR1-compatible
by construction, with no new arm and no request/response.**

**COPY (the shape).** The whole economy is ~2,000 ticks of 90 simulated days run **once at worldgen**,
rayon-parallel per site, with gameplay reading only a normalised price vector — which proves the sim
**need not be live**: a dormant realm can be caught up by N cheap ticks on wake, or run entirely
offline. Also the abstraction ratio: 15 abstract Goods for thousands of items, per-item price by
recipe expansion (§4.4).

**AVOID.** Its own source admits prices end up **nearly identical in every town** and lists the reasons
they did not fix it ⇒ if we adopt this family, **regional divergence must be engineered deliberately**
or there is no trade gameplay. Also f32 everywhere, `hashbrown::HashMap`, `lazy_static`, rayon — all
verified present [V] and all illegal for us.

**LEGAL.** GPL-3.0 [V]. Read and re-derive the design; **do not copy a function, a constant table, or
a data file.** Any port task must be specified as re-derivation from a written spec.

### 5.7 X4: Foundations — the cheapest credible station economy

**COPY.** `price = lerp(max, min, storage_fill_ratio)` with per-ware min/max bands **[L]**, behind a
genuine multi-stage production chain (raw → refined → component → product) moved by NPC traders. The
decisive property: **price is a pure function of a quantity we already persist**, so there is zero
economy-specific state to replicate, checkpoint, reconcile or restore on realm wake.

**AVOID.** With no memory it cannot express expectation, hoarding or speculation, and prices snap
immediately ⇒ pair with the smoothing/hysteresis discipline from §4.2.

### 5.8 Second Life — the definitive cautionary tale for player finance

**COPY.** The structural rule (which is what actually matters): **any yield-bearing instrument must be
SERVER-LEDGERED with the obligation enforced by the simulation** — a dividend is a scheduled, funded,
fence-stamped transfer with a solvency check, never a promise held by a player.

**AVOID.** Everything else. Ginko Financial collapsed with **L$55 M of withdrawal requests against
L$180 M of deposits**, its assets "primarily invested in either things of poor to no liquidity, or
virtual securities … trading at significantly under their purchase price", triggered by Linden Lab
restricting in-world **gambling** — an *unrelated* policy lever [V]. ⇒ **an operator tweak surface is a
systemic-risk instrument**: it needs staged rollout, simulate-before-apply on a forked snapshot, and an
immutable audit trail.

⚠ The frequently-quoted 69.7%/yr rate, ~US$750 k destroyed, and the 2008-01-22 "no interest without a
banking charter" rule are **[U]** — the cited page supports none of them [V]. Also note: real-money
cashout imports real-world legal exposure (§9).

### 5.9 Others, one line each

| Game | Copy | Avoid |
|---|---|---|
| **Entropia Universe** | A hard peg with real redemption (**10 PED = US$1** is the only verified figure [V]) as one legitimate currency-scope option. | Everything about the Calypso Land Deed equity design is **[U]** as cited — re-source before using it as the "stocks" template. Also: real-money withdrawal puts you squarely in financial-regulation territory. |
| **Dual Universe** | Per-location books with **remote order management but LOCAL physical settlement** — the closest existing space-MMO analogue to what sealed shards force, and it maps cleanly (order management = a Signal-class message; settlement = a Transfer-class saga). Three fees: flat per-order + value tax + per-day storage. **[U]** on the exact figures. | The unsolved **bot-to-player-market transition**: NPC market makers are essential to bootstrap, but removing them later is a live-economy migration with no rollback ⇒ decide the bot exit path at design time and make the parameters operator-tunable and reversible. |
| **Elite Dangerous BGS** | The non-farmable coupling rule: faction influence responds to **profitable volume in a scarce commodity**, not raw volume — so round-tripping does not farm it **[L]**. Proves economy→world-state feedback (security, stock, outfitting availability, NPC behaviour) is durable and fun. | Opacity. An unpublished BGS produced a decade of reverse-engineered guides and constant accusations of arbitrariness — the exact opposite of "fully analyzable". Publishing the model is also what makes it testable. **[U]** on details (the cited guide is unreachable). |
| **Path of Exile** | Friction as a deliberate lever: no in-game auction house, barter in currency items, because an efficient market would be flooded and league→Standard merges collapse rare prices [V]. | — |
| **Offworld Trading Company** | The *decision* to collapse bid/ask into a single price (spread pushed to the edge). | The quantitative claim is **[U]** (cited URL 404s). |
| **OpenTTD** | A closed-form **integer** haulage reward capturing distance, transit time and per-cargo perishability, with a floor-at-31 degenerate-case guard — bit-exact across hosts, trivially coverable, and shippable long before any price simulation exists. ⚠ **CORRECTION**: the wiki's `>>7 … >>13` form is historical; current `src/economy.cpp` computes `BigMulS(dist * time_factor * num_pieces, cs->current_payment, 21 + TIME_FACTOR_FRAC_BITS)` with a four-part time-factor curve and `MIN_TIME_FACTOR = 31` [V]. Re-derive from `economy.cpp`, not the wiki. Manhattan distance would be replaced by our tiered-`i64` `LatticePos` distance (D-41), itself exact-integer [R]. | GPL-2.0 ⇒ re-derive. |
| **Freeciv / Cataclysm-DDA / Anno** | Nothing. Surveyed for completeness: no reusable price-formation algorithm beyond what Victoria 3 / Endless Sky / Veloren already give [V]. Recorded so the search is not repeated. | — |

---

## 6. Correctness: moving value in a sealed-shard distributed world

### 6.1 The non-negotiables

**N1 — Money is not a CRDT, and the crisp reason is I-confluence, not "CRDTs are eventually
consistent."** Bailis et al. (VLDB 2015, *Coordination Avoidance in Database Systems*) give the
necessary-and-sufficient condition: a transaction set can execute coordination-free,
transactionally-available and convergent **iff** it is I-confluent w.r.t. the invariant. A counter CRDT
*does* converge on the value; the problem is that `balance ≥ 0` is a predicate over the **merged**
state that merge does not preserve — two replicas each locally deducting 5 from a balance of 8 are both
locally valid and merge to −2. The consequence is an **asymmetry we should exploit**: credits
(increments) are monotone and **are** I-confluent ⇒ they can ride at-least-once + dedup with **zero
coordination**; debits are not ⇒ they must pass a coordination point, and we have exactly one: the
directory CAS. This kills the temptation to make wallets a `FireAndForget` `EffectFree` feed (which
`EffectClass` would have permitted *syntactically*) and gives a principled rule for classifying every
economy flow arm.

**N2 — Double-entry, per currency, as a continuously-asserted global property.** Every committed
entry-set sums to zero **grouped by `CurrencyId`**, with ≥2 entries; a 2-entry cross-currency
transaction is a bug. Distribution changes the *checking*, not the rule: each shard reports a
fence-stamped partial tuple `(Σdebits, Σcredits, Σfaucets, Σsinks, Σpending)` per currency at a
`universe_tick`, and a reconciliation sweep sums them at a fence-consistent cut.

**N3 — Single-writer per account.** An `Account(AccountId)` row in the existing directory key space
(`DirectoryKey` is currently `Session | Entity | Realm | Ship` [R]) gives `verify_authority_unique` for
free.

**N4 — Escrow / two-phase, and NEVER compensate a posted credit.** Sagas (Garcia-Molina & Salem,
SIGMOD Record 16(3), 1987) undo a step **semantically**, "but do not necessarily return the database to
the state that existed when the step began" — and for money that caveat is the whole problem
("cannot un-pay"). Design rule: **every reversible leg must operate on PENDING (escrowed) funds only**,
whose compensation is `void_pending_transfer` — always possible, never negative. Mapped onto our FSM:
reserve during `Preparing`, CAS in `CommittingCas`, post in the `Swapping`/`Promoting` tail, void in
`Aborting` [R].

**N5 — Exactly-once VALUE transfer = at-least-once delivery + idempotent apply.** Exactly-once
*delivery* is an impossibility result (Two Generals / FLP), not an engineering gap. We already own the
full ladder at two altitudes: the wire layer (incarnation + seq + cumulative ack + reconnect replay +
`RecvLedger` high-water in `crates/io-prod/src/mesh.rs`) and the effect layer (`applied_steps` keyed on
`(TransferId, step_id)`, consult-before-effect / record-after-effect) [R].

⚠ **CORRECTION (revision 1 said "an economy needs ZERO new delivery machinery" — the durable half is
owed).** The delivery *ladder* exists; the **durable effect journal does not.** The idempotency journal
the claim rests on is IN-MEMORY at the shard: `crates/sim/src/stub.rs` `AppliedSteps` is a
`BTreeSet<(TransferId, u32)>` whose own doc says "the durable redb `applied_steps` table will use at P3
… (An in-mem backing cannot crash mid-step — the durable crash window is a P3 concern, D-22)" [R], and
**D-22 is 🟥** with the durable table explicitly owed (`docs/design/DEFERRED.md:2766`) [R]. Money is
precisely the workload that needs consult-before-effect / record-after-effect to survive kill-9, so
**the economy is the forcing consumer for that deferral.** Honest restatement: *the delivery ladder
exists; the durable `(TransferId, step_id)` effect journal is owed, and the economy forces it.* It is a
named **blocking dependency of D-48 and slice E-1** (§10.4), and N6's receipt schema must be designed
into that same table so it is designed once.

**N6 — Idempotency must return a stored RECEIPT, not just `AlreadyApplied`.** Stripe's contract is the
reference: the first request's status code **and body** are stored and replayed for ≥24 h, and a retry
with the **same key but different parameters is an ERROR, not a replay** [V]. Our `applied_steps`
answers a boolean today; for economy it must store `{outcome, resulting_fence, balances_after}` so a
client that retried, reconnected or re-homed can be told authoritatively what happened. **That IS
read-your-writes without prediction.** ⇒ **fold N6 into the durable `applied_steps` schema from N5**;
two separate designs of the same table is how the receipt half gets dropped.

**N7 — The CLEARING-ACCOUNT rule: value in flight must always be located in exactly one ledger
account, never "in the wire."** A value saga moves source → `InTransit(TransferId)` (a real account
owned by exactly one shard) → dest. This is what makes the conservation invariant checkable at **every**
tick, including mid-saga and mid-crash, rather than only at quiescence — and a stranded `InTransit`
balance with no live saga becomes a directly detectable, directly repairable anomaly (the value analogue
of the orphan-lock sweep D-6 S3 already performs on directory `in_transfer` locks) [R].

**N8 — Fences everywhere; the directory CAS is the only commit point; postcard v1; TLV-framed blobs;
decode-to-Default BANNED for Durable kinds** [R]. A wallet crossing shards is a TLV-tagged field of the
player blob under the existing version-floor handshake, so a missing REQUIRED tag is a hard
`ABORT_SPATIAL` (the player stays alive on the source) rather than a silently zeroed wallet — i.e. **a
rolling deploy can never zero a wallet.**

**N9 — Escrow rights, not derived authority.** If a player must transact on a shard far from their
wallet's owner, use O'Neil's Escrow method (ACM TODS 11(4):405–430, 1986) in its modern CRDT form, the
**Bounded Counter** (Balegas et al., SRDS 2015): partition the global slack into per-replica **rights to
decrement**, so a local debit is safe because it consumes only locally-held rights, and only *exhaustion*
requires coordination. Note what this is architecturally: **authority is HELD, not derived** — precisely
the fix `generic_transfer.md §A2` already made for transient authority anchored to a realm-lease `Fence`
[R]. Rights transfer is itself a value transfer riding the same saga.

⚠ **A hole revision 1 left open: rights have no recovery on permanent holder death.** If a
rights-holding shard is permanently killed (the D-37 scenario the report elsewhere relies on), the
rights are lost. The stated bound `Σ rights ≤ slack` **still holds** — but the owner's spendable balance
is permanently reduced with no declared sink, i.e. **value destroyed while every stated invariant
passes.** Rights are not entities, so D-37 forward re-home does not recover them. Two mechanisms, both
cheap, and at least one is mandatory:

1. **Revocation on confirmed-dead evidence** — reuse the `is_confirmed_dead` / CA-1 re-solicit probe
   pattern already built for `AwaitAdopt` liveness [R]; the account owner reclaims the grant.
2. **`expiry_universe_tick` on every grant** — after which the owner reclaims unilaterally, no
   coordination and no liveness question. Same rule S9 already applies to resting orders; strictly the
   safer of the two and it composes with (1).

⇒ **I13 is strengthened** to `Σ rights outstanding + Σ reclaimable == slack`, so a lost grant is a
*detectable anomaly* rather than a silent write-off, and rights appear in the I3 sum.

**N10 — Reuse §A5's additive-effect cut verbatim.** `generic_transfer.md §A5` already solved the exact
hard case for damage: the handoff blob carries the authoritative post-drain value **AND**
`applied_damage: BTreeSet<DamageEventId>`; at FLUSH the source drains and folds every event with
`hit_tick < freeze_tick`; events at or after the cut route to the dest only; the CAS fence / `freeze_tick`
IS the cut; a redelivered event dedups against the inherited set [R]. **Money is additive in exactly
the same way** — a wallet crossing shards is `applied_damage` with a different name (`balance_posted` +
`applied_txn_ids`, bounded and pruned by age; see §7.6 for the per-entity byte budget, which revision 1
did not bound). `BTreeSet`, not `HashSet`, because it crosses into the byte-identical-replay surface —
and §A5 says so itself, verbatim: *"NOT `HashSet` — the sim/node clippy `disallowed-types` bans the
default-hasher `std::collections::HashSet`, and this set CROSSES the transfer barrier into the
byte-identical-replay surface"* (`docs/design/generic_transfer.md:165`) [V].

⚠ **Errata for `generic_transfer.md` itself** (surfaced by the review, worth fixing independently of the
economy): the §A5 **spec** at line 165 says `BTreeSet` with that note, but the older
adversary-response **summary** at line 250 of the same file still says `applied_damage:
HashSet<DamageEventId>` [V]. The spec is right and the summary is stale; a stale `HashSet` on a
byte-identical-replay surface is a live foot-gun for the next reader, so line 250 should be corrected.

### 6.2 A concrete gap: the `Store` seam has no point read

`crates/sim/src/io/mod.rs:417` — `Store` is `put` / `delete` / `scan(prefix)` / `commit` / `flush`.
There is deliberately **no `get`** ("no flow point-reads — dead Tier-A surface") [R]. A ledger
fundamentally wants "read account X's balance."

Two honest options; the second is better aligned:

| Option | Cost | Benefit |
|---|---|---|
| Add `get` to the seam | A new code path in **both** `MemStore` and the redb backend, plus HR5 coverage on both | Direct point reads |
| **Keep the authoritative ledger fully in RAM** as a deterministic state machine, journal every committed entry via the per-tick group commit, and rebuild **from a SNAPSHOT + a truncated journal tail** at rehydrate | Bounds the ledger by RAM; needs an explicit per-shard account/order budget in `EconomyTuning` **and a snapshot** — see the ⚠ below | **No seam change** for the ledger itself, Tier-A stays pure, and audit-replay (I12) is free. This is exactly LMAX (in-memory event-sourced processor + journal + snapshot) and exactly what our `rehydrate` already does for sagas/directory/batch-gos [R] |

⚠ **Correction to revision 1's costing of option 2.** It said only "`rehydrate` becomes O(ledger) at
startup ⇒ needs an explicit budget", and deferred snapshotting to D-54 "before the economy carries real
volume". That is too late, for two reasons: `Store::scan` returns a **fully materialised
`Vec<(Vec<u8>, Bytes)>`** [R], and the corrected authoritative ledger is **0.5–2.4 TB/yr** (§2.2). An
economy realm whose boot cost scales with its *lifetime* ledger makes RLM's `boot_ticks` unbounded —
and F7 already downgraded AOI-1 to "a generator sizing constraint under bounded `boot_ticks`" and calls
the warp-arrival hitch load-bearing (`scripts/realm_lifecycle_design.md:161`) [R]. So:

- **Snapshot + truncate is part of E-1**, not of D-54's later retention work: a periodic ledger snapshot
  under its own key (`econ/snap/<lsn>`), with the journal replayed **only from the snapshot LSN**.
- **`EconomyTuning` bounds the snapshot**: `snapshot_period_ticks`, `max_snapshot_bytes`,
  `journal_retention_ticks` — and `max_snapshot_bytes` must be ≤ the per-shard PVC share (§2.3).
- **A gate asserts rehydrate stays inside the RLM `boot_ticks_p99` budget** at
  `max_orders_per_book` + `max_accounts_per_shard`, using the release-only named-budget latency pattern
  in `crates/harness/src/latency.rs` [R].
- ⇒ **D-54 and D-6's owed WAL retention/compaction become blocking dependencies of E-1**, not of "before
  real volume".

### 6.3 Authority-placement options

| | **A — accounts by owner, markets by (realm, commodity)** | **B — one never-dormant economy-capability shard** | **C — hierarchical: per-station books + regional aggregator** | **D — event-sourced single sequencer** |
|---|---|---|---|---|
| **Shape** | `Account(AccountId)` + `Market(RealmId, CommodityId)` become directory key families with OwnerRecords + fences; matching runs in the market key's owner; **escrow locus is D15, NOT pre-decided here** (revision 1 wrote "escrow held on the ACCOUNT owner" while §7.2 simultaneously assumed the venue holds it — see §6.4 rule 1); if escrow sits at the account owner a fill emits a value saga per (buyer, seller) pair, batched like `TransientGo`, and if it sits with the book a fill is a local mutation | One `ShardProfile` instance owns ALL accounts, markets and the journal; every other shard talks to it | Books at the deepest realm (station/area); a regional aggregator maintains a merged view; a galaxy index answers "where is X cheapest" | Not a competing topology — the **internal** architecture all of A/B/C should use |
| **Throughput ceiling** | Matching free; binding limit is the per-shard durable write budget ≈ **13,000 postings/s desktop / 3,000–7,000 on k3d storage** (§2.2, rebuilt), ×N shards | Same **per shard**, but now that is the *whole universe's* ceiling rather than one shard's — and empirically this **is** EVE's architecture (one SQL Server pair behind 195 nodes at ~40 k CCU / 250 M txn/day) [V]. TigerBeetle's own position paper explicitly **rejects** sharding for OLTP ledgers (it destroys cache locality and strict serializability), recommending RSM + LSM + object storage — "diagonal scaling" [V] | Unbounded horizontally (per-station books are independent single-writers) | — |
| **Agent capacity** ⚠ *new row, and it is decisive* | **5×10⁶ evals/round spread over 10³+ shards = 0.0075% of a core each** — free | **1–2 FULL CORES of agent work on one single-threaded sequencer**, before any player traffic; and the escape hatch (agents in realm shards, books here) costs 25–500 MB/s of reviewed-taxonomy egress ⇒ **infeasible on both branches** (§2.3c) | Same as A | — |
| **Failure behaviour** | Identical to today's transfer: fence + CAS + WAL re-drive. ⚠ **BUT a dead market owner is NOT a working re-home today** — see the ⚠ below | A single availability domain — if the economy shard is down **all trade stops**, but nothing is lost (WAL + fence) | A station book is independently recoverable; the aggregator is a rebuildable projection, so losing it is not a value loss. ⚠ Same `ReHomeState` blocker as A | Deterministic replay ⇒ microsecond failover between identical-input replicas |
| **Latency** | One saga = a few ticks (50–150 ms), invisible for a market order | ⚠ **CORRECTED**: "best possible (a local function call)" is true **only for actions occurring ON the economy shard.** Every mining credit, fee, fill and tax initiated by gameplay *elsewhere* becomes a per-action side-effecting cross-shard command through the reviewed taxonomy — one saga each, the same 50–150 ms, **plus** the egress and the idempotency journal cost, **plus** it collapses the "at most one new arm" claim | Local trades fastest; cross-region pays a saga | — |
| **RLM interaction** | Markets keyed by realm ⇒ a non-running realm's market must be hosted by the nearest live ancestor, **or** "has open orders" must be an RLM demand signal (which is the honest answer, and it is a small `LiveFabric` input) | ⚠ **NOT trivial — there is no way to spawn or protect such a node today.** The reconciler is documented the **SOLE kill authority** over a DESIRED live-set that is the *ancestor-closure of realms* (`crates/node/src/rlm_spawn.rs:141`, `scripts/rlm_step3_reconciler_spec.md:301`) [R], and spawning goes through `spawn_realm(coord)` keyed on a **`RealmCoord`** [R]. A non-spatial economy shard is therefore either a **synthetic realm injected into the forest** or a **second parallel lifecycle authority** — revision 1 acknowledged neither | **Worst case** — station realms are exactly the ones RLM kills ⇒ every book must persist and rehydrate byte-identically, and escrow must never be held there | — |
| **HR conformance** | Maximally conformant (a capability + directory key families) | HR3-legal **only** if implemented as a normal `ShardProfile` capability and proven by G-IDENTICAL on ≥2 shard kinds — otherwise it becomes a *distinguished* shard and invites HR3/HR4 violations | The natural realm shape and the natural tax jurisdiction | Compatible with all |
| **Weakness** | Multiplies directory keys (millions of accounts) against a **single-writer** orchestrator directory — which `transfer_protocol.md §4.6` already flags as the D-32 range-partition-by-region blocker [R]. Relieved by N9 (an account needs a CAS only when its rights are exhausted, not per trade) | Re-centralises the hotspot; a gameplay-critical single domain | RLM dormancy bites hardest | — |
| **Empirical support** | — | EVE, TigerBeetle | EVE **also** splits the market by region and pins the hot one: a dedicated node for Jita and a **second dedicated node for The Forge market** out of 195 [V] | LMAX; every exchange ("price-time priority is inherently serial at the symbol level", industry practice being one matcher per symbol) |

⚠ **The `ReHomeState` blocker (it invalidates the failure row of both A and C as written).** Revision 1
wrote "a dead market owner is a re-home (D-37 machinery exists) [R]". **D-37's adopt cannot carry
economic state today.** `ReHomeCmd.state: ReHomeState` has exactly **one** arm —
`ReHomeState::PoseOnly(StampedPose)` — and its own doc says "POSE-ONLY today … the P7 checkpoint slice
grows a `Snapshot(Vec<u8>)` TLV-blob arm ADDITIVELY" (`crates/wire/src/intershard.rs:618–627`) [R].
D-31 is the same hole on the crossing side (`StubCrossing.state` is produced as an opaque `vec![]`) [R].
So re-homing an `Account`/`Market`/book subject after a permanent kill today **reconstructs a pose and
silently loses every balance and every resting order** — value destruction on the exact recovery path
revision 1 cited as the availability answer. Consequences, now stated:

- **`ReHomeState::Snapshot` (P7) and `TransferableKind::serialize` (D-31) are named, blocking
  dependencies of D-48/D-53** (§10.4).
- **Options A and C are BLOCKED on them.** Only option B — where accounts never re-home because they
  never moved — is reachable before P7. That is the one genuine argument left for B, and it is a
  *sequencing* argument, not an architectural one.
- **New invariant (I18):** a re-home whose subject holds ledger state and whose `ReHomeState` arm cannot
  carry it must **FAIL LOUD**, never adopt-to-default. (decode-to-Default is already banned for Durable
  kinds [R]; this makes the *absence of an arm* equally loud.)

**Recommendation (see D2/D3 in §11) — INVERTED IN REVISION 2.** Implement **D inside A**: key
`Account(AccountId)` and **`Market(RealmId, CommodityId)`** as directory key families from day one, with
in-memory event-sourced ledger + journal internals, an `economy`/`venue` capability on the normal
`ShardProfile` lattice (plus the per-realm override, §7.5), and a G-IDENTICAL fixture from day one.
**B then becomes a deployment CONFIGURATION of A** — a `ShardProfile` instance that happens to own many
market keys — which is reversible; **A→B is a config change, B→A is a rewrite.** Two independent
constraints force this direction:

1. **Agents.** Books must be co-located with the agents that quote into them (§2.3c). B's two branches
   cost >100% of one core or ~25–500 MB/s of reviewed-taxonomy egress.
2. **Lifecycle.** B has no representation in the RLM desired-set without either a synthetic realm or a
   second lifecycle authority; A's markets are realms, which the machinery already spawns and protects.

`Market(RealmId, CommodityId)` also **closes the hot-hub question for free** (formerly OQ #2): one
venue's books are already partitioned across K single-writer authorities by commodity, so the hottest
*venue* is never the hottest *writer*. See §12.1.

**If the user prefers B anyway** (a legitimate choice for the pre-P7 window, given the `ReHomeState`
blocker), then B must specify — and revision 1 did not — (a) how the economy node is spawned and
protected inside the RLM desired-set, (b) its command arm(s) and their idempotency keys, (c) the
per-gameplay-action cross-shard rate, and (d) a re-derived latency row for actions **not** on the
economy shard.

### 6.4 The RLM-dormancy problem

⚠ **WORLDLINE (rev 2) — ONE REQUIREMENT HERE IS A DEFECT, AND THE CATCH-UP TIER IS DELETED.** See
`scripts/dormant_world_simulation_design.md` §3.4 and §4.7.
1. **Requirement 2 ("has resting orders / holds escrow" as an AoI/hysteresis input to keep a market realm warm)
   is a LAW-E1 VIOLATION and must not be built.** It makes *which realms are alive* a function of economic
   state — the deepest available violation, since the RLM reconciler is the **sole kill authority** and
   `desired_alive` is a pure function of a demand ledger with no store or `World` access. **LAW-WL-7**: no
   economy value, no life-tier value, and no economy-derived boolean may enter `desired_alive`, `teardown_ready`,
   the `aoi_decide` occupant set, or the `AoiMembership` map. Machine-proved by `G-WL-LIFECYCLE-BLIND` (toggle the
   economy and the life tier; the `LifecycleAction` trace must be **bit-identical**). **The legal alternative**,
   if "keep a busy venue warm" is genuinely wanted: a per-realm **`min_dormant_ticks`** — a *time* hysteresis in
   `RlmTuning`, charged to the game's lifecycle tuning and economy-blind — never an obligation-driven KeepAlive.
   The deeper answer is that the state must simply be **correct across an arbitrary reap**, which replay-from-base
   gives for free.
2. **The dormant catch-up tier (item 4 below) is DELETED as a primary mechanism.** Its own fallback when
   `max_catchup_ticks` is exceeded is *"adopt the pure closed-form field rather than replay"* — i.e. it needs the
   closed form **regardless** — so keeping only the fallback removes the ~52-second wake problem instead of
   bounding it, and deletes two tuning fields and a cliff behaviour. `dormant_catchup_tick_period` and
   `max_catchup_ticks` shrink to a fallback-only knob for the optional coarse-agent tier's abnormal gaps.
3. **The storage-topology blocker below dissolves for the PHYSICAL layer** (the worldline state is four orders of
   magnitude smaller than `journal_bytes_per_realm_budget`) and for **account positions** (zero-rate subjects at
   the custodian, §6.7). It stands for **live order books**.

**The problem.** A market book with long-lived resting orders, a corporate wallet, an escrow, and a
courier-contract deadline must remain live and correct while nobody is present. RLM spins realm shards
up and down by per-realm AoI; LF-1 states a realm is at-least-Dormant **iff** it is an ancestor of (or
equal to) some Active leaf, and killable only if zero occupants AND zero live descendants AND no live
KeepAliver AND declared-parent-live [R]. Note the current implementation status: "v1 runs a full server
per realm even when unoccupied" — the *Dormant-as-a-cheaper-capability* optimisation is itself deferred
[R]. So today the exposure is **kill**, not a partial-capability dormancy.

**What the ancestor-closure invariant gives us — restated honestly.** ⚠ Revision 1 said the up-path is
"guaranteed at-least-Dormant **with the `signal_relay` capability**", and leaned on that in five places.
LF-1 guarantees **at-least-Dormant**, full stop. `signal_relay` is carried by `profiles::galaxy()` and
`profiles::station()` only — `system()`, `planet()`, `ship()` and `area()` have it **false**, with a unit
test asserting exactly that [R] — and `sealed_shards.md:147` documents it as the *Galaxy-Relay
radio-subscriber* capability, not a general hierarchy-routing capability [R]. What a **Dormant** realm can
*process* is also undefined (that optimisation is itself deferred; "v1 runs a full server per realm even
when unoccupied" [R]). So the correct premise is:

> The up-path realms are **at-least-Dormant** (LF-1). **Whether each can RELAY or PROCESS an economic
> Signal is an OPEN REQUIREMENT** to feed into the P9 Signal design (S10), and possibly into
> `profiles::system()`/`planet()`.

Consequently the tax-remittance design does **not** get to assume relay capability on the chain — which
is one of the two reasons T8 was rewritten to push-with-retention (§4.8).

**Candidate answers:**

| Option | Mechanism | Reuses | Risk |
|---|---|---|---|
| **(A) DURABLE-DORMANT** | The realm's redb store survives the kill; the book is re-adopted on spin-up exactly like an orphan. Deadlines falling while down are evaluated on wake against `universe_tick` (closed-form ⇒ safe). | The `/whoami` cookie-probe + launch-ledger rehydrate pattern from RLM 5e-4/5e-5 [R]; `dingir-exchange`'s op-log + fork-and-save snapshot as the reference shape | ⚠ **BLOCKED on a storage-topology prerequisite** — see the ⚠ below. Also: the rehydrate gate must be sized at `max_orders_per_book` (**400 k–1.5 M orders for a hub**), not revision 1's 40 k |
| **(B) DELEGATED EXCHANGE AUTHORITY** | A permanently-resident `ShardProfile` with the economy capability that non-running realms delegate to (EVE's four market nodes, generalised). | Existing profile machinery; HR3-clean (a capability, not a kind) | Re-centralises the hotspot we sharded to avoid |
| **(C) ANCESTOR ESCALATION** | A dormant realm's economic state is held by its nearest ACTIVE ancestor. | LCA routing + `closure_peers`; degrades gracefully | The owning authority **moves**, so every order carries a `Fence` and re-homing a book is itself a transfer saga — most elegant, most novel risk |

⚠ **The storage-topology prerequisite (option A is not implementable on the current topology).**
Revision 1 correctly ledgered RLM Step-4 persistence (D-53) as owed, but never mentioned the *volume*
question, so (A) read as buildable on existing machinery. Verified against the manifests:
`deploy/k3d/50-shard.yaml` gives each shard `volumeClaimTemplates` with `accessModes: [ReadWriteOnce]`
and `storageClassName: local-path` — a **node-local** volume, **256 Mi**, bound to the StatefulSet
**ordinal** rather than to a `RealmId` [R]; the durable root is `/var/lib/vd`
(`crates/io-prod/src/boot.rs`) [R]; and RLM spawns realm shards as child processes on a node, so a
realm's `econ/` state lands on **that node's** disk. Therefore:

- **G4 cannot pass as written** ("bring it up ON A DIFFERENT NODE; assert I11 byte-identical
  rehydration") — a realm respawning on a different node cannot reach its previous store.
- **Realm→PVC identity is by pod ordinal**, so a respawned realm does not deterministically re-attach
  its own book/escrow.
- **256 Mi cannot hold** a hub book + escrow + journal even at revision 1's own volumes, let alone the
  corrected ones (§2.2).

**Blocking prerequisite added to D-53 — pick one:** (i) RWX / networked storage with a **RealmId-keyed
volume identity**; (ii) an explicit **state-handoff step in the RLM spin-down/spin-up saga** that ships
the `econ/` prefix to the new host (40–80 MB for a hub ⇒ ~0.3–0.6 s at 1 Gbit/s — budget it, and it must
be fence-stamped and idempotent like every other handoff); or (iii) restrict economic state to realms
**pinned to a node**. **Until one is chosen, D3 defaults to (B) or (C), which inverts revision 1's D3
recommendation.** Shard PVC size must also become `f(max_orders_per_book, journal_retention_ticks)`.

**Four requirements that hold under ALL THREE options and must be designed, not discovered:**

1. ⚠ **Escrow LOCUS — revision 1 asserted two mutually exclusive things and this is now decision D15.**
   §6.4 said "escrow is held by the ACCOUNT owner, never the market host … the market holds only a
   *reference*"; §7.2 then claimed the single biggest simplification, "with 100% escrow both sides'
   assets are already held by the venue's ledger authority ⇒ a fill is a local mutation, not a saga."
   **Those are mutually exclusive unless account owner == venue** (i.e. only under D2 option B). Under
   the recommended option A, if escrow stays with the account owner then **every fill becomes a
   cross-shard escrow post**, which changes the throughput budget, the per-fill idempotency requirement
   and the E-7 slice shape. **Resolve before E-1 freezes the escrow schema:**
   - **(i) Escrow co-located with the BOOK.** A fill is a local mutation — the big simplification
     survives. **Cost:** a killed market realm must be *proven* never to strand locks (a real gate:
     teardown-with-live-escrow → assert every reservation reaches exactly one terminal), and the escrow
     travels with the book under the D-53 handoff.
   - **(ii) Escrow at the ACCOUNT owner.** A killed market realm can never strand value. **Cost:** delete
     the "fill is local" simplification and re-derive the throughput budget with **one saga (or one
     batched go-token) per fill** — at the hub's 2.3–3.0 trades/s mean that is affordable, but it is a
     different design and a different slice.
   **Do not carry both.** (A defensible hybrid: escrow at the book for *same-realm* actors, at the
   account owner for remote actors — but then the fill path has two shapes and must be gated as such.)
2. **The book is a `Store`-backed Durable TLV table with an explicit schema (never decode-to-Default),
   serialised on teardown, with its existence recorded so the orchestrator knows the realm has open
   economic obligations** — which is also the natural place to make "has resting orders / holds escrow"
   one of the AoI/hysteresis inputs that keeps a market realm warm.
3. **Expiry is mandatory** — every resting order (and every escrow-rights grant, per N9) carries
   `expiry_universe_tick`, so a realm never respawned cannot hold escrowed value forever, and expiry is
   correct on the next wake regardless of wall clock.
4. ⚠ **The dormant-tier catch-up must be structurally bounded** — revision 1's "a dormant realm can be
   caught up by N cheap ticks on wake" is unbounded as written: 30 days at 20 Hz is **51.8 M ticks =
   52 s of blocking CPU on wake** even at an optimistic 1 M ticks/s, which violates both the RLM
   hysteresis budget and the seamlessness hard rule if a player arrives mid-catch-up. The fix is the
   cadence hierarchy (§2.3b): the dormant tier ticks at `dormant_catchup_tick_period` (1 universe-hour
   ⇒ **720 ticks for 30 days ⇒ microseconds**), with a hard `max_catchup_ticks` and a **defined
   behaviour when exceeded**: adopt the pure closed-form field rather than replay.

**Sizing the rehydrate gate honestly** (and the conclusion is *stronger* than revision 1's): at
TLV-decode + `BTreeMap`-insert ~0.5–1 µs/order, a **400 k**-order hub book rehydrates in **0.2–0.4 s**,
and the 1.5 M worst case in **1.5–3 s**, plus a 40–80 MB prefix scan (~0.05–0.5 s) ⇒ **~0.3–1 s
incremental spin-up for a hub, ~2–4 s worst case.** That is *acceptable* and hidden by RLM's
warm-before-reach — but the gate must be **sized at `max_orders_per_book`** and published as a
release-only latency budget in the SPIKE-3a pattern, not written against an arbitrary 40 k.

### 6.5 The invariant + gate list (harness-shaped)

**Invariants** (each checkable by a function over ground truth, not over rendered bytes):

| ID | Invariant |
|---|---|
| **I1 DOUBLE-ENTRY-ZERO** | Every committed entry-set sums to zero per `CurrencyId`/`ItemKind`, ≥2 entries, no 2-entry cross-currency |
| **I2 NO-NEGATIVE** | For every constrained account, `posted − pending ≥ 0` at every committed instant |
| **I3 CONSERVATION** | `Σ_shards (posted + pending + clearing) == Σ declared faucets − Σ declared sinks`, per currency, at any fence-consistent cut |
| **I4 VALUE-LOCATED** | In-flight value sits in exactly one `InTransit(TransferId)` account owned by exactly one shard (so I3 holds mid-saga) |
| **I5 FAUCET/SINK-DECLARED** | Every mint/burn cites a `FaucetId`/`SinkId` from a static registry; an undeclared change to the global sum is a hard failure |
| **I6 SINGLE-WRITER** | Exactly one shard holds `Owned` for each `AccountId`/`Market` key at every committed instant (reuse `verify_authority_unique`) |
| **I7 EXACTLY-ONCE-APPLY** | At most one ledger effect per idempotency key; redelivery returns the **stored receipt** |
| **I8 ESCROW-CONSERVATION** | Every reservation has exactly one terminal (partial-post with remainder returned \| void \| timeout-void); pending buckets equal the sum of live reservations |
| **I9 MONOTONE-FENCE** | No economic state transition accepts `fence < highest_seen` |
| **I10 NO-DUPE** | Every `ItemId` is in exactly one container at every committed instant; split/merge conserve amount per `ItemKind` |
| **I11 ORDER-CONSERVATION** | `escrow_remaining + filled == original commitment` for every open order; a realm teardown/spin-up rehydrates the book **byte-identically** |
| **I12 AUDIT-REPLAY** | Replay from genesis/snapshot reproduces every balance bit-identically; two runs under one seed produce identical traces |
| **I13 RIGHTS-BOUNDED** ⚠ *strengthened* | (if N9 is adopted) `Σ rights outstanding + Σ reclaimable == slack` (**equality**, not `≤`), and rights appear in the I3 sum ⇒ a grant lost to a permanently-dead holder is a detectable anomaly, not a silent write-off (§6.1 N9) |
| **I14 TAX-ALGEBRA** | Depth-invariance; zero-rate-insertion invariance; `Σ shares == total`; `LCA(a,a) ⇒ tariff ≡ 0`; BOM-stage-count invariance |
| **I15 CONTRACT-CLOSURE** ⚠ *new* | Every contract reaches exactly ONE terminal (completed \| expired \| failed \| cancelled), and its collateral has exactly ONE destination; no contract is simultaneously accepted by two haulers (§7.9) |
| **I16 FAUCET-BUDGETED** ⚠ *new* | Every `Faucet` arm carries a per-epoch, per-realm-subtree **budget** in `EconomyTuning`; the source shard **refuses** (fails loud) a mint that would exceed it, and the reconciliation sweep asserts each arm's total ≤ its budget **independently of the net identity**. Without this, a correctly-declared mint passes I3 **and** I5 forever (§4.9) |
| **I17 SELF-REPORT-CROSSCHECK** ⚠ *new* | A shard's reported partial tuple must be **reconstructable from its own emitted journal** (the exporter's copy), so a lying or buggy shard is detectable from OUTSIDE the shard. Under HR1 this is the only external check available |
| **I18 LEDGERED-STATE-CARRIED** ⚠ *new* | A re-home / crossing whose subject holds ledger state and whose `ReHomeState`/`serialize` arm cannot carry it **FAILS LOUD**; adopt-to-default is forbidden (§6.3) |
| **I19 ACCOUNT-ALWAYS-HOMED** ⚠ *new* | Every `AccountId` has exactly one live `Owned` holder **even when its owner has zero sessions and their last realm is dead** (§7.10) |
| **I20 LEDGERED-KIND-LOSSLESS** ⚠ *new* | Any kind carrying a ledger position is `Durable`/`LossBudget::ZERO`, **or** its loss posts to `Sink::TransientLoss` at the moment the loss is accounted (§4.9.1) |
| **I21 DIRTY-SET-SOUND** ⚠ *new* | A book absent from the clearing dirty set is provably unchanged since the last batch (§4.1) |

**Gates.** ⚠ **Revision 2 adds a wall-clock budget column**, because revision 1 specified 11 new gates
with no time budget in a repo where `just gate` is a **human-run local pre-merge step with no CI by
policy**, already running fmt + 3 clippy passes + `cargo test --workspace` + **10** named integration
gates (client-load, orch-crash, spike2a, spike3a, rlm-soak, 3 render smokes, node-per-realm-walk,
rlm-proc-spawn, rlm-kill9) + **3 separately-instrumented llvm-cov builds** over 9 Tier-A crates ≈ **71 k
LOC** and ~2,100 test fns [R, measured]. Adding a 10th Tier-A crate of the most branch-dense code in the
codebase plus 11 gates is a **schedule input**, not a footnote.

**Gate tiering** (also new): an **inner loop** (`coverage-fast` + unit + G-ACCOUNTING + G1) and a
**slower pre-merge tier** (the process/kill-9/RLM cells + G7). A `FaultFabric` proptest at proptest's
default 256 cases over a 12 k-tick ledger scenario is minutes-to-hours unless capped, so **every
proptest gate carries an explicit `PROPTEST_CASES` and a checked-in `proptest-regressions` corpus** — the
repo already uses one (`crates/core/proptest-regressions/tlv.txt`) [R].

| ID | Gate | Tier | Wall-clock budget |
|---|---|---|---|
| **G-ACCOUNTING** | A 12 k-tick harness run must close the books to zero: for every currency and jurisdiction, `Δ money_supply == Σ faucets − Σ sinks`. **No economy feature lands without this.** | inner | **≤20 s** (virtual clock; 12 k ticks is milliseconds of clock, the cost is the postings) |
| **G1 model-based** | `proptest-state-machine` over generated interleavings; reference model = `BTreeMap<AccountId, (posted, pending)>` for the ledger, brute-force matcher for the book | inner | **≤60 s**, `PROPTEST_CASES=64`, regressions corpus committed |
| **G2 bank test** | A `p3_economy_bank` scenario under `FaultFabric` drop/dup/reorder/partition asserting I2 + I3. The canonical adversarial form: `jepsen.tests.bank` transfers random amounts, forbids negative balances, and after **every read** asserts the total equals the initial amount; a partially applied transfer makes the sum drift. It found real anomalies in CockroachDB, Dgraph and YugabyteDB [V]. Reimplement as a scenario — our deterministic in-process harness is strictly better than driving a real cluster. Money is `LossBudget::ZERO`, so unlike the transient invariants there is **no budget arm** | pre-merge | **≤180 s**, `PROPTEST_CASES=32` over a **2 k**-tick scenario (not 12 k — 256 default cases × 12 k ticks is minutes-to-hours) |
| **G3 kill-9 cells** | Kill the account owner mid-reserve; kill the market host between match and settle; kill the orchestrator in `CommittingCas` of a value saga; kill the dest after adopt before ack — all reuse `Fault::KillRebuild` [R] | pre-merge | **≤240 s** for 4 process cells (same family as `rlm-kill9`, the existing slowest gate) |
| **G4 RLM teardown cell** | Tear a market's realm down with open orders and live escrow, bring it up on a different node; assert I11 + no stranded escrow + no double-fill. ⚠ **Sized at `max_orders_per_book`** (400 k–1.5 M for a hub), **and blocked on the D-53 storage prerequisite** (§6.4) | pre-merge | **≤120 s**; the rehydrate assertion itself budgets **≤1 s** (400 k) / **≤4 s** (1.5 M), release-only |
| **G5 G-IDENTICAL** | ⚠ **Now concretely specified** (revision 1 left it as OQ #10 while D2's only HR3/HR4 safeguard depended on it): the **same** `place-order → match → settle → tax → conservation-check` scenario, driven by the identical fixture, on a **station realm** and on a **ship realm**, asserting **identical ledger traces modulo ids** (normalise `AccountId`/`OrderId`/`RealmId` through a canonical renaming, compare the posting sequence byte-for-byte). **Lands in E-1, not E-8** — the fixture is what keeps the capability honest while the code is still small | inner | **≤30 s** |
| **G6 dupe chaos** | Item transfer crossed with kill-9 and with a redelivered batch; assert I10 | pre-merge | **≤120 s** |
| **G6b transient-loss conservation** ⚠ *new* | Kill the dest of a batch of **value-bearing dropped stacks**; assert `Δ money_supply == Σ faucets − Σ sinks` still closes with the abandoned value under `Sink::TransientLoss` and nowhere else (§4.9.1, I20) | pre-merge | **≤60 s** |
| **G7 load** | A 10 k-order burst asserting a DURABLE-UNAFFECTED-BY-BURST analogue plus an fsync-p99 budget, using the SPIKE-3a latency-gate pattern in `crates/harness/src/latency.rs` [R]. ⚠ **Must SHARE spike3a's existing release build** rather than force a second one | pre-merge (release) | **≤90 s** on top of spike3a's existing build |
| **G7b exporter backpressure** ⚠ *new* | Fill the `EventSink` ring / stall the exporter and assert **fail-loud** (the economic action is refused) rather than a dropped posting (§8.1) | pre-merge | **≤45 s** |
| **G7c rehydrate budget** ⚠ *new* | Rehydrate a max-budget ledger + book and assert it fits the RLM `boot_ticks_p99` budget (§6.2) | pre-merge (release) | **≤60 s** |
| **G7d agent tick cost** ⚠ *new* | Measure `vd-econ`'s agent round inside a real `step_tick` against `agent_eval_budget_per_tick`; seeded by the E-(−1) spike | inner | **≤30 s** |
| **G7e tax-path microbench** ⚠ *new* | `path_index(situs)` + composition + Hamilton split at 10²–10³ taxable events/s | inner | **≤20 s** |
| **G9 direct-trade chaos** ⚠ *new* | Kill either side of a P2P trade **between confirm and post** (§7.8); assert exactly one terminal per reservation (I8), no partial swap (I1), conservation across the pair (I3) | pre-merge | **≤90 s** (2 process cells) |
| **G10 contract-deadline-while-dormant** ⚠ *new* | The contract deadline elapses **while the issuer's realm is torn down** (§7.9); assert adjudication still happens at the collateral holder, exactly one terminal (I15), collateral has exactly one destination | pre-merge | **≤90 s** |
| **G11 offline-account fill** ⚠ *new* | **Fill an offline player's resting order while their last realm is torn down** (§7.10); assert the escrow posts, the wallet updates, exactly one terminal, and I19 holds throughout | pre-merge | **≤90 s** |
| **G8 ANTI-THEATER** | Inject a deliberate 1-unit imbalance through a test-only faucet and assert the reconciliation sweep **DETECTS** it — mirroring the D-6 "no-op-stub `Store` goes RED" guard [R]. **Without G8 the whole invariant suite is decorative.** ⚠ **Extended in revision 2** with two cells revision 1 lacked: (b) a **correctly-declared** mint that exceeds its arm budget must be REFUSED (I16), and (c) a shard whose self-reported partial disagrees with its own emitted journal must be DETECTED (I17) | inner | **≤30 s** |
| **G-ECON-REPLAY** | Replay a recorded input log under **unchanged** tuning and assert the emitted econ log is **byte-identical**. Without this the counterfactual twin is fiction | inner | **≤45 s** |
| **G-ECON-ORACLE** (optional, **NOT in `just gate` — binding, not optional**) | Generate order streams (proptest), feed both our Rust matcher and the Coq-extracted OCaml/Haskell binary, assert identical fills. Feasibility depends on (a) license clarity — the repos have **none** [V], (b) an OCaml/Haskell toolchain, (c) an agreed order-stream serialisation, (d) our tie-breaking provably matching theirs (⚠ and per §4.1 the theorems give volume uniqueness, not price uniqueness, so "identical fills" requires *our* tie-break rule to be implemented on their side too). Fallback: a Rust reimplementation of the verified *algorithm* plus proptests against the paper's stated theorems | out-of-band | n/a — must never be a `just gate` dependency |

**Two honest costs:**

1. I3 requires a fence-consistent cut across shards, which is cleanly available only at a quiesced tick
   boundary. So the global sweep is a **periodic sampled** check; the per-tick check is the weaker (but
   still strong) per-shard I1/I2/I4.
2. **HR5's per-monomorphisation rule against an order-type × TIF × self-trade matrix is the difference
   between ~600 and ~2,000 tests.** The branchless-shim discipline (generic fns are straight-line, all
   branching in monomorphic helpers) is therefore a **schedule input** on this crate, not advice.

### 6.6 Fail-loud vs fail-safe for money: what a LIVE shard does when an invariant fires

⚠ **Revision 1 specified 14 invariants and 11 gates and never said what a running production shard does
when one fails with players connected.** That is the single most consequential un-designed behaviour,
because it is exactly the situation the invariants exist for — and §8.6 already predicts the failure
mode ("the gate will now fire spuriously and be disabled by the next person").

**Rule: never panic.** A panicking realm under RLM re-spawns and **retries the same poison entry-set** —
an infinite crash loop that also loses the tick. The correct shape is **reject the entry-set, keep the
frame.**

| Invariant | Detected at | Blast radius | Automatic action |
|---|---|---|---|
| **I1 double-entry-zero** | entry-set construction, pre-commit | the entry-set | **REFUSE** the entry-set; emit `EconIncident`; the initiating action returns a typed error to its caller. Never partially post. |
| **I2 no-negative** | pre-commit balance check | the account | **REFUSE**; the order/trade/withdrawal fails with `InsufficientFunds`. This is a normal outcome, not an incident, **unless** the account was already negative — then it is an incident. |
| **I8 escrow-closure** | reservation terminal | the reservation | **REFUSE** the second terminal; emit `EconIncident`; keep the first terminal. |
| **I10 no-dupe** | container mutation | the `ItemId` | **REFUSE** the mutation; **quarantine the `ItemId`** (a fenced flag: not transferable, not tradeable, visible on the dashboard) rather than deleting it — deletion is itself unexplained destruction. |
| **I3 conservation** (per-shard partial) | the reconciliation sweep, off-tick | the **currency × realm subtree** | **HALT** trading in that subtree for that currency: set a per-market `HALTED` flag that is itself **fenced, CAS-committed and dashboard-visible**; postings that only *move* value (settlement of already-matched fills) continue, mints/burns do not. |
| **I5/I16 undeclared or over-budget mint** | the mint site | the `Faucet` arm | **REFUSE** the mint at source, fail loud to the caller, emit `EconIncident`. This is the whole point of I16. |
| **I6 single-writer / I9 fence** | command admission | the key | **REJECT** as stale (the existing fence discipline); count it. Already the repo's behaviour for every other authoritative action. |
| **I17 self-report mismatch** | external cross-check | the shard | **ALERT** only (never auto-halt on an external check — a broken exporter must not stop the game); page a human. |

**`EconIncident` is an ordinary event in the same journal** (`{tick, shard, invariant, subject, fence,
entry_set_digest, action_taken}`), so a post-mortem is a query and the incident count is a dashboard
series. A halted market's `HALTED` flag is a **fenced CAS record**, not a config flag, so it survives
teardown, replays identically, and cannot be set by a stale actor.

**Reversal is a first-class design item, and wall-clock rollback is never permitted.**

- **Rollback (rewinding the cluster to an earlier wall-clock state) is FORBIDDEN.** Under sealed shards
  it is not implementable honestly: players kept playing, sagas committed, realms were torn down and
  re-spawned with retired NodeIds (F2), and `applied_steps` would have to be un-recorded. I12
  audit-replay makes *reconstruction* possible; it does not make *rewinding* possible.
- **Reverse FORWARD instead**, as a first-class cohort operation: `Reversal { case_id, cohort,
  dry_run }` → an N-party batched distribution (D-50) of `Faucet::AdminGrant` / `Sink::AdminConfiscation`
  postings, each idempotent by `(ReversalId, member_step_id)`, each carrying the operator identity and
  the case id, **dry-runnable** (compute and display the aggregate effect before committing), and
  **N-party-signed** for high-impact cohorts (§4.7's M-of-N).
- **Compensation must be FUNDED** — a transfer from a declared reserve account, never a mint (§9.4).
  Otherwise the compensation for an incident is itself an unmeasured faucet.

---

## 7. The fit to our architecture

### 7.1 Crate topology

⚠ **WORLDLINE (rev 2) — OVERRULED.** `scripts/dormant_world_simulation_design.md` §1.1/§3.1 places `vd-econ`
**ABOVE** `vd-sim`, not below it: an object-safe `EconomyPort` **defined in `vd-sim`**, **implemented in
`vd-econ`**, injected as `Option<Box<dyn EconomyPort + Send + Sync>>` from **`vd-bins`** — so the edge direction
is **econ → sim** and there is **no `vd-sim → vd-econ` edge at all**. The topology below requires one, which
links the economy into every shard binary and reduces LAW-E1 to a runtime flag. The precedent is in-tree twice
(`Store` and `RealmSpawner`, `crates/sim/src/io/mod.rs:415-479`, both documented object-safe *as an HR5
requirement*), and the rule is machine-enforced by `tests/tests/crate_isolation.rs` — as an **allowlist over the
transitive closure**, including dev-dependency edges, since the gate as it stands today skips dev-deps, checks
direct edges only, and is a name blacklist. Consequences for the rows below: `vd-econ` cannot be *driven by*
`vd-sim` on the tick schedule (it is drained through the port); the seam-provided `DetRng` note is moot; and NPC
agent strategies do **not** live in `vd-econ` (see the §7.7 and D8 notes). The physical substrate itself lands in
**`vd-core`**, which the isolation gate already asserts depends on no workspace crate.

| Crate | Tier | Contents | Why here |
|---|---|---|---|
| `vd-core` (existing) | **A, 100%** | `Money` newtype, `CurrencyId` + `minor_exponent`, integer basis-point rate arithmetic, `muldiv` with declared rounding, largest-remainder `allocate`, `Price`/`Qty` tick/lot newtypes | Pure value types; sits below wire, so the frozen contract can reference them. Follows the `core/src/tlv.rs` branchless-shim idiom [R]. |
| `vd-wire` (existing) | **A, 100%** | The economy `InterShardFlow` arm(s) + envelope types + the TLV tags for wallet/tax-credit fields | The reviewed egress taxonomy; adding an arm without classifying it in the exhaustive `effect_class()`/`durability_class()` matches **does not compile** [R] |
| **`vd-econ` (NEW)** | **A, 100%** | The ledger kernel (accounts, entries, pending/post/void, faucet/sink registry), the order book + clearing, tax composition, NPC agent strategies, integer graph algorithms (**hand-written** SSP-with-potentials min-cost flow, Bellman-Ford cycle detection), index/inequality folds | One new Tier-A crate keeps the economy's branch density out of `vd-sim` and makes its coverage independently attributable. Depends only on `vd-core` (+ `serde`, `thiserror`) — **no new third-party deps.** ⚠ This is only true because §3.5's "~250 lines over petgraph" is **retracted**: petgraph is not a workspace dep and must not enter Tier-A (§3.5). ~4–6 k LOC, and it becomes the **10th** Tier-A crate — the justfile pins `tier_a` at exactly the nine existing crates today [R], so adding it is a deliberate edit with a coverage consequence. |
| `vd-sim` (existing) | **A, 100%** | The `economy` capability field on `ShardProfile`; systems that drive `vd-econ` on the tick schedule; the `sim::io` seam usage | `ShardProfile` already lives here with a validating `build()` [R] |
| **`vd-econ-solver` (NEW)** | **B** | Proportional response / tâtonnement / LP calibration, Leontief cyclic fallback, offline parameter fitting | Behind the io seam. Float-tolerant by construction. **Output is quantised to an integer price grid before it re-enters the sim** — the same discipline as "every physics→control boundary quantized to integer grids" [R] |
| **`vd-econ-analytics` (NEW, NOT a server build-graph member)** | **B** | Parquet writer, the exporter sidecar, DataFusion/DuckDB/ClickHouse client, the dashboard API | Keeps arrow/parquet/datafusion (47 direct deps incl. tokio + object_store [V]) out of the server build entirely — load-bearing given the documented 15-minute-build / 208 GB-target history [R] |
| `vd-harness` (existing) | **A, 100%** | `verify_value_conservation` + siblings; the `p3_economy_bank` scenario driver | Joins the existing oracle family [R] |
| `vd-io-prod` (existing) | **B, 90% floor** | The `EventSink`/exporter impl, the `EconSnapshot` admin endpoint, `econ_*` metrics | Where I/O legally lives [R] |

**Hard rule for the crate split:** no analytics dependency may appear in the Tier-A list, which the
justfile pins as exactly nine crates at 100% region+branch [R]. arrow/parquet/DataFusion inside Tier-A
would be an HR5 catastrophe (llvm counts regions **per monomorphisation** and these are generic-heavy).

#### 7.1.1 `vd-econ`'s crate-level rule set (absent from revision 1)

`vd-econ` sits **below** `vd-sim` on the dependency arrow, so it does not automatically inherit the
sim/node `clippy.toml` regime. It must **opt in explicitly**, and the rules are not optional for a crate
that decides money:

| Rule | Why |
|---|---|
| **Inherit the sim/node `disallowed-types`**: no default-hasher `HashMap`/`HashSet` — `BTreeMap`/`BTreeSet`/`DetHashMap` only | Iteration order reaches the wire and the journal. FxHash is **not** architecture-portable either (§12.2), so even a fixed-seed map's iteration must be sorted before it becomes bytes. |
| **Inherit the sim/node `disallowed-methods`**: no ambient time, no ambient rng, no I/O, no allocation-in-hot-loop patterns | Tier-A purity; `vd-econ` must be callable from a Tier-A oracle and from the counterfactual twin with identical results. |
| **`DetRng` is a PARAMETER, never ambient** | The Doran–Parberry mandatory book **shuffle**, every ZI-C/PRZI draw, and the seed-derived auction end-jitter all need randomness. `vd-econ` cannot reach `sim::io` (wrong direction on the arrow), so the seam-provided `DetRng` is **passed in by the `vd-sim` caller** as `&mut dyn DetRng` — object-safe, so no per-monomorphisation region multiplication (the same reason `Store` and `RealmSpawner` are object-safe [R]). |
| **Branchless generic shims** | `vd-econ` is the most branch-dense code in the codebase. Serialize/lookup are straight-line; every `?`, `if`, `match` and error closure lives in a monomorphic helper (the `core/src/tlv.rs` idiom, `field_decode_err` for hoisting closures) [R]. This is the difference between ~600 and ~2,000 tests (§6.5). |
| **No `f32`/`f64` anywhere in the crate**, enforced by a lint | Every rate is integer bp/ppm; every division is an explicit `muldiv` with a declared rounding direction. Float belongs in `vd-econ-solver` (Tier-B) only, and its output is quantised before re-entry. |

### 7.2 The egress / `InterShardFlow` decision

Every economic cross-shard interaction, assigned to an arm. The taxonomy currently has **23 landed
arms** (Ghost, Transfer, Directory, Saga, SagaAck, DirectoryReply, FlushSource, TransferAck, Demote,
Promote, TransientRelease, TransientDrop, ReleaseComplete, TransientAbandon, ReHome, TransientDiscard,
ReSolicitBatch, CrossingRequest, TransientCrossingRequest, TransientCrossingGrant, CrossingAborted,
CrossingAbortedAck, RealmDemand) plus **3 reserved** (`BlockEdit` P6, `Coupling` P8, `Signal` P9) [R].

| Economic interaction | Arm | Effect class | Durability | Note |
|---|---|---|---|---|
| Value (funds) crossing shards | **`Transfer`** (existing) — a `Funds` `TransferableKind` | `SideEffecting{TransferStep}` | as today | **HR2-pure**: value moving between shards has exactly one legitimate mechanism family. No new arm. |
| Item/inventory crossing shards | **`Transfer`** (existing) | `SideEffecting{TransferStep}` | as today | An item stack is just another `TransferableKind`. |
| Escrow reserve / post / void across shards | **`Transfer`** legs + the directory CAS | `SideEffecting{TransferStep}` | as today | Reserve ≈ `FlushSource`/`Frozen`; post ≈ post-CAS tail; void ≈ `ThawSource`. |
| Fill settlement **within one venue** | **D15-dependent**: *(none — purely local)* under escrow-at-the-book, **or** one batched `Transfer` go-token per clearing batch under escrow-at-the-account-owner | — / `SideEffecting{TransferStep}` | — / as today | ⚠ **Revision 1 asserted the local form while §6.4 rule 1 simultaneously required escrow at the account owner** — mutually exclusive unless account owner == venue (§6.4). Under escrow-at-the-book a fill really is a local mutation (the biggest single simplification). Under escrow-at-the-owner every fill is a cross-shard escrow post — affordable at the hub's 2.3–3.0 trades/s, but a different design, a different idempotency requirement and a different E-7 slice. **This is D15; do not carry both.** |
| Order place / modify / cancel from a client | **the D-39.1 reliable client→shard discrete-action carrier** (already owed, with block-edit P6 as first consumer and combat FIRE P11 as named second) [R] | `SideEffecting` | reliable | **The economy becomes its THIRD consumer of one build-once arm** rather than a per-feature fork. HR3-clean. |
| Cross-realm order command (rare) | **ONE new narrow arm** `Market(MarketOp)`, *or* reuse the discrete-action carrier | `SideEffecting{TransferStep}` | `ProducerLessReliable` if shard-initiated ⇒ **must be `Durability::Retained`** | See §7.2.1. |
| Tax remittance / treasury accrual up the tree | **`Signal`** (reserved, P9) carrying an idempotent `(fence, beneficiary)` **ACCRUAL NOTICE** — never an authoritative debit | `EffectFree` for the notice; the money itself moves by `Transfer` when reconciled | idempotently re-derivable | The `Signal` doc already states cross-shard durable signals reuse `(TransferId, step_id)` with a correlation id [R]. Accruals are **pull**, per T8. |
| Price ticks / book depth / index values / dashboard telemetry | **`Signal`** (P9) or a dedicated read-only feed | **`EffectFree`, latest-wins** | `Unreliable` OK | May **NEVER** carry authority-gating state. `sealed_shards.md` documents the exact failure this rule prevents (a discrete `DockState.clamped` riding a lossy datagram) [R]. **A fill is a Transfer; a price feed is a Signal; never merge them.** |
| Neighbour price discovery | **a TLV tag INSIDE the transferred item stack's blob** (the Veloren pattern, correctly placed) | inherits the enclosing `Transfer` arm's `SideEffecting{TransferStep}` | as today | ⚠ **CORRECTED.** Revision 1 wrote "piggyback on the existing goods `Transfer`, `EffectFree` extra field". **Effect class is a property of the ARM, not of a field**: `InterShardFlow::Transfer` is classified `SideEffecting{TransferStep}` (`crates/wire/src/intershard.rs:318`) [R], so there is no such thing as an `EffectFree` field on it — and describing a field as having its own effect class is exactly the confusion the G-SEALED gate exists to prevent. Worse, the row invited a **field append to `TransferEnvelope`**, the frozen contract carried by *every* transfer. The Veloren idea is right; the placement is wrong. Put the price+supply vector as a **TLV tag in the item-stack blob** (the same schema `tax_credit_minor` belongs in, §10.2 S2), consumed at the dest as **advisory latest-known state**. Its loss is tolerable because the enclosing arm's idempotency already covers the goods movement. **Zero new machinery, zero request/response, zero cross-shard read.** Staleness is a feature. |
| Corporation role / division-permission changes | **a fenced, CAS-committed directory-style record** — *not* a Signal | `SideEffecting{FencedCas}` | reliable | Role/permission state is **authority-gating discrete data** ⇒ it may never ride an `EffectFree` arm. This is real design work not yet scoped anywhere in `docs/design/` [R]. |
| Market data to clients (book pages, fills) | **the connection plane** — a **paged, fence-stamped, reliable** carrier | — | reliable | **Never** `InterShardFlow` — that is a shard↔shard taxonomy. ⚠ **CORRECTION**: revision 1 wrote "`MsgClass::Bulk`-style". **There is no `Bulk` arm.** `MsgClass` is `Control, Saga, Snapshot, Input, Membership, GhostReliable, GhostDelta, RealmSnapshot` and it is **WIRE-FROZEN, APPEND-ONLY** (`crates/sim/src/io/mod.rs:51`) [R] ⇒ **the market read path has no carrier today.** It is either a second consumer of D-4's `EventMsg` (fine for digests and fills) **or** a distinct paged class (better for book pages, because a page is a bulk payload that must not head-of-line-block a fill). Either way: **page size bounded well under the 1 MiB stream-frame cap** (`MAX_STREAM_FRAME_BYTES = 1 << 20`, `crates/wire/src/framing.rs:17` [R]), a stated refresh cadence (`market_digest_hz`), per-session subscription caps, and a **bytes/s/client budget** — 1.6 MB/s per 2,000-client hub at 1 Hz, 32 MB/s at 20 Hz (§2.2). Gate it with a load cell: a client browsing while a burst runs, mirroring SPIKE-3a. |
| **Cross-realm market SEARCH** ("cheapest X within N realms") ⚠ *new row* | **a non-authoritative READ MODEL**, projection-fed from the analytics exporter, never a query into a sealed World | — | — | Revision 1 gestured at a "galaxy index" inside §6.3 option C with no owner, no freshness contract and no authority status. Specify all three: the **analytics projection builds it**, its **staleness bound is stated and DISPLAYED to the player**, and it **may never gate a fill** — a stale index shows you where to fly, and the authoritative price is whatever the venue's next batch clears at. |
| Wallet/market state to a client | **needs a reliable, fence-stamped, latest-wins-by-fence carrier** | — | reliable | ⚠ `MsgClass` currently has **no reliable per-session gameplay-state class**: `Control, Saga, Snapshot(U), Input(U), Membership, GhostReliable, GhostDelta(U), RealmSnapshot(U)`, and it is **WIRE-FROZEN, APPEND-ONLY** [R]. D-4 already owes exactly this (`EventMsg`) [R]. **Do not smuggle balances into the `Snapshot` datagram** — D-18/D-36 are the in-repo postmortems for authority-gating state on unreliable datagrams [R]. |

#### 7.2.1 What would need a NEW arm, and why

| Candidate new arm | Needed? | Why / why not |
|---|---|---|
| `Market(MarketOp)` — cross-realm order create/modify/cancel | **Maybe ONE** | Only if a player may manage an order in a realm other than the one they occupy (the Dual Universe "remote manage, local settle" model). If orders are always placed in the occupied realm, the D-39.1 client carrier suffices and **zero new arms are needed.** |
| `Economy(EconomyOp)` — a single catch-all for orders/contracts/payments/tax | **NO — prefer not** | Keeps the reviewed file small but makes idempotency keying **heterogeneous within one arm**, which is exactly what the G-SEALED conformance gate exists to prevent. Prefer `Transfer` for value (HR2-pure) + at most one narrow command arm. |
| A new `MsgClass` for reliable per-session economic state | **YES, but it is already owed** | D-4's `EventMsg`. The economy is a second consumer, not a new requirement. |
| Anything per-economy-feature | **NO — treat as a red flag on the design** | A design that needs a new arm per economy feature should be revised, not accommodated by widening the taxonomy. |

**Net (⚠ recounted honestly in revision 2).** Revision 1 said "at most ONE new `InterShardFlow` arm",
which undercounted by treating a **reserved-but-unbuilt** arm as free: the file header lists `Signal`
under "RESERVED (variant lands with its consumer)" (`crates/wire/src/intershard.rs:28`) [R], and the
economy's plan uses it heavily (price digests, accrual advisories, corp events). The honest delta to the
reviewed surface is:

| Reviewed-surface change | Status |
|---|---|
| `InterShardFlow::Signal` | **RESERVED, unbuilt** — the economy is one of its co-designers (S10), not a free rider |
| `InterShardFlow::Market(MarketOp)` | **at most ONE new arm BEYOND `Signal`**, and only if remote order management is wanted (§7.2.1) |
| `DirectoryKey::Account` (+ `Market`) arms | **appends** to a small closed enum with exhaustive matches [R] — cheap, but they *are* reviewed-surface changes (S6) |
| A reliable per-session `MsgClass` (D-4 `EventMsg`) | **already owed**; the economy is its second consumer |
| A paged market-data class | **new**, or D-4 carries digests and pages both (see the read-path row) |
| The client discrete-action carrier (D-39.1) | **already owed**; the economy is its third consumer |
| New binaries | **ZERO** |

So: **one new arm beyond the already-reserved `Signal` (whose shape the economy co-designs), two
`DirectoryKey` appends, one already-owed `MsgClass` plus possibly one paged class, one already-owed
client carrier, and zero new binaries.** That is still the strongest single argument that this is not a
bolt-on for this architecture — it is just no longer stated as one arm.

### 7.3 Realms as jurisdictions, and how tax rates compose

The realm tree (galaxy > system > planet > station > area) is a real, live hierarchy with
`RealmPath`/`path_for_realm`/`lineage_seeds` and an O(depth·log L) `path_index` since RLM 5e-3a [R]. It
is therefore **already the jurisdiction tree**, and §4.8's eight rules apply directly:

- **Situs** = the deepest containing realm = the containment result the realm layer already computes.
- **Rate composition** = sum `rate_bp` over `path_index(situs)`, applied **once** to one base.
- **Ceilings** compose top-down via a parent-stored `child_cap_bp`; **floors** use nearest-ancestor top-up.
- **Tariffs** are edge taxes at `LCA(src, dst)`, decomposed into export/import chains exclusive of the LCA
  — reusing the same LCA primitive the Signal system needs for up-to-LCA routing.
- **Remittance** rides accruals (T8), and the RLM ancestor-closure invariant guarantees an active
  market's jurisdiction chain is live [R].
- **Rate CHANGES** must themselves be fenced, journaled and ledgered events with an `effective_tick`
  delay — otherwise a station owner front-runs their own tax change against inbound trades.

**One deliberate limitation to state:** our jurisdictions are a strict tree. Eco's overlapping
same-tier influence radii are richer politics but make incidence ambiguous and membership expensive.
Recommend strict hierarchy (D10 in §11).

### 7.4 Where corporations live

A corporation is **not spatial**, so it is neither a Realm nor an Occupant under the current taxonomy —
it is a third thing (an identity/ownership aggregate) with members across many shards, and it must
outlive realm dormancy.

| Option | Mechanism | Pros | Cons |
|---|---|---|---|
| **(i) Identity plane** — alongside player identity in `docs/design/identity_persistence.md` [R] | A corp is a durable identity record with a directory `Owner` key; its wallets are `Account` keys | Reuses the identity/persistence machinery; naturally non-spatial; survives all realm churn | Extends a design doc that currently has no organisational-identity concept |
| **(ii) `TransferableKind` on the economy-authority shard** | A corp is a Durable kind whose authority lives on the economy capability | Reuses HR2 verbatim; the corp's wallets are co-located with the ledger ⇒ zero-saga internal moves | Ties corp availability to the economy shard |
| **(iii) Hosted by the corp's HQ realm** | Authority follows the HQ station | Spatially intuitive; taxes naturally local | **Rejected**: the HQ realm can be killed, and moving HQ becomes a re-home of the whole corporation |

**Recommendation:** (i) for identity + roles + share registry, (ii) for the wallets (i.e. corp wallets
are `Account` keys owned by the economy authority). Roles/permissions are **authority-gating discrete
data** ⇒ a fenced CAS record, never a Signal. Share ownership then implies a cross-shard ownership
graph — the one place per-entity aggregates genuinely break down, and where `disintegrate`'s event-first
"decision model" is worth reading [V].

### 7.5 Where markets live, and how a remote market is browsed

- **A market is a venue capability on a realm — but "per realm" is NOT expressible today, and that
  matters.** ⚠ Revision 1 said: add one field (`market_venue: bool`) to the existing `ShardProfile`
  lattice (`crates/sim/src/capability.rs:71` already has `voxel`, `signal_graph`, `functional_blocks`,
  `block_edit`, `surfaces`, `seats`, `signal_relay`, `hull_host`, all derived and validated by `build()`
  [R]) with a lattice rule enforced at parse time, exactly as `FunctionalBlocksNeedVoxel` is today —
  costed at "~30 lines + tests". **The lattice half is right; the per-realm half does not exist.**
  `ShardProfile` is a pure function of the realm's **KIND**:
  `RealmCoord::profile_kind()` → `profile_kind_of(RealmKindTag)` (`crates/core/src/realm_coord.rs:63,97`)
  → `vd_sim::capability::profile_for(ProfileKind)` (`crates/sim/src/capability.rs:250`), and
  `RealmSpawner::spawn_realm(coord, at_tick)` derives the profile **from the coord alone**
  (`crates/sim/src/io/mod.rs:468`, doc: "The profile is derived from `coord.profile_kind()`") [R].
  A bare `market_venue: bool` on `profiles::station()` therefore makes **ALL stations venues** — never
  "venue-capable realms", which is the entire basis of the recommended D1(d) two-tier design ("real order
  books only where a venue capability is enabled") and of G5's fixture. **S5 splits in two** (§10.2):
  - **S5a** — the `ShardProfile` field + its lattice rule. Cheap, ~30 lines + tests, as described.
  - **S5b** — **a per-realm capability override carried on the spawn path**: a seed-derived venue flag
    from the generator → `spawn_realm`. This is a real **RLM-touching** slice (it changes the
    spawn signature or adds a per-realm capability record the reconciler must carry through rehydrate),
    and it must be ledgered, not costed at 30 lines.
  - **Until S5b exists, venue-ness is per KIND.** Re-check D1(d) under that constraint: a per-kind flag
    still gives "books at stations, ambient prices everywhere else", which is a coherent (if coarser)
    two-tier design — it just cannot express "this station is a hub and that one is not", which is what
    the concentration lever (§5.1 AVOID) actually needs.
  **Never a new binary, never a `match` on shard kind** (HR3). D-39.4 already prescribes the
  capability-config pattern for stations: "a new `RealmId` arm + a `ShardProfile` capability config —
  zero new transfer code, HR2/HR3" [R].
- **Order RANGE is a realm-subtree predicate, not a jump distance** — `this station | this planet | this
  system | this branch | galaxy`. Cheaper than EVE's jump query, composes with `path_index`, and
  symmetric range (or range that **costs money**) is the decentralisation lever EVE never pulled.
- **Browsing a remote market** has exactly three legal shapes, in order of preference:
  1. **Last-known state gossiped on goods deliveries** (Veloren): free, HR1-native, deliberately stale.
  2. **An `EffectFree` periodic price/depth digest** on the Signal plane, rate-limited: gives real
     cross-region comparison; must never gate authority.
  3. **A client-side read model fanned in by the gateway/analytics projection**: best UX for "show me my
     orders across 12 realms", but it is a *read model*, not a query into a sealed World.
  A direct cross-shard market **query** is not available under HR1 and should not be designed for.
- **A hub realm exceeding one shard's capacity — ANSWERED in revision 2** (it was OQ #2). Our answer must
  not be EVE's (TiDi + dedicated hardware), and it does not need to be: **`Market(RealmId, CommodityId)`
  — option A's own key — already partitions one venue's books across K single-writer authorities for
  free.** The hot *venue* is therefore never the hot *writer*. Quantified: The Forge holds 409 k orders
  across ~2–4×10⁴ active types ⇒ **~10–20 orders per book on average**, the hottest single
  (venue, commodity) book is O(10³) orders ≈ 60–200 KB, ~300 ns/order to match; trade flow in the hottest
  region is ~200–260 k trades/day ⇒ **2.3–3.0 trades/s mean** (300/s even at 100× peak) and ~2.5 M
  postings/day ⇒ **29 postings/s** — three orders of magnitude below the corrected per-shard write
  ceiling (§2.2). **The hot market is not a throughput problem; it is a hot REALM problem** (orders +
  clients + escrow + read fan-out concentrated in one realm), and the per-commodity partition plus
  `max_orders_per_book` plus the client-fan-out caps address all four. Admission control + fee-based load
  shedding remain the *policy* backstop, not the mechanism.

### 7.6 Items and inventory on the `TransferableKind` registry

- **A stack is a ledger position, not an entity with a mutable count.** `Container → (ItemKind, amount)`,
  with split/merge as entry-sets summing to zero per `ItemKind` ⇒ split is *exactly* a transfer and
  inherits every invariant, oracle and chaos cell the currency ledger has. That is what makes I10
  provable with the same machinery as I3, and it unifies inventory with money at the economy altitude
  (currency is `ItemKind::Currency(CurrencyId)`; a share is a commodity with an issuer; ore is a commodity).
- **Crafting/refining is a declared faucet+sink pair** (inputs burned, outputs minted) ⇒ it appears in the
  audit as an *explainable* non-conservation rather than as drift.
- **`ItemId` minted like `EntityId`** (`{kind, mint_shard, seq, rand}`, never time-derived, never reused),
  with the **ledger — not the blob — as the uniqueness authority.**
- **Wallets ride the player blob as TLV-tagged REQUIRED fields**, using §A5's cut discipline (N10) and
  the version-floor handshake, so a missing tag is a hard abort rather than a zeroed wallet.
- ⚠ **The per-entity economic-state BUDGET (missing from revision 1, and it converts a rich player into an
  un-transferable player).** `MAX_STREAM_FRAME_BYTES = 1 << 20` (1 MiB) caps a stream frame
  (`crates/wire/src/framing.rs:17`) [R]. A hauler with 10⁴ item positions at tens of bytes each, plus a
  multi-currency wallet, plus an `applied_txn_ids` set, plausibly approaches or exceeds it — and because
  decode-to-Default is banned for Durable kinds and a missing REQUIRED tag is a hard `ABORT_SPATIAL`,
  exceeding the cap turns *"a rich player crosses a shard boundary"* into **"this player can never
  re-home."** So state the budget and enforce it **at mutation time**, not at transfer time:
  `max_positions_per_container` (proposed 2,000), `max_currencies_per_wallet` (8),
  `applied_txn_id_retention_ticks` (bounded so the set is O(10²)) ⇒ worst-case blob well under
  **256 KiB**, i.e. 4× headroom under the frame cap. **Refuse the 2,001st stack** as a normal typed error;
  never discover it during a crossing. **Gate:** a proptest asserting a max-budget entity serialises
  under the cap, plus a Tier-A invariant at every mutation site. If genuinely unbounded inventories are
  wanted, **chunked blob transfer becomes an owed DEFERRED item** — decide which, do not leave it implicit.
- D-31 is directly relevant: `TransferableKind`'s behaviour half (`serialize`/`spawn`/`precondition`/
  `rebind_refs`) is **not built yet**, and `StubCrossing.state` is currently an opaque `Vec<u8>` produced
  as `vec![]` [R]. The economy would be a natural **second** consumer of that trait after the TLV blob
  lands at 1d.6 — and `rebind_refs` is exactly what a container full of items needs.
- ⚠ **This must constrain P6/P7's FIRST line of inventory code (S11), not be retrofitted after it.**
  Revision 1 scheduled items-as-ledger-positions at slice **E-3**, after the ledger — but **P6 (block
  edits) and P7 (checkpoints, whose roadmap row explicitly promises resuming "inventory") land BEFORE the
  economy arc** [R] and will inevitably ship a **mutable-count** inventory first. Retrofitting a live
  inventory into ledger positions is precisely the "live-economy data migration" the report elsewhere
  insists must be avoided — it is the same argument S2 makes for a single TLV tag, applied to something
  ~100× larger. The early-seam cost is **~0 lines of implementation**: a written rule plus the `ItemId`
  minting shape. See S11 in §10.2.

### 7.7 Where the NPC economy sim ticks

Four placements, and the right answer is a mix:

| Placement | Cost | RLM behaviour | Use for |
|---|---|---|---|
| **In the realm's own tick** | Competes for the frame budget; 5,000 noise agents × 10³ markets = 5×10⁶ evaluations/round unless decimated | Dies when the realm is killed | Active, player-present markets only; decimate by a per-agent phase derived from `(seed, agent_id)`; AoI-scale the population |
| **Offline at worldgen** (Veloren) | Free at runtime | Immune | The initial price field and the profession distribution |
| **Lazy closed-form `f(seed, universe_tick)`** for realms with no active market | **O(1) per query, ZERO state** | **Immune, and this is the key** | Every realm nobody is in. A coarse price is derived closed-form from `f(seed, universe_tick, realm_path)` **plus a small persisted deviation** — the same seed-derived pattern as terrain, and fully consistent with our determinism law (celestial math is already Category A closed-form [R]). An unvisited market therefore has a plausible, continuous price with **no simulation and nothing to restore.** |
| **A separate background economy simulator** exchanging only aggregates (Star Citizen's choice) | A new process/capability | Immune | Only if the agent layer grows beyond what a realm tick can host |

**The AoI-gated design that makes cost O(active) rather than O(realms):**

- Tier 1: a realm runs a **real** book and real agents only while Active under the RLM AoI predicate.
- Tier 2: **every other realm is a pure closed form** — `f(seed, universe_tick, realm_path)` plus a small
  persisted deviation, **zero background work, zero state to tick.** ⇒ total cost is **O(active)**.
- **Spin-up RECONSTITUTES** the leaf book from the closed-form field (seed an NPC ladder consistent with
  the coarse price and recorded stock); **teardown COLLAPSES** it back. This is the economic analogue of
  RLM's own states and needs **no new concept**.
- ⚠ **Two mandatory guards.** (a) The coarse-graining literature is explicit that an aggregate needs
  **NEW rules** — the fine-grained agent rules are not valid for the aggregate [L] ⇒ author the aggregate
  as a supply/demand curve, not as a crowd of ZI agents. (b) The **collapse/reconstitute pair must be
  value-conserving**: assert `Σ aggregate stock == Σ leaf stock` and `Σ escrowed currency` across the
  transition, **or the hierarchy is a faucet.**

⚠ **Revision 1 presented TWO mechanisms for the same job and chose neither, and its complexity claim did
not derive.** It offered both the closed-form tier above *and* a **rollup pyramid** of aggregate ports
decimated per level, claiming `O(active·log n) + O(realms/2^depth)`. Two problems:

1. **The formula is not derivable as written.** If level-k-from-the-leaves runs every 2^k ticks over
   N/b^k realms, per-tick work is `Σ_k N/(b^k·2^k) = N·Σ(1/2b)^k ≈ 1.05·N` for b=10 — i.e. **Θ(N)**: the
   decimation is cancelled by the fact that the leaves are the most numerous *and* the fastest. It is
   sublinear only if the **deepest, most numerous level is the SLOWEST** (work ≈ `N/2^D = (b/2)^D`; for
   b=10, D=5 that is ~3.1×10³ of 10⁵ realms per tick, i.e. **a station's ambient economy updates once per
   32 ticks**). That is a perfectly good choice — but it must be **stated as a decision**, not implied by
   a formula.
2. **The pyramid is redundant with the closed-form tier in the same section.** If inactive realms are
   O(1) closed-form with zero state, background work is already **0** and total cost is already
   **O(active)**.

⚠ **WORLDLINE (rev 2) — THIS SECTION IS SUPERSEDED IN THREE WAYS; see
`scripts/dormant_world_simulation_design.md`.**
1. **THE CLOSED-FORM TIER IS A GAME SUBSTRATE, NOT AN ECONOMY COMPONENT.** "Lazy closed-form
   `f(seed, universe_tick)` … zero background work, zero state to tick" is the right answer and it is now the
   **worldline**: a pure integer `evaluate(baseline, deviations_before(t), t)` in **`vd-core`**, serving NPC life,
   material stocks, production and destruction — all of which must advance with the economy **absent** (LAW-E1).
   The economy is consumer #2, reading the same function. So this row's placement column changes from "the
   economy's inactive tier" to "the game's physical layer, which the economy reads".
2. **"NPC agent strategies inside `vd-econ`" IS A LATENT LAW-E1 VIOLATION** (and so is the §7.1 row that says
   so). Switching the economy off would delete the NPCs, directly against LAW-E2's own words ("so players can
   interact with NPCs"). Split: **existence / population / pose / physical needs = GAME**; **trading policy =
   an injected object-safe strategy with a named price-free `NeedsOnlyStrategy` default** (see the D8 note).
   *Prices make NPCs smarter, never alive.*
3. **THE STATE BUDGET IS ANSWERED, AND THE SEAMLESSNESS WORRY WITH IT.** The `(stock, deviation)` budget below
   is the worldline's `Baseline` + bounded absolute-rebase log: **0 bytes for a never-visited realm**
   (gate-asserted), ~3–9 kB for a heavily-touched one, ~3–9 MB for a 100 k-realm universe at 1 % visitation —
   with the log bounded by **player attention** rather than by world size or elapsed time, coalesced on ingest,
   and compacted **lazily on adopt**. The "must not see prices snap" requirement is met by construction, because
   the leaf and the coarse field are **the same function** rather than two approximations of each other. Three
   constraints come with it: the fold must key on **`(realm_fence, tick, seq)` with the fence dominant** (a
   stale incarnation otherwise reverts its successor and mints material); the deviation is an **absolute rebase**,
   never a delta; and touched-then-abandoned realms need a **tombstone-back-to-seed** retention rule (its D-79),
   because the touched set is otherwise monotone.

**Decision: closed-form-only for the inactive tier.** It is strictly simpler, genuinely O(active), and it
is the one that satisfies "nothing to restore on wake". The rollup pyramid is **demoted to a named
alternative** — worth revisiting only if aggregate *interaction between* dormant realms (dormant trade
flows) is ever wanted, at which point the honest derivation above applies and
`ambient_reprice_period_ticks` per level becomes the explicit knob. (The multiport composition law,
arXiv:2512.20600 [L], remains the right formalism *if* that day comes.)

⚠ **And the missing budget is STATE, not compute.** Whichever tier wins, the persisted
`(stock, deviation)` per `(good, realm)` for **every realm that ever hosted a market** is
~10⁶ realms × ~30 goods × 16 B ≈ **480 MB, monotonically growing**, with no eviction policy anywhere in
revision 1. Add:

- `max_deviation_entries_per_realm` and a **global** `deviation_store_bytes_budget` in `EconomyTuning`;
- an **eviction/regeneration rule**: a realm whose deviation decays below a named threshold **reverts to
  pure closed form** and its row is deleted — the deviation is a *correction*, so dropping it is
  lossless by construction, unlike dropping stock;
- stock, by contrast, is **value** ⇒ it may never be evicted, only converted (a declared
  `Sink::AmbientDecay` if ambient stock is meant to decay at all — decide it, do not let it leak).
- ⚠ **Seamlessness**: the standing hard rule forbids visible jumps [R], so the coarse field must be
  consistent with whatever the leaf book last recorded — a player who leaves and immediately returns must
  not see prices snap. Exact reconciliation is an open question (§12).

### 7.8 Player-to-player direct trade (the trade window)

⚠ **Absent from revision 1**, which is a real gap on three counts: it is the **most common economic user
interaction in any MMO**, it is the **classic dupe vector** (revision 1's own §9.1 lists "duping via a
state-machine hole — a packing dialog open during a trade" as an exploit class while never designing the
trade), and it is the **cheapest possible first economy feature** — it needs no book, no venue, no
agents, and no item registry beyond stacks. Its absence also meant revision 1's slice sketch had **no
player-visible economic interaction until E-7**.

**The mechanism** — a two-party atomic swap that may span shards (two players in different realms):

| Element | Design |
|---|---|
| FSM | `Offered → BothPending → Confirmed → Posted` \| `Cancelled` \| `Expired`. One `TradeId`, one `Fence`. |
| Reservation | **each side's assets are reserved by their OWN account owner** (two-phase pending, §4.5), so neither player's shard ever holds the other's value. This is the one place where escrow-at-the-account-owner is unambiguously right regardless of D15. |
| The swap | a **linked entry-set**: both legs post together or neither does. If the two accounts are on different shards it is one batched `Transfer` go-token with two credit legs, each idempotent by `(TradeId, leg_step_id)` — credits are monotone/I-confluent so at-least-once + dedup suffices (N1). |
| **Confirm-lock** | a confirm **invalidates on ANY basket mutation** (add, remove, quantity change, or a reservation terminal), and the invalidation is part of the same fenced state — this is the anti-dupe rule, and it is what "a packing dialog open during a trade" defeats when it is absent. |
| Expiry | `expiry_universe_tick` on the trade, like every order (S9). |
| Receipt | the stored receipt from N6 (`{outcome, resulting_fence, balances_after}`), echoed to both clients with the idempotency key — this is read-your-writes without prediction. |
| Tax | one `TaxableEvent` with situs = the deepest realm containing **the trade**, i.e. `LCA` of the two participants if they are in different realms (T7's edge rule). |

**Chaos cell (G9):** kill either side **between confirm and post**; assert exactly one terminal per
reservation (I8), no partial swap (I1), and conservation across the pair (I3).

**Slice position: E-2b**, after value-crosses-shards and **before** items-as-ledger. It proves the ledger
+ saga composition end-to-end with a real player-visible feature, and it is the natural **first HR6/vdctl
acceptance test** (two real clients, live QUIC, both wallets settle — §8.8).

### 7.9 Contracts, collateral and courier logistics

⚠ Revision 1 concluded — correctly and importantly — that *"contracts are not a separate subsystem, they
are the transfer machinery with an economic policy on top"* and that *"courier contracts are THE
mechanism that makes a distributed market work without a global market"*. Under sealed shards that makes
contracts **more load-bearing than order books**. It then left them at one paragraph: no FSM, no state
table, no invariant, no chaos cell, no DEFERRED entry, and buried in "E-8+".

**The FSM and its fence points:**

```
Posted --accept--> Accepted --deliver--> Delivered --adjudicate--> Completed
  |                   |                                              |
  +--expire-->Expired +--deadline-miss--> Failed(collateral forfeit)  |
  +--cancel-->Cancelled (issuer only, only while Posted)              +--> collateral returned
```

| Element | Design |
|---|---|
| Authority | the contract is a **Durable `TransferableKind`** owned by the **issuer's account owner** (not the issuing realm — realms die). Accepting it is a fenced CAS on the contract key ⇒ **two haulers can never both accept** (I15). |
| **Reward + collateral** | both **held by the ACCOUNT owner** in two-phase pending (the same rule as S9's escrow), with the contract holding only references. Collateral is a **`Transfer`**, never a `Sink`, unless forfeited to an NPC — in which case it is `Sink::CollateralForfeit`, declared. |
| Deadline | `expiry_universe_tick`, evaluated **on wake** against `universe_tick` (closed-form ⇒ correct across an arbitrary down period). No wall clock, no scheduler. |
| **Adjudication** | a **pure function** of `(contract, universe_tick, delivery_evidence)` evaluated at the **collateral holder** (the account owner), which is live by I19 — *not* at the issuer's realm, which may be dormant. This is the fix for "who adjudicates when the issuer's realm is down". |
| Partial delivery | allowed only if the contract declares `allow_partial`, in which case reward and collateral apportion by **largest-remainder** (§4.5) — one rounding point, `Σ shares == total`. |
| Discovery | contracts are **advertised** via the same non-authoritative read model as market search (§7.2), scoped `Public \| Private \| Corp \| Alliance` (EVE's scoping, verified [V]), with a stated staleness bound. Accepting is authoritative; browsing is not. |
| Retention | `contract_retention_ticks` (EVE uses a 30-day window [V]) after a terminal, then the record is archived to the analytics log and dropped from the shard. |
| **Transport RISK** | the point of courier gameplay: cargo can be destroyed or stolen in transit. That makes the loss a **declared `Sink::Destruction` + a `Transfer::CollateralForfeit`**, and it is the natural demand for cargo insurance (§7.14) — the risk-adjusted price gradient *is* the arbitrage gameplay. Revision 1 mentioned this only as an Albion aside [L]. |

**I15 CONTRACT-CLOSURE** (new): every contract reaches exactly one terminal and its collateral has
exactly one destination; no contract is accepted twice.

**Chaos cell (G10):** the **deadline elapses while the issuer's realm is torn down**; assert adjudication
still happens at the collateral holder, exactly one terminal, and conservation.

**Slice position: E-4b**, *before* order books. A distributed market needs hauling more than it needs a
book.

### 7.10 Account authority lifecycle: offline players, dead realms, escheatment

⚠ **The offline-player problem was absent from revision 1, and it is a correctness hole, not a UX one.**
Revision 1 makes wallets a TLV field on the **player entity** blob (§7.6, N10) and simultaneously
requires escrow at the **account owner** (§6.4 rule 1) plus **single-writer per `AccountId`** (I6). But an
offline player **has no live entity in any World**, and RLM will tear down the realm they were last in.
So: which shard holds `Owned` for an offline player's `Account` key? Who honours a fill against their
resting buy order's escrow? Who pays their corp's tax accrual? (OQ #8 asked only the narrower "does an
unreachable account FAIL or QUEUE".)

**The rule (I19 ACCOUNT-ALWAYS-HOMED):** an `Account` key's authority must be independent of the owner's
**session** *and* of any **realm's liveness**. Every `AccountId` has exactly one live `Owned` holder even
when its owner has zero sessions and their last realm is dead.

| Option | Mechanism | Note |
|---|---|---|
| **(i) Accounts owned by the economy capability** | the never-dormant economy shard holds every `Account` key | This is D2 option **B**'s one unambiguous advantage, and revision 1 never said so. Under the recommended option **A** it is not available for free. |
| **(ii) A never-dormant HOME shard per account, resolved via the existing `home_shard`/`coordinator_of` discipline** | the account's home is a **stable function of the `AccountId`** (e.g. the ancestor-closure-pinned Galaxy realm, or a hash-partitioned set of them), independent of where the player is | **Recommended under option A.** Galaxy realms are already ancestor-closure-pinned live whenever anything below them is active, and the pinning is the machinery's own invariant rather than a new one. |
| **(iii) Accounts re-home with the player** | authority follows the avatar | **Rejected**: it is exactly what breaks when the player logs off, and it needs `ReHomeState::Snapshot` (§6.3) to be safe at all. |

**⇒ D2 options A and C must each answer this or be dropped.** Under (ii) they answer it cleanly.

**Chaos cell (G11):** **fill an offline player's resting order while their last realm is torn down.**
Assert the escrow posts, the wallet updates, exactly one terminal, and I19 holds throughout.

**Realm-store lifetime vs ledger permanence** (also missing): RLM tears realms down and F2 retires their
`NodeId`s **forever** [R]. Revision 1 never said who owns a torn-down realm's `econ/` prefix, whether a
realm store is ever GC'd, or what happens to accruals/orders/escrow inside a realm that is **never
respawned** — and D-53 covers rehydration on *wake*, not permanent non-return, which is the case that
decides whether I3 can ever close.

**Escheatment policy (the multi-year tail):**

| Abandoned value | Resolution |
|---|---|
| A **tax accrual / clearing balance** for a realm that never runs again | after `escheat_after_ticks`, a declared `Transfer::EscheatToAncestor` to the **nearest live ancestor treasury** — never a silent drop, never an unbounded accrual counted in I3 forever |
| Orders / escrow in a never-respawned realm | `expiry_universe_tick` (S9) fires on the **next wake of any ancestor** that can adjudicate, releasing escrow to the account owner (who is live by I19) |
| **Scrip issued by a dissolved corporation** | redeem from the declared reserve escrow (§4.6) until exhausted, then **void the remainder as a declared `Sink::IssuerDissolved`** with the loss visible on the dashboard |
| Shares of an abandoned corporation | delisting + liquidation (§7.11), pro-rata treasury distribution via the N-party path, then registry closure |
| A **deleted or banned** account's balances | `Sink::AccountClosed`, declared, with the case id |
| Items stored on a realm nobody visits | this is why **`Sink::StorageRent`** matters: it is one of the very few sinks that scales with *hoarding*, and without it abandoned hoards are permanent state. **Decide whether recurring storage rent exists** (D21). |
| The `econ/` prefix of a permanently-retired realm | owned by the **escheat sweep**, which is the only thing licensed to GC it — and only after every balance inside it has a declared destination |

### 7.11 Territory, realm ownership, alliances — and the corporation ORGANISATION layer

⚠ Revision 1 presupposed all of this in three places and designed none of it: D11(b) (recommended) makes
tax rates "player-settable by whoever holds the realm", §7.5 makes a market a venue capability on a realm,
and §3.8/§9.1 key self-match prevention on "character → corporation → alliance".

**(a) Territory and ownership.**

| Element | Design |
|---|---|
| `RealmOwner` | a **fenced CAS record in the directory**, same class as corp roles — `{realm, owner: AccountId \| CorpId \| Npc, since_tick, fence}`. Not a Signal; it is authority-gating discrete data. |
| **Acquisition** | **a user decision (D18)**: `Claim` (first-come, cheap, land-rush) \| `Auction` (periodic, the same clearing engine — HR3) \| `Conquest` (P11 combat coupling) \| `Rent` (recurring `Sink`, the softest). Each has a very different combat coupling, so it cannot be decided after P11 starts. |
| Unclaimed realms | **NPC-owned with config rates** (`npc_default_rate_bp` per realm kind), so there is always a defined tax rate and always a counterparty. |
| Realm ownership as an **instrument** | selling a station = transferring authority over a live realm = a `RealmOwner` CAS **plus** a value transfer, atomically linked. It trades on the same clearing engine as everything else (HR3). |
| Alliances | **decide (D19)**: a real **jurisdiction tier** (a node above Galaxy? a cross-cutting overlay?) or purely an **identity grouping** for SMP/roles/contract scoping. ⚠ A jurisdiction tier breaks D10's strict tree (alliances are not spatial) ⇒ **recommend identity-grouping only**, with alliance-level taxes expressed as *voluntary transfers from member corps*, not as a tax level. |

**(b) The corporation organisation layer.** Revision 1 delivered wallets (7 divisions), a share registry,
M-of-N officer signatures, votes, and D-51 as a ledger entry honestly admitting "this is real design work
not yet scoped anywhere". What was missing, at implementation depth:

| Element | Design |
|---|---|
| **Permission model** | a **fixed-width bitmask per division** (7 divisions × a `u32` of permissions = 28 B, bounded, trivially serialisable, no dynamic allocation — the same "count is small" decision as the divisions themselves). Permissions compose by OR within a role set; a query is a mask test. |
| **Role changes** | **fenced CAS records with an `effective_tick` DELAY**, and a mandatory audit event. ⚠ Revision 1 applied the front-run fix to tax rates (§7.3) but **not to roles** — a director who can grant themselves withdrawal rights and drain in the same tick is the same exploit. Same machinery, same delay. |
| **Membership lifecycle** | join / leave / kick as a directory-keyed relation, with an **explicit rule for member-held corp assets on departure**: assets held *on behalf of* the corp are a distinct ownership tag (not the member's balance), and departure triggers a `Transfer` back to a corp division — never a silent retention and never a confiscation of personal assets. |
| **Corp-action audit log** | every corp action (withdrawal, role grant, asset move, contract issue) is an event in the same journal with the actor identity ⇒ **"who drained the hangar" is a query**, which is the single most-requested corp feature in the genre. |
| **Corp taxation of member income** | a first-class `Sink`/`Transfer` pair: withholding at the income event (EVE: 0–11% on bounty/mission payouts over 100,000 ISK [V]), assessed and remitted **in the transaction currency** (§7.16). |
| Shared hangars | asset ownership **distinct from wallets** — a container whose owner is a `CorpId` + division, with the permission mask gating access. |
| **Theft stance (D20)** | ⚠ Revision 1 stated a fraud stance for share/bank fraud but not for **corp-internal theft**, which is EVE's most famous economic gameplay. **Recommend: theft IS gameplay, but every corp action is logged and disclosed, so it is detectable after the fact** — the exact mirror of "fraud about the future is gameplay; fraud about the present balance is a missing feature". |

**(c) Equity lifecycle** (revision 1 covered the registry, dilution votes, dividends and disclosure, but
not the lifecycle):

| Stage | Design |
|---|---|
| **Issuance / IPO** | fixed supply at creation + **vote-gated dilution**; the founder allocation is a **declared, ledgered issuance** (`Faucet::EquityIssuance` against a `Sink::EquitySubscription` of the paid-in capital), never a magic balance. |
| **Illiquid valuation** | a batch auction with two orders a week produces a meaningless "market cap" that a dashboard would publish as fact. **Rule: refuse to publish a market cap below a liquidity threshold** (`min_trades_for_valuation`, `min_volume_for_valuation`); display **"no trade"**, never a stale print. |
| **Delisting / liquidation** | the common MMO case (abandoned corps). Procedure: registry freeze → pro-rata treasury distribution via the **batched N-party path** (D-50, largest-remainder) → registry closure → `Sink::CorpDissolved` for any unclaimed residue. |
| **M&A** | an **N-party share swap** — a second consumer of D-50, not new machinery. |

### 7.12 Recipes vs player-built machines (the biggest drift risk against the full end-goal)

⚠ Revision 1's whole production layer (§4.4's Leontief over an acyclic BOM DAG, T3's VAT input credit,
the Doran–Parberry producer agents) assumes **authored recipes with fixed technical coefficients**. But
**this game's factories and refineries are player-built assemblies of blocks** whose throughput is
emergent from the build and driven by the P9 Signal system — that is the standing end-goal
("signal-heavy cross-shard blocks", stations and ships built from blocks). If a player's refinery
throughput depends on their block layout, power signals and cross-shard signal routing, then the
technical-coefficient matrix `A` is **not** seed-derived world data and the Leontief sweep is computing a
fiction. Revision 1 flagged only the weaker "abstraction ratio is a one-way decision" and asked "hand-
authored BOM or derived from recipes" — neither is the sharp question.

**The reconciliation (state it, and it is cheap):**

| Layer | What it is | Who owns it |
|---|---|---|
| **The RECIPE** | a **registry entry**: `inputs → outputs`, acyclic (S3), integer, the taxable base for T3, and **the only thing the economy reasons about**. `A` is over recipes. | `vd-core` registry, seed-derived world data |
| **The MACHINE** | a **player block assembly** that determines *rate*, *efficiency* and *location* — driven by signals, power and layout | `vd-sim` blocks (P6/P9) |
| The coupling | machine efficiency enters the economy as a **per-job multiplier bounded by config** (`machine_efficiency_min_bp`, `machine_efficiency_max_bp`), quantised to an integer grid at the boundary — the same discipline as every physics→control boundary [R] | the job-cost function |

**Hard rule: a recipe may NEVER be created by a player build.** A player build changes *how fast* and
*how efficiently* a recipe runs, never *what transforms into what*. If the user ever wants
player-authored recipes, then S3's acyclicity validator must run **at runtime on player-authored graphs**
— a materially different design (untrusted graph input, a cycle becomes a griefing vector, and the tax
layer's stage-invariance must be re-proven per graph). **Vet the goods-taxonomy decision (OQ #5) against
this before it is fixed**, because the abstraction ratio and the recipe/machine boundary are the same
one-way decision seen from two angles.

### 7.13 Guaranteeing regional price divergence (the mechanism revision 1 never supplied)

⚠ Revision 1 recorded **twice** that this is the failure mode of the family it recommends — Veloren's own
source admits *"prices end up nearly identical in every town"* [V], and §5.6/§4.4 conclude "regional
divergence must be engineered deliberately" — and then specified **no mechanism to ensure it**. Worse,
D1(d)'s ambient closed-form price and §4.4's Victoria 3 MAPI blend (`local = MAPI·market + (1−MAPI)·state`)
actively **pull toward convergence**, and the only divergence force named (transport cost via min-cost
flow) is Tier-B **advisory**. Without divergence there is no arbitrage ⇒ no hauling ⇒ no
courier/contract gameplay ⇒ the load-bearing loop of the whole design is dead.

**The divergence budget — four forces, all deterministic and all in `EconomyTuning`:**

1. **Seed-derived per-realm resource and demand profiles** ⇒ equilibrium prices differ **by
   construction**, not by accident. `f(seed, realm_path)` yields each realm's resource endowment vector
   and its demand vector; these are the same closed forms §4.10 needs for the extraction faucet.
2. **Transport cost as an additive integer term on the ambient price**, derived from the tiered-`i64`
   `LatticePos` distance (D-41, exact-integer [R]) — so distance *is* a price, in the authoritative tier,
   not only in the advisory solver.
3. **Tariffs (T7) as additive integer edge costs at the LCA** — already designed; this is where they earn
   their keep, because they *reshape* trade routes rather than merely taxing them.
4. **A bounded cross-realm coupling weight**: `mapi_weight_per_level`, **low at the leaves** and rising
   toward the root (a station is nearly autarkic; a galaxy price is an average). Cap the total coupling so
   a leaf price can never be pulled more than `max_parent_coupling_bp` toward its parent per repricing
   period.

**And assert it, or it will silently centralise:** a dashboard series **and a gate** — the **median
cross-realm price dispersion for a named basket must stay above `min_price_dispersion_bp`**. If it does
not, the economy has converged and the hauling loop is dead; that is a *test failure*, not a balance
observation.

### 7.14 Insurance as a funded pool (diagnosed in revision 1, not designed)

Revision 1's best cautionary datum is that EVE's insurance has been a **~+2.8 T ISK/month NET FAUCET for
two decades** [V], and §4.9 says our payout pool must be "actuarially closed by construction" — but no
mechanism followed.

| Element | Design |
|---|---|
| **Premium** | `premium = f(observed loss rate over a window, risk class)`, quantised to **integer basis points**. The loss rate is a **Tier-B statistic** (computed over the event log) feeding a **Tier-A rate** — so it crosses the quantisation seam exactly like a solver output: quantise, then commit as an `EconomyEpoch` parameter, never read a float in-tick. |
| **The pool** | a real balance funded by premiums, per risk class. **`payout ≤ pool_balance` is a HARD rule.** |
| **Pool exhaustion** | a **defined, declared behaviour**: `PayoutPolicy::{ DenyUntilFunded, ScalePro Rata(bp), Queue }` — never borrow, never mint. Whichever is chosen is visible on the dashboard as a first-class series. |
| **Moral hazard** | **no payout on self-inflicted or non-hostile loss**, evidenced from the destruction event's own provenance triple (`ref_type, context_id, context_id_type`) — the famous EVE self-destruct-for-payout exploit is closed by *provenance*, not by policy text. |
| Adverse selection | risk classes are **per hull/cargo class × security band**, seed-derived from the realm's danger profile, so a low-risk actor cannot buy a high-risk pool's rate. |
| Coverage granularity | **per-hull** by default (cheap, one record); per-item only for **cargo** insurance, which is where it couples to **courier collateral** (§7.9): a courier's collateral and the cargo's insurance must not *both* pay out for the same loss — the contract declares which is primary. |
| Gate | a **conservation cell**: the pool cannot mint. Inject a payout larger than the balance and assert it is refused (fail-loud), not funded from nowhere. |

### 7.15 Cold start of MONEY: M(0), onboarding, and the long-run policy

Revision 1 answered cold-start **liquidity** well (Albion-style adaptive NPC bid, AMM bootstrap,
closed-form ambient prices) and cold-start **money** not at all — yet §4.5's identity
`M(t) = M(t−1) + Σfaucets − Σsinks` **needs an M(0)** to be assertable.

| Element | Design |
|---|---|
| **M(0)** | a **seeded, declared `Faucet::Genesis`** with a stated per-holder-class distribution (NPC treasuries / faction reserves / player starting balances). Not a magic initial balance — otherwise the very first conservation check needs a plug, which is the failure mode §8.6 Rule 0 exists to prevent. |
| **Starter capital** | a **bounded `Faucet::StarterGrant`** per new account, capped and counted like every other faucet (and therefore budgeted by I16). |
| **First-hour income floor** | mechanism, not hope: **NPC-anchored entry-level BUY orders** (finite quantity, config replenishment budget — never infinite, §4.3) for the goods a new player can produce in their first hour. This gives a guaranteed income floor **that does not scale with veteran wealth**, which is the actual onboarding problem in a 3-year-old economy. |
| **Long-run inflation target** | a stated **net-issuance band** (§4.10c), validated against every config change (§8.6 step 2). |
| **Inequality POLICY** (revision 1 measured Theil and designed no lever) | the available levers, in ascending order of intrusiveness: (a) progressive **fee schedules** (broker/transaction bp rising with order value); (b) **recurring asset sinks** — storage rent (D21) and structure upkeep, which fall on hoards rather than on flows; (c) Eco's **pro-rated multi-owner wealth tax** [V], explicitly copyable and explicitly the most intrusive. ⚠ With the OSRS caution attached: **taxes and sinks are weaker levers than designers assume** [V], so the honest expectation is that (b) does more than (c). |

### 7.16 Multi-currency: who quotes, what governs the numeraire, and which currency a tax is in

Revision 1's multi-currency **mechanism** half is strong (ledger-partitioned currencies, ≥4-entry linked
FX, star topology with N−1 books, Bellman-Ford as a monitoring assertion). Its **policy** half was
missing entirely.

| Question | Answer to present |
|---|---|
| **Who quotes the star's N−1 books?** | Two options with very different consequences. **(a) An NPC market maker** ⇒ a *de facto peg* the operator must **fund and defend**; it must be an **Avellaneda–Stoikov inventory-skewing quoter with a funded budget** (§4.3), never a fixed-price anchor, or it is the money printer of §4.3's reject row. **(b) Player-quoted** ⇒ thin books and manipulable cross-rates, mitigated only by the star topology making triangular arbitrage structurally impossible. **Recommend (a) with a bounded budget per period**, degrading to (b) as volume grows. |
| **What governs the numeraire's supply?** | A stated **monetary-policy rule**, not a config value: net issuance band + the per-arm faucet budgets (I16) + `Faucet::Genesis` for M(0). Revision 1's D9 asked "NPC-issued with a faucet/sink budget, pegged, or a player instrument?" and no rule followed; §8.6 only validated that a config's issuance is "not absurd". |
| **Which currency is a tax assessed and remitted in?** | ⚠ T1–T8 were silent. **Rule: assessed in the transaction currency, remitted in the transaction currency**, with FX only at the **beneficiary's option** as an explicit linked-transfer event (≥4 entries). Assessing in a numeraire would import an FX rate into the tax base and make historical verification impossible — the exact reason §4.5 requires ≥4 entries for FX in the first place. |

### 7.17 Order-entry latency and the pending-ack contract (no client prediction)

Revision 1 fixed the client **contract** (§12.4: expose `posted` and `pending`, compute `available`
server-side, echo the idempotency key + fence) but never budgeted the **felt** latency — which decides
whether the market feels broken.

**The budget:** client → gateway → shard → apply → receipt, **on top of** the 100–150 ms interpolation
buffer, plus (for cross-shard value) "a saga = a few ticks (50–150 ms)". Name it
`ORDER_ACK_BUDGET_MS` and gate it release-only with `crates/harness/src/latency.rs`'s percentile helper
[R] at ~10–20× the observed p99, per the SPIKE-3a pattern.

**The pending-ack UI contract:** an order is unacknowledged for ~200–300 ms and **prediction is
forbidden**. So the server **authors an intermediate state** — `Accepted{order_id, idempotency_key,
fence}` returned immediately on admission, before the batch clears — and the client renders *that real
server state*, never a guess. This is also **an argument FOR D1(a)** that revision 1 left on the table:
under a **batch auction** the player already expects to wait for the next clearing, so a 200–300 ms ack
is invisible; under a continuous book the same latency reads as "the market is laggy".

### 7.18 Economy → world-state feedback (so the economy drives the game)

Revision 1 flagged Elite Dangerous's BGS as proving that economy→world-state coupling (security, stock,
outfitting availability, NPC behaviour) is durable and fun, with the **non-farmable rule** (influence
responds to **profitable volume in a SCARCE commodity**, not raw volume) [L] — and then nothing in the
design consumed economic state to change the world, leaving the economy an accounting system bolted
*beside* the game.

**Name at least one designed consumer, from day one:** the **per-realm depletion/congestion index**
(§4.4) already gates job cost; extend the same index to drive **NPC presence / security response** and
**station service availability** (which services a realm offers is a function of its own economic
activity). With the non-farmable rule written in: the index responds to *profitable volume in a scarce
commodity*, so round-tripping does not farm it. Everything beyond that is deferred and ledgered — but
"the economy changes nothing in the world" must be a **decision**, not an omission.

⚠ **WORLDLINE (rev 2) — ANSWERED, and the answer is a designed mechanism rather than a named index.**
`scripts/dormant_world_simulation_design.md` §3.2 gives the economy an **ACTOR channel**: it enqueues an
ordinary fenced, refusable, idempotent **`EconCommand`** (`MoveStack | GrantHull | DeliverGoods | SetLien |
ClearLien | …`) on the same path a player's action takes, and the **game** applies it under its own authority and
journals the resulting deviation. §2 **row 26** is its verdict row. Five properties make it the only LAW-E1-legal
shape: the game may **refuse** it (typed, counted, and the refusal is itself an observable fact); it is applied by
the game's authority so its effect is replayable **from the game's journal alone**; it is idempotent and
re-drivable; **absence of the economy means the command was never issued**, never that the game blocked; and it is
strictly one-directional (`-> ()`), so no game code path waits on an economy answer.
**The rejected alternative is worth recording**, because it is the obvious one: an *advisory value* the game reads
(a price hint biasing NPC scoring). That was revision 1 of the worldline design's `weights()` and it is deleted —
it made durable physical state a function of unjournaled, off-tick-derived monetary state, which broke both the
live-equals-dormant property and this report's own §8.1 replay obligation. If an advisory input is ever wanted it
must be tick-aligned, fence-stamped, **recorded in the game's journal at the tick it was applied**, and carried by
a reviewed arm (that design's W17(b)).

---

## 8. Fully analyzable: the data, the dashboards, and the tweak loop

### 8.1 The event log must BE the ledger's journal, not telemetry

⚠ **WORLDLINE (rev 2) — SHARPENED INTO TWO RECORDS, both game-owned.**
`scripts/dormant_world_simulation_design.md` §4.6 splits what this section treats as one artefact:
- **The WORLDLINE STATE** — bounded, authoritative, **`LossBudget::ZERO`, never shed**, ~0 B–9 kB per *touched*
  realm. It carries every fact that changes a **conserved total** or founds a **claim** (item mints and burns,
  declared losses, destruction/delivery, balances) under **RULE WL-CONSERVED-FACT**, and it rides the reviewed
  `InterShardFlow` arm.
- **The FACT JOURNAL** — unbounded, historical, **OPTIONAL, always sheddable and never blocking** (drop-and-count
  with a declared `JournalGap`), on a **separate** carrier so a non-sheddable record can never back-pressure the
  shared emit site and stall the tick.
Two consequences for this section as written: (a) *"the log must BE the ledger's journal"* stays true for money,
but the **physical** truth a retro-payout replays from is the STATE record, not the journal — a sheddable journal
records a gap as a **count**, and a count cannot repair a per-`ItemId` identity; (b) the world advances
**byte-identically with the archive absent**, so the archive's availability is never a gameplay precondition.

"Fully analyzable" only has teeth if it means an **assertable accounting identity**. If the log is a
best-effort telemetry side-channel, the identity is unprovable and the dashboard is decoration; if the
log **is** the journal the balances are derived from, the identity is a unit test — and the log becomes
the primary **anti-duping** control (a dupe is a conservation violation detected the tick it happens,
not by an analyst three weeks later).

**Carriage (three conforming options).** HR1 says `Transport::send` is the only egress from sim/node and
every byte crossing a shard boundary is an `InterShardFlow` arm [R], so the log cannot be smuggled onto
the wire — **and it must NOT become an `InterShardFlow` arm** (it is a shard→warehouse fan-in, not a
shard↔shard flow, and it would push ~100× the gameplay traffic through the reviewed taxonomy).

⚠ **REVISION 2 RETRACTS OPTION (A) AS RECOMMENDED — it is technically impossible.** Revision 1
recommended "redb under an `econ/` prefix + a tailing sidecar", justified as "the classic
transactional-outbox/CDC pattern the repo already implements in `crates/io-prod/src/outbox.rs`". Both
halves are wrong, and the whole §8 "fully analyzable" pillar plus slice E-5 rested on it:

- **HARD-VERIFIED: redb takes an exclusive file lock at open.** `redb-2.6.3`
  (the version in our `Cargo.lock` [R]) calls `libc::flock(fd, LOCK_EX | LOCK_NB)` in
  `src/tree_store/page_store/file_backend/unix.rs:37` [V]. **A second process cannot open a live shard's
  store at all**, so an out-of-band sidecar cannot tail it — not slowly, not read-only, not ever.
- **The cited precedent is not analogous.** `crates/io-prod/src/outbox.rs:34` is
  `SharedOutbox = Arc<Mutex<Box<dyn OutboxSink + Send>>>`, read by the **same process's** peer-writers and
  boot replay [R] — never by an external reader. It is an in-process outbox, not CDC.
- **Architecturally it was a second, unreviewed egress of shard-private durable state**, i.e. exactly what
  HR1 forbids ("a shard's World/rapier/redb is private"), and §8.1's own preamble concedes
  `Transport::send` is the only egress.

| Option | Mechanism | Verdict |
|---|---|---|
| **(A) redb under an `econ/` prefix + an out-of-process tailing sidecar** | — | ⚠ **RETRACTED — IMPOSSIBLE.** redb's `flock(LOCK_EX)` at open forbids a second opener [V]. Recorded so nobody re-proposes it. |
| **(B) a 5th `sim::io` seam trait** | `EventSink { fn emit(&mut self, kind: EconEventKind, bytes: &[u8]) }`, **object-safe** (used as `&mut dyn EventSink`, so no per-monomorphisation region gotcha — the same shape as `Store` and `RealmSpawner` [R]). **mem impl** = a deterministic `Vec`, oracle-checkable inside Tier-A; **io-prod impl** = a bounded ring → an **off-tick writer** → rotated segments, mirroring `Store`'s off-tick fsync design (block-on-prior, depth-1) exactly [R] | **RECOMMENDED.** ⚠ **State the HR5 cost honestly** rather than claiming "zero new seam": **two impls to cover**, the mem one inside Tier-A at 100% region+branch and the io-prod one at the Tier-B 90% floor, plus the seam's own trait-object plumbing through `build_app`. That is the real price of the analyzability pillar and it is worth paying. |
| **(A′) keep the journal in the shard's own redb, export IN-PROCESS** | The shard writes rotated segments under `econ/` **and the same process ships them** (an io-prod task holding the same `RedbStore` handle, exactly as `replay_outbox` does today [R]); the sidecar consumes **files the shard has closed**, never the live redb | **VIABLE ALTERNATIVE.** Reuses the store we already crash-test and keeps segment durability inside the existing group commit. Costs less new seam surface than (B) but couples the exporter to io-prod's store handle rather than to a clean trait, and it makes the Tier-A oracle path (a deterministic `Vec` of emitted events) harder to express. **Present both; (B) is cleaner, (A′) is cheaper.** |
| **(C) the P9 `Signal` bus** | — | **REJECT for the log** (wrong routing shape: analytics wants fan-in to one sink; Signal routes up-to-LCA + broadcast-by-frequency, is rate-limited, and is the gameplay plane). **ACCEPT for economic *gameplay* data** (price ticks to market terminals, corp wallet notifications). These are two different planes and conflating them is the mistake to avoid |

#### 8.1.1 Backpressure and overflow: the journal is `LossBudget::ZERO`

⚠ **Neither carriage option in revision 1 specified overflow behaviour, and the omission is fatal to the
section's central claim.** If the log **is** the ledger's journal, then option (B)'s "bounded ring" means
**drop on full**, which is silent money-history loss — and money is `LossBudget::ZERO`. Option (A) put
unbounded growth inside the shard's own store sharing its per-tick commit/fsync budget with no shed rule.

**The policy, stated as a rule and pinned as a gate:**

1. **On ring-full or store-full the TICK MUST FAIL LOUD — refuse the economic ACTION, never drop a
   posting, and never kill the frame.** A refused order/trade/mint returns a typed error to its caller
   (the §6.6 discipline); the world keeps simulating. Dropping a posting is unrepresentable by
   construction, because the emit is part of the same entry-set commit as the balance change.
2. **A watermark + shed-loud rule** reusing the existing `SendError::QueueFull` idiom [R]: at
   `econ_ring_high_water_bp` the shard begins **refusing new discretionary economic actions**
   (order placement, crafting starts) while still allowing **settlement of already-committed
   obligations** — so the system drains rather than deadlocks.
3. **A retention/truncation CONTRACT between the shard and the exporter**: the shard **may not truncate
   `econ/` (or free a ring segment) until the exporter has durably ACKED that segment** — the classic
   outbox ack, which `crates/io-prod/src/outbox.rs` already implements for frames [R]. This is what makes
   a slow/dead exporter a **backpressure** event rather than a data-loss event.
4. **Named budgets, not literals**: `econ_ring_bytes`, `econ_ring_high_water_bp`,
   `econ_segment_bytes`, `exporter_ack_timeout_ticks`, `econ_journal_retention_ticks`.
5. **Gate G7b:** fill the ring / stop the exporter and assert **fail-loud rather than data loss** — plus
   that the frame keeps ticking and the refusal is counted, not panicked.

⚠ Also reconcile the write budget: §2.2's corrected figures give **~2,000 durable puts per 50 ms commit
for the WHOLE shard**, shared with the directory, saga WAL, `applied_steps`, block edits and checkpoints.
Econ postings are **not** free headroom on top; `postings_per_tick_budget` (§2.3) is carved out of that
same commit.

### 8.2 Schema

A Kimball star with **ONE narrow fact table** plus conformed dimensions:

```
econ_posting(
  universe_tick, wall_ts, shard_id, seq, econ_epoch,
  event_kind,                 -- our analogue of ESI's ref_type enum (EVE has 72)
  currency, amount_minor,     -- i128 minor units
  account_from, account_to,
  jurisdiction_path,          -- the realm path: the drill-down axis
  realm_id, subject_kind, subject_id,
  order_id, corp_id,
  fence, correlation_id
)
```

Dimensions: `account`, `corp`, `item_type`, `jurisdiction` (= the realm path), `event_kind`,
`faucet_sink` (the closed taxonomy), `econ_epoch` (the config version — see §8.6).

**There is no global total order available**: the `Transport` contract explicitly refuses cross-`MsgClass`
ordering [R]. So joinability must come from **`(shard_id, seq)` + `universe_tick` + `Fence` +
`TransferId`**, never from arrival order. A per-shard hash chain is an option for tamper-evidence
(open question §12).

### 8.3 Volume math, and two traps

⚠ **Rebuilt in revision 2 from the single consistent chain in §2.2.** Revision 1's §2.2 and §8.3
disagreed with each other by an order of magnitude ("~10⁶ economic txns/day ⇒ 41–47 GB/yr" vs "~5–20 M
econ events/day"), and §8.3's own Trap 1 conceded "tens of millions of individual credit events/day"
two paragraphs later. All three numbers now derive from one place.

| Line | Value |
|---|---|
| Postings (authoritative) | **300–600 postings/s sustained** ⇒ 0.95–1.9×10¹⁰/yr (§2.2) |
| Analytics write rate | **~10–40 M econ rows/day ⇒ ~116–460 rows/s avg, 2–5 k/s peak** |
| Analytics storage | **~123–492 MB/day, 45–180 GB/yr** Parquet+zstd; 5-year retention **0.2–0.9 TB** — still one node |
| Measured columnar win on real EVE order data | row CSV 136.8 B/row; row zstd-3/10/19 = 30.6/25.4/20.3 MB; **naive per-column split at zstd-10 = 19,755,660 B = 12.29 B/row** at a fraction of zstd-19's CPU — and that is still ASCII digits per column, so a *typed* Parquet with dictionary/delta/RLE should land materially below it [V] |
| **Ledger (authoritative, separate from analytics)** | **0.53–2.4 TB/yr** at 56–128 B/record ⇒ **retention/partition/archive is a GATING design decision, not a footnote** (D-54, and it now blocks E-1 via §6.2's snapshot requirement) |
| ⚠ Revision 1's arithmetic error, recorded | "7.3×10⁸ postings/yr … at 56–128 B ≈ **41–47 GB/yr**". 7.3e8 × 56 B = 40.9 GB ✓, but **7.3e8 × 128 B = 93.4 GB**, not 47 — the upper bound was computed against 3.65e8. So even on revision 1's own (10–50× low) input the range should have read **41–93 GB/yr** |
| Reference scale | CCP's **entire three-year** public faucet/sink fact table is 69,695 rows / **4,067,080 bytes**; their whole 9.4-year daily money-supply series is 3,420 rows / **366,270 bytes** [V] |

**Trap 1 — micro-faucets, and the compression claim that was arithmetically impossible.** EVE pays
**1.95 T ISK/day in Bounty Prizes** [V]; at 200–500 k ISK per NPC kill that is **4–10 M individual credit
events/day** — which is precisely why CCP batches them through ESS pools.

⚠ Revision 1's rule was "≤1 row per player per minute per faucet source, flushed on realm exit/teardown"
and it claimed this "turns a ~40 M-row/day firehose into 2–5 M rows/day" (10–20×). **That compression is
impossible under its own rule:** 40×10⁶ rows/day ÷ 1,440 min ÷ **24,518 CCU** (the report's own verified
ACU) = **1.13 events per player-MINUTE**, so a 1-minute window compresses by at most ~1.13×. At the
report's own **100 k CCU** target it is 0.28 events/player-min ⇒ **zero compression**. The 10–20× could
only ever have come from the *other* clause — **session-scoped flush on realm exit** — which reduces
ledger audit granularity from **per-kill** to **per-SESSION**, directly contradicting §9.2 item 5 ("prove
this account earned that ISK" is a query) and I12's per-event replay claim.

**⇒ This is decision D17, and it must be decided rather than asserted:**

| Option | Window | Consequence |
|---|---|---|
| **(a) Per-event, no aggregation** | `faucet_aggregation_window_ticks = 0` | Full per-event provenance; I12 and the moderation appeal both hold as claimed. **Cost: the 0.53–2.4 TB/yr ledger stands.** |
| **(b) Short window** | e.g. 1 minute (1,200 ticks) | ~1.1× compression, i.e. **not worth the granularity loss** — this is the option the arithmetic kills. |
| **(c) Session-scoped** | flush on realm exit / teardown | Real 10–20× compression, at the price of **per-session audit granularity**. Must be published as a **design concession**, and §9.2 item 5 downgraded to "prove this account earned that ISK **within this session**". |
| **(d) CCP's actual answer: change the GAMEPLAY** | ESS-style pooling | Fewer individual credit events *exist*, so nothing is lost. This is a **mechanism decision**, not a logging optimisation, and it is the honest way to get the volume down. |

Whichever is chosen, `faucet_aggregation_window_ticks` is an explicit `EconomyTuning` field, the
aggregate row carries the **count** so the accounting identity is preserved, and **both the ledger and the
analytics volume lines are re-derived from it.**

**Trap 2 — order books.** EVE Ref snapshots all 1.61 M orders 48×/day at ~945 MiB/day compressed [V].
Emitting only **place/modify/cancel/fill deltas** plus an hourly top-N-levels-per-market summary costs
**~12 MB/day** for the same analytical power — a **40× saving**, and depth is reconstructable by replay.

**Storage roles (do not conflate them):** redb = the hot/authoritative per-shard journal (already a dep,
already crash-tested); **append-only Parquet on disk/object-store = the immutable archive of record**;
an OLAP node = a **derived, rebuildable index**, never the source of truth.

⚠ **Amended: build EXACTLY ONE rollup, and treat it as MANDATORY.** Revision 1 said "do NOT build a
rollup pyramid — at 10 M rows/day the raw log is small enough to keep forever and query directly."
The *pyramid* prohibition survives (a naive `(hour × currency × event_kind × jurisdiction)` rollup can
easily **exceed the raw log's cardinality**), but "query directly" does not: a **full-year scan over
180 GB of Parquet at 1–2 GB/s effective is 90–180 s**, which is not an interactive galaxy dashboard.
So:

- **Materialise ONE narrow daily fact table** (EVE's equivalent is 65 rows/day) plus the index series
  (theirs is 16,161 rows total) — and treat it as **required infrastructure, not an optimisation.**
- **State the dashboard query targets as measured requirements**, not as an assertion that partition
  pruning will handle it: **p95 ≤ 2 s for a galaxy-level panel, ≤1 s for system, ≤500 ms for station**,
  over a 1-year window. Publish them, measure them, and treat a regression as a failure.
- **Nothing else is materialised.** Every other panel is a query over the daily table plus a bounded
  recent-raw window.

#### 8.3.1 Journal SCHEMA evolution over a multi-year live economy

⚠ Also missing from revision 1, which versioned the **config** (S4's `econ_epoch`) and one TLV tag (S2)
but never the **record schema of a 10¹⁰-row journal** — while I12 claims replay-from-genesis reproduces
every balance bit-identically, which would pin the decoder for **every historical schema version
forever**.

| Rule | Detail |
|---|---|
| **A version tag per SEGMENT**, not per row | `econ_segment_header { format_version: u16, econ_epoch, shard, first_lsn }`. One tag amortised over ~10⁵ rows. |
| **Decoders retained for all historical versions** | in the Tier-B analytics crate (never in Tier-A — historical decoders would multiply the Tier-A coverage surface without bound). |
| **I12 restated honestly** | *Replay is guaranteed bit-identical from the latest SNAPSHOT plus same-version segments. Genesis replay across format versions is BEST-EFFORT and labelled as such.* This is the only version of I12 that is true after the first schema change, and saying it now is much cheaper than discovering it in year three. |
| **Adding a field** | append-only within a version (postcard discipline), so reader-N tolerates writer-N+1's extra tail; a *semantic* change (splitting a faucet arm, renaming an `event_kind`) **bumps `format_version`** and is recorded as a `SchemaChanged` event in the same journal. |
| **Re-partitioning the archive** | a Tier-B rewrite job producing new segments with a new version tag; the old segments stay readable. |
| **The snapshot is the durability anchor** | which is exactly why §6.2 moves snapshot+truncate into E-1. |

#### 8.3.2 Seasons, wipes and leagues

Absent from revision 1 as a topic — Path of Exile was cited for auction-house friction but never for its
**league → Standard economy reset**, which is the shipped answer to "an economy ages badly". This is a
decision (D22) because it changes the shape of three other things:

- **"Never wipe"** makes retention (D-54), escheatment (§7.10) and inequality policy (§7.15) load-bearing
  **forever**.
- **"Seasonal"** makes `econ_epoch` (or a sibling `season_id`) a **partition key from day one** — free now,
  a migration later — and it makes M(0) (§7.15) a recurring event rather than a one-off.


### 8.4 Three packaged stack options

| | **MINIMAL** | **STANDARD** (recommended) | **HEAVY** |
|---|---|---|---|
| **Ingest** | Shard journals to redb; a sidecar exporter writes postcard segments | + one **ClickHouse** node ingested via the official pure-Rust `clickhouse` 0.15.1 crate (`Inserter`, RowBinary, zstd, rustls — no C++) | + Delta (`deltalake` 0.32.4) or Iceberg (`iceberg` 0.10.0) over `object_store` 0.14.1 for time-travel/ACID |
| **Query** | An offline `vd-econ` tool converts to Parquet (`arrow` 59.1.0 / `parquet`); query with `datafusion-cli` or the DuckDB CLI | ClickHouse SQL + the Parquet archive | + DataFusion embedded in-process for the counterfactual twin's queries |
| **Dashboard** | An additive `EconSnapshot` on the **existing** `axum` admin shell + `econ_*` names in the existing `/metrics` registry, with a `runs/`-style analytics manifest for reproducibility | + **Grafana** 13.0.0 (AGPLv3) with the official `grafana/clickhouse-datasource` (Apache-2.0); + Prometheus 3.11.2 or VictoriaMetrics scraping the `/metrics` we already serve for *operational* series | + a Rust front-end (`leptos` 0.8.20 / `dioxus` 0.7.9) over an axum API, or an `egui` 0.35 + `egui_plot` 0.36 desktop ops tool (**its own binary** — `bevy_egui` 0.39 pins `egui ^0.33` [L]) |
| **Scoring / stats** | none | `tdigest` / `sketches-ddsketch` / `hyperloglogplus` for percentiles + distinct counts (versions/licenses **[U]**) | + OpenTelemetry 0.32.0 for Tier-B tracing; `augurs` 0.10.2 + `linfa` 0.8.1 for advisory scoring |
| **New server-side deps** | **ZERO** | 1 (the `clickhouse` client, in the analytics crate only) | several |
| **New services to operate** | 0 | 2–3 (ClickHouse, Grafana, Prometheus) | 4+ |
| **Buys** | Full analyzability + HR6 agent-operability for almost nothing | Real-time cross-galaxy dashboards, ad-hoc SQL, community panels, zero bespoke UI work | Reproducible as-of snapshots, in-process counterfactual queries, bespoke UX |
| **Costs** | No real-time, no ad-hoc SQL ergonomics | Two more services; dashboards live in an **AGPL process** (fine as a separate binary; a fork/embed is a licensing event) | Build time, operational weight, more deps to justify |
| **Rust-native alternative** | — | **GreptimeDB** (Rust, Apache-2.0 core + separate enterprise license, v1.0 GA, SQL **and** PromQL, built on Arrow/DataFusion/Parquet/object-store) — one engine for metrics *and* events; architecturally elegant, considerably less battle-tested at this workload than ClickHouse [L] | — |

**Partitioning trick that makes the galaxy dashboard cheap:** partition Parquet/ClickHouse by
`(universe_day, realm_path_prefix)`. Because the realm tree is path-indexed, "show me this galaxy /
this system / this station" are all **prefix predicates that prune partitions**, so a galaxy-level
dashboard reads O(partitions at that level) rather than O(all fills).

### 8.5 What a galaxy dashboard must show

EVE's MER, published continuously instead of monthly, **plus everything CCP cannot show because their
pipeline is derived**:

| Panel | Series | Note |
|---|---|---|
| Money supply | Total + Δ, **by currency and by holder class** (character / corp — CCP splits exactly this way [V]) | |
| Faucets & sinks | By named source, with a **reconciliation residual that is provably 0** | CCP's residual needs an "Active ISK Delta" plug; ours must not |
| Velocity | MV=PQ over a window (CCP publishes 0.2792, with and without "accessories" [V]) | Free from the closed ledger |
| Price indices | **Fisher** headline (not Laspeyres) + per-basket decomposition, per region and rolled up | EVE's Mineral/Ship/Module/Consumer split is the proven decomposition |
| Regional dispersion | Price dispersion + **cross-region arbitrage spread as a heatmap over the realm tree** | This is the min-cost-flow dual, i.e. "where is the arbitrage" |
| Trade | Volume and value, by realm subtree | |
| Order books | Depth, spread, **time-to-fill percentiles** (tdigest), place/cancel ratio | |
| Throughput | Production / mining / destruction, by realm and security band | EVE publishes all three |
| **Tax revenue by jurisdiction, with the realm path as the drill-down axis** | Per level, per flow kind | **A structural advantage over EVE's flat region model** |
| Inequality | **Theil per realm subtree** (decomposable) + wealth percentiles + Lorenz | Theil answers "rich station or rich player?"; Gini cannot |
| Activity | Active traders (HLL), new-account inflow percentile | The latter is also the RMT-buyer signature |
| Corps / equity | Share registry, market cap, dividend flow, buyback/issuance, insider concentration (HHI) | |
| Integrity | Bot / RMT / manipulation panel: Benford deviation, wash-ring count, self-trade attempts, spoofing scores | Every model score is itself a logged event |
| Config | The **`econ_epoch` timeline** with every intervention annotated | See §8.6 |
| Health | Conservation residual, stranded `InTransit` balances, escrow age distribution, orphan-lock count | The correctness panel |

### 8.6 The operator tweak surface

**Rule 0 — a tweak is an EVENT through the ledger, never a balance write.** If an operator can write a
balance directly, three things break simultaneously: the double-entry invariant (unexplained value), the
audit-replay invariant (replay no longer reproduces state), and the conservation gate — which will now
fire spuriously and be disabled by the next person, the failure mode that kills invariant gates in
practice. So: `OperatorFaucet(OpId)` / `OperatorSink(OpId)` named arms, carrying the operator identity,
an idempotency key and a `Fence`.

**The loop:**

1. **ONE versioned `EconomyTuning` struct** (per the no-magic-numbers rule) holding every rate, cap,
   band, fee, basket and population parameter named in §4.
2. **Validation that is both structural and economic.** Structural: bounds, monotonicity, cap ≥ floor.
   **Economic: reject any config whose modeled daily net issuance exceeds a stated fraction of money
   supply.** (EVE currently runs +2.16%/month [V]; a config that would run +20%/month should not be
   applyable by accident.)
3. **Rollout as a fenced EPOCH**, reusing the existing `EpochId` + directory-CAS precedent [R]:
   `EconomyEpoch { epoch, tuning_hash, effective_at: UniverseTick, signer, ed25519_signature }` CAS'd
   into the directory; every shard flips at a **known universe tick**; **every emitted event stamps
   `econ_epoch`** so any analysis is always joinable to the config that produced it. Scope by
   realm-path prefix for staged single-region rollouts.
4. **Record the change itself as a `ConfigChanged` event in the same log** ⇒ the audit trail is
   self-analyzing, and the dashboard can answer "what changed when" and A/B a tweak against the journal.
5. **An INTERVENTION LOG designed at the same time as the tweak API**:
   `{tick, realm_subtree, param, old, new, actor, hypothesis}`. The OSRS causal study is the reason:
   the tax barely moved trading at the taxed price points and the item sink **raised** luxury prices
   [V] ⇒ without a measurement design, "apply tweaks" is unfalsifiable.
6. **A mutating surface separate from `/admin/*`.** That router is documented read-only **by
   construction**, loopback-only, with authentication still owed (D-13) [R]. A tweak endpoint needs
   authn/authz, an audit trail, and a two-person or delay-based safety on high-impact parameters.

**Rate ownership is a design decision** (D11 in §11): operator-only config, or player-settable by
whoever holds the realm (EVE's POCO/structure model). If player-settable, the ceiling/floor/top-up
machinery becomes load-bearing immediately.

### 8.7 Faster-than-real-time counterfactual simulation

**Every PART already exists** — `VirtualClock` (12 k ticks in milliseconds), `Topology` with
`trace()`/`trace_bytes()` (a byte-comparable execution trace), `FaultFabric`, `MemStore` seedable from a
redb `scan`, and a recorded `InputLog` [R] — **but "nearly free" was a 29× arithmetic error compounded by
an unmeasured tick rate**; see the corrected budget below.

**The recipe:**

1. **Snapshot** = redb `scan` + seed + `econ_epoch`.
2. **Fidelity gate (G-ECON-REPLAY)** — replay the recorded input log under the **unchanged** tuning and
   assert the emitted econ log is **byte-identical**. *Without this the twin is fiction.*
3. **Counterfactual** — re-run with `EconomyTuning'` and diff **aggregate series**, not events.
4. **Confidence bands** — run M seeds across M **processes** (one seed per process, embarrassingly
   parallel, no shared state). ⚠ **CORRECTION**: revision 1 said "`rayon` is already an approved dep [R]".
   **rayon is NOT a dependency of this workspace** — it appears in no `Cargo.toml`, and
   `docs/design/d6_saga_wal.md:123` says verbatim *"rayon remains un-adopted"* [V]. Adopting it is a user
   decision; process-level parallelism needs no dep at all and is the better fit here anyway (each seed is
   a whole simulation).

**The budget — ⚠ rebuilt; revision 1 was wrong by 29× on its own assumption and by 3–4 orders of
magnitude on a realistic one.** An econ-only `ShardProfile` (no rapier) is HR3-legal, which is what makes
this affordable at all. Revision 1 said: "30 days at 20 Hz is 51.8 M ticks, so at ~1 M ticks/s/core a
single 30-day counterfactual is under a minute of CPU, and **1,000 seeds fit in ~30 core-minutes**".

- 30 × 86,400 × 20 = **51.84 M ticks** ✓; ÷ 1e6 = **51.8 s** ✓ ("under a minute").
- But 1,000 × 51.8 s = 51,840 core-s = **864 core-minutes = 14.4 core-HOURS** — **off by ~29×.**
- And **1 M ticks/s (1 µs/tick) is extrapolated from a harness containing no economy**. With even 5,000
  phase-decimated agent evaluations per tick at 300 ns each, that is 1.5 ms/tick ⇒ **~670 ticks/s ⇒ 21.6
  hours per 30-day seed ⇒ 1,000 seeds ≈ 2.4 core-YEARS.**

**The fix is the cadence hierarchy (§2.3b), and with it the twin genuinely is cheap.** The twin does not
run at 20 Hz — it runs at the economy's cadence:

| Horizon × cadence | Ticks | Per seed (at a measured ticks/s) | 1,000 seeds |
|---|---|---|---|
| 30 days at **20 Hz** (revision 1's implicit assumption) | 51.8 M | 52 s optimistic / **21.6 h** realistic | 14.4 core-h / **2.4 core-yr** |
| 30 days at **`clearing_period_ticks` = 100** (5 s) | **518 k** | ~0.5 s / ~13 min | ~9 core-min / **~9 core-days** |
| 30 days at **1 Hz** ambient repricing | **2.6 M** | ~3 s / ~65 min | ~50 core-min / ~45 core-days |
| 30 days at **`dormant_catchup_tick_period`** (1 universe-hour) | **720** | **microseconds** | **seconds** |

⇒ **State the horizon, the cadence and the seed count as three explicit parameters**
(`twin_horizon_days`, `twin_cadence_ticks`, `twin_seeds`), and **drop the 1 M ticks/s figure entirely** in
favour of a measured ticks/s from the E-(−1) spike. A defensible default: 30 days at the clearing cadence,
64 seeds ⇒ well under a core-hour.

**The honest limitation, to state up front.** Replay-counterfactuals answer **mechanical** questions
correctly (tax yield, sink strength, fee incidence, index response) but **not behavioural** ones
(players stop trading when you raise the fee). For those you need an elastic-demand agent overlay
(krABMaga as a *reference*, or ~500 lines of our own deterministic agents) — and it must be labelled a
**model, not a prediction.** EVE's Scarcity intervention is the reason this matters: it stopped wars,
docked capital fleets and slumped activity for years [V]/[L].

### 8.8 HR6: the agent-operable economy surface

⚠ **HR6 is a HARD RULE and revision 1 mentioned it twice, both times as "an `EconSnapshot` endpoint is
agent-operable".** That is not the project's e2e discipline. Every landed subsystem here is gated through
`vdctl` + `DevState` counters + a `WaitField` arm — the FA-5 S0 pattern (`realm_frames_applied` +
`WaitField::RealmFramesApplied`, with a closed-loop e2e blocking on `realm_frames_applied ge 1` before
capturing) [R]. Without this, **no economy slice as written could pass the project's own gates**, and it
collides with the standing "don't commit broken / the user confirms it works in-game" rule.

**(a) `vdctl` verbs** (the request/response shapes are `vd-devproto` serde types, the same the client
decodes, so the two cannot drift [R]; the CLI help is derived from `WaitField::ALL`/`WaitOp::ALL` so it
can never go stale [R]):

| Verb | Purpose |
|---|---|
| `econ-balance [--account <id>] [--currency <id>]` | print `{posted, pending, available}` — the server-computed triple, never a client-derived one (§7.17) |
| `econ-order-place <venue> <commodity> <side> <price> <qty>` | returns the `Accepted{order_id, idempotency_key, fence}` receipt |
| `econ-order-cancel <order_id>` | idempotent; returns the stored receipt on redelivery (N6) |
| `econ-trade-offer / econ-trade-confirm / econ-trade-cancel` | drives the §7.8 direct-trade FSM from a script — this is the closed-loop two-client acceptance test |
| `econ-contract-post / econ-contract-accept / econ-contract-deliver` | §7.9 |
| `econ-assert-conservation [--currency <id>] [--subtree <path>]` | runs the reconciliation sweep on demand and exits non-zero on a residual — **the single most valuable verb**, because it makes G-ACCOUNTING runnable against a live cluster, not only in the harness |
| `econ-dump-log [--since-tick N] [--kind K]` | the journal as JSON lines, for an agent to assert over |
| `econ-epoch-show` | the active `EconomyEpoch` + `tuning_hash` + `effective_at` (§8.9) |

**(b) `DevState` counters** (following the existing classification discipline — a nonzero **FAULT**
counter is a real problem, **THROUGHPUT**/**BENIGN** ones are not [R]):

| Counter | Class |
|---|---|
| `econ_postings_applied` | THROUGHPUT |
| `econ_orders_accepted`, `econ_fills_applied`, `econ_clearings_run` | THROUGHPUT |
| `econ_actions_refused_budget` (the §8.1.1 shed) | BENIGN (the backpressure working) |
| `econ_actions_refused_stale_fence` | BENIGN |
| `econ_conservation_residual_minor` | **FAULT — must be 0** |
| `econ_incidents` (§6.6) | **FAULT** |
| `econ_markets_halted` | **FAULT** |
| `econ_stranded_in_transit` | **FAULT** |

**(c) `WaitField` arms** (append-only, and the enum's `is_boolean()`/`ALL` discipline must be extended in
the same commit [R]): `EconPostingsApplied`, `EconFillsApplied`, `EconConservationResidual` (a
0/1 *flag*, so `wait econ-conservation-residual eq 0` is expressible), `EconOrdersAccepted`.

**(d) A `runs/` manifest for economic scenarios**, mirroring the existing capture manifests: seed,
`econ_epoch`, `tuning_hash`, the input log digest, the emitted-journal digest, and the final conservation
tuple — so an economic run is **reproducible by artefact**, which is also exactly what G-ECON-REPLAY
needs.

**(e) A per-slice "works in-game" criterion — a COLUMN in the slice table (§10.5), not an afterthought.**
The canonical one: **two real clients over live QUIC complete a P2P trade and both wallets settle**,
asserted by `vdctl econ-balance` on both sides plus `econ-assert-conservation`, with a screenshot at the
aligned tick. Per the standing test-exactly-production law this must drive the **real** shipped path — no
stand-ins, no override hacks.

### 8.9 Where `EconomyTuning` lives, how it loads, and how it reconciles with the epoch

⚠ **Revision 1 named ~80 tunables across §4 and never placed the struct in a crate**, never said how it
is parsed at boot, and never reconciled the **two competing sources of truth** it proposed: a config-file
struct (§8.6 step 1) versus the directory-CAS'd `EconomyEpoch { tuning_hash, effective_at }` (§8.6 step 3).
A shard booting with a config file that disagrees with the committed epoch is an **unspecified, silently
divergent state** — i.e. two shards running different economies while both believing they are correct.

| Question | Answer |
|---|---|
| **Where does the TYPE live?** | `vd-core` (pure data, `serde`, no I/O) — so `vd-econ` (Tier-A), `vd-sim`, the harness and the Tier-B twin all reference **one** definition. Precedents: `TransportTuning` / `TransferTuning` [R]. |
| **Where does the VALUE come from?** | Parsed **once at boot in the bin** (io-prod tier, where I/O legally lives) from the same config path as the existing tuning structs, then passed **by value** into `build_app` — never read from the environment inside sim/node. |
| **Validation** | Two layers, both at parse time: **structural** (bounds, monotonicity, `cap ≥ floor`, `Σ` sanity) and **economic** (reject any config whose modelled daily net issuance exceeds the §4.10c band). A failed validation is a **boot refusal**, loud, with the offending field named. |
| **The reconciliation rule (the gap)** | **The committed `EconomyEpoch` WINS, and a mismatch is a BOOT REFUSAL, not a silent override.** At boot the shard computes `tuning_hash` over its parsed config and compares it to the directory's committed epoch: equal ⇒ proceed; **different ⇒ refuse to serve economic traffic** (it may still simulate physics) and report `econ_config_mismatch` as a FAULT counter. An operator rolling out new tuning therefore always follows the §8.6 order — CAS the epoch first, then roll the shards — and a half-rolled fleet is *visibly* half-rolled rather than quietly bifurcated. |
| **Scoped rollouts** | The epoch record carries a **realm-path prefix scope** (§8.6), so a shard's applicable epoch is the deepest committed epoch whose prefix matches its realm path — resolved with the same `path_index` everything else uses [R]. |
| **Hot changes** | Never in-place. A change is a **new epoch with a future `effective_at`**; every shard flips at a known universe tick, and every emitted event stamps `econ_epoch` (S4) so analysis is always joinable to the config that produced it. |

---

## 9. Integrity

### 9.1 Exploit catalogue → structural defence → detection signal

| Exploit | Structural defence (prevents) | Detection signal (catches what slips) |
|---|---|---|
| **Duping via copy-then-delete** | A move is one entry-set summing to zero; there is no "copy" primitive | I1 fails the tick it happens; `Σ count` per `ItemKind` drifts |
| **Duping via two live sessions** | Ed25519 `SessionTicket` with `conn_binding`; the gateway owns the single connection for the session [R] | Two `Owned` claims on one `Session` key ⇒ `verify_authority_unique` |
| **Duping via state-machine hole** (e.g. a packing dialog open during a trade) | `applied_steps` + fence on every authoritative action; escrow means the item is already committed | I8 (a reservation with two terminals) |
| **Duping via forced server change with the old connection open, the old server's later save overwriting the new** — historically the dominant cause | **Dead by construction**: gateway-owned single connection, silent server-side route swap, ordered demote-before-promote, directory CAS as the only commit point, and the gateway lease-epoch frame fence DROPS a stale source's frames [R]. This is R1/R2 in our own audit vocabulary | I6 + I9 |
| **Duping via opaque blobs with application-level uniqueness** | The **ledger** is the uniqueness authority, never the blob; `ItemId` minted like `EntityId`, never time-derived, never reused | I10: every `ItemId` in exactly one container at every committed instant |
| **NPC price seam / money printing** | Finite-quantity NPC anchors with a replenishment budget; no infinite fixed-price orders | I3/I5: an undeclared change to the global sum is a hard failure. This is what caught nothing at CCP for four years [L] |
| **Rounding-dust harvesting** | ONE rounding point, direction always toward the sink, dust posted to a declared sink account | I1 fails intermittently if violated ⇒ make it a proptest, not a hope |
| **Wash trading / self-trade** | Self-match prevention keyed on the owning legal entity (character → corp → alliance) | A→B→A within a window at off-band prices; wash rings as SCCs in the trading graph (petgraph) |
| **Spoofing / layering** | Order-modification fee + tick size + per-actor cap | Place/cancel ratio and time-to-cancel distributions |
| **Market cornering / manipulation** | Per-item quantity limits per time per account (RuneScape's post-2011 answer); price bands + volatility interruptions | HHI counterparty concentration; robust-z/MAD price outliers vs the regional median |
| **Bot arbitrage at zero margin** | A non-zero per-order cost | Order-rate percentile per account; place/cancel entropy |
| **RMT (gold selling)** | Nothing structural prevents it; make it *visible and expensive* | The published gold-farming literature is explicit that the winning features come from the **trading graph** plus connection patterns: farmer→broker→buyer bipartite motifs, low-diversity high-throughput nodes, assortativity, community detection. We can additionally add **session/gateway co-occurrence edges** that most games cannot [L]. Plus: new-account inflow percentile (the buyer signature) |
| **Unfunded compensation minting** | Payout pools are real balances funded by premiums; overdraw fails loud | I3 |
| **Fraudulent player banks / equity** | Server-ledgered obligations, solvency checks, M-of-N officer signatures, present-balance disclosure | Treasury-vs-liability ratio panel; withdrawal-concentration alerts |
| **Operator error / insider abuse** | Tweaks are fenced, signed, epoch-scoped events with an intervention log; two-person or delay-based safety on high-impact parameters | The `econ_epoch` timeline with every intervention annotated; `OperatorFaucet`/`OperatorSink` totals as a first-class dashboard series |

**Tiered detection** (the tiering matters as much as the signals):

| Tier | Where | What | Cost |
|---|---|---|---|
| **0 — deterministic invariants** | In-shard, in-tick | Conservation, non-negative balances, escrow closure, fence monotonicity | **The highest-value control by a wide margin**: catches duping the tick it happens |
| **1 — statistical batch** | Offline over the event log | Benford on trade amounts; wash-trade windows; self-trade; place/cancel and time-to-cancel distributions; robust-z/MAD price outliers vs regional median; HHI concentration; new-account inflow percentile. Streaming aids: `tdigest`, `sketches-ddsketch`, `hyperloglogplus` | Cheap; all integer-expressible |
| **2 — graph** | Offline, `petgraph` (already approved) | Bipartite farmer/broker/buyer motifs, wash rings as SCCs, low-diversity high-throughput nodes, assortativity, community detection, session co-occurrence | Moderate |
| **3 — offline ML** | Advisory only, human-gated | extended-isolation-forest (port candidate), `augurs` for series outliers, `linfa` for clustering | **Every model score is written back into the log as an event**, so enforcement decisions are themselves auditable |

### 9.2 Moderation tooling requirements

1. **Case view**: for any account, the complete `econ_posting` history with the `(event_kind, context_id,
   context_id_type)` provenance triple resolved, plus the counterparty graph neighbourhood.
2. **Reversal as a first-class ledgered operation**: a confiscation/rollback is
   `Sink::AdminConfiscation` / `Faucet::AdminGrant` with an operator identity and a case id — **never a
   balance edit**, so the accounting identity never needs a plug (unlike CCP's "Active ISK Delta" [V]).
3. **Bulk action with a dry-run**: apply to a cohort, preview the aggregate effect, then commit as one
   batched multi-party operation (§4.7).
4. **Immutable audit of moderator actions**, queryable in the same dashboard as player activity.
5. **Appeal support**: because the log is complete and replayable, "prove this account earned that ISK"
   is a query. That is a direct product benefit of I12.
6. **Enforcement volume planning**: EVE banned **3,165 accounts for macro use in June 2026 alone** [V] ⇒
   this is a permanent operational load, not an incident response.

### 9.3 Legal-edge note (factual only — this needs real legal advice)

**This section flags issues, it does not give advice. Anything in it that becomes load-bearing must be
reviewed by a qualified lawyer in the relevant jurisdictions.**

| Fact | Why it matters |
|---|---|
| Second Life ran a floating internal↔real currency market (LindeX) with real cashout, and after Ginko Financial's collapse Linden Lab restricted in-world banking [V for the collapse; the exact 2008 rule text is **[U]**] | A game that permits *cashout* has historically been forced into financial-institution-shaped rules by its own incidents. |
| Entropia Universe operates a **hard peg with real withdrawals** (10 PED = US$1) [V] | Real-money redemption is the bright line that moves a virtual economy toward money-transmission/e-money regulation in many jurisdictions. |
| EVE has no official cashout; PLEX is a one-way purchase that trades internally, and CCP treats RMT as a bannable offence with published enforcement [V] | The "no official cashout" model appears to be the lower-exposure design. |
| Tradable equity + dividends + player-run deposits is *structurally* a securities-shaped feature set | Even without cashout, "shares paying dividends" invites scrutiny, and (per §4.7) the historical record is that every player financial institution collapsed. |
| Loot-box / lottery mechanics (EVE's Hypernet Relay is a 5%-taxed lottery [V]) | Gambling regulation varies enormously by jurisdiction and is actively changing. |
| Consumer/tax exposure from in-game currency purchases and refunds | Independent of the game design; a business question. |

**Practical recommendation:** design so that **no real-money redemption path exists** and equity is
purely internal. That keeps every mechanism in §4.7 available as *gameplay* while staying far from the
regulated perimeter, and it is a decision that is cheap to make now and expensive to reverse.

### 9.4 Economic incident response (the runbook)

⚠ **Absent from revision 1**, which designed the *planned* tweak loop (epochs, intervention log,
two-person rule) and listed moderation *tooling*, but had no procedure for a live duping exploit or a
runaway faucet at 03:00. §9.1's own last row calls operator tweaks "a systemic-risk instrument" and stops.

**Severity classes and the automatic response** (the automatic half is §6.6; this is the human half):

| Class | Example | Auto | First human action |
|---|---|---|---|
| **S1 LEAK** | conservation residual ≠ 0 in a subtree | `HALTED` flag on the affected currency × subtree (§6.6) | run the standing triage query set; identify the arm |
| **S2 DUPE** | I10 fires, or `Σ count` per `ItemKind` drifts | quarantine the `ItemId`; refuse the mutation | freeze the mechanism (an epoch with the feature's cap at 0), not the whole economy |
| **S3 SEAM** | an NPC buy ≥ NPC sell across a chain | the finite-quantity budget bounds the damage by construction | set that anchor's replenishment budget to 0 via an epoch |
| **S4 BOT-FARM** | order-rate / inflow percentile alarms | none (statistical, offline) | case view + graph neighbourhood; enforcement via `Sink::AdminConfiscation` |
| **S5 OPERATOR ERROR** | a bad epoch applied | boot-refusal on mismatch (§8.9) catches the half-rolled case | CAS the previous epoch forward (never "roll back" — §6.6) |

**Halt primitives, and who may invoke them.** Each is a **fenced, CAS-committed, signed, epoch-scoped,
audited** record — never a config poke and never a balance edit:

| Primitive | Scope |
|---|---|
| `HaltMarket(RealmId, CommodityId)` | one book |
| `HaltVenue(RealmId)` | one venue |
| `HaltCurrency(CurrencyId, realm_path_prefix)` | mints/burns in a subtree |
| `HaltMechanism(FaucetId \| SinkId)` | one arm, via an epoch setting its budget to 0 |
| `HaltAllTrade` | the break-glass; **two-person rule mandatory** |

⚠ **The authenticated surface is a BLOCKER, not a detail.** The existing `/admin/*` router is documented
**read-only by construction**, loopback-only, with **authentication still owed (D-13)** [R]. **D-13 is
therefore a hard blocker on ANY mutating economy surface** — halt primitives, epoch CAS, reversals — and
must be raised to that status in the ledger. A mutating economy endpoint bolted onto the read-only admin
shell would be the single worst security decision available here.

**The standing triage query set** (written once, against the journal, so an incident is a query):

1. `Δ money_supply` vs `Σ faucets − Σ sinks` per currency per hour, with the residual by **arm**.
2. Per-`Faucet` arm totals vs their I16 **budgets**, ranked by utilisation.
3. Top accounts by inflow percentile in the last N hours, with the provenance triple resolved.
4. New-account inflow percentile (the RMT-buyer signature) and session/gateway co-occurrence edges.
5. Wash-ring SCCs and self-trade attempts in the window.
6. `econ_incidents` grouped by invariant and shard.
7. Stranded `InTransit` balances and escrow age distribution.

**Compensation without minting.** A rebate/compensation is a **funded `Transfer` from a declared reserve
account** (`Reserve(CompensationPool)`), never a `Faucet` — otherwise the compensation for an incident is
itself an unmeasured faucet, which is the §4.9 "unfunded compensation" row applied to ourselves. The pool
is funded from sinks by policy and its balance is a first-class dashboard series. If the pool cannot cover
the cohort, the shortfall is **declared and visible**, and topping it up is an explicit `Faucet::AdminGrant`
with a case id — loud, attributable, and counted.

**Player communication + the post-mortem artefact.** Every incident action (halt, un-halt, reversal,
compensation, epoch change) is an event in the **same journal**, so the post-mortem is a query and the
public write-up is generated from it — the `econ_epoch` timeline with every intervention annotated
(§8.5). This is also the only way an intervention becomes *measurable*, which is the whole point of the
OSRS caution [V].

### 9.5 Authorization and authenticity of economic commands

⚠ Revision 1 specified M-of-N officer signatures and a signed `EconomyEpoch` — the exceptional paths — and
nothing about the **ordinary** one: which identity a market order is attributed to, how a gateway session
becomes an `AccountId`, and what stops a compromised or buggy shard from emitting a credit for an account
it does not own. I6 asserts single-writer; nothing bound an *emitted* economic command to the emitting
shard's authority.

| Question | Answer |
|---|---|
| **Session → `AccountId`** | the gateway already owns the single connection per session with an Ed25519 `SessionTicket` + `conn_binding` [R]. `AccountId` is **derived from the durable player identity** (`identity_persistence.md`'s identity record), resolved **once at admission** by the gateway and carried on every client command — the client never *names* an account, it names an *action*, and the account is looked up server-side. A client-supplied `AccountId` must be rejected outright. |
| **Order attribution** | an order carries `{actor: AccountId, on_behalf_of: Option<CorpId>, division}` — EVE's `is_corporation` flag generalised (§4.7) — and the venue **re-derives** the actor from the session at admission rather than trusting the payload. `on_behalf_of` is authorised by the corp permission bitmask (§7.11) at the corp's own authority, not at the venue. |
| **Shard → shard authenticity (the real gap)** | **every economic command carries the emitting shard's authority `Fence` for the subject key, and the receiver rejects a command whose emitter does not hold `Owned` for that key.** That is the mechanism binding an emitted credit to the emitter's authority: a shard can only mint/move value for keys it owns, and ownership is the directory's CAS'd `OwnerRecord`. This is not new machinery — it is the existing fence discipline applied to a new subject family — but revision 1 asserted I6 without stating the *enforcement* side. |
| **Mesh-level authenticity** | shard↔shard traffic already runs over the mTLS mesh [R], so a *foreign* process cannot inject; the fence rule above is what protects against a **legitimate but wrong** shard (a stale incarnation, a buggy feature, a compromised node). |
| **Operator commands** | Ed25519-signed, epoch-scoped, two-person for high impact, and gated behind **D-13** (§9.4). |
| **Invariant** | folded into **I6/I9**: a command mutating account or market state is rejected unless `(emitter holds Owned for the key) ∧ (fence ≥ highest_seen)`. Counted, not silently dropped. |

---

## 10. Roadmap + planted seams

### 10.1 Where this sits

Current position: P0–P3 machinery is largely landed (transfer saga, fences, directory CAS, saga WAL +
redb, transient batches, kill-9 crash matrix, RLM Steps 1–5e) [R]. The roadmap ahead is P4 terrain →
P5 physics → P6 block edits + persistence → P7 checkpoints → P8 ships → **P9 functional blocks +
signals** → P10 warp → P11 combat + load [R].

**The economy's natural home is P9-adjacent**, for three concrete reasons:

1. `Signal` is the reserved carrier the design already nominates for gameplay-level cross-shard data
   including economy [R], and the economy needs it for price feeds, accrual notices and corp events.
2. The economy's most interesting content (thruster-block signals, factories, station logistics) needs
   P6 blocks and P8 ships to exist.
3. The Signal seam is explicitly meant to be **designed before P6** [memory record] — which is exactly
   when the economy's cross-shard requirements should be fed into it, so the arm is shaped once.

**But there are seams that must land BEFORE P4/P6/P11**, because retrofitting them is a versioned wire
migration or a live-economy data migration. ⚠ **Revision 2 moved four items earlier than "P9-adjacent",
and they are not all cheap:** **S12** (closed-form ore distribution) must precede **P4 terrain**; **S11**
(the inventory-as-position rule) must precede **P6/P7**, whose roadmap row already promises resuming
"inventory"; **S13** (the declared destruction sink) must precede **P11**; and **S5b** (the per-realm venue
capability on the RLM spawn path) is a real RLM slice rather than a 30-line field. Separately, **D16**
(does mined terrain regenerate?) is a **P4 design decision**, not a P9 one — the material economy's
geometry is fixed by the terrain that ships (§4.10).

### 10.2 Land EARLY and cheap (the anti-cornering list)

Each item is small, has independent value, and is expensive-to-impossible to retrofit later.

> ⚠ **WORLDLINE (rev 2) — READ FIRST.** `scripts/dormant_world_simulation_design.md` §11.1/§11.2 supersedes or
> constrains **S1, S2, S3, S4, S6, S7, S8, S9, S11, S12, S13** below, and adds pre-P4 seams this list does not
> have (`RealmUid` + `MAX_REALM_DEPTH`, the integer arithmetic kernel, RULE WL-INTQ + `content_epoch`, LAW-WL-8,
> `is_session_occupant`, `EconomyPort`/`EconCommand`, RULE WL-LIVENESS, RULE WL-LIEN, RULE WL-CONSERVED-FACT,
> RULE WL-AGGREGATE, RULE WL-SETTLED-RATE, TLV-framed durable records). Where the two disagree, the worldline
> document is the later analysis and it cites the code; the per-row notes below say what changed.

| # | Seam | Where | Cost | Why it cannot wait |
|---|---|---|---|---|
| **S1** | `Money(i128, CurrencyId)` + `CurrencyId` with `minor_exponent` + integer-bp rate arithmetic + `muldiv` with a declared rounding direction + largest-remainder `allocate` | `vd-core` | ~300 lines, 0 deps | The wire/TLV representation of value freezes with the first use. Getting it wrong is a migration of every historical record. ⚠ **WORLDLINE (rev 2) — CONSTRAINED**: the `muldiv` half is **subsumed** by the worldline's integer arithmetic kernel (`dormant_world_simulation_design.md` §4.2, §11.1), which lands **pre-P4 for the physical layer** and therefore before the economy. Adopt its signatures verbatim — `muldiv_floor/ceil(a: u128, b: u64, c: u64)` plus signed variants, **not** `(u128, u128, u128)` (which needs a 256-bit intermediate and is unimplementable as one 128-bit divide) — and its **derived** overflow caps rather than named ones. |
| **S2** ⚠ **re-aimed** | **DESIGN THE ITEM-STACK TLV SCHEMA**, with `tax_credit_minor`, the Veloren price/supply vector (§7.2) and the position/`ItemId` fields as day-one tags | `vd-core` schema + `vd-wire` tag registry | ~150 lines (schema + codec + tests), not ~20 | ⚠ **CORRECTION**: revision 1 reserved a tag number "on the item-stack blob schema" — **there is no item-stack blob schema.** `TransferableKind`'s behaviour half is unbuilt and `StubCrossing.state` is produced as `vec![]` (D-31, conceded in revision 1's own §7.6), and the per-kind version-floor handshake that would read the tag is explicitly "P-later" (`crates/wire/src/intershard.rs:111`) [R]. **Reserving a number in a nonexistent schema buys nothing.** The real owed artefact is the schema itself, as D-31's first consumer; then decode-to-Default being BANNED for Durable kinds [R] does the work revision 1 wanted. ⚠ **WORLDLINE (rev 2) — CONSTRAINED, with a hard rule.** `dormant_world_simulation_design.md` §7.3(b): the item tag must be **OPTIONAL and forward-skippable**, and a `WorldEpoch`/schema mismatch must degrade to *carry the bytes verbatim, opaque* — **never** `Refused`. A **required** epoch-stamped inventory tag turns a routine rate rebalance or a rolling deploy into a **movement outage** (re-home IS the docking primitive), because PREPARE refuses at the version floor. Also: the binding size cap is `KindDef::max_state_bytes` — `PLAYER_DEF = 4096` B, `SHIP_DEF`/`NAMED_CONSTRUCTION_DEF = 8192` B — **not** `MAX_FIELD_BYTES`, so the in-blob inventory is a few tens of stacks and must be enforced at MUTATION time (§7.3(a), §8.3). |
| **S3** | **Recipe-DAG acyclicity validation** at registry load, in the same shape as `ShardProfile::build`'s lattice validation | `vd-core`/`vd-sim` | ~60 lines | Enforces the property that makes the whole Leontief/VAT layer integer, dependency-free and Tier-A. If cyclic recipes are ever authored *accidentally*, the tax layer's stage-invariance silently dies and sparse float linear algebra becomes mandatory. ⚠ **WORLDLINE (rev 2) — PLACEMENT FIXED**: the recipe registry **and** its acyclicity validator must live in the **GAME** (`vd-core`, next to `KindDef::is_coherent`), never in `vd-econ` — otherwise crafting stops when the economy stops, a LAW-E1 violation (`dormant_world_simulation_design.md` §2 row 11). |
| **S4** ⚠ **re-homed** | **`econ_epoch: u32` stamped on the economy event record from the first event**, **plus the cadence fields** (`clearing_period_ticks`, `agent_decision_period_ticks`, `ambient_reprice_period_ticks`, `dormant_catchup_tick_period`) — the cadence is baked into what an event *means* | **`vd-core`** (the record type) + `vd-econ` — **NOT `vd-wire`** | ~30 lines | ⚠ **CORRECTION**: revision 1 put this in `vd-wire`, but §8.1 insists the econ record must **not** be an `InterShardFlow` arm (it is a shard→warehouse fan-in). The record belongs in `vd-core`/`vd-econ`. Every analysis must be joinable to the config **and the cadence** that produced it; adding either later orphans all prior history. ⚠ **WORLDLINE (rev 2) — BROADENED THREE WAYS**: `econ_epoch` becomes a world-wide **`WorldEpoch`** in `vd-core` stamped on every durable record with **settle-on-change** (the physical layer needs it first); a separate **`content_epoch`** is owed because P4 changes the generator after the first baselines land; and a **`FORM_VERSION`** per closed form is owed because `WorldEpoch` versions the *config*, not the *math*, so a rolling deploy could otherwise integrate one interval with two different closed forms (`dormant_world_simulation_design.md` §9.1, §9.2, §7.5). |
| **S5a** | **The `market_venue`/`economy` field on `ShardProfile`, defaulted false, with its lattice rule** | `crates/sim/src/capability.rs` | ~30 lines + tests | Proves at design time that the economy is a *capability*, never a shard kind (HR3), and gives the G5 fixture a place to attach. Follows D-39.4's prescribed pattern verbatim [R]. |
| **S5b** ⚠ **new — the half revision 1 missed** | **A PER-REALM capability override carried on the RLM spawn path** (a seed-derived venue flag from the generator → `spawn_realm`, surviving rehydrate) | `vd-core` generator + `crates/sim/src/io/mod.rs` `RealmSpawner` + the reconciler's rehydrate | **a real RLM-touching slice — LEDGER IT**, not ~30 lines | `ShardProfile` today is a pure function of realm **KIND** (`profile_kind()` → `profile_kind_of(RealmKindTag)` → `profile_for`, and `spawn_realm` derives the profile from the coord alone) [R] ⇒ **S5a alone makes ALL stations venues.** That defeats the entire basis of D1(d) ("real order books only where a venue capability is enabled") and of G5's fixture. Until S5b lands, venue-ness is per KIND and D1(d) must be re-checked under that constraint (§7.5). ⚠ **WORLDLINE (rev 2) — A SECOND CONSUMER**: the coarse agent tier (W-5 / D-71) also needs a **per-instance** capability and structurally cannot satisfy G-IDENTICAL without one. Until S5b lands its interim is an **injected object-safe port wired only in `vd-bins`** from the static pin config, with G-IDENTICAL declared as *"the identical fixture passes on ≥2 kinds with the port ABSENT"* (`dormant_world_simulation_design.md` §9.3). |
| **S6** | **`Account(AccountId)` reserved as a `DirectoryKey` arm** (inert) | `vd-wire` `DirectoryKey` | ~10 lines | `DirectoryKey` is a small closed enum consumed by `transfer_subject_entity` with an exhaustive match [R]; adding an arm later touches every match site. Reserving it inert costs nothing and makes single-writer-per-account free later. ⚠ **WORLDLINE (rev 2) — REJECTED, and REPLACED.** The directory CAS is the only commit point, so an economy key there makes an economy fact an **authority input** — the coupling LAW-E1 forbids (`dormant_world_simulation_design.md` §8.4, W13). The replacement, which is strictly better: an account's durable **position** is a **zero-rate worldline subject** at the custodian (§6.7), so single-writer-per-account is a property of the fence-ordered absolute-rebase log rather than of a `DirectoryKey` arm — and it also answers **D3** for wallets and treasuries without D3's blocked storage prerequisite. `Market(RealmId, CommodityId)` is untouched. |
| **S7** | **The closed `Faucet`/`Sink` enum + the `(event_kind, context_id, context_id_type)` provenance triple** | `vd-core` | ~80 lines | If any code path outside a named arm can change a balance, the accounting identity stops being CI-assertable and we inherit CCP's forensic problem permanently. Establishing the closed taxonomy *before* the first faucet exists is the only cheap moment. ⚠ **WORLDLINE (rev 2) — SPLIT INTO TWO REGISTRIES.** One enum spanning physical loss channels and monetary sinks means a physical durable record references the monetary taxonomy: a monetary renumbering mis-interprets historical **physical** rows, and a monetary registry change bumps the epoch and rewrites **physical** intervals. Owed: **`PhysicalLossChannel`** in `vd-core` (the only thing a worldline record may name) and **`MonetarySink`** in `vd-econ`, which may project from it but is never referenced by a physical row — with an exhaustive coherence test next to `KindDef::is_coherent` (`dormant_world_simulation_design.md` §9.4). |
| **S8** | **`verify_value_conservation` + G8 anti-theater** (a deliberate 1-unit imbalance must be DETECTED) | `vd-harness` | ~150 lines | The gate must exist before the first feature, or the suite becomes decorative — the exact failure the D-6 no-op-stub-`Store` guard was written to prevent [R]. ⚠ **WORLDLINE (rev 2) — TWO ADDITIONS.** (a) `verify_item_conservation` is provable **before the economy exists**, over captured `InspectReport`s, as a sibling of three landed oracles (`dormant_world_simulation_design.md` §8.4). (b) The identity must be `Σ mint − Σ burn − Σ **declared** loss == Σ positions` **and** every conservation-bearing fact must live in the non-sheddable STATE record (RULE WL-CONSERVED-FACT, §2): a sheddable journal records a gap as a **count**, and a count cannot repair a per-`ItemId` identity — so the gate would be green precisely in the arm where the hazard is absent. Add a **mid-scenario shed cell** with a RED control. |
| **S9** | **`expiry_universe_tick` on any order-like record**, and the rule "escrow is held by the ACCOUNT owner, never the market host" written into the design | design + types | ~0 lines now | These are the two rules that make RLM teardown safe. Discovering them later means a live migration of escrow ownership. ⚠ **WORLDLINE (rev 2) — THE PHYSICAL HALF IS SETTLED, INDEPENDENTLY OF D15.** **RULE WL-LIEN**: an economy object may **never** be the container of record for an item position. Escrow or collateral over **goods** is a lien — a claim in `vd-econ` referencing a stack whose position stays in a **GAME** container — so "the economy is absent" means "the lien is unenforceable", never "the goods are nowhere" (`dormant_world_simulation_design.md` §2 row 28). D15 remains open for the **monetary** half only. |
| **S10** ⚠ **broadened** | **Feed the economy's cross-shard requirements into the P9 Signal design**: price digests as `EffectFree`; the explicit rule that Signals never carry authority-gating economic state **nor the credit half of an applied debit** (§4.8 T8); **and the open requirement of whether System/Planet realms need a relay or processing capability at all** — today only `profiles::galaxy()` and `profiles::station()` carry `signal_relay` [R] | `docs/design/` + the Signal design + possibly `profiles::system()`/`planet()` | design only | The Signal arm should be shaped ONCE, with all its consumers known. This is the single highest-leverage design coupling in the whole report — **and revision 1's version rested on a relay guarantee that does not exist** (§1.4, §6.4). |
| **S11** ⚠ **new — the largest retrofit removed** | **The FIRST line of inventory code (P6/P7) must represent a stack as an APPEND-ONLY POSITION**, with `ItemId` minted like `EntityId` (`{kind, mint_shard, seq, rand}`, never time-derived, never reused) and split/merge as zero-sum entry-sets — **the ledger, never the blob, is the uniqueness authority** | a written rule in `docs/design/` + the `ItemId` minting shape in `vd-core`; **cross-referenced into the P6/P7 roadmap rows** | **~0 lines of implementation now** (a rule + an id type) | ⚠ **P6 (block edits) and P7 (checkpoints — whose roadmap row explicitly promises resuming "inventory") land BEFORE the economy arc** [R] and will otherwise ship a **mutable-count** inventory first. Revision 1 scheduled items-as-ledger-positions at E-3, i.e. **after** the thing it must constrain. Retrofitting a live inventory is the largest single migration in the plan and it is removed for free by writing the rule down now. Give it a DEFERRED pin. ⚠ **WORLDLINE (rev 2) — RESOLVED, AND THIS ROW'S CENTRAL CLAUSE IS INVERTED.** *"The ledger, never the blob, is the uniqueness authority"* is **inverted coupling** under LAW-E1: if items ARE ledger entries, an economy outage stops a player mining a rock. **The append-only-POSITION DISCIPLINE and the `ItemId` minting shape land in the GAME's `vd-core`/store; the economy is a READER/projection** — "the discipline, not the dependency". No-dupe is then provable with the ledger entirely out of the loop, as a sibling of `verify_authority_unique` / `verify_transient_conservation_tick` / `verify_transient_loss_budget` / `verify_input_conservation`. Two further corrections: conservation can **never** be strict `Σ in == Σ out` (the game is already licensed to destroy in-flight items — `DROPPED_BLOCK_DEF` is `LossBudget(2)`, and `TransientAbandon` exists for exactly that), so the identity carries `Σ declared_loss`; and the recommended durability answer is the third option this row omits — **items inherit their CONTAINER's durability**, so only loose world drops are lossy. See `dormant_world_simulation_design.md` §8 in full, and note that this row's *"~0 lines of implementation now"* is true for the rule + the id type and **false** for the shard-side durable substrate it presupposes (a real slice, W-6). |
| **S12** ⚠ **new (pre-P4)** | **Ore/resource distribution as closed-form `f(seed, realm_path, voxel_pos)`** | `vd-core` generator | ~120 lines | The extraction faucet must be deterministic **and** computable before the economy exists, or the net-issuance target (§4.10c) can never be set. Same discipline as celestial math (Category A). A P4 decision. ⚠ **WORLDLINE (rev 2) — PROMOTED TO A HARD PREREQUISITE AND BROADENED.** The production integral's initial stock `S₀` **IS** this function, so the worldline *requires* it rather than benefiting from it. And it must be broadened from ore to **every** worldline input (`K`, extents, construction-slot positions, waypoint coordinates, rate numerators) under **RULE WL-INTQ**: one named quantisation function at the generator boundary. Reason: those inputs are **f64/libm** today (`crates/core/src/taxonomy.rs:8-25`), which would promote SPIKE-6a from determinism-*hygiene* to **authority**-load-bearing, since three different binaries independently recompute them (`dormant_world_simulation_design.md` §9.1). A **cross-binary** determinism gate is a W-0 exit criterion. Also owed with it: **LAW-WL-8** — the voxel edit log is the authority for geometry and the scalar deposit stock is DERIVED from it at declared breakpoints (§7.6). |
| **S13** ⚠ **new (pre-P11)** | **The destruction path emits a declared `Sink::Destruction` per destroyed block from its FIRST commit** | `vd-sim` block/damage path + `vd-core` sink registry | ~40 lines at the emit site | If the primary sink is not declared from day one it is **unmeasurable**, and no material balance is assertable. Retrofitting means re-deriving history. ⚠ **WORLDLINE (rev 2) — PLACEMENT FIXED.** The destruction **fact** is game-owned *and* must live in the non-sheddable **STATE** record, not in the sheddable fact journal (RULE WL-CONSERVED-FACT, `dormant_world_simulation_design.md` §2 rows 21/22). Otherwise "insurance retro-payouts are replayable from a log the GAME owns" is false: the physical journal may shed and the archive is optional, so claims are simply gone after an outage. The `Sink` it posts to is a **`PhysicalLossChannel`**, not a `MonetarySink` (see S7). |

**Total early cost: ~1,100 lines of pure Tier-A code plus design text (revision 1 said ~700, before S2's
real cost, S5b, and S11–S13) — and it removes every expensive-to-reverse decision from the critical path.**
⚠ Note that **S5b is not a "cheap seam"**: it is a real RLM-touching slice and should be planned as one.

### 10.3 Do NOT build yet

| Not yet | Why |
|---|---|
| The order book / matching engine | No venue exists until stations are first-class realms (P8-adjacent, D-39.4) and no items exist until P6. Building a matcher now is speculative scaffolding — the smallest-correct discipline forbids it. |
| NPC agents of any kind | They need commodities (P6) and a price field. And the *aggregate* layer must ship first and stand alone (§5.3). |
| Multi-currency FX, corporations, equity, dividends | Every one of these is additive on the ledger + saga once the ledger exists. Building them now freezes gameplay decisions with no consumer. |
| Any solver, LP, or equilibrium code | Tier-B advisory by construction; nothing consumes advice yet. |
| The analytics stack beyond the MINIMAL option | The MINIMAL option costs zero new deps and is enough to prove the log is right. Adding ClickHouse/Grafana before there are events to look at is operational weight with no payoff. |
| Cyclic recipes | Explicitly deferred by S3, not silently allowed. |
| The counterfactual twin | Needs the event log + G-ECON-REPLAY first; the twin is *cheap* precisely because it comes last. |
| A galaxy-wide market of any kind | It is a request to centralise load; price it as such (§4.9). |
| Player-authored RECIPES | §7.12's hard rule: a player build changes *rate and efficiency*, never *what transforms into what*. Allowing player-authored recipes means running the S3 acyclicity validator on untrusted runtime input and re-proving stage-invariance per graph — a materially different design, not an increment. |
| Cross-shard market QUERIES | Not available under HR1 and not to be designed for (§7.5). Search is a non-authoritative read model (§7.2). |
| Derivatives of any kind | Explicitly out of scope (D23): they multiply the solvency surface and add nothing the requested feature set needs. Stated as a decision rather than by omission. |
| Wall-clock ROLLBACK | Forbidden by §6.6, not deferred. Reverse forward instead. |

⚠ **Two things revision 1 put on this list that revision 2 moves OFF it**, because they are rules rather
than features and cost ~0 lines now: the **inventory-position representation** (S11 — it must constrain
P6/P7's first line, not follow the ledger) and the **destruction sink emit** (S13 — pre-P11).

### 10.4 Proposed DEFERRED.md ledger entries

Highest existing ids are **D-46** plus the `D-RLM-*` family [R], so the economy block starts at **D-47**.
Format follows the file's convention: WHAT is missing / WHERE the interim lives / WHEN it lands /
DEPENDENCY / PIN.

| Id | WHAT is missing | WHERE | WHEN | Dependency | Pin (exists-to-be-flipped) |
|---|---|---|---|---|---|
| **D-47** 🟥 | **The economy design document itself.** `docs/design/` is entirely silent on economy — a grep for market/economy/currency/wallet in `PLAN.md`, `DEFERRED.md` and `realm_lifecycle_design.md` returns nothing relevant [V]. No reserved `InterShardFlow` arm, no `ShardProfile` capability, no roadmap slot. | `docs/design/economy.md` (absent) | Design lands **before P6**; implementation P9-adjacent | This report + the §11 decisions | This ledger entry; `PLAN.md` gains an economy row |
| **D-48** 🟥 | **The double-entry ledger kernel + the closed faucet/sink taxonomy.** No `Money` type, no accounts, no entries, no conservation invariant. | `vd-core` (Money) + `vd-econ` (kernel) — both absent | S1/S7 early; the kernel with the first value-moving feature | S1, S7, **§4.11's action taxonomy** (S7's enum is a function of it), **the DURABLE `(TransferId, step_id)` `applied_steps` table (D-21/D-22 residual, 🟥) — money is the forcing consumer, and N6's receipt schema is part of the same table**, **`ReHomeState::Snapshot` (P7) + `TransferableKind::serialize` (D-31)** for any subject that re-homes | `verify_value_conservation` exists and passes trivially (zero accounts) so it cannot be forgotten |
| **D-49** 🟥 | **Reliable per-session gameplay-state carrier for wallet/market state.** `MsgClass` has no reliable per-session class and is WIRE-FROZEN append-only [R]. | `crates/sim/src/io/mod.rs` `MsgClass` | With D-4 (`EventMsg`) — the economy is its **second** consumer | D-4 | D-4's entry gains "economy wallet/market state" as a named consumer |
| **D-50** 🟥 | **N-party (multi-recipient) atomic distribution.** The saga is 2-party; dividends/splits/mergers/auction settlement are N-party. | `vd-sim` saga | With the first dividend feature | The batched go-token pattern (`generic_transfer.md §A6`) [R] | A doc-comment on the batch machinery naming dividends as the owed consumer, and an explicit "do NOT implement as a multi-key directory CAS (D-32)" note |
| **D-51** 🟥 | **Corporation identity, roles and division permissions as fenced authority state.** Authority-gating discrete data with **no home** in the current design; `identity_persistence.md` has no organisational-identity concept [R]. | `docs/design/identity_persistence.md` + directory | With corporations (P9+) | D-47 | Named in D-47 and in the identity design's KNOWN LIMIT |
| **D-52** 🟥 | **The economy event log + analytics projection**, **including the new `EventSink` `sim::io` seam** (revision 1's sidecar-tails-redb option is impossible — redb `flock`s exclusively [V]) **and the §8.1.1 backpressure/ack contract**. No cross-shard OLAP capability exists; redb is per-shard, and a torn-down realm cannot answer queries ⇒ the log must be **push/projection-fed**. | `vd-econ-analytics` (absent) + `EventSink` in `vd-sim`/io-prod | MINIMAL option with the ledger; STANDARD when there is traffic | D-48, S4 | The `EconSnapshot` admin endpoint exists and reports zeros; **G7b (backpressure fail-loud) exists and passes trivially with an empty ring** |
| **D-53** 🟥 | **RLM teardown safety for economic state** (book rehydration byte-identity **sized at `max_orders_per_book`**, no stranded escrow, `expiry_universe_tick` honoured across a down period). | `vd-econ` + RLM | With the first market | S9, D-48, RLM Step 4b (`StoreKey::Rlm` durable snapshot, D-RLM-2), **`ReHomeState::Snapshot` (P7)**, and ⚠ **A STORAGE-TOPOLOGY PREREQUISITE: pick (i) RWX/networked storage with RealmId-keyed volume identity, (ii) an explicit econ-prefix handoff step in the RLM spin-down/up saga (40–80 MB ⇒ ~0.3–0.6 s at 1 Gbit/s, fence-stamped + idempotent), or (iii) economic state only on node-pinned realms.** Today: `ReadWriteOnce` + `local-path` + **256 Mi** bound to a StatefulSet **ordinal** [R] ⇒ **G4 as written cannot pass** | The G4 chaos cell exists and passes trivially with zero orders; **shard PVC size becomes `f(max_orders_per_book, journal_retention_ticks)` in the manifests** |
| **D-54** 🟥 **raised to BLOCKING on E-1** | **Ledger retention / snapshot+truncate / partition-by-realm / cold archive.** At **0.95–1.9×10¹⁰ postings/yr (0.53–2.4 TB/yr** — revision 1's 41–47 GB/yr was 10–50× low *and* carried a 2× arithmetic error) the full history cannot live in one shard's redb, and **`rehydrate` is O(ledger) without a snapshot**, which makes RLM `boot_ticks` unbounded (§6.2). | `vd-econ` + io-prod store | ⚠ **With E-1**, not "before real volume" — the snapshot is a correctness/latency prerequisite, not a capacity optimisation | D-6's owed WAL version + Tombstone + compaction/retention (currently deferred to P6/P7) [R]; **D17's audit-granularity decision sets the volume** | D-6's entry gains "economy ledger volume" as a forcing consumer; **G7c (rehydrate inside `boot_ticks_p99`) exists** |
| **D-55** 🟥 | **Cyclic recipes** (fuel to make fuel) — deliberately excluded by the S3 acyclicity gate. | recipe registry | P6+ if ever wanted, with a decided sparse-solve strategy | S3 | The acyclicity validator's error type names this entry |
| **D-56** 🟥 | **The verified-matcher differential oracle (G-ECON-ORACLE).** Blocked on license clarity (the Coq repos have **no license** [V]), an OCaml/Haskell CI toolchain, and tie-break equivalence. | `vd-tests` gate (absent) | Opportunistic; the fallback (proptests against the paper's theorems) is not blocked | Order book existing | Named in D-47 as the strongest available correctness lever |
| **D-57** 🟥 | **Behavioural counterfactuals.** The replay twin answers mechanical questions only; elastic-demand agents are needed for "players stop trading when you raise the fee", and any output must be labelled a **model, not a prediction**. | `vd-econ-solver` / an offline harness | After the twin's G-ECON-REPLAY gate | D-52, agents, **a measured ticks/s from E-(−1)** (the twin budget was 29× off without one) | Named in the tweak-surface design |
| **D-58** 🟥 ⚠ *new* | **Direct player-to-player trade** (§7.8) — the most common economic interaction, undesigned in revision 1, and the classic dupe vector. | `vd-econ` + `vd-sim` | **Slice E-2b — the FIRST player-visible economy feature** | D-48, D-49 (the receipt carrier) | The G9 chaos cell (kill either side between confirm and post) exists |
| **D-59** 🟥 ⚠ *new* | **Contracts, collateral and courier adjudication** (§7.9) — FSM, forfeiture, adjudication at the collateral holder, discovery, retention, transport risk. Under sealed shards these are *more* load-bearing than order books. | `vd-econ` + `vd-sim` | **Slice E-4b, BEFORE order books** | D-48, I19 (the collateral holder must be live), the read model for discovery | I15 CONTRACT-CLOSURE + the G10 cell (deadline elapses while the issuer's realm is down) |
| **D-60** 🟥 ⚠ *new* | **The material economy** (§4.10): extraction yield, terrain regeneration policy (**D16**), the loot-drop ratio, wreck persistence, salvage yield, and the two **issuance targets**. P4/P5/P6/P11 fix this geometry irreversibly. | `vd-core` generator (S12) + `vd-sim` destruction path (S13) + `EconomyTuning` | **S12 pre-P4, S13 pre-P11**, the targets with the ledger | D16, D21, the P11 full-loot decision | S12's closed-form ore field exists; S13's `Sink::Destruction` emit exists and sums to zero at zero destruction |
| **D-61** 🟥 ⚠ *new* | **Live-invariant behaviour + the reversal primitive** (§6.6): per-invariant blast radius and automatic action, the fenced `HALTED` flag, `EconIncident`, cohort reversal, and the **rule that wall-clock rollback is never permitted**. | `vd-econ` + `vd-harness` + the ops surface | With E-1 (the invariants are useless without it) | D-48, D-50 (cohort reversal is N-party), **D-13 admin authn for the halt surface** | Every invariant's failure path has a test asserting *refuse-and-continue*, never panic |
| **D-62** 🟥 ⚠ *new* | **Territory / realm ownership / alliances** (§7.11a) — presupposed by D11(b)'s player-settable rates, by venue capability, and by self-match prevention's "character → corp → alliance"; designed nowhere. | directory `RealmOwner` record + `vd-econ` | With player-settable rates (D11b) | D11, D18 (acquisition mechanism), D19 (alliance shape), P11 for conquest | Named in D-47; D11's entry gains "requires RealmOwner" |
| **D-63** 🟥 ⚠ *new* | **The corporation ORGANISATION layer** (§7.11b) — permission bitmask, membership lifecycle + member-held assets, corp-action audit log, **role-change front-run delay**, corp taxation of member income, shared hangars, the theft stance (**D20**). D-51 covers identity/roles as *authority state*; this is the model. | `identity_persistence.md` + directory + `vd-econ` | With corporations (P9+) | D-51, D-50, D20 | Named in D-51 |
| **D-64** 🟥 ⚠ *new* | **HR6 economy surface** (§8.8): `vdctl` verbs, `DevState` econ counters, `WaitField` arms, the `runs/` manifest, and a per-slice "works in-game" criterion. **HR6 is a hard rule; no slice lands without it.** | `vd-devproto` + `vd-bins` + `vd-client-harness` | **With E-1**, incrementally per slice | the counters follow each feature | `vdctl econ-assert-conservation` exists and exits 0 on an empty economy |
| **D-65** 🟥 ⚠ *new* | **`EconomyTuning` home + epoch reconciliation** (§8.9): the struct in `vd-core`, boot-time parse in the bin, structural + economic validation, and the **boot-refusal on config/epoch `tuning_hash` mismatch** (the unspecified divergent state in revision 1). | `vd-core` + the bins + directory epoch record | With E-0/E-1 | S4, the directory epoch CAS precedent [R] | `econ_config_mismatch` is a FAULT counter that reads 0 |
| **D-66** 🟥 ⚠ *new* | **Insurance as a funded pool** (§7.14), **credit/bonds scope** (D23), **equity lifecycle** (§7.11c), **M(0) + onboarding** (§7.15), **escheatment** (§7.10), and **regional-divergence enforcement** (§7.13) — the policy layer, each independently shippable on the ledger. | `vd-econ` + `EconomyTuning` | P9+ | D-48, D-50, D22 (seasons changes escheatment's weight) | The divergence gate (`min_price_dispersion_bp`) exists and passes trivially at one realm |

### 10.5 A slice sketch (for a later arc, not a commitment)

⚠ **Re-sliced in revision 2**: an in-engine measurement spike moved to the FRONT (it replaces three
estimated budgets with one measurement and the review showed those estimates were wrong by 12–50×),
direct trade inserted as the first player-visible feature, contracts moved **before** order books, and an
**HR6 "works in-game"** column added because a slice without one cannot pass the project's own discipline.

| Slice | Content | Gate | HR6 "works in-game" |
|---|---|---|---|
| **E-(−1)** ⚠ *new, do this FIRST* | **A ~200-line in-engine spike**: a stub `vd-econ` posting loop + a stub agent round inside a real `step_tick`, measuring (a) durable postings/s against our own `RedbStore` (not redb's published benchmark), (b) ns/agent-evaluation, (c) ticks/s for the twin, (d) rehydrate ms per 10⁵ ledger rows | The three measured numbers land in `EconomyTuning` as budgets; **no §2.2 figure enters a design doc until this runs** | `vdctl state` shows the stub counters advancing |
| **E-0 (pre-P6, cheap)** | S1–S13: the money type, **the item-stack TLV schema**, the acyclicity validator, `econ_epoch` + the **cadence fields** (in `vd-core`), `ShardProfile` **S5a**, the inert `Account` directory arm, the closed faucet/sink enum **derived from §4.11's action taxonomy**, `verify_value_conservation` + G8, the RLM rules written down, the Signal-design input, **S11's inventory-position rule**, **S12's closed-form ore field**, **S13's destruction sink emit** | `just gate` green; G8 detects an injected 1-unit imbalance **and refuses an over-budget declared mint (I16)**; coverage 100% on the new Tier-A code | `vdctl econ-assert-conservation` exits 0 on an empty economy |
| **E-0b** ⚠ *new* | **S5b**: the per-realm venue-capability override on the RLM spawn path (a real RLM slice) | An RLM cell: two sibling stations, one venue-capable, one not, across a spin-down/up cycle | `vdctl state` reports the venue flag per realm |
| **E-1** | The ledger kernel: accounts, entries, pending/post/void with universe-tick timeout, per-currency conservation, **snapshot+truncate (§6.2)**, the **durable `applied_steps` + receipt table (N5/N6)**, **`EconomyTuning` + epoch reconciliation (D-65)**, and **§6.6's live-invariant behaviour** | G-ACCOUNTING over 12 k ticks; G1 model-based; G2 bank test under `FaultFabric`; **G5 G-IDENTICAL from day one** (moved up from E-4 — it is D2's only HR3/HR4 safeguard); G7c rehydrate budget | `vdctl econ-balance` on a real client returns the server-computed `{posted, pending, available}` |
| **E-2** | Value crossing shards as a `Funds` `TransferableKind` on the existing saga; the clearing-account rule | G3 kill-9 cells; I3 holds mid-saga | a real client's wallet survives a re-home over live QUIC |
| **E-2b** ⚠ *new* | **Direct player-to-player trade** (§7.8): two-sided reservation, confirm-lock, linked post, expiry | **G9**: kill either side between confirm and post; I8 + I1 + I3 | **the canonical acceptance test — two real clients over live QUIC complete a trade and both wallets settle**, asserted by `vdctl` on both sides |
| **E-3** | Items as ledger positions; split/merge conservation; `ItemId` minting; **the per-entity blob budget (§7.6)** | G6 dupe chaos; **G6b transient-loss conservation (I20)**; I10; a proptest that a max-budget entity serialises under the 1 MiB frame cap | `vdctl` inventory dump reconciles with the ledger |
| **E-4** | Tax composition (T1–T8, with T8 as **push-with-retention**) + the five algebraic proptests; per-realm depletion index | I14; **G7e tax-path microbench** | a real trade's tax appears in the beneficiary treasury, asserted after a beneficiary realm restart |
| **E-4b** ⚠ *new, BEFORE order books* | **Contracts + collateral + courier adjudication** (§7.9) | **I15**; **G10**: deadline elapses while the issuer's realm is torn down | a real client posts a courier contract, another accepts and delivers, over live QUIC |
| **E-5** | The event log via the **`EventSink` seam (option B) or in-process export (A′)** — **never a second redb opener** — + the MINIMAL analytics path + `EconSnapshot` + the **§8.1.1 backpressure contract** | The accounting identity reproduced offline from the log alone; **G7b backpressure fail-loud**; **I17 self-report cross-check** | `vdctl econ-dump-log` output reproduces the identity offline |
| **E-6** | Ambient markets: the clamped closed-form price + the lazy `f(seed, universe_tick)` field + collapse/reconstitute conservation + **the §7.13 divergence forces** | Value-conservation across a teardown/spin-up cycle; no visible price jump; **the `min_price_dispersion_bp` gate** | a client sees different prices at two realms and hauls between them |
| **E-7** | The order book + clearing at venue-capable realms (**with the dirty set and the full tie-break total order, §4.1**); escrow per **D15**; expiry; caps incl. **`max_orders_per_book`**; tick/fees; **the paged market read carrier + its per-session caps** | G4 RLM cell (**sized at `max_orders_per_book`**, and gated on D-53's storage prerequisite); G7 load (sharing spike3a's release build); G-ECON-REPLAY | a real client places, sees, and fills an order in a rendered market UI |
| **E-8** | Agents (D8 (a)→(b)→(d)) with **`agent_eval_budget_per_tick`** | **G7d agent tick cost**, seeded by E-(−1) | a client watches an NPC-driven price move while idle |
| **E-9+** | Multi-currency + FX policy (§7.16) · corporations/equity + the organisation layer (§7.11) · insurance (§7.14) · M(0)/onboarding (§7.15) · escheatment (§7.10) · the STANDARD analytics stack · the twin · the incident runbook tooling (§9.4) | each with its own gates | each with its own vdctl verb |

---

## 11. Decisions for the user

**Twenty-four** numbered choices (revision 1 had fourteen; the review surfaced ten more, three of which —
D15, D16, D17 — gate other work). **No library is presented as adopted** — every dependency is a decision
here, per the standing rule. Recommendations are stated but none is decided.

---

**D1. Market mechanism.**

| Option | Pros | Cons |
|---|---|---|
| **(a) Frequent batch / uniform-price auction** | Clearing is a pure function of an unordered order **SET** ⇒ determinism free, no cross-shard total order over arrivals needed, replicable and cross-checkable against the Coq uniqueness theorem. Latency-fair (we have no client prediction + a 100–150 ms buffer). One clearing pass per N ticks instead of per message ⇒ what makes 10⁵–10⁶ markets affordable. One clearing price per market per batch collapses analytics volume by orders of magnitude. | Players from other games expect a live book with visible depth and instant fills. Unfamiliar. |
| **(b) Continuous double auction** | Familiar; immediate fills; visible depth. | Outcome depends on arrival order ⇒ exactly ONE authority, no replication, no oracle cross-check; rewards whoever's packet lands first, which reads as "lag = money" and is farmable by co-located bots; more analytics volume. |
| **(c) EVE-style event-driven with arrival precedence** | Cheapest of the three (no continuous loop); proven at scale. | Unusual pricing semantics; still arrival-order dependent. |
| **(d) Two-tier**: single ambient mid price per (good, realm) everywhere + real books only at venue-capable realms | Cheapest overall; immune to intra-realm self-arbitrage; no per-order state in most realms ⇒ nothing to reconcile on wake; books exist where they are interesting. | Two mechanisms to explain to players (though only one to *implement*, since ambient prices are a closed form, not a book). |

> **Recommendation: (d) with (a) as the book mechanism** — ambient closed-form prices everywhere, and a
> uniform-price batch auction at venue-capable realms. Both run on the same clearing function, so this
> is one implementation, not two. (b) remains buildable on the same `BTreeMap` structure if playtesting
> demands it.

---

**D2. Authority placement.** (Options and the full tradeoff table in §6.3.)

> ⚠ **Recommendation INVERTED in revision 2.** Revision 1 recommended "(D) inside (B)" — a never-dormant
> economy capability owning everything, keyed so A/C is a later re-keying. **Two independent constraints
> make that infeasible:**
>
> 1. **Agents (§2.3c).** 5,000 noise agents × 10³ active markets = 5×10⁶ evaluations/round =
>    **1.0–2.0 core-seconds**. On one single-threaded sequencer that is **>100% of a core** before any
>    player traffic; and the escape hatch (agents in realm shards, books on the economy shard) costs
>    **~25–500 MB/s** of side-effecting `InterShardFlow` egress — the same objection §8.1 uses to keep the
>    event log off the reviewed taxonomy. Revision 1 never carried the agent number into this table, so
>    the "throughput ceiling" row read as if matching headroom settled it.
> 2. **Lifecycle (§6.3).** There is **no way to spawn or protect** a non-spatial never-dormant node: the
>    reconciler is the SOLE kill authority over a desired live-set that is the *ancestor-closure of
>    realms*, and spawning is keyed on a `RealmCoord` [R]. B is therefore a synthetic realm or a second
>    lifecycle authority — neither acknowledged in revision 1.
>
> **Recommendation: implement (D) event-sourced single-sequencer internals inside (A)** — key
> `Account(AccountId)` and `Market(RealmId, CommodityId)` as directory key families **from day one**, with
> the `economy`/`venue` capability on the normal `ShardProfile` lattice (plus S5b's per-realm override),
> and G5 G-IDENTICAL in slice **E-1**. **(B) then falls out as a deployment CONFIGURATION of (A)** — one
> `ShardProfile` instance owning many market keys — which is reversible; **A→B is config, B→A is a
> rewrite.** `Market(RealmId, CommodityId)` additionally closes the hot-hub question for free (§7.5).
>
> **The one real argument left for B** is sequencing, not architecture: A and C both need
> `ReHomeState::Snapshot` (P7) and `TransferableKind::serialize` (D-31) before a market or account owner
> can survive a permanent kill (§6.3), and B avoids that because accounts never move. **If the user wants
> economy before P7, B is the honest interim** — but then it must specify its lifecycle, its command arms,
> its per-action cross-shard rate, and a corrected latency row, and it must accept the agent placement
> consequence (agents cannot live there).
>
> **Also decide: where do offline players' accounts live?** (§7.10 / I19.) Under A the recommended answer
> is a **stable never-dormant home shard per account** derived from the `AccountId` (the
> ancestor-closure-pinned Galaxy realms are the natural set). This is a *consequence* of D2, not a separate
> feature, and revision 1 left it undesigned.

> ⚠ **WORLDLINE (rev 2) — HALF THE MECHANISM IS REMOVED AND REPLACED.**
> `scripts/dormant_world_simulation_design.md` **rejects `Account(AccountId)` as a `DirectoryKey` arm** (W13, §8.4):
> the directory CAS is the **only commit point**, so an economy key there makes an economy fact an authority
> input — the deepest LAW-E1 violation available. `Market(RealmId, CommodityId)` is untouched, so D2's
> hot-hub answer survives.
>
> **The replacement, which also answers the "where do offline accounts live" question above without a
> never-dormant home shard**: an account's durable **POSITION** is a **zero-rate worldline subject** custodied by
> the worldline custodian (§6.7) — `TickRate { num: 0, den: 1 }`, `LossBudget::ZERO`, rehydrated on adopt through
> the existing `WorldlineAdopt`, inheriting the fence-ordered absolute-rebase log, the stale-fence reject and the
> conservation oracle. Single-writer-per-account becomes a property of that log rather than of a directory arm,
> and a reaped realm strands nothing. **Live** book state (resting orders, in-flight matches) stays `vd-econ`'s
> own and still needs D3/D-53's byte-identical rehydration — the split is RULE WL-LIEN's: **positions are
> physical and game-custodied; obligations are monetary and econ-owned.**

---

**D3. RLM teardown answer for books / wallets / escrow.** (Options A/B/C in §6.4.)

> ⚠ **Recommendation CONDITIONED in revision 2.** (A) durable-dormant remains the most consistent with
> sealed shards and reuses the `/whoami` cookie-probe + launch-ledger rehydrate pattern already built and
> kill-9-proven [R] — **but it is BLOCKED on a storage-topology prerequisite that the current deployment
> does not satisfy** (§6.4): shard PVCs are `ReadWriteOnce` + `local-path` + **256 Mi**, bound to a
> StatefulSet **ordinal** rather than to a `RealmId` [R], so **G4 ("bring it up on a DIFFERENT node,
> assert byte-identical rehydration") cannot pass.**
>
> **So D3 is really two decisions:**
> 1. **The storage prerequisite** — (i) RWX/networked storage with RealmId-keyed volume identity, (ii) an
>    explicit econ-prefix **handoff step** in the RLM spin-down/spin-up saga (40–80 MB for a hub ⇒ ~0.3–0.6 s
>    at 1 Gbit/s; fence-stamped and idempotent like every other handoff), or (iii) **economic state only on
>    node-pinned realms**. **Until one is chosen, D3 defaults to (B) or (C).**
> 2. **The dormancy shape** — (A) with the prerequisite met, else (B)/(C).
>
> **Plus the four unconditional rules** (§6.4): the **escrow LOCUS is D15, not pre-decided**; the book a
> `Store`-backed Durable TLV table with byte-identical rehydration **sized at `max_orders_per_book`**
> (400 k–1.5 M for a hub, ~0.3–1 s, not the 40 k revision 1 assumed); `expiry_universe_tick` mandatory on
> orders *and* on escrow-rights grants; and the **dormant-tier catch-up structurally bounded** by
> `dormant_catchup_tick_period` + `max_catchup_ticks` with a defined fallback.
>
> Also raise the shard PVC size to `f(max_orders_per_book, journal_retention_ticks)` in the manifests.

> ⚠ **WORLDLINE (rev 2) — CHANGED for positions; still open for live books.**
> `scripts/dormant_world_simulation_design.md` W2 finds **both** of this decision's storage shapes inadequate for
> the physical layer — the orchestrator is `requests == limits` **512 MiB GUARANTEED QoS** with a 1 Gi PVC shared
> with the directory and the saga WAL, and per-realm Store B is blocked exactly as this section says — and adds a
> **third option: a dedicated `worldline-custodian` StatefulSet** with its own PVC and memory budget, leaving the
> orchestrator as clock/directory/CAS authority only. **Wallets, treasuries and settled escrow positions ride
> that custodian as zero-rate subjects (§6.7), so they need NONE of this section's storage prerequisite.** What
> remains genuinely open here is **live order-book rehydration at `max_orders_per_book`** and
> `expiry_universe_tick` across a down period. Two further constraints it imposes: compaction must be
> **lazy-on-adopt** so no code path prefix-scans a store family (a family `scan` materialises the whole prefix,
> `crates/io-prod/src/store.rs:751-776`), and the **legal** way to keep a busy venue warm is a per-realm
> `min_dormant_ticks` **time** hysteresis charged to the game's lifecycle tuning — never a KeepAlive lever
> charged to an obligation (see the §6.4 note).

---

**D4. Money representation.**

| Option | Pros | Cons |
|---|---|---|
| **(a) `Money(i128, CurrencyId)` minor-units newtype (ours)** | Canonical bytes; exact; deterministic across targets; zero deps; ~21 orders of magnitude of headroom; one monomorphic rounding/overflow site; near-zero HR5 cost; tick-size lattice comes free | We write and cover ~300 lines |
| **(b) `rust_decimal` 1.42.1** | Mature, well-maintained, purpose-built, no float | **Preserves trailing zeros ⇒ non-canonical encoding**, which makes byte-identity gates representation-sensitive and embeds a third-party serde impl in the frozen wire contract; generic-heavy under per-monomorphisation region counting |
| **(c) `fixed` 1.31.0** | Genuinely maintained and stable (⚠ the research's "stale alpha" claim was wrong [V]); `no_std`-shaped deps | **Base-2 fixed point cannot represent 0.01** ⇒ wrong for money. Acceptable for *rates* if confined to a few monomorphic newtypes — and needed at all only if integer bp/ppm in `i128` proves insufficient |
| **(d) floats** | — | Forbidden by our own determinism rules |

> **Recommendation: (a) for the authoritative ledger; (b) permitted in Tier-B display/reporting only;
> (c) held in reserve for rates.** Also decide the **rounding policy** (recommend: always toward the
> sink/house, one rounding point per event, dust to a declared sink account) — this is what stops
> intermittent `Σdebits ≠ Σcredits` failures under load.

---

**D5. Ledger implementation.**

| Option | Pros | Cons |
|---|---|---|
| **(a) Our own kernel over redb + the saga WAL** | Zero new deps; pure Tier-A; ~1,000–1,500 lines; fits the existing WAL/CAS/outbox; no new operational surface; the directory CAS stays the only commit point | We own correctness (mitigated: G1/G2/G8 are strong gates) |
| **(b) External TigerBeetle cluster via `tigerbeetle-client`** | Jepsen-verified strong serializability from 0.16.26 [V] | A 3–6 replica Zig cluster to operate; **Linux-only production support**; tokio-async FFI; a **second commit authority**; only 2 published crate versions and 41 total downloads [V]; a ≥10 s single-node-failure latency tail pre-0.16.43 [V]; and its query surface is minimal so dashboards still need a separate projection. ⚠ The commonly cited TPS figures are **unsourced [V]** |

> **Recommendation: (a), copying TigerBeetle's SCHEMA and two-phase semantics exactly.** That is the
> highest-value reuse in the report and it costs nothing.

---

**D6. Order-book code provenance.**

| Option | Pros | Cons |
|---|---|---|
| **(a) Port the design** (limitbook's structures + matchcore's command/outcome state machine + match-rust's integer tick/lot and golden-replay discipline), writing ~600–1,200 SLoC ourselves | 100% region+branch achievable; determinism ours; no third-party generics in Tier-A; matches our conventions exactly | We write it |
| **(b) Depend on `matchcore`** | Fastest to a working book; deps are Tier-A-compatible | 15 k SLoC we cannot cover; **2 GitHub stars, 1 fork, 158 downloads, no coverage or soundness claims** [V]; a matching bug is directly exploitable for currency creation (a security issue, not just correctness); and the repo/crates.io license metadata disagree [V] |
| **(c) Depend on any active crate** | — | Every active one is lock-free/async (§3.3) |

> **Recommendation: (a).** "Port" means we own the coverage, so it is only affordable at the ~600–1,200
> SLoC scale — not a 15 k-SLoC transliteration. Also decide the price-level structure:
> **BTreeMap + Slab (recommended: empty-cheap, essential at 10⁶ books)** vs a flat tick ladder as an
> opt-in fast path for a hot hub market only.

---

**D7. Analytics stack.** (MINIMAL / STANDARD / HEAVY, fully costed in §8.4.)

> **Recommendation: start MINIMAL (zero new server-side deps), move to STANDARD when there is traffic
> worth looking at.** The specific dependency decisions inside that: `arrow` + `parquet` (Apache-2.0) for
> the archive; **one** of `datafusion` 54.1.0 / DuckDB / `polars` for querying; and for STANDARD, the
> `clickhouse` 0.15.1 Rust client + ClickHouse + Grafana 13.0.0 (**AGPLv3** — fine as a separate process,
> a licensing event if forked/embedded) + Prometheus or VictoriaMetrics. A serious Rust-native
> alternative for the whole STANDARD pair is **GreptimeDB** (one engine for metrics and events, Apache-2.0
> core, less battle-tested) [L]. **All of these must live in a crate that is not a member of the server
> build graph** — `datafusion` alone has 47 direct deps including tokio and object_store [V].

---

**D8. Agent scope and placement.**

| Option | Pros | Cons |
|---|---|---|
| **(a) None initially; closed-form ambient prices only** | Cheapest; ships first; independently valuable | Markets have no counterparty behaviour |
| **(b) ZI-C + PRZI + trivial baselines, in-tick, AoI-scaled and phase-decimated** | ~500 lines, pure, deterministic, gives every active market liquidity and price discovery | Per-tick cost must be budgeted (5,000 noise × 10³ markets = 5×10⁶ evaluations/round undecimated) |
| **(c) + ZIP minority, + GD gated on volume** | Prices converge and respond to shocks; smarter behaviour where the book is deep | Branch-dense (coverage cost); GD is O(H) |
| **(d) + Doran–Parberry production agents** | Closes the loop to production and to the **distribution of professions** — bankruptcy-and-replacement IS the profession allocator | The most parameters; chaotic sensitivity to initial conditions ⇒ replay pins seed AND agent order |
| **(e) A separate background simulator** (Star Citizen's choice) | Immune to realm churn | A new process/capability; the highest-risk, latest-delivering component in the genre [V] |

> **Recommendation: (a) → (b) → (d), in that order, each independently shippable.** Reject GDX outright.
> Port AA only if a specific NPC archetype must visibly outcompete players. Agents are `TraderStrategy`
> enum-registry arms (recommended, HR2-style), not trait objects (rejected: generic-monomorphisation
> coverage cost).
>
> ⚠ **Two things revision 1 left implicit that the review made load-bearing:**
> - **`agent_eval_budget_per_tick` is a named `EconomyTuning` field, and the phase-decimation factor is
>   DERIVED from it** — not asserted. At 200–400 ns/evaluation the budget converts directly into an
>   agents-per-round figure per shard (§2.3).
> - **Agent placement is not free: it CONSTRAINS D2.** Books must be co-located with the agents that quote
>   into them, which is what forces D2 option A (§2.3c). Deciding "agents yes" after deciding "one economy
>   shard" is deciding the same thing twice, incompatibly.

> ⚠ **WORLDLINE (rev 2) — CHANGES THE MECHANISM AND THE PLACEMENT.** Two things:
> 1. **NPC EXISTENCE IS NOT AN AGENT DECISION.** NPC existence, population, cohorts, pose, routines and physical
>    needs are **main-game** subsystems in `vd-core`/`vd-sim` (LAW-E3), advancing identically with the economy
>    off. Only **trading POLICY** is economy-adjacent. Placing "NPC agent strategies" inside `vd-econ` (§7.7,
>    §7.1) is a latent LAW-E1 violation — switching the economy off would delete the NPCs.
> 2. **STRATEGIES ARE AN INJECTED OBJECT-SAFE PORT, NOT A `TraderStrategy` ENUM REGISTRY.** This row rejects
>    trait objects on HR5 monomorphisation grounds; the worldline design **overrules that**, because LAW-E1
>    requires the economy's decision *logic* to be link-time absent-able and an enum registry would put it in
>    Tier-A. The HR5 cost is paid by making the trait **object-safe** (the `Store`/`RealmSpawner` idiom, so there
>    is no per-monomorphisation region multiplication), with a named **price-free `NeedsOnlyStrategy` default**
>    that is the declared degraded mode. The influence direction is also constrained: a strategy may change the
>    world **only** by issuing a journaled, refusable `EconCommand` — never by biasing a value that a
>    deviation-authoring code path reads (`dormant_world_simulation_design.md` §3.2, §2 row 4).
>
> The coarse **aggregate** tier of §7.7 is likewise re-homed: it is a **game** substrate (the worldline's
> closed form), not an economy component, and its optional agent layer (W-5 / D-71) sits behind an injected port
> wired only in `vd-bins`.

---

**D9. Multi-currency scope.**

| Option | Pros | Cons |
|---|---|---|
| **(a) One currency** | Simplest; no FX; no partition key | Fails the stated requirement |
| **(b) A fixed small set with one authoritative mint each, plus the SCOPE taxonomy** (`fungible-global` \| `issuer-scoped-non-transferable` \| `commodity-ised` \| `account-scoped`) | Covers everything EVE does with FOUR currencies at a fraction of the machinery: issuer-scoped needs **no** transfer machinery, commodity-ised needs only item transfer, and only ISK-like needs a cross-shard `Funds` saga. Makes "add a currency" a registry entry, not a feature | Requires the taxonomy to be right up front |
| **(c) Player/corporation-issued scrip** | Rich emergent politics; Eco's shipped precedent | Needs an on-ledger issuance registry, a separate non-reflexive redemption escrow, mandatory reserve disclosure, and no protocol acceptance guarantee — plus an unbounded number of `ledger` partitions |

> **Recommendation: (b), with a numeraire star topology and N−1 books, designing the SCOPE taxonomy as a
> registry before the ledger schema freezes.** (c) as a later opt-in. Also decide: **is there a numeraire,
> and is it NPC-issued with a faucet/sink budget (EVE ISK), externally pegged (Entropia), or itself a
> player instrument?** Note that FX must be ≥4 entries across per-currency pools from day one — a 2-entry
> conversion is historically unverifiable and retrofitting it means rewriting every historical
> transaction.

---

**D10. Jurisdiction shape.**

| Option | Pros | Cons |
|---|---|---|
| **(a) Strict hierarchy** (a realm's applicable laws = the path to the root) | Cheap, deterministic, unambiguous incidence; reuses `path_index`/LCA verbatim; depth-invariant under T2 | Less political texture |
| **(b) Eco-style overlapping same-tier influence radii** | Richer politics; competing governments | Ambiguous membership; ambiguous tax incidence for a single transaction; expensive |

> **Recommendation: (a).** The politics can come from *rate competition* between realms (T5's floor
> prevents the race to zero) rather than from overlapping jurisdiction.

---

**D11. Who sets tax rates, and how a change lands.**

| Option | Pros | Cons |
|---|---|---|
| **(a) Operator-only config** | Simple; safe; the ceiling/floor machinery is optional at first | No player agency; misses the best gameplay |
| **(b) Player-settable by whoever holds the realm** (EVE's POCO/structure model, which is the shipped precedent for additive composition [V]) | Real regional economics; the strongest emergent-politics lever | The T4/T5 ceiling/floor/top-up machinery becomes load-bearing immediately, and a rate change must be a fenced, journaled, ledgered event with an `effective_tick` **delay** — otherwise the owner front-runs their own tax change against inbound trades |

> **Recommendation: (b) with T4+T5 landing at the same time, plus the effective-tick delay.** Also decide
> whether taxes are **evaluated inline at the transaction** (deterministic, but every trade touches the
> jurisdiction chain — cheap, since `path_index` is O(depth·log L)) or **accrued and swept periodically**
> (cheaper per trade, but needs an off-tick idempotent fence-stamped scheduler that survives kill-9).
> Recommend **inline assessment + accrued remittance** (T8), which gets both properties.

---

**D12. Equity / corporations scope, and the fraud stance.**

| Option | Pros | Cons |
|---|---|---|
| **(a) Corporations with shared wallets only** (no tradable equity) | All the co-op gameplay, none of the securities-shaped risk | No "stocks of companies" |
| **(b) + fixed-supply tradable shares with mechanically-funded dividends** (Entropia's safe shape) | Meets the requirement; trades on the same clearing engine (HR3) | Needs the vote/record-tick, M-of-N officer, disclosure and solvency machinery — and the N-party distribution (D-50) |
| **(c) + player-run banks / deposits / promised interest** | Legendary emergent stories (EVE's best) | **Every documented instance collapsed** [V]/[L]; systemic and reputational risk |

> **Recommendation: (b), and explicitly decide the FRAUD STANCE now** because it changes the data model:
> is fraud **intended gameplay** (EVE's answer — then build disclosure and let players be fooled about
> the *future*, never the *present balance*) or **prevented** (then build M-of-N, mandatory reserve
> disclosure, and snapshotted vote weights as hard rules)? Recommend: **fraud about the future is
> gameplay; fraud about the present balance is a missing feature.** Also: **no real-money redemption
> path** (§9.3).

---

**D13. Solver adoption (all Tier-B, offline, advisory).**

| Option | Note |
|---|---|
| **(a) None initially** | Nothing consumes advice yet; proportional response can be written in ~150 lines when it is needed |
| **(b) `microlp` 0.5.0** (Apache-2.0, pure Rust) | The most defensible; **but** it depends on `web-time` and advertises wall-clock time limits [V], pulls `sprs` transitively, and has an optional `highs` feature. Use with time limits DISABLED, a pinned version, and an assertion that sprs's **LGPL Cholesky feature stays off** |
| **(c) `clarabel` 0.11.1** (Apache-2.0) | The only backend exposing **duals** (shadow prices = exactly the dashboard's "analysis" output); optional BLAS backends mean per-CPU divergence |
| **(d) `good_lp` 1.15.2 as a facade** | ⚠ **MANDATORY `default-features = false`**: the default is `["coin_cbc", "singlethread-cbc"]`, which needs C library headers on the build machine [V]. ⚠ And the research's license claim was wrong: coin_cbc is **MIT**, CBC is **EPL**; the LGPL backend is **lpsolve**; SCIP via `russcip` is **Apache-2.0** [V] |
| **(e) `highs`** | **Reject**: multi-threaded float reduction ⇒ two runs on the *same* machine can differ, plus a C++ toolchain in the build |
| **(f) `argmin` 0.11.0** | The right tool for offline **calibration** specifically; pure Rust |

> **Recommendation: (a) now; if/when needed, (b) or (c) directly rather than through the (d) facade** — a
> single pinned backend is easier to keep deterministic-enough and avoids the default-feature trap
> entirely. **Never in the authoritative tick under any option**; output is always quantised to an integer
> price grid before it re-enters the sim.

---

**D14. Test-tooling dependencies.**

| Option | Note |
|---|---|
| **`proptest-state-machine` 0.8.0** (MIT/Apache-2.0, 2026-03-24) | Model-based testing against a reference model (`BTreeMap<AccountId,(posted,pending)>` for the ledger; a brute-force matcher for the book). **Dev-dependency only ⇒ zero HR5 cost**, and we already depend on `proptest` with a checked-in `proptest-regressions/` corpus [R], so it is an in-family addition rather than dep drift. Still needs explicit sign-off |
| **`stateright` 0.31.0** (MIT, 2025-07-27) | An explicit-state model checker — attractive for *proving* the value-transfer FSM over all small interleavings. Cost: the FSM must be re-expressed in its model shape, risking model/implementation divergence (the classic failure of bolt-on model checking) |
| **The Coq-extracted oracle** (D-56) | Blocked on license clarity (**no license** [V]), an OCaml/Haskell CI toolchain, and tie-break equivalence |

> **Recommendation: adopt `proptest-state-machine` as the standing gate; use `stateright` as an optional
> one-off verification on the 3–4 hardest cells only; treat the Coq oracle as opportunistic with the
> proptest-against-published-theorems fallback.** ⚠ **Every proptest gate carries an explicit
> `PROPTEST_CASES`** and a committed regressions corpus (§6.5) — the default 256 cases over a 12 k-tick
> ledger scenario is minutes-to-hours in a repo whose gate is a human-run pre-merge step with no CI.

---

**D15. Where escrow lives** *(new in revision 2 — revision 1 asserted both halves of a contradiction).*

| Option | Pros | Cons |
|---|---|---|
| **(a) Co-located with the BOOK** | **A fill is a local mutation, not a saga** — the single biggest simplification in the report, and the reason §7.2's throughput model works as written | A killed market realm must be *proven* never to strand locks (a real gate, not a hope); escrow travels with the book under the D-53 handoff; a remote actor's value sits on a shard they do not occupy |
| **(b) Held by the ACCOUNT owner** | A killed market realm can never strand value; a remote actor's assets never leave their own authority; unambiguously right for direct P2P trade regardless | **Every fill becomes a cross-shard escrow post** ⇒ re-derive the throughput budget with one saga (or one batched go-token) per fill; a different per-fill idempotency requirement; a different E-7 slice |
| **(c) Hybrid** | book-local for same-realm actors, account-owner for remote | Two fill paths ⇒ two crash matrices; must be gated as two shapes, not one |

> **Recommendation: decide before E-1 freezes the escrow schema, and state it once.** At the hub's measured
> 2.3–3.0 trades/s mean (§2.3), **(b) is affordable** and it is the strictly safer answer under RLM — so
> (b) is the recommendation *if* the throughput budget is re-derived honestly. **(a) is defensible only
> with the no-stranded-locks gate written first.** Do not carry both, which is what revision 1 did.

> ⚠ **WORLDLINE (rev 2) — THE PHYSICAL HALF IS DECIDED AND OUT OF SCOPE HERE.** **RULE WL-LIEN**: an economy
> object may **never** be the container of record for an item position, so escrow over **goods** is a *lien* — a
> claim in `vd-econ` against a stack whose position stays in a **GAME** container (typically the venue station's
> own physical container). Economy absent ⇒ the lien is unenforceable, never "the goods are nowhere"; and an
> econ-owned container would be invisible to the harness `InspectReport` capture the conservation oracles audit,
> so it would read as a phantom (positions < mints) or, on restart against a stale escrow view, a duplicate.
> Gate cell: open an escrow over goods, kill the economy, assert the identity holds and the goods are still in a
> game container. **D15 therefore reduces to the MONETARY half only** (`dormant_world_simulation_design.md`
> §2 row 28).

---

**D16. Does mined terrain regenerate?** *(new — and it is a P4 decision, not a P9 one.)*

| Option | Consequence |
|---|---|
| **(a) Finite per planet** | Depletion is permanent; hauling distance rises monotonically; the depletion index becomes a *map* of history; late-game scarcity is real and un-tunable without a world intervention (EVE's Scarcity is the cautionary tale [V]) |
| **(b) Respawning fields / belts** (`RespawnAfter(ticks)`) | Extraction is a renewable flow; the depletion index is the only scarcity lever, and it works locally and continuously — which is what CCP named as the *absence* that caused their worst crisis [V] |
| **(c) Per-material-class mix** | Bulk ores respawn, rare deposits are finite — the richest, and the one that makes hauling *and* prospecting both matter |

> **Recommendation: (c)**, with `regeneration_policy` per material class in `EconomyTuning`. But this must
> be decided **before P4 terrain lands**, because it fixes the primary faucet's geometry irreversibly
> (§4.10). Also decide `yield_per_voxel` per material and the **net material issuance band**.

> ⚠ **WORLDLINE (rev 2) — SHARPENED, AND THE DEFAULT FLIPS.** (c) is **accepted**, on three conditions from
> `scripts/dormant_world_simulation_design.md` §4.3 / W3:
> 1. **"Respawn" must mean `RespawnAfter(ticks)` — a STEP — never continuous replenishment.** Continuous
>    regeneration racing continuous production has **no general closed form** (the stock solves a coupled
>    recurrence), so it forfeits O(1)-in-Δ evaluation and forces a bounded quantum loop with a `max_quanta`
>    cliff. This is a **determinism-algebra** constraint, not a gameplay preference (D-68).
> 2. **`RespawnAfter` becomes the recommended DEFAULT rather than an option**, because it is the only mechanism
>    in the substrate that restores anything without a player or an agent — otherwise the dormant world is
>    monotone **non-improving** and "the parts of the galaxy you invested in only decay" (§1.2).
> 3. **Its evaluation has TWO REGIMES and the naive formula is wrong in one of them**: if the machine's hopper
>    fills before the deposit dries (the likelier case), a `Stall` machine stops consuming, the deposit **never**
>    depletes, and **no respawn cycle occurs at all**. Both regimes must be in the differential-test
>    cross-product.
> Also owed with this decision: **LAW-WL-8** (the voxel edit log is the authority for geometry; the scalar
> deposit stock is DERIVED from it at declared breakpoints) and the four player-visible states of
> (rock present/absent) × (stock zero/nonzero), two of which must be **impossible** (§7.6, §6.6). `yield_per_voxel`
> is an **integer** minor-unit quantity under RULE WL-INTQ.

---

**D17. Audit granularity** *(new — revision 1 claimed per-event auditability and per-session aggregation simultaneously).*

Options and consequences are tabulated in §8.3 (options a–d). The arithmetic kills option (b): a 1-minute
window compresses by ~1.1× at the report's own ACU, and by nothing at 100 k CCU.

> **Recommendation: (a) per-event, combined with (d) — change the GAMEPLAY so fewer micro-credit events
> exist** (ESS-style pooling, which is CCP's own answer). That keeps I12 and the moderation appeal true as
> claimed, and gets the volume down honestly rather than by discarding provenance. If the storage bill is
> unacceptable, (c) session-scoped is the fallback **and §9.2 item 5 must be downgraded in the same
> commit** to "…within this session".

---

**D18. How is a realm ACQUIRED?** *(new — presupposed by D11(b), designed nowhere.)*
`Claim` (first-come, land-rush) | `Auction` (periodic, on the same clearing engine — HR3) | `Conquest`
(P11 combat coupling) | `Rent` (recurring `Sink`, softest). Each has a very different combat coupling, so
it cannot be decided after P11 begins. > **Recommendation: `Auction` + `Rent` first** (they are pure
economy and need no combat), `Conquest` as a P11 additive path.

---

**D19. Are alliances a jurisdiction TIER or an identity grouping?** *(new.)*
A jurisdiction tier breaks D10's strict tree (alliances are not spatial) and makes tax incidence ambiguous
again. > **Recommendation: identity grouping only** — used for SMP keying, contract scoping and roles,
with alliance-level revenue expressed as **voluntary transfers from member corps**, never as a tax level.

---

**D20. Corp-internal theft: gameplay or prevented?** *(new — revision 1 stated a fraud stance for
shares/banks but not for the most famous corp mechanic in the genre.)*
> **Recommendation: theft IS gameplay, but every corp action is logged, attributable and disclosed**, so
> it is detectable after the fact — the exact mirror of "fraud about the future is gameplay; fraud about
> the present balance is a missing feature". This requires the corp-action audit log and the **role-change
> `effective_tick` delay** (§7.11b) to land with the first corp wallet, not later.

---

**D21. Do recurring asset sinks exist (storage rent, structure upkeep)?** *(new.)*
> **Recommendation: yes.** It is one of the very few sinks that scales with **hoarding** rather than with
> flow, it is the mechanism that makes escheatment (§7.10) tractable, and it is the least intrusive
> inequality lever (§7.15). The only cited precedent figure (Dual Universe's 0.02/day) is **[U]**, so
> calibrate from our own dashboard rather than from it.

> ⚠ **WORLDLINE (rev 2) — SPLIT INTO A PHYSICAL AND A MONETARY HALF, because as an unqualified monetary sink it
> is a LAW-E1 violation.** "Structure decommissioned because rent went unpaid" is a **monetary → physical**
> coupling. `scripts/dormant_world_simulation_design.md` §2 **row 27** splits it:
> - **PHYSICAL (decoupled):** a `Construction`'s **condition** is a worldline stock draining at a per-entity
>   integer rate — an ordinary closed form — so a structure **wears and can physically decommission on its own
>   schedule with the economy off**, which is also what makes it advance while dormant (LAW-E2).
> - **MONETARY (M-internal):** money-denominated rent is a `vd-econ` **receivable**. It may never itself destroy
>   a structure; unpaid rent accrues and is collected or written off when the economy is available.
> This preserves the hoarding-scaled sink and the escheatment mechanism while keeping the world's decay
> economy-independent.

---

**D22. Seasons / wipes / leagues?** *(new.)*
> **Recommendation: decide now, build for it either way.** "Never wipe" makes retention (D-54),
> escheatment and inequality policy load-bearing **forever**. "Seasonal" makes `season_id` a **partition
> key from day one** (free now, a migration later) and makes M(0) recurring. Path of Exile's league →
> Standard merge is the shipped precedent for the seasonal shape [V].

> ⚠ **WORLDLINE (rev 2) — DECIDE THIS BEFORE W-0, because the day-one artefacts are now being frozen.**
> "Free now, a migration later" is exactly right, and the artefacts in question are the worldline's:
> **`RealmUid`**, the `WL_*` `StoreKey` families and the durable record schema
> (`scripts/dormant_world_simulation_design.md` §11.1, §12.3-12). `season_id` is **absent from all three**. If
> seasons are ever possible it joins them now at zero cost; if "never wipe" is decided, say so explicitly,
> because it makes retention (D-54 and the worldline's own **D-79** touched-realm GC), escheatment and inequality
> policy load-bearing **forever**.

---

**D23. Is there a sanctioned CREDIT dimension?** *(new — revision 1 answered banking only by prohibition.)*

| Option | Note |
|---|---|
| **(a) No credit at all** | Simplest; the money-supply model stays purely faucet/sink |
| **(b) Fully-collateralised NPC/corp loans with mechanical liquidation** | Safe: **no endogenous money creation**; a loan is an **asset+liability entry PAIR, not a faucet**, so I3's shape is unchanged; liquidation is a deterministic threshold on the collateral's ledgered value |
| **(c) Fractional / uncollateralised** | Rich, and a solvency-enforcement problem; it **creates money endogenously**, which changes the I3 identity's shape and the money-supply dashboard |

> **Recommendation: (b).** It gives interest as a monetary lever and corporate **bonds** as the natural
> sibling of the requested "stocks of companies" — trading on the same clearing engine (HR3) — without
> endogenous money. **Derivatives stay explicitly out of scope** (they multiply the solvency surface and
> add nothing the requested feature set needs), stated as a decision rather than by omission.

---

**D24. Is a public/community economic data feed published, and at what granularity?** *(new.)*
Revision 1 drew the right lesson from Albion Data Project ("ship the feed ourselves or third parties will
sniff the wire") and modelled the dashboard on EVE's public MER, but never surfaced publishing as a
decision. > **Recommendation: publish AGGREGATES and INDICES only** (per-realm-subtree volumes, prices,
faucet/sink totals, the indices), on a stated cadence with a stated delay, under stated terms; **per-account
data stays operator-only** — a public per-account feed makes targeted scamming and market surveillance
trivial. Cheap now, awkward once third parties depend on a format.

---

## 12. Open questions / what we could not verify

### 12.1 Design questions with no answer yet

1. ~~**Is a fill within one venue purely LOCAL?**~~ **PROMOTED TO DECISION D15.** It was not an open
   question but an unresolved **contradiction** in revision 1 (§6.4 rule 1 required escrow at the account
   owner while §7.2 assumed the venue holds it). It cannot be "confirmed"; it must be **chosen**, and the
   choice changes the throughput model and the E-7 slice shape. See D15.
2. ~~**What is our answer to a hub realm exceeding one shard's capacity?**~~ **CLOSED in revision 2**
   (§7.5): **`Market(RealmId, CommodityId)` — option A's own key — already partitions one venue's books
   across K single-writer authorities**, so the hot venue is never the hot writer. Quantified: The Forge's
   409 k orders spread over ~2–4×10⁴ active types ⇒ ~10–20 orders/book, the hottest single book O(10³)
   orders ≈ 60–200 KB; the hottest region's flow is **2.3–3.0 trades/s mean / 29 postings/s**, three orders
   of magnitude under the corrected per-shard write ceiling. **The hot market is a hot REALM problem
   (orders + clients + escrow + read fan-out), not a throughput problem** — addressed by the per-commodity
   partition + `max_orders_per_book` + the client fan-out caps. Admission control and fee-based shedding
   remain the policy backstop. **Residual open question:** does splitting one venue's books by commodity
   break the "one book per venue" player mental model enough to matter? (A UI question, not a
   correctness one.)
3. **How is the coarse `f(seed, universe_tick)` price field made consistent with the leaf book's last
   recorded state across a teardown/spin-up cycle**, given the seamlessness hard rule (no visible jumps)?
   The sketch is "seed-derived field + a small persisted deviation", but the exact reconciliation, the
   conservation assertions, and what a player observes if they leave and immediately return are
   unspecified.
4. **How is the economic event log made globally mergeable and tamper-evident?** Candidate: per-shard
   append-only stream keyed `(universe_tick, Fence, TransferId, seq)` with a per-shard hash chain. Is a
   hash chain warranted, or are the redb WAL + directory CAS sufficient provenance?
5. **Goods taxonomy and the item→goods mapping.** How many abstract goods (Veloren 15, econsim 4, Victoria 3
   dozens), and is the mapping from concrete blocks/items to goods a hand-authored BOM or derived from
   recipes? This is a **one-way decision** whose cost scales with P6 content growth, and it must be vetted
   against the full end-goal (signal-heavy cross-shard blocks, ships and stations built from blocks).
6. **How much of EVE's mechanism inventory do we actually want?** Contracts and couriers are clearly
   load-bearing for a distributed market; LP stores, PLEX, skill injectors and Hypernet lotteries are
   EVE-specific monetisation/sink artefacts. **Which sinks replace them**, given that block-built ships
   and voxel terrain give us destruction and construction sinks EVE would envy?
7. **How far back does the ledger stay online?** Retention/rollup policy fixes what questions the
   dashboard can ever answer, and it forces D-6's deferred WAL retention/compaction work earlier.
8. **Does an account whose owner is unreachable FAIL or QUEUE?** Orleans' virtual-actor precedent
   (activation-on-demand, never "unavailable") argues for queue, which needs a bounded queue with
   backpressure and a defined shed-loud behaviour — our `SendError::QueueFull` idiom already provides the
   shape [R].
9. **Which distributions do we actually need, and can each be an integer CDF / alias table?** Zipf or
   lognormal for wealth and order sizes; Poisson or exponential for arrivals. If any genuinely needs a
   transcendental, we need a pinned in-house polynomial plus a cross-platform test — this is the
   determinism hole most likely to survive to production and only show up on a different CPU architecture
   (e.g. aarch64 vs x86_64 in the k3d cluster).
10. ~~**What exactly is the G-IDENTICAL economy fixture?**~~ **CLOSED in revision 2** (§6.5 G5), because
    leaving it open while D2's *only* HR3/HR4 safeguard depended on it was untenable — HR4 is a hard rule
    ("or it doesn't land"). The fixture: the **same** `place-order → match → settle → tax →
    conservation-check` scenario, driven by the identical fixture, on a **station realm** and on a **ship
    realm**, asserting **identical ledger traces modulo ids** (canonical renaming of
    `AccountId`/`OrderId`/`RealmId`, then a byte comparison of the posting sequence). **It lands in slice
    E-1, not E-8** — the fixture is what keeps the capability honest while the code is still small.
11. **Should the order book live inside the shard's ECS World at all, or as a separate pure state machine
    outside it?** matchcore's command→outcome model suggests the latter: a pure state machine fed commands
    and emitting outcomes is trivially snapshottable, replayable and coverable, and it decouples the book's
    lifecycle from the realm's. Worth deciding deliberately rather than defaulting to "it is a `bevy_ecs`
    resource".
12. **Does the roadmap get an economy P-slot now, or does the design wait until after P9?** §10 argues for
    the cheap seams now + the design before P6 + the implementation P9-adjacent, but the slot itself is a
    user call. ⚠ Note that revision 2 moves **four** items earlier than "P9-adjacent": S11 (the inventory
    representation rule) must precede P6/P7; S12 (closed-form ore) must precede P4; S13 (the destruction
    sink) must precede P11; and E-(−1) (the measurement spike) should run before any of the §2.2 figures
    is treated as a budget.
13. ⚠ *new* **How does the P9 Signal design answer the relay question?** Only `profiles::galaxy()` and
    `profiles::station()` carry `signal_relay` today [R]. Does an up-path System/Planet realm need a relay
    capability, a *processing* capability, or neither (i.e. does Signal routing tunnel through
    relay-incapable intermediates)? This is now S10's central question and it blocks nothing else — but
    getting it wrong would silently break every cross-realm economic notification.
14. ⚠ *new* **What can a DORMANT realm actually process?** LF-1 guarantees at-least-Dormant, and
    Dormant-as-a-cheaper-capability is itself deferred ("v1 runs a full server per realm even when
    unoccupied" [R]). Until that optimisation lands, "Dormant" means "a full server", and every economy
    design decision that leans on dormancy is really leaning on *kill* vs *not-killed*. When the cheaper
    Dormant arrives, the economy's minimum capability set in that state must be specified.
15. ⚠ *new* **Which per-shard budget numbers are real?** §2.3's proposed defaults are derived from
    published third-party benchmarks and arithmetic, not from our engine. E-(−1) replaces four of them
    with measurements; until then they are **planning figures, not budgets.**

### 12.2 A determinism hazard the research did not flag

⚠ **`rustc-hash` (FxHash) is fixed-seed but NOT architecture-portable, and NOT insertion-ordered.**

The research claimed "FxHasher has a FIXED seed, so iteration is insertion-order-deterministic". The
verifier corrected this [V]:

- **Fixed seed: TRUE** — `FxBuildHasher` is a unit struct with a `const fn default()`.
- **"Insertion-order-deterministic": FALSE** — std `HashMap` iterates in bucket/hash order, so a fixed
  seed makes iteration *reproducible-but-arbitrary* and dependent on capacity and insertion history.
- **The seed constant and rotation DIFFER BY TARGET POINTER WIDTH**: `K = 0xf1357aea2e62a9c5` / `ROTATE = 26`
  on 64-bit vs `K = 0x93d765dd` / `ROTATE = 15` on 32-bit ⇒ **hash order is not identical across
  architectures.**
- `rustc-hash` also ships `FxRandomState` (randomly seeded) and `FxSeededState`, so "rustc-hash is
  deterministic" is a property of *which state type a dependency chose*, not of the crate.

**Consequence for us:** `matchcore` and `pathfinding` remain deterministic *per target*, but any output
derived from Fx-map **iteration** must be sorted before it reaches wire or state, and a 32-bit target
would diverge. This belongs in the economy design's determinism section, and arguably in
`crates/core/src/collections.rs`'s module doc.

### 12.3 Figures that must be re-sourced before entering a design doc

| Claim | Status | What to do |
|---|---|---|
| TigerBeetle "42,000 TPS measured / 60,000 burst / 15,000 batched PostgreSQL / designed for 1 M TPS" | **[U]** — appears nowhere in current TigerBeetle docs or README [V]; likely c.2022 material | Do not cite. If throughput matters, benchmark our own kernel |
| TigerBeetle batch limit | **8,189** per the docs' request table vs **8,190** on the performance page [V] | ⚠ **Do NOT make it load-bearing at all.** It is a **client-request event cap** derived from TigerBeetle's ~1 MB message frame ÷ 128 B record (exceeding it returns `ERR_TOO_MUCH_DATA`) — a wire-protocol constant of a system we are not using. Revision 1 multiplied it by our tick rate to produce the 1.6×10⁵ ledger-ops/s "ceiling", which bounds nothing. **Retracted** (§2.2). |
| ⚠ **Revision 1's "~1.6×10⁵ ledger-ops/s, ~50× EVE's average"** | **RETRACTED** — a non-derivation (above). Re-derived from redb 2.6.3's own benchmark: **~40,000 durable postings/s whole-shard on desktop NVMe, ~13,000 for the economy's share, ~3,000–7,000 on k3d `local-path`** ⇒ **4–14× EVE's 2.9 k/s**, and even that is apples-to-oranges (one EVE SQL txn may hold many postings) | **Measure our own `RedbStore`** (slice E-(−1)) before any figure enters a design doc |
| ⚠ **Revision 1's "41–47 GB/yr" ledger** | **WRONG TWICE**: the upper bound was computed against 3.65e8 instead of 7.3e8 (7.3e8 × 128 B = **93.4 GB**), and the input event count was EVE's **trade count only**, 10–50× low. Corrected: **300–600 postings/s ⇒ 0.53–2.4 TB/yr** | Reconcile the k3d PVC sizing (256 Mi/shard, 1 Gi/orch [R]) against it — currently off by 3–4 orders of magnitude |
| ⚠ **Revision 1's "1,000 seeds fit in ~30 core-minutes"** | **29× arithmetic error** (1,000 × 51.8 s = 14.4 core-**hours**), on top of a 1 M ticks/s figure extrapolated from an economy-free harness (realistically **2.4 core-years** at 20 Hz with agents) | Re-derive from the cadence hierarchy: at the clearing cadence a 30-day seed is ~518 k ticks and the twin is genuinely cheap (§8.7) |
| ⚠ **Revision 1's "~9× value/order" (Jita)** | Does not follow from the stated shares: **68.8/24.6 = 2.80×**; value/order vs rest-of-universe = 1,465/226 = **6.5×** | Use 2.8× (share ratio) or 6.5× (value/order), never 9× |
| ⚠ **Revision 1's "~1.7 M valid (region,type) pairs"** | Conflated the live **order** count (1.55–1.67 M [V]) with a pair count. 114 regions × 50,301 types = **5.73 M possible**; ~353 k **active** [V] | And restate cardinality at **our** venue granularity (station/area × item kinds = 2.5×10⁸–5×10⁹ possible), bounded by config (§2.2) |
| ⚠ **Revision 1's "1 KB per empty book ⇒ 1 GB at 10⁶ books"** | **4–7× pessimistic**: two empty `BTreeMap`s (24 B each) + a `Slab` + an id index ≈ **200 B** ⇒ ~150–250 MB | The conclusion (lazy + empty-cheap) is unaffected |
| ⚠ **Revision 1's Trap-1 "40 M rows/day → 2–5 M" (10–20×) via a 1-minute window** | **Arithmetically impossible**: 40e6 ÷ 1,440 min ÷ 24,518 CCU = **1.13 events/player-minute** ⇒ ≤1.13× compression; **zero** at 100 k CCU | The 10–20× can only come from session-scoped flush, which costs per-event audit granularity ⇒ decision **D17** |
| ⚠ **Revision 1's "petgraph / rayon are already approved deps [R]"** | **FALSE in this worktree** — neither appears in any `Cargo.toml` [V], and `docs/design/d6_saga_wal.md:123` says "rayon remains un-adopted" [V]. Both were recollections from the OLD project's evaluation, mislabelled as in-repo facts | Re-tag as **[U]/user-decision**; hand-write integer min-cost flow in `vd-econ`; keep petgraph Tier-B only; use process-level parallelism for the twin |
| ⚠ **Revision 1's "`Money` is a fixed 16-byte primitive ⇒ canonical postcard bytes"** | postcard v1 encodes `i128` as a **zigzag varint**, not 16 fixed bytes | Restate as "one canonical varint encoding per value, no trailing-zero/scale freedom"; add a golden byte-pin test |
| ⚠ **Revision 1's "the up-path is at-least-Dormant WITH `signal_relay`"** | Only `profiles::galaxy()` and `profiles::station()` carry `signal_relay`; `system()`/`planet()`/`ship()`/`area()` do not, with a unit test asserting it [R] | Restate as at-least-Dormant only; relay/processing capability is an open P9 requirement (§6.4, OQ 13) |
| ⚠ **Revision 1's "a tailing sidecar tails the shard's redb"** | **IMPOSSIBLE** — redb 2.6.3 takes `flock(LOCK_EX \| LOCK_NB)` at open [V]; and the cited `outbox.rs` precedent is an **in-process** `Arc<Mutex<Box<dyn OutboxSink>>>` [R] | Use the `EventSink` seam (B) or in-process export (A′) — §8.1 |
| ⚠ **Revision 1's "`MsgClass::Bulk`-style"** | **No `Bulk` arm exists**; `MsgClass` is wire-frozen append-only [R] | The market read path needs a real carrier decision (§7.2) |
| ⚠ **Revision 1's "a dead market owner is a re-home (D-37 machinery exists)"** | `ReHomeState` has **one** arm, `PoseOnly` [R] ⇒ a re-homed account/book loses every balance and order | `ReHomeState::Snapshot` (P7) + D-31 become blocking deps; I18 makes the gap fail loud |
| ⚠ **Revision 1's `applied_steps` durability claim ("zero new delivery machinery")** | The journal is an in-memory `BTreeSet`; the durable table is **owed (D-22 🟥)** [R] | Name it a blocking dependency of D-48/E-1 and fold N6's receipt into its schema |
| Ginko Financial: 69.7%/yr rate, ~US$750 k destroyed, the 2008-01-22 no-interest-without-charter rule | **[U]** — the cited page supports only L$55 M vs L$180 M and the gambling trigger [V] | Re-source from Linden Lab's 2008 banking policy and contemporaneous reporting, or drop the figures |
| Entropia Calypso Land Deeds: 60,000 deeds, 25% of net income, weekly payout | **[U]** — the cited page supports only 10 PED = US$1 [V] | Re-source from MindArk / Planet Calypso announcements before using it as the equity template |
| Albion fees (2.5% setup on create+edit, 8% / 4% sale tax) | **[U]** — `wiki.albiononline.com` is bot-blocked [V] | Re-source with a browser |
| Dual Universe: flat + 1% + 0.02/day storage tax, 5-minute order-edit cooldown | **[U]** — support site unreachable [V] | Re-source with a browser |
| Elite Dangerous BGS details | **[U]** — the cited guide is unreachable [V] | Re-source if the influence-coupling rule is adopted |
| Offworld Trading Company "buy = 2× sell, then collapsed to one price" | **[U]** — cited URL 404s [V] | Re-source from Soren Johnson's Designer Notes / GDC talks, or drop the quantitative claim |
| `en.wikipedia.org/wiki/Call_auction` | **404** [V] | Cite "Call market" or "Double auction" instead; the mechanism is standard and unaffected |
| Veloren's smoothing constants (0.8 on values and labour; `val = 2^(1−surplus/demand)`; the `sum/1000` industry floor) | **[U]** — not located in `mod.rs`; likely in `context.rs`/`cache.rs` [V] | Re-read those files if the family is adopted. The economy-of-scale constant IS verified verbatim [V] |
| EVE "~524 T ISK/day (Sept 2022)" vs "777.3 T ISK/month (Feb 2026)" | **Mutually inconsistent** [V] | Use only the MER-derived figures in §2.1 |
| Doran & Parberry page range (35–47 vs 35–48) | **[U]** | Trivial; check the PDF if cited formally |
| "Lighting-match-engine-core: 8 ns per order execution" | **[U]** — repo README marketing, no independent benchmark [V] | Do not cite as a performance fact |
| "`limitbook` is 565 SLoC"; "`lobster` is ~1.5 k lines with two f64 uses" | **[U]** — not measured [V] | Measure if the port plan depends on it |
| Eco's current 9.x–11 law/contract/store API surface | **[U]** — `wiki.play.eco` / `docs.play.eco` are Cloudflare-blocked; reconstructed from Steam dev blogs [V] | Re-check with a browser before mirroring any specific API |
| `matchcore`'s benchmark figures (61–142 ns/submit) | Vendor-reported, **not reproduced** [V] | Benchmark our own port |
| `matchcore` license: crates.io "MIT OR Apache-2.0" vs repo Apache-2.0 only | **Ambiguous** [V] | Resolve before porting any *code* (design-porting is unaffected) |
| ClickHouse "2–4 M rows/s ingest"; GreptimeDB maturity; `tdigest`/`sketches-ddsketch`/`hyperloglogplus`/`augurs`/`linfa` versions and licenses; Grafana/Prometheus/`clickhouse`-crate versions; `plotly`/`charming`/`plotters`/`egui` versions; `bevy_egui` 0.39 pinning `egui ^0.33` | **[L]/[U]** | Verify at the moment of the D7/D14 decisions, not now |

### 12.4 What we deliberately did not research

- **A second shipped DECENTRALISED-market reference with primary sources.** EVE is the only fully verified
  data point in the topology dimension and it is a *central-database* data point. Albion and Dual Universe
  are the right candidates but both are currently `[U]`. A verified decentralised-market case study would
  materially de-risk the region-sharded design and is the highest-value follow-up research.
- ⚠ **The actual per-tick cost of an econ system in our engine — PROMOTED to slice E-(−1), do it FIRST.**
  Every per-tick, per-op and per-byte figure in §2.2/§2.3/§7.7/§8.7 is imported from an external
  microbenchmark or from arithmetic, and the review demonstrated that revision 1 then **spent those
  numbers as if they were budgets** — with errors of 12–50× (the commit ceiling), 10–50× (the ledger
  volume), 29× (the twin) and 4–7× (the empty-book footprint). A ~200-line spike measuring
  `vd-econ` inside a real `step_tick` replaces **four** estimates with measurements:
  (a) durable postings/s against our own `RedbStore`, (b) ns per agent evaluation, (c) ticks/s for the
  twin at the economy cadence, (d) rehydrate ms per 10⁵ ledger rows. **No §2.2 figure should enter a
  design doc before this runs.**
- **Client-side UX for markets** (order entry, book display, wallet, contracts). Out of scope as *visual
  design*; but the **contract** and the **latency budget** are now specified (§7.17), because "no client
  prediction" makes them a server-side design decision rather than a UI one: expose `posted` and `pending`
  separately, render `available = posted − pending` from the server only, echo the **idempotency key +
  resulting fence** in a receipt, and author an explicit `Accepted{…}` pending state so the client renders
  a real server state instead of predicting.
- **Player-facing economic *content*** (what specific items are mined, crafted, hauled, or destroyed). That
  is game design, and it depends on P6 blocks. ⚠ Note the distinction from §4.10: the **rates and the
  balance equation** are *not* deferrable content — they are pre-P4/pre-P11 seams.

---

## 13. Review record

Three adversarial critics reviewed revision 1 of this document on 2026-07-26.

| Critic | Verdict | Findings | Disposition |
|---|---|---|---|
| **1** | `SOUND_WITH_FIXES` | 7 HIGH, 6 MEDIUM, 4 LOW + 6 missing sections | 17 of 18 findings sustained and fixed; **1 rejected on the merits** (below); all 6 missing sections written |
| **2** | `MAJOR_REWORK` | 3 CRITICAL, 6 HIGH, 4 MEDIUM, 2 LOW + 12 missing | all sustained and fixed; §2.2 rebuilt from one consistent chain and §2.3 added |
| **3** | `SOUND_WITH_FIXES` | 5 CRITICAL, 12 HIGH, 9 MEDIUM, 2 LOW + 33 missing | all sustained; 11 new sections written, 10 new decisions added |

**The corrections that changed a CONCLUSION, not just a number:**

1. **§8.1 event-log carriage** — the recommended option (redb + an out-of-process tailing sidecar) is
   **impossible**: redb 2.6.3 takes `flock(LOCK_EX | LOCK_NB)` at open [V], and the cited `outbox.rs`
   precedent is an in-process sink [R]. Retracted; the `EventSink` seam (B) is now recommended with its
   HR5 cost stated, plus an in-process-export alternative (A′). **§8.1.1 backpressure** added — the journal
   is `LossBudget::ZERO`, so ring-full **fails the action loud**, never drops a posting.
2. **§6.3/§11 D2 authority placement — INVERTED.** Agents (5×10⁶ evals/round = 1–2 core-s) and RLM
   lifecycle (no way to spawn/protect a non-spatial never-dormant node) both forbid the one-economy-shard
   recommendation. Now: option **A** keyed `Market(RealmId, CommodityId)` from day one, with B as a
   reversible deployment configuration. Its latency row was also corrected.
3. **§2.2 — rebuilt.** The 1.6×10⁵ ledger-ops/s ceiling was a non-derivation (a TigerBeetle *client
   request* cap × our tick rate); re-derived from redb's own benchmark to **~13,000 postings/s for the
   economy's share, ~3,000–7,000 on k3d storage**. The ledger volume was understated 10–50× **and** carried
   a 2× arithmetic error; corrected to **0.53–2.4 TB/yr**, which is 3–4 orders of magnitude above the
   provisioned PVCs. **§2.3 added**: per-shard budgets, the missing cadence hierarchy, and the agent-load
   arithmetic that inverted D2.
4. **§4.8 T8 tax remittance — REWRITTEN.** Pull-from-a-possibly-dead-situs is backwards under HR1+RLM;
   now push-with-retention (`ProducerLessReliable` + `Durability::Retained`), with the explicit rule that a
   Signal may never be the only carrier of the credit half of an applied debit.
5. **§6.4 D3 durable-dormant — CONDITIONED.** Blocked on a storage-topology prerequisite (`ReadWriteOnce` +
   `local-path` + 256 Mi bound to a StatefulSet ordinal [R]) that makes G4 unpassable as written.
6. **§6.3 the `ReHomeState` blocker** — `ReHomeState` has one arm (`PoseOnly`), so re-homing an account or
   book loses all value. `ReHomeState::Snapshot` (P7) + D-31 are now named blocking dependencies, and **I18**
   makes the gap fail loud.
7. **§6.4/§7.2/D15 the escrow contradiction** — revision 1 asserted both "escrow at the account owner" and
   "a fill is local because the venue holds both sides". Promoted to a decision, with both branches costed.
8. **§7.5/S5 venue capability** — `ShardProfile` is a pure function of realm KIND [R], so a bare field makes
   *all* stations venues. Split into S5a (cheap) and S5b (a real RLM slice).
9. **§4.1 batch clearing** — the Coq claim downgraded (uniqueness of *volume*, not of *price*), the full
   tie-break total order named, and the **dirty set** made mandatory because the sweep scales with
   cardinality.
10. **§2.2 conclusion 3** — "analytics is single-node" survives the volume correction; **"no rollup
    pyramid" does not** (a 180 GB year-scan is 90–180 s) ⇒ exactly one mandatory daily fact table plus
    published p95 query targets.

**Sections added in response to the "missing" lists:** §2.3 (per-shard budgets + cadence), §4.9.1
(ledger positions vs the landed transient-loss machinery), §4.10 (the material economy), §4.11 (the
economic action taxonomy), §6.6 (live-invariant behaviour + reversal), §7.1.1 (`vd-econ`'s clippy/DetRng
rule set), §7.8 (direct P2P trade), §7.9 (contracts/collateral/courier), §7.10 (account authority
lifecycle + escheatment), §7.11 (territory/ownership/alliances + the corporation organisation layer),
§7.12 (recipes vs player-built machines), §7.13 (guaranteeing regional divergence), §7.14 (insurance),
§7.15 (M(0) + onboarding + inequality policy), §7.16 (FX policy + tax currency), §7.17 (order-ack latency
+ the pending-ack contract), §7.18 (economy → world-state feedback), §8.1.1 (journal backpressure),
§8.3.1 (journal schema evolution), §8.3.2 (seasons/wipes), §8.8 (the HR6 `vdctl`/`DevState`/`WaitField`
surface), §8.9 (`EconomyTuning`'s home + epoch reconciliation), §9.4 (the incident runbook), §9.5
(authorization/authenticity of economic commands).

**Invariants added:** I15 CONTRACT-CLOSURE, I16 FAUCET-BUDGETED, I17 SELF-REPORT-CROSSCHECK, I18
LEDGERED-STATE-CARRIED, I19 ACCOUNT-ALWAYS-HOMED, I20 LEDGERED-KIND-LOSSLESS, I21 DIRTY-SET-SOUND; I13
strengthened from `≤` to an equality with a reclaimable term.
**Gates added:** G6b, G7b–G7e, G9, G10, G11, plus a **wall-clock budget column** and an inner/pre-merge
tiering for all of them.
**Ledger entries added:** D-58…D-66. **Seams added:** S5b, S11, S12, S13 (and S2/S4 re-homed).
**Decisions added:** D15–D24.

**One finding REJECTED on the merits.** Critic 1 (LOW) claimed §6.1's N10 mis-cites
`generic_transfer.md` §A5, saying the design "actually specifies `applied_damage: HashSet<DamageEventId>`
(`docs/design/generic_transfer.md:250`)" and that the report reads as though a fix is already landed.
**Verified: the report's citation is correct.** §A5's spec at `generic_transfer.md:165` says
`applied_damage: BTreeSet<DamageEventId>` — **BTreeSet**, as the report states — and carries an explicit
type note — *"NOT `HashSet` — the
sim/node clippy `disallowed-types` bans the default-hasher `std::collections::HashSet`, and this set
CROSSES the transfer barrier into the byte-identical-replay surface"* — i.e. exactly the reason the report
gives. Line 250 is the older **adversary-response summary** line, not §A5. The finding did surface a real
*documentation* defect, which is recorded as an errata in §6.1 N10: line 250 of that design is stale and
should be corrected independently of the economy work.

---

*End of report (revision 2, 2026-07-26). The **eight** gating decisions are D1–D5 + D15–D17; the full set
is D1–D24. The cheap anti-cornering seams are S1–S13 (with S5 split into S5a/S5b and S5b explicitly NOT
cheap). The proposed ledger entries are D-47…D-66. Do slice **E-(−1)** — the in-engine measurement spike —
before treating any §2.2/§2.3 figure as a budget.*
