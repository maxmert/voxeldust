# N2 — THE BUILD DESIGN (plan step 4 built; ruling B13 item 3)

**What this file is.** A DESIGN. It changes no source file and no world file. It stands on branch `game`, HEAD
`c399f54`: `GENERATOR_VERSION` 21 (`crates/terrain/src/tag.rs:233`), `ARTIFACT_VERSION` 9
(`crates/terrain/src/artifact.rs:95`), `PROTO_MINOR` 34 (`crates/wire/src/version.rs:349`). Four read-only maps of the
code fed it, and the designer read every anchor below again in the code. Every claim about the code carries a
`file:line`. **ARITH** marks hand arithmetic. **UNMEASURED** marks a number nobody printed. **PREDICTED** marks a
reading of a static file, not a run.

**REVISION 2 (2026-09-26).** A review of the first draft found 29 defects. This revision answers each one. It applies
the fix where the code agrees with the finding, and it states why where the code does not. The **REVIEW LEDGER** at the
end lists every finding with its outcome.

**What the owner ordered** (ruling B13 item 3, `docs/design/owner_decisions_2026-09-22_believability.md:377-381`):
build N2 as the integer shore law in the recipe, on the server and on the client (one generator, two hosts, no drift —
SL10); the bench measures it again against its f64 reference; then the pilot flies ALL stands against BOTH reference
pictures. NARROW WATER at the far swaps stays RED for slice 0b.

**Where N2 lives today.** Only in the bench, in f64: `crates/bins/src/shore_rules.rs` (`candidates` 774-830,
`candidate_for` 832-979, `point_side` 1108-1136, `finish` 1213-1300, `n2_octave_ramps` 1377-1401, `finish_alone`
1508-1514, `fold` 1536-1561) and `crates/bins/examples/shoreline_bench/eval.rs` (`Ctx::eval` 306-652). The shipped path
still runs today's rule R0 (`crates/recipe/src/height.rs:412-434` through `crates/recipe/src/plan.rs:214-247`).

---

## 0. The short answer

1. **The near rungs are buildable now, exactly, with no new data.** On the home planet rungs 0–9 read the rows. The
   tiles a chunk waits for (`crates/terrain/src/artifact.rs:1572-1596`) are exactly the union of N2's 6 × 6 node
   blocks (`crates/bins/src/shore_rules.rs:550-557`). The fold at rungs 8 and 9 reads no extra node on the home planet
   (§3.6).
2. **The far rungs are NOT buildable from what the client holds.** From rung R₁ = 10 up, N2 reads each fine node's
   height and each lake node's level. The client holds only a pyramid level, the 2-bit mask and the counts there
   (`crates/client/src/artifact_book.rs:175-195`). So the build meets an SL6 question EARLIER than the one B13 names:
   **Q1, the shore band** (§2.5, §7). B13 names only narrow water.
3. **The 30 m band makes the far data small and EXACT** (§2.3). A node's banded word is its side's default wherever
   its height stands 30 m or more from the water. So the far rungs need a height only for a node within 30 m of a
   water that its node cells read, plus each lake node's level. With that datum, and ONE stated rule for a water that
   the fold names outside a column's node block (§1.2 item 9), every rung reads the SAME words on both hosts.
4. **The far fold is ONE derived structure per body, and only the shore's own node cells pay for it** (§3). The held
   read keeps every value between its node cell's four corner words, so only the node cells that the shoreline crosses
   need their 64 × 64 samples. The client derives their records and a sample count pyramid ONCE per body, on the
   terrain workers' share. Every reader shares them by pointer: the chunk, the halo, the ladder's span, the morph and
   the eye. **The far globe WAITS for the pyramid**: a body with a sea shows no far globe while its pyramid derives
   (ARITH 2–15 s for a hemisphere on two workers, UNMEASURED). Slice S0 measures it. Q3 sets it against shipping the
   counts, and a gate holds it under the owner's number.
5. **ONE naming rule for the fold at every folded rung** (§1.4). A cell is a LAKE where the lake samples are at least
   half of the wet ones, at the HIGHEST lake level among its lake samples; else it is the sea. A count pyramid carries
   this rule to the top rung, so nothing changes at the 13 → 14 swap. The rule replaces the bench's "the water with the
   most samples" at rungs 8–13, so S0 benches it again (Q7).
6. **The reach becomes a law** (§4, Q2): `reach = C_kind × wavelength`, with one measured correction per octave kind.
   The measurement also REFUTES the spectrum's own sine law by 2.6–3.1 ×. Q2 tells the owner so.
7. **The card** compiles the new kernel in written-out shapes, and a probe measures it (§5). The card still builds no
   artifact chunk (ruling F9, parked). The CPU path of the same source ships (ruling F8).
8. **ONE switch at every rung** (`GENERATOR_VERSION` 22) after the band lands, then the flight (§6). The switch also
   takes the stated-side R0 arms out of the recipe, so no host that holds an artifact can reach R0. The collider slice
   will read the one reader. The identity's measured word moves: two of the six rung-0 golden keys stand on a shore in
   their node blocks (PREDICTED, §6 S4).

**Example.** A pilot lands on the gentle coast. At rung 3 her boots stand on columns whose node blocks come from the
tiles under her; N2 draws the swells down to a smooth beach. She climbs to 1 400 km. At rung 11 the client holds no
tile of that coast. With the shore band it holds each coastal node's height within 30 m of the sea, so rung 11 draws
the same line rung 9 drew, and the coast does not move at the 704 km swap. Without the band, rung 10 draws the node
square again, and the coast jumps about eight pixels there (ARITH, §2.4).

---

## 1. THE INTEGER N2 — every step of the f64 reference in the recipe's integers

### 1.1 The words and their formats

| word | format | bound (proved in the kernel's doc) | anchor |
|---|---|---|---|
| a radius (`R`, the ground `B`, a level `W`, the surface `h`, D7, the sea) | gap steps at 28 fraction bits; 1 m = 2³⁵ raw | the column's own range, as today | `crates/recipe/src/cell.rs:38`; `crates/terrain/src/units.rs:33-37`; `crates/terrain/src/artifact.rs:1230-1231` |
| a node word, RELATIVE to its water | whole metres `d ∈ [−BAND, BAND]`, BAND = 30 m, at 28 fraction bits ("metre-28") | `|p| ≤ 30·2²⁸ < 2³³` | the band, `crates/bins/src/shore_rules.rs:916`; the constant `BAND_M` in `vd-recipe` (§4) |
| a fraction `ta`, `tb` | 28 bits, `[0, 1)` | — | `crates/terrain/src/artifact.rs:1817-1824` |
| a Catmull-Rom weight | 28 bits, the recipe's own | `|w| ≤ 2²⁸`, `Σ|w| ≤ 1.25·2²⁸` (ARITH, the maximum at `t = ½`) | `crates/recipe/src/macro_field.rs:29-40` |
| a derivative weight (NEW) | 28 bits | `|w′| ≤ 1.39·2²⁸`, `Σ|w′| ≤ 3·2²⁸` (ARITH, the maximum at `t = ½`) | the f64 form, `crates/bins/src/shore_rules.rs:579-587` |
| the side field `S` | metre-28 | `|S| ≤ 30 m` (the held read stays inside its corner words) | `crates/bins/src/shore_rules.rs:589-634` |
| its slope `g` (a magnitude) | metre-28 per node | STATED `1 m/node ≤ g ≤ 159 m/node`, `159·2²⁸ < 2^35.4`: each axis is at most `Σ|w|·Σ|w′|·30 = 1.25 × 3 × 30 = 112.5` m per node, so the magnitude is at most `112.5·√2`. The EXACT maximum over the ±30 m vertices is 112.5 m per node (ARITH: every one of the 2¹⁶ sign vertices on a 128 × 32 grid of fractions; where one axis reaches 112.5 the other axis's sum is nought) | the f64 form, `crates/bins/src/shore_rules.rs:651-670` |
| the distance `d = |S|/g` | nodes at 28 bits | `0 ≤ d ≤ 30` nodes (`|S| ≤ 30 m`, `g ≥ 1 m` per node): five integer bits, NO cap | §1.3 step 4 |
| the division's remainder (NEW) | metre-28 | `< 2g ≤ 318·2²⁸ < 2^36.4` | §1.3 step 4 |
| a distance word | nodes at 28 bits, signed | `|word| ≤ √32 − ½ = 5.157` nodes (a stencil node whose only other-side node stands at the block's far corner), or the stated 3.5 where the block holds no such node | `crates/bins/src/shore_rules.rs:952-966` |
| `dist` (the plain read of the sixteen distance words) | nodes at 28 bits | `≤ 1.25² × 5.157 = 8.06` nodes `< 2³¹`: the UNHELD read compounds `Σ|w| ≤ 1.25` on both axes | `crates/bins/src/shore_rules.rs:967` |
| `D = lesser(d, dist)` | nodes at 28 bits | `≤ 8.06` nodes | `crates/bins/src/shore_rules.rs:1394` |
| a share (`Ω`, `β`, `φ`, a fade) | 28 bits, `[0, 1]` | — | `crates/recipe/src/noise.rs:75-87` |
| `q_k` = node over octave `k`'s reach (NEW) | 28 bits | `≤ 256·2²⁸` on the home planet (ARITH: 8 192 / 38) | §4 |
| a ratio product `|S_k|·g_b` (the dry pick) | 128 bits | `≤ 30·2²⁸ × 159·2²⁸ < 2^68.3` | `crates/recipe/src/wide.rs:12-30` |

Every word the recipe multiplies is a `Gi`: it has no division and no negation (`crates/recipe/src/gi.rs:9-12`). A
product goes through `*` and one `>>` where a stated bound keeps it under 2⁶³, and through `Gi::mul_shr` everywhere
else (`crates/recipe/src/gi.rs:116-127`).

### 1.2 The node cell's words (the host, `vd-terrain`, once per node cell)

A "node cell" is the square between four fine node centres: the node `(ni, nj)` at or before the column's centre and
the fractions `(ta, tb)` across it (`node_and_fraction`, `crates/terrain/src/artifact.rs:1817-1824`). Every column
whose centre falls in one node cell reads the same words, so the chunk builder keeps them in a small per-chunk map keyed
by `(face, ni, nj)`. The map is only a memo: every other reader computes the same words with the same function (§1.7).

1. **The site.** The same `(zf, zi, zj)` the column's `Z` read uses (`crates/terrain/src/chunk.rs:359-369`). A halo
   column across a seam therefore reads the PARTNER face's node cell, which is the partner chunk's own
   (`crates/terrain/src/lattice.rs:141-192`). **A CORNER PHANTOM** (a column off the face on both axes,
   `lattice.rs:187-191`) names no cell. Its `Z` reads its KEY face's cell clamped to −1 or to the edge
   (`chunk.rs:361-367`), and `node_at` fills every node that stands off the face on both axes with that face's own
   corner node (`artifact.rs:1838-1840`). **THE STATED RULE:** the phantom's node cell is that same clamped cell, and
   its side is the POINT side of that cell at every rung. A phantom has no fold, because its rung-F samples would be
   phantoms too. So the three chunks that meet at a cube corner hold three different phantoms, as they hold three
   different `Z` today. The owed corner blend (`CORNER_BLEND`, `artifact.rs:109-111`) is the cure (Q12). From rung 16
   up the phantom's whole block stands off the face on both axes, so its words are its face's corner node's own (§2.3).
2. **The block.** The 36 nodes `[ni − 2, ni + 3]²` through the seam table (`crates/bins/src/shore_rules.rs:550-557`;
   `crates/terrain/src/artifact.rs:1831-1865`).
3. **Per node:** its side word (land, sea, pocket, lake; `crates/terrain/src/artifact.rs:296-344`), its lake level
   (the row's `water_m` for a lake word, `crates/bins/src/shore_rules.rs:410-427`), and its height `z_m` (§2.3 names
   where each host finds it).
4. **The waters** of the block: the sea, and each lake level (D1: two lake nodes of one level are one water,
   `crates/bins/src/shore_rules.rs:432-450`, 464-500).
5. **Per water `b`, the sixteen stencil words** (offsets −1…2): the mirror, then the band, on whole metres
   (`crates/bins/src/shore_rules.rs:687-695`, 903-917; the mirror covers all sixteen words, `:181`):
   - a node wet for `b`: `d = max(min(z − W, −1), −BAND)`;
   - a node not wet for `b` at or under `W`: `d = min(max(W − z, 1), BAND)`;
   - any other node: `d = min(z − W, BAND)`.

   This is exact integer work. The f64 reference computes the same whole numbers, so the two hosts and the reference
   read identical words (a check, §1.8).
6. **Per water `b`, the sixteen distance words**: for each stencil node, the least squared offset `d²` to a block node
   of the other side for `b`. The offsets along each axis are 0 … 4 (a stencil node at −1 … 2, a block node at
   −2 … 3), so `d²` takes FOURTEEN values: 1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25 and 32. Nine of them are not
   squares (ARITH). The word is `⌊√d² · 2²⁸⌋ − 2²⁷` from `isqrt(d² << 56)` (`crates/recipe/src/root.rs:46-82`, an exact
   floor), read from a TABLE of fourteen words that the f64 reference reads too (§1.8 item 1). It is `7 · 2²⁷` (3.5
   nodes, a stated choice, §4) where the block holds no such node, and negative for a node wet for `b`
   (`crates/bins/src/shore_rules.rs:952-966`).
7. **Per water `b`, the four presence bits**: a node wet for `b` within the 5 × 5 around each corner node
   (`crates/bins/src/shore_rules.rs:927-941`).
8. **Per water `b`, the UNIFORM bit**: the node cell's four corner nodes are all wet for `b`, or all not wet for `b`
   (§3.2 uses it).
9. **THE DEFAULT RULE — a water with no wet node in the block.** From rung 14 up the fold can name a water whose wet
   nodes all stand outside the column's own 6 × 6 block. ARITH at rung 16: a column's rung-7 samples lie in the node
   cells `8i − 1 … 8i + 7`, their wet corners in `8i − 1 … 8i + 8`, and the column's own block is `8i + 1 … 8i + 6`.
   For such a slot every node word is `W + BAND`, every distance word 3.5 nodes and every presence bit 0 — on EVERY
   host, whatever the field holds. The rule lives in the one reader (§1.7), so a host that reads the band and a host
   that reads the rows read the same words. (The f64 reference reads each stencil node's own height there,
   `crates/bins/src/shore_rules.rs:903-917`, called for such a water at
   `crates/bins/examples/shoreline_bench/eval.rs:508-551`. It takes the same rule as a named choice, §1.8 item 1.)
   Under the rule the slot reads `S = +BAND`, `g = g_min`, `Ω = 0` and `β = 0`: it gives a wet column its level and
   the wet hold `k × BAND` = 7.5 m, nothing more (ARITH). At rungs 0–13 on the home planet the rule never fires (§2.3).

**Example.** At rung 16 a pilot's cell over the plain holds a tarn at its far edge, 30 km from the cell's centre. Most
of the cell's wet samples are the tarn's, so the fold names the tarn. The column's own block holds no tarn node. So the
shard and the client both read the tarn's slot as `W + 30 m`: the column stands under the tarn's level, and the two
hosts agree to the bit.

ARITH: 36 node reads (a `BTreeMap` lookup each on a tile cache, `crates/terrain/src/artifact.rs:724-733`), 16 words
and 16 × 36 integer compares per water — about 2–3 µs per node cell. At rung 9 a node cell holds 256 columns, so a
column pays about 0.01 µs. From rung 13 up each column is its own node cell and pays the whole 2–3 µs where its block
holds a shore (UNMEASURED); §1.5's uniform-sea rule and the no-candidate path cut the open sea and the dry inland to a
scan of the mask.

### 1.3 The column's slots (the recipe, per column, per candidate)

1. **The candidates and the cap.** Each water's nearest wet node by the Chebyshev distance from `(ta, tb)`, a tie to
   the lower node index; the four nearest are kept (`crates/bins/src/shore_rules.rs:774-830`). The distances are exact
   multiples of 2⁻²⁸, so the integer order is the f64 order. A FIFTH slot is held for a water the fold names that the
   cap dropped (`crates/bins/examples/shoreline_bench/eval.rs:508-551`). So the kernel has five fixed slots.
2. **`S`, THE HELD READ, WRITTEN OUT.** One held pass is `v = (Σ pᵢ·wᵢ) >> 28` with `*`, then `v` is held between its
   two middle words (the f64 pass, `crates/bins/src/shore_rules.rs:591-600`). The hold is a SIGN MASK, the shape the
   terrace uses to pick its nearer bed top (`crates/recipe/src/terrace.rs:230-232`), never a branch. The read is TEN
   passes WRITTEN OUT in a row — four row passes and one pass across them rows first, the same columns first (the f64
   loops, `crates/bins/src/shore_rules.rs:605-624`) — so no loop carries a clamped value (§5). Bound:
   `|Σ pᵢwᵢ| ≤ 1.25 × BAND × 2⁵⁶ < 2^61.3`; a `const` assert beside the pass states `1.25 × BAND_M < 128` (§4).
   `S = (v_rows + v_columns) >> 1` (the f64 takes the exact mean, `:627-634`; the floor differs by at most 2⁻²⁹ m).
3. **`g`, THE PLAIN GRADIENT.** The unheld Catmull-Rom with the NEW derivative weights
   `w′(t) = ½[−3t²+4t−1, 9t²−10t, −9t²+8t+1, 3t²−2t]` at 28 bits, `t² = (t·t) >> 28` as `weights` computes it
   (`crates/recipe/src/macro_field.rs:29-40`; the f64 form `crates/bins/src/shore_rules.rs:579-587`, 651-670).
   - The inner row sums use `*`: bound `3 × BAND × 2⁵⁶ < 2^62.5`. A `const` assert beside them states
     `3 × BAND_M < 128`, so a band of 43 m or more fails to compile (§4).
   - The outer sums use `mul_shr`, the way `catmull_rom_16` does (`crates/recipe/src/macro_field.rs:47-63`), because
     their bound (`2^62.8`) leaves no margin.
   - THE MAGNITUDE: take each axis's `unsigned_abs` FIRST, as `slope_share` does
     (`crates/terrain/src/artifact.rs:1998-2007`). A signed `>> 4` floors toward −∞, so without it `g` would depend on
     the axis's sign by one unit. Then shift both axes to 24 fraction bits, square each (`< 2^61.7`), add in an
     unsigned word (`< 2^62.7`), `isqrt`, and shift back to 28 bits.
   - 24 bits and not 20: at 20 bits the relative error of `g` (10⁻⁶) moves `β` by 2 × 10⁻⁶, which moves a column whose
     blend spans up to 2 km of height by 2–4 mm and breaks the check's band; at 24 bits the same path moves it about
     0.2 mm (ARITH).
   - `g = greater(g, ONE)`: the floor `g_min` is one metre per node.
4. **`d = |S| / g`, EXACT, WITH NO CAP.** `d = ⌊|S|·2²⁸ / g⌋`. Because `|S| ≤ 30 m` and `g ≥ 1 m` per node, `d < 32`
   nodes, and the quotient has 33 bits. The division is a RESTORING DIVISION WITH A SHIFTED REMAINDER — the shape
   `recip_pow2` uses (`crates/recipe/src/root.rs:109-127`) — WRITTEN OUT as 33 masked steps, the way `isqrt` writes
   out its 32 (`crates/recipe/src/root.rs:46-93`; `:27-36` records why a loop shape fails on the card):
   - the remainder starts at `|S| >> 5`, which is under `g` (it is under 2²⁸, and `g ≥ 2²⁸`);
   - each step shifts the remainder left by one and brings in the next bit of `|S| << 28` (the five low bits of `|S|`,
     then 28 zeros), compares the remainder with `g` by a mask, subtracts `g` under the mask, and sets the quotient's
     bit under the mask;
   - the remainder stays under `2g ≤ 318·2²⁸ < 2^36.4`, so no step can wrap.

   The isqrt shape — the DIVISOR shifted to the top bit — is REFUSED: `g << 31` passes 2⁶³ for any `g` of 16 m per
   node or more. `recip_pow2` itself does not serve: it is a loop meant once per body; `recip30` covers only `[1, 2)`
   (`:129-151`).
5. **`dist`.** The plain read of the sixteen distance words at `(ta, tb)`, WRITTEN OUT (sixteen products, four row
   sums, four outer products), then its magnitude (the f64 `plain_read`, `crates/bins/src/shore_rules.rs:967`). It
   does not call `catmull_rom_16`, whose nested loops never ran on the card (§5).
6. **`Ω`.** `lerp(lerp(o₀₀, o₁₀, ta), lerp(o₀₁, o₁₁, ta), tb)` on the four bits (`crates/recipe/src/noise.rs:85-87`;
   the f64 four-product sum, `crates/bins/src/shore_rules.rs:942-945`).
7. **`β = (Ω · (ONE − fade(lesser(d, ONE)))) >> 28`** (`crates/bins/src/shore_rules.rs:1099-1102`;
   `crates/recipe/src/noise.rs:75-81`; `fade(ONE) = ONE` exactly, ARITH).
8. **`D = lesser(d, dist)`**, the distance the octave fade reads (`crates/bins/src/shore_rules.rs:1394`).

**Example.** On the steep belt coast the banded side field rises 60 m over one node, so `g` reads about 60 m per node.
A top-bit division would shift `g` past 2⁶³ and read a wild `d`, and the long swell would come back at the water's
edge. The shifted-remainder division keeps its remainder under 120 m per node and reads `d = |S|/60` exactly, so the
pilot's column fades its swell where the reference fades it.

### 1.4 The side

1. **THE POINT SIDE** (rungs up to F; F = 7 on the home planet, §3.6): wet for the lowest-level slot whose `S < 0`,
   a tie of levels to the LOWER water key, the sea first (`crates/bins/src/shore_rules.rs:1108-1126`, the tie at
   `:1117`). Else dry, governed by the least `d`; a tie of `d` goes to the higher level, and a tie of both to the
   earlier slot (`:1127-1135`). The integer compares the ratios without a quotient: `|S_k|·g_best < |S_best|·g_k`
   through the 128-bit `wide::mul_wide` (`crates/recipe/src/wide.rs:12-30`; both products under 2^68.3, §1.1).
   `g_used = S_gov`.
2. **THE FOLD** (rungs over F) — ONE RULE AT EVERY FOLDED RUNG, from rung 8 to the top rung. Over the rung-F point keys
   under the column:
   - the column is WET where the wet samples are at least half (a tie wet, W15.1 item 4);
   - a wet column is a LAKE where the lake samples are at least half of the wet ones (a tie a lake: W15.1's own rule,
     `crates/terrain/src/artifact.rs:474-489`), at the HIGHEST lake level among its lake samples — `block_water`'s
     own rule (`crates/terrain/src/artifact.rs:1598-1623`) applied to samples;
   - else the column's water is THE SEA.

   Every tie is decided: two lakes of one level are one water (D1). A count pyramid carries all three words (the wet
   count, the lake count, the highest lake level) to the top rung (§3.4), so the rule at 13 → 14 is the rule at
   8 → 9. **This rule REPLACES the bench's fold at rungs 8–13**, which names "the water with the most samples, a tie to
   the higher level" (`crates/bins/src/shore_rules.rs:1536-1561`; its tie of count AND level goes to the first water
   scanned, `:1553-1559`). S0 benches the new rule before anything is built on it (Q7).
   - Wet: the named water's slot governs, `g_used = −|S_gov|` — the fifth slot where the cap dropped the water, the
     default rule of §1.2 item 9 where the block holds no wet node of it.
   - Dry with candidates: the least `d` governs, `g_used = +|S_gov|` (`eval.rs:499-560`).

   §3 says how the game counts the keys.
3. **THE DECISION** reaches the kernel as data: a wet mask word, the governing slot's level `W`, `g_used` and `β`.

**Example.** A cell at rung 12 over a coastal marsh holds 40 % sea samples, 35 % of a lagoon at 3 600 m and 25 % of a
pond at 3 610 m. The bench's rule names the sea (the most samples). The new rule names a lake, because 60 % of the wet
samples are lakes, at the pond's 3 610 m. The rung-14 cell over it reads the same three words from the pyramid and
names the same water, so no level jumps at the 13 → 14 swap.

### 1.5 The surface kernel (one recipe entry, in the card's safe shapes)

A NEW entry beside `column_surface_at`. After S4 `column_surface_at` keeps only the ground's own sign, for the card and
for a host with no side data (§6 S4). All heights below are over the ladder radius `R`; `S` becomes a length by `<< 7`,
because one metre is 2²⁸ units at metre-28 and 2³⁵ units of length.

**THE SIDE IS DECIDED FIRST, OUTSIDE EVERY LOOP.** §1.4 reads only the slots and the sample counts, never the octaves.
The kernel then takes ONE of three paths:
- **NO CANDIDATE, AND NOT WET BY THE FOLD** — today's loop (`relief_parts_from`, `crates/recipe/src/height.rs:302-332`)
  and the ALONE arm (step 7).
- **WET** — today's loop and the WET arm (step 6). The WET arm reads only the fine sum `fine_m`: the reference fills
  the per-octave parts only for a dry column (`crates/bins/examples/shoreline_bench/eval.rs:581`) and reads them only on
  the dry arm (`crates/bins/src/shore_rules.rs:1226-1229`). A column whose fold is wet takes this path even with no
  candidate of its own: its named water's slot reads the default rule (§1.2 item 9).
- **DRY WITH CANDIDATES** — the φ loop (step 1), then steps 2–5.

The choice among the paths stands OUTSIDE every loop, so the card's rule holds, and a wet column or a dry inland
column pays today's loop.

1. **THE φ LOOP (the DRY path only).** ONE INDEX LOOP over the octaves `[first, count)` of the fixed table,
   `octave_term` inside it as in `relief_parts_from`, and THREE ADDITIVE ACCUMULATORS: `whole += term`,
   `fine += term & mask`, `kept += mul_shr(term & mask, φ_k, 28)`. `φ_k` is computed fresh in each iteration from the
   five slots, written out slot by slot (no inner loop, no carried product):
   `t = lesser(ONE, mul_shr(D_b, q_k, 28))` — inside the loop the `lesser` is written as a sign mask, the way the
   terrace picks its nearer bed top (`crates/recipe/src/terrace.rs:230-232`) — then
   `factor_b = ONE − ((Ω_b · (ONE − fade(t))) >> 28)`,
   `φ_k = (((factor₁·factor₂ >> 28)·factor₃ >> 28) …)`. An empty slot has `Ω = 0`, so its factor is `ONE` exactly.
   `q_k` is read from the shore charter by the loop index, as `octaves[k]` is. After the loop, `fine_m = mul_shr(fine,
   m, 28)` and `kept_m = mul_shr(kept, m, 28)`: the factor still multiplies ONCE (`crates/recipe/src/height.rs:
   302-332`). This is the only loop shape the card has measured safe: additive accumulators, masks in the data, no
   value a branch assigns inside the loop (`crates/recipe/src/height.rs:191-203`; `crates/recipe/src/cell.rs:254-261`).
2. **`raw = terrace(R + B + fine_m)`** as today (`crates/recipe/src/plan.rs:235-241`).
3. **DRY, WITH CANDIDATES** (`crates/bins/src/shore_rules.rs:1385-1400`): `x = raw − R − fine_m`,
   `y = W_gov + (g_used << 7)`, `node = x + mul_shr(y − x, β_gov, 28)`, `s₁ = node + kept_m`.
4. **THE DRY HOLDS** (`crates/bins/src/shore_rules.rs:1253-1277`, 677-681). For each slot:
   `b1d = B + mul_shr((W_b + (|S_b| << 7)) − B, β_b, 28)`; `hold_b = W_b + λ(b1d − W_b)`, where `λ(x) = x >> 2` for
   `x ≥ 0` and `x << 2` for `x < 0` (k = ¼, K = 4); a lake slot subtracts `mul_shr(A_r, ONE − Ω_b, 28)`. `lb` is the
   greatest. Where no slot is the sea and the body has a sea: `lb = greater(lb, sea + λ(B − sea))`.
5. **THE SIDE HOLD** (`crates/bins/src/shore_rules.rs:1278-1293`): for each sea slot,
   `lb = greater(lb, W + ((|S| << 7) >> 2) − mul_shr(A_r, ONE − Ω, 28))`; where D7 names a water,
   `lb = greater(lb, D7 + ((|g_used| << 7) >> 2))`. Then `s = greater(s₁, lb)` and the standing water is D7
   (`:1294-1299`). Q11 names an exemption for a pocket's floor and two measured variants of the D7 floor (§7).
6. **WET** (`crates/bins/src/shore_rules.rs:1234-1252`): `h_far = terrace(R + B + mul_shr(fine_m, ONE − β, 28)) − R`,
   `h₁ = h_far + mul_shr(y − h_far, β, 28)`, `s = W − greater(W − h₁, (|g_used| << 7) >> 2)`; the standing water is
   `W`. The reference applies the terrace in two orders — `terrace(B + (1−β)f)` here and `terrace(B + f) − f` on the
   dry side — and the port keeps both.
7. **ALONE** (no candidate, a side word held; `crates/bins/src/shore_rules.rs:1508-1514`):
   `s = greater(raw − R, sea + λ(B − sea))` on a body with a sea, `raw − R` on a dry one; the standing water is D7.
8. **NO SIDE DATA** (the golden top level, the card, a body with no artifact): the ground's own sign — today's
   `SIDE_UNKNOWN` arm of `shore()` (`crates/recipe/src/height.rs:412-434`), which S4 keeps as the recipe's only other
   shore rule. N2 needs side words by construction; where a host holds none, the ground's own sign decides, as every
   host did before the mask (`crates/terrain/src/artifact.rs:1454-1460`). **A far field WITHOUT its band and lake
   levels is NOT "no side data"**: there `shore_column` answers `None` and the chunk waits (§3.5). It never answers R0.
9. The biome reads the final surface, as today (`crates/recipe/src/plan.rs:241-246`).

**THE UNIFORM-SEA RULE.** Take a column whose block's ONLY water is the sea, whose 36 block nodes are all wet for it,
and whose sixteen stencil words all stand at the default `W − BAND`. The kernel's own arithmetic then gives:
- `S = −BAND` exactly: each final pass is held between two equal corner results;
- `g = g_min`: the integer derivative weights sum to within two units of nought (each `>> 1` floors once), so the read
  slope of sixteen equal words stays about a hundred units, far under `ONE`, and the floor gives `ONE`;
- `d = 30` nodes and `β = 0` (`fade(ONE) = ONE`).

So the WET arm reduces to `s = lesser(raw − R, W − 7.5 m)` and `l = W` (ARITH). The host tests the rule on the mask's
36 words and on the sixteen stencil nodes' sea words: from the tiles at a near rung, and at a far rung from the band's
membership bit (a node the band does not hold reads its default) — about 0.2 µs (ARITH). A test proves that the closed
form equals the full kernel bit for bit on 10⁶ random deep-sea columns. Its failing control: the closed form on a
column with one CORNER word at `W − 29 m` must differ from the full kernel.

**Example.** A hull cruises 150 km over the open ocean. Every column under it is uniform sea. At rung 16 each one pays
today's loop, one compare and the rule's test — not the 2–3 µs of node-cell words that a coastal column pays.

### 1.6 The standing water, the beach, the cells and the sheet

- The column's standing water becomes N2's `l`: `W` on the wet side, D7 on the dry side
  (`crates/bins/src/shore_rules.rs:1245-1251`, 1294-1299). D7 is today's land rule unchanged
  (`crates/terrain/src/artifact.rs:1239-1243`; the bench reads it at `crates/bins/examples/shoreline_bench/eval.rs:
  393-410`). Its word — POCKET or not — comes from the node counts at a far rung (§3.4).
- The cells (`crates/recipe/src/cell.rs:352-363`), the sheet (`crates/terrain/src/position.rs:406-440`) and the eye
  (`crates/terrain/src/height.rs:138-161`) read that word through `ColumnAt.water` and `SampleBox.water`. Their code
  does not change; their input does.
- The beach: the coast bit AND `s ≥` the hold water — `W_gov` for a column with candidates, the sea for a column with
  none. This is today's rule (`crates/terrain/src/chunk.rs:417-421`) moved onto N2's governing water. No pixel shows
  the beach until 8e (Q9).

### 1.7 THE ONE READER

A NEW function in `vd-terrain`, called here `shore_column`, does steps 1.2–1.5 for one column. It answers the surface,
the standing water, the hold water and the biome — or `None` where its field cannot say:
- a tile is not here;
- a far field holds no band or no lake levels yet;
- a rung's fold reads the sample pyramid (§3.4) and the pyramid of a face it reads has not landed.

The fold's records and the pyramid come from the book's SHARED copies (§3.3–3.4), never from a per-chunk map. Every
reader calls `shore_column`, in one change (ruling W16.2, `docs/design/owner_decisions_2026-09-21_water.md:1076-1082`):
- **`ColumnRead::column`** (`crates/terrain/src/chunk.rs:340-431`): the chunk and its halo
  (`crates/terrain/src/lattice.rs:385-457`), and the ladder's span (`crates/terrain/src/digest.rs:349-398`,
  `crates/client/src/ladder_view.rs:1356`).
- **`field_column`** (`crates/terrain/src/height.rs:166-226`) — THE MORPH TARGET OF EVERY FACE-SEAM VERTEX AND EVERY
  FALLBACK VERTEX, AND THE SOURCE OF THEIR SHADE. A vertex on or past a face's edge skips the parent meshes
  (`crates/client/src/chunks.rs:1257-1267`; the filter at `:1082`) and takes its target from `coarser_field_m` →
  `height_field_m` → `field_column` at the coarser rung (`chunks.rs:1173-1186`, `:1296`). `field_normal` reads four
  more points for its shade (`chunks.rs:1101-1122`). So along all twelve cube edges, in every crossfade band, this
  function places the seam vertices. It reads `Z` at the rung-0 cell under the direction (`height.rs:174-179`), the
  node cell's words at the direction's own fractions (so the target is continuous along the seam), and the SIDE — the
  point side or the fold — of the rung's OWN cell, as it reads the side today (`height.rs:198-202`). Cost (ARITH,
  UNMEASURED): the node-cell words (2–3 µs) and one memo or pyramid read per call, five calls a seam vertex — about
  15 µs a seam vertex, or 1–2 ms more for a chunk on a face edge (62–124 seam vertices).
- **the eye and the ruler** (`crates/client/src/chunks.rs:2470-2509`, `:2575-2594`), through `drawn_height_field_m` →
  `field_column`.
- `column_water` stays for D7 and for the pocket word (§3.4).

`field_column` cannot equal the chunk's column bit for bit above rung 0: it finds its rung-0 cell from a FLOAT
direction (`height.rs:174-178`) and sums the octaves at `direction_of_unit(dir)` (`height.rs:207`), while the chunk
reads the rung-r cell's centre and the integer site direction. A rung-r centre stands on a rung-0 cell corner, half a
rung-0 cell from each of four cells. So S2 tests it bit for bit at rung 0 and within HALF A CELL OF THE RUNG above it
(§6 S2). Reading the node cell at the rung's own centre instead would make the target a step at every cell border,
which a morph cannot carry.

**THE FALLBACKS STAY PROVISIONAL.** The span and the eye also read "the finest whole level" when the rung's own field
is not whole (`crates/client/src/ladder_view.rs:1361-1363`; `crates/client/src/chunks.rs:2503-2507`). That level may
hold no mask and no band, so under N2 the read there is the no-side rule (§1.5 step 8). It stays a STAND-IN. The span
marks it provisional and reads it again at every descent (it does today, `ladder_view.rs:1363`). The eye treats it as it
treats the recipe's own relief that it falls back to after it (`chunks.rs:2475-2476`). No FINAL span and no built chunk
reads it.

**Example.** A coast crosses the edge between two faces of the home planet. The seam vertices of a rung-9 chunk morph
toward the rung-10 surface through `field_column`. The rung-10 chunk draws its columns through the same `shore_column`
on the same shared records. So the seam vertices slide onto the shore that the rung-10 chunk draws, and no stripe shows
along the cube edge at the 704 km swap.

### 1.8 THE CHECK — the integer N2 against its f64 reference

The bench runs it on every column of every window at rungs 0–13 (`crates/bins/examples/shoreline_bench/stands.rs:
123-130`), and from S2 on the seam windows too (§6 S2). Before the switch (slices S1–S3) the integer path is read by
nobody but the bench; after it (S4) the integer path IS the shipped column.

1. **THE SAME LAW ON BOTH SIDES.** The f64 reference reads the SAME words as the integer, through three named choices
   in `NamedChoices` (`crates/bins/src/shore_rules.rs:94-207`):
   - `reach=law`: its `octave_reach[k]` becomes `node / (q_k / 2²⁸)` (the benched reaches are an f64 Monte Carlo,
     `crates/bins/examples/shoreline_bench/main.rs:497-595`);
   - `dist=table`: it reads the host's fourteen 28-bit distance words, as `fade_f64` already reads the recipe's integer
     fade (`crates/bins/src/shore_rules.rs:672-675`), where it read `hypot − 0.5` through libm (`:961-964`);
   - `outside=default`: the default rule of §1.2 item 9.

   A one-time test proves `⌊hypot(Δi, Δj)·2²⁸⌋ = isqrt(d² << 56)` for all fourteen values of `d²` (ARITH: all
   fourteen agree; the nearest root stands 0.006 of a unit from a whole unit, at `d² = 8`, far over f64's rounding).
   Without the three choices the check compares two laws.
2. **CHECK-N2-INT-WORDS** — the node words, the distance words and the presence bits the integer host builds against
   the f64 reference's: 0 differences (band 0). The node words are whole metres, the distance words are one table on
   both sides (item 1), the presence bits are bits, and the default rule applies on both.
3. **CHECK-N2-INT-DECIDE** — the side, per SAMPLE and per CANDIDATE:
   - A column, or a fold's sample, is a NEAR TIE where ANY candidate's f64 `|S_b| < 10⁻⁶ m` (a non-governing water
     near nought can change which water governs), or where two candidates' f64 distances meet
     `|d_a − d_b| ≤ 10⁻⁶ · max(d_a, d_b) + 10⁻⁷` nodes. The relative term covers `g`'s 24-bit root (about
     1.2 × 10⁻⁷ of `d`, and the dry pick compares `d` up to 30 nodes, `crates/bins/src/shore_rules.rs:1127-1135`). The
     absolute term covers `S`'s floors (at most 2⁻²⁷ m over `g ≥ 1` m per node).
   - The point side and its governing water: 0 differing columns outside the near-tie set (band 0).
   - The fold is compared on its SAMPLES: every sample's key at band 0, except the near-tie samples (counted and
     printed beside the line); then the fold's three words and its decision at band 0 on every column whose samples all
     agree.
   - THE FAILING CONTROL: one sample key flipped inside the integer fold must read RED.

   **Example.** A lagoon at 3 600 m lies beside the sea at 3 591 m. A column stands 5 m deep in the lagoon and exactly
   on the sea's line: `S_sea = +3·10⁻⁹ m` in f64, one unit under nought in the integer. The f64 column is governed by
   the lagoon and the integer column by the sea. The first draft's set tested only the governing water's `S`, so this
   column was no near tie, and the check went red on correct code. Now the sea's own `|S|` puts it in the set.
4. **CHECK-N2-INT-VALUE** — on every column whose decision agrees: `max |s_int − s_f64| ≤ 0.001 m`, the band
   CHECK-R0-F64 already uses (`crates/bins/src/shore_rules.rs:184`; census line 23). The standing water is equal word
   for word (band 0: both are whole-metre levels or ZERO).
5. **CHECK-N2-INT-DRAWN** — the drawn wet flag (`s < l`) is equal on every agreeing column, except where
   `|s − l| < 1 mm` (a tie at the level, counted).
6. **CHECK-N2-RECORD-KEYS** — band 0, with NO near-tie exemption: on every window, every sample key that a node-cell
   record holds equals the rung-7 column kernel's own point key at that sample. The records call the column kernel's
   own functions (§3.3), so any difference is a defect of the batching. Its failing control: a record built with the
   rows-first passes alone must differ.
7. **THE FAILING CONTROL.** The host's word builder takes the band as an argument; the shipped value is `BAND_M`. With
   31 m it must read RED on CHECK-N2-INT-WORDS and CHECK-N2-INT-VALUE. A check that cannot go red proves nothing.

Why 1 mm is enough (ARITH): each integer step rounds at 2⁻²⁸ of its own unit; the widest carrier is `β` times a
height difference of up to about 2 km near a steep coast (the census reads RAISE-OVER-RAW up to 935 m,
status §8e); with `g` at 24 bits the error there is about 0.2 mm.

### 1.9 What the build does NOT port

Form 2 never reads the lift, the bank slope, the octaves' RMS heights, the angle of repose, O1's residual `ρ`,
`n2_theta` or `n2_cross_sigma` (`crates/bins/src/shore_rules.rs:1302-1347`, 1403-1504), and it never reads
`wet_corner` (`:946-948`, printed only by the probe, `eval.rs:734`). Forms 0, 1 and 3, the signed-distance words and
the smoothed mask stay in the bench as refuted controls (status §8e). After S4 today's stated-side R0 stays there too,
as the bench's own control (§6 S4). `tanh` never enters the recipe (`crates/recipe/clippy.toml:33`).

---

## 2. THE DATA EACH HOST READS AT EACH RUNG

### 2.1 The home planet's rungs

The body: 19 rungs, 1 216 nodes a face edge, 8 192 m a node, 6 pyramid levels, R₁ = 10, F = 7, R_P = 14, the first
fine octave 4, the octave counts by rung `[14,13,12,11,10,10,9,8,7,6,4,3,2,1,1,1,1,1,1]`
(`docs/investigation/2026-09-26/shoreline_bench_n2_census.txt:64`). A rung reads the rows while four nodes are wider
than its chunk, else the finest level whose node is at least its cell (`crates/terrain/src/artifact.rs:853-876`).

| rungs | cell | the field the chunk reads | N2's side | fine octaves summed | what N2 reads | the client holds it? |
|---|---|---|---|---|---|---|
| 0–6 | 1–64 m | the rows (tiles) | the point side | 10 … 5 | the 6 × 6 block: heights, side words, lake levels | YES: every tile of `tiles_of_chunk` (`crates/client/src/artifact_book.rs:175-187`) |
| 7 (= F) | 128 m | the rows | the point side | 4 | the same | YES |
| 8–9 | 256–512 m | the rows | the fold of 2 × 2 and 4 × 4 rung-7 samples, per column | 3, 2 | the same block (§3.6) | YES on the home planet |
| 10–12 | 1–4 km | level 1 (16 km means), the mask, the node counts | the fold of 8 × 8 … 32 × 32 samples inside the column's node cell, from the shared memo | 0 | the same fine words | **NO** |
| 13 | 8 km (one node square) | level 1 | the fold over four quadrant records of four node cells | 0 | the fine words of four node cells | **NO** |
| 14 (= R_P) | 16 km | level 1 | the sample count pyramid (§3.4) | 0 | the column's own block, and the pyramid | **NO** |
| 15–18 | 32–262 km | levels 2–5 | the same | 0 | the same | **NO** |

At rungs 10 and up the fine half is nought on the home planet (census:64-65: the lift reads 0 from rung 10), so N2
there is N1's blend, the holds and the fold. Every one of them reads the node words.

### 2.2 What each host holds

- **The server (the planet's shard)** holds the whole artifact: the rows, the pyramid, the mask
  (`crates/bins/src/bin/shard.rs:326-360`). No shipped server crate builds a chunk today; the world identity is the
  server's only chunk path (`crates/bins/src/lib.rs:1194-1204`; the gateway `crates/bins/src/bin/gateway.rs:193`).
  When the collider slice lands (slice 11, `docs/design/DEFERRED.md:7763`), collision reads rung 0 (ruling W10 item 4,
  `docs/design/owner_decisions_2026-09-21_water.md:204-205`), where the rows are the words — through `shore_column`.
  After S4 the recipe holds no other shore rule that a host with an artifact can reach (§6 S4).
- **The client** holds the head, every pyramid level, the mask and the counts it folds, and the tiles within the
  ring of rung 9 (`crates/client/src/artifact_book.rs:83-99`; `crates/terrain/src/artifact.rs:936-945`, 1018-1042; the
  gateway `crates/connection-plane/src/gateway/artifact.rs:279-303`). The tile reach ends near 823 km (ARITH: 627 km ×
  1.1 × 1.17 + two nodes), so most of the rung-10 band has no tile. A pyramid level holds no fine height
  (`crates/terrain/src/artifact.rs:788-820`), and its water word is the HIGHEST lake of its block
  (`:1598-1623`).
- **v2 finding 14 holds by construction**: a far rung reads the far field and never a tile an earlier flight left,
  because `field_for` picks the field by the rung (`crates/client/src/artifact_book.rs:175-195`).

### 2.3 Why the 30 m band makes the far words exact

Take a node `n` and a water `b` at level `W`. By §1.2 step 5 the banded word depends on the height only through
`clamp(z − W, −30, 30)` and the node's wet bit:
- a node wet for `b` reads the default `W − 30` wherever `z ≤ W − 30`;
- any other node reads the default `W + 30` wherever `z ≥ W + 30` OR `z ≤ W − 30` (the mirror).

So a node reads its side's default for every water whose level stands 30 m or more from its height. A host that knows
the side words (the mask), the lake levels, and the height of every node that stands WITHIN 30 m of a water it is read
for, reads EXACTLY the words the rows give.

**"Read for", and the walk's range.** Node `n` is read for water `b` where `n` stands in the stencil of a node cell
whose 6 × 6 block holds a node wet for `b`. The server computes that membership by walking EVERY NODE CELL THAT ANY
COLUMN OF ANY RUNG CAN NAME, each block through the seam table, as the kernel reads it:
- the regular node cells: both indices from −1 to `edge − 1` on each face (a column in a face's first or last half
  node names −1 or `edge − 1`, `crates/terrain/src/artifact.rs:1811-1824`);
- the CORNER PHANTOMS' node cells. A phantom's cell is clamped to −1 or to the face's edge
  (`crates/terrain/src/chunk.rs:363-366`). From rung 13 up, `node_and_fraction` then names a node outside the regular
  range (ARITH, the home planet: −1 or 1 216 at rung 13, −2 or 1 216 at 14, −3 or 1 217 at 15, −5 or 1 219 at 16,
  −9 or 1 223 at 17, −17 or 1 231 at 18) — 132 new node cells on the home planet (ARITH: 22 a face). At rungs 13–15
  such a block holds nodes past the edge on ONE axis, which `node_at` maps up to FOUR nodes inward on the partner face
  (`artifact.rs:1841-1864`); a regular block reaches two. From rung 16 up the whole block stands off the face on both
  axes, and `node_at` names the face's corner node for all 36 (`artifact.rs:1838-1840`).

S2 checks the membership EXHAUSTIVELY over that enumeration (§6 S2, CHECK-SHORE-BAND).

**The water outside the block.** At every rung whose cell is no wider than a node (rung 13 and under on the home
planet), the fold names only a water with a wet corner in a sample's node cell (§3.2), and every such corner stands
inside the column's own block. From rung 14 up the fold may name a water outside the block. The DEFAULT RULE (§1.2
item 9) reads that slot on both hosts alike, so the band never needs a height for it, and the rows' heights are never
read there.

**Example.** The gentle coast rises one metre in fifty. A land node 4 km from the shore stands 80 m over the sea: its
word is the default `sea + 30` at every rung, and nothing about it crosses. The land node beside the shore stands 12 m
over the sea: its height rides in the band, and rung 11 reads `sea + 12`, exactly as rung 3 reads it from the tile.

### 2.4 The formulations for the rungs from R₁ up

| | formulation | the 9 → 10 swap | verdict |
|---|---|---|---|
| F1 | N2 under R₁, today's R0 from R₁ up | R0 at rung 10 draws the nearest node's word (`crates/terrain/src/artifact.rs:1346-1371`): the node square, up to half a node (4 096 m) from N2's rung-9 line; at that swap one pixel is about 512 m, so up to 8 px (ARITH). The deciding number includes 9 → 10 (census:12876: N2 0.50 px). The bench's R0 reads 8.4 px at 10 → 11 on the gentle coast (status §8e). | REFUSE: a W16 breach at the swap the owner saw pass |
| F2 | words from the mask alone (the signed-distance words, `word_sdf`, `crates/bins/src/shore_rules.rs:880-901`) at every rung from R₁ | on a gentle coast the banded heights put the line at the height crossing, the mask words put it midway: up to half a node apart, up to 8 px (ARITH); the round measured this form as a control and did not keep it (status §8e: lean 0.32, the steep coast flattened) | REFUSE, unless the owner wants it at EVERY rung (a different N2) |
| F3 | **the shore band** (Q1): the heights of the nodes within 30 m of a water they are read for, and every lake node's level | the same words at every rung (§2.3), so the benched swaps (0.50 px at 9 → 10, 0.66 px at 10 → 11 on the gentle coast, census:12876-12877) are what ships | **RECOMMEND** |
| F4 | the tiles out to the top rung's ring | exact | REFUSE: about half the rows of 8 871 936 nodes × 10 B, some 45 MB (ARITH) |

### 2.5 The SL6 ask, as the owner would read it (Q1)

- **What data.**
  - (a) THE LAKE LEVELS: the level, in whole metres, of every lake node, in node order of the mask's lake words (the
    membership is derivable from the mask, so no index crosses).
  - (b) THE SHORE BAND: the height `z_m` of every fine node that stands within 30 m of the level of a water it is read
    for (§2.3), as a sorted node list with one height each. The membership depends on the height, so the list carries
    its own indices, delta-coded.
  - (c) THE BAND'S DIGEST, in the head: two FNV-1a words folded over the lake levels and the band's entries in order,
    the way the artifact's own digest is folded (`crates/terrain/src/artifact.rs:1756-1781`).
- **From which realm to which.** The planet's shard to every client, once, with the artifact's head, like the mask —
  one new bulk arm `BulkMsg::ArtifactShore { realm, part, parts, bytes }` in parts of at most 32 KiB, and head fields
  for the band's parts, its node count and its digest.
- **Why the receiver cannot compute it.** The heights and the lake levels are the solve's output, and the solve runs
  only on the shard. The client holds them only in the tiles under the eye (§2.2).
- **Why the digest.** The head's `digest` — the word the client echoes in `ArtifactHeld` — folds the rows, the pyramid,
  its water words and the mask, and nothing else (`artifact.rs:1756-1781`). A size check sees no wrong height. And the
  gateway files a coast part under the head it holds now (`crates/connection-plane/src/gateway/artifact.rs:176-197`);
  the arm carries no digest (`crates/wire/src/channels.rs:471-476`), and a band part inherits that. So the client
  checks the assembled band against the head's band digest and REFUSES a mismatch by name (a counter). The same check
  catches a straggler part of an older head (the W16.2 lesson, `docs/design/owner_decisions_2026-09-21_water.md:
  1099-1104`).
- **The size.** UNMEASURED. ARITH for (a): about 94 000 lake nodes (G-LAKE-SHARE 4.828 % of the land, the land 21.89 %
  of 8 871 936 nodes by `HOME_PLANET_OCEAN_SHARE_Q4` 7 811, `crates/terrain/src/home.rs:40`) × 2 B ≈ 0.19 MB. ARITH
  for (b): between a twentieth and a fifth of the nodes, as design v2 D8 assumed, at about 3 B an entry ≈ 1.3–5.3 MB;
  the 30 m band should put it at the low end, because a steep coast crosses the band inside one node. The mask is
  2.2 MB (`docs/design/owner_decisions_2026-09-21_water.md:1096-1097`). Slice S0 prints BAND-NODES and BAND-BYTES
  before the owner answers.
- **What doing without costs.** F1 or F2: a seam at the 9 → 10 swap (about 8 px, ARITH), and the far swaps the owner
  saw measured are not the far swaps that ship.
- **What doing it costs.** `PROTO_MINOR` 35 with `PROTO_MINOR_FLOOR` 35 (a field append to `ArtifactHead`, which minor
  34 shipped, `crates/wire/src/channels.rs:417-437`; R2 also plans 35, so whichever lands first takes it); a gateway
  cache entry; the far view waits for the band as it waits for the mask today (ruling F9), and then for the derived
  pyramid (§3.5, Q3). `ARTIFACT_VERSION` stays 9: the shard derives the band from its stored rows at boot and stores
  nothing new. Any change to the band's rule or to the shore law bumps `GENERATOR_VERSION`, as every change of the
  shape does.

---

## 3. THE FAR RUNGS — the game's fold

### 3.1 Why a column cannot fold its own samples

At rung `r > F` a column takes the majority of `4^(r − 7)` rung-7 point keys
(`crates/bins/examples/shoreline_bench/eval.rs:480-497`): 4 at rung 8, 16 at 9, 64 at 10, 256 at 11, 1 024 at 12,
4 096 at 13, and 2²² per column at rung 18 (ARITH). The bench read them from a window's rung-7 map
(`eval.rs:264-280`); the far window W3 ran R0 alone for that reason (`eval.rs:217-218`;
`crates/bins/examples/shoreline_bench/main.rs:797`). A chunk cannot pay 4 096 point keys a column.

### 3.2 The held read's maximum principle: only the shore's own node cells pay

Each held pass stays between its two MIDDLE words. The rows-first read's last pass stays between the two middle rows,
and each of those stays between its row's two middle words: the four corner words of the node cell. The columns-first
read obeys the same bound, and so does their mean (`crates/bins/src/shore_rules.rs:589-634`). A word wet for `b` is at
most `W − 1` and a word not wet for `b` at least `W + 1` (whole metres, §1.2). So:
- where a node cell's four corners are all wet for `b`, `S_b ≤ −1 m` at every point of the cell;
- where they are all not wet for `b`, `S_b ≥ +1 m` at every point of the cell.

In integers the clamp comes after the rounding, so the bound is exact. Where every water of the block is UNIFORM over
the cell (§1.2 step 8) and the block holds at most four waters (the cap), every sample in the cell draws one key. A slot
under the default rule (§1.2 item 9) is uniform by construction. Only a node cell whose corners differ for some water —
a cell the shoreline crosses — needs its samples. (With more than four waters the cap can drop a water from some
samples and not from others, so those cells are sampled whole; the census counted no column over four candidates,
line 114 and its 175 siblings.)

### 3.3 One node cell's samples, and THE RECORD MEMO

On the home planet a node cell holds 64 × 64 rung-7 samples (ARITH: 8 192 / 128). Their fractions take only 64
values along each axis. So the host computes, per water, the four rows-first first passes at each of the 64 `ta`
values and the four columns-first first passes at each of the 64 `tb` values (512 passes), then two final passes per
sample (8 192 passes). ARITH: about 8 700 passes of about 5 ns, 30–90 µs per water per cell with the key logic.
UNMEASURED.

**The key is computed JOINTLY per sample.** A sample's key is the point side over ALL the cell's slots at once — the
lowest-level wet slot (§1.4) — so a sample wet for two waters (a lagoon inside the sea's presence) counts ONCE, under
one key (`crates/bins/src/shore_rules.rs:1108-1126`, 1536-1561). A count per water apart cannot give that.

**The same functions as the column.** The record calls the column kernel's own functions: the passes, the weights and
the point side. The batching only reuses the first passes — the four row passes at each of the 64 `ta` values, the four
column passes at each of the 64 `tb` values — and those are the same calls on the same words. CHECK-N2-RECORD-KEYS
(§1.8 item 6) holds the two paths equal at band 0.

**The record.** Per group of the cell's samples — the 8 × 8 groups of 8 × 8 samples, the 4 × 4 groups of 16 × 16 and
the 2 × 2 quadrants of 32 × 32, 84 groups in all — THREE words: the wet samples, the lake samples, and the highest lake
level among the lake samples (the fold's own words, §1.4). ARITH: 84 × 6 B ≈ 0.5 KB a mixed cell. Rungs 10, 11 and 12
read the groups; rung 13 reads the quadrants.

**THE MEMO.** The records live in the book, ONE MEMO PER BODY, shared by pointer as the coast counts are
(`crates/client/src/artifact_book.rs:93-96`). The book is copied on write (`artifact_book.rs:13-16`), so the memo is a
lock-guarded map behind its own pointer: a record is the memo of a pure function, not state. Whoever needs a record first
fills it: a chunk, the span, the morph, or the derivation job (§3.5). A record is a pure function of its node cell's
words, so the order of filling cannot change a word. So a rung-11 chunk and a seam vertex's morph toward rung 11 read
ONE fold, and neither pays for it twice.

### 3.4 The fold, rung by rung (home planet; §3.6 for another body)

| rungs | a column's samples | where they come from | cost per column (ARITH, UNMEASURED) |
|---|---|---|---|
| 8, 9 | 2 × 2, 4 × 4, all in the column's own node cell | computed per column from the cell's words | 0.1–0.5 µs where the cell is not uniform; nothing where it is |
| 10–12 | 8 × 8 … 32 × 32 inside its node cell | the memo's record (§3.3) | one read |
| 13 | 64 × 64 over four quadrants of four node cells — one NODE SQUARE | four quadrant records | four reads |
| 14–18 | the node squares under the cell: 2 × 2 at 14, 32 × 32 at 18 | THE SAMPLE COUNT PYRAMID: per node square the three words; each level up the SUMS of its four children's counts and the MAX of their levels | one read |

ONE rule (§1.4 item 2) reads the same three words at every folded rung, so the rule does not change at 13 → 14. The
first draft changed it there. It named the far water by W15.1's lake share and took its level from the pyramid's block
word from rung 14 (`crates/terrain/src/artifact.rs:1141-1155`, `:1598-1623`), after "the most samples" at rungs 8–13.
Three failures follow from the code, each UNMEASURED:
- a lower lake that shares a 16 km block with a higher tarn jumps to the tarn's level at 13 → 14;
- a cell of 40 % sea and 35 % + 25 % lakes changes from the sea to a lake there;
- a lake-majority cell whose block holds no lake NODE reads `DRY_M` and loses its water through the land rule
  (`artifact.rs:1255-1262`).

Today rungs 10–14 read one field and one word (`artifact.rs:862-867`), so the draft would have moved today's
G-RUNG-LAKE-SHIFTED step from 9 → 10 to 13 → 14, where no line reads it. Where two lakes share a cell, the lower one's
water still steps up to the higher level at the swap where they merge: that is `block_water`'s own stated price
(`artifact.rs:1605-1609`), and S0 counts it at every swap.

**The count word.** The pyramid holds every level the edge halves to, as `CoastCounts` does — six on the home planet
(census:64) — so its top word reaches 4⁶ × 4 096 = 2²⁴ (ARITH), under a `u32`. On the largest lattice the code allows
(2 048 nodes an edge, eleven levels, `crates/terrain/src/artifact.rs:358-362`) and up to 128 × 128 samples a node, the
top reaches 2³⁶ (ARITH), past a `u32`: the pyramid's word is a `u64`, or its width is chosen per body from that bound.

**THE NODE COUNTS STAY.** `CoastCounts` (`crates/terrain/src/artifact.rs:379-490`) keeps all three of its words — the
wet nodes, the lake nodes and the POCKET nodes (`:382-386`) — BESIDE the new sample pyramid. The column's own SIDE WORD
at a far rung still comes from them: `sample_side` reads the counts from rung 14 (`artifact.rs:1346-1370`;
`coast_level`, `:511-520`). D7 reads that word, and a POCKET column reads INLAND with no water and no hold
(`artifact.rs:1196-1203`, `:1253`; the bench reads the word at `crates/bins/examples/shoreline_bench/eval.rs:393-398`).
The sample pyramid names only the fold's water. Without the node counts a far pocket cell would read LAND, D7 would
take the sea through the level's stencil (`artifact.rs:1240-1243`, `:1280-1311`), and N2's side hold
(`crates/bins/src/shore_rules.rs:1290-1292`) would lift the floor to the sea's level.

**Example.** The second pocket's floor stands 8.2 km under the sea's level. At rung 13 it is a basin. With the node
counts kept, its rung-14 cells still read POCKET, D7 names no water, and the floor stays a basin at every swap — never
the mesa at each level swap that the pocket word was made to stop (`artifact.rs:246-253`).

### 3.5 What it costs, and THE FAR VIEW'S WAIT

**The costs** (ARITH, UNMEASURED):
- A chunk at rungs 10–13 reads the memo. A mixed cell under it costs 30–90 µs the first time any reader asks. A rung-12
  chunk covers about 31 × 31 node cells; if a tenth are mixed, 100 × 60 µs ≈ 6 ms, against 11.5 ms for a whole rung-9
  build today (census:17674).
- The pyramid needs every mixed node cell that the far globe shows: 0.2–0.5 million mixed cells on the globe (an
  assumption of the earlier designs, never counted) × 1.3 waters × 30–90 µs ≈ 8–60 s of one core, half that for a
  hemisphere, divided by the terrain's worker share (ruling F6, `docs/design/owner_decisions_2026-09-12_frozen_patches.md:
  84-89`): 2–15 s for a hemisphere on two workers.
- Memory: about 0.5 KB per mixed cell for the records, plus the pyramid; tens of MB (ARITH).

**THE BOOK'S NEW STATE.** Beside `coast` and `counts` (`crates/client/src/artifact_book.rs:88-96`) the `ArtifactCache`
holds:
- the band (a membership bitmap with a rank index over the heights, one read a node) and the lake levels, each `None`
  while its parts assemble;
- the record memo (§3.3);
- the sample pyramid PER FACE, `None` while that face derives.

The pyramid is DERIVED ON THE TERRAIN WORKERS' SHARE (ruling F6), never on the receive path. Today the one derived far
datum, `CoastCounts::of`, runs inline where the last mask part lands, and then the epoch moves
(`artifact_book.rs:613-625`); 8–60 s of one core cannot run there. Six jobs, one per face, start when the mask, the band
and the lake levels are whole. The faces the eye faces go first. Each landed face bumps the cache's `epoch`, so the lane
takes the new cache (`crates/client/src/chunks.rs:2200-2204`), and its `far_epoch` (item 4 below).

**THE WAITS.**
1. `masked_level` — which BOTH `field_for` and `field_at_rung` read (`artifact_book.rs:185`, `:220`) — answers a level
   only where the level, the mask, the band and the lake levels are all whole. So a far chunk, the span, the morph's
   coarser field and the eye wait alike. A head that states no band part (a dry body) has nothing to wait for, as
   `coast_held` answers for a head with no coast part today (`artifact_book.rs:131-133`).
2. From R_P (the first rung whose cell is wider than a node: 14 on the home planet, §3.6) `field_for` answers Ready
   only when the pyramid of EVERY face the chunk reads — its own face and the partner faces its halo reaches — has
   landed. `ParentMesh::build` calls `field_of` → `field_for` (`chunks.rs:473-478`, `:1153-1168`), so it waits too, and
   no worker blocks. `field_at_rung` serves any direction, so it cannot name one face: it hands out the level, and
   `shore_column` answers `None` for a column whose face's pyramid has not landed. The morph, the span and the eye then
   treat that column as not here: the span stays provisional, and the chunk that read it is stamped (item 4).
3. **READY MEANS FINAL.** Once a field answers for a rung, nothing it reads changes for that head. So no chunk and no
   parent mesh is ever built on a half-derived pyramid, and the parent cache — which keeps a mesh until `forget`
   (`chunks.rs:899-929`) — never holds a wrong one.
4. **A CHUNK WHOSE MORPH READ A MISSING FAR FIELD IS PROVISIONAL.** Two cases:
   - a rung-9 chunk built before the band: its seam and fallback targets read rung 10 (`chunks.rs:1185`, `:1296`);
   - a rung-13 chunk built before the pyramid: its rung-14 parents wait.

   Such a chunk keeps its vertices' own radii where the field answered nothing, as today (`chunks.rs:1296`). The lane
   STAMPS the build with the book's `far_epoch`: a counter bumped when the mask, the band, the lake levels or a face's
   pyramid lands, never when a tile lands. The stale test reads it beside the digest (`chunks.rs:2344-2346`). When the
   far epoch moves, the lane asks for every resident provisional chunk again, and it draws the old mesh until the new
   one lands, so no hole opens. This also heals today's rung-9 chunks built before the mask: nothing rebuilds them,
   because `state_artifact` drops chunks only for a new head (`chunks.rs:2205-2210`).

**THE HOLE, STATED.** At the TOP rung nothing coarser stands. When the head lands, the recipe's chunks are dropped
(`chunks.rs:2206-2210`), and every far rung waits. Today the globe stands empty while the pyramid and the mask arrive.
N2 adds the band's transfer and the derivation, for EVERY body with a sea that the sky shows, not only the home planet.
Ruling F9 names a hole a defect (`owner_decisions_2026-09-12_frozen_patches.md:283-290`). The owed W18 item — draw a far
body from its coarsest level alone the moment it lands — cannot hold under N2 while its pyramid derives, because the top
rung's fold reads the pyramid. So:
1. **The hole is a GATE**, G-EMPTY-GLOBE: the time from a head's arrival to the first far picture, per body with a sea,
   against THE OWNER'S NUMBER (owed, Q3).
2. **S0 measures it first**: for every body of the home system that states water, its mixed node cells, its derivation
   at the worker share, and its band's bytes and parts.
3. **Where S0's time passes the owner's number, Q3's shipped counts become the default.** The client then only sums, as
   `CoastCounts::of` does today.
4. **The eye's faces derive first, and a far chunk never builds on a partial pyramid.**

**Example.** A pilot logs in over the moon. The home planet hangs in her sky. Its head lands, its recipe chunks drop,
and her client derives the planet's sample pyramid on two workers. Until the faces she sees have landed, the planet
shows no globe — G-EMPTY-GLOBE states for how long — and it never shows a coast built on a half-derived pyramid. If the
time passes the owner's number, the shard ships the counts instead.

**Slice S0 measures it first** (CORNER-MIXED-CELLS, FAR-FOLD-COST, the hemisphere's time, EMPTY-GLOBE per body). If
the time is too long for the owner, the fallback is a second SL6 ask (Q3): the shard ships each mixed node square's
three words (6 B, ARITH about 1.2–3 MB, with their own digest in the head). That amends W15.1 item 5 ("the counts are
derived, never shipped", `docs/design/owner_decisions_2026-09-21_water.md:810-812`). Rungs 10–12 still fill the memo per
chunk.

### 3.6 Another body (SL5)

On the home planet a node is 2¹³ cells, so every node-cell edge falls on a rung-`r` cell edge for `r ≤ 12` (ARITH from
`crates/terrain/src/artifact.rs:1817-1824`): a column's samples never leave its node cell. On a body whose node is
not a power of two — the stated 5 km body has 714 cells a node (`crates/terrain/src/macro_lattice.rs:80-82`, 132-141) —
a sample can fall one node cell over, and its block can reach one node past `tiles_of_chunk`'s span
(`crates/terrain/src/artifact.rs:1572-1596`). The span then comes from the rung-F samples of the halo's end columns,
`node_F(first · 2^(r−F)) − 2` to `node_F((last + 1) · 2^(r−F) − 1) + 3`; on the home planet that is today's span, a
test proves it, and `tile_reach_m` adds a third node for the fold (`:1018-1042`). The fold always nests by rung
cells, so it is exact on every body; the sample count pyramid uses the footprint rule rounded outward
(`coast_level`, `:511-520`).

**F AND R_P FROM THE NODE'S SIZE ALONE.**
- R₁ is the first rung whose chunk spans at least four nodes: the test `level_for` makes (`artifact.rs:853-876`), read
  WITHOUT its `levels == 0` early answer (`:858`). That early answer gives level 0 at every rung on a body with no
  pyramid, where R₁ would not exist.
- F = R₁ − 3.
- R_P is the first rung whose cell is wider than a node: the first rung whose footprint level (`coast_level`,
  `:511-520`) is one or more (W15.1 item 3: the level comes from the node's own size,
  `owner_decisions_2026-09-21_water.md:803-806`).

On the home planet R₁ = 10, F = 7 and R_P = 14, as before. The 5 km body's edge (11) never halves, so it has no pyramid
level; there R₁ = 6, F = 3 (ARITH: 62 × 2⁶ = 3 968 ≥ 4 × 714) and R_P = 10 (ARITH: 2¹⁰ = 1 024 > 714). On a body with no
pyramid the fold from R_P up sums the node squares' three words under the cell DIRECTLY: at most (cell / node + 2)²
reads, a few dozen on the 5 km body (ARITH). A test in S2 runs R₁, F, R_P and the direct sum on the 5 km body.

### 3.7 Narrow water stays RED

The fold is a majority, so water about one far column wide flickers under it: the gentle coast 4.84 px at 11 → 12
(census:12878), the plain's lakes 8.3 px at 9 → 10 (status §8e). Ruling B13 item 3 keeps this RED for slice 0b (the
per-water fold with a crossing byte; `docs/design/DEFERRED.md:10212-10217`). This design adds nothing for it.

---

## 4. THE CONSTANTS

| constant | value on the home planet | class | where it lives | source |
|---|---|---|---|---|
| k | ¼ (`>> 2`) | a stated choice (W6); the owner's word is still OWED | `SHORE_SHIFT`, `crates/recipe/src/height.rs:336` | W6 (`owner_decisions_2026-09-21_water.md:93-116`); `shore_rules.rs:172-176` |
| K | 4 (`<< 2`) | a stated choice (D5, 1/k); the word OWED | a new recipe constant | design v2 §15 Q4; `shore_rules.rs:176` |
| the band | 30 m | the owner's stated choice, GIVEN | ONE constant `BAND_M` in `vd-recipe`, which the host reads (one writer). Beside the kernel's `*` sums stand `const` asserts `3 × BAND_M < 128` and `1.25 × BAND_M < 128`, so a band of 43 m or more fails to compile. The S1 extreme-word test reads the constant | B13 item 1 (`believability.md:371-373`) |
| g_min | 1 m per node | the rows' own unit (whole metres a node) | a recipe constant | D4; `shore_rules.rs:177` |
| the mirror's metre | 1 m | the rows' own unit | the host | `shore_rules.rs:687-695` |
| the cap | 4, plus one slot for the fold | a stated choice (finding 23); never bound | the kernel's five slots | `shore_rules.rs:178`; census:114 |
| F = R₁ − 3 | 7 | R₁ from the node's size alone (§3.6); the 3 is typed (finding 7) — the owner's word OWED | `vd-terrain` | `main.rs:441-444`; W15.1 item 3 |
| R_P | 14 | the first rung whose cell is wider than a node (§3.6) | `vd-terrain` | `artifact.rs:511-520` |
| the distance's ½ | — | the block's own geometry: a line midway between two nodes | the host's table | `shore_rules.rs:964` |
| the distance's 3.5 | — | a STATED CHOICE (T9): the bench's word for "no other-side node in the block". It is NOT the block's reach — a found node can stand 5.16 nodes off — so it is not monotone with the found distances. On the home planet it saturates every octave's fade (the longest reach, the 25 km wave's, is 2.4 nodes) | the host's table | `shore_rules.rs:964` |
| the presence radius | 2 nodes | the stencil's own reach | the host | `shore_rules.rs:927-941` |
| the ties | the point side: the sea first, then the higher level, then the earlier slot; the fold: a tie wet, a tie a lake, the highest lake level | W15.1's stated rules and `block_water`'s | the recipe | `shore_rules.rs:1108-1136`; `artifact.rs:474-489`, `:1598-1623` |
| 15/8 | exact | the quintic fade's own peak slope | folded into `C` | `shore_rules.rs:1314-1315` |
| `C_smooth`, `C_ridged` (NEW) | 0.74–0.80 and 0.91–0.92 of the wavelength (census:66-75) | ONE CORRECTION PER OCTAVE KIND: the noise's realized reach over the spectrum's sine-law reach (`15/8 · λ/τ ≈ 0.298 λ`) — about 2.6 × for the smooth octaves and 3.1 × for the ridged ones (ARITH). Measured PER AXIS in FACE-CENTRE LATTICE METRES: the step between two rung-0 columns two cells apart over `2 · cell_m(0)` (`crates/bins/examples/shoreline_bench/main.rs:521-586`), at share one, over 2 048 seeded sites at least 1 024 cells from every face edge. The sample's own spread is about 1.6 % (ARITH, `1/√(2 · 2 048)`) | two code constants at 28 bits beside the test that re-measures them from the recipe's integer noise over the same seeded sample and pins them to the unit | Q2 |
| `q_k` (NEW) | 8 192 m ÷ reach: 0.41 for the 25 km wave, 215 for the 49 m wave | a CHARTER WORD BY LAW: `q_k = π·f_k / (2·C_kind·edge)` — the radius cancels; `f_k` is the octave's own frequency. Drawn ONCE per body at the body's draw through the fenced float `Gf` and rounded once into a 28-bit word, the door `slope_charter` uses for its reciprocal (`crates/terrain/src/artifact.rs:1943-1949`); π is the constant the fence names, as `crates/terrain/src/body.rs:78` names τ. It is the FACE-CENTRE value: the node's size is the arc at the face centre (`macro_lattice.rs:177-182`), and the bend stretches a cell away from it, so the reach in true metres varies over a face. The bench measured in the same face-centre metres, so the check agrees | a new `ShoreCharter` row; a SHORE-CHARTER PIN states the home body's words as literals and runs on every target (the `charter-pin` pattern, `justfile:806-807`) | `macro_lattice.rs:177-182`; `crates/recipe/src/height.rs:3-4` |
| `A_r` | the live amplitudes plus the bench's pull bound: the coarse octaves' relief plus the fine octaves' 1 532 m at rung 0 (`crates/terrain/src/body.rs:1033`); the exact word is UNMEASURED here | seed-derived, an integer word per rung | the `ShoreCharter` row | `crates/terrain/src/body.rs:1527-1529` |
| R₁ | 10 | derived from the node's size (§3.6) | `crates/terrain/src/artifact.rs:853-876` | — |
| the sea, the rows, the mask, the pyramid | solve products | shipped (approved rows) | the head, the tiles, the parts | — |
| the shore band, the lake levels, the band's digest (NEW) | solve products; the digest derived | shipped only on Q1's yes | a new bulk arm and head fields | Q1 |
| the lift (571.8 m) | — | NOT READ by form 2 (a reading) | nowhere in the build | census:65; `shore_rules.rs:1377-1401` |

No N2 word enters the realm's surface statement or the body's charter (SL10's twenty integers,
`docs/design/owner_decisions_2026-09-07_voxels.md:535`): `q_k` and `A_r` follow from the octave table and the lattice
that every host already draws from the seed and the charter, as `slope_ref` does (`crates/terrain/src/body.rs:
1003-1016`). Ruling T9 holds: the reach is not a physical fact but a statistic of the generator's own noise, stated with
its measurement; the band, k and K are named choices, each with its ruling or its owed word. The measurement also
CONTRADICTS the spectrum's own sine law, on which the spectrum's slope anchor and `slope_ref` stand; Q2 puts that to the
owner, and S4 registers it in DEFERRED.

---

## 5. THE CARD

- **The kernel compiles for the card.** Every new recipe function is WRITTEN OUT in the card's measured-safe shapes:
  - the ten held passes in a row, each hold a sign mask (§1.3 step 2);
  - the `dist` read as sixteen products and four row sums in a row (§1.3 step 5) — NOT `catmull_rom_16`, whose inner
    accumulator is read after the inner `while` ends (`crates/recipe/src/macro_field.rs:47-63`), the shape
    `crates/recipe/src/root.rs:27-36` records as a card fault;
  - the division's 33 steps and the root's 32 steps (§1.3 steps 3–4);
  - the φ loop in the octave loop's own shape: one index loop, `octave_term` inside it, additive accumulators, masks in
    the data, no value a branch assigns inside it (`crates/recipe/src/height.rs:191-203`;
    `crates/recipe/src/cell.rs:254-261`);
  - comparisons written out (`crates/recipe/src/gi.rs:56-78`);
  - wherever a loop carries the fold's highest lake level, the ADDITIVE form of a max,
    `best += if v > best { v − best } else { 0 }` — the form the carvers' hollow needed on the card, where
    `best = greater(best, v)` read back ZERO (`crates/recipe/src/cell.rs:254-261`).

  `just recipe-gpu` (`justfile:842-843`) must still build the module.
- **`catmull_rom_16` on the card is UNMEASURED.** It never ran there: the card's entry points are `relief_columns`,
  `cell_field`, `column_pass` on `FieldRead::none`, `node_pass` and two probes (`crates/recipe-gpu/src/lib.rs:41-246`),
  and none reads `Z`. The card's owed artifact path reads it, so the probe below measures it for the first time.
- **The card cannot BUILD an N2 chunk yet.** It binds no `Z`, no row, no side word and no mask
  (`crates/recipe-gpu/src/lib.rs:41-203`; `crates/terrain/src/gpu.rs:372-379`), and it takes no job with an artifact
  (`crates/client-render/src/terrain.rs:1343-1353`; `crates/client/src/chunks.rs:269-274`). Its column pass reads
  `FieldRead::none` (`crates/recipe/src/plan.rs:127-132`, 161-176) — the no-side rule of §1.5 step 8, unchanged.
- **What the card gets now: a PROBE.** A `shore_probe` entry point in `vd-recipe-gpu`, in the pattern of `isqrt_probe`
  (`crates/recipe-gpu/src/lib.rs:205-221`), runs the slot kernel, the side, the surface kernel and `catmull_rom_16` on
  stated WORDS: node words, distance words, presence bits, fractions, `B`, `R`, `W`, the charter's octave table, `q_k`
  and a DIRECTION. The φ loop runs the REAL octave loop from the table and the direction. It never takes stated octave
  terms, which would skip `octave_term` and not test the shipped loop's shape. The gpu-drift gate
  (`crates/bins/tests/gpu_no_drift.rs`) gains a leg: 2¹⁶ inputs, 0 differing output words. Its FAILING CONTROL
  compares the card's words with the CPU's words for inputs with one bit flipped, and must read differing words. The box
  sweep stays 5 979 of 5 979 (DOC, `docs/investigation/2026-09-24/shoreline_design.md:126`), because the card's column
  pass reads the no-side rule before and after S4, and `PlanCharter` does not change (the `ShoreCharter` is a separate
  row, so `PLAN_CHARTER_WORDS` and the size test stand, `crates/terrain/src/gpu.rs:56-57`, 718-726).
- **The extra input buffers, when the card takes artifact chunks** (owed; not in this build): the owed artifact path
  first (the `Z` stencil, the slope share, the side and water words); then one `ShoreCharter` row (17 words); a
  per-box table of the fine node cells under the box (about 10 × 10 cells at rung 9, ARITH) with their words; and the
  side decision per column, either from the host or from a sample pass that writes keys and a reduction that adds
  them.
- **THE COST — against today's COLUMN PASS alone.** The 11.5 ms p50 for a rung-9 chunk (census:17674) is the WHOLE
  `geometry_with`: the column pass, the cells, the mesh and the parents
  (`crates/bins/examples/shoreline_bench/cost.rs:65-68`; the census line says "parents-included"). "About 3 µs a
  column" is that whole build over the columns, so it hides how the column pass itself grows. On the CPU (ARITH,
  UNMEASURED):
  - **the node-cell words**: 2–3 µs a node cell. At rung 9 a node cell holds 256 columns (0.01 µs a column). FROM RUNG
    13 UP EVERY COLUMN IS ITS OWN NODE CELL and pays the whole 2–3 µs where its block holds a shore — more than today's
    far column pays in all. The uniform-sea rule (§1.5) cuts the open sea to its test (about 0.2 µs), and a column with
    no wet node in its block pays the scan of its 36 mask words.
  - **"near water" is most of the planet**: every column whose 6 × 6 block holds one sea node carries the sea as a
    candidate (`crates/bins/src/shore_rules.rs:774-830`); the home planet is 78 % sea; 64.5 % of the plain's rung-9
    chunks hold a wet node in their node block (census:17675).
  - **the slots**: about 0.2 µs per candidate (the passes, the gradient, the division, `dist`).
  - **the side**: the point side's compares; the fold 0.1–0.5 µs at rungs 8–9; one memo or pyramid read above.
  - **the φ loop, on the DRY path only**: five slots × about 8 ns × the rung's fine octaves (10 at rung 0, 2 at rung 9).
    A WET column and a column with no candidate pay today's loop.
  - Ruling F9's bounded ask shortens the fine ring by what a slower column pass costs
    (`owner_decisions_2026-09-12_frozen_patches.md:283-290`), so the cost is a gate: COST-N2INT at rungs 0, 9, 13 and 18
    for an OPEN-SEA, a COASTAL and an INLAND chunk, each against today's column pass alone (S0 on the what-if, S1 on
    the kernel), and COST-SHIPPED per rung with the unpaced frame cost in S4. The bench's f64 what-if for ALL rules
    together read 1.7–2.7 µs per column near water (census:17681-17686). On the card: UNMEASURED.

  **Example.** A hull cruises 150 km over the open ocean. At rungs 13–18 every column carries the sea as its candidate.
  Without the uniform-sea rule each would pay 2–3 µs of new words, more than today's far column pays in all; with it,
  each pays today's loop and about 0.2 µs.
- **The fallback** (ruling F8, `owner_decisions_2026-09-12_frozen_patches.md:143-147`): the CPU path of the same
  source. It is the only builder for a body with an artifact today, and it stays so after this build.

---

## 6. THE SLICES, in build order

Every slice ends with its numbers, the status file changed in the same change (B6 item 4,
`owner_decisions_2026-09-22_believability.md:113-118`), and HR5: `cargo llvm-cov clean --workspace` first, then
`just coverage-fast` (`justfile:30-41`) with 0 real misses on every line the slice touched in `vd-recipe`,
`vd-terrain`, `vd-client`, `vd-wire` and `vd-connection-plane` (Tier-A, `justfile:12`), and the `vd-terrain` baseline
no larger. No slice runs while another job runs on the machine.

### S0 — MEASURE FIRST (the bench; no world change; `vd-bins` only)
1. **The reach law.** A test measures `C_smooth` and `C_ridged` over the recipe's own integer noise (per axis,
   face-centre metres, 2 048 seeded sites, §4). The bench prints each octave's law reach beside its benched reach
   (census:66-75) AND beside the spectrum's sine-law reach and its stated slope (`crates/terrain/src/body.rs:50-66`). It
   runs N2 again with `reach=law`, `dist=table` and `outside=default`: the deciding number, the far swaps to 12 → 13,
   WALL, CHECK-DRAWN-IS-SIDE, G-SEA-OVER-DRY, RAISE-OVER-RAW and FINE-SLOPE-NEAR-SHORE must stay where N2 stood.
2. **The far data and the hole.** BAND-NODES, BAND-BYTES (the delta-coded list and a reach bitmap, both), LAKE-NODES,
   CORNER-MIXED-CELLS, FAR-FOLD-COST (µs per mixed cell). And EMPTY-GLOBE for every body of the home system that states
   water: its solve, its mixed node cells, its derivation at a two-worker share, and its band's parts at the gateway's
   pace (ARITH from those numbers, until S3 measures a login).
3. **The pocket's holds** (Q11, widened by review finding 27). Three named variants:
   - (i) a POCKET-word column exempt from the sea's side hold;
   - (ii) (i), and the D7 floor only where the SEA governs;
   - (iii) (i), and the D7 floor relaxed by the sea's presence, `− mul_shr(A_r, ONE − Ω_sea, 28)`, as the sea slot's own
     hold is relaxed (`crates/bins/src/shore_rules.rs:1284-1293`).

   Each is read with G-POCKET-HELD, G-RUNG-HOLD, WALL, CHECK-DRAWN-IS-SIDE and G-SEA-OVER-DRY at the second pocket, at
   every swap the bench reaches and on W4. POCKET-LAKES-NEAR-SEA counts the lakes under the sea's level within three
   nodes of a sea node; if one exists, it becomes a stand and is read the same way.
4. **THE ONE NAMING RULE** (Q7, review findings 5 and 22). The fold with W15.1's lake share on the samples and the
   highest lake level among the lake samples, at rungs 8–13: the deciding number, the far swaps 7 → 8 … 12 → 13 and
   the lakes' lines must stay where N2 stood. G-RUNG-LAKE-LOST/MIXED/SHIFTED at EVERY swap 8 → 9 … 17 → 18 on the one
   reader: today `lake_census` reads only 9 → 10, 14 → 15 and 15 → 16 (`crates/bins/examples/lake_census.rs:1628-1665`).
5. **The far windows W4** at rungs 13–18 on the three coasts, the second pocket's floor and the high lake:
   - SHORE-STEP 13 → 14 … 17 → 18 under the sample pyramid, beside today's rule;
   - beside each, WATER-STEP: the rung-(r+1) columns whose water or level differs from rung r's at the same place;
   - G-POCKET-HELD and G-RUNG-HOLD at every swap 13 → 14 … 17 → 18;
   - CHECK-SHORE-BAND on W4 too: the bench's own band, built from its rows, read at rungs 13–18 against the rows'
     read (the host's exhaustive check follows in S2).
6. **THE SEAM AND THE CORNER.** EDGE-COAST counts the node pairs where a coast crosses a cube edge and names one as a
   stand. CORNER-COAST finds any cube corner within three nodes of a coast. W-SEAM is a window on each of the two faces
   against the shared edge, not pulled inward (the bench pulls every window onto its face today,
   `crates/bins/examples/shoreline_bench/stands.rs:104-108`).
7. **The integer kernel's what-if cost** (re-run after S1): COST-N2INT at rungs 0, 9, 13 and 18 for an open-sea, a
   coastal and an inland chunk, each against today's column pass alone (§5).

GATES: the bench's own checks (CHECK-R0 176 of 176, CHECK-R0-F64 0.000 m). THE OWNER SEES: numbers, the same panels,
Q1 with its measured size, and Q3 with its measured seconds.

### S1 — THE INTEGER KERNEL (`vd-recipe`; new entries only; no world change; `GENERATOR_VERSION` stays 21)
Builds: `BAND_M` and its asserts; the derivative weights; the held passes, written out; the plain gradient
(`unsigned_abs` first); the written-out division (33 steps); the written-out `dist` read; the slot kernel (§1.3); the
point side and the fold's rule on the three words (§1.4); the surface kernel with its three paths, its one loop and the
uniform-sea closed form (§1.5); the `ShoreCharter` row (`q_k` through `Gf`); the `shore_probe` entry point.
GATES:
- recipe unit tests, each with a failing control:
  - the held read stays inside its four corners on 10⁶ random stencils (control: the plain read leaves the range);
  - the read is continuous across a node line (the last fraction of one cell against the first of the next, within the
    stated rounding);
  - the extreme words (±`BAND_M` alternating, at the worst fractions) wrap no pass (control: the same passes fed
    UNBANDED words of a 3 km steep coast wrap under `*`);
  - THE DIVISION equals `⌊a·2²⁸/b⌋` (computed in `u128` inside the test) on 10⁶ pairs spanning `g ∈ [2²⁸, 159·2²⁸]`
    and `|S| ∈ [0, 30·2²⁸]`, with the four corners of that box as fixed cases (control: the top-bit division differs
    for `g ≥ 16` m per node);
  - the ties of the point side and the fold;
  - the dry hold is continuous as `B` sweeps across `W`;
  - an empty slot leaves `φ` unchanged bit for bit;
  - the uniform-sea closed form equals the full kernel on 10⁶ deep-sea columns (control: a corner word at `W − 29 m`);
  - the default rule leaves a slot inert but for its level (control: a slot fed its rows' heights differs);
- CHECK-N2-KERNEL in the bench: the integer kernel on the bench's own inputs against the f64 reference with
  `reach=law`, `dist=table` and `outside=default`, every column of every window at rungs 0–13, the bands of §1.8;
- `gpu-drift` (`justfile:289-290`): 5 979 of 5 979 boxes; the probe's 0 differing words over 2¹⁶ inputs, with
  `catmull_rom_16` and the real octave loop; the probe's failing control reads differing words; `just recipe-gpu`
  builds;
- the SHORE-CHARTER PIN: the home body's `q_k` and `A_r` words as literals, on every target;
- `terrain-pin` (`justfile:797-800`), `terrain-link-scan`, `terrain-fence-control` (`justfile:808-823`): unmoved,
  because nothing calls the new entries;
- HR5 on `vd-recipe`: 0 misses.
THE OWNER SEES: numbers only; no picture moves.

### S2 — THE HOST (`vd-terrain`; derived and checked, read by nobody; no world change)
Builds: the node cell's words (§1.2) from any field that holds rows (the artifact, the tile cache, the golden rows); the
corner phantom's rule and the default rule (§1.2 items 1 and 9); the candidates and the slots per column; the point
side; the fold at rungs 8–9 per column; the node-cell records (the column kernel's own functions, the keys jointly per
sample) and the memo; the sample count pyramid (three words) beside the node counts (§3.3–3.4), fed by rows for now;
R₁, F and R_P from the node's size and the direct sum for a body with no pyramid (§3.6); the tile span rule over F
(§3.6); `body.shore_charter(rung)`; `shore_column`, the one reader (§1.7) — beside today's reader, called only by tests
and the bench; the band's membership walk over the full enumeration of §2.3, on the server's side.
GATES:
- `vd-terrain` tests, each with a failing control:
  - a uniform node cell draws one key on all 4 096 samples (control: a corner flipped makes it mixed);
  - a parent's counts are the sums, and its level the max, of its children at every level;
  - the 714-cell body: R₁ = 6, F = 3, R_P = 10, no pyramid, and the direct sum equals a brute-force fold (control: a
    sum that skips one node square differs); its samples cross a node cell, and the widened span holds them;
  - a halo column across a face seam reads the partner chunk's column bit for bit;
  - a golden-rows chunk and a whole-artifact chunk read the same words;
  - `field_column` at a column centre equals the chunk's column BIT FOR BIT AT RUNG 0, and within HALF A CELL OF THE
    RUNG above rung 0 (§1.7 states why it cannot be exact there; control: `field_column` on today's `shore()` exceeds
    the tolerance at the gentle coast);
- the bench's integer row N2i through `shore_column`: CHECK-N2-INT (§1.8) at rungs 0–13 on the stands AND on W-SEAM and
  the corner window (moved forward from S4: the first draft first crossed a seam after the band had shipped);
  CHECK-N2-RECORD-KEYS; N2i's own deciding number;
- **CHECK-SHORE-BAND, EXHAUSTIVE OVER THE GLOBE** (moved forward from S3). For every node cell of §2.3's enumeration —
  the 132 phantom cells included — and every water of its block: the 16 banded words, the 16 distance words and the 4
  presence bits built from the band (with the mask, the lake levels and the default rule) equal those built from the
  rows. These are integer compares over at most 8.9 million node cells, most with no water (ARITH about 1–2 s). The
  failing control: one band node removed at a cube-edge coast reads RED;
- **G-N2-PIN (NEW)**: a table of chunk digests built through `shore_column` on a stated golden patch of coastal rows —
  keys at rungs 0, 3, 7 and 9, and ONE CUBE-CORNER COAST KEY (the home planet's rows where CORNER-COAST found one, else a
  stated patch: the pin measures drift, not the look) — with their rows as text beside `golden_home_z.txt`. It runs in
  debug, release and native like `terrain-pin`, and on the other targets by `terrain-legs` (`justfile:816-818`). A
  DERIVED-PYRAMID LEG adds the digest of the records and of the pyramid's levels over one stated level-3 block of coast
  (64 node squares), recorded from the rows' path. Today no gate can see a shore: `terrain_pin` and `mesh_pin` build a
  body with no sea (`crates/terrain/src/home.rs:171-178`; `crates/terrain/src/body.rs:1113-1119`);
- `terrain-pin`, `mesh-pin`, `artifact-pin` (`justfile:828-829`), `charter-pin` (`justfile:806-807`), the identity:
  unmoved;
- HR5 on `vd-terrain`: 0 new misses on the touched lines.
THE OWNER SEES: numbers (CHECK-N2-INT, CHECK-SHORE-BAND); no picture moves.

### S3 — THE SHORE BAND ON THE WIRE (after the owner's yes to Q1; no world change)
Builds:
- the shard runs S2's band derivation (the membership walk of §2.3, ONE function) on its rows at boot, with the lake
  levels and the band's digest;
- `BulkMsg::ArtifactShore` and the head's fields — the parts, the node count, the digest (`PROTO_MINOR` 35); the shard's
  source ships the band after the mask (`crates/bins/src/artifact_source.rs:106-138`); the gateway caches it like the
  mask and counts it in its whole-check (`crates/connection-plane/src/gateway/artifact.rs:463-465`);
- the client's book holds the band, the lake levels, the memo and the per-face pyramid (§3.5); the receiver checks the
  assembled size AND the digest and refuses a mismatch by name; `whole()` waits for the band
  (`crates/client/src/artifact_book.rs:123`), so the client states `ArtifactHeld` only then;
- `masked_level` waits for the band and the lake levels, so `field_for` AND `field_at_rung` wait alike (`:175-221`);
- the derivation jobs on the workers' share, the eye's faces first;
- the lane's `far_epoch` stamp and its provisional re-ask (§3.5).

The wait for the PYRAMID switches on with its reader, in S4; S3 builds the derivation and measures it.
GATES:
- CHECK-SHORE-BAND ON THE WIRE: the band the receiver ASSEMBLED equals the shard's derivation (every entry and the
  digest), and far columns read through it equal the rows' read bit for bit on every coastal column of the bench's far
  windows and of W4 (the CHECK-CACHE-FIELD pattern, `crates/bins/examples/shoreline_bench/eval.rs:887-954`); a wrong
  size and a wrong digest are refused by name; a far chunk waits for a missing band part;
- THE WAITS, each with a failing control:
  - a span read before the band is PROVISIONAL, tested with the band landing after the levels (control: a span read on
    the level alone must not be final);
  - a rung-9 chunk built before the band is stamped provisional and built again when the band lands (control: with the
    stamp off, it stays);
- the pyramid derived from the band equals the rows' derivation on G-N2-PIN's leg;
- G-EMPTY-GLOBE: the band's bytes and parts, the derivation's time, and the time from login to the first far picture,
  per body with a sea, against the owner's number (Q3);
- a CONTROL picture on the stands equal to today's once the band has landed (nobody reads the band yet);
- HR5 on `vd-terrain`, `vd-client`, `vd-wire`, `vd-connection-plane`; the wire's own version tests.
THE OWNER SEES: the control picture, the band's measured cost and the hole's seconds.

### S4 — THE SWITCH (`GENERATOR_VERSION` 21 → 22, ONE change) AND THE FLIGHT
Builds:
- every reader switches at once — `ColumnRead::column` (the chunk, the halo, the span), `field_column` (the morph, the
  seam, the eye), the standing water, the beach — and the wait for the pyramid from R_P switches on (§3.5);
- the instruments move to N2 (`crates/bins/src/shore_geometry.rs:297-320`'s `base_agrees` becomes an N2 drift alarm;
  the bench's R0 row becomes the shipped N2; `shore_step`, `water_edge_step`, `lake_census`, `far_side_probe` read the
  one reader);
- the unit tests that state today's law are restated (`crates/recipe/src/height.rs:672`, 720, 765;
  `crates/terrain/src/height.rs:540`, 624, 716, 757, 813; `crates/terrain/src/chunk.rs:1371`, 1450, 1636;
  `crates/terrain/src/artifact.rs:2133`, 2255, 2316, 2463; `crates/terrain/src/position.rs:576`, 633, 714);
- **THE STATED-SIDE R0 LEAVES THE RECIPE.** `shore()` keeps only the ground's own sign — today's `SIDE_UNKNOWN` arm, the
  card's rule and the no-side host's — and `FieldRead` loses `side` (`crates/recipe/src/plan.rs:150-158`, `:214-247`;
  the stated-side arms, `crates/recipe/src/height.rs:417-427`). `column_water` keeps its own `Side` answer for D7 and the
  pocket word. So no host that holds an artifact can reach R0. The bench keeps today's stated-side law as its own R0
  control in `vd-bins`. The kernel's doc stops telling a shard author to build a chunk through it
  (`crates/recipe/src/plan.rs:206-212`);
- **DEFERRED**, beside the collider consumer (`docs/design/DEFERRED.md:7763`): the collider slice reads `shore_column`
  at rung 0, and its gate G-COLLIDE-N2 holds a shard-built rung-0 column equal to the client's chunk column bit for bit
  on G-N2-PIN's keys.

**Example.** A hull lands on the steep coast. Without the removal, a collider written on `column_surface_at` would stop
it on R0's ground while the pilot sees N2's, up to 935 m apart (RAISE-OVER-RAW p99, status §8e). After it, the shard has
only `shore_column` to call, and the hull stops on the ground she sees.

THE PINS:
- `GENERATOR_VERSION` 22 with its doc paragraph (`crates/terrain/src/tag.rs:3-7`, 233); the version's assert and
  `DECLARED_PIN` (`tag.rs:287-290`, 322). Every realm re-solves at its first boot, because the declared tag keys the
  stored artifact (`crates/bins/src/artifact_store.rs:191-203`; `crates/bins/src/artifact_boot.rs:65-80`); the
  store stamps and the inter-shard ALPN fold the version (`crates/bins/src/lib.rs:1002-1011`;
  `crates/io-prod/src/trust.rs:41-43`).
- **The identity's measured word MOVES (PREDICTED).** Of the six rung-0 golden keys (`crates/terrain/src/digest.rs:
  85-94`), the golden rows (`crates/terrain/tests/golden_home_z.txt`: 207 rows, 93 sea, 2 lake) put water in four
  node blocks: PosX (300, 700) stands on a coast (25 sea nodes and 5 dry on its face) and NegY (2000, 2000) beside a
  lake (2 lake nodes), so N2 changes their columns; PosY (5, 5) and PosZ (999, 1) stand over the sea, where N2 changes a
  column only in water shallower than about 30 m (in deep water both rules keep the raw surface, §1.5 steps 6 and 8).
  The two top keys read no mask and keep the ground's own sign (§1.5 step 8). So `just golden-z-record`
  (`justfile:833-834`, a whole home solve) re-records `HOME_IDENTITY_MEASURED` (`crates/terrain/src/home.rs:31`), and
  its three asserts change together (`crates/terrain/src/tag.rs:316`; `crates/bins/tests/home_artifact_pin.rs:110-114`;
  `crates/bins/tests/home_body_pin.rs:209-213`). The golden rows' text does not change.
- `terrain-pin` and `mesh-pin` UNMOVED (no sea, so no shore: `crates/recipe/src/height.rs:413-415`); `artifact-pin`'s
  digest and sea UNMOVED (the solve does not change); `charter-pin` moves only its identity assert; `gpu-drift`
  5 979 of 5 979 and the probe; `ARTIFACT_VERSION` 9 unmoved.
- G-N2-PIN re-recorded on the shipped path and extended by FAR KEYS at rungs 10 and 13. A rung-18 chunk covers a whole
  face — 38 of its 62 columns stand on the face (ARITH) — so no small patch holds one; the derived-pyramid leg covers
  rungs 14–18. Each far key carries its own patch: the level-1 words (heights and water words), the mask under D7's
  sixteen-node level stencil, and the band and the lake levels for its node cells.

THE PYRAMID'S WAITS (gates, each with a failing control): a rung-14+ chunk and its parent mesh wait for the pyramid of
every face they read (control: a build allowed on a partial pyramid differs from the build on the whole one); a rung-13
chunk built before the pyramid is stamped provisional and built again when its faces land (control: with the stamp off,
it stays).
THE MEASUREMENTS: the census on the SHIPPED path — CHECK-N2-SHIPPED (the bench's integer column equals `column_field`
bit for bit: the old CHECK-R0's role), CHECK-N2-INT, CHECK-CACHE-FIELD with the band, the deciding number, every far
swap, WALL, CHECK-DRAWN-IS-SIDE, G-SEA-OVER-DRY, G-POCKET-HELD, RAISE-OVER-RAW, FINE-SLOPE-NEAR-SHORE, the lakes'
lines, G-RUNG-LAKE-LOST/MIXED/SHIFTED and WATER-STEP at every swap, a NEW G-LAKE-DRAWN (the drawn lake share, for B9
item 1, `believability.md:265-267`), SPAN-MISS and HIDE-MISS (§8); `rung_swap` (the handover's height step);
`water_edge_step` at every swap; **G-SEAM-MORPH** on EDGE-COAST at every swap 0 → 1 … 17 → 18 — each seam vertex's
target against the coarser chunk's own mesh along the same radial, within ONE cell of the coarser rung (the ladder's own
tolerance at a handover), with its failing control (`field_column` on today's `shore()` must read RED); G-EMPTY-GLOBE
at a login over the moon; COST-SHIPPED per rung; the frame cost with `VD_CLIENT_UNPACED=1` on the coast leg.
THE FLIGHT (B13 item 3): every stand against BOTH reference pictures — the Crimson Desert vista and the valley-country
frame (status §9, `terrain_believability_status.md:1000-1013`) — the three stands, the reference stand and its skyline
numbers, ONE CUBE-EDGE COAST STAND (EDGE-COAST), the checklist, each picture captioned "no far shadow" (B6 item 2 and
B6.5, `believability.md:100-106`, 140-143); `just flights` (`justfile:320`) green; the pictures refrozen only on the
owner's look (`justfile:292-296`).
THE DOCS in the same change: §9 below; the status file's §8 row 4 and §8e; DEFERRED D-TERRAIN-9's shoreline entries;
DEFERRED: the spectrum's sine law against the measured reach (Q2), the collider entry, the corner phantom (Q12).
THE OWNER SEES: the pictures and the census; G-LOOK for step 4 is asked.

### S5 — AFTER THE OWNER'S LOOK
The beach (design v2 D9) and the far-swap gates restated as the census's lake lines; slice 0b (narrow water) when the
owner asks for it; the card's artifact path when the owner reopens it; the corner blend.

---

## 7. THE OPEN QUESTIONS FOR THE OWNER

**ANSWERED (ruling B14, 2026-09-26, on the S0 numbers of §10):** Q1 YES (the shore band ships); Q2 the LAW, and the
spectrum's conflict registered; Q3 SEND THE COUNTS TOO (0.636 MB; the owner prefers no empty far globe — W15.1 item 5
amended for the samples' counts; S3 ships the counts and the level word beside the band, and the client derives no sample
pyramid); Q11 the pocket exemption KEPT as a stated rule (variants (i) and (ii); variant (iii) refuted). STILL OPEN: Q4–Q10,
Q12, and Q13–Q14 from S0 item 5; S0 items 4–6 are measured (§10), so Q7 and Q12 have their numbers; Q1's and Q3's sizes
were corrected after the ruling (0.81 MB and 1.10 MB, §10), and the owner's re-confirmation is asked.

**ANSWERED (ruling B15, 2026-09-26):** Q7 — CURE THE FAR LAKE FLICKER FIRST: the bench measures a fold that does not
flicker on nested footprints BEFORE S1, and the one naming rule is not built until then; Q13 and Q14 — BOTH CURES JOIN N2
(the bench's defaults since: `n2_wet_strict`, `n2_lone_holds`, with B14's `n2_reach_law` and `n2_pocket = 2`); Q1 and Q3
stand at the corrected sizes; Q4, Q5, Q6, Q8, Q9, Q10 and Q12 accepted as recommended.

| # | question | recommended answer | its price |
|---|---|---|---|
| Q1 | **SL6: the shore band.** Ship each lake node's level and the height of every fine node within 30 m of a water it is read for, once, with the head, and the band's digest in the head (§2.5)? | YES (F3). It is the only exact road: every rung reads the same words, so the swaps the owner saw measured are the swaps that ship. | UNMEASURED bytes (ARITH 0.19 MB of lake levels and 1.3–5.3 MB of band; S0 measures first); `PROTO_MINOR` 35; one bulk arm and three head fields; the far view waits for the band as it waits for the mask, then for the derived pyramid (Q3). Refusing it costs a seam at the 704 km swap (F1, about 8 px ARITH) or a different N2 at every rung (F2). |
| Q2 | **The reach law, and what it says about the spectrum** (T9). `reach = C_kind × wavelength`, with one measured correction per octave kind, stated as constants? | YES, and REGISTER THE CONFLICT. Both hosts draw `q_k` from the octave table; nothing crosses. | THE CONFLICT: the spectrum states each fine octave as a sine, `a = s·λ/τ`, anchored at `TALUS_RMS × TAN_REPOSE = 0.70` (`crates/terrain/src/body.rs:50-66`, 79-85), so its reach would be `15/8·λ/τ ≈ 0.30` of the wave. The noise realizes 0.74–0.92 of the wave (census:66-75): about 2.6 × (smooth) and 3.1 × (ridged) the sine's. The fine octaves' realized RMS slope at share one is 0.139 per axis (census:76) against the anchor's 0.70, and the 25 km octave realizes 0.0128 per axis against its stated 0.125 (`body.rs:63-65`). The angle-of-repose anchor and `slope_ref` (`body.rs:1016`, the divisor of the roughness share) stand on the refuted law. So `C_kind` is stated as the realized-over-sine ratio, which a later spectrum fix reads too, and DEFERRED registers the conflict with these ratios (S4). The fades move by a few per cent against the benched per-octave statistic (up to about 4 % from a common constant, ARITH); S0 re-runs the census under the law. The other lawful roads: the same statistic in integers on both hosts at the body's draw; or the sine law itself, about 0.30 of the wave — a third of the measured reach, a new look, benched again, on a law the noise does not obey. |
| Q3 | **The far fold's cost, and the empty globe.** Derive the sample count pyramid on the client (W15.1 item 5 kept)? And what is THE OWNER'S NUMBER for G-EMPTY-GLOBE, the seconds a body with a sea may show no far globe after its head lands? | YES, if S0 measures the hole within the owner's number for every body with a sea; else ship each mixed node square's three words. | Deriving: an EMPTY GLOBE per body with a sea for the derivation's time (ARITH 2–15 s for a hemisphere on a two-worker share, UNMEASURED), after the band's transfer. Meanwhile NOTHING stands at the top rung, today and under N2; the owed W18 item (a far body drawn from its coarsest level the moment it lands) cannot hold while its pyramid derives. Shipping: about 1.2–3 MB more (ARITH, 6 B a mixed node square) and W15.1 item 5 amended; the hole shrinks to the transfer. |
| Q4 | **The rulings N2 amends** (v2 finding 13, `shoreline_design.md:995`): W10 items 1–2 — between node centres the side is the SIGN of the held node-scale field, which equals the row's word at every node centre; W15.1 item 1 — a far cell's side is the wet fraction of its rung-F SAMPLES, not of its fine nodes (the node counts stay for the pocket word and D7, §3.4); W15.1 item 3 — rungs 8–13 fold samples where today they read the nearest node's word. | ACCEPT as stated. | Three items of two rulings are re-worded; no new data from them alone. |
| Q5 | **k = ¼ and K = 4** (W6's quarter is still owed, `owner_decisions_2026-09-21_water.md:103-116`). | KEEP both as benched. | None new: every N2 number stands on them. |
| Q6 | **F = R₁ − 3**: the 3 is typed (finding 7); R₁ now comes from the node's size alone, so F exists on every body (§3.6). | KEEP. | A typed offset that W15.1 item 3 frowns on; its reason is measured (the 9 → 10 swap needs samples an eighth of R₁'s cell). |
| Q7 | **Rungs 14–18 have no reference, and ONE naming rule from rung 8 up.** One fold to the top through the sample count pyramid; the water named by W15.1's lake share on the samples, at the highest lake level among the lake samples, at EVERY folded rung (it replaces the bench's "most samples" at rungs 8–13); and the default words for a named water outside the column's block, on both hosts? | YES, after S0 benches the rule at 7 → 8 … 12 → 13 and W4 reads 13 → 14 … 17 → 18. | A change to the benched N2 at rungs 8–13, which S0 must show holds N2's numbers. Where two lakes share a cell, the lower one's water steps up to the higher level at the swap where they merge (`block_water`'s price, counted at every swap). The known lake losses at 14 → 15 and 15 → 16 (`DEFERRED.md:10097-10105`) are measured again. If S0 finds the rule worse, the exact alternative is a per-water count list carried up the pyramid, at a memory cost S0 prints. |
| Q8 | **No side data** (the golden top level, the card, a body with no artifact): keep the ground's own sign, as S4's only other shore rule in the recipe? | YES. | None: the identity's top keys and gpu-drift do not move. |
| Q9 | **The beach under N2**: the coast bit and `s ≥` the governing water (the sea with no candidate)? | YES, as the port of today's rule; the full law (D9) waits for 8e. | None visible: no pixel shows the beach before 8e. |
| Q10 | **The switch order**: one switch at every rung after S3, or the near rungs first with a stated seam at the 9 → 10 swap until S3? | ONE switch (design v2 §9 split slices 3a and 3b for this reason, `shoreline_design.md:704-705`). | The flight waits for Q1's answer and S3. |
| Q11 | **The pocket, and the pocket lake, beside the sea** (widened by review finding 27). (a) N2's side hold stands a pocket's floor over the sea where the sea's presence is whole: G-POCKET-HELD reads 175 columns at the second pocket against R0's 16 (census:14174, 14291; band 0, ruling B7). (b) The D7 floor (`crates/bins/src/shore_rules.rs:1290-1292`) has no presence relaxation and reads the GOVERNING water's field, which can be a lake's. A LAKE-word column that N2 calls dry takes the land rule's sea as D7 (`eval.rs:393-398`; `artifact.rs:1240-1243`), so beside a lake under the sea's level near the ocean its dry shore rises to the sea's level while the water stands at the lake's — a wall; today a lake-word column is never lifted over the sea. And D7's stencil is the field level's 4 × 4 (`artifact.rs:1280-1311`): it widens at 9 → 10 and at 14 → 15 … 17 → 18, so the lifted set changes there. (At rungs 0–9 a sea node in D7's stencil always stands in all four corners' presence squares, so there the floor adds nothing where the sea governs — ARITH from the stencils.) Exempt a POCKET-word column from the sea's side hold, and take one of the two D7 variants? | YES to the exemption; the D7 variant that S0's numbers favour (S0 item 3). | A small change to the benched N2 at pocket floors and pocket-lake shores only; the census runs again. |
| Q13 | **The wet tie** (S0 item 5, §10). From rung 14 up a column's centre is its node cell's middle, where a straight coast's clamped words read `S = 0` exactly; the blend then puts the surface ON the water's level and the sheet draws a side-wet column dry. A wet column stands at least one length unit under its water (`n2_wet_strict`)? | YES. | None measured: 2 of 16 804 near lines move (one a cure); the far side mismatches at rungs 14–16 fall to 0–5. |
| Q14 | **The lone column** (S0 item 5, §10). A far column with no water in its own block, whose fold says dry, stands over its standing water by `k × band`, the default rule's own slot (`n2_lone_holds`)? | YES — it is N2's own rule, "the side decides over the standing water", on the one path that missed it. | Some larger floor jumps at the rung 15–18 swaps (the largest 6.3 km on a 262 km column), the size of today's rule's own (6.7 km). |
| Q12 | **The corner phantom** (review finding 28). A corner phantom reads its key face's clamped cell, so the three chunks that meet at a cube corner hold three phantoms — today in `Z`, under N2 also in the side. Keep that (the stated rule of §1.2 item 1) until the owed corner blend (`CORNER_BLEND`, `artifact.rs:109-111`)? | YES. G-N2-PIN's corner key pins it on every target, and S2 counts the corners whose three phantoms disagree (CORNER-PHANTOM-SIDES). | A tooth of water at a coastal cube corner, at most one far column wide, until the blend. |

---

## 8. THE RISK REGISTER

| # | what can go wrong | how it shows | the gate that catches it |
|---|---|---|---|
| 1 | The far rungs read words the client lacks | a far chunk waits for ever (a hole), or it builds from a tile an earlier flight left (a picture that depends on history) | `field_for` picks by rung (§2.2); `masked_level` waits for the band and the lake levels (§3.5); CHECK-SHORE-BAND (S2 exhaustive, S3 on the wire) and CHECK-CACHE-FIELD with the band (S4): 0 differences |
| 2 | The owner refuses Q1 | no exact far N2 | S0 benches F2 as a what-if against the deciding number before the owner answers; the switch waits (Q10) |
| 3 | The far fold costs too much | the far globe stands empty for seconds after login; far coastal chunks slow | S0's FAR-FOLD-COST and EMPTY-GLOBE; G-EMPTY-GLOBE against the owner's number (S3, S4); Q3's fallback |
| 4 | The band is bigger than ARITH | a long login | S0's BAND-BYTES before Q1 |
| 5 | The reach law moves the look | the deciding number or the texture lines change | S0's census under `reach=law` |
| 6 | A product or the division wraps | a wild surface on a steep coast | the extreme-word tests and the division's box test (S1); the `BAND_M` asserts; CHECK-N2-INT-VALUE (1 mm) |
| 7 | A decision flips at a tie | a column wet on one host path and dry on the other | CHECK-N2-INT-DECIDE (the near-tie set per sample and per candidate, relative and absolute; its failing control) |
| 8 | The card mis-spells the new loop, the division or `catmull_rom_16` | a card box differs | the written-out shapes; the `shore_probe` leg of gpu-drift (0 words; its failing control); the box sweep 5 979 of 5 979 |
| 9 | Two hosts drift on another chip | the identity refuses a client; the far view differs between chips | the identity (two golden keys stand on a shore, S4); G-N2-PIN and its derived-pyramid leg on `terrain-legs`; the SHORE-CHARTER PIN |
| 10 | A seam or a corner reads two blocks for one column | a crack or a step on a cube edge | S2: the halo test, CHECK-N2-INT on W-SEAM and the corner window, CHECK-SHORE-BAND over the phantom cells; S4: G-SEAM-MORPH (the bench's windows never crossed a seam before, `crates/bins/src/shore_geometry.rs:253-256`) |
| 11 | The morph or the eye reads another rule | a vertex morphs toward a shore the chunk does not draw (a W16 breach) | the one reader and its shared records (§1.7); `field_column` bit for bit at rung 0 and within half a cell above it (S2); G-SEAM-MORPH, `water_edge_step` and `rung_swap` (S4) |
| 12 | The span or the hide bound no longer holds | a hole in a ladder slice; a sheet triangle cut that shows | SPAN-MISS and HIDE-MISS in S4 (band 0): no built column outside the span its 25 samples stated (`crates/terrain/src/digest.rs:349-398`); no cut sheet triangle whose ground comes within reach of its water (`crates/client/src/chunks.rs:1340-1349`) |
| 13 | A coastal chunk costs too much | builds lag the ring; F9's bounded ask shortens the fine ring | COST-N2INT against the column pass alone (S0, S1); the uniform-sea rule; COST-SHIPPED per rung and the unpaced frame cost (S4) |
| 14 | The instruments go void | a check that cannot fail | S4 rewrites `base_agrees` and CHECK-R0 as CHECK-N2-SHIPPED, each with a failing control |
| 15 | Coverage, the float fence or the division lint breaks | a red gate | HR5 per slice; `terrain-link-scan`; `terrain-fence-control`; the recipe's `deny(float_arithmetic, integer_division)` (`crates/recipe/src/lib.rs:33-35`) |
| 16 | The inherited REDs get worse | G-POCKET-HELD 175 (census:14291), G-SEA-OVER-DRY 11 columns (e.g. census:12441), CHECK-DRAWN-IS-SIDE 7 ties (census:3384), the walls at the second pocket (68 pairs) and on the belt at rung 3 (16 %), the steep coast's p99 from 9 → 10 to 12 → 13 | the S4 census states each against N2's benched value: no line worse; Q11 for the pocket |
| 17 | The wet hold shrinks to 7.5 m (k × the band) where today it is a quarter of the ground's depth | a fine crest in deep water stands as a shoal 7.5 m under the sea (read in the code, `crates/bins/src/shore_rules.rs:1245-1251`; no census line) | a new reading SHOAL-HELD in S4; no picture shows it until 8o's clarity |
| 18 | Two lakes share a cell at a folded rung | the lower lake's water steps up to the higher level at the swap where they merge | ONE rule at every folded rung (§1.4); G-RUNG-LAKE-SHIFTED and WATER-STEP at every swap (S0, S4) |
| 19 | The far caches take too much memory | the client's memory grows with the coast | S3 measures the memo and the pyramid (ARITH tens of MB) |
| 20 | R2 takes `PROTO_MINOR` 35 first | a version clash | whichever lands first takes 35; the other 36 |
| 21 | The re-record chain is long | a slow switch | one bump, one `golden-z-record` run, the three identity asserts in one change (S4) |
| 22 | The drawn lake share changes before the owner judges the ceiling | B9 item 1 judged on stale numbers | G-LAKE-DRAWN in S4 (the band draws a lake deeper than 30 m by its node's footprint: 16 of N1's 20 at rung 9 on the plain, B13 item 1) |
| 23 | A chunk or a parent mesh is built on a half-derived pyramid | a far coast that no epoch ever rebuilds | READY MEANS FINAL; the pyramid wait from R_P, in `ParentMesh::build` too (§3.5, S4) |
| 24 | A morph target is read before the band or the pyramid | seam vertices that keep a target toward a shore no rung draws | the `far_epoch` stamp and the provisional re-ask (S3's gate and its control) |
| 25 | The collider reads R0 | a hull stops on ground up to 935 m from the drawn ground on the steep coast | S4 removes the stated-side arms; the DEFERRED entry and G-COLLIDE-N2 |
| 26 | A wrong band height, or a straggler part of an older head | a far coast drawn off its rows while every size check is green | the band's digest in the head; the receiver's refusal by name (S3) |
| 27 | A far pocket reads as LAND | the second pocket rises to a mesa at rung 14 | the node counts kept (§3.4); G-POCKET-HELD and G-RUNG-HOLD at 13 → 14 … 17 → 18 (S0) |
| 28 | The three corner phantoms disagree on the side | a tooth of water at a coastal cube corner | Q12; G-N2-PIN's corner key; CORNER-PHANTOM-SIDES (S2) |

---

## 9. The docs that disagree with the code (fix them in the change that touches them)

- `BulkMsg::ArtifactCoast`'s doc says one bit a node, 1.1 MB and 34 parts (`crates/wire/src/channels.rs:461-470`); the
  mask holds TWO bits a node (`crates/terrain/src/artifact.rs:220-239`), 2.2 MB in 68 parts
  (`owner_decisions_2026-09-21_water.md:1096-1097`). Fix in S3.
- `ArtifactCache::coast`'s doc says one bit per fine node (`crates/client/src/artifact_book.rs:88-92`). Fix in S3.
- `artifact_source`'s module doc says the pyramid ships "coarsest level LAST" (`crates/bins/src/artifact_source.rs:
  6-7`); the code ships the coarsest level FIRST (`:106-110`, `.rev()`). Fix in S3.
- `tile_rung`'s doc says "THE FINEST RUNG THAT READS THE ROWS" (`crates/terrain/src/artifact.rs:936-938`); the code
  answers the COARSEST rung that reads them (rung 9 on the home planet, `:939-945`). Fix in S2.
- `FieldRead::side`'s doc says the side is the column's own FINE node's word (`crates/recipe/src/plan.rs:150-158`);
  since W15 a far rung reads the footprint's majority. S4 removes the field.
- `column_surface_at`'s doc tells a shard author to build a chunk through it: "The shard builds a chunk over the belt …
  hands the very same three words to this kernel" (`crates/recipe/src/plan.rs:206-212`). After S4 a host that holds an
  artifact builds through `shore_column`. Fix in S4.
- The status file (`docs/investigation/2026-09-24/terrain_believability_status.md`) names B13's SL6 question as
  narrow water only; this design adds Q1. Fix when the owner answers Q1.

**Example.** After S4 a pilot starts on the beach of the gentle coast and climbs to 1 400 km. Rungs 0 to 9 draw the
shore from the tiles under her, rungs 10 to 13 from the shore band and the shared records, rungs 14 and up from the
sample count pyramid — one set of words and one fold rule at every rung. The gateway refused no client, because the two
hosts folded the same golden coast and lake into one identity word.

---

## REVIEW LEDGER (revision 2)

Each finding of the design review, by its number, with its outcome. APPLIED = the fix is in this file (the code agreed).
REFUTED = the code disagrees, with the reason. OPEN = the fix is in this file, and the owner decides a named part.

| # | finding (short) | outcome | where, and why |
|---|---|---|---|
| 1 | The derived pyramid has no home, no state and no wait; the globe is a hole | APPLIED; the owner's number OPEN (Q3) | §3.5 THE BOOK'S NEW STATE, THE WAITS, THE HOLE: the pyramid is a per-face book state beside `counts`, derived by six jobs on the workers' share (never inline like `CoastCounts::of`, `artifact_book.rs:613-625`), the epoch and `far_epoch` bump when a face lands, `field_for` and `ParentMesh::build` wait from R_P, and `field_at_rung`'s readers get `None` per column on a face with no pyramid. One part is done differently from the finding's text: a rung-13 chunk does NOT wait for its parents, because rung 14 waits too and a wait would leave rung 13's ring empty; it builds, is stamped provisional and is built again when the pyramid lands (§3.5 THE WAITS item 4). The gate "a rung-14+ chunk waits for the pyramid" stands in S4 (THE PYRAMID'S WAITS), where the wait switches on with its reader; S3 builds and measures the derivation. S0 item 2 (EMPTY-GLOBE per body with a sea); S3/S4 G-EMPTY-GLOBE; Q3 states the seconds against 1.2–3 MB and says that nothing stands at the top rung meanwhile |
| 2 | `field_at_rung`, the span, the morph and the eye read far fields that do not wait for the band; the fallbacks read any level | APPLIED | §3.5 THE WAITS item 1 (`masked_level`, which both readers share, waits for the band and the lake levels); §1.5 step 8 and §1.7 (`shore_column` answers `None` on a far field with no band, never R0); §1.7 THE FALLBACKS STAY PROVISIONAL; §3.3 THE MEMO (the records live in the book, shared by pointer, so the morph and the chunk read one fold); S3's gate "a span read before the band is provisional" |
| 3 | After S4 the recipe keeps a callable R0, and the collider has no gate | APPLIED | §6 S4: `shore()` keeps only the ground's own sign and `FieldRead` loses `side`; the kernel's doc is corrected (§9); DEFERRED entry beside `DEFERRED.md:7763` with gate G-COLLIDE-N2; §2.2; risk 25 |
| 4 | The fold's default words for a water outside the block are not the reference's; no gate sees rungs 14–18 | APPLIED | §1.2 item 9 (THE DEFAULT RULE inside the one reader, on both hosts, and in the reference as `outside=default`); §1.8 items 1–2; S0 item 5 (CHECK-SHORE-BAND on W4); S2 G-N2-PIN's derived-pyramid leg on `terrain-pin` and `terrain-legs`; S4 far keys with the level-1 words and the mask under D7's level stencil. One correction: a rung-18 chunk covers a WHOLE face (38 of its 62 columns stand on it), not two thirds; the conclusion stands |
| 5 | The water's name and level change source at 13 → 14; the fold's tie is not total | APPLIED (with 22) | §1.4 item 2 (one rule at every folded rung; every tie decided); §3.4 (the sentence corrected); S0 item 5 (WATER-STEP beside SHORE-STEP) |
| 6 | The band's membership walk has no stated range; corner phantoms name node cells outside it | APPLIED, numbers corrected | §2.3 "Read for, and the walk's range" (the phantom cells enumerated, 132 on the home planet); S2 exhaustive CHECK-SHORE-BAND with a cube-edge failing control. Correction: a phantom's block reaches at most FOUR nodes inward on the partner face (rung 15), not eighteen; from rung 16 up every node of its block is off the face on both axes and `node_at` names the corner node (`artifact.rs:1838-1840`; ARITH in §2.3). The defect (cells no regular walk names at rungs 13–15) stands |
| 7 | The band sits outside every digest; a size check sees no wrong height | APPLIED | §2.5 (c) and "Why the digest": the band's digest in the head, the receiver refuses a mismatch by name; the version rule stated; S3's gate reworded as "the band the receiver assembled equals the shard's derivation, and far columns read through it equal the rows' read" |
| 8 | CHECK-N2-INT-WORDS at band 0 cannot hold for the irrational distance words | APPLIED (with 17), count corrected | §1.2 item 6 and §1.8 items 1–2: the reference reads the host's 28-bit table (`dist=table`); a one-time test proves the floor equal for every `d²`. Correction: `d²` takes FOURTEEN values (9 not squares), not 32 |
| 9 | F is undefined on a body with no pyramid level | APPLIED | §3.6: R₁ from `level_for`'s test without its `levels == 0` answer, F = R₁ − 3, R_P from the node's size; the direct sum on a body with no pyramid; S2 test on the 5 km body (R₁ = 6, F = 3, R_P = 10); §4 |
| 10 | The records and the rung-7 column compute one sample key on two paths | APPLIED | §3.3 (the records call the column kernel's own functions); §1.8 item 6 CHECK-N2-RECORD-KEYS at band 0 with no exemption and a failing control |
| 11 | The division's form overflows; the magnitude bound; the signed shift | APPLIED in substance; one claim REFUTED | §1.3 step 4 (33 written-out steps with a SHIFTED REMAINDER under `2g < 2^36.4`, no cap because `d < 32` nodes; the top-bit form refused); §1.3 step 3 (`unsigned_abs` before the shift); §1.1 (stated bound `g ≤ 159 m/node`, products `< 2^68.3`); S1's division test over `g ∈ [2²⁸, 159·2²⁸]`, `|S| ∈ [0, 30·2²⁸]` with the four corners. REFUTED: the magnitude does NOT reach 159 m/node — its exact maximum over the ±30 m vertices is 112.5 (ARITH, exhaustive over the 2¹⁶ sign vertices: where one axis reaches 112.5 the other's sum is nought). The design states the looser, provable 159 anyway |
| 12 | The near-tie set has holes and cannot go red on the fold | APPLIED | §1.8 item 3: a near tie per sample and per candidate (any `|S_b|`), a relative-plus-absolute tie on `d`, the fold compared sample by sample, and a failing control (one flipped sample key) |
| 13 | "Near water" is most of the planet; a wet column pays the φ loop; far columns pay the node-cell words; the cost base is the whole build | APPLIED | §1.5 (the side decided before the loop; three paths; a WET column takes today's loop; THE UNIFORM-SEA RULE with its bit-for-bit test and control); §5 THE COST restated against the column pass alone, far rungs included; S0 item 7 and S1: COST-N2INT at rungs 0, 9, 13, 18 for open-sea, coastal and inland chunks |
| 14 | The reach law refutes the spectrum's sine law; the design shows it only as a look; C's measure is not stated | APPLIED | §4 (`C_kind` as one realized-over-sine correction per kind; per axis; face-centre lattice metres; 2 048 seeded sites; the sample's spread; the pin); Q2 restated with the ratios and the refuted anchor and `slope_ref`; DEFERRED entry in S4 |
| 15 | The `*` bounds depend on a band that lives in another crate | APPLIED | §4 `BAND_M` in `vd-recipe` (one writer) with `const` asserts `3 × BAND_M < 128` and `1.25 × BAND_M < 128`; §1.3 steps 2–3; S1's extreme-word test reads the constant |
| 16 | `dist ≤ 4.4` is wrong; 3.5 is called the block's reach | APPLIED | §1.1 (a word ≤ 5.157 nodes, `dist ≤ 8.06` nodes; `d` has no cap, so no cap can bind `D`); §4 (3.5 a stated choice, not the block's reach, not monotone with found distances) |
| 17 | The distance words are not whole numbers; libm's `hypot` is not stated to round correctly | APPLIED (with 8) | §1.8 item 1 (`dist=table`, as `fade_f64` reads the integer fade; the one-time floor test over every `d²`) |
| 18 | `q_k` needs π and a door; the law holds only at the face centre | APPLIED | §4 `q_k`: drawn through `Gf` and rounded once, as `slope_charter`'s reciprocal is; π the fence's constant; the face-centre value stated; S1's SHORE-CHARTER PIN on every target |
| 19 | `catmull_rom_16` and a branch-assigned held value on the card; the probe skips the real loop | APPLIED | §1.3 steps 2 and 5 (ten passes and the `dist` read written out, holds as sign masks); §5 (`catmull_rom_16` named UNMEASURED on the card, the probe its first measurement; the probe runs the real octave loop from a table and a direction; its failing control) |
| 20 | Per-water records cannot give the fold's keys; the top count is 2²⁴ | APPLIED | §3.3 (keys computed jointly per sample; the record's three words per group); §3.4 (the top word 4⁶ × 4 096 = 2²⁴ on the home planet) |
| 21 | `field_column` cannot equal the chunk's column bit for bit above rung 0 | APPLIED; the alternative REFUTED | §1.7 and S2: bit for bit at rung 0, within half a cell of the rung above it, with a failing control. The other fix (read the node cell at the rung's own centre) is refused: it makes the morph target a step at every cell border, which a morph cannot carry |
| 22 | The fold's rule changes at 13 → 14 (the level from the block word; the name from the lake share) | APPLIED; OPEN for the owner (Q7) | §1.4 item 2 (ONE rule at every folded rung: the lake share on samples, the highest lake level among lake samples, carried by sums and a max); §3.4 (the three failures stated, the sentence corrected); S0 item 4 (the bench's swaps 7 → 8 … 12 → 13 again; G-RUNG-LAKE-* at every swap); S0 item 5 (the water-level step on W4). It changes the benched N2 at rungs 8–13, so the owner rules on it in Q7 |
| 23 | `field_column` is the morph target of every face-seam vertex and has no source for the fold | APPLIED | §1.7 (the seam vertices, the shade, the source and the cost stated); §3.3 THE MEMO (records shared per body by pointer); S4 G-SEAM-MORPH at every swap with its failing control; the flight's cube-edge coast stand; S0 item 6 |
| 24 | `field_at_rung` shares `masked_level`; the lane never rebuilds a target read before the band | APPLIED | §3.5 THE WAITS item 1 (`masked_level` waits for the band and the lake levels, so both readers wait) and item 4 (the `far_epoch` stamp beside the digest, the provisional re-ask; it also heals today's rung-9 chunks built before the mask); S3's gate with its control |
| 25 | The sample pyramid drops the pocket count | APPLIED | §3.4 THE NODE COUNTS STAY (all three words beside the sample pyramid, for the side word, the pocket and D7); S0 item 5 (the second pocket's floor on W4; G-POCKET-HELD and G-RUNG-HOLD at 13 → 14 … 17 → 18); risk 27 |
| 26 | Every exactness line runs on windows that never touch a seam | APPLIED | S2 CHECK-SHORE-BAND exhaustive over the globe's node cells, the phantom cells included, with a failing control at a cube-edge coast; W-SEAM and the corner window moved forward to S2, so CHECK-N2-INT crosses a seam |
| 27 | The D7 floor has no presence relaxation and lifts a pocket lake's dry shore to the sea's level | APPLIED as a measurement; OPEN for the owner (Q11) | S0 item 3 benches the two D7 variants (the floor only where the sea governs; the floor relaxed by the sea's presence) with the pocket exemption, on the second pocket and any pocket lake near the sea, at every swap; Q11 widened. Noted from the stencils: at rungs 0–9 a sea node in D7's stencil stands in all four corners' presence squares, so the floor there adds nothing where the sea governs |
| 28 | The three corner phantoms at a cube corner can disagree on the side | APPLIED; OPEN for the owner (Q12) | §1.2 item 1 (THE STATED RULE: the phantom's cell is its key face's clamped cell, its side the point side of that cell, no fold); S2 G-N2-PIN's cube-corner coast key; CORNER-PHANTOM-SIDES; Q12 until the owed corner blend |
| 29 | The far globe's wait is a hole and a pop, unbounded by any gate | APPLIED; the owner's number OPEN (Q3) | §3.5 THE HOLE (G-EMPTY-GLOBE against the owner's number; S0 measures first; the shipped counts become the default past the number; the eye's faces first; never a build on a partial pyramid); S3/S4 gates; Q3 |

---

## 10. S0 RESULTS — items 1–6 MEASURED (2026-09-26; the bench, no world change)

The instruments: `crates/bins/examples/n2_far_data.rs` (new) and the shoreline bench's choices `n2_reach_law` and
`n2_pocket`. The censuses: `n2_s0_far_data_census.txt`, `n2_s0_reach_law_census.txt`, `n2_s0_pocket_v{1,2,3}_census.txt`
in this folder. The artifact: the pinned generator-21 home artifact, READ from a scratch file (digest `1eb53c78…d2a0e4`).

**Item 2, THE FAR DATA (Q1, Q3) — the design's arithmetic was pessimistic:**

| line | measured | the design's ARITH |
|---|---|---|
| BAND-NODES | 202 431 of 8 871 936 nodes (2.28 %) | a twentieth to a fifth |
| BAND-BYTES | **0.588 MB** as a delta-coded list (0.201 MB of deltas + 0.386 MB of heights); 1.444 MB as a reach bitmap | 1.3–5.3 MB |
| LAKE-NODES | 93 225 nodes, **0.178 MB** | 0.19 MB |
| Q1's datum in all | **about 0.77 MB** (the coast mask that ships today: 2.2 MB) | |
| CORNER-MIXED-CELLS | 111 089 of 8 886 534 node cells (1.25 %) | |
| FAR-FOLD-COST | 49–54 µs a mixed cell with the words once a cell (the §3.3 record); 3.3–3.6 ms when the words are read again for every sample. Both paths key all 1 638 400 samples alike (749 460 wet each) | |
| EMPTY-GLOBE (the derived pyramid) | **about 3 s on two workers** for the whole home planet (5–6 s on one core; ARITH from the two measured numbers) | 2–15 s |
| FOLD-SHIPPED-BYTES (Q3's alternative) | **0.636 MB** (three 2-byte counts a mixed cell) | 1.2–3 MB |

EMPTY-GLOBE for the home system's other bodies with water is UNMEASURED (each needs its own solve).

**Item 1, THE REACH LAW (Q2) — the law holds N2's numbers:** measured over UNIT octaves of the recipe's own noise
(1 000 m wave, 2 048 seeded sites × 4 seeds, the slope per axis over two rung-0 cells): **C_smooth = 0.7723,
C_ridged = 0.9107**. The law's reach against the per-octave statistic, octave by octave: −3.1 % (the 25 km smooth
wave) to +3.7 % (the 195 m smooth wave); the ridged octaves within ±1.5 %. The spectrum's sine law gives 0.298 of the
wave — a third of what the noise realizes (the conflict of Q2). The whole bench with N2 on the law: of 2 243 N2 lines
2 000 are equal and 243 move, **none changes its verdict**; the deciding number 1.459 px / p99 0.355 px (the statistic:
1.398 / 0.354) — it still passes; the largest step change 0.06 px (the gentle coast at 6 → 7); G-POCKET-HELD at the
second pocket 175 → 141 columns.

**Item 3, THE POCKET (Q11):**
- POCKET-LAKES-NEAR-SEA: **0** lake nodes under the sea's level within three nodes of a sea node on the home planet, so the
  wall the review feared (a lake's dry shore lifted to the sea's level) does not occur here.
- Variants (i) and (ii) change NO line on the second pocket, the steep coast or the gentle coast (identical censuses):
  the pocket's held columns do not come from the sea's side hold.
- Variant (iii) (the standing water's floor relaxed by the sea's presence) is **REFUTED**: at rungs 10–12 on the steep
  coast it draws 90 / 22 / 9 side-dry columns wet and puts the sea over 84 / 20 / 1 dry columns — the far fields' D7
  stencil reaches past the sea's presence, and the floor is needed there in full.
- WHERE N2's 141 held pocket columns come from: the band's own pull (POCKET-FLOOR-MOVED: N2 moves the second pocket's floor
  by up to 3 318 m at rung 7, N1 up to 1 069 m, today's rule 55 m) — near a pocket's lakes the banded side field stands at
  the lake's level ± 30 m and the node-scale blend pulls a deep floor toward it; and the filled troughs
  (RAISE-OVER-RAW p99 134 m). Both are prices ruling B13 accepted (items 1 and 2).

**TWO CORRECTIONS to item 2 (found by S0 item 5, 2026-09-26; both sizes grow, no verdict of the checks changes):**
- **THE BAND'S RULE HAD AN OFF-BY-ONE.** The first count joined a node wet for `b` only for `W − 30 < z < W − 1`, so a wet
  node at `W − 1` (its word `W − 1`, not the edge `W − 30`) was left out. The band now has ONE definition,
  `shore_rules::band_needs` / `shore_band` (a wet node joins for `z > W − 30`; any other node for `W − 30 < z < W + 30`),
  held equal to the mirror and the clamp by a unit test over every height from `W − 100` to `W + 100`. MEASURED
  (`n2_s0_far_data_v2_census.txt`): **217 742 band nodes (2.45 %), 0.631 MB** as a delta-coded list (the first count:
  202 431, 0.588 MB; 15 311 wet nodes were missing). Q1's datum in all: **about 0.81 MB** (was 0.77 MB). The bench's
  failing control proves the missing nodes mattered: with the first rule N2 draws another far column in 1 466–9 158 of
  1 397 760 W4 columns per stand (CHECK-SHORE-BAND-CONTROL).
- **THE SHIPPED COUNTS' UNIT IS THE NODE SQUARE, NOT THE NODE CELL.** A rung-13 column is a node square: it holds a
  quarter of each of four node cells (§3.4). A square needs its three words shipped where any of those four cells is
  mixed. MEASURED: **192 271 mixed node squares (2.17 %), 1.100 MB** (the first count, 111 089 mixed cells × 6 B =
  0.636 MB, named the wrong unit). With the head the planet then ships about 0.81 + 1.10 = **1.91 MB** beside the 2.2 MB
  mask (ruling B14's example said 0.77 + 0.64 MB).

**Item 4, THE ONE NAMING RULE (Q7) — it changes nothing at rungs 8–13:** the whole bench twice with the reach law and
pocket variant (ii), once with the record's fold (the most samples) and once with the one rule (W15.1's lake share on the
samples, the highest lake level among the lake samples): **0 of 16 748 census lines differ** (`b14_fold0` against
`b14_fold1`, scratch censuses; the deciding number, every far swap 7 → 8 … 12 → 13, every lake line equal). The rule is
`shore_rules::one_rule_counts`, which the per-sample fold and the count pyramid both call.

The direct count agrees: FOLD-RULES-DIFFER, the two rules run side by side on N2's own samples of every folded column,
reads **0 of 10 524 872** folded columns (rungs 8–13, every window of every stand).

**Item 5, THE FAR WINDOWS W4 (Q7, and the far half of Q1 and Q3).** The bench's new `pyramid.rs` keys each node cell of a
window once (one sample where the four corners stand on one side of every water, all 4 096 where the cell is mixed), keeps
three words per QUARTER cell (a rung-12 column), and a far column sums the quarters under it and reads the one rule
(`one_rule_counts`). W4 (`far_window.rs`) is 1 024 node squares a side at rung 13 (about 8 300 km), rungs 13–18, today's
rule beside N2. Eight stands give four distinct windows (the belt, the plain, the reference and the gentle coast share
one on face −Z; the steep coast and the second pocket share one on face −Y). The censuses: `n2_s0_w4_census.txt` (the
benched N2) and `n2_s0_w4_cures_census.txt` (with the two cures below).

THE CHECKS — every one green on every window, each with a failing control:

| check | what it holds | measured |
|---|---|---|
| CHECK-PYRAMID-UNIFORM | a uniform cell's one key is every sample's key (the maximum principle, §3.2) | 0 samples differ, 9 502–11 266 uniform cells keyed whole per window |
| CHECK-PYRAMID-FOLD | the pyramid's summed words = the column pass's own keys | 0 of 64 columns at rung 13 and 0 of 16 at rung 14 per window (524 288 samples through `Ctx::eval`) |
| CHECK-PYRAMID-BAND | the pyramid keyed over a band-holding host's nodes = over the rows | 0 of 4 194 304 quarters per window |
| CHECK-SHORE-BAND | N2 over the band-held nodes = N2 over the rows, bit for bit | 0 columns at every rung 13–18 of every window |
| CHECK-SHORE-BAND-CONTROL | the band under S0's first (off-by-one) rule must differ | 1 466–10 476 of 1 397 760 columns differ: the check sees a wrong band |
| MIXED-NODE-SQUARES | the shipped unit | 15 563–70 027 of 1 048 576 squares per window |

TWO FAR DEFECTS FOUND IN N2, AND THEIR MEASURED CURES (two named choices, both OFF in the benched N2 until the owner rules):
- **THE WET TIE (`n2_wet_strict`).** From rung 14 up a column's centre is its node cell's MIDDLE (`t = ½`), where a straight
  coast's clamped words are symmetric, so `S = 0` EXACTLY; then `d = 0`, `β = Ω`, the blend puts the surface exactly on
  the water's level, and the sheet's own rule (`s < l`) draws the column dry against its side. Example (measured): a sea
  column at rung 14 whose ground stands 1 067 m under the sea is drawn as dry land AT the sea's level. At rungs 0–13 the
  centre never stands at `t = ½` on the home planet, so the tie cannot occur there. CURE: a wet column stands at least one
  length unit under its water. MEASURED: CHECK-DRAWN-IS-SIDE at rung 14 falls 242–461 → 0 per window; at rungs 15–16 to 0–5.
- **THE LONE COLUMN (`n2_lone_holds`).** A far column with no water in its own 6 × 6 block whose fold says dry falls back
  on the land rule, which lets its ground stand under the sea that D7 reads: the sea is drawn over a column its side calls
  dry (10–31 columns at rungs 17–18 per window). CURE: such a column stands over its standing water by `k × band`
  (7.5 m), the default rule's own slot. MEASURED: CHECK-DRAWN-IS-SIDE 0–1 column at every far rung of every window;
  WATER-STEP 0 at every far swap but one (1 footprint). ITS PRICE: some larger floor jumps at the rung 15–18 swaps (the
  largest 6 312 m at 17 → 18 on a 262 km column), the same size as today's rule's own (6 736 m).

THE FAR LINES, N2 (both cures ON) AGAINST TODAY'S RULE, at the swaps 13 → 14 … 17 → 18:
- **WATER-STEP** (one-water footprints whose coarser column draws another water, or water from nothing): N2 **0** at every
  swap of every window but one footprint; today's rule 10 / 295 / 8 / 385 at 13 → 14 and up to 25 above.
- **AREA-NEST** (the drawn wet area, coarser against finer): at the swaps 13 → 14 … 15 → 16 N2 within 0.34 %, today's
  within 0.90 %; at 16 → 17 and 17 → 18 both within 1.34 % (N2) and 1.03 % (today's). THE
  LAKES: N2 +1.8 / −6.9 / +1.8 / −4.1 % on the steep coast's window (today's +7.6 / −13.0 / −1.8 / +0.7 %) and +0.7 / −5.0 /
  +2.2 / −4.9 % on the high lake's (today's +8.4 / −11.4 / −1.5 / −2.7 %). Where lakes are few and small (the gentle coast,
  the first pocket) BOTH rules lose them at the far swaps (−20 % … −100 %): the narrow water of B13 item 4.
- **G-SEA-OVER-DRY** at rung 13: N2 45 / 708 / 73 / 1 069 columns, today's 98 / 1 570 / 142 / 2 395 — about half. Rungs
  14–18: alike (0–203).
- **G-POCKET-HELD**: 0 at every far rung of every window, both rules.
- **G-RUNG-HOLD** (dry pairs under the sea's level lifted at one rung and not the other by more than the extractor step):
  RED for both. At 13 → 14 N2 has more jumps (785 against 647 on the gentle coast's window; 304 against 219 on the steep
  coast's) with a larger largest jump (437 m against 306 m); above it the two rules are alike (the largest 6 312 m against
  6 736 m at 17 → 18).
- **SHORE-STEP** on far transects (256 per water kind, 1 049 km a side, matched only inside the transect): RED for both
  rules alike — p99 2.6–7.7 px, the largest ~8 px, which is the match's own reach (four coarser cells). The far coasts
  change their topology from rung to rung (the narrow water of B13 item 4); the step's median stays 0.28–0.82 px.
- **THE LAKES' FLICKER — N2 IS WORSE HERE.** LAKE-FLICKER at rung 14: N2 260 of 693 lake bodies on the steep coast's
  window and 531 of 1 211 on the high lake's, against today's 53 of 714 and 87 of 1 116. FOLD-RETURN: 463 of 1 659 against
  182 of 1 751, and 890 of 2 888 against 343 of 2 770. The bodies are ONE-COLUMN FRAGMENTS at a big lake's ragged edge,
  and the cause is the majority on nested footprints. Example (measured): a fragment of the 5 882 m lake is 64 % wet in its
  16 km column, 40 % wet in the 32 km column over it (so dry), and 59 % wet in the 65 km column over that (a lake again).
  Today's rule draws that lake at all three rungs from the level's own lake word. A one-column fragment at the 14 → 15
  swap is one or two pixels: the narrow water of B13 item 4 (slice 0b), now measured at the far rungs.
- **THE POCKET VARIANTS ON W4**: (i) and (ii) differ in 33 of 1 149 far lines, by at most one column each — ruling B14's
  variant (ii) holds at the far rungs too.

THE NEAR WINDOWS UNDER BOTH CURES: the whole bench with both ON (`n2_s0_w4_cures_census.txt`) against both OFF
(`n2_s0_w4_census.txt`): **2 of 16 804** lines of W0–W3 move — CHECK-DRAWN-IS-SIDE on the steep coast's W2 at rung 8
goes RED → GREEN (the 7 ties risk 16 lists: the wet tie cures them too) and FINE-RMS-NEAR-SHORE there reads 282.1 m for
282.0 m. The default rule (`n2_default_rule`, §1.2 item 9, ON in both) changes none of the 16 748 lines the B14 run
printed (it fires only where a far fold names a water outside the column's block).

**Item 6, THE SEAM AND THE CORNER (Q12)** (`n2_s0_far_data_v2_census.txt`, `n2_s0_seam_census.txt`; both cures ON):
- **EDGE-COAST: 223 node pairs** straddle a cube edge with another water on each side (`shore_rules::edge_coast`); the
  busiest stretch is face −Y's `v−` edge at node 1 097, 21 pairs in 32 nodes (its partner: face −Z's `u−` edge).
- **CORNER-COAST: ONE cube corner is coastal** — its three face corners PosX (0, 0), NegY (1 215, 0) and NegZ (0, 0) each
  hold a coast within three nodes. So Q12's price (a tooth of water at a coastal cube corner, until `CORNER_BLEND`) is
  real on the home planet, at one place.
- **W-SEAM** (the bench's new `seam.rs`: two windows flush against that edge, one on each face, rungs 3–8 and 7–13). Each
  edge column is paired with its partner across the edge by the cube net's own crossing rule (`vd_seed::seam::across`);
  the pairing is checked by the solved ground itself, which reads the same on both faces (6 974.4 m on both).
  - **N2 IS SEAMLESS THERE:** at rungs 3–11 it draws the same water on both faces in EVERY pair (0 of 1 024 … 0 of 64) and
    the step across the edge is the step inward (rung 7: p99 28.0 m across, 26.4 m inward). At rungs 12–13 a column is
    half a node or a whole node square, each centred on its own face's edge node, so a pair compares two nodes and the
    differences are the real lake shore that runs along this edge (1 of 32 pairs at rung 12; at rung 13 13 of 16 for both
    rules). The seam windows show NO red line for N2.
  - **TODAY'S RULE DRAWS ITS SHORE ON THE SEAM.** Its side is the nearest node's word, and beside the edge each face's
    nearest node is its own edge node — land on −Y, lake on −Z. So it draws a lake at 6 968 m on one face and dry land at
    7 300 m on the other: every pair of 1 024 differs at rungs 3–8, with a step of about 290 m (the largest 392 m) against
    about 1 m inward, and 81 % of pairs differ at rungs 7–10. The seam windows show 21 red lines, all today's rule's (its
    walls: 1 637 of 2 240 shore pairs on −Y at rung 7).

**S0 IS MEASURED** except item 7 (the integer kernel's cost, after S1).

**B15 ITEM 1 — THE FAR LAKE FLICKER, THREE CURES MEASURED** (`n2_s0_fold_kernels_w4_census.txt` holds the far window
under each kernel; `n2_s0_fold_tent_census.txt` and `n2_s0_fold_lone_census.txt` the whole bench under the two cures;
the choice `n2_fold_kernel`, the rule `shore_rules::fold_by_kernel`, both fold paths). THE CAUSE: a majority on each
column's own footprint is additive across rungs (a parent's words are its four children's sums), so a footprint whose
finer columns all agree never changes (WATER-STEP 0), but a ragged edge's share crosses one half up and down as the
footprints nest — a one-column fragment shows, hides and shows again. A kernel that removes the fragment before it can
flicker must read beyond the column, and then the sums no longer nest: some one-water footprints change at a swap. THE
NUMBERS (the steep coast's window / the high lake's; N2 with both far cures):

| fold | LAKE-FLICKER at rung 14 | FOLD-RETURN | WATER-STEP at 13 → 14 | lakes' area at 15 → 16 and 16 → 17 | the near windows |
|---|---|---|---|---|---|
| the box (the column's own footprint) | 260 / 531 | 463 of 1 659 / 890 of 2 888 | 0 / 0 | +1.8, −4.1 % / +2.2, −4.9 % | the benched N2 |
| **the tent** (the column and its eight neighbours, 1-2-1 per axis) | 28 / 49 | 38 of 486 / 68 of 764 | 187 / 316 (mostly vanish) | −8.0, −22.8 % / −8.1, −18.7 % | 256 N2 lines move; the deciding number's p99 0.355 → 0.430 px (its maximum 1.459 px stays); 3 lines RED → GREEN; some lake steps grow (the steep coast's lakes at 10 → 11: 0.70 → 2.13 px), some shrink; the plain's small lakes gone at rung 13 (12 → 0) |
| the tent where the column's own samples disagree | 28 / 49 | 38 of 519 / 68 of 811 | 183 / 311 | as the tent | not run (the far lines equal the tent's) |
| **no lone column** (the box, and a wet column with no wet edge neighbour at its rung is dry) | **15 / 34** | 28 of 470 / 62 of 710 | 276 / 468 (mostly appear) | +1.2, −3.9 % / +2.2, −3.5 % | **32 lines move, all at rungs 12–13** (one-column lakes, 8 km, not drawn at rung 13: the plain's 12 bodies → 2); 1 line RED → GREEN; the deciding number unchanged |
| today's rule | 53 / 87 | 182 of 1 751 / 343 of 2 770 | 295 / 385 | −1.8, +0.7 % / −1.5, −2.7 % | — |

READING: both cures bring the far flicker under today's rule. The TENT is a smoothing: it removes a fragment by reading
its neighbours, so its water vanishes at a swap where every finer column was wet (a clean single vanish) and the lakes
shrink at the coarsest swaps (−23 % at 16 → 17), and it moves 256 near lines. NO LONE COLUMN keeps the box's own area and
the near windows almost untouched, flickers least, and pays with water that appears at a swap where its finer columns were
lone and so dry — as often as today's rule (276 / 468 against 295 / 385). The owner chooses (B15 item 1).
