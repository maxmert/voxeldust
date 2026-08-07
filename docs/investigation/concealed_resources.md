> ### ⚠ INVESTIGATION BASE — NOT A DECISION RECORD
>
> **This document is input to an investigation, not the output of one.** It is analysis produced to
> explore a problem space, and it is deliberately more confident in tone than its status warrants —
> that was useful for finding defects and is misleading for planning.
>
> **Nothing here is committed.** Every ruling, recommendation, number and "settled" verdict is a
> *proposal to be re-validated by the pre-implementation investigation for its phase*, including the
> owner rulings recorded at the top of `block_system_design.md`, which record the owner's direction at
> the time rather than a frozen commitment.
>
> **The binding specs are elsewhere:** `docs/design/PLAN.md`, `docs/design/roadmap.json`,
> `docs/design/integration.json`, `docs/design/DEFERRED.md` and the hardened subsystem designs in
> `docs/design/`. Where this document and a binding spec disagree, **the binding spec wins** until an
> investigation says otherwise.
>
> **What this document IS good for:** the prior art it collected, the arithmetic it did, the failure
> modes it found, the one-way doors it named, and the questions it framed. Reuse those. Re-derive the
> conclusions.
>
> See `docs/investigation/README.md`.

# Concealed resource placement — can it be hidden, and what it costs

**Status:** ruling, for owner decision. Adjudicates `[USER DECISION 3-G]` (§3.8 of
`docs/investigation/block_system_design.md`) and the owner's follow-up question about hiding the algorithm.
**Date:** 2026-08-03.
**Scope:** whether ore placement can be hidden from a client that holds the terrain generator; what the
achievable form of concealment is; what it costs against every contract this project has already made.

---

## The owner's question

> *"Ideally players should not be able to compute it — is there a way to make it absolutely hidden? For
> example the algorithm is executing on the server, but then somehow 'transferred' for the execution to
> the client? Or any other possible mechanics?"*

---

## The direct answer

**No — nothing you put inside the client hides anything, and the reason is stronger than "obfuscation is
weak".** Transferring the algorithm to the client for execution does not fail because someone might read
the code. It fails because the thing you want to keep secret is not the code — it is the *answer the code
produces*. To draw a world the client must be able to work out that answer. Anybody who can make it work
out one answer can make it work out all of them: run it over every chunk in the galaxy and write the
results to a file. That attacker never has to read a single line, never has to find a key, and never has
to understand anything. A mathematically perfect hiding scheme would not stop him. This is why no shipped
game has ever hidden a world generator inside its own client and kept it hidden, and why the two most
serious attempts at exactly this problem — the branch of cryptography built to hide a key inside code the
attacker runs, and secure enclaves on consumer hardware — are respectively broken by automated tools and
no longer sold on the machines your players own.

**But "absolutely hidden" is achievable, and the achievable version is better than the one you asked
for.** The one thing that cannot be recovered is a fact the client was never told. That is not a hope, it
is arithmetic: a program can only work out what follows from the bits it was given. So the question stops
being "how do we hide it" and becomes "what does the client actually need to be told" — and here this
game is in an unusually lucky position. **Ore is inside opaque rock. The player cannot see it. So the
client never needs to be told where it is at all.** Withholding it removes no pixel, causes no gap, no
stutter and no waiting; it fits the seamless rule for free. Minecraft can never fix this properly because
the client must hold a block in order to draw it, which is why its servers resort to *inventing fake ore*
in the data they send. We simply never send the real thing.

**Your instinct — hide the key, not the code — is right, and it points one step further than you stated.**
Do not transfer the algorithm for execution. Keep the key *and* the working on the server, and transfer
only the answer. Then look at how much answer actually has to travel, and the answer turns out to be
**none at all**, if the strategic materials are treated as *what a rock assays to when you break it*
rather than as *a coloured block you can see*. That single reframe removes the whole problem. It is also
more realistic, not less: real ore is rock. You do not see veins of pure metal in a cliff; you see rock
that tests positive, and prospecting is the profession of finding out which rock.

**So the recommended design is this.** Common materials — iron, copper, coal, the ordinary building stock
— stay exactly as they are: made from the seed, computed in the client, visible in a cliff face, costing
nothing. The strategic tier the economy's prices actually depend on is never placed in the world the
client builds at all. It exists only as a server-held table saying which rock, when broken, yields it.
Players find it three ways: by breaking rock and seeing what comes out; by using a survey instrument that
returns a deliberately blurry direction-and-depth reading; and by learning to read the visible geology,
because the public generator deliberately places *visible clues* — discoloured, altered rock — that make
a deposit a few times more likely underneath. The clue is a hint, never the answer, and how strong a hint
it is is a number you choose rather than something that leaks by accident.

**What this costs: essentially nothing.** Not one extra byte crosses the network. Not one extra byte is
stored. The rule that terrain never streams survives with no exception at all — which is *better* than
the current §3.8 recommendation, which was going to need one. The detail ladder, the edit pyramid, the
collision system, the mesher and the save format are untouched, because the hidden data never enters the
world grid in the first place. The determinism test that proves every player's terrain is byte-identical
keeps covering one hundred per cent of what a player can see, unchanged. The cryptography needed is one
standard keyed hash, and the library for it is **already in this project's dependency list** for identity
work, so it is not even a new dependency. The real cost is a survey instrument and an assay loop — that
is *gameplay to build*, not architecture to bend, and the honest warning is that shipping the
concealment without shipping the discovery would be the worst outcome available: it would remove an
activity and give nothing back.

**Two things must be settled before the first world is saved, and they are cheap now and expensive
later.** First, *which* materials are in the hidden tier — because moving a material from visible to
hidden after players have mined means moving ore out from under their tunnels. Second, and this is the
one most likely to be under-weighted: **hiding changes how ore is PLACED, not just how it is sent.** If
the hidden deposits are positioned the geologically believable way — at the contact between two rock
layers, along faults, near caves — then anyone can compute them from the visible terrain, which every
player already has exactly. My arithmetic below shows a believable placement rule can leave the position
determined to within a factor of a thousand *regardless of how good the key is*. The perverse
consequence is that the more realistic the placement, the less is hidden. The fix is to place the hidden
deposits from a coarse depth band and nothing finer, and to buy back the realism deliberately through the
visible clue material — where the strength of the hint is a declared number with a test on it.

**And one honest caveat.** No scheme, at any budget, keeps a deposit secret from an organised group that
simply plays. Fifty accounts mining and pooling results will map their own territory, and that is
legitimate. What concealment actually buys is that such a map is *an asset somebody earned* instead of a
file that exists the day the client is extracted — and that the frontier, the part nobody has visited,
stays dark. That is worth having. It is not the same as secrecy, and the design should say so out loud.

---

## The threat model

An ore map is *data*, not an exploit. It is extracted once and copied for free, forever, because the
galaxy is deterministic and the map never expires. So the usual question — "how much effort will each
class of player spend?" — is the wrong one. The security of any client-side scheme equals the effort of
the single most capable interested person on Earth, divided across the whole player base.

| Tier | Who | Effort to extract | Effort to *use* someone else's extraction |
|---|---|---|---|
| T0 | Ordinary player | — | zero: downloads a web tool |
| T1 | Modder with a disassembler | an evening to a weekend | zero |
| T2 | Organised guild, 50 accounts | weeks, and can crowdsource legitimately | zero |
| T3 | Commercial cheat/RMT operation | person-months, funded, precedented | zero |

Two project-specific facts make T3 certain to exist rather than hypothetical. The currency is real
(local and global, with banks), which makes an ore map a *saleable financial product* — the equivalent of
insider information in a commodity market. And the precedent is exact and already ran: Minecraft's
Chunkbase maps an entire world offline from a seed, is free, and takes roughly 18 million visits a month
from people who could not write it.

The corollary that decides everything: **effort estimates per adversary are irrelevant. The only question
is whether the information is present in the client at all.**

---

## Why shipping the algorithm to the client hides nothing

### The one-line argument

The ore map is the graph of the generation function over the chunk key space. The client must be able to
evaluate that function or it cannot draw the ore. An adversary who can read no code and recover no key
can still **call the function in a loop and record the outputs**. Every scheme in the family — obfuscated
native code, WebAssembly, encrypted bytecode with a virtual machine, a server-supplied program, a
server-supplied shader, per-session streamed code — is defeated by that single move, before any question
of reverse-engineering arises. Attacking a generator is fundamentally easier than reverse-engineering it,
because the attacker does not need to understand it, only to *harness* it: set a breakpoint on the output
buffer, or pattern-scan for the entry point and call it. Harnessing is hours of work no matter how the
code is protected.

The theory says the same thing and it is settled. General virtual-black-box obfuscation — "the program
reveals nothing but its input/output behaviour" — was proved impossible (Barak, Goldreich, Impagliazzo,
Rudich, Sahai, Vadhan; CRYPTO 2001 / JACM 2012). Indistinguishability obfuscation does exist from
well-founded assumptions (Jain–Lin–Sahai, STOC 2021) but is both astronomically impractical and
semantically useless here: it guarantees only that two *functionally equivalent* programs look alike, and
explicitly does not hide what the program computes — which is the only thing we want hidden.

### Form by form, fairly

| Form | Why it fails | Worse than status quo? |
|---|---|---|
| Obfuscated native code | Hours to harness; the ore is in a CPU buffer anyway because collision needs it. Commercial protectors (VMProtect, Themida) have a public body of automated devirtualisation work against them (Rolles, WOOT 2009, and successors) | no, but no gain |
| WebAssembly | **Strictly easier** to reverse than native: structured control flow, typed locals, no register allocation; standard tools decompile it readably. The sandbox protects the host from the module, never the module from the host | yes |
| Encrypted bytecode + a VM in the client | The interpreter is in the client, so one print in the dispatch loop yields a full instruction trace — and a trace of a straight-line numeric program *is* the program. **You ship the adversary a free tracer** | yes, badly |
| Server-supplied shader | Minutes with a frame-capture tool, which captures the shader and every buffer; and the result must reach the CPU for collision regardless | yes |
| Code streamed per session | Raises the cost per rotation, not per extraction — extraction is automated once. The cheat market ships updates within 24 h of a game patch. Costs a build pipeline, a new determinism axis (each rotation is a binary whose byte-identity must be re-pinned) and a seamless-law hazard if a rotation lands mid-flight | yes |
| White-box cryptography | This is the discipline built precisely to hide a key inside code the adversary runs, and its public record is total defeat. Chow et al.'s 2002 white-box AES fell to the BGE attack (2004) at ≈2³⁰ work; then **generic automated** attacks arrived — Differential Computation Analysis (Bos, Hubain, Michiels, Teuwen; CHES 2016) broke essentially every public scheme in seconds to minutes on a laptop with no knowledge of the design. Every WhibOx contest submission (2017, 2019, 2021, 2024) has been broken. Its own industry deploys it as *delay*, always paired with server-side analytics and revocation | no gain |
| Consumer secure enclave (TEE) | Not available: Intel removed SGX from consumer CPUs at 11th generation (2021); AMD's equivalent is a server/VM technology; ARM TrustZone is not exposed to third-party apps on any general consumer platform. Where it exists it has been broken repeatedly (Foreshadow, Plundervolt, SGAxe, ÆPIC Leak, Downfall). **And it would not help anyway**: an enclave cannot render, so the ore must exit to the untrusted GPU, and nothing stops the adversary asking the enclave for chunk after chunk | not applicable |
| Server-side rendering (pixel streaming) | Genuinely works — the client receives only frames — at the cost of a GPU per concurrent player and 10–30 Mbit/s per session | not viable |
| Kernel anti-cheat | Does not touch a data-extraction attack under break-once-run-everywhere; breaks the Linux and macOS targets; a legal and support liability | no gain |

### The precedent, both directions

**Where the generator ships, the map exists.** Diablo II sent the client a map seed and let the client
generate the level; maphack appeared immediately and is still maintained today, with public clientless
renderers that reconstruct any act from a sniffed seed. Minecraft's server-side anti-xray (Orebfuscator,
Paper's engine modes) works purely by rewriting outgoing packets, and Paper's own documentation concedes
the defeat in one sentence: *"If the client is able to obtain the world seed, it is able to know the real
location of every generated ore, completely bypassing Anti-Xray."* And the seed is itself recoverable
from ordinary play — SeedCrackerX harvests bits from observed structures and cross-checks against the
hashed seed the server sends every client. The only countermeasures that ever worked were changes to the
*key* (per-structure seed tables), never to the code. Across ten surveyed games — Minecraft, Terraria,
Valheim, 7 Days to Die, Factorio, Satisfactory, No Man's Sky, Deep Rock Galactic, Astroneer, Dwarf
Fortress — the rule is exceptionless: wherever placement is client-computable, an offline map tool
exists. Factorio ships the preview *itself*; Satisfactory's community map became the de-facto planner.

**Where the generator stays on the server, it holds.** Elite Dangerous runs its Stellar Forge as a
service; after eleven years and millions of players the community's crowdsourced star map holds **0.024%
of the galaxy**. The part of Elite that *did* ship to the client — the procedural system-naming scheme —
was decoded by players in 2015 and is now a documented public algorithm. The line held exactly where the
code stopped shipping.

**And the industry's own conclusion, arrived at twice, twenty years apart, is: cull server-side.** RTS
map hacks (StarCraft, Warcraft III, Age of Empires) were structurally unpatchable because lockstep
peer-to-peer meant every client held the full state and fog of war was a rendering filter; the response
was detection and bans, never prevention. The fix came from the other direction as server-side
relevancy — Quake and Source visibility culling, Unreal's network relevancy and dormancy, Valve adding
occlusion culling so servers stop sending fully-occluded enemy positions, and most quotably Riot's
Valorant "Fog of War", moved into the server explicitly on the reasoning that the only way to beat a
cheat that reads memory is for the data not to be in memory. That is decisive because it comes from the
company shipping the most invasive kernel anti-cheat in the industry: even they treat the driver as
defence in depth *on top of* not sending the data.

---

## The correlation problem — and yes, it is the real constraint

**Say this loudly: concealment is spent by the PLACEMENT RULE, not by the key.** The residual secrecy of
a hidden deposit is not set by how long the key is; it is set by how tightly the placement is conditioned
on terrain the client already has *exactly and bit-identically* (the cross-build determinism gate
guarantees the adversary's copy is exact). Biome, strata, depth below the surface and carver geometry are
all in the client. So an attacker computes the *admissible set* for every chunk offline, and what is left
is only the choice within that set.

Worked over one 62³ chunk (238,328 cells), with a 3 m-radius deposit (113 cells):

| Placement rule | Admissible cells | Prospecting advantage from public data alone | Bits of real secrecy |
|---|---|---|---|
| Unconditioned within the chunk | 238,328 | 1× | 17.86 |
| Coarse 30 m depth band | 115,320 | **2.1×** | 16.82 |
| 8 m stratum band | 30,752 | **7.8×** | 14.91 |
| "Believable geology", 1% of the chunk | 2,383 | **100×** | 11.22 |
| "At a stratum contact within 5 m of a cave" (~200 cells) | 200 | **1,192×** | 7.64 |

At the bottom row the deposit is *determined* by public data and the concealment has been destroyed by
the placement rule, against a 256-bit key. **The more geologically believable the placement, the less is
concealed: the AAA instinct is the attack.**

**Two clarifications, because the numbers above are easy to misread.**

1. *As §3.5.1 step 7 is written today*, nothing implies anything tighter than a depth band, so the
   current exposure is the benign 2–8× row. The risk is the **next revision** adding believability rules,
   not the committed one. That is exactly why this must be settled before the ore table's shape freezes.
2. **The per-cell position is the least important of the three quantities.** Two planet-scale channels
   matter more and neither is about position:
   - **The material map.** If "titanium appears in biome X within depth band Y" is a public table, and
     that biome covers 5% of the surface and the band is 30 m of a 16,864 m column, the search is
     narrowed **11,243×**, offline, galaxy-wide. That is worth more than any per-chunk entropy.
   - **The rate.** If the number of deposits per chunk is a public function, every chunk in the galaxy
     can be ranked offline by expected yield — and that ranking is ninety per cent of what an ore map is
     worth. It survives any key length, because it is a different quantity.

**The rule that follows, and it is a placement rule:**

> The concealed table owns the **count**, the **material** and the **richness**. It takes the public
> world as input only through a **coarse integer depth band** — crust versus mantle — and nothing finer:
> no biome, no stratum, no cave proximity, no fault geometry.

**The realism you lose is bought back deliberately, and this is the design's nicest property.** Run the
correlation the other way. The *public* generator places a visible **indicator** material — a gossan, an
alteration halo, a discoloured stratum — from the seed, as ordinary terrain. The *concealed* table then
places a deposit with a probability that depends on whether an indicator is present. Choose, say, a 4×
enrichment: reading the rock is then a genuine, learnable, tradeable skill worth 4×, and the offline map
of every indicator on every planet — which will exist — is worth 4× and nothing more. Geology becomes an
advantage a player earns rather than a file somebody sells.

That enrichment factor is a **declared field of the ore table** (no magic numbers) with a property test
behind it: run the real concealed table over N chunks and assert that the best public predictor's
digs-to-first-hit advantage stays under the declared bound. **The target is not zero.** A 2–4× advantage
is good for the economy; a 1,192× advantage is a product someone sells.

---

## The mechanisms compared

All figures are per player, at 1% ore by volume, at the design's own 8-byte record, and at the **only
legal reveal scoping** (see the note below the table). Tier-0 residency radius 785.7 m; chunk edge 62 m;
new columns entering the tier-0 disc = 0.409 × speed per second.

| # | Mechanism | Bytes per revealed chunk | Walking (5 m/s) | Ground vehicle (30 m/s) | Flight (200 m/s) | Server CPU, 10k players | Persistent storage | What the adversary learns per entitled chunk | Verdict |
|---|---|---|---|---|---|---|---|---|
| **A** | **Public placement** (§3.8 A) — all ore in the client | 0 | 0 | 0 | 0 | 0 | 0 | the entire galaxy, offline, forever | defensible; see the strategic section |
| **M1** | Full-volume reveal of the chunk | 19,064 B | 117 KB/s | 701 KB/s | 4.68 MB/s | as M3 | 0 | all 238,328 cells | **reject** — 62× the bytes *and* 62× the leak of M3 |
| **M2** | Give the client a per-chunk key and the concealed generator | 8 B | 0.05 KB/s | 0.3 KB/s | 2 KB/s | ~0 | 0 | all 238,328 cells, **transferably** — publish 8 bytes and anyone regenerates it offline forever. The whole starter planet's surface-band key database is 2.4 GB: a torrent | **reject** — and it deletes the very gate §3.8 built to catch a concealed generator in the client |
| **M3** | Reveal only air-adjacent (visible) ore cells | 308 B | 0.49 KB/s | 2.9 KB/s | 19.6 KB/s (49 KB/s at a 500 m/s skim) | **54 cores** naive; ~0.2 cores if written as a per-deposit exposure test | 0 | ~1.6% of the chunk — the cells you could see anyway | **reject**, on the three grounds below |
| **M7** | Reveal deposit *descriptors* (centre, radius, material) | ~6 B/deposit, ~222 B/chunk | as M3 | — | — | as M3 | 0 | the deposit's **whole extent** from one exposed cell | optional flavour inside M3 only; a gameplay question (vein-following), not a cost one |
| **★ M12** | **Concealed materials never enter the world grid at all** — a server-held drop table, not an overlay | **0** | **0** | **0** | **0** | ~1.7% of one core (one keyed hash per touched chunk, cached) | **0** | only what this player personally extracted | **RECOMMENDED** |
| **M4** | Survey instrument (rides M12) | 128 B per survey | 12.8 B/s at a 10 s cooldown | — | — | negligible | 0 | a quantised bearing/depth/confidence reading | **adopt, with M12** |

**Why M3 is rejected — three independent grounds, and the first is the decisive one.**

1. **It cannot be scoped at reach; it must be scoped at 786 m.** §3.8(B) says foreknowledge is "bounded
   at tens of metres" because the reveal follows the small collider/edit radius. That is not
   implementable. The client draws tier 0 out to 785.7 m, so an exposed ore cell in a cliff face or a
   cave mouth is *drawn* at up to 786 m. A 67 m reveal radius makes one player at 60 m and another at
   700 m see different terrain **facts** — forbidden outright by Addendum 2 §C.7 — and produces a visible
   colour change on approach, which the seamless law forbids. So both of §3.8(B)'s cost claims
   ("bounded at tens of metres", "bandwidth proportional to digging, not to terrain") are false as
   written, and every number in option (B) must be re-derived at 786 m. Re-derived, the traffic is
   **movement-bound, not dig-bound**: digging contributes ~0.05 reveals per block broken (breaking a cell
   in solid rock exposes at most 5 new cells), while traversal contributes everything in the table.
2. **It is farmable.** At 100 m altitude the tier-0 disc gives a 1,559 m ground swath. At 200 m/s a bot
   sweeps 311,720 m²/s against the starter body's 3.28 × 10¹¹ m² of surface: **12.2 bot-days solo, 2.9
   hours with 100 accounts**, from a packet capture, with no modified client. It harvests not only
   surface outcrops but every ore cell exposed in every cave inside the tier-0 volume beneath the flight
   path — which is a large fraction of the ore a miner actually reaches.
3. **It is structurally expensive.** It gives the server a *terrain-generation workload proportional to
   players × draw area* (54 cores at 10,000 walking players if written the obvious way — generate the
   chunk and scan it — and 2,171 cores if those players are flying). It makes the D-9 / D-39.5 per-cell
   area-of-interest reshape a **security precondition** rather than a bandwidth one, because on today's
   whole-realm broadcast one player standing in a mine hands the shaft's ore to every session on the
   planet. It forces the entitlement set to be dilated by one cell for the mesher's apron. And it puts a
   documented exception into "terrain never crosses the wire".

Against all of that, M3 buys one thing: a glinting vein you can see in a cave wall, for one tier of
materials. That is a bad trade — and the next section gets the same visual reward for free.

---

## The recommended design, exactly

### The one rule everything follows from

> **A concealed material is never placed in the block grid by the generator, at any tier, at any
> distance, exposed or not. It exists only as a server-held table saying what a given generated rock cell
> yields when it is removed. A concealed material may appear in the grid only as a cell a player
> authored.**

That is the whole design. Everything below is consequence.

Note what it is *not*: it is not "the client is told the host rock instead of the ore" (a substitution),
and it is not "the reveal is withheld until you are close" (an overlay with a gate). Both of those put
concealed data into the grid and then try to keep it out of the client. This puts it in a different place
entirely — a drop table — so there is nothing to keep out. That is why the cost is zero rather than
small, and why it deletes several defects that every other option has to fix.

### The material classes

`MaterialClass::{Public, Concealed}` — the two values §3.11 already reserves, with the semantics fixed:

| Class | Placed by | In the client | Visible | Bytes on the wire | Determinism gate | Examples |
|---|---|---|---|---|---|---|
| **Public** | the public generator | yes | yes, as an ordinary block | 0 | fully covered, unchanged | stone, dirt, water, iron, copper, coal, and the **indicator** materials |
| **Concealed** | never placed; a drop table only | no | never, as generated terrain (a *player-placed* block of the same material is ordinary and visible) | 0 | its own pin against a committed test secret | the strategic/exotic tier the price signal depends on |

A third value (`Revealed`, exposure-visible) is deliberately **not** added. If the owner ever wants the
reveal lane, appending a registry discriminant later is a client-update event, not a world migration —
the expensive parts (the generator split and the placement rule) are already in place.

### What the client is told, when, by what path

**Nothing, ever, about concealed placement.** There is no message, no field, no tag, no timing signal and
no size signal. TLV tag 3 on the chunk-delta envelope, currently reserved for "concealed reveals", stays
**reserved and unused** — it is the escape hatch if the reveal lane is ever wanted, and it must not be
consumed.

The three discovery paths, and what each costs:

| Path | Carrier | Payload | Cost per player per second |
|---|---|---|---|
| **Extraction (assay)** | the existing block-edit acknowledgement and item grant | the item you receive | **0 extra bytes** — the drop already had to be sent |
| **Survey** | `WorldAction::Survey` on the reliable, sequenced, acked, reach-checked, rate-capped carrier already being built for block edits (its second consumer, P11's fire trigger, is already reserved) | `SurveyResult` — 32 bearings × 8 depth bands × 4 bits = **128 B**, plus a substance class and a confidence class. Range, precision and cooldown are fields of the instrument's registry row, never literals | **12.8 B/s** at a 10 s cooldown |
| **Reading the geology** | the public generator | the indicator material, drawn as ordinary terrain | **0** |

The survey answer is lossy **by construction**, not by tuning: a cheap scanner returns 8 bearings, 2
bands and "metal present"; an expensive one returns 64 bearings, 16 bands, a named substance and a
richness class. Prospecting becomes a tech tree; instrument power draw is an ordinary signal, so it is a
real ship system rather than a UI button; and the result is a *tradeable good*, which is how a
player-driven economy gets information asymmetry from player labour rather than from a formula. This is
EVE's scanning loop and Vintage Story's prospecting pick, both battle-tested — and note that Vintage
Story's readings survive being pooled across a whole server, because a probability field pooled is still
a probability field. **Concealment by resolution degrades gracefully under crowdsourcing; concealment by
secret does not.** That is the argument for making the survey coarse.

### What the server keeps

- **A master secret**, in the deployment secret store (a Kubernetes secret / environment injection),
  reached only through a trait below the `sim::io` seam like a TLS key — so no simulation code ever holds
  it, no shard's private state contains it, and **zero `InterShardFlow` arms are added** (HR1 clean).
  Never in a realm's database file: those live on a shared network volume, and a snapshot leak would
  otherwise be a galaxy leak.
- **Per-realm subkeys**, `realm_key = PRF(master, RealmUid)`, one keyed hash at realm spin-up on the
  existing realm-assignment path. A compromised node then leaks one body, not the galaxy — an HR1-shaped
  property obtained for one line.
- **A key id** — `concealed_key_id: [u8; 8]`, a truncated keyed hash of the master — recorded in
  `RealmDescriptor` beside `last_owner_fence` and the generator schema version, and **refused loudly on
  open** if it disagrees. Without this, two nodes holding different secrets during a rolling deploy fork
  a planet's ore *invisibly to every digest gate*, because the secret is an input to the function rather
  than part of it.
- **The concealed table generator**, in its own server-only crate.

### The cryptography, and the trap

**HMAC-SHA256. Zero new dependencies.** `hmac = "0.12"` and `sha2 = "0.10"` are already declared at
`Cargo.toml:77-78`, already used by `vd-connection-plane`, and §7.5 already recommends HMAC for signal
grants — so this choice also leaves the project with **one** keyed-MAC primitive rather than two (HR3).
What is needed is a pseudorandom function, not encryption: the client never decrypts anything, so there
is no key exchange, no AEAD and no new primitive class.

The performance argument that decided §7 does not exist here. The table is consulted once per touched
chunk (cached), on a path where generating that chunk already costs 0.885 ms; HMAC-SHA256 at 345 ns is
**0.039%** of it. Keyed BLAKE3 at 51 ns saves nothing measurable and `blake3 1.8.5` is in `Cargo.lock`
only transitively (via `bevy_asset`), so promoting it would be a new declaration the owner would have to
make for no gain.

> **THE TRAP, and it must be written into the design because it is the shortcut an implementer will
> take.** Do **not** build the concealed table on `SplitMix64` / `child_seed`. **I built and ran the
> inversion.** `child_seed(parent, salt, index)` at `crates/core/src/rng.rs:70-74` is three rounds of the
> SplitMix64 finaliser with published constants; every stage is a bijection on `u64` (xor-shift-right is
> invertible by iteration, multiplication by an odd constant is invertible modulo 2⁶⁴ by Newton's
> method). With secret `0xdeadbeefcafef00d` and a public key `(0x12345678, 0x00abcdef)`, one observed
> output inverted to `0xdeadbeefcafef00d` exactly, in microseconds. Since the chunk key is public and a
> player legitimately observes outputs by mining, **one mined deposit recovers the master secret**. This
> is not a weakness, it is a complete break, and it is an easy mistake because `child_seed` is the
> codebase's established idiom and §3.5.1 routes all generation through it. Enforce it structurally: the
> concealed crate has **no dependency on `core::rng`**, so the mistake does not compile.

### Placement, and why the generator is integer-only

The concealed table is parametric, not a field: per chunk, a keyed draw yields 0..n deposits, each a
centre (an integer cell index), a radius (integer decimetres) and a material (a table row). The
containment test is `|cell − centre|² ≤ radius²` in integer units. **There is no float anywhere on that
path** — which means the whole `Gf` / libm / SIMD-dispatch divergence class does not apply and
cross-target divergence is impossible *by construction* rather than merely tested for. Enforce it with a
scoped clippy `disallowed-types` ban on `f32`/`f64` in that crate, the same idiom as `EffectFree` and
`Gf`. This is also exactly why the placement rule of the previous section (condition only on a coarse
depth band) is free: `GeneratorBand`'s `min_r`/`max_r` are already **integer metres**, so the one input
the concealed generator is allowed to read is already the right type.

### Cost summary, per player per second

| Quantity | Cost |
|---|---|
| Concealment itself, bytes on the wire | **0** |
| Concealment itself, persistent storage | **0** |
| Concealment itself, inter-shard bytes | **0** |
| Survey, at a 10 s cooldown | **12.8 B/s** (0.05% of the 24 KB/s snapshot lane) |
| Server CPU, whole galaxy, 10,000 players mining | ~1.7% of one core |
| Latency exposure | none — nothing arrives, so nothing can be late |

---

## What it costs us

### The determinism gate — unchanged, and one new sibling

**`terrain_pin.rs` and `just terrain-crossbuild` are completely unchanged, and P4's Definition of Done
survives verbatim.** This is the single best property of the recommended design and it is *not* true of
any reveal-based option. Because concealed materials never enter the grid, `gen_public` still produces
the entire client-visible world, still byte-identically across server, client and every target-cpu build,
and still pinned at every legal tier. There is no "public half" of the gate; there is just the gate.

One new sibling is needed, for one reason: **a re-shard must not move ore.** A realm can be re-homed onto
a different CPU (§3.9.4), and if two hosts disagreed about deposit placement a player's mine would change
under them.

- **`ore_pin`** — digests over `(test_secret, chunk_key)` in the server-only crate, run by
  `just terrain-crossbuild` on the same 3 targets × 2 optimisation levels. **Two** committed test
  secrets, so the gate also proves the secret actually changes the output. There are no tiers to multiply
  by, so this is 128 digests. The production secret never enters version control.
- **A random-secret property test** over ≥2¹⁶ secrets asserting no panic and bounded output — because
  the production secret is untested by construction, and a key-dependent integer overflow would otherwise
  ship undetected. This is the honest residual: for the public half, millions of players continuously
  re-run the function and a divergence surfaces as a visible seam; for the concealed half, CI is the only
  witness and it only ever witnesses the test secret.
- **A dependency-closure assertion** — a ~20-line `cargo metadata` test asserting the concealed crate is
  absent from the client's transitive dependencies. This turns §3.8's aspiration ("the gate that fails
  the day someone links the second into the client") into a mechanism. `client_server_terrain_parity`
  keeps its existing job unchanged.

Under HR5 the concealed crate is Tier-A at 100% region+branch. That is achievable — it is small,
monomorphic, integer-only and fully exercisable with the test secret — provided the branchy parts (the
admission predicate, the deposit draw) follow the branchless-shim discipline.

### The terrain-never-streams rule — no exception at all

§3.9.3 property (1) says literally that no encoding contains an unedited cell, and §3.10 restates it as
"P4's *only the seed crosses the wire* and P6's *binding no chunk streaming rule* both hold without
exception". **Under the recommended design both sentences stay literally true, unamended.** This is
strictly better than §3.8's option (B) or (C) as currently written, each of which was going to buy a
documented exception for one material class. Say plainly in the design that the exception was
contemplated and is **not needed**.

### Persistence, and the delta path — untouched

Zero bytes. The concealed table is a pure function of `(realm_key, chunk_key)`, regenerated on demand,
never persisted. Mining an ore cell writes an ordinary edit ("this cell is now air"), which §3.9 already
budgets. The per-realm delta budget, the pyramid's ~1.1 entries per edit, `BLOCK_STORE_FLUSH_STEP = 19`,
the three-step compaction checkpoint and the ~1.7 GB per heavily-played planet figure are all unaffected.

**And note what this deletes.** One investigation proposed *materialising* reveals into the delta store
(≈7.7 MB per heavily-explored planet in parametric form, 615 MB expanded — the latter breaking the
§3.9.6 budget outright) so that the secret could be rotated without moving ore under existing mines. That
whole mechanism is unnecessary, for a reason worth stating: **the concealed table is consulted only for
cells nobody has touched.** Once a cell is mined it is an ordinary edit, resolved before any table
lookup. So a rotation cannot move ore under an existing shaft; it can only relocate deposits nobody has
begun to work. That in turn preserves the design's own purity rule — **no observer-triggered writes** —
which a materialised reveal would have violated (a durable write caused by someone standing nearby is the
same anti-pattern the tier-hint rule forbids when it demands a shard's trajectory be byte-identical for
every value of every observer's `tier_floor`).

### Dormancy — works, for a reason worth noticing

§3.9.5 already puts the fold in exactly the right place: a dormant realm's advance is a closed-form
`f(realm_seed, tick_span)` evaluated **by the owner at spin-up**, emitting ordinary WAL edits under its
own fence; the dormant scheduler is not a shard and never opens a block store. The owner holds the realm
subkey, so an NPC settlement's mining output can be folded correctly; the scheduler stores only a summary
and never needs the secret at all. Had the design taken the rejected "dormant writer" alternative, key
distribution would have had to cover the whole **dormant** galaxy rather than the played one. Two things
to write down: the economy-side fold now transitively reads a game-side server secret (the permitted
direction, econ → game, and no economy state lands on the realm/physics/lifecycle path); and the fold
must run where the key is, or NPC output decouples from what a player actually finds on arrival.

### The detail ladder — provably blind

Because concealed materials never enter the grid, the edit pyramid, `coarsen`, `COARSE_SHAPE`,
`summary_of_tier0_cell`, the generator band, the min/max relief pyramid and `vdctl world verify-pyramid`
are all blind to concealment **by construction**. No projection function, no special case, no second
code path. Nothing is added to `BlockRegistryHash`.

> **This is worth dwelling on, because under any other option it is a live defect and a one-way door.**
> The pyramid leaf reads the cell's substance; `coarsen` propagates the dominant substance by summed
> fill; the entry is persisted and shipped to clients as TLV tag 4. I ran the walk-up: **one** block
> break drives rung-1 fill 255 → 223 and rung-2 255 → 251, and rung 3 returns to 255 and prunes — so a
> single break forces a rung-1 and a rung-2 entry, and **two** breaks in the same 8 m cell reach rung 3.
> A deposit of radius ρ dominates a tier-L cell when ρ ≥ 0.492 × 2^L m, and tier L is drawn out to
> 785.7 × 2^L m, so a leaked entry is visible at **1,596 × ρ, independently of the rung**: a 3 m vein at
> 4.8 km, a 10 m body at 16 km, a 30 m body at 48 km. The bigger the deposit, the further it advertises
> itself — server-authored, rendered by an unmodified client, immune to any reveal radius, any padding,
> any per-cell area-of-interest fix and any key length. And it will not show up in a naive test: with an
> ore-inclusive summary an *unedited* coarse cell over a deposit prunes and looks clean, so the leak
> appears only where a player has dug — which is exactly where prospecting information is valuable and
> exactly where an acceptance test would not look. `PyramidEntry` is register row 37 / one-way door 43
> and Addendum 2 §D.1 freezes it before the first world is written. **If any reveal-based option is ever
> chosen, the leaf, the prune comparison and the verify-pyramid audit must all read a public-substance
> projection, and that is a semantics change that costs nothing now and a galaxy-wide pyramid rebuild
> after P6.** The recommended design closes it structurally instead.

Two smaller channels close the same way and should still be written down as invariants, because they are
what a future "just make it visible when exposed" change would trip over:

- **The rim warp is a silhouette tell.** §5.6.1's amplitude table gives ore 0.08 m and loose stone
  0.10 m, and `A(p)` is the minimum over the solid blocks touching a corner — so revealing an exposed ore
  cell moves shared corners by up to 2 cm, splits greedy runs and changes the emitted quad set. A reveal
  is therefore a **geometry** change, not a colour change, and the claim that "a reveal dirties nothing"
  is true only for buried cells and false for exactly the cells a reveal is about.
- **The damage model is a free, repeatable, non-destructive oracle.** §2.3.8 ships an 8-stage crack
  overlay as a change-driven delta; the *timing* of stage changes under a known tool damage rate reads
  the block's integrity to 3 bits, and natural terrain heals on the random tick with an effective time
  constant of 546 s, so the probe costs nothing and can be repeated forever. Under the recommended design
  this is moot (the client and the server both see host rock, and the damage curve is the host's). Under
  any reveal design it must be an explicit registry invariant, or "you must mine it to reveal it" becomes
  "you must tap it once to reveal it", automatable at the edit rate cap over the whole exposed surface.

### The seamless law — zero exposure

Nothing arrives, so nothing can arrive late. There is no reveal to prefetch, no mesher to block, no
re-mesh (which would itself be an ore oracle), no interaction with the residency governor, and no
crossfade band to fit a reveal into. Compare the reveal lane, which needs an entitlement radius of
R₀ × (1 + lead) with the lead as a tuning field, a manifest announcement so the mesher blocks rather than
re-meshes, and an unexamined interaction between the residency governor (which can *shrink* the tier-0
radius under memory pressure) and that lead.

### Two things the recommended design does still cost

1. **The CAS guard is fine — but only because the grid is identical.** §3.9.7's `expected: BlockState`
   compares the client's view against the server's. Under the recommended design both are the host rock,
   byte-identically, so mining works and there is no oracle. **Under any design that puts concealed ore
   into the grid, this breaks twice over**: every first dig into an unrevealed vein is *rejected* by the
   CAS, and the refusal is itself a material oracle (bounded by the server-side reach test to the ~524
   cells of a 5 m reach sphere, enumerable in about 26 s at a 20/s rate cap). Recording this because it
   is the second independent reason a reveal design needs prefetch, and because the fix — masking the
   comparison to the public projection of the state word — is free at P4 and a protocol change after P6.
2. **HR3/HR4 compliance is not automatic.** A ship or station realm has no body and therefore no
   concealed table. That must be a `ShardProfile` capability field, **never a match on shard kind**, or it
   is a straight G-NO-SHARD-FORK violation — and a shard-kind check is the natural thing to write. The
   `assert_feature_anywhere` fixture must exercise the concealed lane on a Spherical **and** a Cartesian
   profile in the slice that lands it, not at P8.

---

## The strategic alternative: accept public placement

This deserves to be argued at full strength, because it is genuinely defensible and it is free.

**The economy argument for concealment is the weakest part of §3.8, and the evidence contradicts it.**
EVE Online is the only large-scale player-driven economy in the genre and its ore placement is **100%
public**: asteroid belts are static, named, permanently public objects; the ore mix in a belt is a
published function of the system's location and security status; third-party sites have catalogued every
belt for two decades and CCP has never treated it as a problem. Every economic lever CCP has pulled has
been quantity and distribution, never concealment — the 2020–21 Resource Redistribution raised some
minerals 400% and cut others 75% and built "Primary Supply Zones" so different space is worth different
things. The economy remains the genre's benchmark, with a February 2026 economic report recording 777
trillion ISK of trade. Value lives in **access, logistics and territorial control**, exactly what §3.8's
option (A) says.

**And the one genuinely concealed thing in EVE shows the ceiling of concealment.** Moon composition was
never computable — you needed a survey probe launcher and trained skills — and it went public almost
immediately anyway, through entirely legitimate crowdsourcing: public moon databases seeded from survey
dumps as early as 2009, alliance-internal scan-sharing tools, corp-wide scan mergers. The 2017 rework did
not restore secrecy; it re-seeded every moon from a seed the community *voted on at a convention* and
made extraction an active cycle instead. **The concealment's real product was never secrecy. It was that
the organisation which did the scanning owned an asset.**

**Factorio is the other end of the same argument.** Its authors concluded the information was not worth
hiding and built the reveal into the product: the map preview and the exchange string make ore layout a
pre-game planning input. The difficulty is throughput, ratios, logistics and pressure — all strictly
downstream of knowing where the iron is — so publishing the answer removed a chore rather than a
challenge. Satisfactory's hand-placed nodes are fully enumerated on a community map that became the
de-facto factory planner, and the developers leaned into it.

**Voxeldust's interesting decisions are all downstream of discovery too**: what you build, how you route
it, whether your ship can get there, who holds the ground, what a consignment market will pay. On the
evidence, the marginal value option (A) destroys is small.

**Does it win?** **No — but the gap is much narrower than §3.8 implies, and it is a gameplay gap rather
than an economic one.** Three things decide it:

1. The economic case for concealment is not made out. §3.8 should stop describing public ore as a threat
   to the price signal; the largest natural experiment in the genre says otherwise. Delete that argument.
2. The **gameplay** case is strong and is what should carry the decision. Elite Dangerous built an entire
   profession and a first-discovery race on server-side generation; Vintage Story built a beloved
   prospecting loop on a coarse probability reading. Concealment is worth paying for **if and only if you
   intend to build the discovery gameplay that consumes it**. As pure denial it is worth nothing.
3. The recommended design's cost is close enough to zero — no bytes, no storage, no wire exception, no
   determinism change, no ladder change — that the gap is almost entirely "do you want a prospecting
   profession?" That is a question about the game you want, not about engineering.

So: **take concealment, in the recommended form, for a small strategic tier only, and only if the survey
instrument and the assay loop ship with it.** If the answer to prospecting-as-a-profession is "not now",
then take option (A) for everything and preserve the option to change your mind by taking rows C-3 and
C-4 of the register below — which cost nothing and are the only things that are expensive to add late.

---

## The decision register

| # | Decision | Recommendation | One-way door? | Cost of deferring |
|---|---|---|---|---|
| **C-1** | Can placement be made "absolutely hidden" on the client? | **No**, by anything the client evaluates — and **yes**, information-theoretically, for anything it is never told. Close the question; do not spend engineering on obfuscation, WASM, a bytecode VM, a server-supplied shader, per-session code streaming, white-box crypto, a consumer enclave or kernel anti-cheat | no | none — this is an answer, not a choice |
| **C-2** | Which §3.8 option? | **(C), redefined**: the concealed half is *never revealed at any range*, not "revealed by exposure or survey". Reject the exposure-reveal lane (M3), the per-chunk key (M2) and full-volume reveal (M1) | the *definition* is, once worlds are saved | see C-3/C-4 |
| **C-3** | **Which materials are concealed?** — **OWNER DECISION** | The small strategic/exotic tier the price signal depends on. Iron, copper, coal and all building stock stay public and visible | **YES** — moving a material from public to concealed after players mine moves ore out from under their tunnels | must be taken **before the first world is written**. Free now; a galaxy-wide resource regeneration under existing builds afterwards |
| **C-4** | The concealed half never enters the block grid — a drop table, not an overlay | **Adopt as a rule.** This is what makes the cost zero and what structurally closes the pyramid leak, the CAS break, the crack-timing oracle and the rim-warp tell | **YES**, in effect: it fixes the semantics of a frozen record | free now. If concealed ore is put in the grid first, the pyramid semantics fix is a galaxy-wide pyramid rebuild after P6 |
| **C-5** | Placement conditioning | Concealed placement owns the **count, material and richness**, and reads the public world **only** through the coarse integer depth band. Realism is bought back through a **public indicator material** with a declared enrichment factor | **YES** — the ore table's shape | free now; invisible when wrong, and only discovered when players publish a prediction tool |
| **C-6** | The enrichment factor / max prior gain — **OWNER DECISION (a number)** | Target a **2–4×** advantage for a player who reads geology. Declare it as a field of the ore table with a property test measuring digs-to-first-hit with and against public knowledge | no | cheap to change while it is a field; expensive once players have learned the geology |
| **C-7** | Cryptographic dependency — **OWNER DECISION, and smaller than it looks** | **HMAC-SHA256 — zero new dependencies** (`hmac 0.12` + `sha2 0.10` are already declared and already used; §7.5 already recommends HMAC, so the project ends with one keyed-MAC primitive). Keyed BLAKE3 buys 0.03% of a chunk's generation cost and would be a new declaration. **Forbid keyed `SplitMix64`/`child_seed` explicitly** — one observed output recovers the master secret, demonstrated | no | none; but the forbidding must be written down now, because it is the shortcut an implementer will take |
| **C-8** | Key management | Master in the deployment secret store, never in a realm database file. Per-realm subkeys via the same PRF. `concealed_key_id: [u8; 8]` in `RealmDescriptor`, refused loudly on open | the key id field is a record change | 8 bytes in a record that already exists is free now; a migration later. Without it, a rolling deploy mid-rotation forks a planet's ore invisibly to every gate |
| **C-9** | The survey instrument and the assay loop | **Ship with the concealment or do not ship the concealment.** Survey rides the block-edit carrier already being built; the assay needs no new machinery at all — the item grant already exists on the edit acknowledgement path | no | this is the real cost. Concealment without discovery removes an activity and returns nothing |
| **C-10** | Gates owed | `ore_pin` on two committed test secrets across the same 3 targets × 2 opt levels; a ≥2¹⁶-random-secret no-panic property test; a dependency-closure assertion that the concealed crate is absent from the client; a scoped clippy ban on `f32`/`f64` in that crate; the concealed lane exercised on a Spherical **and** a Cartesian profile | no | small, and they are what make the concealed half's determinism real rather than assumed |
| **C-11** | If the reveal lane is ever built anyway | Then, and only then: the pyramid leaf + prune + verify-pyramid must read a public-substance projection; the CAS must be masked to the public projection; the delta manifest must not name reveal-only chunks; every entitled chunk must get a fixed-size padded payload (presence and size are both signals); the D-9 per-cell reshape becomes a **security** precondition; entitlement dilates by one cell for the mesher apron; and exposure must be computed **per deposit, locally**, never by generating and scanning a chunk (30–270× worse, and the version an implementer writes first) | several | all free at P4, all expensive after P6 |

**What must land at P4, and it is genuinely small:** the public/concealed field in the registry (already
reserved), the rule that the public generator never places a concealed material, the rule that the
concealed half never enters the grid, and the key-id field. Everything else — the concealed crate, the
survey, the pins — is **purely additive** and can land at P6 or later without a migration, precisely
because concealment does not touch the grid, the delta format, the pyramid or the wire. **That is the
strongest practical argument for this shape: it makes `[USER DECISION 3-G]` itself cheap to defer, so
long as C-3, C-4 and C-5 are taken now.**

---

## Adjudicated objections

Where the reviewers were wrong, or right for the wrong reason. Recorded so they are not re-raised.

- **"Reach-gated reveal bounds foreknowledge at tens of metres" (§3.8 B, and two investigations)** —
  **wrong, and it invalidates option (B)'s whole cost model.** Tier 0 is drawn to 785.7 m, so a 67 m
  reveal makes two players see different terrain *facts* (Addendum 2 §C.7) and produces a colour change
  on approach. Re-derived at the only legal scoping, reveal traffic is **movement-bound, not dig-bound**.
  The adversary is upheld.
- **"Concealment costs ~1.1% of one core galaxy-wide at 10,000 players"** — understates by roughly 50×,
  because it assumed the unimplementable 64 m radius. At 786 m it is 6.13 chunks/s per walking player,
  i.e. **54 cores** at 10,000 walking players if written naively.
- **"54 cores at ten thousand *flying* players"** (adversary) — arithmetic slip; 54 cores is the
  **walking** figure. Ten thousand players at 200 m/s is **2,171 cores** naive. Direction right, number
  understated; it strengthens his own conclusion.
- **"The tier-0 vertical extent is about 186 m", bounding what a bot flyover harvests** (adversary) —
  wrong and again self-undermining: at 100 m altitude the tier-0 set reaches ~686 m below the aircraft
  directly beneath it. The farm's depth reach is larger than claimed.
- **"The pyramid leak is immune to any reveal radius, padding, per-cell AoI fix or key length"**
  (adversary) — right about every reveal-based design, and I verified the walk-up and the 1,596 × ρ
  visibility range independently. **But not immune to the recommended architecture**: if the concealed
  half never enters the grid there is nothing for the leaf to read. The fix is structural, not a
  projection.
- **"Materialise reveals into the delta store (≈7.7 MB/planet) so the secret can be rotated"**
  (investigation 4) — **wrong, and it would have violated the design's own no-observer-triggered-writes
  rule.** The concealed table is consulted only for un-edited cells, so rotation moves only un-mined ore.
  The whole mechanism, its TLV tag and its budget accounting are unnecessary.
- **"A reveal dirties nothing and needs no re-mesh"** (investigation 2) — wrong for exactly the cells a
  reveal is about. §5.6.1 gives ore a 0.08 m rim-warp amplitude against loose stone's 0.10 m, and `A(p)`
  is a minimum over the corner's solid blocks, so revealing an exposed ore cell is a **geometry** change
  that splits greedy runs.
- **"A believable geology rule leaves 7.6 bits and a 1,192× advantage against the committed text"**
  (investigation 1) — right arithmetic (I reproduced it), wrong target. §3.5.1 step 7 as written implies
  nothing tighter than a depth band, so today's exposure is the benign 2.1–7.8× row. It remains the
  binding constraint because the *next* revision will add believability rules — which is why C-5 must be
  taken before the ore table freezes rather than audited afterwards.
- **"Never-revealed concealment has a hard dependency on a survey instrument AND an assay-on-extraction
  loop that do not exist"** (adversary) — half wrong. The assay needs **no new machinery**: the item
  grant already rides the block-edit acknowledgement. Only the survey is new, and it rides a carrier
  already being built with its second consumer already reserved.
- **"Ore concealment protects the price signal, and public ore threatens it"** (§3.8's framing) — not
  supported. EVE runs the genre's benchmark player-driven economy on fully public static belts and has
  never once reached for concealment as an economic lever. Decide 3-G on the **gameplay** argument.
- **"White-box cryptography / a client TEE might carry this"** (raised implicitly by the owner's
  question) — closed. White-box schemes have fallen to *automated* attacks needing no knowledge of the
  design since CHES 2016, and every WhibOx contest submission has been broken; consumer SGX was removed
  from Intel client CPUs in 2021 and an enclave cannot render anyway.
- **"A third `Revealed` material class is needed so rare ore keeps its visible vein"** (investigation 2)
  — rejected. The visual reward is obtained for free by keeping the common tier public **and** by placing
  visible **indicator** materials in the public generator, which is better gameplay (a hint you learn to
  read) at zero cost. Two material classes suffice, and a third can be appended later as a client-update
  event if the owner ever wants the reveal lane.
