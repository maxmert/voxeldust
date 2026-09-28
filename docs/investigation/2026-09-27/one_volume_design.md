# The one-volume world model — the final design

*Written 2026-09-28 on branch `game` at `e8d1747` (GENERATOR_VERSION 21, ARTIFACT_VERSION 9), from the code and the
rulings as they stand. This document builds nothing, measures nothing and changes no other file. It closes workflow
`wf_a9730eb2-4b9` (ruling B21): three readers, three competing designs (procedural first, stored truth first, learned
from shipped games), two adversarial reviews (laws and behaviour; scale and cost), and this synthesis.*

**THE SECOND ROUND (the same day).** After the run started, the owner added four requirements: (1) "Ok, please make
sure that we are extremely performant!!!"; (2) "We can comment out all current special shores and sea models."; (3) "Also
while performant we should not build seams. Also there are should not be jumps or flicker between rungs"; (4) "And we
SHOULD NOT break ground laws and rules of course". §8.3 already deletes the shore and sea models of requirement 2. Three
drafts (performance, seams, laws) and one review of them answer requirements 1, 3 and 4: §10A (the budgets and the
levers), §10B (the seam mechanisms and their gates) and §10C (the full compliance table). Every section that they change
says so in place. §9 holds ONE gate table for the whole design. The second round of §11 answers every finding of the
review. §0.8–§0.10 and questions Q8–Q23 are new for the owner.

**Marks.** MEASURED = a run printed it; the source is named. ARITH = hand arithmetic; the sum is shown. DOC = a number
that a ruling or a code comment states; nobody measured it again for this document. ASSUMED = an input of an ARITH line
that nobody measured. UNMEASURED = nobody printed it. `path:line` = read in the code on this branch.

**Binding.** B21 (design the one volume; "don't try to connect": an existing part stays only if it fits the new model
unchanged; N2 is parked). W16 items 1–3 and B16 (the bar: no flicker and no change of shape between rungs). B18 Q4 (near
slides are judged in flight) and Q5 (no rung decides a hand-over). W2 (live water). The law of B7 (the sea over the
connected ocean; a pocket is a closed basin). W11 and T9 (every level comes from a published law). F6 (the average
machine; the terrain gets a share of the cores), F9 (the bounded ask; the card as a budgeted builder, its worth still the
owner's word), B19, B20 (the coast first; half the cores). The owner's four later requirements (above). SL1–SL10,
HR1–HR6, S5-2, S5-3, S6-6, V4, V5, V6, V10.

---

## 0. For the owner

You said: *"The quality and speed is quite terrible at the moment."* *"I wonder how all that will work when we will edit
the terrain using different tools?"* *"I feel that you don't build a generic solution, but some parts that you try to
fit together."* And you asked: *"Can we generate terrain, but then 'fill' with water, or air, or other substances? And
don't try to connect."* This section answers in plain words. The sections after it give the proof.

### 0.1 What the model is

Yes. The game makes a planet in two passes.

1. **The ground pass** makes the solid ground: the rock, the soil, the sand, the hills and the caves. It never looks at
   water.
2. **The fill pass** gives every open space one substance.
   - An open space below the level of its water holds that water: the sea or a lake, and later lava.
   - Every other open space holds the air, or nothing (vacuum) on a body with no air.

A shore is then only the place where water touches ground. The game draws the top of the ground and the top of the water
as ONE surface, from the same cells. There is no second water surface, and no rule moves the ground to fit the water.

**Example.** A walker stands on the belt's beach. The cells under her boots hold sand. In front of her, the open cells
below the sea's level hold sea water, and the cells above them hold air. The game draws one surface: the sand, the
waterline, then the flat sea. No hole and no wall can stand between the water and the land, because they are one
surface. At your stand today, the same place shows a hole, dark walls and a 208 m wall of land.

### 0.2 Where each water level comes from

The planet's server calculates the planet once (205 seconds for the home planet, measured). It finds the sea's level from
the planet's water, and each lake's level from the lake's water budget. This stays. Two things are new.

1. Each point of the server's large map (one point every 8 km) now says which water its rain runs to: the sea, a lake, or
   a dry pit. So a hollow beside a lake knows the lake's level, also where its own map point is dry.
2. A new check finds, for each lake, the lowest point of its rim on the ground that you see. A lake stands no higher than
   that point, so a lake never stands as a wall over lower ground. If the rim of a dry basin under the sea's level dips
   under the sea's level, the sea fills the basin.

### 0.3 Where your first model needed a correction

1. **"A material and a fill amount per cell"** can place only one boundary in a cell. At a coast, one large far cell
   holds rock, water and air: two boundaries. So each cell keeps its distance to the ground, and the game computes its
   distance to the water from the water's level. Nothing more is stored.
2. **"Water where it connects to the ocean"** is not a fact about one cell. It is a flood over the whole planet. The
   server's map points and the rim check give each place its water instead.
3. **"Every distance is a copy made by one rule"** is possible only where the copy is cheap. The home planet has about
   600 million million columns of 1 m. So from about 450 km out, every far cell is the exact average of one reference
   view. Nearer, each detail step draws its own small hills. Question 1 is about this.

### 0.4 What changes for the player

- **The coast** is one surface at every distance: no hole, no dark walls and no wall of land. The first gate checks
  exactly this, 100 m over the sea at your stand.
- **Hollows and caves.** On a low coast, a hollow below the sea's level holds a small pond at the sea's level. A cave
  below its water's level is full of water. A lake can end a little lower where its rim has a notch. We measure how much
  of this happens before we build. Question 3 asks you to let us find a real groundwater law first.
- **Digging** (your ruling W2). A player who digs a trench from the sea sees the sea flow into it. A pit dug far from
  water stays dry. A sealed room dug below the sea's level stays dry until it opens to water. A dig that would leave a
  wall of water is refused for now, and the player is told why (question 4).
- **A moon with no air** holds vacuum over its ground and in its caves. Today the game writes air there.
- **The far view.** See question 1.
- **A far planet** appears whole when its data arrives: its six faces in one frame, never face by face. It still
  appears at once when it comes into range, about 23 pixels wide. The cure for that belongs to the visibility rules (a
  range that comes from brightness), not to the ground.
- **Speed.** Today one piece of ground takes 11 to 22 ms to build, and most of that time repeats work that was done a
  moment before. The plan first removes the repeated work, with no change to the world, and then adds the model's own
  small work. The separate water surface and its three cuts go. The wider ring of detail at 450–700 km goes (it cost 8 %
  of a frame, measured). §0.8 gives the targets in numbers.

### 0.5 What it costs

- **The planet's server:** more work once per planet and world version. It is not measured yet. Our estimate is a few
  minutes to about one and a half hours of one processor core, split over the server's cores. Question 21 asks you for
  its ceiling.
- **Download near a planet:** up to about 15 MB more for a pilot about 3 000 km up, because the map pieces reach to the
  horizon. The data that comes with the planet's first message gets smaller: 8.9 MB instead of 14.0 MB, and its coarsest
  part (26 KB) comes first.
- **The game's memory:** about 36 MB less, because today's coast counts go. Each drawn point grows by 4 bytes (about
  11 % more drawn ground), so that the water's colour can blend between detail steps (question 16). A little more memory
  while each piece of ground is built.
- **The pictures:** the ground moves once (question 5), so every recorded check and every judged picture of the terrain
  is made again.

### 0.6 What stays

The planet's calculation, its sea and its lakes. The rock layers and the caves. The 19 detail steps, from 1 m cells near
you to 262 km cells for a whole planet. The blend between two detail steps and the half-second rise of a late piece. The
worker threads. How the planet's data travels. The edit messages. The checks that prove that the server and the game
compute the same ground to the byte.

### 0.7 The questions

**Q1 — THE FAR COAST.** The planet is too big to build its far view from its near cells, so the game makes the far view
from the same data with coarser cells. Choose one:

- **(A) EXACT AT EVERY DISTANCE.** The far view is always the exact average of the near view. No coast moves, and no lake
  appears, disappears or flickers at any distance.
  Price: new small hills, built from the large ones downward, so the land looks different and you judge it again; a
  stored map of every shore at 32 m, made once per planet (estimate: 20 to 110 minutes of one processor core, and 57 to
  150 MB stored per planet); about 9 MB more with the planet's first message, and the map pieces to the horizon (up to
  about 15 MB); memory in the graphics card for exact colours (17 to 79 MB at each stand, measured). Even then, water
  under 32 m follows a simple local rule, and a water step of up to about 1 m can stand in rare places.
- **(B) EXACT FROM FAR, SMOOTH NEAR.** From about 450 km out, every coarser view is the exact average of one reference
  view, by one rule. There a coast does not move and no lake flickers: a lake smaller than a pixel is a tint that fades
  as you fly away. Nearer than about 450 km, each detail step draws its own small hills, so a coast slides a few pixels
  while you fly through a blend band. It never pops.
  Price: the near slide. Old measurements, taken before today's shore rules: 3.7 pixels at 56 km and 6.6 pixels at
  111 km; 22 pixels at 450 km, but most of that came from a second far map that this design deletes. We measure the new
  slide on your four stands before you choose. Also up to about 15 MB more download at the worst height.

Your later words, *"no jumps or flicker between rungs"*, change what (B) must show:
- **A slow slide is not a jump.** Your ruling B18 Q5 read the same words so: a change that no detail step decides and
  that is gradual. In B18 Q4 you accepted near slides of at most 1.5 pixels for the sea and 3.3 and 4.8 pixels for
  lakes, judged in flight. Those sizes were measured under a shore rule that this design deletes, and your ruling B16
  says *"no flicker or changing of the shapes between rungs at all"*. So the new slide needs its own numbers and a stated
  limit. Choose the limit: (i) the sizes that you accepted in B18 Q4, as the most that a coast may slide at any band; or
  (ii) the stricter line of the ground (1 pixel for 99 % of the coast, 2 pixels at most). **Recommendation: (i), judged
  in your flight.**
- **A pond that comes back is a flicker.** With (B), a small pond can fade out at one band and fade back in at the next
  while you climb. We do not know yet how often. Your ruling B16 forbids that. So (B) stands as designed only if the
  bench finds no such pond on your stands and on a sample of the planet. If it finds some, (B) needs your change of B16
  for such ponds; the smaller hand-over of question 2 makes them rarer, but not zero; (A) removes them.
- **One new rule, named.** The rim check (§0.2, item 2) compares the server's 8 km map with the drawn ground and lowers a
  lake to its drawn notch. It applies the lake's own law, but it is the kind of rule that your ruling B21 asked us to
  stop. A finer server map (1 km, the next step of the terrain plan) or (A) makes it unnecessary.

**Recommendation: (B).** Build it, measure the near slide and the ponds on your stands, and then decide. If you reject
the measured slide or the pond count, (A) comes on top of (B): it replaces only the small hills and adds the shore map.

**Q2 — A SMALLER HAND-OVER (only with B).** Today a small hill leaves the view when its wave covers fewer than 49 cells
of the next detail step (your ruling T7). If it stays until its wave covers fewer than about 4 to 8 cells, each
hand-over is about 6 to 12 times smaller in pixels, so the near coast slides much less. The exact far view then starts
at about 3 500 to 7 000 km instead of about 450 km. May the bench measure it? **Recommendation: yes, measure only; you
rule after.**

**Q3 — STILL WATER IN HOLLOWS AND CAVES.** In the fill pass, every natural hollow and cave below its water's level holds
that water. This needs no stored water, and no wall of water can stand. But:
- on a low coast, hollows below the sea's level become ponds at the sea's level, also some way inland;
- caves under the sea or under low land are full of water;
- a lake stands no higher than the lowest point of its rim, so some lakes end a little lower;
- a dry basin whose rim dips under the sea's level fills with the sea, and the sea's level drops a little to pay for it.

One flat level in every hollow is a simplification, not a law of physics. Your ruling T9 says that a number which
stands in for a missing law is a defect, and your ruling W11 says that the water supply decides whether a hollow holds
water, never the shape of the hole. Choose:
- (a) we find and measure a published groundwater law first (a candidate: the depth of the water table from the rain
  and the shape of the ground, a published study of 2013), and a hollow holds water only where that table stands over
  its floor;
- (b) a hollow stays dry unless it connects to water: this needs a stored map of every shore at fine detail, with the
  price of Q1 (A);
- (c) you accept the flat level now, as a named simplification.

**Recommendation: (a). The bench measures the ponds, the flooded caves, the lowered lakes and the filled basins under
the law and under the flat level, beside your lake limit (4 % of the land), before you choose.**

**Q4 — DUG HOLES (your ruling W2).** A dug hole fills only when the server's flood reaches it from a water that it
touches, cell by cell, at that water's level. A sealed room dug below the sea's level stays dry. A flood that is too
large for one tick continues over the next ticks, so the water flows in. No wall of water may stand, so the first
version's two walls become two refusals:
- (a) a flood that is too large for the server (a canal into a deep dry basin opens up to about a thousand million
  million cells in the largest basin, an estimate) is refused, and the player is told why, until the digging work
  designs one live level for a whole basin;
- (b) a dig that joins two waters at different levels (a canal from a lake down to the sea) is refused, and the player
  is told why, until flowing water exists (your ruling W3, later). Without the refusal, the canal holds the higher water
  and ends as a wall at the lower end.

**Recommendation: accept both refusals for the first slices.** The first trench, from the sea at your stand, reaches
neither one (the bench checks it).

**Q5 — THE SMALL HILLS' AVERAGE.** The crests of the small hills lift the land by up to 572 m on average above the
large-scale ground that the server placed the water against (measured, ruling B13). Without today's shore rule, many
lakes would drain and coasts would move. We shift the small hills so that their average is zero at every detail step.
Their shape stays; the land near water moves down to where the server put it; every picture is judged again.
**Recommendation: accept.**

**Q6 — NEW DATA FROM THE PLANET'S SERVER.**
- (a) Each 8 km map point's water number changes its meaning: the level of the water that its rain runs to. No new
  bytes.
- (b) The map pieces reach to the horizon, not only to about 450 km: up to about 15 MB more for a pilot about 3 000 km
  up.
- (c) A new far map (the average height of the top, and the share of water) replaces today's far map: 8.9 MB instead of
  14.0 MB. Its coarsest part (26 KB) comes first, so a planet appears whole after its first half second.
- (d) The coast map (2.2 MB) is deleted.
- (e) With the digging slice: the dug cells that hold water; the water's share (later the ground's material shares) in
  the far view of edits; and, with each edit, how many parts it has at each detail step, so that your game shows a dig
  only when all of its parts are here.
- (f) Withdrawn: the planet's map data for district realms (question 19).

**Recommendation: accept (a) to (d) now, and (e) with the digging slice.**

**Q7 — THE TRENCH NEEDS A SMALL PART OF THE EDIT SLICES.** Your ruling T1 puts the block store and the edit lane after the
whole terrain arc, and ruling V4 asks for a detailed discussion before any slice starts. The trench in the first slice
needs a minimal store and a minimal lane for one planet, and first the player's own dig input. **Recommendation: build
the coast first (it is your first gate, ruling B20); discuss the minimal store, the lane and the dig input in detail;
then build the trench as their first piece.**

**Q8 — THE FRAME TARGET.** Is the target 60 frames a second at the window's size today (2 560 × 1 440 pixels), at the
full screen (3 456 × 2 234), or 120 frames a second (the display can show it)? **Recommendation: 60 at today's window
size now; the bench also measures the full screen.**

**Q9 — THE TARGET MACHINE, AND FULL DETAIL AT SPEED.** Your ruling F6 says that the game targets an average machine, and
the numbers of that machine are still owed. Please name it: the cores, the memory and the class of graphics card. On
your Mac (7 builders) the plan aims at full detail at 528 m/s. On an 8-core machine (4 builders) the builders make about
520 to 760 pieces of ground a second (an estimate), at or under the need, so there the finest detail comes later at
speed: no hole and no pop, but less detail. **Recommendation: full detail at speed is the check on your Mac; on the
average machine the rule of your ruling F9 that brings the finest detail in later at speed stays the stated fallback,
measured when you name the machine.**

**Q10 — THE COAST FIRST, OR THE SPEED FIRST.** Your ruling B20 says: the coast first. The speed work on the game's
builder (no change to the world) can come before the coast or after it. **Recommendation: the coast first, as you
ruled, and the builder work right after it. The coast changes the same code, so this order does that work once. If you
want the speed first, the coast waits for about one step.**

**Q11 — THE 8 ms BUDGET.** In September you set 8 ms for one piece of ground (ruling V10). At that time the budget counted
only the piece's own cells and surface. The full work for one piece, with its coarser neighbours, is 16 to 22 ms today
on a moving eye. Should 8 ms become the target for the full work, at every detail step? **Recommendation: yes, as a
target that the bench reports for each detail step. The check of the cells and the surface stays at 8 ms, as you
ruled.**

**Q12 — THE GRAPHICS CARD AS A SECOND BUILDER (your ruling F9, still open).** Measured: beside three builders, the card
built 95 pieces a second, and it cut the worst gap at 528 m/s by two thirds. But the widest pop grew on four flights of
five, and the slow flight's detail went incomplete. Today the card builds nothing on a planet, because it has no planet
map data. The plan can give it that data (up to about 15 MB). **Recommendation: measure the card on the new code first.
It stays a switch until you decide on the numbers.**

**Q13 — THE HELPER THREAD AND THE SPEED MARGIN.** May the calculation of which ground is needed run on one of the
terrain's own 7 threads, not on the main thread? And may each ring be asked a little wider by the eye's speed times the
measured delay? The speed comes from two delivered positions, as the rule of your ruling F9 reads it today.
Nothing is predicted. **Recommendation: yes to both.**

**Q14 — THE EFFICIENCY CORES.** Your Mac has 10 fast cores and 4 efficiency cores. May the terrain's builders ask the
system for the fast cores? **Recommendation: measure first; ask for the fast cores only if an efficiency core builds
1.5 times slower or more.**

**Q15 — THE BORDER SHADE.** Where a piece's coarser neighbour comes from the ground that the coarser step really draws, a
point on a piece's edge can take a slightly different shade: a few pixels at the edges can change. **Recommendation:
judge it by the picture check and by your look.**

**Q16 — THE COST OF THE COLOUR BLEND.** Each drawn point grows from 36 to 40 bytes (about 11 % more drawn ground), so that
the water's colour can blend between detail steps. Without the blend, every near coast pops its colour at a band, as it
does today. **Recommendation: accept, with the net frame time measured at the coast stand first.**

**Q17 — A PLANET WITHOUT ITS SOLVED DATA (only in development, after a change of the world).** Do we draw nothing until
its data arrives, or do we keep its old shape and change the whole planet in one frame? **Recommendation: draw nothing.
The planet then appears whole, never as a mix of old and new pieces.**

**Q18 — A DIG SEEN FROM FAR.** A dig appears in the same frame at every distance, so the near view and the far view
always agree. Do you accept a far view that changes in one frame when the world itself changes? **Recommendation: yes.
This is a change of the world, not of a detail step.**

**Q19 — DISTRICT REALMS (AREAS).** In September you ruled that an Area builds its own floor and gets no planet map data
(R-16). The first version of this design asked to send that data to each Area. **Recommendation: keep your ruling. An
Area's water comes only from its own rows, and a flood stops at its edge.**

**Q20 — THE FAR COLOUR OF LAKES: PER CELL OR PER PIXEL.** The far view's lake colour is exact per cell: a soft edge up to
two pixels wide, and no extra memory. Exact per pixel (the windows of your ruling B17) costs 17 to 79 MB of graphics
memory at each stand (measured). **Recommendation: per cell, measured first; the windows come back only if your flight
shows the soft edge.**

**Q21 — THE SERVER'S TIME BEFORE A NEW PLANET SHOWS.** The new rim check adds server time before a planet that nobody
visited can show (an estimate of 2 minutes to about 1.5 hours of one processor core, not measured). The time is split
over the server's cores. What is the ceiling of the added time? **Recommendation: not more than the planet's own
calculation today (205 seconds for the home planet), on the production server's cores.**

**Q22 — THE DIG MESSAGE.** A dig with a brush shape (a new message of about 30 bytes) gives smooth cuts. A dig cell by
cell gives stepped cuts and about 1 000 messages for the trench. **Recommendation: the brush.**

**Q23 — THE OLD SHORE CODE.** After the new model lands, the old shore test code stays in the tree as a second shore
model. May it go then, as your second requirement allows? The history keeps it. **Recommendation: yes.**

### 0.8 How fast it is

Today, measured on your Mac (7 builders of 14 cores):
- One piece of ground takes 11 to 22 ms to build on a moving eye, and up to about 100 ms when its coarser neighbours are
  not ready. The rule that makes the ground is fast. The time goes to repeated work: a new piece builds its eight coarser
  neighbours again, reads the same map points thousands of times, and runs the full rule also on cells that are only air
  or only rock.
- The builders make about 374 pieces a second. A pilot at 528 m/s, 537 m over the plain, needs about 525 to 1 030 new
  pieces a second (an estimate; the bench measures it).
- To find which ground is needed, the game's main thread spends 7.7 ms of each frame on average at 528 m/s, and 33 ms in
  the worst frame. You see that worst frame as a jump.
- A frame at your four stands takes 9.5 to 13.4 ms, so the game holds 60 frames a second there.

The targets on your Mac:
- 60 frames a second. No stand slower than today, and at most one frame in a thousand longer than 16.7 ms. Later,
  11 ms at every stand, so that a third of each frame stays free for the game that is not built yet.
- The terrain's work on the main thread under 1 ms a frame on average, and under 2 ms in 99 frames of 100.
- About 1 000 pieces of ground a second: full detail at 528 m/s, 537 m over the plain. The estimate after the changes is
  800 to 1 300 a second, if the bench confirms the gains.
- A late piece that you need now is on the screen within a quarter of a second.
- The first picture with no hole within 3 seconds after login, and full detail within about 15 seconds (with warm
  servers; the estimate is 12 to 20 seconds).
- Your own dig on your screen within about 0.18 seconds (on a link of 30 ms each way).

What reaches them:
1. Each piece reads its map points once, not once for each column.
2. A missing coarser neighbour is built only over the part that the piece needs, never as eight whole neighbours.
3. The full rule runs only near a surface. The cells that are only air or only rock take their answer at once.
4. The surface search visits only the cells where a surface crosses.
5. A helper thread, inside the terrain's own 7 threads, finds which ground is needed, so the main thread never waits
   for it.
6. The separate water surface goes.
7. A planet's data comes coarsest first.

Changes 1 to 5 and 7 do not move one byte of the world. The byte checks that we have today prove each one. On an
average machine with 8 cores (4 builders) the estimate is 520 to 760 pieces a second: there the finest detail comes a
little later at speed, with no hole and no pop (your ruling F9). Question 9 asks which machine the targets are for.

**Example.** A pilot flies at 528 m/s, 537 m over the plain. Today the builders make about 374 pieces a second, so the
finest detail comes late. After the changes they make about 800 to 1 300 a second, and her frame never waits 33 ms for
the calculation of which ground she needs.

### 0.9 No seams, no jumps, no flicker

What guarantees it:
1. Each change from one detail step to the next is a slow blend across a band of distance. The ground, the water and
   now also the water's colour blend. The finer ground blends onto the top of the coarser ground and water, never onto
   the sea floor.
2. A piece that arrives late rises into its place in half a second, with its colour. A piece that is built again keeps
   its rise.
3. When you stand still or move slowly, nothing changes back and forth: every decision has a safety margin.
4. A far planet appears whole: its six faces in one frame, never face by face.
5. A dig appears in the same frame at every distance (question 18).
6. No speed change may make a seam. A speed change may make the picture coarser or later, never broken. The speed
   tricks that measured badly before stay out.

The checks (each one can fail, and each one is first shown to fail on a broken version):
- a pop check that now also sees the water;
- a flicker check for an eye that stands still, and one for a walk and a slow climb;
- a crack check at the edges of the pieces, at the edges of the cube faces and at the one coastal cube corner;
- your coast check, 100 m over the sea;
- a slide check during a descent, and a pond check during a climb;
- a trench check from 1 m to orbit;
- a check when a ship leaves the planet's realm;
- a check of a far planet's first frame.

What stays, stated:
- Nearer than about 450 km, a coast can slide a few pixels while a band passes over it. It never jumps (question 1).
- Today's pop check reads more steps at some detail steps than a still picture has (up to 1.6 % of the pixels of one
  band at 528 m/s). The bench first finds the cause. Until then the check makes sure that nothing gets worse.
- A far planet still appears at once when it comes into range, about 23 pixels wide. Its cure belongs to the
  visibility rules.

**Example.** A pilot climbs from the beach at the 15° stand to orbit. Ground more than about 490 km from her never moves.
Nearer, the coast can slide a few pixels while a band passes over it. It never jumps, and the sea never flickers.

### 0.10 No broken law

We checked the whole design against the six hard rules, the standing laws, the conventions and your rulings (§10C). The
first version broke six of them, and each one now has a cure: small ponds that fade out and back in during a climb (a
check that must read zero, or your word, question 1); a wall of water at the end of a dig (the server refuses such a
dig for now, question 4); a far planet that appears late or face by face (its coarsest view first, all six faces at
once); an old game client that could read the new water data in the old way (one version change for the whole change of
the world); planet data sent to district realms against your ruling (withdrawn, question 19); and a flood on a ship with
no "down" (the flood reads "down" from the realm's own gravity, and a ship joins that check when its gravity exists).
The first version also called the unused graphics card a broken rule. It is not, because your ruling F9 made the card a
switch, and your answer on its worth is still open (question 12). The planet's server stays the only writer of its
ground and water, no player position goes into another realm, and every computer draws the same ground and water, to
the byte.

The rulings that the design asks you to change:
- **Retire their mechanisms** (the purpose stays where it is a law): W6, W10 items 1–2, W15, W16 item 2's three-way side
  and its water sheet, W17, W18, B7 item 1's mechanism, B8 item 1's lake hold, B13 (the shore round that B21 parked),
  B14 items 1–4, B15 items 1, 2 and 4, B17's windows (question 20), and B18 Q2–Q3 as separate data.
- **Amend:** W2 (a flood past its limit and two waters are refused, never a wall, question 4; the first trench before
  8e, question 7), W4 item 3, T1's order (question 7), T1 item 6 and T2 (the water's top is part of the one surface), S5-3
  (the far fold above the reference step; the material shares in an edit's far entry), B21 items 1 and 2 (one named new
  rule and two ladder rules, question 1).
- **Only on your word:** T9 and W11 (question 3), B16 (question 1), B20 item 1 (question 10), V10 (question 11), F9's
  open item (question 12), T7 rule 1 (question 2).
- **Kept:** R-16 (question 19), V4 rows 13 and 14, every hard rule and every standing law.

---

## 1. The cell and the extraction

### 1.1 What one cell holds

The WORKING cell is what a build reads: the sample box (`crates/terrain/src/lattice.rs:66-91`), the no-drift digest
and the extractor. It grows from two bytes (`crates/terrain/src/chunk.rs:93-98`) to three.

| Byte | Name | Meaning | Stored or shipped |
|---|---|---|---|
| 0 | substance | The substance at the cell's centre: the solid's code where the cell is solid; otherwise the fill's code — the liquid of the column's water body, or the medium (air, or `Empty` for vacuum). Today's append-only codes (`crates/terrain/src/strata.rs:13-117`). | Yes, as an edit row's kind (Format B). |
| 1 | solid gap `s` | The signed radial distance from the centre to the SOLID surface, in 1/128 of a cell, negative inside the solid. Today's gap byte with today's meaning (`crates/recipe/src/cell.rs:494-525`). | Yes, as Format B's density byte (V6 B-3). |
| 2 | top gap `t` | The signed radial distance to the column's TOP: the higher of the solid surface and the water's level. `t = min(s, f)`, with `f = floor((r − λ)/cell)` in 1/128 of a cell, clamped, and `λ` the column's water level. A column with no water has `t = s`. An EDITED cell takes its `λ` from its row: a liquid row's level, or no level for a dug cell (§3.1). | Never. Every host derives it. |

**Why `t = min(s, f)` is exact.** In a column with ground `h` and level `λ`, `s = (r − h)/cell` and `f = (r − λ)/cell`,
so `min(s, f) = (r − max(h, λ))/cell`: one signed distance to one surface. Both bytes are floored once, and the minimum of
two floors is the floor of the minimum, so `t` is exact to the byte. A still water's surface is flat at its level, so
its gap needs no stored byte.

**The classes, read from the bytes** (gate G-CLASS checks them on every pinned cell):
- SOLID: `s < 0`, or a catalogue block or a sub-metre block stands in the cell.
- LIQUID: `s ≥ 0` and `t < 0`.
- GAS (the medium, or vacuum): `t ≥ 0`.
- WET TOP: `t < s`. The water's level stands over the ground in this column. The test is true at every cell near the
  top, also at a cell whose centre is inside the rock under a thin film of water.

**Format B does not change.** The 12-byte record keeps one density byte, and it is `s` with convention tag 1
(`crates/core/src/registry/mod.rs:51-58`). `t` never enters a record, a store row or a wire message.

**Memory.** ARITH: a box of 64³ = 262 144 cells grows from 512 KiB to 768 KiB for each build. The box is transient;
a built chunk keeps only its mesh (`crates/client/src/chunks.rs:88-199`). Each worker OWNS one box slot, its extraction
table and its geometry vectors, and reuses them (lever E10, §10A.3): no build allocates a box, a core copy or the 1 MB
extraction table (ARITH today: about 3 MB a build, 1.2–3 GB a second of allocator traffic at 400–1 000 builds a second).
Every slot is sized to its own box, because a pool slot that grew and never shrank once gave the card another box's
words (MEASURED: 1 022 of 6 049 boxes drifted with the grown pool, 0 with every slot sized to its box;
`docs/design/DEFERRED.md:8375-8382`).

**Example.** At rung 0 on the belt's beach, the cell under a walker's boots has its centre 0.2 m under the sand's surface
and 1.5 m over the sea's level: `sand`, `s = −26`, `f = +127` (clamped), `t = −26`: solid, dry. A cell 20 m seaward
has its centre 0.3 m over the sea floor and 1.2 m under the sea's level: `water`, `s = +38`, `f = −128` (clamped),
`t = −128`: liquid, wet.

### 1.2 One extractor, one surface: the top

The extractor is exact-integer surface nets. Today it reads ONE sign field, the gap byte, through `is_rock(gap)`
(`crates/terrain/src/extract.rs:79-83`, used at `:170-199` and `:481-508`). The top mesh reads the top byte `t` where it
reads the gap byte today. Nothing else in its arithmetic changes: the exact average of the crossings, the one rounding to
1/256 of a cell, the prism at a cube corner, the edge ownership and the shorter diagonal
(`crates/terrain/src/extract.rs:98-566`).

What follows:
1. **One mesh for the ground and the water.** The top is where not-gas meets gas: the ground under the air, the water
   under the air, and the shoreline where they meet. A shore's group gets ONE vertex, and the ground's quads and the
   water's quads end on it. So a hole between the water and the land (ruling B20) cannot exist.
2. **The water stays at every rung.** The top stands at `max(ground, level)` within the byte's quantum, however thin
   the water is. ARITH at rung 18: the band holds two cells, with centres 131 072 m under and over the ladder radius
   (`crates/seed/src/ladder.rs:257-268`); the sea stands 3 591 m over the radius (`crates/terrain/src/home.rs:32-40`).
   Over sea floor 2 000 m under the radius, `t` reads −66 at the lower centre and +62 at the upper centre, so the
   crossing lands at the sea's level within the byte's step of 2 048 m. With `s` alone the crossing lands on the sea
   floor, and a pilot 10 000 km out sees no ocean: the blocker of both reviews (§11, L1 and L2).
3. **The colour.** Each vertex carries a WET SHARE: the share of its crossed edges whose not-gas end is wet (`t < s`).
   The renderer blends the ground's look and the water's look by it. Today the ground is one colour and the water
   another (`crates/client-render/src/terrain.rs:1996-2090`); 8e and 8o give both their real looks later. A shore is a
   soft colour edge one cell wide. The share counts EVERY crossed edge of a group, owned or not, as the vertex position
   does (`crates/terrain/src/extract.rs:4-10`), so the two chunks of a shared vertex give it one share at a chunk edge and
   at a face seam. A Tier-A test proves one position, one `t` and one pair of shares (its own and its morph target's) for
   every shared vertex at chunk edges, face seams and the corner prism (§10B.4, [SKIRT SHARES]).
4. **The facing.** A quad faces from not-gas to gas (today from rock to air).
5. **Only the crossed cells** (lever E5, §10A.3). The fill records each column's lowest and highest sign change of `t`.
   The extractor visits only those cells, in today's (c, b, a) order, and reuses its worker's table instead of filling
   1 MB for each call (`crates/terrain/src/extract.rs:458-518`). MEASURED today: 1.1–1.97 ms a chunk for 64³ visits
   (`docs/investigation/2026-09-08/landforms/12_chunk_budget_base.md:139-192`); a surface crosses about 4 cells of a
   64-cell column (ARITH), so about 0.3 ms. The first-use vertex order and every mesh byte stay; the full scan stays in
   the tests as the oracle (`mesh_pin`; bench P4).

**Example.** A pilot 100 m over the belt's bay looks at the coast. A group on the waterline holds rock cells, water cells
and air cells. Its one vertex is the average of the rock–air crossings and the water–air crossings. The beach's last quad
and the sea's first quad share it, so the pilot sees the sand run into the water with no sky between them.

### 1.3 The bed: the ground under the water

The BED is the boundary between SOLID and not-solid: the same extractor, run on `s`. The mesh draws it only where an eye
can see it: an eye under the water's level, or water shallower than the visible depth when 8o gives water its clarity.
In the first slice the water stays opaque and nobody dives, so the bed is not drawn. Today's floor cut
(`crates/client/src/chunks.rs:1427-1436`, `:1662-1690`) does this job with a special rule on the side words; here it
follows from the one top.

- A column asks only for the chunk of its TOP. Over the sea the ladder asks for one chunk per column (the sea's
  surface), and never for the floor's chunk while the bed is not drawn.
- Collision never reads a mesh. It reads `s` and the blocks at rung 0 (§6.2).

### 1.4 Flat water, drawn with few triangles

The one extractor puts one quad on each open-water cell: ARITH 62 × 62 × 2 = 7 688 triangles for an open-water chunk.
That shape was MEASURED once already: on the coast leg, 33 million sheet triangles and a 32.2 ms frame, against 0.59
million triangles and 23.2 ms after the sheet's deep-water blocks (`docs/design/DEFERRED.md:9063-9084`). So the first
slice carries ONE generic merge:

- A FLAT REGION is a set of groups whose not-gas cells are all liquid at one level. Its vertices stand on the level's
  sphere.
- The mesh draws a flat region as blocks of n × n cells with `n² ≤ R / (16 · cell)`: the widest block whose chord dips
  under the level by no more than the extractor's own step. The bound exists already (`sheet_block_cells`,
  `crates/terrain/src/position.rs:311-333`); its arithmetic stays and the sheet that used it goes. ARITH on the home
  planet: the whole chunk (62 cells) at rungs 0–6, 55 cells at rung 7, 27 at rung 9, 9 at rung 12, 6 at rung 13, 3 at
  rung 15, 1 from rung 17.
- A block keeps the one mesh's own vertices on its rim, so no crack opens against a neighbouring block or chunk.
- **THE MERGE KEEPS THE MORPH** (§10B.4, [MERGE]). A vertex joins a flat block only when its morph metre is within one
  vertex quantum of nought AND its target share equals its own share: the coarser rung is flat water at the same level
  there too. Where the coarser rung holds an island under a finer flat block, the vertices over it stay, so the island
  never pops at a band's far edge (§10B.6 X7; bench B-S3 counts what the condition costs).
- ARITH: about 250 triangles for an open-water chunk at rungs 0–6 (a rim of 248 vertices), and up to about 500 at
  rung 13.
- A MEASURED lesson stays in force: a block whose chord dips more than the extractor's step reads through the air as a
  lighter square (`docs/design/DEFERRED.md:8850-8880`). So the bound is a law of the picture, not a taste.
- **THE OPEN-WATER CHUNK IS CHEAP** (lever E4, §10A.3). An open-water column whose ground bound lies more than one cell
  under the chunk needs no octave sum and no cell kernel: its `s` clamps, and its `t` is the level's gap. So an
  open-water chunk costs one `λ` read and one crossing a column, and the merge: ARITH under 1 ms, reported per rung
  (budget B-BUILD, §10A.2). The sea is 78 % of the home planet (MEASURED, `docs/design/DEFERRED.md:9521`).

Gate G-FRAME (§9) compares the frame and the mesh bytes at the coast stand with today's sheet BEFORE the sheet is
deleted. Bench P7 first measures the sheet's share of TODAY's frame: the 3.0 ms of §7 was measured before the sky fix
(§10A.1.3).

---

## 2. Generation

Every pass is a pure function of (seed, address, charter, the body's solved data) in the recipe's fenced integers
(`crates/recipe/src/gi.rs:1-53`). One source serves the shard's CPU, the client's CPU and later the card (SL10; F7; F8).

### 2.1 The solid pass

The solid pass is today's column pass (`crates/recipe/src/plan.rs:214-247`) with two changes.

1. **The shore call goes** (`crates/recipe/src/plan.rs:236-241`). No water word enters the solid pass. `FieldRead`
   loses `water` and `side` (`:137-175`).
2. **THE CENTRING.** Each ridged term is re-centred on its own mean: `(1 − 2|n|) − μ`, where `μ` is one pinned word —
   the mean of the recipe's own ridged fold over a fixed lattice, measured once (bench M1). The mean of a smooth term is
   zero by the noise's symmetry; gate G-MEAN checks it. The change is one constant in the ridged branch of
   `octave_term` (`crates/recipe/src/height.rs:460-467`); the mask shape that the card needs stays.

**The word is tested, never trusted** (§10C LAW-11). A test derives `μ` again from the recipe's own ridged fold and fails
on any difference from the pinned word (gate G-MU). Bench M1 derives it over TWO independent fixed lattices: if the two
words differ by more than the word's own quantum, `μ` is a property of the lattice and not of the fold, and the centring
is refused. B14 set the precedent when it made `C_smooth` and `C_ridged` a law.

**Why the centring is necessary.** The solve places the sea and every lake against `Z`. The ridged fine octaves stand on
a positive mean: MEASURED 571.8 m at share one (ruling B13; `crates/terrain/src/body.rs:132-135`). So today the drawn land
near every water stands `m × 572 m` over the ground that the solve put the water against, and the shore law hid it.
Without the shore law and without the centring, a lake shallower than the lift drains in the drawn world (review finding
L4). ARITH: `μ ≈ 571.8 / 1 025.3 ≈ 0.56`, where 1 025.3 m is the ridged band's amplitude sum: octaves 5–9 carry
401.8, 319.6, 198.9, 79.9 and 25.1 m by the spectrum at `crates/terrain/src/body.rs:50-66`.

**What the centring gives.** The drawn mean equals `radius + Z + coarse octaves` at EVERY rung, rung 0 included. The far
rungs need no tail. The shape of every crest stays; the land moves down by `m × Σ(a · μ)` over the rung's live ridged
octaves. It reads no water, so it is not a shore rule.

**Unchanged in the solid pass:** the octave sum and the roughness factor (`crates/recipe/src/height.rs:52-332`), the
survival table (`crates/terrain/src/body.rs:143-308`; see §4.6), the cap-rock bench (`crates/recipe/src/terrace.rs`), the
C1 read of `Z` (`crates/terrain/src/artifact.rs:1811-2012`), the strata, the beds and the provinces
(`crates/recipe/src/cell.rs:322-445`), and the cavern and tube hollows (`:447-457`, `:588-621`). One change there: a
hollow OPENS its cell (`s ≥ 0`) and no longer stamps it air (`:521`). The fill pass decides what fills it.

**The frozen patch (F1).** A frozen patch keeps its octave count; its centring uses that count's own table. No new owner
word is needed.

**Example.** A lake stands 60 m deep at node scale on ground whose roughness factor is 0.3. Today its drawn floor stands
about 112 m over its own level (ARITH: 60 − 0.3 × 572 ≈ −112), and only the shore law held water there. After the
centring, the drawn floor averages the solve's floor, and the pilot sees the lake where its budget holds water.

### 2.2 The level map: which water each place drains to

**Definition.** For each 8 192 m node: the level of the FIRST standing water that the node's water reaches along the
solve's own water tree — the sea on the connected ocean, a lake's budget level, a pocket lake's level — or DRY where the
water ends at a dry pit. The tree already carries the water that way and "ENDS it where the budget holds it"
(`crates/terrain/src/lakes.rs:47-54`).

**Every node of a catchment carries its level**, not only the nodes under the water. Today `water_level` answers DRY for
every node that stands over its water (`crates/terrain/src/solve.rs:838-857`), so a column whose own node is dry cannot
know the lake beside it (reader finding 2).

**Where it lives: the row's water word, with a new meaning.** The row keeps its ten bytes
(`crates/terrain/src/artifact.rs:116-136`); `water_m` now holds the level map. A node under water still reads its own
water's level; a dry node now reads the level of the water that it drains to. No byte is added. ARTIFACT_VERSION goes to
10.

**Cost.** One memoized walk over the tree after the tree is built (`crates/terrain/src/solve.rs:1953-1962`): 8 871 936
steps, well under one second (ARITH).

**It states B7's law as data.** The sea stands over the connected ocean only. A pocket's catchment carries DRY or its own
lake's level, so no host can put the sea into a pocket.

### 2.3 The rim pass: a still water stands no higher than its lowest drawn spill

**The problem.** The level map is an 8 km fact. Under a node, the fine octaves carry up to 1 533 m of relief at share one
(ARITH from the spectrum, `crates/terrain/src/body.rs:50-66`). So the drawn ground can cut a notch into a rim that the
solve saw closed. Where two neighbouring columns take different levels and both stand under the higher one, the higher
water ends in a vertical face against air (review finding L6). A wall of 208 m on the coast (B20) is what stopped the
arc (B21).

**THE LAW.** A still water stands at most at the lowest point of its rim on the drawn ground. Two waters that the drawn
ground joins below both of their levels are one water. This is the law that the solve applies on its nodes
(Fill–Spill–Merge, `crates/terrain/src/lakes.rs:14-23`), read on the ground that a player sees.

**THE PASS** (in the solve, once per body, after the level map):
1. Find every boundary between two catchments with different levels (DRY counts as no level) where the drawn ground CAN
   come near the higher level: `Z − B < ℓ_high`, with `B` the column's fine-relief bound plus the bench's bound. Most
   divides stand far over their lakes and drop out here.
2. Along each remaining boundary, read the drawn ground on both sides at every rung up to the reference rung `L_ref`
   (§4.1), and take the lowest `max(h(a), h(b))` over the pairs of neighbouring columns that straddle the boundary. That
   is the drawn spill `p`, floored to whole metres.
3. If `p < ℓ_high`:
   - a lake lowers its level to `max(p, ℓ_low)`;
   - if the higher water is the sea, the lower region JOINS the sea, and its level becomes the sea's.
4. Run the lakes' budget again for every changed body, and the sea's bisection again when a region joined the sea,
   inside the existing lakes-to-sea fixed point (`crates/terrain/src/solve.rs:1904-1943`; at most `LAKE_SEA_SOLVES`
   re-solves, with its convergence flag).
5. Write the final levels into the level map.

**What it guarantees.** At the surface, no still water stands against open air at the same radius on a catchment
boundary, at any rung up to `L_ref`. Above `L_ref` the far fold (§4.3) averages a picture that holds no such face. Gate
G-NO-WALL checks every rung, whole on the stands and on a sample of the planet.

**Its price** (question 3):
- Some lakes end lower: a full lake drops to its drawn outlet. That is physical, because a spilling lake stands at its
  outlet.
- A pocket whose drawn rim dips under the sea's level fills with the sea, and the sea's level drops a little to pay for
  it (the inventory; T9). MEASURED at generator 19: 148 pockets, 59 470 nodes, 3.2 million km²
  (`docs/investigation/2026-09-24/terrain_believability_status.md:538`). How many join is UNMEASURED (bench M3).
- The time is UNMEASURED. ARITH on ASSUMED inputs — 10 000 to 50 000 lakes, 5 to 30 km of low boundary each, both sides,
  every rung up to `L_ref` (about twice rung 0's count), 0.67–0.84 µs a column (`docs/design/DEFERRED.md:9533-9535`):
  0.2–6 × 10⁹ reads, about 2 to 84 core-minutes, split over the server's cores. A planet that nobody visited draws
  nothing until its head lands (§5.2), so this time delays its first picture. Bench M3 prints its WALL time on the
  production server's core count, and the owner rules a ceiling for that wall time (budget B-BAKE, §10A.2; question
  21). ARITH: under the recommended ceiling of 205 s, the pass alone needs about 1 to 25 cores (120–5 040 core-seconds).
- **A split is deterministic** (§10C LAW-20). The pass and the far bake run on a FIXED partition of the nodes with an
  ORDERED merge, so the artifact's digest is one number at any core count (gate G-SPLIT, at 1 and 7 workers; bench
  M11). The solve's other-host leg G6 has never run (`docs/design/DEFERRED.md:7721-7736`); it runs before any second
  shard derives an artifact.
- **It is a new rule between two sources of truth, and the design names it** (§10C LAW-28; B21 item 1: "STOP ADDING
  TERRAIN RULES"). It reads the 8 km level and the drawn ground, and it moves the level to fit the ground. It applies the
  lakes' own law (Fill–Spill–Merge) on the drawn ground and runs the budget again (T9), but it is the kind of rule that
  the owner asked to stop. Two representations make it unnecessary: a finer solve (R2's 1 km level: the fine relief
  under it falls from 1 533 m to about 315 m, ARITH §2.4) or option (A) (§4.5). Question 1 states it.

**Example.** A tarn in the belt's pass is full, so its budget level equals its saddle node's height, 2 100 m. At rung 0
the fine octaves cut a notch 14 m deep into the saddle's crest. The pass finds the notch and lowers the tarn to 2 086 m.
At every rung the tarn's water now ends at ground that stands at or over 2 086 m, and no sheet of water hangs in the
notch.

### 2.4 The fill pass: every open cell gets its substance

**Once per column** (in the column pass):
1. Read the signed gradient of `Z` at the column. The slope read already takes two signed central differences before
   their absolute value (`crates/terrain/src/artifact.rs:1985-2012`); the fill keeps their signs.
2. THE DESCENT STEP: go one node length downhill along `−∇Z` and take the nearest node there. The seam table carries the
   step across a face edge (`node_at`, `crates/terrain/src/artifact.rs:1831-1870`). On an exact flat, take the column's
   own nearest node.
3. `λ` = the level map at that node.

`λ` is a function of the column's site on ONE field (the rows' spline). So every rung that reads the rows gives a column
the same `λ` (gate G-ONE-LEVEL), and two chunks on either side of a seam agree.

**Once per cell** (rank 1 only: the seed's own cells; §3 treats edited cells):
- an open cell (`s ≥ 0`) whose centre stands under `λ` holds the body's liquid;
- every other open cell holds the medium;
- `t = min(s, f)`.

**The reads are gathered once per box** (levers E2 and E4, §10A.3). The fill's `λ` read and the descent step's gradient
signs read the box's gathered node window: one lookup per node per box, never a map lookup per column (today a node
read is a `BTreeMap` lookup with a modulo and a divide, `crates/terrain/src/artifact.rs:724-733`, 102–162 ns MEASURED,
`docs/design/DEFERRED.md:9533-9535`). The fill's runs join the per-column collapse: a run of cells more than one cell
over the column's top is gas by rule; a run more than one cell over the ground's upper bound and under `λ` is liquid by
rule; a run under the strata's and the caves' reach is bedrock by rule. Only the bands near a surface run the cell
kernel. The plain per-cell pass stays in the tests as the oracle, byte for byte (benches P2, P3).

No branch names a landform.

**Why one descent step and not the nearest node.** A divide stands on a crest of `Z`. A column on a lake's inner slope
drains into the lake; a column on the outer slope drains away. With the nearest node, the rim node's square reaches past
the crest, and part of the outer slope under the lake's level would hold perched water. The descent step puts the
boundary near the crest, where the ground is highest, so the rim pass (§2.3) lowers a lake only for a real notch.

**THE SUB-NODE CHOICE, named** (reader finding 1). Of the three options — a node rule, finer shipped water facts, or a
solid law near water — this design takes the node rule (the level map through the descent step), cures its walls with
the rim pass, and states its price:
- **It fills every natural open space under its water's level: the WATER TABLE of porous ground.** A hollow of the fine
  octaves beside the sea is a pond at the sea's level. A cave under the sea floor or under low land is full of water.
- **It never fills a pocket.** A pocket's catchment carries DRY or its own lake's level.
- **It is a STATED SIMPLIFICATION, not a published law** (review finding L15; T9; §10C LAW-1). A real water table
  follows the ground and stands over its outlet inland. Where the fine relief can reach the level — a coastal lowland, a
  lake's shore plain — ponds stand at the catchment's level. T9: "A band that stands in for a missing law is a defect:
  the law is owed." W11 item 2: a hollow ends dry, partial or spilling, "decided by the water supply, never by the shape
  of the hole". So the flat level does not ship on its own authority. THE ORDER (question 3):
  (a) a PUBLISHED WATER-TABLE LAW with a calibration body first. The candidate is the equilibrium water-table depth from
  recharge and terrain (Fan, Li & Miguez-Macho 2013, Science 339, 940–943; to be read before use), fed by the row's own
  runoff (W11's Turc–Pike partition). A column away from standing water takes the lower of its catchment's level and
  the law's table, so a hollow holds water only where the table stands over its floor (bench M2b).
  (b) The connected-only fill of option (A)'s shore map (§4.5).
  (c) The flat level, only on the owner's amendment of T9 and W11, after M2's numbers.
  The pond area and the flooded-cave share are UNMEASURED (bench M2; the drawn pond area joins the lake share against
  G-LAKE-SHARE's 4 % of land). Two cures of the cause exist, both outside the fill: R2's finer solve (a 1 km level: the
  fine relief under it falls from 1 533 m to about 315 m, ARITH from the spectrum), or option A's stored shore map (§4.5).
- **One level per column.** A pond over a flooded cave at another level cannot exist, and a perched lake over a dry cave
  cannot exist.

**Caves at a catchment boundary.** A cave that runs under a boundary between two levels can hold the higher water on one
side and air on the other: a face inside the cave. Caves carve only at rungs 0–3 and only in a depth band
(`crates/recipe/src/cell.rs:346-350`; `crates/terrain/src/body.rs:1151-1176`), so such a face shows only to an eye inside
the cave. Its count is UNMEASURED (bench M10). If the count is not small, the cure is to take cave hollows into the rim
pass: a lake then drains through a karst outlet.

**Example.** A tarn stands at 500 m behind a rim whose crest is at 520 m, and the ground falls to 300 m on the far side.
Halfway down the outer slope the ground is at 410 m. The descent step there points away from the tarn, so the column
takes the outer water's level and stays dry. A column on the inner slope at 490 m points into the tarn and holds tarn
water up to 500 m.

### 2.5 Other substances

- **Vacuum.** The medium of a body without the air flag (`crates/core/src/look.rs:168`) is `Empty` (V4 row 13: a removal
  and vacuum are `Empty`; `Air` is the atmosphere). Today the cell charter writes air on every body
  (`crates/terrain/src/chunk.rs:477`). One word changes.
- **Air and other gases.** The medium's substance is the charter's atmosphere. A gas that pays is live state from the
  owning shard, never seed-only (S5-6; the seed and secrecy ruling, S1).
- **Ice.** A water body in a frozen climate takes ice at its surface, from the climate rows and a growth law; a glacier
  is a layer of the solid pass. Both belong to 8e.
- **Lava or another liquid.** A water body is a level plus a liquid. A lava lake is one more body with its own liquid,
  when a volcanism law exists (T9).
- **Salt and fresh water.** A closed lake holds salt water (Langbein 1961, the budget's own source) and so does the sea;
  the liquid per body comes from a law, never a draw. The registry holds both liquids
  (`crates/core/src/registry/substance.rs:228-240`). Not in the first slice: every body on the home planet draws as water.
- **Rivers.** A river is a fill whose level falls along its path: the same per-cell rule with a sloped level. It belongs
  to 8d and 8o; the drawn rivers stay retired (W16 item 4).
- **Tides and waves.** Never in the static cells (SL10; T3). A tide or a wave is ONE closed-form function of
  (direction, tick) that moves every wet top in the vertex and fragment stages, the same at every rung — never a change
  of cells and never a rebuild, so two rungs never disagree between rebuilds (§10B.6 X20; 8o). Buoyancy reads the same
  function.
- **Every later fill substance** (lava, a salt, a liquid metal) first passes the seed-and-secrecy test (the ruling of
  2026-08-27, S5 item 4): *"if this were printed on a public wiki tomorrow, would the game still work?"* (§10C LAW-24).

---

## 3. Edits and live water

### 3.1 The composition order does not change

S6-6 stays: the generated shape, then terrain cell edits, then catalogue blocks, then sub-metre blocks, then attachments
(`crates/terrain/src/compose.rs:1-86`). THE STATIC FILL IS PART OF RANK 1 (the generated shape). LIVE WATER IS A RANK-2
ROW (a terrain cell edit that the shard writes). So the order gains no rank. The FUNCTION changes (§10C LAW-25): the
cell grows to three bytes, and an edited cell takes its `t` from its row, so either `compose()` carries `t`, or one fill
step after `compose()` derives `t` for the edited cells. The ORDER is S6-6's; the first round's "`compose()` stays
unchanged" is withdrawn.

The fill of an edited cell comes from its row, never from the static rule:
- a row of a solid kind: solid;
- a row of kind `Empty` (dug): the medium, also under `λ`;
- a row of a liquid kind (only the shard's flood writes one): that liquid, at the flood's level.

### 3.2 What a tool does

Every tool changes the solid: a dig writes `Empty` with its gap; a place writes a solid with its gap. The registry already
refuses a hand-placed liquid, gas, void or ore (`crates/core/src/registry/kind.rs:118-121`), so only the owning shard's
flood makes water.

### 3.3 The flood (ruling W2; no water face)

When an edit opens a cell, the owning shard runs a BOUNDED flood.
1. It starts at every liquid cell — static or live — that touches an open cell under that liquid's level.
2. It steps cell by cell through OPEN cells, DUG AND NATURAL, that stand under the level. Its neighbours are the
   extractor's own: `seam::across` at a face edge and the three-column prism at a cube corner
   (`crates/terrain/src/lattice.rs:11-21`; `crates/terrain/src/extract.rs:13-17`). So the flood never wets a cell that the
   mesh does not draw (review finding L28).
3. "UNDER THE LEVEL" is read along the realm's DOWN, from the realm's gravity FUNCTION (V4 15.2, V11), through a seam
   BELOW `GridMapping` (ruling V6 A2: "any physics anchor sits BELOW the seam"). The planet's function is the radial.
   `GridMapping::outward` is `None` on a flat grid (`crates/core/src/grid/mod.rs:200-209`), and no realm gravity FUNCTION
   (a "down" at a cell) exists in the code today (MEASURED, grep: only scalar magnitudes, such as
   `surface_gravity_mps2`, `crates/physics/src/taxonomy.rs:719`), so the flood names no grid (§5.5; §10C LAW-9).
4. It runs on the shard's COMPOSITION WORKER (§3.6), an injected seam: inline in a Tier-A test, a thread in the binary,
   as the artifact worker is (`crates/bins/src/artifact_worker.rs:1-9`). The dig's rows and the flood's rows commit
   TOGETHER when the worker answers. The dig parks meanwhile, bounded by a tuning field of its own — not
   `BlockStoreTuning::parked_ticks_max`, whose stated meaning is the stalled disk
   (`crates/sim/src/stub/built_store.rs:196-198`). The visited set is a `BTreeSet` or a `DetHashSet`
   (`crates/sim/clippy.toml:24-25`). (§10C LAW-22.)
5. It runs where the shard's profile has `block_edit` (`crates/sim/src/capability.rs:141`), never where "the realm is a
   planet" (HR3; §10C LAW-23).
6. It wets at most a stated number of cells in one tick, and at most a stated total. Both numbers live in the shard's
   tuning struct (no magic numbers); bench M9 sizes them. A flood over the per-tick number CONTINUES over the next ticks
   on the worker, so the water flows in over a few ticks.
7. It writes each wetted cell as a row: the Format B record with the liquid's kind and the cell's own `s`. A cell whose
   flood level differs from its column's `λ` also gets a level index (the chunk-row tag `FLUID_LEVELS`, a REQUIRED tag;
   SL6 ask A4; §6.3).

What follows:
- **A crater far from any water stays dry** (W2 rule 1): no liquid cell touches it.
- **A sealed room dug under the sea stays dry** until a dig opens it to water.
- **W2's own canal example holds:** a canal from a lake into a dry basin wets the basin's natural air under the lake's
  level, inside the total bound.
- **NO WATER FACE STAYS** (SL8; §10C LAW-3; question 4). A flood that would pass its TOTAL bound is refused before any row
  is written, with a typed reason (a new appended `ActionRefusal` arm; SL6 ask A8), until the digging slice designs one
  live level for a whole catchment. ARITH: a canal into the largest pocket (1.35 million km², MEASURED B7) opens about
  10¹⁵ open cells at a mean depth of 1–3 km (ASSUMED), so a flood cell by cell can never cover every case. The first
  round's "vertical face at the bound" is withdrawn.
- **TWO WATERS.** A dig that would join two waters at different levels (a canal from a lake down to the sea) is refused
  with a typed reason (A8) until W3's flowing water exists (8o). Without the refusal the canal would hold the higher
  water and end as a wall at the lower end. Bench M9's two-waters census states how often a dig can meet this.
- **A dam** (W2 rule 1): a placed wall across standing water replaces the water cells that it fills. Nothing re-solves.
  W2's river half waits for rivers (§2.5), so the first W2 fixture is the trench from the sea.
- Gate G-NO-FACE (§9): after a dig settles, 0 liquid–gas faces taller than one cell where a flood ran.

The static rule never applies to dug cells. Applying it was review finding L3: procedural first flooded every dug cell
under `λ` at once, against W2.

### 3.4 The trench from the sea, step by step

A miner stands on the belt's beach and digs a trench 2 m wide, 5 m deep and 100 m long, from the waterline inland.

1. **Her client** (an occupant of the home planet's realm) sends `WorldAction::BlockEdit` with `Empty` from the PLAYER'S
   OWN DIG INPUT (a key and a target; built first, after its own discussion, V4 row 14; §10C LAW-16), per cell or as a
   brush (`crates/wire/src/channels.rs:85-95`; SL6 row R-5 YES). `BlockEdit` places one kind at ONE cell
   (`crates/wire/src/channels.rs:135-148`); a brush is a new appended `WorldAction` arm (SL6 ask A7; question 22). Today
   the gateway counts a world action and drops it (`crates/connection-plane/src/gateway/client.rs:233-240`); slice 10
   routes it.
2. **The gateway** routes it to the planet's shard, the only writer of those cells (HR1; V4 row 8). No pose crosses (SL2).
3. **The shard** checks her reach against the pose that it holds, and checks that no Area owns the cells by ONE query of
   the planet's child index (`crates/core/src/child_index.rs`; SL9; §10C LAW-18). It composes the cells around the dig
   on its composition worker, off the tick (§3.6).
4. **It writes the rows.** Each dug cell becomes `Empty`, with a gap from the brush, so the cut stays smooth. Without the
   brush each cell takes gap +1 (`crates/terrain/src/compose.rs:6`): a stepped cut.
5. **The flood, on the same worker** (§3.3). The trench's seaward end touches the sea. Every dug cell under the sea's
   level becomes sea water (a row); every dug cell over it keeps the medium (air). A sealed pit that she dug beside the
   trench stays dry. The dig's rows and the flood's rows commit together.
6. **The commit.** Once a tick, one store transaction, durable first
   (`docs/investigation/2026-09-07/06_storage_diff_lane.md:363-395`), under the store's owner fence: its writer and its
   refusal come with this slice (`crates/sim/src/stub/built_store.rs:125-130`; §10C LAW-14). The shard folds the change
   into its edit pyramid by S5-3: per coarse cell, the mean change of `s`, and the top's MATERIAL SHARES — today the WET
   COVERAGE (the mean of the children's wet tops, a fraction and never a majority), later the surface materials of 8e.
   No dominant-solid threshold reaches the picture: a threshold on nested footprints swaps a mixed cell's colour at a
   band, the kind of fold that B16 refused (§10B.6 X6; SL6 ask A5, widened).
7. **Next tick.** `ChunkRows` go on the Bulk class with the realm's fence (`BulkFor { realm_fence, .. }`,
   `crates/wire/src/session_flow.rs:552-556`): rung-0 rows to the sessions whose windows hold those chunks, and coarse
   entries to the windows that hold coarser rungs. Each tick states, for each rung, how many parts that (tick, rung) has,
   and the mask of the rungs that the tick publishes (SL6 ask A9). Each `BulkFor` carries the rows of ONE rung and states
   that rung in a plain header beside its opaque bytes. The gateway filters per session by that header, as the owner
   ruled (*"The gateway still FILTERS per session what the realm chose to publish"*,
   `docs/design/owner_decisions_2026-09-07_voxels.md:318-324`). It never decodes a row, and it forwards one shared buffer
   per bulk message (V8 addendum item 3). So each filtered subset is whole by itself.
8. **Every client** applies a tick when every (tick, rung) of its mask that the client holds over the dig is whole (THE
   EDIT TICK, §10B.4). It lays the rows over its generated box (rank 2 over rank 1), runs the fill and extracts again,
   evicts the touched parent meshes, rebuilds every touched chunk and every finer chunk that reads a touched parent, and
   swaps the whole set in ONE frame. A row with a stale fence is refused (gate G-FENCE). The trench's water and the sea
   stand at one level, so their top is one surface (gate G-WATERTIGHT across the mouth).
9. **What each observer sees.**
   - The walker beside her sees water in the trench within budget B-EDIT (§10A.2): ARITH 140–175 ms from the click on a
     link of 30 ms each way (ASSUMED) — the link, the tick, durability, two rounds of rebuilds (the parents, then the
     children) at 3.3–11 ms a build (MEASURED: `docs/design/DEFERRED.md:7760`; ruling B18), a wait for a free worker, and
     two frames.
   - A pilot 50 km up draws rung 6 there (ARITH: a pixel is 50 000 × 0.0011506 = 57.5 m; switch(5) = 27.8 km <
     50 km ≤ switch(6) = 55.6 km). The trench is 1/29 of a pixel wide. Its wet coverage raises each crossed rung-6 cell
     by about 2/64 ≈ 3 %: at most a faint tint. A canal about 58 m wide shows as a line of water.
   - A pilot 450 km up draws rung 10 (1 024 m cells). The trench is 0.4 % of a pixel: nothing, which is correct.
10. **Collision** (slice 11) reads the same composed rung-0 cells, so the miner stands in water where she sees water.
    Buoyancy waits for 8o (R-21 NO).
11. **Stored:** ARITH 1 000 dug cells, about 500 of them wet; 1 000 × (12 B record + 3 B index) ≈ 15 KB of rows; a level
    index only for cells whose level differs from their column's `λ` (none here); about 240 coarse entries over rungs
    1–18 (S5-3 prunes a zero delta).

B21's example ("a pilot 50 km up sees a thin line of water") is true only for a canal at least one pixel wide. A 2 m
trench is a sub-pixel tint.

### 3.5 An Area, a hull, and the flood's edge

- **An Area realm** owns the planet cells inside its box (V5). Its floor is BUILT, and it gets no planet rows: the owner
  refused "the planet's diff under an area" (R-16 NO, `docs/design/owner_decisions_2026-09-07_voxels.md:227`), and the
  approved format sitting refuses any realm-to-realm chunk fetch without an ask
  (`docs/investigation/2026-09-07/topic_00_format_sitting.md:306-309`). So an Area's cells are its own rows, and its
  liquids come from rows only, as a hull's do (§5.5). A flood stops at the realm boundary before any row is written
  there (v1 refuses cross-realm edits, V4 row 8): a dig whose flood would cross it is refused like a flood past its bound
  (§3.3). The first round's ask A6 (the planet's rows under an Area's box) is WITHDRAWN (§6.4; §10C LAW-4; question 19).
  A way with no crossing exists for later: the Area's shard derives the planet's static shape itself (SL10 clause 5),
  but only after the solve's other-host leg G6 passes (§2.3). How an Area's own edit entries reach a far observer is a
  question for the Area slice. OPEN.
- **A hull or a station** has no generated water; its liquids come only from rows. The flood kernel is the same on both
  grid arms and reads "down" from the realm's gravity function (§3.3). A hull's down exists in the code only as
  `ChildFelt`, the felt acceleration that its parent states down to a driven child (SL6 R-20, YES ON TRIAL, behind the
  `felt_down` capability, default OFF; `crates/wire/src/intershard.rs:606-630`; `crates/sim/src/capability.rs:95-100`),
  or later as a graviblock. The hull's shipped water source (a tank block) is owed. OPEN; see §5.5.

### 3.6 The shard's composition budget and the client's rebuild cadence

**The shard composes cells for the first time.** Today no shard builds a chunk: `vd-sim`, `vd-node` and `vd-physics` have
no terrain edge. The flood, and later the collider, need generated cells. ARITH: one rung-0 chunk costs about 5 ms today
(the column pass with the artifact 3.2 ms, MEASURED `docs/design/DEFERRED.md:9534-9535`; the cell pass about 2 ms,
MEASURED `:7760`), and about 2 ms with the gathered read and the per-column collapse (levers E2 and E4, §10A.3; ARITH,
UNMEASURED), against a 20 ms tick at 50 Hz (`crates/bins/src/lib.rs:446`). So:
- compositions and the flood run on the COMPOSITION WORKER, off the tick, like the artifact worker (§3.3 item 4);
- a flood evaluates the generator at the columns and cells that it visits, never whole chunks that it does not touch;
- the box cache is bounded in the shard's tuning struct (ARITH worst case without a bound: 128 players × 27 boxes ×
  0.77 MB ≈ 2.7 GB);
- the storage design's load bench M-1 runs before slice 10 (§10), and slice 10 opens with that load harness (V8
  addendum item 2).

**The client's rebuild cadence.** A built chunk keeps only its mesh, so each edit re-runs the generator for the chunk and
its halo neighbours. ARITH at M-1's load (100 builders at 2 edits a second, 128 observers,
`docs/investigation/2026-09-07/06_storage_diff_lane.md:780`): 200–500 rebuilds a second × 3.3–6.5 ms = 0.7–3.3
core-seconds each second for each observer — up to half of the seven workers. So:
- the client applies the FIRST whole edit tick of a burst at once, and joins the next ticks of the burst into one
  rebuild set per chunk inside a join window in the client's tuning (THE EDIT TICK, §10B.4);
- a chunk that holds edits keeps its generated box in a bounded cache, so a rebuild runs only compose, fill and the
  RANGED extraction (lever E5);
- edit rebuilds ride the head of the urgent class, never behind the moving eye's queue;
- the gates of S4: M-1; G-EDIT (budget B-EDIT, §10A.2: the digger ≤ 180 ms p99 from the click to the pixel on a link of
  30 ms each way, a pilot 50 km up ≤ 250 ms); and edit rebuilds use at most ONE of the seven workers under M-1's load.

---

## 4. The ladder

### 4.1 Two rules, and why not one

B21 asks that every distance be a copy made from the same volume by one rule. That is possible only where the fold is
affordable.

- **Why not at every rung.** ARITH: rung 0 of the home planet holds 6 × 9 961 472² = 5.95 × 10¹⁴ columns. One byte
  each is 595 TB, and one pass over them at 0.84 µs is 15.8 core-years. An exact fold of one rung-13 cell alone reads
  4¹³ = 67 million rung-0 columns. The mean of the fine octaves over a cell has no closed form, because a cell of the
  bent cube is not a box in the noise's space (`crates/seed/src/bend.rs:1-46`). So under the reference rung the
  unedited ground is the generator at the rung's own cell, as S5-3 rules.
- **Where the fold is affordable.** On the home planet, from rung 10 up the ground holds no fine octave: the survival
  rule keeps only the coarse octaves (`crates/terrain/src/body.rs:143-308`), and `Z` replaces those. There is no bench
  from rung 5 up, and there are no caves from rung 4 up. So the drawn ground is the solved field alone, and the mean of
  that field over a cell is a fixed linear read of the rows (§4.3).

So:
- **Rungs 0 to `L_ref`: the generator at the rung** — the solid pass with the rung's live octaves, the fill, the
  extractor.
- **Rungs above `L_ref`: THE FAR FOLD** — each cell is the exact mean of the rung-`L_ref` cells under it.
- **Edits at every rung:** the S5-3 delta fold, with the wet coverage (§3.4).

**THE REFERENCE RUNG `L_ref`** is the finest rung whose ground holds no live fine octave: the solved field alone. It
comes from the body's own table and is never typed. Home planet at today's survival constant: `L_ref = 10`. With the
smaller constants of question 2: 12 (K = 8) or 13 (K = 4). A body with no such rung has no far fold, and the generator
draws every rung.

### 4.2 ONE FIELD at every rung

Today rungs 10–14 read pyramid level 1, a floored 2 × 2 mean (`crates/terrain/src/artifact.rs:849-876`, `:1695`), and
rungs 15–18 read coarser means: a second field. MEASURED: the ground moves up to 1 442.5 m (1.41 cells of rung 10) at the
9 → 10 swap (W17), and before the coast mask the coast moved a median 11.5 km there
(`crates/recipe/src/height.rs:393-397`). W17's cure pushed that swap from 445 km to 704 km and cost 8 % of a frame
(MEASURED: chunks 7 304 → 8 926, frame 23.16 → 25.12 ms at the 20 km stand;
`docs/design/owner_decisions_2026-09-21_water.md:1446-1453`).

In this design:
- **Rungs 0 to 13 read the ROWS** (the C1 spline over the 8 192 m nodes, `crates/terrain/src/artifact.rs:1811-2012`).
  At rung 13 a cell is a node. So the tiles reach out to rung 13's ring, bounded by the horizon (SL6 ask A2). ARITH (the
  scale review's figure, re-derived here): the worst eye stands about 3 193 km up, where the horizon equals rung 13's
  switch distance of 7 120 km. The visible cap holds 84.6 million km²: about 1.49 million nodes of 56.96 km² on average,
  about 363 tiles, about 14.9 MB. At 1 MiB a half-second beat (`crates/connection-plane/src/gateway/artifact.rs:40-50`)
  that is about 7 s. The coarser rung stands until the tiles arrive (F9).
- **Rungs 14 to 18 read THE FAR MAP** (§4.3), shipped with the head for the whole body.
- **Deleted:** the mean pyramid and its water words, and `field_step_m` (`crates/terrain/src/artifact.rs:936-1003`).
  `tile_rung` and `switch_floor_m` stay: T7 rule 3's floor is inert once every step is under one cell. Every hand-over's
  step is then only the octaves that the coarser rung drops, which T7 keeps under one cell of the rung that takes over
  (MEASURED 0.64 px; `crates/terrain/src/body.rs:190-194`). At 9 → 10 that bound is 897.7 m, 0.88 of a rung-10 cell
  (DOC, `crates/client/src/ladder_view.rs:225-234`).
- **The gateway's tile cache** gets an eviction bound in its tuning. ARITH without one: 2 166 tiles × 40 960 B = 88.7 MB
  per planet.
- **The pace moves into the gateway's tuning.** `ARTIFACT_BYTES_PER_BEAT` is an inline constant
  (`crates/connection-plane/src/gateway/artifact.rs:50`) against the gateway's own rule, "ONE reviewed struct, no inline
  literals" (`crates/connection-plane/src/gateway/config.rs:22`), and the "about 7 s" above rests on it. It becomes a
  field of `TransportTuning`; bench M7 states its value (§10C LAW-11).
- **THE SERVE ORDER** (lever E12, in S3 with the far map; §10A.3). The gateway serves each session the coarsest far
  levels and the near tiles first, and a far body only to the far-map level that its drawn rungs read (the gateway alone
  knows each session's angular size of a realm, `docs/investigation/2026-09-07/06_storage_diff_lane.md:506-507`). It is
  a per-session filter and pace on what the realm published, never a clamp on what the realm publishes (the owner's
  answer, `docs/design/owner_decisions_2026-09-07_voxels.md:318-324`). No new data crosses (SL6).
- **A CHUNK WAITS FOR ITS PARENTS' FIELDS** (§10B.4, [PARENTS]). A chunk builds only when the field of every parent that
  it reads is whole (`crates/client/src/chunks.rs:1106-1141`). Today a vertex with no row under its radial keeps its own
  radius (`crates/client/src/chunks.rs:1366-1372`), and a tile's arrival never rebuilds the chunk (the head's digest does
  not change), so that vertex never gets its morph (§10B.6 X8). The coarser rung stands whole meanwhile (F9).

### 4.3 THE FAR FOLD

**The rule.** Above `L_ref`, a cell's TOP is the mean of the tops of the rung-`L_ref` cells under it, and its WET SHARE
is the share of those cells whose top is wet. One rule — the mean — at every far rung, for ground and water alike, and
later for every surface material of 8e.

**How it is computed.** Every level is computed from the rung-`L_ref` cells directly and rounded ONCE (never a mean of
rounded means), so a parent is the mean of its children to the rounding.
- A far cell whose reference samples are all dry, or all wet at one level, has a closed form: the mean of point samples
  of the spline is a fixed weighting of the nodes around the cell — a stencil of about 5 × 5 nodes (ARITH), one table
  per rung and phase. A bound of the stencil's heights against the levels decides which cells are uniform.
- A MIXED far cell (a coast or a divide inside it) reads its reference samples one by one: 4 at rung `L_ref + 1`, and up
  to 64 at rung 13 on the home planet.
- **The client** computes rungs `L_ref + 1` to 13 from the rows that it holds, through the box's gathered node window
  (lever E2). Its cost is UNMEASURED (bench M6). ARITH: a uniform cell costs about one spline read — 102–162 ns through
  today's per-column read (MEASURED, `docs/design/DEFERRED.md:9534`), about 10–40 ns through the window (ASSUMED) — so
  0.04–0.6 ms for the uniform cells of a chunk. A mixed cell reads up to 64 reference samples, and each sample's top
  needs `Z` and the descent step's gradient: about five spline reads, at least 50 ns (at the ASSUMED 10 ns a read). So
  a coastal rung-13 chunk with a 10 % mixed share (ASSUMED, about 410 of its 4 096 columns) costs at least 1.3 ms for
  its fold before its geometry (ARITH: 410 × 64 samples × 50 ns = 1.31 ms, plus the uniform cells; the first round
  priced it at under 0.7 ms, with 25 ns a sample).
- **The shard** computes rungs 14 to 18 once, at the bake, and ships them as THE FAR MAP: per far cell the mean top (i16
  metres) and the wet share (one byte). ARITH: 2 954 424 far cells (16 km to 262 km) × 3 B = 8.9 MB, in place of today's
  11.8 MB pyramid and 2.2 MB coast mask — about ONE BYTE PER MACRO NODE (8 871 936 nodes), the rule that sizes every
  planet's far map (budget B-LOGIN). The bake: ARITH 5.68 × 10⁸ rung-10 cells × 0.1–0.8 µs = 1–8 core-minutes, less
  where the stencil answers, on §2.3's fixed partition with an ordered merge (gate G-SPLIT).
- **The far map ships COARSEST RUNG FIRST, in its own parts** (§10B.4, [EPOCH]; §10C LAW-8). Rung 18 is 8 664 cells ×
  3 B ≈ 26 KB (ARITH: 6 × 38² × 3 B), so it lands in the head's first beat, and a body's first frame holds all six
  top-rung chunks at once — never a globe with a face missing. The client draws each far rung when its part lands and
  fades the next rung in across its band (gate G-FAR-FIRST).
- **The mesh of a far rung** comes from the same extractor: each far cell's top byte is its gap to the folded top, and
  each vertex's wet share is the mean of its group's folded shares. So a far rung is a height field of mean tops, with
  no overhang and no cave.

**What the far fold guarantees** (gate G-FAR-EXACT, in two parts; §10C LAW-7):
- PER CELL: a parent's top and share are the mean of its children's, to the rounding (0 misses beyond 1 m and 1/255).
- PER PIXEL: no share step at a band's far edge (the wet morph, §4.4), and a monotone change inside the band.
- Above `L_ref`, a coast holds its place within its one-cell soft edge at every swap.
- No lake appears, vanishes or flickers at a far swap. A lake smaller than a cell is a tint at its exact share. As the
  pilot climbs, the pixel grows, and the tint fades by the lake's true share of the pixel. This is B18 Q1's far look,
  made by one rule.
- Each far cell stands at the mean height of what a pilot sees over it. This is B18 Q2's lid, now the fold itself. The
  dry word (B18 Q3) is not needed, because the mean of the land's own tops IS the land's height.

**Where the far fold is not exact** (stated):
- The share is exact per CELL, not per pixel. At a swap, 2 × 2 cells become one, so a pixel shows its parent's share: a
  monotone blur inside two pixels, never a flicker. Pixel-exact windows cost 17–79 MB per stand plus an 11.3 MB page
  (MEASURED, B17). This design retires B17's windows; that is a trade the owner rules (question 20), not a silent loss.
- The rounding: at most one metre of height and 1/255 of share, once.

**Why the far fold is not a special water rule.** It reads no side word and no water word of its own. It averages the one
top and the one wet test of the reference rung, the same for water and ground.

**Example.** A pilot climbs over the belt's high lakes from 500 km to 60 000 km. A tarn 1.5 km across covers 44 % of one
rung-11 cell (2 × 2 km), so at 890 km that cell is drawn 44 % blue. Its parent at rung 12 is 11 % blue, and at rung 13,
3 % (ARITH: π × 0.75² / 4, / 16, / 64). The pilot sees the tarn fade as it shrinks under a pixel. It never goes and comes
back. Today B15 MEASURED 53 of 714 far lake bodies flickering at rung 14.

### 4.4 The seams between rungs

**Unchanged tools:** the crossfade band at 0.9–1.1 of the switch distance, the finer rung's morph onto the coarser mesh,
the sink, the skirts, the band table and the 0.5 s arrival ramp (`crates/client/src/ladder_view.rs:36-54`, `:143-160`;
`crates/client/src/arrival.rs`; `crates/client-render/src/ladder_fade.wgsl:140-198`). They read distances and bounds,
never water. §10B.2 lists what each one guarantees and the measurement behind it.

**Re-derived sizes:**
- The sink and the skirts read the rung's step bound (`crates/client/src/chunks.rs:57-77`). Under the centring a dropped
  octave varies around zero, and the field swap is gone, so the bound is the dropped octaves' own amplitude.
- **THE FAR SINK** (§10B.4, [FAR SINK]). Above `L_ref` no octave drops, so today's formula gives two cells
  (`chunks.rs:69-77`). The fold's own step is the top's variation inside one coarser cell: if bench B-S6 reads it under
  one cell, the two cells stay; otherwise the body states that step for each far rung, and the sink and the skirts read
  it.

**What the second round adds** (§10B.4 holds each one with its evidence; §10B.6 the defect that each one cures):
- **TOP ONTO TOP** ([TOP]). The parent mesh that the morph reads is the coarser rung's TOP mesh after the flat merge,
  with one share byte a vertex. Every fallback reads the coarser TOP, `max(ground, λ)`: `coarser_field_m`
  (`crates/client/src/chunks.rs:1247-1261`), `field_normal` (`:1175-1196`) and the face-seam rule (`:1325-1341`). Today
  they read `height_field_m`, the GROUND (`crates/terrain/src/height.rs:115-122`); in the one mesh a water vertex on a
  cube-face seam would morph down to the sea floor inside every band (X1). Over open water both tops stand at the level,
  so the morph is nought there.
- **THE WET MORPH AND THE WET RAMP** ([WET MORPH]). Each vertex carries its own share and a TARGET share: the parent's
  share at the radial's hit, weighted as the shade is weighted there (`chunks.rs:713-717`), or the coarser top's wet
  test where a field answers. The vertex stage blends `share = mix(target, own, whole(d) · ramp)`, as it blends the
  shade (`ladder_fade.wgsl:180`). So at the far edge the finer share IS the coarser's, and a late chunk starts with its
  stand-in's colour; the probe reads the same blend. This cures today's near-shore pop (🟥,
  `docs/design/DEFERRED.md:10248-10252`; X2). One rule serves every surface material that 8e adds.
- **THE FACE SEAM READS THE NEIGHBOUR FACE'S PARENT** ([SEAM TARGET]). A vertex on a cube-face edge reads its target —
  position AND share — from the neighbour face's parent mesh (the extractor already maps cross-face sites): the exact
  cure that DEFERRED names (`docs/design/DEFERRED.md:7910-7913`). Today it reads the coarser FIELD from both faces, a
  crease along the twelve cube edges; with the wet share, its binary wet test would stand against the coarser rung's
  fractional share and step at the far edge. 223 coastal node pairs straddle a cube edge, and one cube corner is coastal
  (MEASURED, `docs/investigation/2026-09-24/terrain_believability_status.md:1009`).
- **A REBUILT KEY KEEPS ITS RAMP** ([STAMP]). A rebuilt key inherits the arrival stamp of the entity that it replaces
  while that ramp runs. Today a rebuild enters whole (`crates/client/src/arrival.rs:42-51`;
  `crates/client-render/src/terrain.rs:3215-3225`), so a chunk rebuilt 0.2 s after it landed jumps (X4).
- **HYSTERESIS AT EVERY DECISION** ([HYSTERESIS]). The want and the split at a band's edge get a keep margin of one
  column span of the rung; today they have none (`crates/client/src/ladder_view.rs:1499-1515`), so an eye that sways on
  the edge asks, releases and asks again (X15).
- **THE SKIRTS IN THE ONE MESH** ([SKIRT SHARES]). A skirt copies its edge's two shares, as it copies its normals and its
  morph (`chunks.rs:1771-1792`).

**The rule for every speed lever** (§10B.5): a lever may make the picture coarser or later; it may never make it
discontinuous. It changes no world byte, moves no drawn band in one frame, removes no stand-in, and takes no frame over
the period.

### 4.5 THE FAR COAST: what differs, where, and the owner's choice

| What | Rungs 0 to `L_ref` (0–10 today; the last slide is the 9 → 10 band at about 450 km) | Above `L_ref` (the far fold) |
|---|---|---|
| The water's level | The same number at every rung. | The same. |
| Which water a column holds | The same at every rung: the descent step reads one field. | The fold of the reference rung. |
| The ground's detail | Each rung keeps fewer small hills. The step is at most one cell of the rung that takes over (T7; MEASURED 0.64 px). | The exact mean of the reference rung. |
| The coast | Slides across each band by the dropped relief over the coast's slope. New numbers UNMEASURED (bench M4). Old MEASURED medians, before W6 and with the lift: 234 m at 6 → 7 and 848 m at 7 → 8 (`crates/recipe/src/height.rs:359-366`), which is 3.7 px and 6.6 px at the finer cell (ARITH 234/64, 848/128). At 9 → 10: 11.5 km, 22 px at 445 km, but that number includes the field swap that §4.2 deletes. | Holds within its one-cell soft edge. |
| Water under about two cells | Can fade out and back in across successive bands (a pond made by one octave). Count UNMEASURED (bench M4). | A tint at its exact share; it fades as the pixel grows and never comes back. |
| The colour | The vertex's wet share at the rung. | The fold's share. |

**THE OWNER'S LATER WORDS** (*"no jumps or flicker between rungs"*), read with the rulings (§10B.7):
- A gradual, one-way slide is neither a jump nor a flicker. B18 Q5 read the same words so ("no rung decides it", a
  gradual change). B18 Q4 accepted near slides of at most 1.46 px (the sea) and 3.29 and 4.79 px (lakes), "judged in the
  flight" — sizes MEASURED under N2, which this design retires. B16 says "No flicker or changing of the shapes between
  rungs at all". So the new slides need their own numbers (bench M4) and a stated line (gate G-SLIDE; question 1).
- A pond that fades out at one band and back in at the next is a FLICKER: B16 item 1 names "show, hide and show", and
  W16 item 2 says that a finer rung never contradicts a coarser one. So under (B) such ponds are allowed only where gate
  G-POND-FLICKER reads 0 on the stands and the planet sample, or on the owner's amendment of B16 (question 1). A smaller
  K (question 2) makes them rarer, not zero; option (A) removes them.

**THE TWO SIDES OF QUESTION 1**, each with its price:

**(A) EXACT AT EVERY DISTANCE** (the stored-truth design, with the scale review's corrected prices):
- A new fine ground, THE CONSERVATIVE SYNTHESIS: each rung's four children average exactly to their parent, from the
  solve's 8 km field down to 1 m. The look changes, and every judged picture is judged again. Its zero-sum rule must
  spread the remainder by a hash of the block's address (review finding L18), and G-GRID and G-SPECTRUM must pass on a
  bench first.
- A stored water tree at 32 m: which fine column belongs to which water, and its folds. The bake: 21–112 core-minutes
  (ARITH, corrected for the rung-0 count, review finding C2), with a 0.7–7 GB working set. Stored: 57–150 MB per planet
  (ARITH), on a count of mixed cells that B14 measured in the N2 world (review finding C8). With the head: 8.8–9.5 MB
  more, and the tiles to rung 13 as here.
- The client derives the folds at rungs 6–9: 0.5–3 s a ring near a coast (ARITH, corrected).
- Pixel-exact colour at a swap: share windows, 17–79 MB per stand plus an 11.3 MB page (MEASURED, B17).
- Still not exact: under 32 m a local rule, with a water step of up to 1.19 m in rare places; the bench at rungs 0–4;
  the caves.

**(B) EXACT FROM FAR, SMOOTH NEAR** (this design):
- Above `L_ref` (about 450 km out today): the far fold. Nothing slides and nothing flickers.
- Nearer: the slides of the table above, gradual and one-way inside each band (the wet morph, §4.4). Price: their size,
  measured on the four stands before the owner decides (bench M4), against a stated line (gate G-SLIDE). Question 1
  offers two lines: B18 Q4's accepted sizes (1.46 px for the sea, 4.79 px for lakes) as the most that a coast may slide
  at any band, or the stricter line of W17.1 and T7 (p99 ≤ 1 px, max ≤ 2 px per band, at the finer cell). The lever of
  question 2 shrinks them.
- Ponds that fade out and back in: gate G-POND-FLICKER at 0, or the owner's amendment of B16 (above).
- Data: the tiles to rung 13 (≤ about 14.9 MB at the worst height, ARITH) and the far map (8.9 MB, replacing 14.0 MB).

**RECOMMENDATION: (B).** It keeps today's judged ground (with the centring), stores nothing new per planet, and meets
B16 wherever the octave sum allows — above `L_ref` always, below it where G-POND-FLICKER reads 0. If the owner rejects
the measured near slides or the pond count, (A) comes ON TOP of (B): it replaces the fine ground and adds the shore
tree; the cell, the fill, the extractor, the edits and the far fold stay. The tension, stated: requirement 1 favours
(B), which stores nothing new for a planet; B16 and W16 favour (A) nearer than about 450 km. The bench prices both
before the owner rules.

### 4.6 The survival constant K (question 2)

Today an octave lives at rung L while its wavelength (twice it for a ridged octave) covers K = 48.83 cells
(`crates/terrain/src/body.rs:182-205`, `:226-233`; ruling T7 rule 1). The step of a hand-over is the dropped octave's
amplitude, measured in cells of the rung that takes over. A smaller K keeps each octave longer and drops it where the
cells are bigger, so the step is a smaller share of a cell. ARITH for the 3 125 m crest (198.8 m): at K = 48.83 it drops
at 7 → 8, 0.78 of a 256 m cell; at K = 4 its reach of 6 250 m still covers 4 cells at rung 10 (1 024 m cells, a threshold
of 4 096 m) and it drops at 10 → 11, 0.097 of a 2 048 m cell — about 8 times smaller. In general the drop is about
48.83 / K times smaller in cells (ARITH: 12 at K = 4, 6 at K = 8). The comment at `crates/terrain/src/body.rs:195-202`
says that a typed 4 "makes that step many cells high"; that does not match this arithmetic, and the comment marks its
own number UNMEASURED.

The prices: a few more octaves per column (ARITH: about three more, at 12.19 ns a noise call, MEASURED
`docs/investigation/2026-09-07/00_proposed_voxel_foundation.md:379`, against 839 ns a column); a ridged octave at K = 4
lives down to two cells a wavelength, where its kink is sampled coarsely (so the bench also runs K = 8); and `L_ref` moves
up to 12 or 13, so the far fold starts at about 3 500 or 7 000 km. T7 is the owner's ruling, so the bench only measures
(bench M5).

---

## 5. The far view and other bodies

### 5.1 From orbit
Rungs 14–18 read the far map from the head; no tile is needed. Where a far cell's share is one, the sea is the flat top at
its level, drawn on the rung's lattice within the chord bound. A coast is a one-cell soft edge at its exact share.
**Example.** A pilot 20 000 km over the home planet sees the continents' coasts at their true places and the high lakes as
faint tints. When she dives, each ring swaps inside its band, and the coasts do not move until the last 450 km.

### 5.2 Another planet
Its own shard, solve, level map, rim pass and far map. No far chunk waits for a coast mask any more
(`crates/client/src/artifact_book.rs:158-192` is deleted): the far map carries everything that the far rungs read.
THE ARTIFACT EPOCH (§10B.4, [EPOCH]):
1. A body with a macro lattice draws nothing until its head and the top rungs of its far map land (DEFERRED's road (a),
   `docs/design/DEFERRED.md:9129-9133`; question 17), so no chunk builds on the recipe's relief and pops away later.
2. The far map ships coarsest first. Rung 18 (26 KB) lands in the head's first beat, and the body's first frame holds all
   six top-rung chunks at once. Each finer far rung fades in across its band as its part lands. Gate G-FAR-FIRST: from
   the first far part HELD by the client to the body's first drawn frame, at most one beat (0.5 s); 0 frames with a held
   first part and no drawn rung.
3. An artifact change under a drawn body (development only: a new world) keeps the whole old set drawn until the new
   set's band is whole, then swaps the realm in one frame (gate G-EPOCH). Today the swap goes key by key, and old and new
   ground stand side by side until the last new chunk lands (MEASURED, `docs/design/DEFERRED.md:9137-9145`).
4. A STATED RESIDUE: a body enters a window at its reach — by the size rule 1.5°, about 23 px (ARITH: 0.02618 ÷
   0.0011506; `crates/core/src/geometry.rs:1140-1146`) — and its artifact is served only after that: at least one beat,
   and 77–382 s when its shard solves for the first time (MEASURED, `docs/design/DEFERRED.md:9137-9139`). That pop-in
   belongs to the REACH, not to a rung. Its cure is the reach ruling's brightness radius ("A planet is visible by
   brightness far beyond the distance at which its size makes a dot",
   `docs/design/owner_decisions_2026-09-02_reach.md:79-81`); the first solve's wait is budget B-BAKE (§10A.2). The
   DEFERRED entry of the far planets' pop-in (`docs/design/DEFERRED.md:8937-8947`) keeps this residue.

### 5.3 A moon with no air
No water inventory: every level-map word is DRY, and no body holds a liquid. The medium is `Empty`, so its ground and its
caves hold vacuum. The far fold is the fold of its ground. **Example.** An astronaut on the home moon digs into a crater
wall. The dug cells hold vacuum, and the flood never runs.

### 5.4 A gas giant
No solid-surface flag (`crates/core/src/look.rs:170`): no solid pass, no cells, nothing extracted. The sky draws it (8s).

### 5.5 A hull or a station that holds terrain (V4 15.2)
The flat arm of the grid. There is no generator: every solid cell is a row. The medium is the realm's own (`Empty` in a
hull without life support; a sealed room's air is a later life-support feature). Liquids come only from rows. "Under the
level" means deeper along the realm's stated gravity function than the free surface (V4 15.2, added), read through the
seam below `GridMapping` (§3.3); with no gravity there is no free surface. The fill, the flood and the extractor are the
same kernels on both arms. A hull's "down" exists in the code only as `ChildFelt`, the felt acceleration that its parent
states down to a driven child (SL6 R-20, YES ON TRIAL, `felt_down` default OFF; §3.5), or later as a graviblock.
**OPEN:** the hull's shipped water source (a tank block) does not exist. So this design does NOT claim HR4 for a hull
yet (§10C LAW-9): G-IDENTICAL runs the dig and the flood on the planet's shard and on an Area's shard (V5: an Area uses
the planet's grid and gravity), the Area fixture GIVEN its source row (an Area gets no planet rows, §3.5). The hull's
flat arm joins the gate when its down and its source exist, and the owner's word on the source comes with the hull
slice.
**Example.** A builder digs a tank into a hull's terrain cells beside a source row. The same flood kernel wets the tank
along the hull's graviblock down, as it wets a trench on the planet along the planet's down.

### 5.6 An Area realm
See §3.5.

---

## 6. Server and client

### 6.1 Who computes what

| Host | Computes | Holds |
|---|---|---|
| The planet's shard | Once per body, off the tick: the solve (205 s MEASURED, `docs/design/DEFERRED.md:9138-9139`), the level map, the rim pass, the far map. Per edit: the check, the rows, the flood, the S5-3 fold. Later: the rung-0 collider from the composed cells. | The artifact (rows and far map), the edit rows, the edit pyramid. |
| The gateway | Nothing new in kind. It caches each realm's artifact parts (now with an eviction bound), paces them at 1 MiB a half-second beat per client (a field of its tuning struct, §4.2), serves the coarsest far levels and the near tiles first (lever E12), and filters chunk rows per session by the plain rung header of each `BulkFor` (the owner's answer: *"The gateway still FILTERS per session what the realm chose to publish"*). It never decodes a row, and it forwards one shared buffer per bulk message (V8 addendum item 3). | Artifact parts. |
| The client | Every chunk: the solid pass, the rows, the fill, the extractor, the wet share, the flat merge. The far fold at rungs `L_ref + 1` to 13. The S5-3 fold of the diffs that it holds (V4 row 7). | Meshes, the tiles in reach, the far map. |

The client never solves, never floods and never states a pose. Its artifact book forgets a realm on the ladder's grace
(one interpolation buffer), never on one scene delta as today (`crates/client/src/net.rs:375-378`): a one-tick absence
at a realm hand-over would cost the head's far data (8.9 MB) and the tiles again, and a band gap meanwhile (§10B.4,
[BOOK GRACE]; §10B.6 X16).

**The card.** Today the card builds no solved chunk: "A job with an artifact is the CPU builders' alone: the card holds no
Z" (`crates/client/src/chunks.rs:343-348`; the code names the state "ruling F9, the card builder parked",
`crates/client-render/src/terrain.rs:1343-1346`). Once each box request carries its gathered node window (lever E2), the
card COULD build solved chunks again from the same recipe source (`crates/recipe-gpu`, one crate, two compilations).
Whether it should is ruling F9's open owner question (question 12), not a broken law: MEASURED on cured code, the card
built 95 boxes a second beside three workers and cut the 528 m/s leg's worst gap from 144 to 44 urgent chunks, but it
raised the pop detector's widest step on four legs of five and took the slow hull's band from holding to 3 357 urgent
chunks (`docs/design/owner_decisions_2026-09-12_frozen_patches.md:489-503`). The CPU share stays the default; bench P12
measures first (§10A.3, E9).

### 6.2 Collision
The collider reads the same composed rung-0 cells: `s`, the blocks and the rows (SL10: the server computes collision on
the same shape; W10 item 4; collision never coarsens). Liquids are not solid. Buoyancy and drag wait for 8o (R-21 stays
NO). Nothing collides today (no terrain edge in `vd-sim`, `vd-node` or `vd-physics`); slice 11 builds the collider.

### 6.3 The wire

| Part | Today | This design |
|---|---|---|
| `ArtifactHead` | …, `sea_m`, `coast_parts` (`crates/wire/src/channels.rs:415-431`) | Unchanged. `coast_parts` is always 0 (a tombstoned field). The client refuses a head of another ARTIFACT_VERSION by name, as the store does (`crates/bins/src/artifact_store.rs:197-203`); today `accept_head` checks no version (`crates/client/src/artifact_book.rs:447-465`). |
| `ArtifactTile` | 64 × 64 rows of 10 B (40 960 B) | Unchanged format; the water word carries the level map (ARTIFACT_VERSION 10). Reach: out to rung 13's ring (SL6 A2). |
| `ArtifactPyramid` | heights and water words, levels 1–6, 11.8 MB | Tombstoned (never produced). |
| NEW `ArtifactFar` (an appended arm) | — | Per far cell of rungs 14–18: the mean top (i16) and the wet share (u8); 8.9 MB, in its own parts, COARSEST RUNG FIRST (SL6 A3; a new arm, a PROTO_MINOR bump). The gateway serves it to each session only down to the level that the session's drawn rungs read (E12), so it is never sent whole to every session at once. |
| `ArtifactCoast` | 2 217 984 B in 68 parts | Tombstoned (never produced). |
| `ChunkRows` (slice 10) | planted, no producer (`crates/wire/src/channels.rs:398-414`) | The sparse Format B cells (a wet cell has a liquid kind). A NEW REQUIRED tag `FLUID_LEVELS` (8) for cells whose level differs from their column's `λ` (SL6 A4). The `PYRAMID` tag's entries gain the top's material shares, the wet coverage first (SL6 A5). Each tick states its part count for each rung, and its rung mask (SL6 A9). |
| `BulkFor` (slice 10) | planted, no producer (`crates/wire/src/session_flow.rs:544-556`) | One rung's rows per message, the rung in a plain header beside the opaque bytes, so the gateway filters without decoding (SL6 A9; a new appended arm, never a field on the planted one). |
| `WorldAction`, `ActionRefusal` | `BlockEdit` at one cell; five refusal reasons (`crates/wire/src/channels.rs:85-173`) | A brush arm if question 22 picks it (SL6 A7). Two refusal reasons — a flood past its bound, two waters joined — if question 4 picks the refusals (SL6 A8). Both appended. |

A new arm, not a re-used one: an old peer would read a changed field's old meaning (the standing rule: never repurpose a
message). The chunk row's seven tags are numbered once and never reused (`crates/core/src/chunk_row.rs:31-62`), so the new
tag is 8. It is REQUIRED (§10C LAW-13): a reader that skipped it would draw a flooded cell at its column's level, or as
gas — a durable fact read as a default — so an older reader refuses the row (`crates/core/src/tlv.rs:19-22`; gate
G-REQUIRED).

**ONE WORLD EPOCH** (§10C LAW-5). S1 gives the row's `water_m` a new meaning and deletes the coast mask; the shore law that
reads both goes only at S2; the far view gets its new field only at S3. Without the pyramid, `level_for` sends every
rung to the rows and `tile_rung` climbs to the top rung (`crates/terrain/src/artifact.rs:857-859`, `:939-945`), so the
tiles would grow to the whole visible body (up to 2 166 tiles, 88.7 MB, about 42 s at 2 MiB/s, ARITH), and rungs 10–18
would read point samples: B15's far lake flicker again. So S1, S2 and S3 land as ONE epoch: GENERATOR_VERSION 22 with
ARTIFACT_VERSION 10, and `PROTO_MINOR_FLOOR` moves to the minor that adds `ArtifactFar` (the owner's posture on this
class: "raise the floor, we don't have old clients", `crates/wire/src/version.rs:375-376`). An old client passes the
world hello today (the declared tag folds only the generator version and the seed, `crates/terrain/src/tag.rs:237-242`);
after the epoch it is refused by name (gate G-VERSION).

### 6.4 SL6 asks — one table for the whole design (default NO; each says why the receiver cannot compute it)

| Ask | What; from which realm to which; the bytes | Why the receiver cannot compute it | Without it | State |
|---|---|---|---|---|
| A1 | The row's water word as the level map; the planet → its observers; 0 new bytes. | It needs the whole solve (205 s, MEASURED) and the rim pass. | A dry node cannot know the lake beside it: node-square water (B8). | ASKED, with the one-epoch version guard (§6.3). |
| A2 | Tiles out to rung 13's ring; the planet → its observers; ≤ about 14.9 MB at about 3 193 km (ARITH). | The rows are the solve's output. | Rungs 10–13 read a second field; the 9 → 10 swap moves the ground up to 1 442.5 m (MEASURED W17.1). | ASKED; the pace moves into the gateway's tuning; the coarser rung stands whole meanwhile (F9). |
| A3 | `ArtifactFar`, a new appended arm; the planet → its observers; 8.9 MB in place of 14.0 MB (ARITH), coarsest rung first. | It needs every row of the body (88.72 MB, ARITH §7). | Rungs 14–18 have no field, or a second far field, and far lakes flicker (53 of 714, MEASURED B15). | ASKED. |
| A4 | `FLUID_LEVELS`, chunk-row tag 8, REQUIRED; the owning realm → its observers, inside R-3's rows; 1 B per wet cell off its column's level (DOC §7). | Live water is live state (W2). | A flood at another level cannot be drawn. | ASKED, with the edit slice. |
| A5 | The top's material shares in a `PYRAMID` entry — the wet coverage now, the surface materials with 8e; the owning realm → its far observers; about 1 B per share per coarse entry (ARITH). | A far client holds no rung-0 rows; a near client folds its own (V4 row 7). | Sub-pixel live water shows no tint far away; a dominant-solid threshold swaps a mixed cell's colour at a band (B16). | ASKED, with the edit slice; it amends S5-3's entry (§10C.5). |
| A6 | The planet's rows and levels under an Area's box; the planet → the Area, shard to shard. | — | — | WITHDRAWN: R-16 is NO ("an area's floor is built"); SL3 (§3.5; question 19). |
| A7 | A brush dig: a new appended `WorldAction` arm (a centre, a radius, a kind); the client → the gateway → the owning realm; about 30 B an action (ARITH). | The brush's shape is the player's intent. | A dig is per cell with gap +1: a stepped cut and about 1 000 actions for the trench (ARITH). | ASKED (question 22). |
| A8 | Two new appended `ActionRefusal` reasons (a flood past its bound; two waters joined); the owning realm → the client; 1 B. | The refusal is the realm's decision. | The player does not learn why a dig failed. | ASKED ONLY IF question 4 picks the refusals. |
| A9 | THE EDIT TICK's counts: for each (tick, rung), its part count, and the tick's rung mask (32 bits), in `ChunkRows`; and a plain rung header on a new appended `BulkFor` arm (shard → gateway); the owning realm → its observers; a few bytes a tick (ARITH). | Which parts a tick has is the realm's own fact; a `ChunkRow` already carries its tick (`crates/core/src/chunk_row.rs:8-10`), not its count. | A client cannot tell a whole tick from a part of one, so the rungs of one dig land in different frames and a finer chunk morphs onto a coarser surface that no longer stands (§10B.6 X5); the gateway cannot filter by rung without decoding. | ASKED, with the edit slice. |
| No ask | The coast arm and the pyramid arm go (tombstoned). R-21 (buoyancy up) stays NO. The serve order (E12) is a pace and a filter on approved parts. `ArtifactHeld` keeps its meaning only while the artifact's digest folds the far map (a CHECK in gate G-VERSION's fixture). | — | — | — |

### 6.5 The laws, checked

Replaced by §10C, the full compliance table: every hard rule, standing law, convention and ruling, each with its
verdict and its evidence, over the merged design (the model, the speed levers and the seam mechanisms). The first round's
short list here omitted A6's missing arm, the missing "down", the red coverage, the version gap and the card's state.

---

## 7. Costs on the home planet

(R = 6 341 670 m, N = 9 961 472, 19 rungs, 8 871 936 nodes: `crates/seed/src/ladder.rs:232-273`,
`crates/terrain/src/macro_lattice.rs:21-25`.)

| Item | Number | Mark and source | Budget (§10A.2) |
|---|---|---|---|
| Rung-0 columns | 5.95 × 10¹⁴ | ARITH, 6 × N² | — |
| One pass over them | 15.8 core-years at 0.84 µs | ARITH | — |
| **Server, once per body and world version** | | | |
| The solve | 205 s | MEASURED, `docs/design/DEFERRED.md:9138-9139` | — |
| The level map walk | under 1 s (8 871 936 steps) | ARITH | — |
| The rim pass | 2–84 core-minutes | ARITH on ASSUMED inputs (§2.3); UNMEASURED | B-BAKE: the added wall time ≤ the owner's ceiling (question 21) |
| The far map bake | 1–8 core-minutes | ARITH (5.68 × 10⁸ rung-10 cells × 0.1–0.8 µs) | B-BAKE |
| Solve memory | per-node vectors about 727 MB before the lakes; the rim pass adds its boundary lists | ARITH (reader); UNMEASURED | — |
| **Stored per planet** | | | |
| Artifact today | 102.8 MB (rows 88.72 + pyramid 5.91 + water words 5.91 + mask 2.22) | ARITH; the MEASURED 94.63 MB (`docs/design/DEFERRED.md:9167`) is ARTIFACT_VERSION 5 | — |
| Artifact after | 97.6 MB (rows 88.72 + far map 8.86) | ARITH | — |
| **Download per client** | | | |
| Head's far data | 14.0 MB → 8.9 MB | ARITH | B-LOGIN: every planet whole at its drawn level ≤ 30 s |
| Tiles | today to rung 9's ring (91 tiles at the 20 km stand after W17, MEASURED W17); after: to rung 13's ring, up to about 363 tiles ≈ 14.9 MB at about 3 193 km | ARITH (scale review, re-derived) | B-WIRE: the fill ≤ 2 MiB/s |
| Delivery at the worst height | about 7 s at 1 MiB a half-second beat | ARITH | B-WIRE (the pace a field of `TransportTuning`, §4.2); the coarser rung stands whole meanwhile (F9) |
| Gateway tile cache | bounded (value UNMEASURED); unbounded it tends to 88.7 MB per planet | ARITH | a bound in the gateway's tuning |
| **Client, per chunk** | | | |
| Build box | 512 KiB → 768 KiB (+256 KiB a build; about 1.8 MiB on 7 workers), in a worker-owned pool slot (E10) | ARITH | B-MEM |
| Vertex | 36 B → 40 B at the packed rungs, 44 B → 48 B at the exact rungs: the own share, the target share, the skirt flag and one spare byte in one 4-byte attribute (§10B.8) | ARITH on the MEASURED 36 B (`crates/client/src/chunks.rs:195-198`) | B-MEM and B-FRAME (question 16) |
| Build today | 3.3 ms rung-0 surface, 6.1 ms cave-dense (`docs/design/DEFERRED.md:7760`); 11 ms median rung 9 with the artifact (ruling B18); 16–22 ms on a moving eye with parents (`docs/design/DEFERRED.md:7936-7941`); warm 9.68 ms and cold 33.14 ms median, cold p99 100 ms, over 1 600 chunks before the solve (`docs/investigation/2026-09-08/landforms/12_chunk_budget_base.md:162-170`) | MEASURED | B-BUILD: ≤ 6.5 ms ask-weighted wall time at 7 workers |
| Why a cold chunk is slow | it runs the sample box NINE times, its own and its eight parents'; the parents are 69 % of the time over the 1 600 chunks (`12_chunk_budget_base.md:169`) | MEASURED | B-BUILD (E1) |
| The column read | 839 ns a column and 3 227 µs a chunk at generator 17; a node read 102–162 ns (`docs/design/DEFERRED.md:9533-9535`) | MEASURED | B-BUILD (E2) |
| New work per chunk (fill, `t`, wet share, flat merge) | UNMEASURED (bench M8); the shore law and the side reads go | — | B-BUILD |
| Chunk rate on 7 workers | still eye about 636 a second; moving eye 318–437 a second; 374 a second MEASURED on the 528 m/s leg before the solve (`docs/investigation/2026-09-08/landforms/slice_08_ladder_discussion.md:2044-2056`) | ARITH from the MEASURED 11 ms and 16–22 ms; MEASURED | B-RATE: ≥ 1.3 × the measured ask (680–1 340 a second, ARITH) |
| After the levers (§10A.3) | about 790–1 330 a second, weighted by the ask's rung mix at 537 m | ARITH on one worker's quiet costs; conditional on P1, P2 and E6 | B-RATE |
| The 528 m/s leg | 2 023 urgent chunks (MEASURED, `docs/design/DEFERRED.md:10454-10458`, worker count not stated) → 4.6–6.4 s to clear | ARITH | G-RATE: 0 gap frames, the bounded ask inert |
| Far fold, rungs 11–13 | 0.04–0.6 ms a chunk for uniform cells; at least 1.3 ms more for a coastal rung-13 chunk's mixed cells | ARITH on ASSUMED reads; UNMEASURED | B-BUILD, reported per rung |
| **Frame** | | | |
| Unpaced window today | 11.3 / 13.4 / 11.4 / 9.5 ms at the berth / 6 376 / 11 499 / 16 954 km | MEASURED, `docs/design/DEFERRED.md:8906-8908` | B-FRAME: no stand worse than today; p99 ≤ 16.7 ms; the target 11.0 ms |
| The sheet's share | 3.0 ms at 6 376 km and 0 ms on foot, on the window's frame BEFORE the sky fix (20.2 ms at 6 376 km); on today's 13.4 ms frame UNMEASURED (bench P7) | MEASURED, `docs/design/DEFERRED.md:8894-8898` | — |
| The main thread at 528 m/s | the descent 7.695 ms a frame and its worst frame 33.4 ms (`slice_08_ladder_discussion.md:2591-2596`); the release-and-request walk 1.2 ms a frame on foot (`docs/design/DEFERRED.md:8896`); the harvest LOOP 0.07 ms an upload, with the engine's own upload outside it (`slice_08_ladder_discussion.md:665-667`) | MEASURED | B-MAIN: ≤ 1.0 ms mean, ≤ 2.0 ms p99 |
| Open water per chunk | 7 688 triangles unmerged → about 250–500 merged; the sheet's blocks read 0.59 million triangles on the coast leg | ARITH; MEASURED | B-BUILD: an open-water chunk under 1 ms (ARITH) |
| W17's ring, deleted | its cost was 8 % of a frame at the 20 km stand | MEASURED (W17) | — |
| **Memory** | | | |
| The coast counts, deleted | about 35.5 MB | ARITH (2 956 590 × 12 B) | — |
| The coast mask, deleted | 2.2 MB | ARITH | — |
| Drawn ground at the walk | 16 001 chunks × 168 KB ≈ 2.7 GB at 36 B a vertex (plus 12 B of 16-bit indices); ≈ 2.9 GB at 40 B; both before the sheet's and the floors' savings | ARITH on MEASURED counts (32 M vertices over 9 136 chunks at the berth, `docs/design/DEFERRED.md:8894`; 16 001 chunks on the walk, `:10406`) | B-MEM |
| Footprint | 6 611 MB at the ground stand with 6 659 chunks (before the packed normal took the drawn bytes from 2 460 to 2 080 MB); at the walk UNMEASURED (bench P11) | MEASURED (`slice_08_ladder_discussion.md:897-912`; `docs/design/DEFERRED.md:7990-7993`) | B-MEM (the ceiling for the target machine, question 9) |
| **Edits** | | | |
| An edited cell | 12 B record + 3 B index, + 1 B level index where the level differs | DOC + ARITH | — |
| The trench (1 000 cells) | about 15 KB of rows and about 240 coarse entries | ARITH | — |
| A shard's rung-0 composition | about 5 ms today; about 2 ms with the gathered read and the per-column collapse (E2, E4) | MEASURED parts (`docs/design/DEFERRED.md:9534-9535`, `:7760`); ARITH | off the tick, on the composition worker (§3.3) |
| The digger's click to pixel (B-EDIT) | 140–175 ms on a 30 ms one-way link (ASSUMED); the budget ≤ 180 ms p99 | ARITH (§10A.2) | B-EDIT: ≤ 180 ms p99; a pilot 50 km up ≤ 250 ms |
| Rebuilds at M-1's load, before coalescing | 0.7–3.3 core-seconds a second per observer | ARITH | G-EDIT: ≤ one of the seven workers |
| **Option A (question 1), in addition** | bake 21–112 core-minutes; 57–150 MB stored; head +8.8–9.5 MB; windows 17–79 MB per stand; client folds 0.5–3 s a ring | ARITH (corrected) / MEASURED (B17) | |

Every budget is stated for the owner's Mac, the measuring instrument, and is owed for the target machine of
ruling F6 (question 9); §10A.2 derives each one, and §10A.4 states what it becomes on an eight-core machine.

---

## 8. What stays, what changes, what goes, and the rulings

### 8.1 Stays unchanged (fits the model as it is)
- The fenced integers and the arithmetic: `crates/recipe/src/{gi,wide,root,rng,noise,bend}.rs`; `vd_seed::{bend, seam,
  ladder, digest}`.
- The roughness factor and the survival table (`crates/recipe/src/height.rs:95-160`; `crates/terrain/src/body.rs:143-308`,
  unless the owner rules K), the cap-rock bench (`crates/recipe/src/terrace.rs`), `BodyDefinition::from_seed`
  (`crates/terrain/src/body.rs:929-1335`).
- The C1 read of `Z`: `catmull_rom_16`, `sample_z`, `node_and_fraction`, `node_at`, `slope_charter`
  (`crates/terrain/src/artifact.rs:1811-2012`).
- The solid half of the cell kernel: strata, bed rock, provinces, the cavern and tube hollows, `trilinear8`, `point_at`,
  `pack`, `gap_code` (`crates/recipe/src/cell.rs:105-166`, `:168-265`, `:322-457`, `:588-621`); `crates/terrain/src/strata.rs`;
  the substance registry.
- The solve's land, craters, erosion and climate; the sea's bisection over the connected ocean; the lakes' budget; the
  water tree (`crates/terrain/src/{land,craters,solve,climate,lakes}.rs`) — as sources.
- The composition's rank order (S6-6) and the seating rule (`crates/terrain/src/{compose,seat}.rs`). `compose()` itself
  changes in place: it carries `t`, or a fill step follows it (§3.1, §8.2; the second round withdrew "unchanged").
- The chunk topology and the halo (`crates/terrain/src/chunk.rs:80-172`; `crates/terrain/src/lattice.rs:46-120`); the vertex
  position (`crates/terrain/src/position.rs:1-168`; lever E6 changes its code, never its bytes).
- The extractor's integer arithmetic (`crates/terrain/src/extract.rs:98-566`; lever E5 visits only the crossed cells, and
  the arithmetic, the order and the bytes stay).
- The client's drawing tools: normals, packing and the floating origin (`crates/client/src/chunks.rs:1809-1935`); the
  worker pool and the core share (`crates/client-render/src/terrain.rs:59-86`, `:460-585`); the ladder's choice rules,
  the bounded ask, the band table, the arrival ramp, the shadow ladder (`crates/client/src/{ladder_view,ask_pace,arrival}.rs`);
  the closed-cave shade at a ring's far edge (`crates/client/src/chunks.rs:1175-1199`), because caves still end at
  rung 3. They stay as tools; the second round changes some of them in place (§8.2): the pools and the descent's thread
  (E8, E10), the keep margin at a band's edge, the rebuilt key's stamp and the late caster's ramp (§10B.4).
- The artifact transport: the worker, the store, the source, the gateway's serve
  (`crates/bins/src/{artifact_worker,artifact_store,artifact_source}.rs`; `crates/connection-plane/src/gateway/artifact.rs`).
- The diff lane's plants (`crates/wire/src/channels.rs:85-173`, `:398-497`; `crates/core/src/chunk_row.rs`), the store seam
  and its planted block families, the grid seam, the world handshake.
- The recipe-gpu shell pattern (`crates/recipe-gpu/src/lib.rs`) and every determinism gate (their tables are recorded
  again).

### 8.2 Changes in place
- `octave_term`: the ridged branch is centred by one pinned word (`crates/recipe/src/height.rs:460-467`).
- `column_surface_at`: no shore call; `FieldRead` loses `water` and `side`; `PlanCharter` loses its shore `sea_radius`
  (`crates/recipe/src/plan.rs:61-65`, `:137-175`, `:214-247`). The biome's own sea datum stays (`BiomeCharter.sea_radius`,
  `crates/recipe/src/height.rs:474-486`) until 8e.
- The cell kernel: a hollow opens its cell; the fill kernel (the column's `λ`, the charter's medium) replaces the
  side-chosen water input of `fluid_at`; the top byte `t` (`crates/recipe/src/cell.rs:352-370`, `:494-534`).
- `compose()`: it carries the top byte `t` from an edited cell's row, or one fill step after it derives `t` for the edited
  cells (`crates/terrain/src/compose.rs`; §3.1; §10C LAW-25).
- The column read: `λ` from the level map through the descent step replaces the side, hold and coast chain; the cell
  gains the top byte; the chunk skip tests the top (`crates/terrain/src/chunk.rs:93-98`, `:340-431`, `:554-603`).
- The extractor reads the top byte, adds the wet share, faces from not-gas to gas (`crates/terrain/src/extract.rs:79-83`,
  `:443-566`).
- The column span asks for the top, not the ground (`crates/terrain/src/digest.rs:258-330`).
- The artifact: the row's water word is the level map; `level_for` gives the rows to rung 13 and the far map from 14;
  the far map; ARTIFACT_VERSION 10 (`crates/terrain/src/artifact.rs:95`, `:849-876`).
- The solve: after the water tree, the level map walk, the rim pass inside the lakes-to-sea fixed point, the far map bake
  (`crates/terrain/src/solve.rs:1904-1966`).
- The client: one mesh, the wet share, the flat merge, the wet morph, the far fold at rungs `L_ref + 1` to 13, the eye's
  surface as the top (`crates/client/src/chunks.rs`; `crates/client/src/ladder_view.rs` `handover_step_m` without its
  field term).
- The renderer: one ground material with the wet share (`crates/client-render/src/terrain.rs:1996-2090`).
- THE BUILDER (§10A.3; no lever moves a world byte, and each keeps its plain form as the test oracle): `ColumnRead` over
  a gathered node window (E2, in S2); a bounded column-field cache for `sample_box` (E3, in S2); per-column runs in
  `cell_pass` and the halo (E4, in S2); a ranged `extract_with` with a reused table (E5, in S3); in `geometry_from`, the
  parent SUB-BOX on a miss, and the drawn chunk's parent form only where bench P5 shows hits (E1); the skirts from the
  border groups, each radial computed once, `vertex_position` without a `Vec` (E6); worker memory pools (E10); in
  `sync_terrain`, the descent on one of the terrain's own threads, the release walk on any change of its inputs over a
  difference of sorted sets, and the harvest by a derived time budget with worker-built mesh buffers (E8); the serve
  order at the gateway (E12, in S3); the speed margin (E11, in S3).
- THE SEAMS (§10B.4): the parent mesh as the coarser TOP mesh with a share byte, and every fallback reading the coarser
  top ([TOP]); the target share and the wet ramp in the vertex stage and the probe ([WET MORPH]); the merge condition
  ([MERGE]); the chunk's wait for its parents' fields ([PARENTS]); the face seam's target from the neighbour face's parent
  mesh ([SEAM TARGET]); the rebuilt key's stamp ([STAMP]); the edit tick ([EDIT TICK], with S4); the keep margin at a
  band's edge ([HYSTERESIS]); the far sink ([FAR SINK]); the skirts' shares ([SKIRT SHARES]); the book's grace
  ([BOOK GRACE]); the late caster's ramp ([CASTER RAMP]); the harness probe's kinds 3–5 (wet top, skirt, underside) beside
  today's 0–2 (`crates/client-harness/src/probe.rs:38-46`), and a pop detector that compares wet pixels and counts wet
  jumps (`crates/client-harness/src/pop.rs`).
- THE LAWS (§10C): the client refuses an artifact head of another version by name; `FLUID_LEVELS` is required; the
  artifact pace in `TransportTuning`; the `μ` test; the flood on the composition worker, gated by `block_edit`, with its
  own parking field; the owner fence's writer (S4); the player's dig input (S4).
- GENERATOR_VERSION 22; every pin recorded again (terrain-pin, mesh-pin, artifact-pin, charter-pin, the golden fields,
  gpu-drift).

### 8.3 Deleted (no adapter to any of them)
- **The shore law:** `shore`, `SHORE_SHIFT` and the four side words (`crates/recipe/src/height.rs:334-434`); the shore
  input of the column pass (`crates/recipe/src/plan.rs:61-65`, `:137-175`, `:222-241`).
- **The coast mask:** `Artifact::coast`, `Side`, `coast_side`, `side_word`, `coast_word`, `side_of_word`, `coast_bytes`
  (`crates/terrain/src/artifact.rs:220-344`, `:1709-1716`); its store rows (`crates/bins/src/artifact_store.rs:51-56`;
  `crates/sim/src/stub/built_store.rs:59-62`); the client's `accept_coast` and the far chunk's wait for the mask
  (`crates/client/src/artifact_book.rs:158-192`, `:563-616`); `BulkMsg::ArtifactCoast` tombstoned.
- **The footprint counts:** `CoastCounts`, `coast_level`, `footprint_level`, `PyramidField.coast` and `.counts`,
  `ZField::coast_counts` and `water_side`, `sample_side` (`crates/terrain/src/artifact.rs:346-520`, `:586-600`,
  `:776-785`, `:1314-1379`).
- **The column water chooser:** `ColumnWater`, `column_water`, `stencil_holds_sea` (`:1157-1312`); the nearest-node water
  read `sample_row` (`:1135-1155`); `side_of_row`, `TileCache.sea_m`, `SparseRows.1` (`:621-684`, `:747-761`).
- **The second far field:** the mean pyramid, its water word `pyramid_water` and `block_water` (`:210-219`, `:1597-1623`,
  `:1665-1708`), `field_step_m` (`:936-1003`); `BulkMsg::ArtifactPyramid` tombstoned.
- **The column wiring of the shore:** the side, the hold, the coast and the beach override
  (`crates/terrain/src/chunk.rs:350-421`); the shore in `height`, `field_column` and `drawn_height_field_m`
  (`crates/terrain/src/height.rs:70-93`, `:124-226`; the drawn surface becomes the top).
- **The column-word fluid pick and the air stamp:** `fluid_at`'s side-chosen input, `above_cell_word`'s water,
  `CellAt.water` as the side chose it, and a hollow's air stamp (`crates/recipe/src/cell.rs:352-370`, `:472-475`, `:521`,
  `:527-534`).
- **The water sheet:** `WaterSheet`, `water_sheet`, `water_sheet_sided`, `GroundWord`, `WaterSource`, `WaterSide`,
  `water_radius`, `water_quantum_m`, `SHEET_BLOCK` (`crates/terrain/src/position.rs:182-498`; the chord bound's arithmetic
  moves to the flat merge); `SampleBox.water`, `.surfaces`, `.sea` (`crates/terrain/src/lattice.rs:74-91`).
- **The sheet's client parts:** its build, hide bound, exact cut, deep-water blocks and overlap, its morph, its own box
  (R1b) and the floor cut (`crates/client/src/chunks.rs:140-160`, `:185-188`, `:255-298`, `:1415-1436`, `:1469-1690`); its
  entity and material (`crates/client-render/src/terrain.rs:2059-2090`, `:3238-3265`).
- **The golden fields' coast exception** (`crates/terrain/src/artifact.rs:1454-1460`).
- **The bench's shore rules and design D's bench code, deleted in S3 on the owner's word** (question 23; the owner's
  requirement 2: *"We can comment out all current special shores and sea models."*): N1, O1 and N2
  (`crates/bins/src/shore_rules.rs`, `shore_shape.rs`, `shore_geometry.rs`); design D's skin, lid and dry word
  (`crates/bins/examples/shoreline_bench/{skin,skin_lines,lid,far_window}.rs`; `crates/bins/examples/far_water_share.rs`).
  After S2 they measure nothing that ships, so kept they would be a second implementation of one concept (SL5; the
  FINAL-backend law; §10C LAW-21). Until the owner's word they stay in the tree as instruments; git keeps them after.

### 8.4 The rulings

**RETIRED (their mechanisms; the owner's word is needed):**
- W6 (the quarter hold). Its PURPOSE — no coast slide — is the far fold's above `L_ref` and question 1's below it (gate
  G-SLIDE's line).
- W10 items 1–2 (the coast mask; "no host derives a side from a mean"): no side exists any more. W10 item 3 is met by the
  one field and the far fold; item 4 (collision reads rung 0) stands.
- W15 (the footprint counts).
- W16 item 2's mechanism (the three-way side), W16's sheet on the ground's grid, and W18 (the sheet on the ground's
  lattice, its morph, its cuts). W16 items 1–3 as LAW stand as the bar; item 4 (the drawn rivers retired) stands.
- W17 (the field step and its floor at the rows → level-1 swap): its cause, the second field, is deleted.
- B7 item 1's MECHANISM (`SIDE_INLAND`, `Side::Pocket`) and B14 item 4 (the pocket exemption). B7's LAW stands and the rim
  pass applies it at the drawn rims.
- B8 item 1's "the lake level held by its dry neighbours (W6's owed law)". Its "one shoreline from one field" is met by
  construction.
- B13 (N2; B21 item 1 "parked" it, and the owner's word retires it); B14 items 1–3 (the shore band; THE REACH LAW of
  N2's fade, whose registered spectrum conflict stays in DEFERRED for R2 — the first round omitted it; the far counts);
  B15 items 1–2 and 4; B17 (the skin's windows: a trade the owner rules, question 20); B18 Q2–Q3 as separate data (the
  lid and the dry word). The far fold keeps B18 Q2's intent: a height at every rung.

**AMENDED:**
- W4 item 3 ("the one sheet, the one coast band and beach"): its principle — one water mechanism, no second surface — is
  the model's own; its named parts retire. T1 item 6 and T2 ("the water sheet drawn as a real surface at the sea radius
  with the coast clipped"): the water's top is part of the one mesh; 8o's "surface as a SHAPE" fits.
- S5-3: above `L_ref` the coarse answer IS the fold of the reference rung, and an edit's coarse entry carries the top's
  material shares beside its dominant substance, so no dominant-solid threshold reaches the picture (A5). The generator
  is still never asked to fold fine cells; R-7 YES ("an exact cheap coarse fold inside the generator", "the bench decides
  whether it exists", `docs/design/owner_decisions_2026-09-07_voxels.md:215`) already allows the fold.
- S6-6: the ORDER stands; the static fill is inside rank 1, and live water is a rank-2 row. `compose()` carries `t` or a
  fill step follows it (§3.1).
- V6 B-3: the density byte stays the gap to the SOLID; the top gap is derived and never stored.
- W2: the flood wets connected open cells, dug and natural, inside a stated bound, on the shard's composition worker; a
  flood past its bound and a dig that joins two waters are REFUSED with a reason, never a wall (question 4); the river
  half waits for rivers; the first trench comes before 8e (question 7).
- T1 (the order): the trench needs a minimal store and lane before 8e (question 7, after the detailed discussion that V4
  row 14 asks for).
- B18 Q1 (the far look) and Q5 (the hand-over by the pixel's size): made by the far fold.
- B21 item 1 (the rim pass is a new rule, named, §2.3) and item 2 (two ladder rules, §4.1): question 1.
- B16 and W16 items 1–3: met above `L_ref`; below it, as the owner answers question 1 (the slide's line and the ponds).
- ONLY ON THE OWNER'S WORD: T9 and W11 (the flat level, question 3); B16 (ponds that come back, question 1); B20 item 1
  (the builder before the coast, question 10); V10 (the whole build at 8 ms, question 11); T7 rule 1's constant (K,
  after bench M5, question 2).

**OPEN UNDER F9:** the card as a second builder on solved bodies (question 12). F7 asks for the GPU "to the maximum"; F9,
later, made the card a budgeted knob and left its worth to the owner.

**STAND UNCHANGED:** SL10; S5-2 (zero tolerance, so this is a world epoch); T3; T9 and W11 (unless question 3 answers
(c)); W3; V4 rows 13 and 14; V5 (the caves rule; the Areas rule); R-16 (question 19); F1–F2; F6 (its target machine
owed, question 9); F9 item 1 (the bounded ask); B19; B20 (the coast first — the recommended order keeps it; half the
cores); B21 item 3 (the method; items 1–2 as amended above); V10 (the cell path's 8 ms, gate G-BUDGET).

**READ AGAIN AFTER THE EPOCH:** G-LAKE-SHARE (B3) and its ceiling (B9 item 1): the rim pass lowers lakes and the ponds add
water, so the lake share moves (§10C LAW-27).

§10C.5 holds the full table of the rulings that the design amends or retires, with the owner's words beside each.

---

## 9. The first vertical slice: one coast and one trench, correct at every distance

**Where.** The owner's 15° stand at the belt's coast (B19 item 3), the camera about 100 m over the sea and looking at the
coast (B20 item 1), then the climb to orbit, then one trench dug from the sea at the same beach.

**The order** (one job at a time; each step ends with its gates, its numbers, its pictures and one line in the living
status file; the owner's word opens the next step):
- **S0 — THE BENCH AND THE FAILING COAST GATE** (§10; no world change). The benches M1–M11, P0–P14 and B-S1–B-S11.
  First, as B20 item 1 rules, the gate that stands about 100 m over the sea and FAILS on today's picture: the probe's
  new kinds (wet top, skirt, underside) and G-COAST-100, run red on today's code (bench B-S4). P0 records today's
  performance and pop baselines on the same tree, on pairs of flights. The owner answers questions 1–23 on the numbers
  and the pictures.
- **S1–S3 — ONE WORLD EPOCH** (§6.3; §10C LAW-5). Built and gated as three steps on the working tree, merged ONCE:
  GENERATOR_VERSION 22, ARTIFACT_VERSION 10, the protocol floor at the minor that adds the far map, every pin recorded
  again (terrain-pin, mesh-pin, artifact-pin, charter-pin, the golden fields, gpu-drift). S1 and S2 end with bench
  numbers and bench pictures; S3 ends with the flown world.
  - **S1 — THE SOLVE.** The level map in the row's word; the rim pass inside the lakes-to-sea fixed point, on a fixed
    partition with an ordered merge (G-SPLIT); the far map for rungs 14–18, cut coarsest rung first; the coast mask, the
    pyramid's water words and the mean pyramid deleted. The home solve runs again; M3 prints its wall time.
  - **S2 — THE RECIPE.** The shore law deleted; the centring with its `μ` test (G-MU); the fill kernel (`λ` through the
    descent step on the rows) and the top byte; the medium from the air flag; a hollow opens its cell; the far fold's
    kernels. The new column read is written GATHERED (E2) with its column cache (E3), and the new cell pass with the
    per-column collapse (E4), from the start: nothing is fitted onto the shore chain that S2 deletes (B21: "don't try to
    connect").
  - **S3 — THE ONE MESH AND THE LADDER.** The extractor reads the top byte over each column's crossed range (E5); the wet
    share; the flat merge with its morph condition; the span asks for the top; rows to rung 13 and the far map from 14,
    served coarsest first (E12); `field_step_m` deleted; the sheet, its cuts, its box and the floor cut deleted (E7); the
    eye's surface is the top; the speed margin (E11); every seam mechanism of §10B.4 except the edit tick; the client
    refuses a head of another artifact version; the bench's shore code deleted on the owner's word (question 23). **The
    owner looks** (the coast from 100 m, the climb to orbit, the descent).
- **S0b — THE BUILDER** (placed by question 10; recommended right after the epoch, because S3 rewrites the same builder
  code). Levers E1, E6, E8 and E10: the client's builder and frame, which the model keeps. No world byte moves: the pins
  stay green without a re-record, and the picture gate, the pop gate and the moving-eye gate judge the floats and the
  timing. G-MAIN, G-BUILD, G-LAT and G-RATE become binding here.
- **S4 — THE TRENCH.** The minimal real edit path for one realm, after the detailed discussion of slices 9 and 10
  (question 7; V4 row 14), opened by the storage design's load harness (V8 addendum item 2). The player's own dig input
  first (§10C LAW-16); then `vdctl` presses it at the input-resource seam, never a raw world action past the client. The
  brush or the per-cell dig (question 22). The planet's shard checks the dig (the reach; the Area lookup), writes the rows
  under its owner fence (§10C LAW-14), runs the flood on its composition worker where its profile has `block_edit`,
  refuses a flood past its bound and two waters (question 4), commits durably, folds the S5-3 entries with the top's
  shares, and ships `ChunkRows` per (tick, rung) with their counts on the Bulk class; the gateway filters by the rung
  header; the client applies each whole edit tick in one frame. No stand-in (the standing rule: test exactly
  production). **The owner looks** (the trench from the beach, from 50 km and from 450 km).
- **E9 — THE CARD WITH ROWS**: only on question 12's word, after S3 (its kernels need the fill and the top byte).

**Every step's report** (B6 item 2; §10C LAW-26): the three stands' pictures (the belt, the coast, a plain); one picture
on the fourth stand framed like the reference, with its skyline numbers; the reference checklist brought up to date. A
step that turns a gate green but moves that picture AWAY from the reference says so. The centring (question 5) moves the
land, so the skyline moves: S2 states by how much.

**Coverage** (HR5; §10C LAW-15): every step ends with 0 misses on the lines that it touches; the full `just
coverage-fast` runs before the epoch merges, and before S0b and S4 merge. B4 holds the pass since the last one as owed
(vd-terrain read 111 real misses at step 2b, MEASURED, `docs/investigation/2026-09-24/terrain_believability_status.md:1253`);
it runs first, before the epoch's merge. The design adds no deferral.

**THE GATES — one table for the whole design.** Each gate can fail, and each names the failing control that proves it:
the same run with the cure switched off must fail where today fails, or the gate proves nothing. "Today" says what
today's code reads where it is known. "Binds" is the step from which the gate must pass. HR5 covers the Tier-A halves
(the probe's codec, the flicker judge's pure half, the book's grace, the keep margin, the stamp rule, the merge
condition) at 100 % region and branch.

| Gate | What it checks | Pass | Failing control | Today | Binds |
|---|---|---|---|---|---|
| G-COAST-100 (B20) | A readback 100 m over the sea at the owner's 15° stand, looking at the coast; six bands in view (ARITH: rungs 0 → 6; the horizon 35.6 km). The probe's kinds count the sky between the water and the land, skirt pixels and underside pixels; G-NO-WALL counts on the chunks in view. | 0, 0, 0, 0 | Today's code: bench B-S4 must read each class that B20 saw over nought, or the probe cannot see the defect | FAILS (MEASURED B20: the hole, the walls, the 208 m wall) | built red in S0; green in S3 |
| G-WATERTIGHT | At every rung 0–18 on the stands, every edge of a wet top quad off a chunk seam is shared with another top quad. | 0 open edges | The sheet beside the ground (two meshes) | FAILS by construction | S3 |
| G-TOP | At every rung on the stands, the drawn wet share over the stand's footprint equals the share of that rung's cells whose top is wet. | equal to the rounding | The extractor on `s` in place of `t` (thin water lost, review blocker L1) | UNMEASURED | S3 |
| G-FLAT | Every vertex of a group whose not-gas cells are all liquid stands at its level within one vertex quantum. | 0 off | A flat block one cell wider than the chord bound | — | S3 |
| G-NO-WALL | Tangential liquid–gas edges (a vertical water face) taller than one cell, at every rung up to `L_ref`, whole on the stands and on a planet sample after the rim pass. | 0 on the stands; the sample's count with its stated error | The rim pass off | FAILS (the 208 m wall) | S1 (bench), S3 (drawn) |
| G-ONE-LEVEL | Every column of a stand keeps one `λ` at every rung up to `L_ref`. | 0 changes | `λ` from the nearest node at each rung's own cell | — | S2 |
| G-MEAN | Over 9 600 directions a rung pair, the mean drawn ground less `radius + Z + coarse`. | within 1/8 of a cell of the rung | The centring off | FAILS (up to `m × 571.8 m`) | S2 |
| G-MU | `μ` derived again from the recipe's own ridged fold against the pinned word. | equal | The ridge's fold changed by one term | — | S2 |
| G-LAKES-DRAWN | The drawn lake and ocean shares at rung 0 (a sample) and at `L_ref`, against the level map's shares. | within the sample's stated error | The centring off (lakes drain) | UNMEASURED | S3 |
| G-LAKE-SHARE (B3; B9 item 1) | The lake share of the land, read again with the rim pass and the ponds. | ≤ 4 % of land, or the owner's word (B9) | — (B3's own gate) | RED 4.828 % (B9) | read after the epoch |
| G-FAR-EXACT (a) PER CELL | Over every far cell of the stands (and M6's whole planet): a parent's top and share against the mean of its children's. | 0 misses beyond 1 m and 1/255 | A parent read at one point sample | FAILS (B15 MEASURED 53 of 714 far lake bodies flicker) | S3 |
| G-FAR-EXACT (b) PER PIXEL | The climb from 450 km to 60 000 km over the four stands: the share step at each band's far edge; the change inside the band. | 0 steps; monotone | The wet morph off | — | S3 |
| G-SLIDE | The descent from 450 km to 100 m over the four stands: per band the coast's median, p99 and largest motion in pixels at the finer cell; frames outside every band with a coast change over one pixel. | the line of question 1 (proposed: B18 Q4's accepted sizes, the sea ≤ 1.46 px and lakes ≤ 4.79 px at most; or W17.1's p99 ≤ 1 px, max ≤ 2 px); 0 frames outside the bands | The centring off (the one-sided lift moves every coast, review findings L4 and L5) | — | S3 |
| G-POND-FLICKER | The descent and the climb over the four stands and a planet sample: per band, the area of water that appears or vanishes, in pixels at the band's distance; the columns that go dry and then wet in one climb. | under 1 px per band; 0 columns — or the owner's amendment of B16 (question 1) | The centring off (the lift changes with the rung) | UNMEASURED | S3 |
| G-CLASS | The class from `s`, `t` and the block lane agrees on every pinned cell. | 0 | One cell's `t` written over its `s` with a liquid substance | — | S2 |
| G-NO-DRIFT | The three cell bytes, the mesh and the far fold on the stand's chunks: shard CPU against client CPU; debug, release, native; the aarch64 and x86-64 legs. | byte-identical | The tampered golden table (`just terrain-legs`' own control) | green on today's bytes | every step |
| G-SPLIT | The home artifact's digest from the rim pass and the far bake at 1 and at 7 workers. | equal | An unordered merge | — | S1 |
| G-VERSION | A client of artifact version 9 against a version-10 head; `ArtifactHeld` against a digest that folds the far map. | refused by name; 0 chunks drawn | Today's `accept_head` (checks no version) | FAILS (no check) | the epoch |
| G-REQUIRED | A reader that does not know tag 8 meets a row with `FLUID_LEVELS`. | it refuses the row | Tag 8 written as optional | — | S4 |
| G-FENCE | A row with a stale realm fence. | refused by the gateway and the client; 0 applied | The fence check off | — | S4 |
| G-NO-FACE | After a dig settles: liquid–gas faces taller than one cell where a flood ran; the refusals of a flood past its bound and of two waters. | 0 faces; each refusal typed | The refusal off (the flood stops at its bound) | — | S4 |
| G-TRENCH-SHARD | A dug cell under the sea's level that connects is water; a dug cell over it is air; a sealed pit is air. | all | The static rule applied to dug cells (review finding L3) | — | S4 |
| G-TRENCH-CLIENT | Two clients' composed rung-0 boxes equal the shard's, byte for byte; the rows arrive within two ticks of durability. | all | A client that skips the fill on edited cells | — | S4 |
| G-TRENCH-FAR | At 50 km (rung 6) and 450 km (rung 10) the client's coarse entries equal the shard's; no drawn vertex moves more than one pixel; a second canal 128 m wide shows as a line at 50 km. | all | Coarse entries without the wet coverage | — | S4 |
| G-TRENCH-SEAM | The trench of S4: the dig's frame, then a climb from 1 m to orbit over it. | the whole rebuild set swaps in one frame; nothing outside its footprint moves past the floor; 0 holes and cracks; G-POP and G-WET-JUMP over the trench | The rung-6 copy rebuilt one frame late | — | S4 |
| G-IDENTICAL (HR4) | The dig and its flood on the planet's shard and on an Area's shard, the Area fixture given its source row; the hull's arm when its down and its source exist (§5.5). | identical rows | An Area profile without `block_edit` | — | S4 |
| G-POP | The moving-eye legs and a descent from 450 km to 100 m over the four stands and the coast stand: runs of 60 consecutive frames every 10 s, every pair judged (`crates/bins/examples/pop_run.rs`), ground AND water in the probe, split by kind and by boundary. | NOW: no leg and no boundary worse than P0's baseline read by the same probe (the share of pixels past the floor, and the widest step), on pairs of flights; at least 10⁶ compared pixels on every leg. TARGET, after the cause is found (P0, B-S5) and on the owner's word: ≤ 0.1 % past the floor at every boundary (the still pixels' own rate, `crates/client-harness/src/pop.rs:54`) | The morph metre and the target share forced to nought | Reported only, one pair every 2 s; blind to water; the walk 0.67 %, 240 m/s 0.21 % (rung 6 0.36 %), 528 m/s 0.98 % (rung 6 1.6 %); both descents 0 pixels compared (MEASURED, `docs/investigation/2026-09-26/r1_renderer_pops/r1_flights.txt:25-29`) | S0 (baseline), S3 |
| G-WET-JUMP | The same pairs: pixels whose wet or dry kind matches no pixel of the 3 × 3 neighbourhood in the previous frame. | 0 at every swap | The wet morph off (the target share forced to the own share) | no water kind | S3 |
| G-STILL | A still eye at the five stands after the band settles: 120 consecutive frames, every pixel. | 0 pixels with two or more reversals past the floor; 0 changes of a pixel's kind or rung | One chunk's arrival stamp restarted every other frame | UNMEASURED | S3 |
| G-SLOW | A walk at 1.4 m/s and a climb at 1 m/s through every band of the coast stand; an eye that sways ±1 m for 10 s across each switch line and one skyline crest. | 0 reversals past the floor inside the ramp's 0.5 s; no key asked twice after the first sway | The sink forced to nought (the rungs z-fight in the band); the skyline's keep margin set to its want margin (MEASURED once: 4 chunks missing on 80 of 1 383 samples) | UNMEASURED | S3 |
| G-BAND | The band's gap every frame, and hole pixels on every recorded frame. | 0 gap frames on the walk and the 1.4 m/s hull; 0 hole pixels on every leg | A worker share of one | the walk 0 gap frames on 1 202 samples, the 1.4 m/s hull 0 on 1 289 (MEASURED, `docs/design/DEFERRED.md:10402-10407`) | every step |
| G-CRACK | Stands on a chunk edge, a cube-face edge and the one coastal cube corner, with eyes 1 m, 100 m, 10 km and 1 000 km up; a Tier-A test that both chunks give a shared vertex one position, one `t` and one pair of shares. | 0 hole pixels; 0 skirt pixels where the neighbour stands; equal bytes | Skirts off (MEASURED once: one crack pixel at 970 m) | rungs 0–3 measured by `crates/bins/examples/skirt_gap.rs` | S3 |
| G-FAR-EDGE | Tier-A: every vertex of the stands' chunks at wholeness nought, against the parent's surface, shade and share — seam vertices included (the neighbour face's parent). | float rounding; the octahedral quantum; 1/255 | A vertex that keeps its own share; a seam vertex that reads the field | position and shade only; seam vertices read the field | S3 |
| G-PROBE-KINDS | The probe's codec: 0 nothing, 1 dry top, 2 ruler, 3 wet top (share ≥ ½ after the morph and the ramp), 4 skirt, 5 underside (a front face whose normal points to the body's centre); a fixture whose picture and probe agree per pixel. | exact round trip; 0 disagreeing pixels | A wet fragment coded as dry | kinds 0–2 only (`crates/client-harness/src/probe.rs:38-46`) | S0 |
| G-CROSSING-TERRAIN | A hull crosses the planet's realm boundary both ways, the pop detector running. | 0 pops; no ladder forgotten; no artifact served again; no key asked twice | The ladder forgotten at the crossing | UNMEASURED | S3 |
| G-EPOCH | A body's first drawn frame; a re-solve under a drawn body (development). | all six top-rung chunks in the first frame; the swap in one frame, with no frame of old and new chunks together | Today's key-by-key swap | FAILS (`docs/design/DEFERRED.md:9137-9145`) | S3 |
| G-FAR-FIRST | A far planet from its first far part HELD by the client. | its first drawn frame within one beat (0.5 s); 0 frames with a held first part and no drawn rung | The far map shipped finest first | FAILS (a far body draws nothing until its head lands, and its far chunks wait for the coast mask, `crates/client/src/artifact_book.rs:158-192`; `docs/design/DEFERRED.md:8937-8947`) | S3 |
| G-FRAME | The window, unpaced, at the four stands, the coast stand and every moving-eye leg; the mesh bytes at the coast stand. | per stand, p50 not worse than today's MEASURED p50 (11.3 / 13.4 / 11.4 / 9.5 ms) with the 40 B vertex in, on pairs of runs; per leg, p99 ≤ 16.7 ms and at most 1 frame in 1 000 over 16.7 ms. The TARGET, reported: p50 ≤ 11.0 ms at every stand, the berth included, once P7 names the berth's lever | The sheet kept beside the one mesh at the coast stand | p50 MEASURED; p99 and the count UNMEASURED in the window | S3 |
| G-MAIN | The terrain's work on the main thread per leg. | ≤ 1.0 ms mean and ≤ 2.0 ms p99 | The descent on the main thread | 7.7 ms mean, 33.4 ms worst at 528 m/s (the descent alone) | S0b |
| G-BUDGET (V10) | The CELL PATH of a coast chunk (the box, the fill, the extraction, the merge) on one worker. | under 8 ms | The synthetic checkerboard (MEASURED 26 ms, D-TERRAIN-2) | 3.3 ms surface, 6.1 ms cave-dense (before the fill) | S3 |
| G-BUILD | The ask-weighted mean WALL time of one build, parents included, at 7 workers on the moving-eye legs; per rung reported. | ≤ 6.5 ms (B-BUILD); every rung ≤ 8 ms only if question 11 so rules | The parent path without the sub-box | 14.6 ms of wall time at 7 workers (528 m/s, before the solve) | S0b |
| G-RATE | The moving-eye legs at 1.4, 240 and 528 m/s, 537 m over the solved 15° stand, 7 workers. | 0 gap frames on every leg; the bounded ask inert at 240 and 528 m/s; the queue's peak under one second of capacity. On the target machine (question 9): the band whole at the DRAWN rung (F9) | A worker share of three (F6's old quarter) | 491 urgent at 240 m/s, 2 023 at 528 m/s (MEASURED peaks) | S0b |
| G-LAT | P9 on the fast legs. | p99 ≤ 250 ms from the ask to the first drawn frame | The speed margin off | UNMEASURED | S0b |
| G-MEM and G-SOAK | The footprint at the walk and the four stands; a 10-minute soak. | ≤ today's footprint (P11) + 11 % of the drawn bytes − the sheet's bytes; no growth over the soak; the ceiling for the target machine when question 9 names it | An unbounded done queue (MEASURED once: +2.2 GB over a minute) | UNMEASURED at the walk | S3 |
| G-LOGIN | P13, warm store and gateway. | ≤ 3 s to the first frame with every wanted column drawn or stood in for; ≤ 15 s to zero urgent; ≤ 30 s to every planet whole at its drawn level | The far map shipped finest first | 45.3 s for five planets whole (MEASURED) | S3 |
| G-EDIT | P14 (the storage design's M-9). | the digger ≤ 180 ms p99, a pilot 50 km up ≤ 250 ms; edit rebuilds ≤ one worker under M-1's load | The key-by-key swap, or edits queued behind the moving eye | no edit path | S4 |
| G-WIRE | The bytes a second a client on each leg. | the fill ≤ 2 MiB/s; steady flight ≤ 128 KB/s | The artifact served again on each crossing | the pace MEASURED; steady UNMEASURED | S3 |
| The byte gates | `terrain-pin`, `mesh_pin`, `artifact-pin`, `gpu-drift`, G-NO-DRIFT, after every lever. | green with no re-record on today's code; inside the epoch no change beyond the epoch's own | Their tampered-table controls | green | every lever |
| `gpu-drift` on solved chunks | Only on question 12's word: every key of the stands, the card against the CPU. | 0 differing bytes | A pool slot that grew (MEASURED: 1 022 of 6 049 boxes drifted) | the card builds no solved chunk | E9 |
| M-1 (the storage design) | 100 builders at 2 edits a second, 128 observers. | messages a tick ≤ dirty chunks × holders; never observers × edits | A fan-out per observer | UNMEASURED | before S4 |
| HR5 | The new kernels and every touched line. | 100 % region and branch | An uncovered branch | vd-terrain 111 real misses (MEASURED at step 2b); vd-client UNMEASURED | every merge |

---

## 10. The bench before any build

No bench changes the world. Each bench states its time and its core count, runs on a quiet machine (the load under 1.5
for two samples), and can come out either way. A planet-wide count samples every k-th tile and states its error (review
finding C19). A flight bench runs as a PAIR of flights of one binary: single flights misled once (ruling F9's card judge,
`docs/design/owner_decisions_2026-09-12_frozen_patches.md:497-503`).

**The model's benches (M).**

| Bench | Measures | Decides |
|---|---|---|
| M1 THE CENTRING | The ridged fold's mean `μ` over a fixed lattice (the pinned word), and again over a SECOND independent lattice (if the two words differ by more than the word's own quantum, `μ` belongs to the lattice and the centring is refused); the smooth octaves' mean; G-MEAN at every rung pair; pictures of the four stands before and after. | Question 5; G-MU. |
| M2 THE WATER TABLE | On the four stands and a planet sample: the area of ponds (drawn water at rung 0 in natural hollows that are not connected at rung 0 to the node-scale water); the share of cave volume that floods; the drawn pond area added to the lake share against G-LAKE-SHARE's 4 % of land (over it, the flat level fails); pictures. | Question 3. |
| M2b THE WATER-TABLE LAW | The same census under a published water-table law fed by the row's own runoff (the candidate: Fan, Li & Miguez-Macho 2013, read before use): the pond area and the flooded-cave share beside the flat level's; the lake share against 4 % of land. | Question 3. |
| M3 THE RIM PASS | Its run on the home planet: its WALL time on the production server's core count and on one thread, its memory peak; the lakes lowered (count, depth); the pockets joined (count, area); the sea's level change; the fixed point's convergence; G-NO-WALL on the stands after it. Over the owner's ceiling, it fails. | Questions 3 and 21; the S1 budget; B-BAKE. |
| M4 THE NEAR WATER (= B-S2; the pond-flicker bench) | The descent from 450 km to 100 m and the climb back over the four stands and the coast stand, at today's K, with the centring, the one field and no shore law: per band, the coast's median, p99 and largest motion in pixels at the finer cell; the water bodies that show, hide and show over three successive rungs (count, pixels) and the columns that go dry and then wet in one climb; the wet jumps with the wet morph on and off (the control). | Question 1 (the slide's line, the ponds); G-SLIDE; G-POND-FLICKER. |
| M5 THE SURVIVAL CONSTANT | M4 again at K = 4 and K = 8 (the pond count too); T7's rung-disagreement line; G-GRID; the column cost; the new `L_ref`. | Question 2. |
| M6 THE FAR FOLD | G-FAR-EXACT on the climb; PER CELL over every far cell of the home planet at rungs 11–18, the parent's top and share less the mean of its children's (fails beyond 1 m and 1/255); the largest per-pixel share change at each far swap on the climb, printed; the client's cost per rung-11–13 chunk (the stencil, the mixed cells); the far map's bake time and bytes. | Questions 1 and 20; the S1 and S3 budgets. |
| M7 THE ONE FIELD | Tiles per client over the moving-eye legs and at the worst height; the delivery seconds at today's pace and at two other paces, and the sends shed against the transport's retry buffer (fails on a shed send, or on a frame with a hole while the tiles arrive); the gateway cache's peak; the 9 → 10 step without the field swap. | Question 6 (b); the pace's value in `TransportTuning`. |
| M8 THE ONE MESH | A coast chunk's cell path on one worker (G-BUDGET); the whole build with parents on the moving eye; open-water triangles per chunk; the frame at the coast stand against today's sheet; the 40 B vertex against 36 B (= B-S9). | G-FRAME, G-BUILD, question 16. |
| M9 THE FLOOD | The trench and a large dig (10⁶ cells): flood time, wetted cells, the per-tick and total bounds; the shard's composition budget. THE EXTENT CENSUS: for a canal from the sea or a lake into each of the ten largest dry basins and pockets, the connected open cells under the water's level at rung 0 (sampled, the error stated). THE TWO-WATERS CENSUS: on the four stands, the length of catchment boundaries where two levels differ by more than 1 m within 100 m (0 on every stand lets the refusal wait for the digging slice). | Question 4; the S4 tuning. |
| M10 CAVE FACES | Faces inside caves at catchment boundaries on the stands, at rungs 0–3: count and height. | Whether caves join the rim pass. |
| M11 SPLIT DETERMINISM | The rim pass and the far bake at 1, 2 and 7 threads: the three artifact digests. | G-SPLIT (fails if they differ). |

**The performance benches (P; §10A).** Each lever bench also proves that the lever changes no world byte.

| Bench | Measures | Refutes the plan, or decides |
|---|---|---|
| P0 THE BASELINE FLIGHT | The moving-eye gate on the solved 15° stand at 7 workers, a pair of flights: per leg the builds a second, the WALL ms a build (p50, p99), the parent hit rate, the queue's peak, the gap frames, THE FRAME'S WORK per piece, and the pop detector read by the NEW probe (water counted) per boundary, with its compared-pixel count. | — (the numbers that every other bench and gate is judged against; G-POP's baseline). |
| P1 THE SOLVED CHUNK BUDGET | `chunk_budget` rebuilt to read the home artifact, at the 15° stand, the coast stand, the belt and the plain: per rung the median, p90, p99 and max, cold and warm, on one worker AND at 7 loaded workers, with the artifact read, the cells, the halo, the parents, the extraction and each geometry piece (positions, rays, field reads, packing, skirts, bounds) timed apart. | Refutes the plan if the artifact read is under half of the column pass, or the parents under half of the cold build. Sets E6's reach at rungs 1–2, which B-RATE depends on. |
| P2 THE GATHER | The column pass with one node window per box against today's, on the 1 024-key square and every key of the stands; the cost of one spline read through the window. | One cell byte differs, or the column pass is less than 2 times faster. The read's time replaces the ASSUMED 10–40 ns. |
| P3 THE COLLAPSE | The kernel calls and the time per box per rung with per-column runs, over the seam set's 6 049 boxes and the stands, against the plain cell pass. | One cell byte differs, or rungs 4 and up need less than 5 times fewer calls. |
| P4 THE RANGES | The ranged extraction against the full scan. | One mesh digest differs, or it is less than 3 times faster. |
| P5 THE PARENTS | On recorded moving-eye legs, per rung: the asks at a ring's LEADING edge and at its TRAILING edge, apart; the age of each needed parent form (the time since its drawn build); the hit rate with and without the drawn parent form; the sub-box's cost against a whole parent build; every morph metre against today's path on 1 000 chunks (a border normal counted apart). | The line comes from the geometry, not from a chosen rate (ARITH: 24–29 % of each rung's new chunks are trailing-edge asks at 537 m, 16 % of all): a trailing-edge or aged miss costs ≤ 25 % of a whole parent build (ARITH: the sub-box is about 18 % of the parent's cells). A morph metre that differs away from a border refutes the sub-box. |
| P6 THE MAIN THREAD | THE FRAME'S WORK per piece (the descent, the release walk, the harvest, the placement, the stamp, the band tables) per leg in the window, p50 and p99; again with the descent on a terrain thread and the walk run on any change of its inputs; and the render world's cost per uploaded chunk, apart. | Refutes E8 if the terrain's main-thread p99 stays over 2 ms. Sets the harvest's time budget and the upload's share of B-FRAME. |
| P7 THE WINDOW FRAME | Unpaced, at the four stands, the coast stand and the walk: the p50, the p99, the max and the count of frames over 16.7 ms per 60 s leg; one GPU capture per stand to split the vertices from the per-entity work; the sheet's share of today's frame (the sheet off). | The baseline is wrong if a p50 differs from today's MEASURED 11.3 / 13.4 / 11.4 / 9.5 ms by more than 1 ms. Names the berth's lever (E14, or the vertices). |
| P8 THE ASK | The new keys a second per rung on each leg against the ARITH sum at the leg's height, with `z` measured and the trailing edge apart. | Before the owner rules B-RATE: the measured ask replaces the ARITH range (525–1 030 a second at 528 m/s and 537 m). |
| P9 THE LATENCY | For each urgent chunk: the ask, the build's start and end, the harvest, the first drawn frame; p50 and p99 per leg. | Sets E11's input; G-LAT. |
| P10 THE CORES | The build time per worker by core type (performance, efficiency) at 7 workers. | Question 14 closes if efficiency-core builds are less than 1.5 times slower. |
| P11 THE MEMORY | The footprint by category at the walk (16 001 chunks) and the four stands, the drawn bytes a chunk per rung, and a 10-minute soak. | Sets G-MEM's baseline; G-SOAK fails on any growth. |
| P12 THE CARD WITH ROWS | The column and cell passes on the card with a gathered window, over every key of the stands; boxes a second; the frame's cost at the quarter budget; the pop detector and the band on pairs of flights, card on and off (F9's two measured harms). | Question 12. One differing cell keeps the card parked; the harms decide the rest. |
| P13 THE LOGIN | From the world handshake to the first frame with every wanted column drawn or stood in for, to zero urgent chunks, and to every planet whole at its drawn level; warm store and gateway; with and without the coarse-first order. | G-LOGIN. |
| P14 THE EDIT | Click to pixel for the digger and for a pilot 50 km up (the storage design's M-9), with the edit slice. | G-EDIT. |

**The seam benches (B-S; §10B).** Each one runs on today's code first where the text says so, so that its control reads
over nought.

| Bench | Measures | Decides |
|---|---|---|
| B-S1 THE FAR EDGE | On today's code at the four stands and the coast stand: at wholeness nought, the largest gap between a finer vertex and its parent's surface (m) and shade (rad); the fallbacks at the face seams and for missing tiles; the crease's specks in pixels. | Any gap over float rounding or the octahedral quantum, or any missing-tile fallback, is a defect to cure before S3; whether the crease's exact cure is needed. |
| B-S2 THE NEAR WATER | = M4. | Questions 1 and 2. |
| B-S3 THE MERGE AGAINST THE MORPH | At the coast stand and the belt's lakes: the share of would-be merged water vertices whose morph metre passes one vertex quantum, or whose target share differs from its own. | Nought: the merge condition costs nothing; else the islands that a merged block would pop. |
| B-S4 TODAY'S COAST, READ BY THE NEW PROBE | The hole, skirt, underside and wall counts at the owner's coast stand on today's code. | G-COAST-100's failing control: each class that B20 saw must read over nought. |
| B-S5 STILL AND SLOW | Today's code, the band settled: 120 still frames at the five stands (reversals; kind or rung flips); a ±1 m sway across each switch line and a skyline crest (keys asked twice); with P0, the cause of today's pop excess at rung 6 on the fast legs. | Which decisions lack hysteresis; G-POP's target line. |
| B-S6 THE FAR STEP | The largest difference between a fold cell's top and its reference cells' tops, rungs 10 → 11 up to 17 → 18, in cells of the coarser rung. | Under one cell keeps the two-cell sink; else the body states a far step. |
| B-S7 THE FOLD'S NEST | The client's rung-13 fold against the far map's rung-14 cells on the four stands: the largest top gap and share gap. | ≤ 1 m and ≤ 1/255, or a defect in one fold. |
| B-S8 THE WATER'S STEP | The coast stand with every third finer chunk held back: the water-coloured step in pixels at each rung, and the ramp's rise. | X9's price against the ARITH 4–5 px at rung 2. |
| B-S9 THE VERTEX COST | = the 40 B part of M8: the harvest's bytes and the unpaced frame at the five stands, net of the deleted sheet. | Question 16; G-FRAME. |
| B-S10 THE EDIT TICK | For the trench: the chunks and parent meshes that one edit tick touches for an observer at 1 m, 50 km and 450 km; the set's build time on 7 workers; the frames between the first and the last swap on today's key-by-key path. | The set's size against the ARITH 4–18 builds; today's path must read over nought frames (the control). |
| B-S11 THE CROSSING | A hull crosses the planet's realm boundary on today's code: artifacts served again, ladders forgotten, pops. | Whether X16 shows today; G-CROSSING-TERRAIN's control. |

---

## 10A. PERFORMANCE — THE BUDGETS AND HOW THE DESIGN MEETS THEM

*The second round, for the owner's requirement 1: "Ok, please make sure that we are extremely performant!!!". Nothing
here is built or measured again. Every number carries a mark of the header. "Today" is branch `game` at `e8d1747`; most
MEASURED build numbers come from flights before the solve (8c), and the text says so where the artifact changes them.
The review's findings on the draft (§11, the second round) are answered here, each in place.*

**The instrument.** Apple M4 Pro: 14 cores (10 performance, 4 efficiency), a 20-core GPU, 48 GiB of memory (MEASURED
2026-09-28 by the performance draft, `sysctl hw.memsize hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu` and
`system_profiler SPDisplaysDataType`). The terrain has 7 workers (ruling B20; `crates/client-render/src/terrain.rs:4257-4260`).
The window draws 2 560 × 1 440 physical pixels at four samples a pixel (MEASURED, `docs/design/DEFERRED.md:8892-8893`),
capped at 60 frames a second (`:8919-8922`). **The target is NOT this machine.** Ruling F6: "The Mac's fourteen cores
are the measuring instrument, not the target"; the average machine's numbers (the cores, the memory, the GPU class) are
owed by the owner (`docs/design/owner_decisions_2026-09-12_frozen_patches.md:84-94`; question 9). Every budget below
is stated for the instrument, and §10A.4 states what it becomes on an eight-core machine.

### 10A.1 Today's hot paths

#### 10A.1.1 One chunk build: where the 11–22 ms go

MEASURED sizes. A rung-9 chunk takes a median of 11 ms on one worker (ruling B18 Q7,
`docs/design/owner_decisions_2026-09-22_believability.md:518-520`). On a moving eye a chunk takes 16–22 ms of wall time
at parent-cache hit rates of 89–76 % (14 workers; `docs/investigation/2026-09-08/landforms/slice_08_ladder_discussion.md:645-647`),
and 14.6 ms of wall time at 7 workers on the 528 m/s leg (`:2047-2050`). Over 1 600 chunks at rungs 0–3 the median is
9.68 ms warm and 33.14 ms cold, and the cold p99 is 100 ms (`docs/investigation/2026-09-08/landforms/12_chunk_budget_base.md:162-170`).

The recipe itself is fast: 12.19 ns a noise call (MEASURED, `docs/investigation/2026-09-07/00_proposed_voxel_foundation.md:379`)
and 14 ns a cell on one core (MEASURED, `docs/investigation/2026-09-08/landforms/slice_08_integer_bench.md:440`). A chunk
is slow because it repeats work. Nine causes (H1–H9), each read in the code:

| # | Cause | Where in the code | Size |
|---|---|---|---|
| H1 | A cold chunk runs the recipe NINE times: its own box and one box for each parent mesh. | `geometry_from` asks up to eight parents (`crates/client/src/chunks.rs:1317-1320`; `parent_keys`, `:1106-1141`). Each miss runs `ParentMesh::build`: `sample_box` and `extract_all_edges` (`:547-554`). | MEASURED: the parents are 81.35 of 101.97 ms on the named chunk and 69 % of all the time over 1 600 chunks; a hit saves 83 % (`12_chunk_budget_base.md:69-80`, `:169`, `:204-211`). |
| H2 | The artifact is read per COLUMN, never per box. | `ColumnRead::column` (`crates/terrain/src/chunk.rs:340-431`) calls `sample_side`, `column_water` (with the 16-node `stencil_holds_sea`, `crates/terrain/src/artifact.rs:1280`), `slope_share` (four `sample_z`, `:1985-2012`), `sample_province` and `sample_z` (16 node reads, `:1871-1888`) for every column. On the client each node read is a `BTreeMap` lookup with a modulo and a divide (`TileCache::row`, `:724-733`). | ARITH: 80–100 lookups a column, so 330 000–410 000 a box, while a rung-0 box (64 m) stands inside one window of at most 7 × 7 nodes (8 192 m each). MEASURED: 839 ns a column, 3 227 µs a chunk, at generator 17 with a drainage read of about a fifth that W16 retired (`docs/design/DEFERRED.md:9533-9535`). Today's number: UNMEASURED. |
| H3 | The column pass runs per BOX, never once per column of chunks. | `sample_box` runs `column_field` for every key (`crates/terrain/src/lattice.rs:499`); the halo's 252 columns run the column rule again (`lattice.rs:436-457`). | ARITH: a column holds about 1.6 chunks (§10A.2), so about 37 % of the column passes repeat one done a moment before; the halo adds 6 % to each box. |
| H4 | Every cell of a crossed chunk runs the cell kernel. The skip works only for a WHOLE chunk. | `generate_in` skips a chunk only when all of it stands over or under every surface (`crates/terrain/src/chunk.rs:578-601`); else `cell_pass` runs 238 328 cells (`:677-779`) and the halo 23 816 more (`lattice.rs:518-570`). The core is built in one vector and copied into a second (`lattice.rs:500`, `:528-529`). | ARITH: 262 144 × 14 ns = 3.7 ms a box on one core. Caves carve only at rungs 0–3 (§2.4) and the strata reach under 100 m (`crates/terrain/src/body.rs:2620`), so at rung 4 about 8 of a column's 64 cells need the kernel, and at rung 6 about 4. |
| H5 | The extraction visits every cell. | `extract_with` visits 64³ cells and three edges each, and fills a table of 63³ words (1 MB) for each call (`crates/terrain/src/extract.rs:458-518`). | MEASURED: 1.1–1.97 ms a chunk (`12_chunk_budget_base.md:139-192`). A surface crosses about 4 cells of a 64-cell column (ARITH). |
| H6 | The geometry step is scalar and repeats work. | Per vertex: a position from 8 cell centres through a new `Vec` and a sort (`crates/terrain/src/position.rs:83-160`); a ray against the parents' triangles (`chunks.rs:1342-1400`); for a vertex with no parent hit, one coarser-field read for its target and FOUR more for its normal (`field_normal`, `:1175-1196`, called at `:1387-1392`), each a whole column read through H2's map; two precise normal packings of four candidates each (`:1401-1414`, `:1819-1851`); the bounds normalise each radial four more times (`:303-328`). The skirts put every edge of every triangle into a `BTreeMap` (`:1730-1740`). | MEASURED: the geometry is 2.0–9.9 ms per rung (means) and 21 ms on the worst chunk (`12_chunk_budget_base.md:139-192`) — 4.09 ms at rung 0 and 8.56–9.72 ms at rungs 1–2 on the walk stand, before the solve. The four field reads a vertex came on 2026-09-24, after that bench: UNMEASURED. ARITH for the skirts: 3 × 10 800 map entries at 50–100 ns = 1.6–3.2 ms on a plain chunk. |
| H7 | A worker waits for a sibling's parent. | The single-flight claim (`chunks.rs:936-965`). | MEASURED: 3 561 waits in 22 549 builds at 528 m/s (`slice_08_ladder_discussion.md:2052`). The stamp's wall time holds the wait. |
| H8 | Each build allocates and frees about 3 MB. | The box (512 KiB), the core copy (466 KiB), the 1 MB table, the mesh and a dozen geometry vectors. | ARITH: 3 MB × 400–1 000 builds a second = 1.2–3 GB/s of allocator traffic. MEASURED symptom: a fill dirties about 2 GB of small heap blocks (`slice_08_ladder_discussion.md:911`). |
| H9 | The water sheet (this design deletes it). | Its triangles, cuts and deep blocks per wet chunk (`chunks.rs:1415-1436`, `:1482-1655`); a second entity and material; a normal packing ON THE MAIN THREAD (`crates/client-render/src/terrain.rs:2286-2291`, `:3246-3269`). | MEASURED: 3.0 ms of the window's frame at 6 376 km on the frame BEFORE the sky fix (20.2 ms), 0 on foot (`docs/design/DEFERRED.md:8894-8898`); its share of today's 13.4 ms frame is UNMEASURED (bench P7). |

**Example.** A pilot at 240 m/s crosses the belt 537 m up. A new rung-1 chunk enters her ring. Its worker finds two of its
eight parents missing and builds both parent meshes, box and all (6.4–14.7 ms each, MEASURED
`12_chunk_budget_base.md:229-244`). It reads the same 16 nodes again for each of the chunk's 4 096 columns. It runs the
cell rule on all 262 144 cells, also on the cells high in the air whose answer is plain air. Then it maps every triangle
edge to find a few hundred border edges.

#### 10A.1.2 The main thread in every frame

| Piece | Where in the code | Size |
|---|---|---|
| The descent | Runs when the lead eye moves 0.5 m (`EYE_STEP_M`, `crates/client-render/src/terrain.rs:108`; `moved`, `:2511-2513`; the call, `:2809-2820`), so in every frame at 240 and 528 m/s. | MEASURED: 3–5 ms a run (`crates/client/src/ladder_view.rs:341`); 7.695 ms a frame and a WORST FRAME OF 33.4 ms at 528 m/s (`slice_08_ladder_discussion.md:2591-2596`); a cold descent on the ground 19.23 ms (`docs/design/DEFERRED.md:7873`). |
| The release-and-request walk | Visits every held chunk, every pending job and every wanted key in each frame (`terrain.rs:2976-3056`), and puts each pending job back in the queue under the pool's lock (`chunks.rs:2338-2340`, `terrain.rs:657-666`). | MEASURED: 1.2 ms a frame on foot (`DEFERRED.md:8896`). |
| The stamp | Assembles the dev stamp. | MEASURED: 4.1 ms on foot (`DEFERRED.md:8895-8896`); now once per core step (`:8922-8923`). |
| The harvest | Up to 48 chunks and 10.8 MB a frame (`terrain.rs:81`, `:91`); `mesh_of` clones every attribute vector (`:2236-2247`). | MEASURED: 0.07 ms of the main thread an upload (`slice_08_ladder_discussion.md:647`), so ARITH up to 3.4 ms in a full frame. That is the harvest LOOP only: "the frame's cost sits in the engine's own GPU upload of the changed assets and in the draw count, outside that loop" (`:665-667`). |
| The placement | Visits every drawn chunk (`terrain.rs:4166-4198`). It writes only on a change, but a moving eye changes every chunk. | UNMEASURED alone. The walk holds 16 001 chunks (MEASURED, `DEFERRED.md:10406`). |
| The band tables | One row per rung per realm (`terrain.rs:2834-2856`). | MEASURED: no frame change within noise (`docs/investigation/2026-09-26/far_water_seamless_design.md:958-961`). |

#### 10A.1.3 The GPU frame, the memory and the wire

- **The frame** (`VD_CLIENT_UNPACED=1`, the window): 11.3, 13.4, 11.4 and 9.5 ms at the berth, 6 376, 11 499 and 16 954 km
  (MEASURED, `docs/design/DEFERRED.md:8906-8908`), after the sky fix (the air over a pixel marched once: the berth
  30.0 → 11.3 ms, 6 376 km 20.2 → 13.4 ms). HDR and bloom cost about 5 ms (`:8933`). The ablations that priced the
  sheet at 3.0 ms and the shadows at 0 ran on the frame BEFORE that fix (`:8894-8898`), so they are not today's shares.
  The rest at the berth (32 M vertices, 9 136 chunks, `:8894`) is not split between the vertices and the per-entity work
  (owed, `:8886-8888`). The worst frame in the window is UNMEASURED; headless, the worst frame of a second reads
  17.2–20.3 ms (median) and up to 70.2 ms (`docs/investigation/2026-09-26/r1_renderer_pops/r1_flights.txt:6-12`).
- **The memory.** The footprint was 6 611 MB at the ground stand with 6 659 chunks and 2 460 MB drawn (MEASURED,
  `slice_08_ladder_discussion.md:897-912`). The packed normal took the drawn bytes to 2 080 MB (`DEFERRED.md:7992`).
  The upload pool holds 0.64 × the drawn bytes (`slice_08_ladder_discussion.md:925`); the parent cache 256 MB
  (`terrain.rs:364`). The bytes of today's 16 001-chunk walk: UNMEASURED.
- **The wire.** One budget of 1 MiB a half-second beat for each client (`crates/connection-plane/src/gateway/artifact.rs:41-50`),
  an inline constant today (§4.2). Five planets are whole 45.3 s after a login behind a warm gateway (MEASURED,
  `DEFERRED.md:10532`).

### 10A.2 The budgets on the instrument

**The ask, derived.** The ladder's own estimate of new chunks a second is `q_L = 2 · g(R_L) · v · z / A_L`, with
`R_L = 1.1 · switch(L)`, `switch(L) = 869.1 m · 2^L`, `g(R) = √(R² − h²)` and `A_L = (62 · 2^L m)²`
(`crates/client/src/ladder_view.rs:422-433`; `crates/terrain/src/artifact.rs:890-908`; the pixel is `2 · tan(22.5°) / 720`,
`crates/core/src/geometry.rs:1160-1173`). It counts a ring's LEADING edge only ("it counts the whole circle's leading
edge", `ladder_view.rs:426-428`). ARITH, summed over the rungs:
- At `h` = 537 m, the gate's flight height today (`DEFERRED.md:10457`): `Σ = 0.898 · v · z`. Rungs 0, 1, 2, 3 and 4 up
  are 45.8, 26.6, 13.7, 6.9 and 6.9 % of it, so rungs 1 and 2 are 40 % of the ask.
- `z`, the chunks a column, is not measured on the solved ground. The first draft fitted `z ≈ 1.66` to the MEASURED 374–400
  chunks a second at 528 m/s (`slice_08_ladder_discussion.md:2044-2056`) at "860–1 700 m up"; that table states no
  height (the 858 → 1 713 m pass is another run's, `:474`), and over that band `Σ` falls from 0.685 to 0.348, so `z`
  runs from 1.11 to 2.18 (ARITH). The anchor also flew the rough world before the solve (7 253 vertices a chunk at the
  ground stand, `:699`, against 3 503 at today's berth, `DEFERRED.md:8894`).
- So the 528 m/s ask at 537 m is 525–1 030 chunks a second (ARITH; 758 at `z` = 1.6); 240 m/s asks 240–470; the walk
  (`h` = 1.8 m, `Σ ≈ 0.995 · v · z`) asks about 2.2.
- THE TRAILING EDGE. For every rung L ≥ 1 the eye also asks rung-L chunks BEHIND it, at `0.9 · switch(L − 1)`, where the
  finer disc leaves ground. ARITH at 537 m: 24–29 % of each rung's new chunks, 16 % of all (0.172 against the leading
  edge's 0.898). A fitted `z` absorbs it only at the anchor's own height. Bench P8 measures the ask, the trailing edge
  apart, BEFORE the owner rules B-RATE.
- A straight dive at 528 m/s through the rung-0 ring (`h` = 500 m): `Σ 2π · h · w · z / A_L ≈ 920` a second for about
  1.8 s (ARITH at `z` = 1.6).
- REFUTED in the review: the sum may leave out the descent's 0.17 slack, because the code applies the slack only to a
  rung that the bound pulls in (`ladder_view.rs:709-711`).

| Budget | Value (on the instrument) | Derivation | Today |
|---|---|---|---|
| B-FRAME | (1) per stand, the p50 not worse than today's MEASURED p50 (11.3 / 13.4 / 11.4 / 9.5 ms at the berth / 6 376 / 11 499 / 16 954 km), the 40 B vertex included; (2) per stand and leg, the p99 ≤ 16.7 ms and at most 1 frame in 1 000 over 16.7 ms; (3) THE TARGET, reported and not yet a gate: p50 ≤ 11.0 ms at every stand, the berth included. | 60 Hz is 16.67 ms (the rate and the window's size are question 8). 11 ms keeps a third of the period for the game still to come (F6 item 2). The draft took 13.4 − 3.0 = 10.4 ms, but the 3.0 ms sheet was measured before the sky fix, the berth reads 11.3 ms and its sheet costs 0 there. So (3) waits for its lever: P7's GPU capture splits the berth's frame (32 M vertices, 9 136 chunks, the per-pixel sky, HDR and bloom about 5 ms), then E14 or a vertex lever. A card slice (E9, only on question 12) must fit beside the drawing: at 11 ms, a quarter of the period (4.2 ms) fits. The engine's upload of the harvested chunks is budgeted here too (about 180 MB a second at 990 chunks of 182 KB, ARITH; P6, P7). | p50 MEASURED; the p99 and the count UNMEASURED in the window; headless worst-of-second frames 17.2–20.3 ms median, 70.2 ms max; the descent's worst frame 33.4 ms |
| B-MAIN | The terrain's work on the main thread ≤ 1.0 ms mean and ≤ 2.0 ms p99 per frame. | B-FRAME leaves about 5 ms between the p50 and the period; the terrain may use 2 of them. | 7.7 ms mean, 33.4 ms worst at 528 m/s (the descent alone) |
| B-BUILD | The ask-weighted mean WALL time of one build, parents included, at 7 workers on the moving-eye legs: ≤ 6.5 ms; per rung reported. | 6.5 workers (7, less about half a worker for E8's descent thread) ÷ 990 chunks a second = 6.6 ms: B-RATE at the ask's midpoint. ONE REGIME (the review): the budget and its gate both read loaded wall time; the one-worker quiet costs of §10A.3 are the instrument that splits it (14.6 ms of wall time at 7 workers stood beside a 9.68 ms quiet warm median in another world). Question 11 asks whether V10's 8 ms becomes a per-rung line for the whole build. | 14.6 ms wall at 7 workers (528 m/s, before the solve); warm 9.68 / cold 33.14 ms median on one worker, cold p99 100 ms; 11 ms median at rung 9 with the artifact |
| B-RATE | The capacity at 7 workers ≥ 1.3 × the MEASURED ask at 528 m/s, 537 m over the solved stand (bench P8), sustained on a moving eye; the bounded ask inert at 240 and 528 m/s. | 1.3 ≈ 1 ÷ 0.77: a pool of 7 at 77 % use keeps the mean wait at about a quarter of a build (Erlang C, ARITH: 0.26). ARITH today: 680–1 340 a second (990 at the ask's midpoint); it also covers the dive's 920. Ruled on P8's number, not before. | 374 a second at 7 workers (MEASURED before the solve, `slice_08_ladder_discussion.md:2049`) |
| B-LAT | An urgent chunk, from its ask to its first drawn frame: p99 ≤ 250 ms. | At 528 m/s and 537 m the rung-0 ring gives (791 − 683 + 63) m ÷ 528 m/s = 0.32 s of warning (the ring's outer edge, its switch edge, the 0.12 s lead; ARITH), less two frames. | UNMEASURED |
| B-MEM | The footprint at the walk ≤ today's (bench P11) + 11 % of the drawn bytes (the 40 B vertex) − the sheet's bytes; no growth over a 10-minute soak; the absolute ceiling for the target machine's memory (question 9). | ARITH: 16 001 chunks × 182 KB (the berth's 3 503 vertices a chunk × 52 B: 40 B of attributes and 12 B of 16-bit indices, `crates/client-render/src/terrain.rs:2212-2219`) ≈ 2.9 GB drawn at 40 B, 2.7 GB at 36 B, before the sheet's and the floors' savings. The draft's 2.5 GB ceiling left the 40 B vertex out, and it is withdrawn. | 6.6 GB footprint at 6 659 chunks (MEASURED, before the packed normal); the walk UNMEASURED |
| B-EDIT | The digger: click to pixel p99 ≤ 180 ms on a 30 ms one-way link (ASSUMED); a pilot 50 km up ≤ 250 ms. | 2 × 30 ms + ≤ 20 ms to the tick + ≤ 20 ms to ship after durability + 0 ms (the first whole tick applies at once) + two build rounds, the parents then the children, 2 × 3.3–11 ms (MEASURED single builds) + a wait for a free worker ≤ one build's p99 (≤ 20 ms loaded after the levers, ASSUMED) + two frames (33 ms) = 140–175 ms (ARITH). The set is 4–18 chunk builds and 1–4 parent builds, by how many rungs the observer draws over the dig (ARITH; bench B-S10), on 7 workers at the head of the urgent class. The pilot adds a busier queue. The storage design's estimate of the lane is 73–123 ms plus one tick (`docs/investigation/2026-09-07/06_storage_diff_lane.md:788`). | UNMEASURED (no edit path) |
| B-LOGIN | Warm store and gateway: (1) the first frame in which every wanted column is drawn or stood in for ≤ 3 s after the world handshake; (2) zero urgent at full detail ≤ 15 s; (3) every planet whole at the level that its drawn rungs read ≤ 30 s. | (1) The head's first beat (≤ 0.5 s) carries the far map's top rung (26 KB) and the six top-rung chunks build in one round; the near tiles (91 tiles, 3.7 MB, at the 20 km stand) take 1.8 s at 2 MiB/s; the coarse-first builds of the rings over the eye are UNMEASURED. (2) 16 001 chunks ÷ 790–1 330 a second = 12–20 s (ARITH): 15 s needs about 1 070 a second. (3) BY ONE RULE: a far map is about ONE BYTE PER MACRO NODE (8.9 MB for 8 871 936 nodes), 0.64 × today's far data (14.0 → 8.9 MB). Today five planets were whole after 45.3 s at 2 MiB/s (MEASURED, all parts): about 90 MB if the pace bound it (ARITH), so about 58 MB and 29 s after; with E12, a planet 0.02 px wide needs only its top rungs. | 45.3 s for five planets MEASURED; the rest UNMEASURED |
| B-BAKE | The added server time of a first solve (the level map, the rim pass, the far bake), in WALL time on the production server's cores, ≤ the owner's ceiling (question 21; the recommendation: the planet's own solve time today). | A planet that nobody visited draws nothing until its head lands (§5.2). Today the home planet's first solve takes 205 s and the far planets' 77–382 s (MEASURED, `DEFERRED.md:9137-9139`). The rim pass is 2–84 core-minutes and the far bake 1–8 (ARITH on ASSUMED inputs), so the pass alone needs about 1 to 25 cores to stay under 205 s. | UNMEASURED |
| B-WIRE | The fill ≤ 2 MiB/s (today's pace); steady flight ≤ 128 KB/s for each client. | Steady, ARITH: tiles 0.04–0.08 KB/s (a 40 960 B tile every 993 s along each axis at 528 m/s, one or two axes; the draft's 0.2 KB/s was a slip); edits 4 KB/s at M-1's load; poses 61 KB/s (128 occupants × 20 Hz × 24 B, ASSUMED); times 2. | the pace MEASURED; steady UNMEASURED |

**What the one-volume model adds, inside these budgets** (ARITH): the fill and the top byte, about 1 ns a cell where the
kernel runs (§2.4); the wet share, one count a vertex (§1.2); the flat merge, one pass over the flat groups (§1.4); the
far fold at rungs 11–13, 0.04–0.6 ms a chunk for its uniform cells and at least 1.3 ms more for a coastal rung-13
chunk's mixed cells (§4.3; the 10 % mixed share and 10 ns a read ASSUMED); the box's third byte, 256 KiB in each
worker's pool slot (§1.1); 4 B a vertex (§10B.8). **What it removes:** the sheet, whose 3.0 ms a frame at 6 376 km was
measured before the sky fix (its share of today's frame UNMEASURED, P7); W17's ring, 8 % of a frame and 1 622 chunks at
the 20 km stand (MEASURED, `docs/design/owner_decisions_2026-09-21_water.md:1446-1453`); the coast chain's 17–19 node
reads a column (H2); the coast counts' 35.5 MB (ARITH, §7); one entity for each wet chunk; the floor chunks under open
water.

### 10A.3 The levers, and how the design meets each budget

| # | Lever | What changes | Expected gain | Risk | Law kept, and the gate that proves it |
|---|---|---|---|---|---|
| E1 | THE PARENT SUB-BOX FIRST; THE DRAWN PARENT ONLY WHERE IT HITS | On a parent miss with no queued sibling that needs the same parent, the worker builds only the parent's cells over the child's footprint and a two-cell ring (ARITH: about 36³ of the parent's 64³ cells, 18 %). When siblings are queued for it (the queue already orders by the parent's Morton code, `crates/client/src/ladder_view.rs:1833-1842`), it builds the whole parent once and they share it, as the single-flight claim does today. Where bench P5 shows that a drawn coarser chunk's parent form is asked inside the cache's history, the worker that builds the drawn chunk also keeps that form (positions, buckets, normals from the SAME box). | Removes most of H1 and H7. MEASURED today: a hit saves 83 % of the cold build (`12_chunk_budget_base.md:204-211`); the all-edges extraction costs what the owned one costs (`:251-253`). ARITH: a miss falls from a whole parent build to about a fifth of one. THE GEOMETRY (the review, and this round's ARITH): the trailing-edge asks (16 % of all at 537 m) have no drawn parent at all; even at the leading edge a rung-L child is asked about 1.8 × 2^L s after its drawn parent entered its ring at 528 m/s (1.1 · switch(L) ÷ v), against about one second of cache history (256 MB ÷ 239 KB ≈ 1 070 forms, about 990 inserts a second). So the drawn-parent hits come mostly from siblings and from fills (a login, a descent's first rings), and the MISS PATH carries the gain. The draft's "a finer chunk finds its parent built" and its 95 % line are withdrawn. | The sub-box must keep the parent's own origin and cell indices, or a target moves; its normals near the ring's edge must equal the whole parent's (P5 compares every morph metre). A drawn parent's form takes its border shade from its own triangles only, so a few border pixels can change (question 15). | Display only (the morph target is the client's arithmetic). P5; the picture gate and the pop gate (V16, V18). |
| E2 | THE BOX READS ITS NODES ONCE | `ColumnRead` reads a node window gathered once per box into a flat array (heights, levels, provinces). Each column reads its 16 and 64 words from the array. | Removes H2. ARITH: the column pass from 3.2 ms to 4 096 × (13 noise calls × 12.19 ns + 5 spline reads × 10–40 ns) ≈ 0.85–1.47 ms (the 13 calls and the 10–40 ns ASSUMED; bench P2 measures the read). A gather once a box was MEASURED for the retired drawn drainage: 18.7–28.7 µs a box, five to seven nanoseconds a column (`docs/design/DEFERRED.md:9531-9533`). The same window feeds the fill's `λ` (§2.4), the far fold (§4.3) and the shard's composition (§3.6). | A window past a face edge must follow `node_at`'s seam rule exactly (`crates/terrain/src/artifact.rs:1831-1870`). | SL10: the same kernel reads the same words. The plain per-column read stays as the test oracle; P2 and a unit test compare the two byte for byte; the pins move by no more than S2's own change. HR5: the gather is a monomorphic helper. |
| E3 | ONE COLUMN PASS FOR EACH COLUMN OF CHUNKS | A bounded cache of column fields, keyed by realm, face, rung, x, y and the artifact's digest; halo columns come from a neighbour's entry when it is held. Written in S2, on the new column read: today's column field carries the side and coast reads that S2 deletes (the review). | Removes H3. ARITH: about 37 % of the column passes, and 6 % of each box. | Memory: 4 096 × 60 B = 240 KB an entry (ARITH); the bound lives in the one config struct. | SL10 (a pure function of seed, address, charter and digest); a stale entry is refused like a stale build (`crates/client/src/chunks.rs:2418-2433`). The same pins. |
| E4 | THE PER-COLUMN COLLAPSE | `generate_in`'s skip rule, applied to runs inside a column: a run more than one cell over the column's top is gas, and a run over the ground's bound and under `λ` is liquid, by rule; a run under the strata's and the caves' reach is bedrock by rule; only the bands near a surface run the kernel. An open-water column whose ground bound lies more than a cell under the chunk needs no octave sum (§1.4). | Removes most of H4. ARITH: about 2 times fewer kernel calls at rungs 0–3 (the caves keep the rest), 8 times at rung 4, 16 times from rung 6. The sea is 78 % of the home planet (MEASURED, `DEFERRED.md:9521`); its open-water chunks need no octave sum and no kernel. | The reach bounds must be the ones the skip already uses (`crates/terrain/src/chunk.rs:617-630`). | SL10 and the skip's own law, "a skip writes exactly the bytes the cell pass would" (`chunk.rs:12-17`), tested per column like `a_skipped_chunk_is_byte_identical_to_its_cell_pass` (`chunk.rs:1252`), with the plain cell pass as the oracle. The same pins. |
| E5 | THE EXTRACTION VISITS ONLY CROSSED COLUMNS | The fill records each column's lowest and highest sign change. `extract_with` visits those cells only, in the same (c, b, a) order, and reuses a worker's table instead of filling 1 MB (§1.2 item 5). In S3. | Removes H5. ARITH: 64³ visits become about 4 096 × 4; 1.1–1.97 ms becomes about 0.3 ms. | The first-use vertex order must not change. | `mesh_pin` (the mesh digests), with the full scan kept as the test oracle (P4). |
| E6 | THE GEOMETRY WITHOUT MAPS | The skirts come from the box's border groups in one linear pass (no `BTreeMap`); each radial is computed once; `vertex_position` uses a fixed array and a sorting network in place of a `Vec` and a sort (the same sum order); after S2 the coarser-field reads go through E2's window; the ray tests run in f64 lanes (E13). | Removes most of H6. ARITH: 2.0–9.9 ms becomes about 1–4 ms. UNMEASURED until P1 splits the step. ITS REACH AT RUNGS 1–2 DECIDES B-RATE (the review): the geometry there is 8.56–9.72 ms against 4.09 ms at rung 0 (walk means, before the solve), and rungs 1–2 are 40 % of the ask. | A change of float order can move a pixel. | The integer positions are SL10's (`mesh_pin`); the floats are display: the picture gate and the pop gate (V16, V18). |
| E7 | THE ONE MESH | This design's own change: the sheet, its cuts and its entity go (§8.3); open water is one flat merge (§1.4). | The sheet's cost on today's frame is UNMEASURED (P7; 3.0 ms at 6 376 km before the sky fix). ARITH: 7 688 → 250–500 triangles an open-water chunk. | The merge's chord (MEASURED lighter squares, `DEFERRED.md:8876-8885`). | G-FLAT, G-WATERTIGHT, G-FRAME (§9). |
| E8 | NOTHING IN A FRAME GROWS WITH THE CHUNKS | (a) The descent runs on one of the terrain's own threads, inside the share (F6, B20), and publishes a whole wanted set; the main thread takes the newest set. (b) The release-and-request walk runs on ANY change of its inputs — a new wanted set, an arrival, a release, a tile landing, a cancel — over the difference of two sorted sets, and re-queues only keys whose priority changed. The hold waits on arrivals (`crates/client-render/src/terrain.rs:2993-2998` reads `is_resident`), so a walk run only after a new set would keep a coarser chunk drawn after its finer chunks land, until the eye moves 0.5 m; DEFERRED names the trigger: "only when an input changed ... a missed input is a hole" (`DEFERRED.md:8931`). (c) The harvest stops at a TIME budget derived from B-RATE: 990 ÷ 60 frames × the main-thread cost of one upload. At today's 0.07 ms that is 1.2 ms a frame, over B-MAIN, so the worker builds the engine's mesh buffers and the main thread moves them and clones nothing (target ≤ 0.03 ms an upload, UNMEASURED: 0.5 ms a frame). The engine's own upload lies outside that loop and is budgeted under B-FRAME (P6, P7). The harvest's byte budget grows with the vertex, as it grew by an eighth for 32 → 36 B (`terrain.rs:88-90`): +11 % for 40 B. (d) The stamp's heavy half only while a dev tool reads (owed, `DEFERRED.md:8930-8931`). (e) The placement is measured first (P6): at 16 001 chunks and about 50 ns a chunk it is about 0.8 ms (ARITH, most of B-MAIN); if P6 confirms it, a per-realm anchor that moves in whole kilometres, re-based over several frames. | MEASURED today: 7.7 ms mean and 33.4 ms max (the descent), 1.2 ms (the walk), up to 3.4 ms (the harvest loop, ARITH). Target: B-MAIN. | A wanted set one descent late: E11 covers it by derivation. The descent's thread costs about half a worker at 528 m/s (7.7 ÷ 16.7 ms, ARITH). An anchor's re-base must not burst in one frame. | SL10 clause 7 and F9 item 1: the descent reads delivered poses only; no prediction. The moving-eye gate; G-MAIN. |
| E9 | THE CARD AS A SECOND BUILDER — ruling F9's OPEN question | After E2 each box request can carry its node window (at most about 4 900 node words at rung 13, 49 at rung 0), so the card COULD build solved chunks. Today it builds none (`crates/client-render/src/terrain.rs:1343-1346`; `crates/client/src/chunks.rs:343-348`). | MEASURED in the shipped client beside three workers (F9): 95 boxes a second at a quarter of each frame AND at the whole frame (36 % of the harvest; about a tenth of B-RATE's 990); on cured code at the quarter, the 528 m/s leg's worst gap fell 144 → 44 urgent and its queue 1 550 → 401 (`docs/design/owner_decisions_2026-09-12_frozen_patches.md:400-408`, `:489-497`). The unwired chain's "about 490" a second took "about all of the card", which then could not draw (`DEFERRED.md:8397-8399`; the draft's "0.03 of one core" left that out). NOT counted in B-RATE. | MEASURED HARMS: the pop detector's widest step above the card-off flight on four legs of five (46 against 40 at 528 m/s), and the slow hull's band from holding to 3 357 urgent chunks, so "THE CARD STAYS A KNOB" (`frozen_patches.md:489-503`). A card box once differed from the CPU's (1 022 of 6 049 with a pool that grew, `DEFERRED.md:8380`). Its GPU time sits inside B-FRAME. | SL10 and F8: one source through rust-gpu (`crates/recipe-gpu/src/lib.rs:1-4`), no port. F9: a budgeted slice of each frame, the CPU as the default and the fallback, a detach on any difference. Question 12; bench P12 on pairs of flights; `gpu-drift` over every key of the stands. |
| E10 | WORKER MEMORY POOLS | Each worker owns its box, table and vectors, and reuses them (§1.1). | Removes H8: ARITH 1.2–3 GB/s of allocator traffic; the time gain UNMEASURED. | A pool that grows and never shrinks gave the card stale words (E9's row). Every slot is sized to its own box. | HR5, and the pool's own test (a small box after a large one). |
| E11 | THE SPEED MARGIN | The descent asks each ring wider by the eye's delivered speed × (the measured p99 latency + the descent's period), derived as `ASK_BOUND_SLACK` is derived today (`ladder_view.rs:545-550`). The speed is read as F9 item 1 reads it: "the PEAK of the lead's metres over the interpolation buffer's seconds — two DELIVERED poses, SL10 clause 7" (`frozen_patches.md:306-310`). In S3. | ARITH: at 528 m/s and 0.25 s, 132 m on the rung-0 ring's 791 m, so the ask grows by 17 % at rung 0. The late-ask residue (`slice_08_ladder_discussion.md:2064-2068`) goes. | More chunks are resident (the ring's added strip). | The owner's movement answer 2 (2026-08-27): "grow the interest radius with the closing speed; never cap the speed". SL10 clause 7 and F9 item 1: two delivered poses, no extrapolation. Question 13. |
| E12 | THE STREAMING ORDER | The gateway serves each realm's coarsest far levels and the near tiles first, and a far body only to the level that its drawn rungs read (the gateway alone knows each session's angular size of a realm, `06_storage_diff_lane.md:506-507`). The worker queue orders the urgent class by the time to need. In S3, with `ArtifactFar`: today's pyramid and coast parts go in S1 (the review). | ARITH: a first globe with no hole from about 26 KB, not 8.9 MB; a far planet 0.02 px wide needs its rung-18 cells, not its 11–23 MB of today's far data (MEASURED, `DEFERRED.md:10470`). | A body whose view grows must receive its finer levels before a finer rung needs them; the coarser rung stands meanwhile (F9). | SL6: no new data, only the order and a per-session filter on approved parts. SL3 and the owner's answer: the gateway "FILTERS per session what the realm chose to publish" (`docs/design/owner_decisions_2026-09-07_voxels.md:318-324`). |
| E13 | SIMD | The float geometry in f64 lanes (ray tests, normals, bounds). The integer recipe can gain little on the owner's aarch64: every recipe step is a 64-bit multiply (`slice_08_integer_bench.md:440-445`), and NEON has no 64-bit lane multiply (the instruction set; not measured here). | UNMEASURED. | A lane order changes a float. | Integers: SL10 and the pins on the aarch64 and x86-64 legs. Floats: the picture gate. |
| E14 | FEWER DRAW CALLS | Today each chunk is one entity. Merge a rung's chunks for each realm into shared buffers with indirect draws ONLY if P7 shows that the per-entity work is a real share of the frame; it is the berth's candidate lever. | UNMEASURED (the split is owed, `DEFERRED.md:8886-8888`). | The render seam must stay engine-agnostic (V4). | V4; G-FRAME. |

**The chunk after the levers, weighted by the ask** (ARITH, one worker on a quiet machine; the inputs are the walk
stand's means before the solve, `12_chunk_budget_base.md:139-148`, and the levers' ARITH above):

| Rung | Share of the ask at 537 m | Geometry today (mean) | E6 reaches every rung like rung 0 (× 0.37) | E6 does not reach rungs 1–3 |
|---|---|---|---|---|
| 0 | 45.8 % | 4.09 ms | 4.8 ms | 4.8 ms |
| 1 | 26.6 % | 8.56 ms | 6.4 ms | 11.8 ms |
| 2 | 13.7 % | 9.72 ms | 6.8 ms | 13.0 ms |
| 3 | 6.9 % | 2.04 ms | 4.0 ms | 5.3 ms |
| 4 up | 6.9 % | — | 2.5 ms | 2.5 ms |
| **Ask-weighted mean** | | | **5.3 ms** | **7.6 ms** |
| **Capacity, 7 workers (6.5)** | | | **1 330 (1 240) a second** | **920 (850) a second** |

Each rung's sum is the columns 0.85 ms (E2), the cells 1.5 ms (E4 at rungs 0–3), the fill 0.3 ms, the extraction
0.3 ms (E5), the geometry (the column above) and the parents 0.3 ms (a hit; about 0.7 ms on a miss through E1's
sub-box). At a spline read of 40 ns (the ASSUMED upper end) each build costs 0.6 ms more: 5.9–8.2 ms, 850–1 190 a
second on 7 workers. **So the capacity after the levers is about 790–1 330 chunks a second (ARITH: 6.5–7 workers, the
two geometry cases, 10–40 ns a read), against B-RATE's 680–1 340.** At the ask's midpoint (990 a second on 6.5 workers,
6.6 ms a build) B-RATE holds only if E6 cuts the geometry at rungs 1–2 to about 6.5 ms or less (ARITH: the other rungs
take 2.62 ms of the mean, so 0.403 × x ≤ 6.57 − 2.62 ms gives x ≤ 9.8 ms for the whole rung-1/2 build) and the spline
read stays near 10 ns. Bench P1 splits the geometry and bench P2 times the read before the owner rules B-RATE. The
draft's "about 1 460" (rung 0 alone) is withdrawn. B-BUILD's loaded regime is not this table's quiet one; P0 and P1
measure both on one tree.

**Which levers meet which budget.** B-FRAME: E7, the harvest's buffers of E8(c), E14 (the berth's lever, after P7).
B-MAIN: E8. B-BUILD: E1–E6, E10, E13. B-RATE: B-BUILD (E9 only on question 12). B-LAT: E8, E11, E12. B-MEM: E7, E10,
a bounded upload ring (DEFERRED item 14). B-EDIT: E2, E4, E5, the cached box of §3.6, and the edit tick's set at the
head of the urgent class. B-LOGIN: E12, the far map coarsest first, B-RATE. B-BAKE: the fixed partition over the
server's cores. B-WIRE: E12.

**Example.** A pilot at 528 m/s, 537 m over the plain. About 525–1 030 new chunks come into her rings each second
(ARITH), two fifths of them at rungs 1 and 2, which cost about twice a rung-0 chunk today. Today seven workers build about
374 a second, so the bound pulls the finest ring in and full detail arrives late. With the levers they build about
790–1 330 a second: enough at the ask's midpoint only if the geometry of rungs 1–2 also falls. The descent runs on a
terrain thread, so her frame never waits 33 ms for it.

### 10A.4 The target machine (F6)

ARITH on an eight-core machine (B20: four workers, "four on an eight-core machine",
`docs/design/owner_decisions_2026-09-22_believability.md:568-569`): 4 ÷ 5.3–7.6 ms = about 520–760 chunks a second, at
or under the ask's midpoint (758 at 528 m/s and 537 m). There the bounded ask pulls the finest ring in at 528 m/s:
lawful and seamless (F9 item 1: "the detail at speed drops on a weak machine; the seamlessness does not"), but with less
detail. A weaker GPU changes B-FRAME. So every budget is ruled twice: for the instrument now, and for the target machine
when the owner names it (question 9). The first draft asked only for the machine's memory. Whether the builders may ask
the system for the fast cores is question 14, after bench P10.

### 10A.5 The benches and the gates

The benches P0–P14 are in §10. The gates G-FRAME, G-MAIN, G-BUILD, G-RATE, G-LAT, G-POP, G-MEM and G-SOAK, G-LOGIN,
G-EDIT, G-WIRE and the byte gates are in §9's one table, each with its failing control: the same flight with the lever
switched off must fail the gate where today fails it, or the gate proves nothing.

### 10A.6 The order

In §9. S0 takes P0–P14 beside M1–M11 and B-S1–B-S11, and the owner rules the budgets on their numbers. S2 writes E2, E3
and E4 into the new column read and cell pass from the start (B21: "don't try to connect": no lever is fitted onto the
shore chain that S2 deletes). S3 writes E5 into the top extraction and carries E7, E11 and E12. S0b (E1, E6, E8, E10: the
client's builder and frame; no world byte moves) comes where question 10 puts it: the recommendation keeps B20 item 1's
"THE COAST FIRST" and puts S0b right after the epoch, because S3 rewrites the same builder code. E9 comes only on question
12's word; E13 and E14 only after their benches.

---

## 10B. NO SEAMS, NO JUMPS, NO FLICKER — THE MECHANISMS AND THE GATES

*The second round, for the owner's requirement 3: "Also while performant we should not build seams. Also there are should
not be jumps or flicker between rungs". This section builds nothing and measures nothing. Its marks are the header's;
`path:line` is read in the code on this branch. It completes §4.4, and the sections that it changes say so in place. The
review's findings on the draft (§11, the second round) are answered here, each in place.*

**The bar.**
- SL8: a seam is a defect, and its tolerance is physical: one pixel of the reference view, 0.0011506 rad (ARITH:
  2 · tan 22.5° / 720, `crates/core/src/geometry.rs:1160-1173`).
- W16 items 1–3: one shape from one field; a finer rung refines a coarser one and never contradicts it; a rung boundary
  is invisible (`docs/design/owner_decisions_2026-09-21_water.md:1016-1022`).
- B16: "No flicker or changing of the shapes between rungs at all" (`docs/design/owner_decisions_2026-09-22_believability.md:455-466`).
- B18 Q5 read the owner's same words, "no jumps or flicker between rungs", as met where "no rung decides it" and the
  change is gradual (`owner_decisions_2026-09-22_believability.md:514-515`). B18 Q4 accepted near slides at sizes
  measured under N2, "judged in the flight" (`:511-513`). §10B.7 states what that leaves to the owner.

### 10B.1 The two properties

Every replacement of one drawn thing by another must be:
1. **CONTINUOUS.** No pixel changes in one frame by more than the picture's own noise. No step, hole, crack or wall. No
   colour or brightness step. No water appears or vanishes in one frame.
2. **STABLE.** Nothing oscillates for an eye that stands still, or that moves slowly on a boundary. Every decision has
   hysteresis, or both sides of the decision draw the same pixels.

**Example.** A walker stands on the belt's beach at the 15° stand and looks at the bay for one minute. Every frame shows the
same picture, except for the sun's slow motion. Then she steps 2 m forward and back across the band where the bay hands
over from rung 0 to rung 1 (782–956 m away, ARITH). The water's colour follows her steps smoothly, and her client asks for
no chunk twice.

### 10B.2 The mechanisms that stay (today's code)

| Mechanism | What it guarantees | Evidence | Measured |
|---|---|---|---|
| The crossfade band | Rungs L and L + 1 are both drawn over 0.9–1.1 of the switch distance. The weight is a clamp of each vertex's own distance: no decision, so nothing flips. The band's edges are the hysteresis. | `crates/client/src/ladder_view.rs:37-53`, `:81-84`; `crates/terrain/src/artifact.rs:908`; `crates/client-render/src/ladder_fade.wgsl:114-125` | A dither in its place left 116, 895 and 47 holes (`ladder_view.rs:47-51`) |
| The morph | A finer vertex slides on its own radial onto the coarser rung's MESH, and its shade onto the parent's shade at the hit. At the band's far edge the finer lies on the coarser. | `ladder_fade.wgsl:137-181`; `crates/client/src/chunks.rs:686-718`, `:1308-1400` | Without the shade's morph, 13 % of the band's pixels stepped up to 48 levels at 240 m/s (`chunks.rs:132-139`); the field as the target left 34 dark specks (`chunks.rs:449-459`) |
| The far edge | A finer fragment is discarded at the band's far edge, where it coincides with the coarser. | `ladder_fade.wgsl:188-196` | — |
| The sink | The coarser rung stands under the finer by the dropped step plus a cell of each rung, and rises to its own radius at the finer's far edge. | `chunks.rs:58-77`; `ladder_view.rs:99-128`, `:764-772` | A sink that ended past the edge drew a ring 347 km wide at the wrong altitude (`ladder_view.rs:104-114`) |
| The skirts | Under every boundary edge a strip hangs into the rock on its triangle's side, as deep as the coarser sink plus a cell. | `chunks.rs:1693-1807` | Before: one crack pixel at 970 m (`chunks.rs:1719-1721`), skirt bottoms 1.50–25.36 m over the sunk coarser ground; after: none (`docs/investigation/2026-09-24/terrain_believability_status.md:188-214`) |
| The band table | One table per realm. Every terrain material reads its row. The render world rewrites it every frame, in step with the placements. | `ladder_view.rs:143-168`; `ladder_fade.wgsl:62-78`; `crates/client-render/src/terrain.rs:2839-2853` | The 5 % band jumps and the water's stale bands are gone (`docs/design/DEFERRED.md:10238-10247`) |
| The arrival ramp | A late chunk starts on its stand-in's drawn surface and shade, and rises over 0.5 s. The start is exact: rung L's wholeness is rung L + 1's unrisen share at every distance. | `crates/client/src/arrival.rs:26-126`, the proof `:171-184`, the control `:186-222`; `ladder_fade.wgsl:80-88`, `:150-159` | Still eye, about 1 000 chunks landing: pixels past 3 levels 1 040 → 78 (`DEFERRED.md:7898-7901`) |
| The release hold | A drawn chunk leaves only when every wanted chunk over its footprint has arrived. | `terrain.rs:2973-3019`; `ladder_view.rs:1273-1299` | Before the descent: 4 818 and 16 297 hole pixels (`ladder_view.rs:1606-1609`) |
| The ask bound | A rung's horizon slides by a quarter of its length a second and never jumps; a move under 0.5 % is not adopted; the descent's slack covers its bracket (asserted at compile time). | `ladder_view.rs:360-385`, `:500-550`, `:656-698` | — |
| The culls' hysteresis | The skyline keeps a column within 0.05 rad and wants a new one within 0.01 rad; a culled column stays culled until 0.75 of the horizon. | `ladder_view.rs:1317-1340`, `:1388-1398` | Before: 4 chunks missing on 80 of 1 383 samples (`ladder_view.rs:1321-1323`) |
| The realm's picture | The scene swap is atomic; a realm missing from a swap level is not forgotten; a ladder outlives a missing row by one interpolation buffer; the gateway draws only whole ticks (B19 item 5). | `crates/client/src/net.rs:311-352`; `ladder_view.rs:1908-1925`; `terrain.rs:2928` | The walk's 203 gap frames from blinking planets → 0 (`DEFERRED.md:10459-10492`) |

### 10B.3 Every replacement

For each place where one drawn thing replaces another (SW1–SW12, a swap each): why it is CONTINUOUS, why it is STABLE,
what this design ADDS (the tags point to §10B.4), and what stays (the X numbers point to §10B.6).

| # | What replaces what | Continuous by | Stable by | Adds | Residue |
|---|---|---|---|---|---|
| SW1 | A ground chunk and its coarser copy, rungs 0 → `L_ref` | The morph of position and shade; the sink; the far edge. The step is the octaves that the coarser rung drops, under one cell of it (MEASURED 0.64 px, `crates/terrain/src/body.rs:190-193`). | A continuous weight; no decision. | [TOP] | The crease's sagitta: at most a cell at the far edge (X18). |
| SW2 | A water chunk and its coarser copy, and the wet colour | Over open water both tops stand at the level, so the morph is nought. At a shore the top slides onto the coarser top, and the share blends to the coarser share. | The same weight. | [TOP], [WET MORPH], [MERGE] | Below `L_ref` a shore slides across the band (§10B.7); a pond can show, hide and show (X11). |
| SW3 | The generator at `L_ref` → the far fold at `L_ref` + 1; the client's fold at rung 13 → the far map at rung 14 | The same morph. A fold cell is the mean of the reference cells under it, rounded once, so a parent is the mean of its children to 1 m and 1/255 (§4.3). | No decision. | [WET MORPH], [FAR SINK] | The step is the top's variation inside one coarser cell (bench B-S6). |
| SW4 | A far fold rung → the next, up to rung 18 | The same morph. A coast holds its place per cell; its soft edge only widens. | No threshold exists. | [WET MORPH] | A pixel shows its parent's share: a monotone blur inside two pixels (§4.3; question 20). |
| SW5 | A late chunk: ground, water, colour | The ramp from the stand-in's surface and shade, and now from its share. | A clock, never a decision. | [WET MORPH] (the wet ramp) | Over water a late chunk rises by the stand-in's sink: ARITH 15.6 m at rung 2, about 4–5 px at 3 km from an eye 100 m up (X9). |
| SW6 | A chunk rebuilt at its own key | Byte-identical where nothing changed (one recipe, SL10). | Deterministic. | [STAMP] | X4. |
| SW7 | A chunk edge at one rung | Neighbours share vertices in exact integers (`crates/terrain/src/extract.rs:1-31`); the skirts close the floating origin's rounding. Two rungs never share an edge: the band overlaps them (`ladder_view.rs:44-47`). | Deterministic. | [SKIRT SHARES] | X9. |
| SW8 | A cube-face edge and a cube corner | One face owns each seam edge; three faces compute one prism vertex at a corner (`extract.rs:13-26`). | Deterministic. | [SEAM TARGET] (the review's cure); [TOP] where a field still answers | The crease's sagitta (X18). |
| SW9 | An edit: its chunk and its coarser copies | Today each key swaps whole when its own rebuild lands. | — | [EDIT TICK] | The edit itself appears in one frame at every rung: a change of the world, not of a rung (question 18). |
| SW10 | A realm hand-over: a pilot flies from the planet's realm into the system's | The scene swap is atomic; the ladder, the band table and the parent cache are keyed by the body's realm; the eye in the body's frame stays continuous, because the parent authors the placement (SL1). | The row's grace. | [BOOK GRACE]; gate G-CROSSING-TERRAIN | — |
| SW11 | A solved artifact arriving; a world epoch | No chunk of a solved body builds on the recipe's relief (§5.2; C13), so today's patchwork of old and new chunks (`DEFERRED.md:9122-9145`) cannot form. | — | [EPOCH] | An artifact change under a drawn body (development only) is one declared change of the whole realm (question 17). |
| SW12 | A late shadow caster | It enters whole today (`crates/client-render/src/ladder_fade_prepass.wgsl:125-127`): its shadow appears in one frame. | — | [CASTER RAMP] | — |

### 10B.4 What the design adds

**[TOP] TOP ONTO TOP.** The parent mesh (`crates/client/src/chunks.rs:544-641`) is the coarser rung's TOP mesh after the
flat merge: the mesh that rung draws. Every fallback reads the coarser TOP, `max(ground, λ)`: `coarser_field_m`
(`chunks.rs:1247-1261`), `field_normal` (`:1175-1196`) and the face-seam rule (`:1325-1341`), where [SEAM TARGET] does not
reach. Today they read `height_field_m`, which is the GROUND (`crates/terrain/src/height.rs:115-122`).

**[WET MORPH] THE WET MORPH AND THE WET RAMP.** Each vertex carries two shares: its own, and the TARGET share. The target
share is the parent's share at the radial's hit, weighted as the shade is weighted there (`chunks.rs:713-717`), or the
coarser top's wet test where a field answers. The shader blends the two exactly as it blends the shade
(`ladder_fade.wgsl:180`): `share = mix(target, own, whole(d) · ramp)`. So at the far edge the finer's share IS the
coarser's, and a late chunk starts with its stand-in's colour. The probe (`crates/client-render/src/probe.wgsl`) reads
the same blend, so its kind follows the picture.

**[MERGE] THE FLAT MERGE KEEPS THE MORPH.** A vertex joins a flat block (§1.4) only when its morph metre is within one
vertex quantum of nought and its target share equals its own: the coarser rung is flat water at the same level there too.
A block keeps every original vertex on its rim, so no crack opens against its neighbours.

**[PARENTS] A CHUNK WAITS FOR ITS PARENTS' FIELDS.** A chunk builds only when the field of every parent that it reads is
whole (`chunks.rs:1106-1141`). So no vertex keeps its own radius for a missing tile (`chunks.rs:1366-1372`).

**[STAMP] A REBUILT KEY KEEPS ITS RAMP.** A rebuilt key keeps the arrival stamp of the entity that it replaces while that
ramp runs. Today a rebuild gets nought and enters whole (`crates/client/src/arrival.rs:42-51`;
`crates/client-render/src/terrain.rs:3215-3225`).

**[EDIT TICK] THE EDIT TICK** (corrected in the review: the draft counted the parts per tick, which fails under the
gateway's per-session filter). `ChunkRows` state, for each (tick, rung), its part count, and each tick states the mask
of the rungs that it publishes (SL6 ask A9; a `ChunkRow` already carries its tick, `crates/core/src/chunk_row.rs:8-10`).
Each `BulkFor` carries one rung and states it in a plain header, so the gateway's per-session filter — the owner's own
answer (`docs/design/owner_decisions_2026-09-07_voxels.md:318-324`) — forwards a (tick, rung) whole or not at all: each
filtered subset is whole by itself, and the realm needs no per-session state (one window serves every session behind a
gateway, `docs/investigation/2026-09-07/06_storage_diff_lane.md:500-503`). The client applies a tick when every
(tick, rung) of the mask that it holds over the dig is whole. It evicts the touched parent meshes, rebuilds every
touched chunk and every finer chunk that reads a touched parent, and swaps the whole set in ONE frame. THE JOIN: while a
set builds, a newer whole tick that touches a key of the set joins it; the set restarts with both ticks' rows and swaps
once. A join window in the client's tuning bounds the delay under steady digging: when it ends, the set takes no more
ticks, swaps when its builds end, and the next ticks form the next set. The first whole tick of a burst never waits for
the window (B-EDIT). Edit rebuilds ride the head of the urgent class.

**[EPOCH] THE ARTIFACT EPOCH.**
1. A body with a macro lattice draws nothing until its head and the top rungs of its far map land (DEFERRED's road (a),
   `docs/design/DEFERRED.md:9129-9133`).
2. The far map ships coarsest first. Rung 18 is 8 664 cells, 26 KB (ARITH: 6 × 38² × 3 B), so it lands in the head's
   first beat. The body's first frame holds all six top-rung chunks at once: never a globe with a face missing.
3. An artifact change under a drawn body keeps the whole old set drawn until the new set's band is whole, then swaps the
   realm in one frame (S5-2: a new world).
4. A body enters a window at its reach. By the size rule that is 1.5°, about 23 px (ARITH: 0.02618 / 0.0011506;
   `crates/core/src/geometry.rs:1140-1146`), so its first frame is a pop-in of the REACH, not of a rung. The reach ruling's
   own cure is the brightness radius (`docs/design/owner_decisions_2026-09-02_reach.md:79-81`, `:234`). ONE RULE for
   §10B and §10C (the review): gate G-FAR-FIRST starts its clock when the first far part is HELD by the client, never
   when the body enters the window, because the gateway serves the artifact only after that (at least one 0.5 s beat,
   `crates/connection-plane/src/gateway/artifact.rs:41-50`; 77–382 s when the shard solves for the first time,
   `DEFERRED.md:9137-9139`).

**[HYSTERESIS] HYSTERESIS AT EVERY DECISION.**

| Decision | Where | Hysteresis |
|---|---|---|
| Which rung draws a pixel | `ladder_fade.wgsl:114-125` | None needed: a continuous weight. |
| Split a column; draw it at a band's far edge | `ladder_view.rs:1499-1515` | NONE today. Invisible (the weight is nought there), but it wastes builds. ADD a keep margin of one column span of the rung. |
| Release a drawn chunk | `terrain.rs:2993-2998` | The hold. |
| The skyline's and the horizon's culls | `ladder_view.rs:1738-1742`, `:1778-1787` | 0.05 / 0.01 rad; 0.25 of the horizon. |
| The deliverable horizon; the descent's re-run | `ladder_view.rs:500-550`, `:645-698`; `terrain.rs:105-107` | 0.5 %, a quarter of a length a second; the 10 % bracket with its 0.17 slack; 0.5 m of eye motion. |
| The sun's shadow map on or off | `terrain.rs:3435-3449` | None, and none needed: when it is off, no ground stands inside the cascades. |
| A realm's ladder; a realm's artifact | `terrain.rs:2928`; `net.rs:375-378` | One interpolation buffer; the artifact NONE today. ADD the ladder's grace to the artifact ([BOOK GRACE]). |
| A chunk's field ready | `crates/client/src/artifact_book.rs:175-189` | Monotone: tiles only arrive. [PARENTS] adds the parents' fields. |
| Wet or dry; a flat block | §2.4; §1.4 | Static per cell and rung: no runtime flip. |

**[FAR SINK] THE FAR SINK.** Above `L_ref` no octave drops, so today's formula gives two cells (`chunks.rs:69-77`). The
fold's own step is the top's variation inside one coarser cell. If bench B-S6 reads it under one cell, the two cells
stay. Otherwise the body states that step for each far rung, and the sink and the skirts read it.

**[SKIRT SHARES] THE SKIRTS IN THE ONE MESH.** A skirt copies its edge's two shares, as it copies its normals and its
morph (`chunks.rs:1771-1792`). The wet share counts every crossed edge of a group, owned or not, as the vertex position
does (`crates/terrain/src/extract.rs:4-10`; `chunks.rs:449-459`). So two chunks give a shared vertex one colour, at a chunk
edge and at a face seam.

**[SEAM TARGET] THE FACE SEAM READS THE NEIGHBOUR FACE'S PARENT** (new in the review). A vertex on a cube-face edge
never reads a parent mesh today: it reads the coarser field from both faces (`chunks.rs:1325-1341`), so its target
stands up to a cell off the coarser surface (DEFERRED item 6, `docs/design/DEFERRED.md:7910-7913`), and under
[WET MORPH] its target share would be a binary wet test against the coarser rung's fractional share: a colour step at
the far edge on every coastal seam. 223 coastal node pairs straddle a cube edge, and one cube corner is coastal
(MEASURED, `docs/investigation/2026-09-24/terrain_believability_status.md:1009`). So the seam vertex reads its target —
position AND share — from the neighbour face's parent mesh, the exact cure that DEFERRED names ("the extractor already
maps cross-face sites"). A missing neighbour parent is built by E1's sub-box. Gate G-FAR-EDGE then holds seam vertices
with no exception.

**[BOOK GRACE] THE BOOK FORGETS ON THE LADDER'S GRACE.** The client's artifact book forgets a realm after one
interpolation buffer, as the ladder does, never on one scene delta (`crates/client/src/net.rs:375-378`).

**[CASTER RAMP] A LATE CASTER RAMPS.** A late shadow caster ramps from its stand-in caster, or the finer chunks keep
casting until it lands.

### 10B.5 The rule for every speed lever

A lever may make the picture coarser or later. It may never make it discontinuous.
1. **A lever changes no world byte.** E2–E5, E9 and E10 give the bytes of the plain form of the same kernel, which stays
   in the tests as the oracle (P2–P4, P12). On today's code the pins (`terrain-pin`, `mesh_pin`, `artifact-pin`,
   `gpu-drift`, G-NO-DRIFT) stay green without a re-record; inside the epoch they move by the epoch's own change and by
   nothing more. A lever that changes a byte is a world change: it waits for the owner and the pictures.
2. **A lever that changes a float of the drawn mesh** (E1's border normals, question 15; E6; E13) passes the picture
   gate inside its measured noise floor (V16, V18) and the pop gate on every leg, or it is refused.
3. **No drawn band is ever unasked.** The descent's slack stays derived (`crates/client/src/ladder_view.rs:545-550`). E8's
   late set and E11's margin join that derivation, and a unit test walks the worst stand at every rung.
4. **A late chunk never pops.** The coarser rung stands whole under it (F9), and the late chunk rises over 0.5 s
   (`crates/client/src/arrival.rs:27`; B18 Q7). No lever removes a stand-in to save memory or time.
5. **No band jumps.** The bounded ask slides at a quarter of its length a second (`ladder_view.rs:378`). A lever never
   moves a band in one frame.
6. **A frame over the period is a jump in time.** B-FRAME's p99 and its count of frames over 16.7 ms belong to this
   requirement. The descent's 33.4 ms frame is the largest known jump today, and E8 takes it off the main thread.
7. **The far fold stays exact above `L_ref`** (§4.3). A faster fold (E2's window) gives the same rounded means
   (G-FAR-EXACT).
8. **Late, never broken.** The worker share (B20: half the cores) and the harvest's budget make chunks late, and a late
   chunk ramps (SW5). The parent cache rebuilds a mesh byte for byte, one build at a time (`crates/client/src/chunks.rs:914-965`).
   The culling boxes hold every place the vertex stage can draw, the ramp's start included (`chunks.rs:300-328`). The
   shadow switch changes no pixel (`crates/client-render/src/terrain.rs:3435-3449`).
9. **Refused by measurement, and never again a speed trick:** the 16-bit position (the crease flips,
   `docs/design/DEFERRED.md:7996-8003`); the coarser FIELD as the morph target (dark specks, `chunks.rs:450-459`); a
   removed skirt (cracks, `chunks.rs:1715-1725`); a flat block whose chord dips past the extractor's step (lighter
   squares, `DEFERRED.md:8876-8885`). **Refused by reason, not by measurement** (the review: no measured frustum cut
   exists): a view-frustum cut in the descent. The descent reads the eye point only, never a facing
   (`docs/investigation/2026-09-08/landforms/12_chunk_budget_base.md:48-49`), so a turn would show holes.

### 10B.6 The adversarial search

| # | Finding | Evidence | Cure, or the price |
|---|---|---|---|
| X1 | The morph's fallback reads the GROUND. In the one mesh a water vertex on a cube-face seam would morph down to the sea floor inside every band: a trench along every coastal face edge. | `chunks.rs:1247-1261`, `:1325-1341`; `height.rs:115-122` | CURED by [TOP] and [SEAM TARGET]. |
| X2 | A near shore pops at a ring's front today: the sheet morphs with its own rung's side, and the parent holds no water. | `DEFERRED.md:10248-10252` (🟥) | CURED by [TOP] and [WET MORPH]. |
| X3 | At a band's far edge two surfaces coincide by construction, and the depth test picks either one. Position and shade agree today. A share that did not agree would speckle a coast for a moving eye. | `ladder_view.rs:115-119`; `ladder_fade.wgsl:188-196` | [WET MORPH] makes the share agree; [SEAM TARGET] makes the seam vertices agree; gate G-FAR-EDGE proves all three, seams included. |
| X4 | A rebuild enters whole, also in mid-ramp: a chunk rebuilt 0.2 s after it landed jumps by 60 % of its rise. Rare today; every edit makes it common. | `arrival.rs:42-51`; `terrain.rs:3215-3225` | [STAMP]. |
| X5 | An edit's rungs can land in different frames, and a parent mesh leaves the cache only by its budget or with its whole realm. A finer chunk then morphs onto a coarser surface that no longer stands. | `chunks.rs:996`, `:1055-1066`, `:2255-2271` | [EDIT TICK], counted per (tick, rung) with the tick's rung mask. |
| X6 | The coarse edit entry's DOMINANT solid is a threshold. When 8e paints substances, a mixed coarse cell swaps its colour at a band: the fold that B16 refused. The first slice draws one ground colour, so nothing shows yet. | §3.4 step 6 | The entry carries the top's material SHARES, blended as in [WET MORPH] (SL6 ask A5, widened, with 8e). |
| X7 | The flat merge drops the vertices that must morph. Where the coarser rung holds an island inside a finer flat block, the island pops at the far edge. | §1.4 | [MERGE]. |
| X8 | A chunk built before its parents' tiles keeps no morph, and a tile's arrival never rebuilds it (the head's digest does not change). | `chunks.rs:1366-1372`, `:2277-2297` | [PARENTS]. |
| X9 | A skirt under a water edge hangs into the water. Opaque water hides it. While a finer neighbour is missing, it closes a water-coloured step of the stand-in's sink. With 8o's clear water and a diver it becomes a curtain. | `chunks.rs:1756-1768`; `DEFERRED.md:10370-10374` | PRICE now: the band gap's transient (bench B-S8). OWED to 8o: a skirt of one vertex quantum under a wet edge. A ceiling skirt through a roof thinner than two cells stays owed to slice 9. |
| X10 | Without the sink, two rungs' water tops are coplanar in the band and z-fight. | `ladder_fade.wgsl:15-25` | The sink stays over water; its price is X9. |
| X11 | Below `L_ref` a pond made by one octave can show, hide and show across three bands. A cure inside (B), "wet only where every coarser rung is wet too", makes water monotone, but then a dry hollow under `λ` beside a wet column is a vertical water face (G-NO-WALL): REFUTED. | §4.5 | A FLICKER under B16, not a price (the review): gate G-POND-FLICKER at 0 on the stands and a planet sample, or the owner's amendment of B16, or option (A); a smaller K makes it rarer, not zero (question 1). |
| X12 | The rim pass moves a level only at a bake, and an edit never re-solves (W2). | §2.3 | No runtime seam. A level changes only with a world epoch ([EPOCH] item 3). |
| X13 | The far fold's soft edge widens by two at each far swap. That is monotone: no threshold, so no flicker. [WET MORPH] makes it gradual. | §4.3 | None. |
| X14 | A late shadow caster enters whole: its shadow appears in one frame. | `ladder_fade_prepass.wgsl:125-127` | [CASTER RAMP]. |
| X15 | The band edge's want and split have no hysteresis: an eye that sways ±1 m there asks, releases and asks again. Invisible, but builds are wasted (requirement 1). | `ladder_view.rs:1499-1515` | [HYSTERESIS]. |
| X16 | The book forgets a realm's artifact on one delta, while the ladder waits one buffer. A one-tick absence at a realm hand-over costs the head's far data (8.9 MB after this design) and the tiles again, and a band gap meanwhile. | `net.rs:375-378` | [BOOK GRACE]. |
| X17 | The pop detector is a report, and it is blind to water. The gate prints it and asserts nothing; the sheet has no probe twin, so water reads as "nothing", and from 100 m over the sea every water pixel under the far land counts as a hole. | `crates/bins/tests/terrain_moving_eye.rs:1622-1643`, `:2552-2561`; `terrain.rs:3240-3269` against `:3289-3316`; `crates/client-harness/src/pop.rs:282`; `crates/client-harness/src/probe.rs:165-219` | The one mesh gives the water a twin; the probe gains kinds (G-PROBE-KINDS); G-POP is asserted with the lines of X21. |
| X18 | The face seam's field target and the far edge's crease sagitta. | `DEFERRED.md:7902-7913` | The face seam: CURED by [SEAM TARGET]. The crease: at most a cell at the far edge; its exact cure (split the finer triangles that straddle a coarser crease) is built only if bench B-S1 shows it on the stands. |
| X19 | A top-rung chunk has no stand-in and enters whole. | `arrival.rs:42-51` | [EPOCH] items 2 and 4: all six at once, at the reach's size. |
| X20 | 8o's tide and waves move the water's top with time. Baked into cells, they would rebuild chunks every tick, and two rungs would disagree between rebuilds. | §2.5; T3 | OWED to 8o: one closed-form function of (direction, tick) moves every wet top in the vertex and fragment stages, the same at every rung; never a cell and never a rebuild. |
| X21 | (the review) G-POP's proposed line — at every boundary no more past the floor than the still pixels' 0.1 % — fails on today's code on every leg: the walk 173 of 25 914 (0.67 %); 240 m/s 700 of 334 352 (0.21 %, rung 6 0.36 %); 528 m/s 2 375 of 242 098 (0.98 %, rung 6 1.6 %). No cure above targets these ground boundaries, and both descent legs compared 0 pixels, so a descent passed by default. | `crates/client-harness/src/pop.rs:54` (`FLOOR_SHARE` 0.999); `docs/investigation/2026-09-26/r1_renderer_pops/r1_flights.txt:25-29` | OPEN as a cause: P0 and B-S5 find it on pairs of flights. The gate NOW: no leg and no boundary worse than P0's baseline read by the new probe, and at least 10⁶ compared pixels on every leg. The 0.1 % line becomes the target only after the cause is named and the owner rules. |

### 10B.7 The far-coast question, restated

**The sizes.**
- **(B), for ground farther than the 9 → 10 band (400–490 km, ARITH):** no slide. The fold holds a coast per cell, to 1 m
  and 1/255 (§4.3), and no lake flickers. Today B15 MEASURED 53 of 714 far lake bodies flickering at rung 14.
- **(B), nearer:** a shore slides across each band. The one-volume sizes are UNMEASURED (bench M4). The nearest numbers:
  under N2, the sea ≤ 1.46 px and the lakes ≤ 3.29 and 4.79 px (MEASURED, B18 Q4); the older medians 3.7 px at 6 → 7 and
  6.6 px at 7 → 8 (ARITH from MEASURED metres, §4.5). Ponds that show, hide and show: UNMEASURED.
- **(A):** exact at every distance. Its price: new small hills (the look is judged again); 57–150 MB stored for each
  planet; a bake of about 20–110 core-minutes; 8.8–9.5 MB more in the head; 17–79 MB of GPU windows for each stand
  (MEASURED, B17); and still a water step of up to 1.19 m under 32 m (§4.5).
- **Question 2 (a smaller survival constant):** each hand-over's step is 6–12 times smaller in cells (ARITH, §4.6), and
  `L_ref` moves to 12 or 13, so the exact far view starts at about 3 500 or 7 000 km.

**What the owner's words settle.**
1. "No jumps": every replacement must be continuous. Both options meet it only with §10B.4. Without [WET MORPH] a near
   shore pops today (X2).
2. "No flicker": nothing may oscillate for a still or a slow eye. Both options meet it by determinism and by
   [HYSTERESIS]; G-STILL and G-SLOW prove it.
3. A gradual, one-way slide is neither a jump nor a flicker by B18 Q5's reading of the same words ("no rung decides it",
   a gradual change). But the item that accepted near slides is B18 Q4 (the review; the draft cited Q5 for it), and it
   accepted SIZES — 1.46, 3.29 and 4.79 px, measured under N2, which this design retires — "judged in the flight". And
   B16 says "No flicker or changing of the shapes between rungs at all". So whether (B)'s new slides are lawful is NOT
   settled here: it is question 1, with B16's words beside it and G-SLIDE's line for the owner to rule.
4. A pond that fades out at one band and back in at the next is a flicker: B16 item 1 names "show, hide and show", and
   W16 item 2 says that a finer rung never contradicts a coarser one. So (B) with such ponds needs the owner's amendment
   of B16. ONE POSITION for §10B and §10C (the review): no price is offered for them; G-POND-FLICKER reads 0, or the owner
   amends B16, or (A) is built; a smaller K makes them rarer, not zero.

**What they leave to the owner** (question 1): the slide's line (B18 Q4's sizes as the most that a coast may slide at any
band, or W17.1's p99 ≤ 1 px and max ≤ 2 px); the ponds' position (above); and the tension — requirement 1 favours (B),
which stores nothing new for a planet, while B16 and W16 favour (A) nearer than about 450 km. The bench prices both
before the owner rules.

**Example.** A pilot climbs from the beach at the 15° stand to 60 000 km. Ground more than about 490 km from her eye never
moves: the coast keeps its place, and the high lakes fade as they shrink under a pixel. Nearer, the coast slides a few
pixels while each band sweeps over it; it never jumps, and the sea never flickers. Whether a small pond may fade out and
back in on the way is the owner's word under B16, not a price.

### 10B.8 What it costs (requirement 1)

| Item | Cost | Mark |
|---|---|---|
| [WET MORPH] and [SKIRT SHARES]: two shares and a skirt flag, one 4-byte attribute | 36 → 40 B a vertex (44 → 48 B at the exact rungs): +11 % vertex and upload bytes. The harvest's byte budget grows by the same 11 %, as it grew by an eighth for 32 → 36 B (`crates/client-render/src/terrain.rs:88-90`). B-MEM (about 2.9 GB drawn at the walk) and B-FRAME's non-regression line include it (the review). | ARITH on MEASURED 36 B (`crates/client/src/chunks.rs:195-198`) |
| The share in a parent mesh | 1 B a parent vertex, about 5.6 KB a parent (ARITH: about 5 600 vertices, `extract.rs:28-31`) | ARITH |
| The target share at build | One weighted read at the hit that the shade already finds. | UNMEASURED (M8) |
| [MERGE] and [PARENTS] | Fewer merged cells in shallow water; later finer chunks at the tiles' edge; no extra build. | UNMEASURED (B-S3) |
| [SEAM TARGET] | The neighbour face's parent mesh for seam chunks: a cache lookup, a sub-box build on a miss (E1). | UNMEASURED (B-S1) |
| [EDIT TICK] | 4–18 chunk builds and 1–4 parent builds for each observer and edit tick, by how many rungs the observer draws over the dig; 13–200 ms of one worker, two rounds on 7 workers; up to about 5 MB held until the swap. | ARITH on MEASURED 3.3–11 ms a build (`DEFERRED.md:7760`; B18) |
| [EPOCH] item 3 | Up to two copies of one band, in development only. | ARITH |
| [HYSTERESIS]'s keep margin | Fewer builds. | UNMEASURED (B-S5) |
| [BOOK GRACE] | One realm's artifact held one interpolation buffer longer. | ARITH |
| The gates | Test time only. | — |
| Deleted with the one mesh | The separate sheet, its cuts, its box, its morph, its entity: 3.0 ms of the window's frame at 6 376 km before the sky fix, 0 on foot; today's share UNMEASURED (P7). | MEASURED (`DEFERRED.md:8894-8898`) |

### 10B.9 The laws, checked

- **SL10.** The wet share is a recipe kernel in integers, pinned by G-NO-DRIFT. The target share, the ramp, the swaps and
  the margins are the client's drawing: they move no cell, no collider and no pose.
- **SL6.** Two asks ride their slices: A5 widened (a coarse edit entry carries the top's material shares, with 8e) and A9
  (the edit tick's counts per (tick, rung) and the rung header), both in §6.4's one table.
- **SL1, SL2, SL3.** Nothing here states or derives a pose. A realm hand-over keeps the parent's authored placement, and
  the planet draws itself. The gateway's per-session filter is the owner's own answer (§10C LAW-17).
- **SL9.** The rebuild set is a lookup by key, never a walk over the children.
- **HR5, HR6.** The Tier-A halves at 100 %; every gate drives the shipped client through `vdctl`.
- **No magic numbers.** The keep margin is one column span of its rung; the flicker window is the ramp's length
  (`arrival.rs:27`); the join window lives in the client's tuning; every floor is measured on its own run.
- **S5-2, T9.** An artifact change is a world epoch, and [EPOCH] item 3 names it. Nothing here moves a level or a
  height; it draws.

§10C holds the full check.

### 10B.10 The benches and the gates

The benches B-S1–B-S11 are in §10 (B-S2 is M4; B-S9 is part of M8). The gates G-POP, G-WET-JUMP, G-STILL, G-SLOW, G-BAND,
G-CRACK, G-FAR-EDGE, G-PROBE-KINDS, G-SLIDE, G-POND-FLICKER, G-TRENCH-SEAM, G-CROSSING-TERRAIN, G-EPOCH, G-FAR-FIRST and
G-COAST-100 are in §9's one table, each with its failing control.

---

## 10C. THE LAWS — THE FULL COMPLIANCE TABLE

*The second round, for the owner's requirement 4: "And we SHOULD NOT break ground laws and rules of course". This section
builds nothing and measures nothing new. It reads the code on branch `game` (the working tree of 2026-09-28) for what IS,
and the rulings for what SHOULD be. It checks the MERGED design: the model (§1–§9), the speed levers (§10A) and the seam
mechanisms (§10B). It replaces the first round's short §6.5. The review's corrections of the draft's verdicts are applied
in place (§11, the second round).*

**Three verdicts.**
- **KEEPS**: the design obeys the rule, and the evidence is named.
- **BENDS**: the design obeys the rule only after a change or a gate that this section names, or on an open owner
  question that it names.
- **BREAKS**: the design, as written, does what the rule forbids. Each break gets a cure (a design change) or an owner
  question (an amendment, with its reason and the price of not amending). No break is turned into a "stated price".

**Example.** A miner digs a trench from the belt's beach into a dry hollow. HR1 asks: does only the planet's shard write
those cells? SL2: does her pose stay inside the planet's realm? SL10: does every client draw the same water, byte for byte?
SL8: does a wall of water stand where the flood stops? Each row of §10C.1 asks one such question of the whole design.

### 10C.1 One row for each rule

| Rule | What it demands (quoted) | How the design keeps it (§) | Evidence for the parts that stay | Verdict |
|---|---|---|---|---|
| HR1 sealed shards | "a shard's World/rapier/redb is private; inter-shard bytes exist only as `InterShardFlow` arms (one reviewed file in `vd-wire`)" | The level map, the rim pass, the far bake and the flood run in the planet's own shard, on its own data (§2.2, §2.3, §3.3, §6.1). A6 is withdrawn, so no shard-to-shard row flows (§3.5). | `crates/wire/src/intershard.rs:1-8`; no arm carries terrain data (`:131-633`). | KEEPS. (The draft's "BREAKS for A6" is withdrawn: an arm added under an SL6 ask is HR1's lawful path; A6's real conflicts are R-16 and SL3, LAW-4.) |
| HR2 generic transfer | "ANY entity kind crosses shards via the `TransferableKind` registry" | No new entity kind. A wet cell is a row of its realm's store, never an entity. A flood stops at the realm boundary (§3.5). | `crates/core/src/entity_kind.rs:1-17` (the registry; its trait is owed as D-31, older than this design). | KEEPS. |
| HR3 one tooling | "shard types are `ShardProfile` capability configs. Never `match` on a shard kind in features." | One shard binary. The solve starts where the body has a macro lattice; the flood runs where the profile has `block_edit` (§3.3 item 5). | `crates/bins/src/artifact_boot.rs:62-64`; `crates/sim/src/capability.rs:116-160` (`block_edit` needs voxel geometry, `:141`). | KEEPS after LAW-23. |
| HR4 features once | "every feature ABOVE the seam passes the identical fixture on ≥2 shard kinds (G-IDENTICAL) or it doesn't land" | G-IDENTICAL runs the dig and the flood on the planet's shard and on an Area's shard, the Area fixture given its source row; the flood reads "down" from the realm's gravity function through a seam below `GridMapping` (§3.3, §5.5). | `crates/core/src/grid/mod.rs:200-209` (`outward` is `None` on a flat grid: "gravity is the realm's function, not the grid's"); a hull's down exists only as `ChildFelt` on trial (`crates/wire/src/intershard.rs:606-630`). | BENDS (the draft read BREAKS): the hull's arm joins when its down and its source exist (LAW-9). |
| HR5 100 % coverage | "Tier-A crates at 100% region+branch" | 0 misses on the lines that each step touches; the full gate before the epoch merges, and before S0b and S4 merge (§9). | MEASURED at step 2b: 111 real misses in vd-terrain, 34 in `position.rs` (`docs/investigation/2026-09-24/terrain_believability_status.md:1248-1253`); vd-client UNMEASURED; B4 holds the pass as owed. | BENDS until B4's owed pass lands, before the epoch's merge. No new deferral (the review: HR5 is a gate before every merge, LAW-15). |
| HR6 agent-operable E2E | "client ships the dev-control harness (`vdctl`): input injection at the input-resource seam, wgpu readback" | G-COAST-100 is a readback; S4 builds the player's dig input first, and `vdctl` presses it at the input-resource seam (§9). | MEASURED (grep): no client code sends a `WorldAction`; the gateway counts and drops it (`crates/connection-plane/src/gateway/client.rs:238-240`). | BENDS until S4 builds the dig input (LAW-16). |
| SL1 told, never decides | "A child NEVER derives, adjusts, computes or states its own placement" | No row, tile, far map, chunk row or flood row carries a placement. An Area reads its told berth as an instrument (V5). | `crates/core/src/grid/addr.rs:9` ("The address holds no parent frame"); the fence test `tests/tests/crate_isolation.rs:202`. | KEEPS. |
| SL2 no occupant pose crosses | "no occupant pose may enter another REALM's simulation" | The dig goes client → gateway → the shard that holds her pose; a dig into another realm is refused (V4 row 8; §3.4). | `crates/wire/src/channels.rs:120-124` (a `BlockEdit` "in the session's OWN realm"), `:168` (`NotYourRealm`). | KEEPS. |
| SL3 a realm draws itself | "A parent's per-child message carries a PLACEMENT and nothing else, and must shrink toward that, never grow." | The planet ships its own far map by its own reach (§6.3). A6 is withdrawn. The realm publishes its rows; the gateway filters per session and serves coarse first (E12) — a filter and a pace, never a clamp. | `crates/wire/src/session_flow.rs:600-606` (`BulkAudience::Realm`: "a realm draws itself, whoever looks"); the owner: "The gateway still FILTERS per session what the realm chose to publish; that is not a clamp" (`docs/design/owner_decisions_2026-09-07_voxels.md:318-324`). | KEEPS. (The draft's BENDS on the gateway filter is withdrawn, LAW-17.) |
| SL4 physics apart from re-home | "No orbit/gravity/thrust symbol on the crossing path" | The flood writes rows; it moves nothing and touches no crossing code. Its gravity seam sits below `GridMapping`, off the crossing path. Buoyancy waits for 8o (R-21 NO). | `tests/tests/crate_isolation.rs:133` (`sl4_the_crossing_path_cannot_name_a_motion`). | KEEPS. |
| SL5 one world | "No scale knob, no preset, no reduced or test-only variant, no second generator." | One recipe, one fill, one extractor. K = 4 and K = 8 run only as bench instruments (§4.6). Option (A) replaces; it never runs beside (§4.5). | `tests/tests/crate_isolation.rs:264-325`. | BENDS until S3 deletes the bench's shore rules on the owner's word (LAW-21; question 23). |
| SL6 ask first | "Default NO ... State what data, from which realm to which, why the receiver cannot compute it, and what doing without costs." | One ask table (§6.4): A1–A5 and A7–A9 asked, each with its reason and its price; A6 withdrawn. | — | BENDS until the owner answers the asks (question 6; the edit slice's asks with it). |
| SL7 proxy at the parent's scale | "What crosses: ONE BIT of occupancy, upward ... Nothing else." | No change to liveness or interest. | — | KEEPS. |
| SL8 seamless | "a seam is a defect ... Tolerances are always physical ... Never ship a capability without its continuity" | One mesh (no coast hole, no wall), one field, the far fold, the wet morph, every mechanism of §10B.4; no water face after a dig (§3.3); the far map coarsest first. | MEASURED today: the 208 m wall (B20), 1 442.5 m at 9 → 10 (W17.1), 53 of 714 far lake bodies flicker (B15). | BENDS: ponds that come back need G-POND-FLICKER at 0 or the owner's word (LAW-2); the near slide needs its line (LAW-6); the per-cell far colour is the owner's trade (LAW-7); a far body's reach pop-in stays a stated residue owed to the reach ruling (LAW-8); a first draw waits for the bake (LAW-19). The draft's three BREAKS are cured: the water face is refused (LAW-3), the far map ships coarsest first (LAW-8), the ponds are a gate or the owner's amendment, never a price (LAW-2). |
| SL9 unbounded children | "never a per-tick walk of all of them ... Finding which child holds a point is a LOOKUP" | Every read is a lookup by address (a node, a tile, a far cell); the Area check is one query of the child index (§3.4 step 3). | `crates/core/src/child_index.rs:1-2` ("answered without touching them all"). | KEEPS after LAW-18. |
| SL10 seed-shaped world | "ONE Rust generator crate ... no-drift as a byte-for-byte MEASURED gate on every target ... The server computes collision on that same shape." | Every new kernel is a recipe kernel in fenced integers; every lever keeps its plain form as the oracle (§10B.5 item 1). The collider reads the same composed rung-0 cells (§6.2). S1–S3 are one world epoch with a version guard (§6.3). | `tests/tests/crate_isolation.rs:283-325`; `crates/terrain/src/tag.rs:237-242`; `crates/bins/src/artifact_store.rs:197-203` (the store refuses another artifact version); `crates/client/src/artifact_book.rs:447-465` (the client checks none). | BENDS: the version guard (LAW-5); the x86-64 leg is emulated and the solve's other-host leg G6 never ran (`docs/design/DEFERRED.md:7721-7736`; LAW-20). |
| No I/O outside the seam | "sim/node code gets time/rng/persistence/transport ONLY through `sim::io` traits" | The flood and the compositions run on the injected composition worker (§3.3 item 4, §3.6). | `crates/bins/src/artifact_worker.rs:1-9`; `crates/sim/clippy.toml:6-31`. | KEEPS after LAW-22. |
| No magic numbers | "operational params in ONE config struct (`TransportTuning`/`TransferTuning`), never inline literals" | The flood's bounds and its parking field, the box cache, the tile cache, the join window, the harvest's time budget and the rebuild rate go into tuning structs; every budget is derived (§10A.2). | `crates/sim/src/stub/built_store.rs:178-199` ("none has a default here"); against it `crates/connection-plane/src/gateway/artifact.rs:50` (inline) and `crates/connection-plane/src/gateway/config.rs:22` ("ONE reviewed struct, no inline literals"). | BENDS until the pace moves into `TransportTuning` and `μ` gets its test (LAW-11). |
| Fence discipline | "every authoritative action carries its Fence; receivers reject stale" | The trench's rows ride `BulkFor` with the realm's fence (§3.4 step 7); gate G-FENCE. | `crates/wire/src/session_flow.rs:552-556`; the owner-fence row, whose "writer and its refusal land with slice 9" (`crates/sim/src/stub/built_store.rs:125-130`). | BENDS until S4 builds the owner-fence writer (LAW-14). |
| Postcard wire rules | "new data rides a new trailing variant, never a new field" | `ArtifactFar`, the brush, the refusals and the rung-headed `BulkFor` are appended arms; `ArtifactPyramid` and `ArtifactCoast` are tombstoned; `coast_parts` stays a dead field; `FLUID_LEVELS` is tag 8 (§6.3). `water_m`'s new meaning lands inside one epoch behind a moved floor. | `crates/wire/src/version.rs:346-348`, `:357`, `:385`, `:369-370` ("Raise it only alongside a change of the same kind"); `crates/core/src/chunk_row.rs:5-9`. | KEEPS after LAW-5. |
| Decode-to-Default banned | "decode-to-Default is BANNED for Durable kinds" | The stored artifact refuses another version; `FLUID_LEVELS` is a REQUIRED tag (§6.3). | `crates/core/src/tlv.rs:19-22` (`required_max_tag` refuses, "never decoded to Default"); `crates/bins/src/artifact_store.rs:197-203`. | KEEPS after LAW-13. |
| Determinism | "closed-form `f(seed, universe_tick)` ... every physics→control boundary quantized to integer grids" | The fill, `t`, the flood and the far fold are integer; the rim pass writes whole metres; a split bake uses a fixed partition and an ordered merge (§2.3). | MEASURED (grep): no thread in `crates/terrain/src`; the artifact digest is pinned on one host (`home_artifact_pin`). | BENDS until G-SPLIT holds (LAW-20). |
| No client-side prediction | "NO client-side prediction (interpolation on a 100–150 ms buffer)" | The client never floods; a dug cell changes only when its row lands (§3.4, §6.1). E11 widens an ASK by a speed read from two delivered poses; it moves no pose. | `crates/wire/src/channels.rs:159-160` ("Success needs no message: the block arrives on the diff lane"); F9 item 1 reads the speed so today (`docs/design/owner_decisions_2026-09-12_frozen_patches.md:306-310`). | KEEPS (LAW-32). |
| The server does all math (SL10 clause 7) | the client "never derives a pose, a velocity, or any state ... The exception is exactly the static shape of the seed" (`docs/design/owner_decisions_2026-09-07_voxels.md:51-54`) | The client derives only the static shape (`λ`, `t`, the wet share, the fold of rungs 11–13 from rows) and folds the diffs it holds (V4 row 7). The ask's speed is a reading for a request, as F9 item 1 already reads it ("two DELIVERED poses, SL10 clause 7"), never a state. | `docs/design/owner_decisions_2026-09-07_voxels.md:132` (V4 row 7). | KEEPS (question 13 asks the owner's word on E11 all the same). |
| Seed and secrecy | "A block's SUBSTANCE may not be a pure function of (position, seed)" where it is valuable | Water, air and vacuum are bulk stock; "a gas that pays is live state" (§2.5); later fill substances pass the wiki test. | `crates/core/src/registry/kind.rs:118` (no hand-placed liquid, gas, void or ore). | KEEPS (LAW-24). |
| F6 and B20, the core share | "Half: 7 of 14" (B20 item 2); "The pool never takes one thread per core" (F6 item 2) | The client's workers stay at half the cores; E8's descent thread counts inside the share (§10A.3). Every budget is stated for the instrument; the target machine is owed (question 9). | `crates/client-render/src/terrain.rs:4257-4260` (`worker_share`). | KEEPS (LAW-30); F6's machine OPEN (question 9). |
| F7 and F9, the card | F7: "we utilise gpu to the maximum"; F9, later: the card as a budgeted second builder, and "SO THE CARD STAYS A KNOB ... OWED BY THE OWNER" | The card stays the knob that F9 left: it builds no solved chunk until each box request carries its node window and the owner rules its worth (question 12). | `crates/client/src/chunks.rs:343-348`; `crates/client-render/src/terrain.rs:1343-1346` ("ruling F9, the card builder parked"); `docs/design/owner_decisions_2026-09-12_frozen_patches.md:489-503`. | OPEN under F9 (question 12), not a break (the review; LAW-10). |
| V10, the 8 ms budget | "I think 8 should be ok" (`docs/design/owner_decisions_2026-09-07_voxels.md:441-443`) | G-BUDGET: a coast chunk's cell path under 8 ms; G-BUILD reads the whole build in the loaded regime; question 11 asks whether 8 ms becomes the whole build's line. | `docs/design/DEFERRED.md:7760` (the owner set 8 ms on the box and the extraction). | KEEPS as ruled. |
| B20 item 1, the coast first | "THE COAST FIRST ... first a gate that stands about 100 m over the sea and looks at the coast, and that FAILS on today's picture" (`docs/design/owner_decisions_2026-09-22_believability.md:565-567`) | S0 builds G-COAST-100 red on today's code first; the epoch cures it; S0b comes after the epoch in the recommended order (§9). | — | KEEPS in the recommended order; BENDS only if question 10 puts the builder first (LAW-29). |
| B21 items 1–2 | "STOP ADDING TERRAIN RULES"; "every distance is a copy made from the same volume by one rule" | The rim pass is a new rule, named (§2.3); two ladder rules (§4.1). | — | BENDS: question 1 (LAW-28). |
| T9 and W11 | "we don't randomly generate any physic numbers or physic laws — always calculate"; a hollow's water is "decided by the water supply, never by the shape of the hole" | A published water-table law first; the flat level only on the owner's amendment (§2.4). | — | BENDS until question 3 (LAW-1). |
| B16 | "No flicker or changing of the shapes between rungs at all" | Met above `L_ref` by the far fold; below it, G-POND-FLICKER at 0 and G-SLIDE's line, or the owner's word (§4.5). | — | BENDS until question 1 (LAW-2, LAW-6). |

### 10C.2 The design, line by line

| Part of the design | Rules it touches | Verdict | Finding |
|---|---|---|---|
| The fill pass (§2.4) | T9, W11 item 2, B3 G-LAKE-SHARE, B16, SL8, SL10 | KEEPS SL10 (one integer kernel on one field). BENDS T9 and W11: the flat water table is no law, so a published law comes first (question 3). BENDS B16: ponds that come back need the gate or the owner's word. | LAW-1, LAW-2 |
| The level map (§2.2) | B7, SL6 A1, S5-2, "never repurpose a message" | KEEPS B7's law as data. KEEPS the wire inside one epoch with a moved floor. | LAW-5 |
| The rim pass (§2.3) | T9, B7, B21 item 1, SL8, determinism | KEEPS T9: a level moves to its drawn outlet, and the laws run again inside the existing fixed point. BENDS: a new rule between two sources of truth; its time delays a first draw; a split must be deterministic. | LAW-28, LAW-19, LAW-20 |
| The far fold (§4.3) | B16, B17, B18 Q1/Q2/Q5, S5-3, R-7, SL8 | KEEPS B16 above `L_ref`. BENDS: B17's per-pixel windows retire only on the owner's word; G-FAR-EXACT is split per cell and per pixel. | LAW-7 |
| The edits and the flood, W2 (§3) | W2, W3, SL8, HR3, HR4, fence, decode-to-Default, SL6, no I/O | KEEPS SL8 (no water face: refusals). BENDS: the hull's down; the owner fence; the brush verb; the required tag; the worker; the gate. | LAW-3, LAW-9, LAW-12, LAW-13, LAW-14, LAW-22, LAW-23 |
| The collision (§6.2) | SL10 clause 5, W10 item 4, "collision never coarsens" | KEEPS: the collider reads the same composed rung-0 cells, and a liquid is not solid. Owed to slice 11: `vd-sim`, `vd-node` and `vd-physics` have no terrain edge today (MEASURED, their `Cargo.toml`). | — |
| The wire, every new byte (§6.3, §7) | postcard rules, SL6, the floor rule, the `BulkFor` contract, V8 addendum item 3 | KEEPS after the floor moves; the gateway never decodes rows and forwards one shared buffer; `FLUID_LEVELS` is required; the far map ships coarsest first. Every new byte is in §6.4. | LAW-5, LAW-8, LAW-13, LAW-17 |
| The card path (§6.1) | F7, F9, B20, SL10 | OPEN under F9. | LAW-10 |
| A hull that holds terrain (§5.5) | V4 15.2, V4 row 13, V11, HR4 | KEEPS V4: every solid cell is a row; vacuum is `Empty`. BENDS HR4 until the hull's down and source exist. | LAW-9 |
| An Area realm (§3.5) | V5, R-16, E4 of the format sitting, SL3, SL6 | KEEPS R-16 and SL3: A6 is withdrawn. | LAW-4 |
| Two ladder rules (§4.1) | B21 item 2 | BENDS B21: two rules, not one; the owner rules through question 1. | §10C.5 |
| The first slice's order (§9, question 7) | T1, W2 ("It belongs to SLICE 9"), V4 row 14, B20 item 1 | Amends T1 and W2's slice. KEEPS V4 row 14: the detailed discussion comes first. KEEPS B20 item 1 in the recommended order. | §10C.5, LAW-29 |
| The speed levers (§10A) | SL10, F6, B20, F9, SL3, SL6, V1 clause 7, T7, HR5, "no magic numbers" | KEEPS: every lever changes no world byte, the descent's thread counts inside the share, the serve order is the gateway's lawful filter and pace, the speed margin reads delivered poses, no lever moves a drawn band. E11 widens the ASK, never a drawn band or a rung's switch distance or survival table (T7); the collider keeps rung 0, and E4 changes no byte of it (collision never coarsens). OPEN: E9 under F9. | LAW-10, LAW-29 – LAW-32 |
| The seam mechanisms (§10B) | SL8, SL6, SL10, S5-3 | KEEPS SL10 (the drawing moves no cell); BENDS SL6 until asks A5 (widened) and A9 are answered; amends S5-3's entry (the material shares). | LAW-33 |

### 10C.3 The findings

Each finding names the rule, the evidence, and the cure or the owner question. An owner question asks for an amendment of
a ruling; its price is the cost of NOT amending.

**LAW-1 BENDS T9 and W11: the flat water table is not a law.** The fill (§2.4) puts water into every natural hollow and cave
under the column's level. The shape of the hollow decides; no water supply decides. The design calls it "a STATED
SIMPLIFICATION, not a published law". T9: "A band that stands in for a missing law is a defect: the law is owed." W11 item
2: every hollow ends dry, partial or spilling, "decided by the water supply, never by the shape of the hole". B3's
G-LAKE-SHARE counts lakes "only where the budget fills a hollow", and it is RED already at 4.828 % (B9). Cure (a): a
published water-table law with a calibration body — the candidate is the equilibrium water-table depth from recharge and
terrain (Fan, Li & Miguez-Macho 2013, Science 339, 940–943; to be read before use), fed by the row's own runoff (W11's
Turc–Pike partition, `docs/design/owner_decisions_2026-09-21_water.md:251`); a hollow then holds groundwater only where the
table stands over its floor (bench M2b). Cure (b): fill only the open cells that connect to the node-scale water (option
(A)'s shore map, §4.5). Question 3. *Example.* On the coast's low plain a hollow 2 km inland dips 3 m under the sea's
level. Under the flat level it holds a pond at the sea's level. Under cure (a) it holds a pond only where the plain's own
water table reaches over its floor.

**LAW-2 BENDS B16 and W16 item 3: ponds that fade out and back in.** §4.5's own table: water under about two cells "can
fade out and back in across successive bands (a pond made by one octave)". B16: "No flicker or changing of the shapes
between rungs at all". W16 item 3: "no water appearing or vanishing". B18 Q1: water thinner than a pixel is "a pale tint
that follows the pixel's size". The owner's requirement 3 says it again. Today the shore law hides this class, so the
design brings it in. ONE POSITION (the review): no price is offered; gate G-POND-FLICKER (§9) must read 0 on the stands
and a planet sample, or the owner amends B16 for such ponds, or option (A) is built; a smaller K (question 2; the pond at
its drop gets about 48.83 / K times smaller in cells, ARITH §4.6) makes them rarer, not zero. Question 1. *Example.* A
pilot climbs from the belt's lowland. At rung 5 a pond stands in one octave's hollow. At rung 6 that octave drops, and the
pond fades. At rung 7 another octave's trough takes the column under its level again, and a pond fades in.

**LAW-3 (the draft's BREAK, now KEEPS) SL8: a wall of water at the flood's bound and between two waters.** The first round's
§3.3 let the water end "in a vertical face" at the bound and between two levels. SL8: a seam is a defect; the 208 m wall
on the coast (B20) is the defect that started B21. A cell-by-cell flood cannot complete in every case: a canal into the
largest pocket (1.35 million km², MEASURED B7) opens about 10¹⁵ open cells (ARITH: 1.35 × 10¹² m² × a mean depth of 1–3
km, the depth ASSUMED). Cure, now in §3.3: a flood over its per-tick number continues over the next ticks; a flood past
its total bound is refused with a typed reason (A8) until the digging slice designs one live level for a whole
catchment; a dig that joins two waters at different levels is refused (A8) until W3's flowing water exists. Gate
G-NO-FACE. Question 4.

**LAW-4 (the draft's BREAK, now KEEPS) R-16 and SL3: ask A6.** The owner refused R-16: "the planet's diff under an area —
NO; an area's floor is built" (`docs/design/owner_decisions_2026-09-07_voxels.md:227`). The approved format sitting
refuses "Any realm-to-realm chunk fetch" without an ask (`docs/investigation/2026-09-07/topic_00_format_sitting.md:306-309`).
SL3: a parent's per-child message "carries a PLACEMENT and nothing else". Cure, now in §3.5: A6 is WITHDRAWN. An Area's
cells are its own rows (the built floor), its liquids come from rows only, as a hull's do (§5.5), and the flood stops at
its boundary. A way with no crossing exists for later: the Area's shard derives the planet's static shape itself (SL10
clause 5), but only after G6 passes (LAW-20). Question 19. (The draft also cited HR1: withdrawn, because an arm added
under an SL6 ask is HR1's lawful path.)

**LAW-5 (the draft's BREAK, now KEEPS) S5-2 and "never repurpose a message": a version guard for the whole change.** S1
gives the row's `water_m` a new meaning and deletes the coast mask; the shore law that reads both goes only at S2; and
the far view's new field comes only at S3. Without the pyramid, `level_for` sends every rung to the rows and `tile_rung`
climbs to the top rung (`crates/terrain/src/artifact.rs:857-859`, `:939-945`), so the tiles would grow to the whole visible
body (up to 2 166 tiles, 88.7 MB, about 42 s at 2 MiB/s, ARITH) and rungs 10–18 would read point samples, B15's far lake
flicker (the review extended the draft's S1 + S2 to S3). An old client passes the world hello (the declared tag folds only
the generator version and the seed, `crates/terrain/src/tag.rs:237-242`) and accepts a version-10 head (`accept_head`
checks no version, `crates/client/src/artifact_book.rs:447-465`). Cure, now in §6.3: (1) S1, S2 and S3 land as ONE world
epoch, GENERATOR_VERSION 22 with ARTIFACT_VERSION 10; (2) the client refuses a head of another artifact version by name,
as the store does (`crates/bins/src/artifact_store.rs:197-203`); (3) `PROTO_MINOR_FLOOR` moves to the minor that adds
`ArtifactFar`, the owner's own posture ("raise the floor, we don't have old clients", `crates/wire/src/version.rs:375-376`).
Gate G-VERSION.

**LAW-6 BENDS SL8: G-SLIDE's line.** The first round's G-SLIDE passed on "question 1's numbers", so it could not fail.
Retiring W6 opens again what W6 cured: medians of 234 m and 848 m at 6 → 7 and 7 → 8 before W6 (DOC,
`crates/recipe/src/height.rs:359-366`), 0 m after (MEASURED, W6). The draft proposed W17.1's line (p99 ≤ 1 px, max ≤ 2 px
per band) and named B18 Q4's 1.46 / 3.29 / 4.79 px as the owner's accepted ceiling in the same breath; a max of 3.29 or
4.79 px fails a 2 px line (the review). W17.1's line was written for the ground's rung disagreement
(`docs/design/owner_decisions_2026-09-21_water.md:1342-1344`), not for a coast. Cure: ONE line, put to the owner in
question 1 — (i) B18 Q4's accepted sizes as the most that a coast may slide at any band (the recommendation, judged in
flight), or (ii) W17.1's line. Over the chosen line: question 2's K, then option (A).

**LAW-7 BENDS: G-FAR-EXACT against §4.3, and B17's windows.** §4.3 states that the share is exact per cell, not per pixel
("a monotone blur inside two pixels"); the first round's G-FAR-EXACT demanded 0 "pixels whose share changes at a swap
beyond the fold's rounding", so it failed by the design's own words. The trade also retires B17's per-pixel windows
(MEASURED 0 steps; 17–79 MB per stand). Cure, now in §9: the gate is split — (a) per cell, 0 misses beyond 1 m and 1/255;
(b) per pixel, no share step at a band's far edge and a monotone change inside the band. B17's retirement is the owner's
trade (question 20).

**LAW-8 (the draft's BREAK, now KEEPS) SL8: a far planet pops in.** The first round: "Until its head lands, nothing of it is
drawn; then rungs 14–18 draw at once". DEFERRED keeps this seam as owed (`docs/design/DEFERRED.md:8937-8947`). Cure, now in
§4.3 and §5.2: the far map ships coarsest rung first, in its own parts (rung 18 is 8 664 cells × 3 B ≈ 26 KB, ARITH); the
client draws each far rung when its part lands and fades the next rung in across its band. Gate G-FAR-FIRST, with ONE
rule in §10B and §10C (the review): its clock starts when the first far part is HELD by the client. The body's entry into
the window at its reach (about 23 px), before its artifact is served, stays a stated residue of the REACH ([EPOCH] item
4), owed to the reach ruling's brightness radius; the draft's "frames with the body in view and no drawn rung: 0" could
not pass and is withdrawn.

**LAW-9 BENDS HR4: the flood's "down" on a flat grid.** `GridMapping::outward` is `None` on a flat grid
(`crates/core/src/grid/mod.rs:200-209`), and no realm gravity function exists in the code (MEASURED, grep). §5.5 agrees:
"with no gravity there is no free surface". So G-IDENTICAL on a hull could not give identical results. Cure, now in §3.3
and §5.5: the flood reads a cell's height along "down" from the realm's gravity FUNCTION (V4 15.2, V11) through a seam
below `GridMapping` (ruling V6 A2: "any physics anchor sits BELOW the seam"); the planet's function is the radial.
G-IDENTICAL's second shard kind is an Area's shard, which uses the planet's grid and gravity (V5); its fixture GIVES its
source row, because an Area gets no planet rows (LAW-4). A hull's down exists today only as the planted `ChildFelt` (SL6
R-20, YES ON TRIAL, behind `felt_down`, default OFF; `crates/wire/src/intershard.rs:30-38`, `:606-630`;
`crates/sim/src/capability.rs:95-100`) — the draft missed it — or later as a graviblock. The hull's flat arm joins the gate
when its down and its source exist; until then the design does not claim HR4 for a hull.

**LAW-10 OPEN under F9 (the draft read BREAKS F7): the card builds no solved chunk.** F7 item 2: "THE GPU DOES THE
CLIENT'S CHUNK WORK TO THE MAXIMUM". But F9, later in the same file, made the card a budgeted second builder at a quarter
of each frame, with the CPU share as the default and the fallback; flown on cured code it raised the widest pop on four
legs of five (46 against 40 at 528 m/s) and took the slow hull's band from holding to 3 357 urgent chunks, so "THE CARD
STAYS A KNOB", and the owner's decision on its worth is still owed
(`docs/design/owner_decisions_2026-09-12_frozen_patches.md:275-298`, `:400-408`, `:489-503`). The code names that state:
"ruling F9, the card builder parked" (`crates/client-render/src/terrain.rs:1343-1346`); "A job with an artifact is the CPU
builders' alone: the card holds no Z" (`crates/client/src/chunks.rs:343-348`). The design keeps the recorded state and
makes the question askable on the new code: once each box request carries its node window (E2), the card can build
solved chunks from the same recipe (`crates/recipe-gpu/src/lib.rs:1-2`, "one crate, two compilations"), and bench P12
measures it on pairs of flights, with the MEASURED 95 boxes a second and the two measured harms beside it. Question 12.

**LAW-11 BENDS "no magic numbers": the artifact pace and `μ`.** `ARTIFACT_BYTES_PER_BEAT` is an inline constant
(`crates/connection-plane/src/gateway/artifact.rs:50`) against the gateway's own rule, "ONE reviewed struct, no inline
literals" (`crates/connection-plane/src/gateway/config.rs:22`); §4.2's "about 7 s" rests on it. Cure: move it into
`TransportTuning`; bench M7 states its value. The centring word `μ` (§2.1) is measured once and typed. Cure: a test
derives it again from the recipe's own ridged fold and fails on any difference (gate G-MU; B14's precedent for
`C_smooth` and `C_ridged`).

**LAW-12 BENDS SL6: the brush has no wire verb.** `BlockEdit` places one kind at ONE cell (`crates/wire/src/channels.rs:135-148`),
and a mined cell becomes `Empty` with gap +1 (`crates/terrain/src/compose.rs:6`). §3.4 digs "per cell or as a brush", with
"a gap from the brush". No brush exists on the wire. Cure: ask A7 (§6.4), or per-cell digs: about 1 000 reliable actions
and a stepped cut for the trench (ARITH). Question 22.

**LAW-13 (now KEEPS) "decode-to-Default is banned": `FLUID_LEVELS` is required.** A chunk row's reader skips an unknown tag
(`crates/core/src/chunk_row.rs:5-9`). A reader that skipped `FLUID_LEVELS` would draw a flooded cell at its column's level,
or as gas: a durable fact read as a default. Cure, now in §6.3: a REQUIRED tag, so that such a reader refuses the row
(`crates/core/src/tlv.rs:19-22`). Gate G-REQUIRED.

**LAW-14 BENDS fence discipline: the minimal store needs the owner fence.** The owner-fence row's "writer and its refusal
land with slice 9" (`crates/sim/src/stub/built_store.rs:125-130`), and S4's minimal store writes the trench's rows. Cure:
S4 includes that writer and its refusal. The rows ride `BulkFor` with the realm's fence
(`crates/wire/src/session_flow.rs:552-556`), and the gateway and the client refuse a stale fence (gate G-FENCE).

**LAW-15 BENDS HR5: the coverage is red, and nothing is deferred.** The first round said the new kernels "stay" at 100 %.
vd-terrain is red (111 real misses at step 2b, MEASURED), and vd-client is UNMEASURED. The draft moved the full gate to
after S3 with no owner question; the review: HR5 is a gate before every merge (`just gate` is the pre-merge gate). Cure:
every step ends with 0 misses on the lines that it touches; the full `just coverage-fast` runs before the epoch merges and
before S0b and S4 merge; B4's owed pass ("still owed for everything since the last pass",
`docs/design/owner_decisions_2026-09-22_believability.md:55-58`) runs first. The fill's field reader follows HR5's
generic rule: a branchless generic shim, every branch in monomorphic helpers.

**LAW-16 BENDS HR6 and "test exactly production".** No player dig tool exists. A `vdctl` verb that sends a raw
`WorldAction` past the client would be a stand-in. Cure: S4 first builds the player's dig input (a key and a target, after
its own discussion, V4 row 14); then `vdctl` presses it at the input-resource seam.

**LAW-17 KEEPS SL3 and the `BulkFor` contract: the gateway's filter is the owner's own answer** (the draft read BENDS and
moved the choice to the realm; the review reversed it). The owner: "The gateway still FILTERS per session what the realm
chose to publish; that is not a clamp"; only a floor pushed DOWN to the realm (R-4) was withdrawn
(`docs/design/owner_decisions_2026-09-07_voxels.md:232`, `:318-324`). The realm could not do the draft's cure anyway: one
window serves every session behind a gateway (`docs/investigation/2026-09-07/06_storage_diff_lane.md:500-503`), so the
realm cannot name each session's rung without new upward data (SL6). The requirement that stays: the filter reads a PLAIN
RUNG HEADER, never decodes a row (`crates/wire/src/session_flow.rs:545-549`), and forwards one shared buffer per bulk
message (V8 addendum item 3). That header is part of ask A9.

**LAW-18 KEEPS SL9 after the cure: the Area check is a lookup.** §3.4 step 3: the shard "checks that no Area owns the
cells". Cure, now in §3.4: one query of the planet's child index (`crates/core/src/child_index.rs`), never a walk over all
Areas.

**LAW-19 BENDS SL8 (requirement 1): the solve's added time delays a first draw.** The rim pass adds 2–84 core-minutes (ARITH
on ASSUMED inputs, §2.3) to the 205 s solve (MEASURED). A body draws nothing until its head lands (§5.2), so a pilot who
flies to an unvisited planet waits longer for it. Cure: M3 prints the rim pass's WALL time on the production server's
core count, and the owner rules a ceiling for it (question 21; budget B-BAKE). The draft measured it on one thread against
205 s while LAW-20 splits it: the 2–84 core-minutes fail a one-thread 205 s ceiling by 0.6–25 times, so the ceiling is
stated for the production core count (the review).

**LAW-20 BENDS determinism and SL10: a split bake, and the solve on another host.** The solve runs on one thread today
(MEASURED, grep), and its digest is pinned on one host. The design splits the rim pass and the far bake over the server's
cores. The solve's other-host leg G6 never ran (`docs/design/DEFERRED.md:7721-7736`). Cure: a fixed partition and an
ordered merge; the artifact pin at 1 and at N workers reads one digest (gate G-SPLIT; bench M11); G6 runs before any
second shard derives an artifact (LAW-4's way with no crossing).

**LAW-21 BENDS SL5 and the FINAL-backend law: the bench's shore rules outlive the shore law.** After S2,
`crates/bins/src/{shore_rules,shore_shape,shore_geometry}.rs` and design D's bench code (§8.3) measure nothing that ships:
a second implementation of one concept. The owner's requirement 2: "We can comment out all current special shores and
sea models." Cure: delete them in S3 on the owner's word (question 23); git keeps them.

**LAW-22 KEEPS "no I/O outside the seam" after the cure: the flood's worker.** The first round ran the flood "in the same
commit" and the compositions off the tick. One composed chunk costs about 5 ms (ARITH §3.6) against a 20 ms tick
(`crates/bins/src/lib.rs:446`, 50 Hz), and SL8's seam list forbids unbounded work in a tick. Cure, now in §3.3: the flood
runs on the composition worker, an injected seam (inline in a Tier-A test, a thread in the binary, as
`crates/bins/src/artifact_worker.rs:1-9`); the dig's rows and the flood's rows commit together when the worker answers;
the dig parks meanwhile, bounded by a tuning field OF ITS OWN (the draft reused `BlockStoreTuning::parked_ticks_max`,
whose stated meaning is the stalled disk, `crates/sim/src/stub/built_store.rs:196-198`); the visited set is a `BTreeSet`
or a `DetHashSet` (`crates/sim/clippy.toml:24-25`).

**LAW-23 KEEPS HR3 after the cure: the flood's gate.** The first round said "the owning shard". Cure, now in §3.3: the
flood runs where the shard's profile has `block_edit` (`crates/sim/src/capability.rs:141`), never where the realm is a
planet.

**LAW-24 KEEPS seed and secrecy after the cure: later fill substances.** §2.5 stated the rule for a gas only. Cure, now in
§2.5: every later fill substance (lava, a salt, a liquid metal) first passes S5 item 4's test of the seed-and-secrecy
ruling: "if this were printed on a public wiki tomorrow, would the game still work?"
(`docs/design/owner_decisions_2026-08-27_seed_and_secrecy.md:67-68`).

**LAW-25 BENDS the text of S6-6: `compose()` changes.** The cell grows to three bytes, and an edited cell takes `t` from its
row (§3.1). So `compose()` carries `t`, or a fill step after it derives `t`. The ORDER stays; the first round's claim
"`compose()` stays unchanged" is withdrawn in §3.1.

**LAW-26 BENDS B6 item 2: each step's report.** B6: every step ends with the three stands, a fourth stand framed like the
reference, "the skyline numbers on that stand", and "the reference checklist ... brought up to date"; "A step that turns a
gate green but moves that picture AWAY from the reference says so." The first round's §9 named pictures only. The
centring (question 5) moves the land, so the skyline moves. Cure, now in §9: both at the end of every step.

**LAW-27 BENDS the ruling ledger: two omissions of §8.4.** B14 item 2 (the reach law of N2's fade) retires with N2; its
registered spectrum conflict stays in DEFERRED for R2. B9 item 1 (the lake ceiling, judged after the shoreline step) must
be read again after the epoch, because the rim pass and the ponds move the lake share. Cure, now in §8.4.

**LAW-28 BENDS B21 item 1: the rim pass is a new rule between two sources of truth.** B21: "STOP ADDING TERRAIN RULES". The
rim pass reads the 8 km level and the drawn ground and moves one to fit the other. It obeys T9 (it applies Fill–Spill–
Merge's law on the drawn ground and runs the budget again), but it is the kind of rule that the owner asked to stop. The
one representation that makes it unnecessary is a finer solve (R2's 1 km level shrinks the fine relief from 1 533 m to
about 315 m, ARITH §2.4) or option (A). Cure, now in §2.3 and question 1: stated to the owner, not hidden inside the fill.

**LAW-29 (new in the review) BENDS B20 item 1 only if the builder goes first.** The performance draft put S0b (the builder)
right after the bench and before the coast. B20 item 1: "THE COAST FIRST". Cure, now in §9: S0 builds the failing coast
gate first; the recommended order puts S0b after the epoch (S3 rewrites the same builder code); question 10 asks the
owner.

**LAW-30 (new in the review) KEEPS F6 and B20: the descent's thread.** E8 moves the descent off the main thread onto one of
the terrain's own threads, inside the seven-worker share (B20 item 2), so the terrain still takes half the cores and
never more (F6 item 2). The target machine is F6's owed number (question 9).

**LAW-31 (new in the review) KEEPS SL3: the serve order.** E12 serves each session the coarsest far levels and the near
tiles first, and a far body only to the level that its drawn rungs read. The realm still publishes its whole artifact to
the gateway's cache (`BulkAudience::Realm`, `crates/wire/src/session_flow.rs:600-606`); the gateway filters and paces per
session, which is the owner's own answer, never a clamp on what the realm draws.

**LAW-32 (new in the review) KEEPS V1 clause 7 and "no client prediction": the speed margin.** E11 widens the ASK by the
eye's speed times the measured delay. The speed is read as F9 item 1 already reads it: "the PEAK of the lead's metres over
the interpolation buffer's seconds — two DELIVERED poses, SL10 clause 7"
(`docs/design/owner_decisions_2026-09-12_frozen_patches.md:306-310`). It moves no pose and predicts no position; the owner's
word is asked all the same (question 13).

**LAW-33 (new in the review) BENDS SL6 and S5-3: the seam mechanisms' data.** [EDIT TICK] needs each (tick, rung)'s part
count, the tick's rung mask and a plain rung header on `BulkFor` (ask A9); X6's cure needs the top's material shares in a
coarse edit entry (ask A5, widened), which amends S5-3's "the dominant substance" for the picture. Cure: both asks in
§6.4's one table; the draft's "A7" named two different asks, and the one table numbers them apart (A7 the brush, A9 the
edit tick).

### 10C.4 The SL6 asks, checked again

§6.4 holds the one table (A1–A5 and A7–A9 asked, A6 withdrawn, and the "no ask" row), each with its data, its direction,
its bytes, why the receiver cannot compute it, and the price of doing without. One CHECK rides gate G-VERSION's fixture:
`ArtifactHeld` keeps its meaning only while the artifact's digest folds the far map.

### 10C.5 The rulings the design amends or retires

| Ruling | The owner's words | What the design does | Proposed change |
|---|---|---|---|
| W6, the quarter hold | "Again, after your changes when I'm flying, the shores are changing all the time." | Retires the shore law. | Retire. Its purpose (no coast slide) moves to G-SLIDE's line (LAW-6; question 1). Price of keeping W6: the 208 m wall (MEASURED, B20) stays. |
| W10, the coast mask | "Perfect" (after "during flight the shores changes again all the time") | Retires items 1–2 (the side bit; no side from a mean). Keeps item 3 (one field) and item 4 (collision reads rung 0). | Retire items 1–2; items 3–4 stand. |
| W15, the footprint | "shows squares of water on the land"; "flips between water and land"; "We need to fix it" | Retires the counts. | Retire. The far fold's share folds as the counts did: a parent is the mean of its four children. |
| W16, one shape | "Please solve it once and for all distances!" | Keeps items 1–3 as the bar, and item 4. Retires item 2's three-way side and the sheet on the ground's grid. | Retire the mechanisms. The bar stands; below `L_ref` the design meets it only through G-POND-FLICKER and G-SLIDE, or the owner's word (LAW-2, LAW-6). |
| W17, the field step | "for some of the far-view rungs the change is still visible" | Retires the field step; its cause, the second field, goes. | Retire; `switch_floor_m` stays, inert. |
| W18, the sheet on the lattice | "Not fixed"; "It changes still when I'm leaving farer" | Retires the sheet, its morph and its cuts. | Retire; the wet morph (§4.4) takes over the water's morph. |
| W2, live water | "Ok, agree with that one" (the ruling: "inside the edit's own box"; "It belongs to SLICE 9") | Widens the box by a margin, wets natural cells, REFUSES a flood past its bound and two waters (no water face), builds the flood before 8e. | Amend by question 4 (the refusals) and question 7 (before 8e). |
| W4 item 3 | "Agree with all: 1, 2, 3, 4, 5, 7" | Keeps "No second water surface"; retires the sheet, the coast band and the beach. | Amend. |
| W11 and T9 | T9: "we don't randomly generate any physic numbers or physic laws — always calculate" | The flat water table is a stated simplification; a published law comes first. | Question 3: a published law first; an amendment only on M2's and M2b's numbers. |
| B3 G-LAKE-SHARE, B9 item 1 | "Agreed, add the ruling and the gates" (B1) | The ponds put water where no budget filled a hollow; the rim pass moves the lake share. | Read the gate again after the epoch (LAW-27). |
| B7 item 1 | "I agree with your 2 recommendations, if it will lead us to believable terrain" | Retires `SIDE_INLAND`; keeps the law (the sea over the connected ocean). | Retire the mechanism. |
| B13, N2 | Every recommended option chosen; B21: "N2 IS NOT BUILT (B13's build is parked, not retired ...)" | Retires N2. | The owner's word turns "parked" into "retired". |
| B14 items 1–4 | The ruling's words: "SEND THE SHORE BAND"; "THE REACH IS A LAW"; "SEND THE FAR FOLD'S COUNTS TOO"; "KEEP THE POCKET EXEMPTION" | Retires all four (the first round's §8.4 omitted item 2). | Retire; item 2's spectrum conflict stays in DEFERRED for R2 (LAW-27). |
| B15 items 1, 2, 4 | "CURE THE FAR LAKE FLICKER FIRST" (the ruling's words) | Retires the wet tie and the lone hold. The far fold meets item 1's intent above `L_ref`. | Retire. |
| B16 | "We need seamless experience - no flicker at all please ... No flicker or changing of the shapes between rungs at all" | Met above `L_ref`; below it only through G-POND-FLICKER at 0 and G-SLIDE's line. | No amendment proposed; question 1 asks it only if the bench finds ponds that come back (LAW-2). |
| B17, the skin | "Shader's own level"; "86.4° now, judge in flight" | Retires the per-pixel windows. | Question 20: per-cell exactness (no memory) or B17's windows (17–79 MB per stand, MEASURED). |
| B18 Q1–Q5 | Q2: "So I'd have heights at any rung"; Q4: the slides "judged in the flight"; Q5: "no jumps or flicker between rungs" | Keeps Q2's intent (a height at every rung) and retires the lid and the dry word as data. Makes Q1 and Q5 by the fold. Q4's sizes were measured under N2. | Amend as named; Q4's sizes are one of G-SLIDE's two proposed lines (question 1). |
| B20 item 1 | "The coast first" | S0 builds the failing coast gate first; S0b after the epoch in the recommended order. | Keep, unless question 10 puts the builder first (LAW-29). |
| B21 items 1–2 | "don't try to connect"; the ruling: "every distance is a copy made from the same volume by one rule" | Adds the rim pass, a new rule between two sources of truth (LAW-28). Uses two ladder rules: the generator per rung up to `L_ref`, the fold above (§4.1). | Question 1 decides. Option (A) meets "one rule"; option (B) amends it. |
| T1, the order | "we need to work on the full 8 before 9; move all the topics before that" | The trench needs a minimal store and lane (slices 9 and 10) before 8e. | Question 7, after the detailed discussion that V4 row 14 asks for. |
| T1 item 6 and T2 | "I'd really like to give an ability to build real ships and sail" (T2) | The water's top is part of the one mesh, not a sheet "at the sea radius with the coast clipped". | Amend; 8o's "surface as a SHAPE" fits. |
| T7 rule 1 (K) | "Agree with all 3 ... all 3 will increase the visual quality and strengthen the existing machinery" | Question 2 measures a smaller K. | Measure only (M5). K is also a lever for LAW-2 (rarer ponds, not none). |
| S5-3 | "Then all is approved for the slice, please implement." (the ruling: "The generator is never asked to fold fine cells"; an entry holds "the dominant substance") | Above `L_ref` the coarse answer IS the fold of the reference rung. A5 adds the top's material shares to an edit's entry. | Amend. R-7 YES ("an exact cheap coarse fold inside the generator") already allows the fold. |
| R-16 (V6 Part E) | "The rest is approved." (R-16: "NO; an area's floor is built") | A6 is withdrawn. | Keep R-16 (LAW-4; question 19). |
| V4 row 14 | "we need to discuss IN DETAIL each of the topics before we start any of them." | Keeps it (question 7). | Keep. |
| V4 row 13 | "Remove is not Air — Air block means atmosphere. When we remove in space — it's simply EmptyBlock." | Keeps it: vacuum is `Empty`, and the medium is the realm's. | Keep. |
| V10 | "I think 8 should be ok" | Keeps 8 ms on the cell path (G-BUDGET). | Question 11: 8 ms for the whole build, per rung, as a target. |
| F6 | "We are building a game that can be run on an average machine." | Every budget stated for the instrument; the target owed. | Question 9: the owner names the machine. |
| F7 and F9 | F7: "we utilise gpu to the maximum"; F9: the card as a budgeted knob, its worth "OWED BY THE OWNER" | Keeps the card parked on solved bodies until the owner rules on P12's numbers. | Question 12 (F9's open item). |

### 10C.6 The gates this section adds

G-POND-FLICKER, G-SLIDE's line, G-FAR-EXACT in two parts, G-FAR-FIRST (its clock from the first far part held), G-NO-FACE,
G-VERSION, G-SPLIT, G-IDENTICAL on the planet's and an Area's shard, `gpu-drift` on solved chunks (only on question 12),
G-MU, G-REQUIRED and G-FENCE. All are in §9's one table, each with its failing control.

*Example.* The pilot of G-POND-FLICKER climbs from the belt's lowland to 450 km. The gate counts every pond that fades out
and every pond that fades in on the way. If one pond wider than a pixel fades out at rung 6 and in again at rung 7, the
gate is red, and the owner chooses: amend B16 for such ponds, or build option (A).

---

## 11. The review ledger

Every finding of both first-round reviews, with its answer, and below them the second round's review of the three
drafts. Where the second round changed a first-round answer, the row says so (SECOND ROUND). CURED = the design removes
it. ACCEPTED = a stated price, put to the owner where it is his to choose. OPEN = owed, with its home. N/A = it belongs to
a design part that this document does not adopt (its price stays in question 1 (A) where it applies).

**Review "laws and behaviour"**

| # | Sev. | Design | Finding | Answer |
|---|---|---|---|---|
| L1 | blocker | procedural | Water thinner than one cell is lost; almost no ocean from rung 13 up; coasts on isobaths. | CURED. The top byte `t = min(s, f)` draws the top at `max(ground, level)` at every rung; the colour from `t < s` (§1.1–1.2). Gates G-TOP, G-FLAT. |
| L2 | blocker | reference | The same loss; a solid cell carries no level. | CURED. Every cell, solid cells included, carries `t`; the extractor reads it (§1.1–1.2). |
| L3 | high | procedural | Every dug cell under its catchment's level floods, against W2. | CURED. The static fill covers the seed's own cells only (rank 1); a dug cell holds the medium until W2's flood reaches it (§3.1, §3.3). |
| L4 | high | procedural | The ridged lift drowns lakes and moves coasts at every rung. | CURED. The centring at every rung, rung 0 included (§2.1). Gates G-MEAN, G-LAKES-DRAWN; bench M1. |
| L5 | high | reference | Branch B: the lift shows lakes far and none near. | CURED by the centring; the bench measures slides without the one-sided mean (M4). |
| L6 | high | procedural | Spill faces at outlets and pocket rims, changing with the rung. | CURED. The rim pass (§2.3); G-NO-WALL at every rung up to `L_ref` as a gate of S1, with bench M3 before any build. |
| L7 | high | procedural | The breach row floods every open cell of a box. | CURED. The flood stores its result as rows of wet cells; no box row exists (§3.3). |
| L8 | high | reference | Divides on node midlines; sills cut lakes; the sill's width follows the rung. | N/A. No sill. Divides come from the descent step on one field (rung-independent) and the rim pass (§2.3–2.4). |
| L9 | high | reference | Branch A brings back the shore law's purpose. | N/A. Not adopted (B21). |
| L10 | high | reference | Point-sampled far rungs bring back the far-lake flicker. | CURED above `L_ref` by the far fold's exact share (§4.3); below `L_ref`, the fades are question 1's price (M4). Gate G-FAR-EXACT. |
| L11 | high | stored truth | The fold's unit: the client folds fine generated cells, about a thousand times the estimate. | N/A for (B): its fold reads rung-`L_ref` cells only. For (A) the corrected price is in question 1 (21–112 core-minutes; 0.5–3 s a ring). |
| L12 | high | stored truth | The cave rule floods a 512 m region per chunk. | N/A. Caves take their column's `λ`, chunk-local (§2.4). The price, faces inside caves at catchment boundaries, is counted (M10). |
| L13 | medium | procedural | A column's water body changes with the rung. | CURED. The descent step reads one field at every rung up to `L_ref`; the far fold above (§2.4, §4.2). Gate G-ONE-LEVEL. |
| L14 | medium | procedural | The far ground stays a second field; W17 stays. | CURED. Rows to rung 13, the far map from 14; `field_step_m` deleted (§4.2). |
| L15 | medium | procedural, reference | The flat water table is not a law; ponds at sea level inland. | ACCEPTED as a stated simplification (question 3), measured first (M2); the cures named (R2; option A) (§2.4). SECOND ROUND: no longer a price — a published water-table law comes first (M2b), and the flat level ships only on the owner's amendment of T9 and W11 (§2.4; §10C LAW-1; question 3). |
| L16 | medium | stored truth | The fill byte points at the ground over thin water. | N/A. Here `t` is the gap to `max(ground, level)` in every cell; the colour reads `t < s`, not a class (§1.1). |
| L17 | medium | stored truth | EXACT keeps the share, not the pixel; the windows' price is missing. | ACCEPTED. The far fold's share is exact per cell; the pixel's blur is stated; the windows' MEASURED price (17–79 MB) is in question 1 (§4.3). SECOND ROUND: B17's retirement is the owner's trade (question 20); G-FAR-EXACT is split per cell and per pixel (§10C LAW-7). |
| L18 | medium | stored truth | The zero-sum rule puts a pit under one child: a lattice. | N/A for (B). For (A): the hashed remainder and G-GRID/G-SPECTRUM on a bench are part of its price (§4.5). |
| L19 | medium | stored truth | The new fine ground is tied to both answers; F1's meaning. | CURED. (B) keeps the octave sum (centred); (A)'s new ground is its own choice; F1 keeps its octave count (§2.1, §4.5). |
| L20 | medium | stored truth, reference | W2 floods dug cells only; W2's canal example fails. | CURED. The flood wets connected open cells, dug and natural, in its bound (§3.3; question 4). SECOND ROUND: the bound's face and the two waters are no longer stated prices; each is REFUSED with a typed reason, never a wall (§3.3; §10C LAW-3). |
| L21 | medium | all | An Area realm owns cells but gets no water facts. | OPEN with its shape: SL6 ask A6; the flood stops at the realm boundary; the far view of an Area's edits belongs to the Area slice (§3.5). SECOND ROUND: A6 is WITHDRAWN (R-16 NO: an Area's floor is built); an Area's liquids come from its own rows (§3.5; §10C LAW-4; question 19). |
| L22 | medium | all | A hull has no water source; HR4's fixture is not identical. | OPEN: the kernels are identical given a source row; the hull's shipped source (a tank block) is owed with the hull slice (§5.5). SECOND ROUND: the flood reads "down" from the realm's gravity function; G-IDENTICAL runs on the planet's and an Area's shard; a hull's down is `ChildFelt` on trial, and the hull joins when its down and its source exist (§10C LAW-9). |
| L23 | low | stored truth | The tree needs new wire arms; its far reach is unstated. | N/A for (B). This design lists its arms and reaches (§6.3–6.4). |
| L24 | low | stored truth | Lowered lakes and joined pockets skip their laws. | CURED. The rim pass runs the budget and the sea's bisection again inside the existing fixed point (§2.3). |
| L25 | low | procedural | 50 km draws rung 6; rung 14 reads level 1. | CORRECTED (§3.4 uses rung 6; §4.2 reads `level_for` as coded). |
| L26 | low | reference | The sill's 0.70 is not the solve's repose; a u16 basin id caps basins. | N/A. No sill; the level map carries a level, not an id. |
| L27 | low | all | W2's river half has nothing to act on. | ACCEPTED. It waits for rivers (8d, 8o); the dam claim is removed; the first W2 fixture is the trench (§3.3). |
| L28 | low | all | The flood's neighbours at a face edge and a cube corner are unstated. | CURED. The flood uses the extractor's topology (`seam::across`, the three-column prism); the W2 fixture holds a corner case (§3.3). |

**Review "scale and cost"**

| # | Sev. | Design | Finding | Answer |
|---|---|---|---|---|
| C1 | high | all | Open water as one quad per cell: a measured 9 ms and about 1 GB of mesh. | CURED. The flat merge is in the first slice (§1.4); G-FRAME before the sheet is deleted. The "3.9 ms saved" claim is not made: after its blocks the sheet costs 3.0 ms at 6 376 km and 0 ms on foot (MEASURED). SECOND ROUND: that 3.0 ms was measured on the frame before the sky fix; its share of today's frame is UNMEASURED (bench P7; RV-1). |
| C2 | high | stored truth | The bake leaves out the rung-0 count. | ACCEPTED into option A's price: 21–112 core-minutes, 0.7–7 GB working set (§4.5). |
| C3 | high | stored truth | Tiles to the horizon are not priced. | PRICED. This design has the same reach: about 363 tiles, about 14.9 MB at about 3 193 km; the gateway cache bound (§4.2, §7). |
| C4 | medium | all | The chunk rate is quoted from a still eye. | ACCEPTED. §7 quotes 318–437 chunks a second on the moving eye; G-RATE on the 528 m/s leg; over water one chunk per column (the bed is not drawn). SECOND ROUND: the capacity after the levers is weighted by the ask's rung mix (790–1 330 a second, ARITH; §10A.3; RV-2). |
| C5 | medium | all | The shard composes cells for the first time, with no budget. | CURED. Off-tick workers, a bounded box cache in the shard's tuning, point evaluation for floods; M-1 before slice 10 (§3.6). SECOND ROUND: the flood runs on the injected composition worker, with its own parking field (§3.3; §10C LAW-22). |
| C6 | medium | all | No rebuild cadence for many diggers. | CURED. Coalesced rebuilds and a cached box for edited chunks; M-1 a gate (§3.6). |
| C7 | medium | procedural, reference | The old slides are converted to pixels at half their size. | CORRECTED: 3.7 px and 6.6 px at the finer cell; the bench prints pixels at the finer cell (§4.5, M4). |
| C8 | medium | stored truth | The tree's size rests on a count from the N2 world. | ACCEPTED into option A's price as a bracket (§4.5). |
| C9 | medium | stored truth | Exact integer folds cannot prune an edit. | For (B): edits fold by S5-3's rounded delta, pruned where it rounds to zero (under one gap step of that rung); the far fold's exactness is for the generated cells. For (A): a pruning rule is part of its price. |
| C10 | medium | stored truth | The column pass is not 20 times cheaper. | N/A. Not claimed. |
| C11 | medium | procedural | Side B's bake time is missing. | PRICED for this design's far map: 1–8 core-minutes, split over the server's cores (§4.3). |
| C12 | medium | procedural | The live-level box floods every open cell of the box. | CURED. Rows of wet cells (§3.3). |
| C13 | medium | procedural | Pockets show as sea until the level map lands. | CURED. The level map rides the rows and the far map, which the ground needs anyway; no chunk builds without its own field (§5.2). |
| C14 | medium | reference | Tiles are about 20 % more than stated; the gateway cache is not priced. | PRICED: about 363 tiles, about 14.9 MB; an eviction bound (unbounded: 88.7 MB per planet) (§4.2). |
| C15 | low | reference | A trench row priced at 7 B, not 12 B. | PRICED at 12 B + 3 B index (+1 B level index) (§3.4, §7). |
| C16 | low | reference | G-BUDGET's scope is unstated; it is red today as a whole-build gate. | STATED. G-BUDGET times the cell path against 8 ms; G-RATE times the whole build with parents (§9). SECOND ROUND: G-BUILD times the whole build in the loaded regime at 7 workers; G-RATE reads the band on the fast legs (§9; RV-2, RV-26). |
| C17 | low | procedural | A pilot at 50 km draws rung 6. | CORRECTED (§3.4; G-TRENCH-FAR at rung 6). |
| C18 | low | procedural, stored truth | 94.63 MB is the ARTIFACT_VERSION 5 figure. | CORRECTED: 102.8 MB today, 97.6 MB after (ARITH, §7). |
| C19 | low | procedural | The planet-wide face count costs hours. | STATED. Every bench states its time and cores; planet-wide counts sample tiles with a stated error (§10). |

### THE SECOND ROUND (2026-09-28): the review of the three drafts

The owner's four later requirements produced three drafts (performance, seams, laws). One review checked them against the
code and the rulings. It found no blocker, four highs, and medium and low findings; this round answers every one, not
only the blockers and the highs. CURED = the merged design removes it (the section named). ACCEPTED = a stated price, put
to the owner where it is his to choose. OPEN = owed, with its home. CORRECTED = a wrong number or citation, fixed in
place. The review also REFUTED one suspicion of its own: the ask sum may leave out the descent's 0.17 slack, because the
code applies the slack only to a rung that the bound pulls in (`crates/client/src/ladder_view.rs:709-711`); no change.

| # | Sev. | Area | Finding | Answer (the second round) |
|---|---|---|---|---|
| RV-1 | high | performance | G-FRAME's p50 ≤ 11.0 ms fails at the berth today (11.3 ms), and its sum takes an ablation from before the sky fix (the sheet's 3.0 ms on the 20.2 ms frame) away from a frame after it; a single max per 60 s leg against an UNMEASURED window maximum and headless worst frames up to 70.2 ms. | CURED (§10A.2 B-FRAME; §9 G-FRAME). Derived from every stand, the berth included: the gate is no stand worse than today's MEASURED p50 (11.3 / 13.4 / 11.4 / 9.5 ms), the 40 B vertex in, plus a p99 ≤ 16.7 ms and at most 1 frame in 1 000 over 16.7 ms per leg. 11.0 ms at every stand is the TARGET, reported, until P7's GPU capture names the berth's lever (then E14 or a vertex lever). The sheet's share of today's frame is UNMEASURED, and P7 measures it before the sheet goes. |
| RV-2 | high | performance | The throughput promise (about 1 460 a second; "about 1 400" to the owner) uses the rung-0 chunk only; rungs 1–2 are 40 % of the ask at 537 m and their geometry costs 8.56–9.72 ms against 4.09 ms; the 10 ns spline read is an unmarked ASSUMED; B-BUILD and G-BUILD read two regimes. | CURED (§10A.3, the ask-weighted table; §0.8). The chunk cost is weighted by the ask's rung mix (45.8 / 26.6 / 13.7 / 6.9 / 6.9 %): 5.3–7.6 ms a build, about 790–1 330 a second on 6.5–7 workers (ARITH), with the spline read ASSUMED at 10–40 ns. B-RATE and the owner's "about 1 000" are stated as conditional on P1, P2 and E6 (the rung-1/2 build must fall to about 9.8 ms). B-BUILD and G-BUILD read one regime: loaded wall time at 7 workers. |
| RV-3 | high | performance | E9 misquotes the card: the chain's 490 boxes a second took "about all of the card"; the shipped card built 95 a second at a quarter AND at the whole frame, raised the widest pop on four legs of five and broke the slow hull's band (3 357 urgent). | CURED (§10A.3 E9; §6.1 "The card"; question 12). E9 states the MEASURED 95 boxes a second, the chain's "about all of the card", and both measured harms; it is ruling F9's open owner question, not counted in B-RATE; its GPU time sits inside B-FRAME (a quarter of the period fits beside an 11 ms drawing). |
| RV-4 | high | seams | G-POP's line (≤ the still pixels' 0.1 % past the floor at every boundary) fails on today's code by 2–16 times (the walk 0.67 %; 240 m/s 0.21 %, rung 6 0.36 %; 528 m/s 0.98 %, rung 6 1.6 %); no cure targets these ground boundaries; the descent legs compared 0 pixels. | CURED as a gate; OPEN as a cause (§9 G-POP; §10B.6 X21). The gate NOW: no leg and no boundary worse than P0's baseline read by the same new probe, on pairs of flights, with at least 10⁶ compared pixels on every leg. The 0.1 % line becomes the TARGET only after P0 and B-S5 find the cause and the owner rules. |
| RV-5 | medium | performance | The ask's anchor height is unsourced; over the stated 860–1 700 m band `z` runs 1.11–2.18 and the 528 m/s ask 525–1 030 a second; the anchor flew the rough world; the code's estimate counts the leading edge only. | CURED (§10A.2 "The ask, derived"; P8). The ARITH range is carried; P8 measures the ask, `z` and the trailing edge before the owner rules B-RATE. |
| RV-6 | medium | performance | E1's "a finer chunk finds its parent built" fails at every ring's trailing edge (24–29 % of each rung's new chunks, 16 % of all); the cache keeps about one second, so P5's 95 % is out of reach. | CURED (§10A.3 E1; P5). This round's ARITH goes further: even at the leading edge a rung-L child is asked about 1.8 × 2^L s after its drawn parent was built, against about one second of cache history. So E1 is now THE SUB-BOX FIRST (about 18 % of a parent's cells on a miss); the drawn parent form stays only where P5 shows hits; P5's line comes from this geometry. |
| RV-7 | medium | performance | E8(c) budgets the harvest LOOP (0.07 ms an upload), not the engine's upload, which lies outside it. | CURED (§10A.3 E8(c); B-FRAME; P6, P7). The render world's cost per uploaded chunk is measured apart and budgeted under B-FRAME (about 180 MB a second to the card at 990 chunks of 182 KB, ARITH). |
| RV-8 | medium | performance | E8(b) runs the release walk only after a new wanted set, so a still eye keeps a coarser chunk drawn after its finer chunks land. | CURED (§10A.3 E8(b)). The walk runs on ANY change of its inputs: a new set, an arrival, a release, a tile landing, a cancel (DEFERRED's own trigger, `docs/design/DEFERRED.md:8931`). |
| RV-9 | medium | performance | Every budget binds on the owner's Mac, but F6 names an average machine; on 8 cores B20 gives 4 workers, 800 a second at 5 ms. | CURED (§10A intro, §10A.4; question 9). Each budget is stated for the instrument and owed for the target; ARITH on 4 workers: about 520–760 a second, so the bound binds at 528 m/s there (lawful, F9). The question names the machine (cores, memory, GPU class) and asks full detail at speed on it. |
| RV-10 | medium | performance | B-LOGIN mixes old and new sizes; its 3 s has no derivation; no budget covers a first solve and the rim pass. | CURED (§10A.2 B-LOGIN, B-BAKE). The far maps by ONE rule (about one byte a macro node, 0.64 × today's far data); the 3 s derived from the head's first beat, the six top-rung chunks and the near tiles (the ring builds UNMEASURED); zero urgent from the measured capacity (12–20 s ARITH, the line 15 s); B-BAKE with the owner's ceiling (question 21). |
| RV-11 | medium | performance, seams | B-EDIT counts one 5 ms rebuild, but the edit tick swaps a whole set (18 chunk builds and 4 parent builds in the draft's ARITH), with ticks joined and a queue wait. | CURED (§10A.2 B-EDIT; §3.6; §10B.4 [EDIT TICK]). Derived from the set on 7 workers (two rounds, parents then children), the join window (0 for the first whole tick) and a wait for a free worker: 140–175 ms ARITH, so the digger's budget is ≤ 180 ms p99 and the pilot's ≤ 250 ms; the set is 4–18 builds by the rungs the observer draws (B-S10). |
| RV-12 | low | performance | A frustum cut is listed as "refused by measurement", but no measurement exists. | CORRECTED (§10B.5 item 9): a reasoned refusal (the descent reads the eye point only; a turn would show holes). |
| RV-13 | low | performance | S0b fits E12 and E3 onto parts that S1 and S2 delete. | CURED (§9; §10A.6): E3 moves into S2, E12 into S3 with `ArtifactFar`. S0b keeps E1, E6, E8 and E10. |
| RV-14 | low | performance | The far fold's mixed cells and the wire sums use per-item costs that are too low (a sample needs about five spline reads; one tile every 990 s is 0.04 KB/s, not 0.2). | CORRECTED (§4.3; §10A.2): a coastal rung-13 chunk's fold is at least 1.3 ms before its geometry (ARITH, 410 × 64 × 50 ns, the inputs ASSUMED), so the far rungs are reported per rung, not gated at 1.5 ms; the steady tiles are 0.04–0.08 KB/s. |
| RV-15 | medium | seams | Seam vertices break G-FAR-EDGE by the draft's own X18: a face-seam vertex reads the field, with a binary share against the coarser fractional share; 223 coastal node pairs straddle a cube edge. | CURED (§4.4; §10B.4 [SEAM TARGET]): the seam vertex reads its target — position AND share — from the neighbour face's parent mesh, the exact cure that DEFERRED names; G-FAR-EDGE holds seam vertices with no exception. |
| RV-16 | medium | seams | The edit tick's part count fails under the gateway's per-session filter, which the owner approved; ChunkRow carries its tick already; no rule joins ticks while a set builds. | CURED (§3.4 steps 7–8; §6.4 A9; §10B.4 [EDIT TICK]): counts per (tick, rung) and the tick's rung mask; one rung per `BulkFor` with a plain rung header, so each filtered subset is whole by itself and the realm needs no per-session state; a join window in the client's tuning. |
| RV-17 | medium | seams | G-SLIDE still has no pass line for the slide. | CURED (§9 G-SLIDE; question 1): one line, put to the owner — B18 Q4's accepted sizes as the most that a coast may slide at any band (recommended), or W17.1's p99 ≤ 1 px and max ≤ 2 px. |
| RV-18 | low | seams | The draft settled coast slides with the wrong ruling item (B18 Q5, where Q4 accepted the slides, at N2's sizes). | CORRECTED (§10B.7; §4.5; question 1): Q5 read the words; Q4 accepted sizes measured under N2, "judged in the flight"; stated only as question 1, with B16's words beside it. |
| RV-19 | medium | laws | LAW-17 contradicts the owner's own answer on the gateway filter, and its cure needs per-session state that the realm does not hold. | CORRECTED (§10C LAW-17; §6.1): the verdict is KEEPS; the filter reads a plain rung header, never decodes rows and forwards one shared buffer. |
| RV-20 | medium | laws | LAW-10 calls the idle card a break of F7, but F9 is newer and kept the card a knob. | CORRECTED (§10C LAW-10; question 12): an open F9 owner question, with the measured 95 boxes a second and the two measured harms. |
| RV-21 | medium | laws | LAW-6's line (max ≤ 2 px) rejects the slides that the owner accepted (up to 4.79 px). | CURED with RV-17: one line, the owner's choice. |
| RV-22 | medium | laws | G-FAR-FIRST's "0 frames with the body in view and no drawn rung" cannot pass: a body enters the window at its reach (about 23 px) before its artifact is served; the seams draft names that pop-in as the reach's. | CURED (§5.2; §10B.4 [EPOCH] item 4; §10C LAW-8): the clock starts when the first far part is HELD; the reach's pop-in stays a stated residue owed to the brightness radius; one rule in both sections. |
| RV-23 | medium | laws | LAW-5's one-epoch cure (S1 + S2) leaves the far view without a field until S3: the tiles grow to the whole visible body (up to 2 166 tiles, 88.7 MB) and rungs 10–18 read point samples. | CURED (§6.3; §9): S1, S2 AND S3 land as ONE world epoch, built and gated as three steps and merged once. |
| RV-24 | low | laws | LAW-9 misses the planted `ChildFelt` down, and its Area fixture conflicts with LAW-4 (an Area gets no planet rows). | CURED (§3.5; §5.5; §10C LAW-9): `ChildFelt` (R-20, YES ON TRIAL, default off) is named as a hull's down; the Area fixture is given its source row. |
| RV-25 | low | laws | Four verdict errors: LAW-4 cites HR1; the rim pass's ceiling on one thread against a split pass; `parked_ticks_max` reused for the worker's wait; the coverage gate moved after S3 with no owner question. | CORRECTED: (1) HR1 dropped from LAW-4; (2) the ceiling is stated in wall time on the production cores (LAW-19; question 21); (3) the flood's wait gets its own tuning field (§3.3; LAW-22); (4) no deferral: 0 misses on touched lines, the full gate before every merge, B4's owed pass first (§9; LAW-15). |
| RV-26 | medium | all three | The drafts give different lines for G-FRAME, G-POP, G-RATE/G-BAND and G-BUDGET. | CURED (§9): ONE gate table, one line and one failing control for each gate. G-BUDGET keeps V10's cell path; G-BUILD is the whole build (question 11); G-BAND asserts the walk and the slow hull; G-RATE is the performance target. |
| RV-27 | medium | seams, performance | The 40 B vertex is missing from B-MEM and B-FRAME. | CURED (§10A.2; §10B.8; §7): B-MEM counts 52 B a vertex (about 2.9 GB drawn at the walk, ARITH; the draft's 2.5 GB ceiling withdrawn); B-FRAME's non-regression line includes it; the harvest's byte budget grows by the same 11 %. |
| RV-28 | medium | seams, laws | On ponds, the seams draft offered a price that the laws draft says the owner cannot accept. | CURED (§4.5; §10B.7; §10C LAW-2; question 1): one position — no price; G-POND-FLICKER at 0, or the owner's amendment of B16, or option (A); a smaller K makes them rarer, not zero. |
| RV-29 | low | all three | "A7" named two different asks; the law check left out the seams draft's data and the performance levers (S0b against B20 item 1, E8's thread, E12's serve, E9 under F9). | CURED (§6.4; §10C): one ask table (A7 the brush, A8 the refusals, A9 the edit tick's counts and the rung header, A5 widened, A6 withdrawn) and one law check over the merged design (LAW-29 to LAW-33). |

**The drafts' own findings.** The seams draft's search (X1–X20, and X21 from the review) is answered in §10B.6, and its
replacements (SW1–SW12) in §10B.3. The laws draft's findings (LAW-1 to LAW-28, and LAW-29 to LAW-33 from the review) are
answered in §10C.3. The performance draft's causes (H1–H9) are answered by the levers E1–E14 of §10A.3.

---

## 12. The three designs and this one

| | Procedural first | Stored truth first | Reference first | THIS DESIGN |
|---|---|---|---|---|
| The cell | substance, `s`, a derived fill gap `f` | a 16-bit kind, a fill to the top of its class run, a class byte | a solid layer and a fluid layer, two codes and two gaps | substance, `s`, a derived top gap `t = min(s, f)` |
| The top surface | class edges; thin water lost (blocker L1) | top of the class run; wrong over thin water (L16) | class pairs; thin water lost (blocker L2) | ONE sign field `t`: `max(ground, level)` at every rung |
| The fine ground | the octave sum plus a far "mean tail" | a new conservative synthesis | the octave sum (Branch A fades it near water) | the octave sum, centred at every rung |
| Water under a node | the level map by a descent step; spill faces (L6) | a stored 32 m tree and a local law | the basin share on node midlines, and a sill (L8) | the level map by a descent step on one field, and the rim pass |
| Dug cells | the static water table (against W2, L3) | W2 on dug cells only (L20) | W2 on dug cells only (L20) | W2: connected open cells, dug and natural, bounded; rows |
| The far ground | the mean pyramid and W17 (L14) | exact sums, a fold from rung 5 | rows to the horizon and a sample pyramid | rows to rung 13 and the far fold above `L_ref` |
| The far water | point samples, or a stored share | the tree's shares | point samples (flicker, L10) | the far fold's exact share (no flicker) |
| New data | a 1–3 MB level map | +8.8–9.5 MB head; 57–150 MB server | basin ids; ~15 MB of tiles | the row word re-meant; ≤ ~14.9 MB tiles at the worst height; 8.9 MB far map in place of 14.0 MB |
| Server bake | none (side B unpriced) | 21–112 core-minutes (corrected) | none | the rim pass (2–84 core-min ARITH) and the far map (1–8 core-min ARITH) |
| Reviews (both; C1 applies to all three) | 1 blocker, 5 highs (L1; L3, L4, L6, L7, C1) | 0 blockers, 5 highs (L11, L12, C1, C2, C3) | 1 blocker, 5 highs (L2; L5, L8, L9, L10, C1) | every finding answered (§11); the second round's review of its three drafts: 0 blockers, 4 highs, every finding answered (§11, RV-1 to RV-29) |

**Why this design.**
1. **Procedural first gives the base.** Its solid pass reads no water, it stores the least, and the scale review ranked it
   cheapest. Every defect that the reviews found in it has a local cure that adds no terrain rule: the top gap (L1), the
   centring (L4), W2 as written (L3, L7), the rim pass (L6) and one field (L13, L14).
2. **Reference first gives the one field** (rows out to rung 13, a sample of one field above), which removes W17's second
   field, and it gives W2's live rows and the wet coverage. Its sill and its Branch A read the water from the solid pass,
   so they stay out (B21).
3. **Stored truth first gives the far fold's idea:** a coarse cell as the exact mean of finer cells, and a coast cell at
   the mean height of what a pilot sees. This design uses that idea only where it is cheap and exact — above the
   reference rung, where the ground is the solved field alone — so it needs no new fine ground and no stored tree. The
   full stored-truth design stays as question 1's option (A), with the scale review's corrected prices.
4. **The owner's words decide the rest.** The design stops fitting parts together: one cell, one fill, one surface, one
   field, one far rule. The far-coast question is a clear choice with a price on each side, and every unmeasured number
   has a bench before any build.
5. **The owner's later words change no choice of the model; they add its budgets, its seam mechanisms and its law
   check.** The one cell, the fill, the one surface and the far fold stay. The second round adds the speed levers that
   remove today's repeated work without moving a world byte (§10A), the mechanisms that keep every replacement
   continuous and stable (§10B), and one compliance table over the merged design (§10C). It withdraws or corrects what the
   review found in the first round: the Area's planet rows (A6, against R-16), the water faces at a flood's bound and
   between two waters (now refusals), the flat water table as a price (a published law first), and the claim that
   `compose()` stays unchanged.

---

*The living status file (`docs/investigation/2026-09-24/terrain_believability_status.md`, ruling B6) owes one line for
this document when the owner reads it. Both rounds wrote only this document, as the workflows asked.*
