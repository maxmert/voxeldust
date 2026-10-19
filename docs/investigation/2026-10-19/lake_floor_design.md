# A lake's floor and its shore — a short design (the owner, 2026-10-19: "Design lake floors")

Binding context read first: B105 Q1 (b) (every standing water's points under its level are outlets, its shore the smooth
base's contour), B115 (a lake stands at its 128 m spill), W11 (a lake is water that stands), W16 (one shape at every
distance), B23 / B38 (no caps, no knobs, published laws), B76 / B93 item 2 (the pictures decide, from the side), SL10 (the
same bytes on every host).

## 0. For the owner

### 0.1 What the window showed

At 14.5 km over the basin lake (2026-10-19): *"Doesn't look believable"*. The node-square plates were a defect of form (a′),
cured in FINE_VERSION 7. Two older forms remain:

1. **Small lakes have rounded or star-shaped shores.** On the bench's map of the basin's 25 tiles, a lake of one 8 km node is
   a rounded square, and a lake of a node and its neighbours a four-pointed star.
2. **The dry pockets have broad smooth floors** (B114's parked basin defects; not this design).

### 0.2 Why the shores are rounded

Under B105 Q1 (b) each lake spreads from its own 8 km nodes through the 128 m points whose SMOOTH BASE (the 8 km rows read
by Catmull-Rom) stands under its level, and those points are outlets fixed at that base. So the lake's floor is the smooth 8 km
surface, and its shore is that surface's contour at the lake's level — a rounded square around one low node, a star around a
low node's cross of neighbours. The land around the lake is solved by the law and drains to that smooth edge.

Real lakes in hilly land are not like this: a lake floods an ERODED landscape, so its shore runs up the drowned valleys and
around the drowned spurs, and branches (a reservoir behind a dam, a lake in a tectonic basin, a fjord-lake). Its deep floor
is flat where sediment fills it.

### 0.3 The candidate: the drowned basin

The solve takes each lake's basin as LAND that drains to the lake's deepest node — that node's floor is the basin's only
outlet inside the lake (the dry end's own rule) — and the water map then floods the eroded ground at the lake's level. The
shore follows the drowned valleys. The sea, the rivers' exits and the dry ends stay as today. B115's spill law is
unchanged: the lake's level is the lower of its 8 km word and its 128 m pass.

**Example.** A one-node lake at 270 m with its floor row at 263 m: today its shore is the smooth surface's 270 m contour, a
rounded square 8 km wide. In the drowned basin, the node's ground is eroded by the law, its mean held at 263 m, its valleys
draining to the node's lowest point; the water at 270 m fills the valley floors and stops at the spurs, so the shore is a
branching line.

### 0.4 What it does not cure

- **The deep floor** is the eroded land's form under water, not flat sediment; it is never seen through deep water (B109's
  opaque depth law), but a shallow lake shows it.
- **The dry pockets' smooth floors** (B114).

### 0.5 The bench (bench only; `fine::l1::OutletRule::Drowned`, `planet_stand water-game … drowned`)

MEASURED on the basin lake's 25 tiles (8 lakes in their domains), beside today's FINE_VERSION 7 rule:
- **The levels.** In the drowned basin the 128 m rims cut fewer lakes: the basin lake holds its 8 km level −2 002 m (today
  −2 721.25 m), one −341 m lake holds (today −731 m), the other drops to −487.5 m (today −775.5 m); −1 312 → −1 691.5 m
  (today −1 529.5 m); −2 984 → −3 053.25 m (today −2 985.5 m).
- **The shores (from above).** The basin lake floods eroded valleys: branching shores, peninsulas and islands, no rounded
  blob or star. The water covers 523 000 cells of the ring against today's 122 000.
- **A defect seen**: a large flat dry area with straight edges on node lines at the lower left (a dry end's playa beside the
  drowned lake — cause NOT MEASURED).
- **From 14 km**: the drowned shores are steep, with a regular saw-tooth along some spurs.
Pictures: `~/Downloads/voxeldust_lakes_2026-10-19/basin_from_above_v7_today.png`, `…_drowned_candidate.png`,
`basin_14km_south_drowned_candidate.png`.

### 0.6 The flat straight-edged floors: their cause found (2026-10-19)

MEASURED (`planet_stand dry-ends`, the home planet's 8 km world, no solve): 6 584 nodes end their water on dry ground (no
water, no receiver), and 5 451 of them lie in 148 clusters of more than one node (the largest 2 322 and 1 106 nodes): the 8 km
world ends the water of EVERY node of a dry hollow, not of its lowest alone. The fine level gave every such node a dry end's
playa (its points under its row fixed at the smooth 8 km surface), so each hollow's floor stood as smooth plates one node
square each — the drowned bench's flat area at the lower left, and the broad smooth floors in every map of the basins (today's
rule too). The same mechanism as form (a′)'s node plates (cured in FINE_VERSION 7 for a lowered lake's floor).

THE CURE, written (bench only until the owner's word; FINE_VERSION 9): from a node that ends dry the path goes down to its
lowest neighbour that also ends dry and stands lower, until none does — the hollow's PIT (`l1::Waters::dry_pit`). Only the pit
keeps a dry end's playa; the hollow's other nodes are land whose base level is the pit's floor (Powell). Its pictures beside
today's are OWED (the bench runs when the window's warm-up is done).
