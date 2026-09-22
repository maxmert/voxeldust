# Tetrahedral cages for ray-traced animation — a note for later (2026-09-22)

**What this is.** A reading of one paper, and the answer to one question: *could we use it in Bevy?*
Nothing here is built, and nothing here is binding. The verdict today is **not now**, so this
document exists to be re-read when the three conditions in section 8 hold.

**The paper.**

> Holger Gruen, Carsten Benthin, Michael Kern, David McAllister.
> **"Ray Tracing Massive Amounts of Animated Geometry."**
> Proc. ACM Comput. Graph. Interact. Tech. 9, 4, Article 49 (July 2026), 18 pages.
> DOI: <https://doi.org/10.1145/3820014>
> Author's version (free): <https://gpuopen.com/download/TetrahedralMeshes_AuthorsVersion.pdf>
> Read on 2026-09-22. The AMD copy is the one to fetch again; it needs no account.

---

## 1. What the paper does

A ray tracer must hold a tree over the triangles it traces. When the triangles move, that tree must
be rebuilt. The cost of the rebuild grows with the number of triangles. The paper breaks that link.

1. Before the game runs, a **cage of tetrahedra** goes around the object in its rest pose. The method
   cuts the object into voxels, throws the empty voxels away, and cuts each remaining voxel into six
   tetrahedra.
2. The triangles are **clipped** against the tetrahedra. Each tetrahedron then holds its own small
   mesh and its own small tree. That tree is built once and never changes again.
3. Each frame, the animation moves **only the cage's points** — about two thousand of them — instead
   of a million vertices.
4. Each tetrahedron goes to the graphics interface as **one instance**. Its matrix is the basis
   change from the rest tetrahedron to the moved tetrahedron. The hardware bends the ray into the
   untouched inner tree. The deformation is therefore piecewise linear.
5. Only the **top tree over the tetrahedra** is rebuilt each frame.

A second variant keeps the ray in world space and stores the inner tree in 4D barycentric form. It
is watertight, and it is a reference only (see the numbers below).

## 2. The numbers the paper measured

On a Radeon RX 9070 XT, at 1080p, two rays per pixel:

| Item | Value |
|---|---|
| The combined scene | 585 million animated triangles, 2.8 million tetrahedra |
| Frame time | 12.43 ms (≈ 60 fps), of which the update is 9.66 ms (78 %) |
| Memory | 770 MB |
| Gain against the standard method | up to 9× in total time, 16× in memory |
| Cost of the clip step | 1.3–2.3× more triangles, 1.4–3.7× more vertices |
| The watertight variant | 19–80× slower to render; 2.3–3.2× the memory |

The paper's own limits: the method suits dense foliage, crowds and bone animation. It does not suit
explosions, cloth folds below the cage scale, or **rigid objects — for those the paper says ordinary
instancing is simpler and usually better**.

## 3. What the method demands of the graphics interface

1. Hardware ray tracing, at least a ray query inside a shader.
2. A two-level tree with a 3×4 matrix per instance.
3. A top tree rebuilt every frame over millions of instances.
4. The instance matrices written **by the card**, in a compute shader. The paper dispatches one
   compute shader to move the cage points and a second to build the matrices.
5. For the watertight variant only: boxes as geometry, plus a custom intersection shader.

## 4. What Bevy gave us on 2026-09-22

Read from the pinned sources on this machine, not from the documentation. **These facts go stale —
re-check them (section 9) before you trust them.**

Versions: `bevy 0.18.1` → `wgpu 27.0.1` (the lock file).

| Need | wgpu 27.0.1 | Evidence |
|---|---|---|
| Ray query, BLAS and TLAS | Yes on Vulkan and DX12 (tier 1.1, shader model 6.5) | `wgpu-hal-27.0.4/src/vulkan/adapter.rs`, `dx12/adapter.rs:525` |
| The same on the Mac | **No** | `wgpu-hal-27.0.4/src/metal/device.rs:1650` — `unimplemented!()` |
| A 3×4 matrix per instance | Yes | `wgpu-27.0.1/src/api/blas.rs:49` — `TlasInstance.transform: [f32;12]` |
| Boxes as geometry | **No** — triangles only | `wgpu-types-27.0.1/src/lib.rs:7898` — `BlasGeometrySizeDescriptors::Triangles` is the only arm |
| Instances written by the card | **No** — the list is a CPU `Vec`, and each instance becomes its own `Vec<u8>` before one staging copy | `wgpu-27.0.1/src/api/tlas.rs:25`, `wgpu-core-27.0.3/src/command/ray_tracing.rs:294`, `wgpu-hal-27.0.4/src/vulkan/device.rs:2814` |

Bevy's own ray tracing, `bevy_solari` (an optional 0.18 feature we do not enable), creates a **new**
top tree every frame and writes one instance per mesh from the CPU
(`bevy_solari-0.18.1/src/scene/binder.rs:72,166`). It accepts a mesh only when its attributes are
exactly position, normal, UV and tangent.

**The consequence, as arithmetic (UNMEASURED).** A Vulkan instance record is 64 bytes. The paper's
scene holds 2.8 million tetrahedra. Through wgpu the CPU must therefore write about 180 MB, and make
2.8 million small allocations, every frame, to fill a list the paper fills with one compute dispatch.
That deletes the paper's headline. A small scene stays in reach: 25 trees at 2 320 tetrahedra each is
58 000 instances, about 3.7 MB per frame.

**The escape hatch.** `Tlas::as_hal::<Vulkan>()` and `Blas::as_hal()` exist. A raw Vulkan build from a
card-written instance buffer is therefore possible, outside wgpu's own tracking. That is a large and
unsafe piece of work, and it ties the client to one back end. We do not want it.

**The Mac's path.** Metal acceleration structures landed in wgpu 30
(`wgpu-hal/src/metal/device.rs:2100` on the v30 tag), together with boxes as BLAS geometry
(`docs/api-specs/ray_tracing.md` on v30). Bevy 0.19 pins wgpu 29; Bevy 0.20-rc.1 pins wgpu 30. So the
Mac gets hardware ray tracing two Bevy steps out, behind the Bevy upgrade gate (no 0.18 → 0.19 before
S11 and S12 land and a hull is flown in a window). The card-written instance list is **still absent**
in wgpu 30 (`wgpu/src/api/blas.rs` on v30 holds the same CPU `TlasInstance`).

## 5. Does our world have the problem this paper solves? Today, no

- **The client rasterises.** The words `ray_query`, `acceleration_structure` and `solari` appear
  nowhere under `crates/`. There is no ray-traced light to accelerate.
- **No animated mesh exists.** The morph targets in `crates/client/src/chunks.rs` are the ladder's
  blend between rungs, not animation.
- **The land never moves.** T3 forbids erosion, and the authored override is the only shape change
  after the freeze. A shape that never changes needs no rebuild, and the rebuild is the whole prize.
- **A hull is rigid.** The star system authors the hull's placement and ships it down as one
  placement. The paper itself rules rigid objects out.

## 6. Where it would fit later

**The example, in the game's words.** The pilot lands a hull on the crimson desert. A slope of grass
and a stand of trees bend in the wind in front of the cockpit. Ruling V4 states that grass, trees and
decoration are art assets, blended dynamically. Suppose the client one day traces the star's light
for a shadow or one bounce. Every patch of grass then needs its tree rebuilt each frame, and a
hillside holds a million blades. This paper is the cure for exactly that picture: the client moves
about 700 cage points per patch, instead of 60 000 vertices, and the patch's inner trees are built
once when the art asset loads. The same cage serves every copy of the patch with its own wind, which
is where the memory saving comes from.

**The second candidate, weaker: the ocean.** The sea surface is a shape that moves. W3 rules that the
waves and the current are look, and a water sheet is cheap geometry, so its tree is small. The cage
would buy little here.

**Not a candidate: a hull that bends.** A damaged hull changes its blocks, which changes topology.
The paper's cage cannot express that; the inner mesh would have to be rebuilt.

## 7. What we would steal even without ray tracing

Little. A rasteriser bends the grass in the vertex shader for free. The cage earns its keep only
where a tree over triangles must be rebuilt, which means ray tracing or a collision structure. Keep
the idea in the drawer, and do not build it for the rasteriser.

## 8. The verdict, and the three conditions to revisit

| Question | Answer (2026-09-22) |
|---|---|
| Can the fast variant map onto Bevy at all? | Yes. It needs only BLAS, TLAS and a ray query, which wgpu 27 has on Vulkan and DX12. |
| Can it run at the paper's scale through the safe interface? | No. The instance list stays on the CPU. |
| Is the watertight variant possible? | Not on wgpu 27 (no boxes). Possible from wgpu 30, and 19–80× too slow anyway. |
| Should we build it now? | **No.** We hold no ray tracer, no animated mesh, and no ray tracing on the owner's Mac. |

Re-read this document when **all three** hold:

1. The client owns a ray-traced light path (a shadow or a bounce), not only a rasteriser.
2. Dense art assets — the grass, the trees — bend in the wind, in large numbers, each one its own.
3. Bevy sits on wgpu 30 or later, so the Mac can trace at all.

## 9. What to measure again when we come back

Every wgpu fact above is a reading of one pinned version. Check each again:

1. Does the TLAS instance list still live on the CPU? Look for a card-written instance buffer in
   `wgpu/src/api/tlas.rs` and `blas.rs`.
2. Does the Metal back end build acceleration structures, and does the Mac in the room report the ray
   query feature?
3. Does the BLAS accept boxes, and does the ray query report a box candidate?
4. Does Bevy's own path (`bevy_solari` or its successor) still rebuild the whole top tree each frame
   from the CPU, and does it accept a mesh with our attributes?
5. Then measure, never argue: build one cage around one grass patch, write N instances per frame from
   the CPU, and record the frame time against N. The number that decides everything is the instance
   count the CPU can write inside our frame budget.
