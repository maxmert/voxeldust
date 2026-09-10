# Slice 8, step 5's second half — the refutation and its answers (2026-09-10)

The refuter (read-only, Opus) attacked the parent mesh's shrink, the render-world-only mesh, the
byte-budgeted parent cache, the memory readers and §17.6. Twelve findings; every one answered
below. The gate chain runs on the answered tree.

| # | Finding | Verdict | Answer |
|---|---|---|---|
| 1 | `ParentMesh::build` guarded both bucket passes with `if let Some(c) = parent_bucket(a, b)`, and the `None` side is unreachable (the extractor keeps every vertex inside the box, its own test asserts the bound) — a never-taken branch, the coverage gate goes red. | CONFIRMED | Both passes index through `bucket_of(a, b)`, which `expect`s the bound with the reason named; a vertex outside the box stops the build instead of skipping a bucket silently. `parent_bucket`'s `None` stays on the lookup path, where a neighbour parent reaches it in production and a test covers it. |
| 2 | The "256 MB budget" was a COUNT of entries at an ESTIMATED size (`PARENT_MESH_BYTES_ESTIMATE`), computed once; the cache never read `ParentMesh::bytes`. A parent larger than the estimate overshot the budget silently, and the shipped 1 024-entry setting had no memory measurement behind it. | CONFIRMED | The cache now bounds itself by the meshes' OWN bytes: `set_budget_bytes(bytes, floor)` keeps `held_bytes` on every keep, trim and forget, trims the least recently used while the bytes pass the budget and more than the floor is held, and lifts the count bound out of the way; the engine passes the 256 MB and the workers' working set as the floor. The estimate constant is deleted. A unit test covers the trim, the floor, a replaced key and a forgotten realm. The final chain (pictures, the moving eye) flies the byte-bounded cache and its numbers replace §17.6's "512 entries" row. |
| 3 | §17.6 said the `f32` offsets "stay under a few kilometres" and the step "under a millimetre at every rung"; the code's own comment says 8 mm at 127 km. | CONFIRMED | The paragraph now states the offset (half the box: 65 km at the home planet's rung 11, 1 050 km at rung 15) and the step (15 µm at rung 1, 7.8 mm at rung 11, 12.5 cm at rung 15), and that the step is 7.4e-6 of the cell at every rung. |
| 4 | §17.6 credited the hand-built parent test with finding the `RAY_SLACK` miss; that test uses exact `f32` coordinates and passes at 1e-9. The real-parent radial test is the one that failed. | CONFIRMED | The paragraph names the real-parent test. |
| 5 | `read_band` reads the footprint right before the leg's clock starts; the tool takes a task snapshot of a multi-GB client, and the walk leg is then asserted on `max_urgent == 0` from its first sample. | PLAUSIBLE | MEASURED: the walk leg flew with the read in place and peaked at 0 urgent chunks over 1 377 samples, the first included (the moving-eye log of 2026-09-10, Docker on). The read costs the client nothing it draws on: the tool reads the task's memory map, the client keeps rendering. Left as built; if a first-sample gap ever appears, the read moves before the throttle. |
| 6 | `RAY_SLACK` at 1e-5 is 2.5× the per-component `f32` step, three vertices perturb independently, and a sliver triangle's barycentric error is unbounded by the constant. | PLAUSIBLE | Answered by the measurement and the fallback: a miss falls back to the field and the census counts it (`morph fallbacks`), unchanged across the four stands before and after the shrink (ground 7 985 of 48 M). The refuter's own check stands: an accepted hit stays within 1e-5 of an edge of its own triangle and the candidate set is unchanged, so the slack cannot return a wild radius. |
| 7 | The live memory test fails off macOS (`footprint`, `heap`, `vmmap` are the platform's own). | CONFIRMED risk | The test is `#[cfg(target_os = "macos")]`; the parsers' tests run everywhere. |
| 8 | `footprint_category` split the row at the last digit, so a category with a digit in its name (`tag 22`) could never match. | CONFIRMED, harmless today | The parser now reads the counted columns as words (number, unit, number, unit, number, unit, count) and joins the rest as the category; `tag 22` is in the test table. |
| 9 | `assert!(matches!(…))` in the new tests, against HR5's discipline (d). | CONFIRMED, not a red gate | The house form (`format!("{:?}").contains(…)`). |
| 10 | `pack` takes the Wide form at exactly 65 536 vertices, which `u16` could still index. | CONFIRMED, safe direction | Left: the doc's "fewer than 65 536" describes the code. |
| 11 | `vmmap_rows` dropped the first of the two column-name lines. | Cosmetic | The row above `REGION TYPE` is kept when it is not blank; a test covers both shapes. |
| 12 | Item 10 flipped 🟩 before a gate ran; the shrink test promised a hit half it did not assert; `parent_shrink_tests` re-implements `surface_key`. | Process | The flip stands with the gate chain's result recorded below; the test now asserts the hit half (some cell of the grid answers the ray through vertex 0 with its own radius); the helper stays (three lines, a different module). |

**Sound, per the refuter:** the render-world-only mesh (no read of a chunk mesh after insertion
anywhere in the client crates; the culling box is explicit on the chunk and its probe twin; the
ruler ball keeps the default usage); the CSR construction and its edges; `parent_bucket` a
bijection on the grid; `bytes()` counts what the struct holds; `position()` loses nothing past
the narrowing; the Narrow pack cannot truncate; the pid in the moving-eye gate is the client's;
the lints; the coverage of the new `ParentTriangles` surface.

**The gate chain on the answered tree:** recorded in §17.7 when it ends.
