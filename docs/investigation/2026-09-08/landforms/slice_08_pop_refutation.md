# Slice 8, step 6 — the pop detector's refutation and its answers (2026-09-12)

The refuter (read-only, Opus) attacked the step 6 change set. Fourteen findings; each answered.

| # | Finding | Verdict | Answer |
|---|---|---|---|
| 1 | The camera, the wanted set and the chunks' placement each sampled the display time and the delivered snapshot on their own, so the stamped eye and the drawn ground were a few milliseconds apart — a metre at 528 m/s, a pixel at a kilometre, straight into the detector's floor. | CONFIRMED | ONE MOMENT PER FRAME: `place_camera` samples once into the render eye (`RenderEye::moment`); the wanted set, the stamp and the placement read that sample and never a second one. |
| 2 | `sum_reads` took the worst per-pair floor beside the sum of per-pair counts past their own floors: a leg line read "floor 65, 54 past it, the widest 64". | CONFIRMED | A reading carries HISTOGRAMS (still, boundary, per boundary); a leg's sum pools them and every number — floor, count past it, widest — is derived once from the pool. |
| 3 | On a spinning parent the stamped eyes' travel is the interpolation chord, not the motion, so the near limit ate a third of the picture on the turning leg. | CONFIRMED by the run | The chord is item 21's defect, not the detector's; the near limit each pair applied is now on the leg line so a runaway limit is visible. |
| 4 | The `hold_s < SPIN_HOLD_MIN_S` guard could never fire (the rest band stops first), and the recorded turning leg flew with constants the tree no longer held. | CONFIRMED | The dead guard is deleted; the rest band is the one stop rule; the legs are re-flown on the shipped tree and the record restated from that flight. |
| 5 | A captured frame with no stamp kept the poll's stamp of a later frame beside its own capture index. | CONFIRMED | The dump beside a capture carries the captured frame's stamp or NONE. |
| 6 | The box's facing was narrowed to f32: one ulp of a unit quaternion on a 6 371 km lever is 0.7 m of stamped eye, per frame, on a turning hull. | PLAUSIBLE | `RealmBox::facing` is f64 end to end; the render primitive narrows once where it needs f32. |
| 7 | A pair with under a thousand still pixels reads its floor as its single worst pixel; the leg's max of those was the bar. | CONFIRMED | Pooled histograms (finding 2). |
| 8 | The colour matched the best of the 3 × 3 neighbourhood but the rung was read from the centre pixel alone, so a boundary that slid a pixel and a half counted as still and put its handover into the floor. | CONFIRMED | A pixel is a boundary pixel when ANY terrain pixel of its neighbourhood was drawn by another rung; keyed by the finer rung. |
| 9 | The readback ring held 62 MB for the whole session, capture pending or not. | CONFIRMED | The ring is kept only while a job is pending or within a ring's worth of frames after the last served capture; otherwise a slot holds its freshest readback alone. |
| 10 | `cfg.shot` now counts served jobs, written or not. | CONFIRMED, low | Named as the serve counter (the fallback stem). |
| 11 | A probe of a different size than its picture would index out of bounds in a Tier-A function. | PLAUSIBLE, low | A frame that is not one picture and one probe of its stated size reads nothing (tested both ways). |
| 12 | §22.3's after-the-fix numbers were another run's; the turning leg's lead excursion (2 741 m) went unreported. | CONFIRMED | §22 is restated from the final flight; the excursion is item 21(a)'s own measurement, recorded there. |
| 13 | Ruling V14 D8-5 flies the slow leg in a HULL at 1.4 m/s, never a walking dot; the moving eye walks a character. | PLAUSIBLE | Recorded for the owner's word (§22.5): the walk leg is the band's own gate from step 4, the detector reads it; a hull at 1.4 m/s is a fifth leg if the owner wants the ruling's letter. |
| 14 | Small doc-vs-code mismatches: "one cell coarse" (half a cell); the ladder's doc claimed the stamp's horizon floored (it is not); §22.4's rung numbers are a comment, not a gate; the example still spelt 1.6. | CONFIRMED | All four corrected; the example reads `EYE_HEIGHT_M`. |

**Sound, per the refuter:** `unproject` is the exact inverse of the ruler gate's projection (same
basis, focal, aspect and y flip; the probe's distance is Euclidean from the camera, which is the
render origin); `frame_camera` rebuilds the renderer's basis and field of view; the stamped eye
is the camera's, offset included; the stamp rides the readback from the extract and every path
of the placement leaves this frame's stamp in the resource; the exact-frame path leaves the
pending job consistent and cannot hang; the encode thread borrows nothing; the live facing has
no mixed consumer; the ladder's floor touches only the wanted set; HR5 both sides on every new
branch; the probe's cell ladder is the judge's; the spin's sign and the hull-in-planet reading
are right; the leg order and labels are clean.
