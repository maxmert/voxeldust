# Batch-2 closure sweep — Stage C audit, final status (2026-08-14)

Verified read-only against this tree: HEAD `27defe9` ("The five criticals" = batch 1) **plus the
uncommitted lane batch and standalone batch**. Every status below was re-derived from the code /
tests / ledgers the finding names — not from the fix reports. Nothing was executed (a test battery
runs concurrently); where a claim would need a run it is marked UNMEASURED.

**Index convention** (the one the task, the batch-1 sweep and DEFERRED.md itself use — confirmed by
`DEFERRED.md:3984` "Stage-C SL4 critical 24 + majors 25/26/27 + minor 28"): findings are grouped by
law (HR1→HR6, SL1→SL7, composite-law findings last), NOT by literal section position. Each row
carries the report anchor `§<line>` of `docs/audit/stage_c_full_law_audit_2026-08-14.md` so the
mapping is unambiguous.

Statuses: **CLOSED-B1** (fixed by commit 27defe9) · **CLOSED-B2** (fixed by the uncommitted lane /
standalone batches) · **DISSOLVED** (closed as a side effect — of what is stated) · **HELD-OWNER**
(blocked on a named owner decision) · **PARTIAL** (half-covered; residue stated).

## Status table

| # | § | Sev | Law | Finding (short) | Status | Evidence (file:line) | Residue |
|---|---|-----|-----|-----------------|--------|----------------------|---------|
| 0 | §95 | MAJ | HR1 | Step-5 lanes sender-blind; down-lane target = peer self-assertion | **CLOSED-B2** | up: `stub.rs:8657-8668` (`child_live_unattested` + head re-read), `:8716-8719` (`realm_observation_unattested`), `:8808-8812` (shape); down: `:8905-8913` (`child_scene_unauthored`), `:7620-7626` (`cascade_unauthored`); `from` now reaches every receiver (`stub.rs:2611/2625/2642/2655`, `:6438`); D-LANE-1 `DEFERRED.md:4095-4104` | NodeId itself self-asserted until P7 — stated in D-LANE-1 ("NOT a security boundary"), D-RLM-8 unchanged + scope note `DEFERRED.md:3325` (OD-6 owner question) |
| 1 | §650 | MIN | HR1 | Nested tripwires 3 of 8; DirectoryReply silent end-to-end | **CLOSED-B2** | five new tripwires `intershard_closed.rs:599-657` (`directory_op/directory_reply/transfer_control/transfer_control_ack/transfer_ack`), header `:545-546` "completed to ALL EIGHT … finding :650" | cosmetic: `intershard.rs:642-643` still says "NO `_` wildcard anywhere" over a match whose grouped arms wildcard 5 payloads (the compile-force now lives in the tripwires); `every_arm` still samples nested variants (a new one can no longer land silently) |
| 2 | §675 | MIN | HR1 | Comment: source "tells its parent where they are" (deleted lane) | **CLOSED-B2** | `bins/src/lib.rs:2020` rewritten ("counting itself occupied" — no pose claim); `bins/src/bin/shard.rs:200`; `stub.rs:645/:661` ("no pose leaves the shard — the per-occupant up-relay is DELETED"); `stub.rs:7816-7817` past tense | — |
| 3 | §13 | CRIT | HR3 | D-37 re-home target cannot host the subject's realm; stale roster mitigation | **CLOSED-B1** | realm-aware target from the flushed pose's frame realm via directory head (`saga_runtime.rs:1915/:2016/:2043`); **plus B2 closed the fix's own regression** (batch-review MAJOR: no same-entity guard) — `re_home_apply` now flips the held same-entity dot in place (`stub.rs:3904-3925`, pinned `:12802`) | — |
| 4 | §105 | MAJ | HR2 | Transient arrival is a FORK — adopt has no receiver-side conversion | **CLOSED-B2** | `adopt_transient_batch` runs `place_arriving_pose` per item, convert-or-refuse, `transient_arrivals_unplaceable` counted+logged (`stub.rs:5913-5960`, counter `:5941`); dispatch reads `to_realm` (`stub.rs:4703-4710`, "it used to be discarded by `..`") | — |
| 5 | §685 | MIN | HR3 | `to_parent` dead field documented as live in the frozen wire | **CLOSED-B2** | ★DEAD tombstones on all four arms: `intershard.rs:966` (FlushSource), `:1086-1088`, `:1106`, `:1126`; `version.rs:22-24`; `saga.rs:56-61`; the three pin tests renarrated as SHAPE pins (`stub.rs:18587`); flag-day removal ledgered D-WIRE-1 `DEFERRED.md:4121-4126` | field bytes stay on the wire until the flag-day MAJOR (postcard-positional — deliberate, ledgered) |
| 6 | §139 | MAJ | HR4 | D-38 🟩 rests on a fixture that never varies NodeKind | **PARTIAL** | B1 added a REAL two-kind fixture: `drive_swept_crossing_feature` on `StubShard` AND `Shard(planet)` (D-PLACE-1 `DEFERRED.md:3984`) | `assert_feature_anywhere` (`stub.rs:19059`) still drives `drive_inward_crossing_feature` (`:18909-18912`, `Rig::new()`) — both runs one NodeKind, profiles decorative; D-38 🟩 (`DEFERRED.md:299-312`) unamended; `capability.rs:13` KNOWN-LIMIT contradiction stands |
| 7 | §713 | MIN | HR4 | AoI/scene machinery hard-panics on a Ship realm, unledgered | **CLOSED-B2** | `region_level → Option` (`stub.rs:8573`); four lane guards skip+count `ship_child_regions_excluded` (`:7064/:7292/:8067/:8258`); pinned `stub.rs:20358`; D-SHIP-1 `DEFERRED.md:4127-4133` (P8 lineage arm + payload widening named) | Ship own-realm still boot-refused — deliberately, ledgered in D-SHIP-1 until P8 |
| 8 | §161 | MAJ | HR5 | C-6c 🟩 cites two tests that no longer exist | **CLOSED-B2** | row amended in place, names the stale clause + audit :161, re-pins to existing tests (`DEFERRED.md:3411`): `geometry.rs:1636` (`container_folds_the_deepest_member_from_the_root_identity`), `stub.rs:19732/:19764` (slice4b escape/dock-undock) | — |
| 9 | §185 | MAJ | HR5 | A permanently-red test inside `just gate`, unledgered | **HELD-OWNER** | test half closed B1: smoke rebased on THE world + renamed (`dual_cluster_crossing_smoke.rs:195`), D-WORLD-8 🟩 (`DEFERRED.md:3962`) + standing rule "a red gate is ledgered with a RED row"; false ledger claims struck (`DEFERRED.md:2981-2982`) | residue: the gate's **coverage step** stays red on 18 tool-artifact regions (per-crate-hash duplicate instantiations, no source line) "pending the owner's call … (ledgered as a task)" (`step5_sl7_lane_deletion.md:199-202`) — no such DEFERRED row exists, breaking D-WORLD-8's own rule. Blocked on: the owner's call on the coverage-artifact row |
| 10 | §195 | MAJ | HR5 | io-prod Tier-B floor never ratcheted (90 vs ≥94.4) | **CLOSED-B2** | `justfile:45` `tier_b_floor := "94"` with the derivation comment citing audit :195 and the recorded series (`justfile:40-44`) | D-40 %c process-tier merge still owed (pre-existing, ledgered) |
| 11 | §741 | MIN | HR5 | Branch half of coverage-fast unscoped to Tier-A | **CLOSED-B2** | `justfile:26` report step now carries `{{tier_a}}`; comment `justfile:19-22` cites audit :741 | — |
| 12 | §27 | CRIT | HR6 | Clusters name realms THE world does not contain | **CLOSED-B1** | `ClusterShape` = Single/Dual/Chain/Demand booking `world_roster` realms only (`bins/src/lib.rs:255-296`, roster `:274/:294`, "the only place allowed to name a realm"); `--forest` errors loudly; smokes rebased | — |
| 13 | §275 | MAJ | HR6 | rlm-proc-spawn ledgered cause is a copied guess | **DISSOLVED** (of 12 / D-WORLD-8) | coord derived from `world_roster` (`rlm_proc_spawn_smoke.rs:162-186`); the port-band guess struck + RETRACTED in `rehome_one_mechanism.md` (proc_launch row) | — |
| 14 | §304 | MAJ | HR6 | resurrect-guard row-drop invisible to the HR6 surface | **CLOSED-B2** | `devproto/src/state.rs:169/:174/:178` (`resurrect_rows_dropped`/`foreign_space_rows`/`echo_rows_dropped`), builder `client/src/net.rs:618-622`, JSON pin `state.rs:249-251` | note: `WaitField` (`devproto/src/predicate.rs:14-32`) still has no row-drop arm — visible via `vdctl state`, not `wait-until` |
| 15 | §328 | MAJ | HR6 | emit-seed-fixtures ships the walk forest as "single-sourced" | **DISSOLVED** (of the one-world emitter deletion, B1) | `write_seed_regions`/`emit-seed-fixtures` deleted (`vd-devcluster.rs:97-99` tombstone); ONE emitter `emit-world-scene` off `world_roster`+`boot_regions_and_movers` | cosmetic: `geometry.rs:930`, `realm_scene.rs:221/:348` still name `realm_regions_for` as the detector/client source |
| 16 | §790 | MIN | HR6 | visual-run.sh + write_visual_regions second world config | **PARTIAL** | emitters deleted, visual-run.sh rewritten (B1) | residue: `UniverseConfig::visual_scale()` survives (`physics/worldgen.rs:1294`; only test callers: `gateway.rs:8924` and `frame_conversion_e2e.rs:1263`, both cfg(test)) with no ledger row; `scripts/demand-visual-run.sh:66-68` still narrates deleted `VD_UNIVERSE_SCALE`/`resolve_universe_scale` |
| 17 | §338 | MAJ | SL1 | Down shape lane tells every nested realm its own placement | **CLOSED-B2** | THE SL1 SELF-PLACEMENT FILTER at the reflect merge: `stub.rs:8265-8277` (`.filter(\|s\| s.realm != config.realm)`, comment names finding 17); the gate INVERTED: `frame_conversion_e2e.rs:2488-2495` asserts `!drawn.contains_key(&PLANET)` ("used to read `drawn[&PLANET] == −5` and asserted the breach"); sibling-box two-subtraction asserts strengthened `:2496-2506` | — |
| 18 | §364 | MAJ | SL1 | Own position planted in RealmRegion.center; boot reads own value | **PARTIAL** | fence half closed B1 (`ChildReach::Excursion(sma·(1+ecc))`, boot-stated; size-only fence gone) | the field + own-row plant + boot read stand; ledgered 🟥 D-PLACE-2 (`DEFERRED.md:4019-4028`, "every shard is still planted with its own row's centre"); nit: the row names deleted `VD_REALM_BOUNDARIES` as a live reader (`:4025`) |
| 19 | §384 | MAJ | SL2 | Transient dest ingress: no conversion, no guard | **CLOSED-B2** | same mechanism as #4: convert-or-refuse at adopt (`stub.rs:5934-5957`); only PLACED poses enter `OwnedTransients` (`:5958-5966`) — AoI/containment can no longer read a child-frame number | — |
| 20 | §812 | MIN | SL2 | Frozen wire + ledger assert a rebase no code performs | **CLOSED-B2** | contract comments rewritten to ★DEAD tombstone (`intershard.rs:1086-1093`, `:1106-1110`); ledger corrected: `DEFERRED.md:3047` (stale-at-HEAD note on the old claim), `:3583` ("★DEAD wire field (D-WIRE-1)") | — |
| 21 | §394 | MAJ | SL3 | SL3 shape debt "ledgered in DEFERRED.md" — no entry existed | **CLOSED-B2** | the entry now exists: D-LANE-4 🟥 (`DEFERRED.md:4106-4110`); the pointer fixed: `step5_sl7_lane_deletion.md:254` "ledgered as D-LANE-4 in DEFERRED.md (held on an owner world-design ruling)" | the debt itself is D-LANE-4 (see #23) — correctly a red row, not a claim |
| 22 | §414 | MAJ | SL3 | Owner Q2 (shape ∥ position never in one message) unimplemented | **HELD-OWNER** | `RealmShape` still packs `center` (`channels.rs:309`) + `shape` (`:310`); now ledgered 🟥 D-PLACE-3 (`DEFERRED.md:4030-4060`) with the actual blocker recorded: **the client-origin statement** — how the client learns which realm is the scene's origin, options (a)/(b)/(c), "None is free; the owner decides" (`:4041-4048`) | DE-ESCALATED note (`:4053-4058`): the SL1 leg is independently closed by #17's filter — the row is now shrink-and-simplify, not an SL1 emergency |
| 23 | §424 | MAJ | SL3 | Nothing in the draw path requires a realm to be running | **HELD-OWNER** | breach still live: draw fold = AoI band ∧ client route only (`stub.rs:8109-8111`), no liveness term; now ledgered 🟥 D-LANE-4 (`DEFERRED.md:4106-4110`) — the structural cure (a realm ships its OWN outline; not-running ⇒ nothing to draw) is **held on the owner world-design ruling**: is an empty unvisited system invisible, or does distance belong to an always-on starfield layer? Adjacent D-LANE-6 (`:4116-4119`) ledgers the `--realm-boxes` no-evidence draw | — |
| 24 | §37 | CRIT | SL4 | Kepler solve inside the placement lookup, on a handed clock | **CLOSED-B1** | placement arc: celestial/worldgen moved to vd-physics; vd-sim/vd-node carry it DEV-ONLY; ONE placement writer (`author_placements`), motion opaque `MotionFn`; executable ban `crate_isolation.rs:138-150` (`sl4_the_crossing_path_cannot_name_a_motion`, OBSERVED-FAILING record); D-PLACE-1 🟩 `DEFERRED.md:3984` | — |
| 25 | §454 | MAJ | SL4 | Four rival has-orbit tests | **DISSOLVED** (of 24) | `frame_context` deleted; movers-only feed filter deleted (D-FO-7 `DEFERRED.md:3837`); one motion test `placement_row`, both arms one row type | — |
| 26 | §482 | MAJ | SL4 | SL4 structural enforcement does not exist | **DISSOLVED** (of 24) | `crate_isolation.rs` SL4 gate (universal over vd-*); **and B2 deleted the batch-review carve-out**: vd-connection-plane's vd-physics edge moved to `[dev-dependencies]` (`connection-plane/Cargo.toml:22-27`), gateway now RECEIVES lowered `WorldRealms`; OBSERVED-FAILING record for this crate too (`crate_isolation.rs:74-82`) | — |
| 27 | §502 | MAJ | SL4 | Stored centre means two things; fence size-only for movers | **PARTIAL** | fence half closed B1 (apoapsis `ChildReach`, world numbers re-solved, tripwires) | `region_center_of` still branches Orbital⇒ZERO vs StaticOffset (`physics/worldgen.rs:201-206`) — ledgered 🟥 D-PLACE-2; nits: D-45(d) (`DEFERRED.md:3418`) still ledgers the now-live geometric check as P4/P5-deferred; D-PLACE-2 names deleted `VD_REALM_BOUNDARIES` |
| 28 | §836 | MIN | SL4 | Orbital variant marked dead while production produces it | **DISSOLVED** (of 24) | `#[allow(dead_code)]` gone; truth-stating doc (`physics/worldgen.rs:174-179` names audit finding 28); measured pin `the_world_produces_an_orbital_planet_in_every_system` | — |
| 29 | §846 | MIN | SL4 | "Band is speed-sized" comment over a v_rel=0 band | **DISSOLVED** (of 24/B1) | replacement comment states the truth ("STATIC widths … OWED, D-WORLD-4b"); D-WORLD-4b 🟧 `DEFERRED.md:3923` | `DEFERRED.md:3418` C-3 sentence still argues from a "velocity-safe ContainmentBand" (stale, separate line) |
| 30 | §47 | CRIT | SL5 | Live world scale knob in the production shard boot | **CLOSED-B1** | no `VD_VISUAL_ORBIT_SLOWDOWN`/`VD_TEST_ORBIT_SLOWDOWN` reader anywhere (grep: only the deletion note `bins/src/lib.rs:2242`); no `central_mass /=` divisor | — |
| 31 | §73 | CRIT | SL5 | Walk gate stands on four realms THE world lacks | **CLOSED-B1** | `node_per_realm_walk.rs` rebased: every name via `world_roster` (`:6/:17/:24`), full-speed `rendezvous_into_planet` (`:44`), flight-law margins asserted at derivation | — |
| 32 | §512 | MAJ | SL5 | Second, seedless world generator under every in-process scenario | **PARTIAL** | bin half closed B1; **B2 closed the gateway half**: `SeedInjectorConfig::inert(world)` REPLACED `Default` ("this crate cannot build one at all — no edge to the generator", `gateway.rs:162-184`) | residue: `generate_walk_forest` + `realm_regions_for` (= `walk_scale`) still build the vd-tests scenario tier (`physics/worldgen.rs:807-824`; `tests/src/lib.rs:893`); the false claim "the EXACT geometry the production `shard.rs` boot computes" survives at `tests/src/lib.rs:883` and `:901`; D-WORLD-5 🟧 (`DEFERRED.md:3932`) flip condition unmet |
| 33 | §546 | MAJ | SL5 | Shard binary can be told to replace THE world (VD_REALM_BOUNDARIES) | **DISSOLVED** (of 12/30, B1) | override family deleted (zero hits for `resolve_realm_boundaries`/`override_regions_for_boundaries`); shard boots `boot_regions_and_movers` unconditionally; env assertions pin no BOUNDARIES key; D-WORLD-4 🟩 | — |
| 34 | §556 | MAJ | SL5 | Two of three gated crossing tests at 300× | **DISSOLVED** (of 30) | both round-trips fly full-speed `rendezvous_into_planet` (`rlm_demand_login.rs:782/:1042`; `flight.rs`); the slowdown recipe retired; **B2 purged the residual stale comments** (grep `OrbitSlowdown`: one historical mention `:509` only) | — |
| 35 | §856 | MIN | SL5 | Crossing playground: walk scene + cannot come up | **PARTIAL** | both code halves closed B1 (`--demand` playground, streamed scene, emitters deleted, `--forest` errors loudly); D-WORLD-2/D-WORLD-4 | residue: `DEFERRED.md:2999-3003` still names "the live `--forest` window" as the interim VISUAL proof and owes the GPU walk re-base onto that now-deleted cluster shape — a ledger paragraph pointing at machinery that no longer exists |
| 36 | §866 | MIN | SL5 | canonical()/seed_derived() presets public, callerless | **CLOSED-B2** | both DELETED — sole remaining trace is the tombstone comment `physics/worldgen.rs:926`; their Tier-A coverage load removed | (`walk_scale`/`visual_scale`/`walk_demand` are #16/#32's residue, not this finding's) |
| 37 | §616 | MAJ | SL7 | A realm demands its own PARENT stay alive | **CLOSED-B2** | `push_demand` structural gate: own realm or DIRECT child only, else refused+counted `demand_refused_not_own_or_child` (`stub.rs:8596-8628`, counter `:8609`); redrive doc states the outward refusal + the lawful liveness chain (`stub.rs:5622-5636`); tests: inward keep-alive still asserted (`:19493`), outward = ZERO demands + counted refusal (`:19547`), `rlm.rs:1157` parent-stays-alive-without-upward-demand | ingest stays source-blind by design — D-RLM-8 unchanged, scope note (`DEFERRED.md:3325`) records the P7 path + OD-6 |
| 38 | §588 | MAJ | SL6 | `RealmShapeObservation` — wire arm with no approval trail | **HELD-OWNER** | still no approval citation on minor 11 (`version.rs:102-107`) while minors 10/13/14 carry one; no ask recorded anywhere in docs (DEFERRED mentions are mechanism-only). Materially narrowed by B2: the lifted-set flatten is DELETED — the payload is now the sender's OWN roster only, no longer growing with the live subtree (D-LANE-3, `stub.rs:7355-7380`), and the sender is attested (#0) | blocked on: the owner's retroactive SL6 approval (or refusal) of the arm |
| 39 | §606 | MAJ | SL6 | Up-observation relay unconditional and recursive | **CLOSED-B2** | both up-re-relays DELETED ("THERE IS DELIBERATELY NO RE-RELAY HERE ANY MORE — finding 39b", `stub.rs:7332-7340`); interior fan bounded per-observer by `AoiMembership.was_in` (`stub.rs:7296-7305`); shape ship = own roster, held sets stay home (`:7346-7360`); wire contract rewritten to the two-level law (`intershard.rs:375-377`, `:389-391`); depth-3 = owner decision with a stated revisit trigger, D-LANE-3 (`DEFERRED.md:4101-4104`) | — |
| 40 | §876 | MIN | SL6 | `ShardRoster` appended with no approval trail | **HELD-OWNER** | minor 9 (`version.rs:94-97`) still carries no approval marker; `grep ShardRoster docs/design/DEFERRED.md` — mechanism rows only, no ask/approval | blocked on: the owner's retroactive SL6 approval of the arm |
| 41 | §896 | MIN | SL7 | Bit ships every tick, not the stated cadence; no adopt-edge | **CLOSED-B2** | emit gated `bit_due \| adopt_edge` (`stub.rs:7918`, `:8030`); `WasOccupied` edge (`stub.rs:491`, re-armed at Empty `:8001-8003`) — "no hook in any adopt path"; contract text matches code (`intershard.rs:362`); the RATE is now the assertion, ±1 beat (`frame_conversion_e2e.rs:2219-2230`, cites finding 41); D-LANE-5 🟩 (`DEFERRED.md:4111-4115`) | priced residual (ledgered): consumers read up to one cadence of staleness |
| 42 | §886 | MIN | SL6 | Dup of 21 — false "ledgered in DEFERRED.md" pointer | **CLOSED-B2** | same cure as #21: D-LANE-4 exists; `step5_sl7_lane_deletion.md:254` names it | — |
| 43 | §920 | MIN | SL7 | Bit's return address = unauthenticated sender; re-targets both down-lanes | **CLOSED-B2** | `home` stored only when the sender matches the directory head (`stub.rs:8657-8676`) — a forged high-`(fence,at)` bit from an unattested node is refused + arms a head re-read; down-lanes believed only from `ParentRealmNode` (#0); route-vs-admission split + zombie-window closure stated in D-LANE-1 (`DEFERRED.md:4095-4099`) | same ledgered NodeId-until-P7 residual as #0 |
| 44 | §229 | MAJ | HR5/★ | Band-exit capstone can no longer fail for its named property | **CLOSED-B2** | rewritten as `p2_dod_band_exit_tears_down_the_retained_dot_after_the_leaver_vanished_at_hold_closure` (`p2_transfer_gates.rs:625-740`): hold-closed-at-settle assert, zero-before-walk-out, mechanism counters `ghost_band_exits==1`/`ghost_despawns==1`, anti-vacuity precondition kept, `max_absent_run` kept as an honestly-labelled continuity floor (`:721`); ledger claim amended (`DEFERRED.md:579`, cites audit :229) | — |
| 45 | §374 | MAJ | SL1/SL2/HR2 | Adopt stores wire pose verbatim; p3 gate structurally blind | **CLOSED-B2** | conversion as #4/#19; the gate is no longer blind: `p3_transient.rs:74-80` asserts `dest_pose.frame == dest_stub_config().frame`; NEW refusal test drives a sibling-frame batch and asserts the refusal (`:206-231`); the stale "one rebind, no per-path fork" ledger claim corrected (`DEFERRED.md:3583`) | — |
| 46 | §255 | MAJ | HR5/wm | Only process-tier crossing gate red across the arc, unledgered | **DISSOLVED** (of 12 / D-WORLD-8) | smoke rebased+green-able, still ungated in `just gate`; D-WORLD-8 🟩 + the standing RED-row rule; opposite ledger claims struck (`DEFERRED.md:2981-2982`) | cosmetic: `rehome_one_mechanism.md:162-164` table row (dual) and `:1339` "sole red" narration still print the old red un-annotated (the proc_launch row DID get the RETRACTED strikethrough); post-fix greenness is ledger-claimed, UNMEASURED here (read-only) |
| 47 | §219 | MAJ | HR5/§4f | Round-trip gates only at 300×; re-aim owed with Step 5 | **DISSOLVED** (of 30) | the knob is gone from code; the whole `rlm_demand_login` binary — both return legs — runs full speed inside `just gate`; live tick-aware aim in `flight.rs:62-68` | — |
| 48 | §626 | MIN | ledger | DEFERRED still records slice-F/minor-14 landings as OWED | **CLOSED-B2** | the 🟥 bullet rewritten "🟩 SUPERSEDED / LANDED (status corrected 2026-08-14 — Stage-C audit :626 …)" with all three sub-claims corrected and "no 'S0' arc exists or is owed" (`DEFERRED.md:3414`) | — |
| 49 | §940 | MIN | §4u | Production docs state the deleted emit rule / ghost writer | **CLOSED-B2** | `authority.rs:18`, `stub.rs:310`, `stub.rs:6932` now state the real two-term rule (`simulates() \| (is_retained_ghost & Source-hold-open)`); `refresh_source_ghost` survives only in past-tense tombstones (`stub.rs:518`, `:3976`); `is_fed_ghost`/`SourceGhostMirror`: zero live hits; gate narration fixed — eviction attributed to hold closure (`frame_conversion_e2e.rs:3115-3116`) | — |
| 50 | §774 | MIN | HR5/★ | P1 parity gate silently swallows two lanes | **CLOSED-B2** | RealmSnapshot lane: mechanism ESTABLISHED in-comment (cites audit :774), every datagram must decode, `realm_frames`/`realm_rows` asserted `> 0` (`process_parity.rs:100-115`, `:369-381`); Event lane: `EntityRemoved` applied + counted, zero-evictions asserted, unknown events panic (`:161-171`); the false "tracks no bystander figures" justification removed | — |

## Non-CLOSED findings — exact residue

### [6] PARTIAL — HR4 (§139)
Batch 1 delivered the missing substance elsewhere: `drive_swept_crossing_feature` runs ONE body on
`NodeKind::StubShard` AND `NodeKind::Shard(planet)` and is truthfully ledgered (D-PLACE-1,
`DEFERRED.md:3984`). The finding's actual target is untouched: `assert_feature_anywhere`
(`crates/sim/src/stub.rs:19059`) still calls `drive_inward_crossing_feature`
(`stub.rs:18909-18912`), which builds `Rig::new()` — both runs the SAME NodeKind, the profile
constructions still decorative `.voxel()` asserts. The D-38 🟩 row (`docs/design/DEFERRED.md:299-312`)
still presents that fixture as running "on a Spherical Shell + a Cartesian Aabb profile", and
`crates/sim/src/capability.rs:13` still carries the KNOWN-LIMIT text that contradicts the 🟩.
Residue = ledger honesty of D-38 (amend the row to cite the swept fixture, or wire the profile's
NodeKind through `drive_inward_crossing_feature`).

### [9] HELD-OWNER — HR5 (§185)
The named red test is cured (B1) and properly ledgered (D-WORLD-8 🟩, `DEFERRED.md:3962`). The
surviving half: `just gate` cannot pass while `coverage-fast` is red on the 18 residual regions that
"own NO source line in the merged lcov (per-crate-hash duplicate instantiations — a tool artifact
class)"; `docs/design/step5_sl7_lane_deletion.md:199-202` says "the gate stays honestly red on them
pending the owner's call … (ledgered as a task)" — and no DEFERRED row exists (grep for the 18
regions / coverage artifact: zero), which breaks the RED-row rule D-WORLD-8 itself established.
Blocked on: the owner deciding how the branch/region gate treats tool-artifact regions (exemption
row vs tooling fix), and that decision being written as the missing RED row.

### [16] PARTIAL — HR6 (§790)
Closed by B1: `write_visual_regions` / `emit-visual-fixtures` deleted; `scripts/visual-run.sh`
rewritten. Residue: (a) the preset itself survives — `UniverseConfig::visual_scale()`
(`crates/physics/src/worldgen.rs:1294`) has zero non-test callers (`gateway.rs:8924` and
`frame_conversion_e2e.rs:1263` are both inside `#[cfg(test)]`) and appears in no DEFERRED row;
(b) `scripts/demand-visual-run.sh:66-68` still narrates the deleted `VD_UNIVERSE_SCALE=visual` /
`resolve_universe_scale` machinery.

### [18] PARTIAL — SL1 (§364)
The load-bearing reader is cured (B1): the boot fence consumes kind-blind `ChildReach`, movers judged
at apoapsis. The stored datum stands: `RealmRegion.center` still exists, every shard is still planted
with its own row's centre (and its ancestors'), and the boot still reads it for static rows. This is
now honestly ledgered 🟥 as D-PLACE-2 (`docs/design/DEFERRED.md:4019-4028`: "the FIELD survives …
every shard is still planted with its own row's centre"), owed with the motion-roster boot rework.
Nit inside the fresh row: `:4025` lists `VD_REALM_BOUNDARIES` — a deleted symbol — among the
readers at HEAD.

### [22] HELD-OWNER — SL3 (§414)
`RealmShape` still ships `center` + `shape` in one message (`crates/wire/src/channels.rs:309-310`),
so owner decision Q2 is still undischarged — but it is now fully ledgered as D-PLACE-3 🟥
(`DEFERRED.md:4030-4060`) with the genuine blocker stated: **the client-origin statement** — the
row lane carries direct children only, so removing `center` requires an answer to "how does the
client learn which realm is the origin of the scene it was handed" (options (a) repurpose `pin`,
(b) an origin marker, (c) keep a zero centre for the own outline only); "None is free; the owner
decides." The SL1 urgency is gone (the #17 filter closed the self-placement leg — DE-ESCALATED note
`:4053-4058`); the flag-day cost (PROTO_MINOR_FLOOR moves) is recorded.

### [23] HELD-OWNER — SL3 (§424)
The draw fold is still liveness-free: `crates/sim/src/stub.rs:8109-8111` pushes `child_shape` on
AoI-band ∧ client-route alone. The structural cure (a realm authors its own outline; a not-running
realm ships nothing, so nothing can draw it — no liveness term needed) is designed and ledgered 🟥
as D-LANE-4 (`DEFERRED.md:4106-4110`), explicitly "held on an owner world-design ruling": whether an
empty, unvisited star system is invisible until spun up, or whether distant unoccupied bodies belong
to an always-on starfield/chart layer that is not the realm scene. D-LANE-6 (`:4116-4119`) ledgers
the same law for the `--realm-boxes` boot file (every pixel gate currently stands on it). Blocked
on: that owner statement about what the client may draw without running evidence.

### [27] PARTIAL — SL4 (§502)
The fence half is cured (B1) and the world's numbers moved so the check is live. The two-meanings
field survives: `region_center_of` still branches `Orbital ⇒ ZERO` vs `StaticOffset ⇒ offset`
(`crates/physics/src/worldgen.rs:201-206`) — ledgered 🟥 in D-PLACE-2. Stale nits: D-45(d)
(`DEFERRED.md:3418`) still ledgers the geometric child⊆parent check as "P4/P5" although it runs at
every boot; D-PLACE-2 names the deleted `VD_REALM_BOUNDARIES`.

### [32] PARTIAL — SL5 (§512)
B1 removed the bin half; B2 removed the gateway half — `SeedInjectorConfig::Default` is replaced by
`inert(world)` and the crate's vd-physics edge is dev-only (`connection-plane/Cargo.toml:22-27`,
`gateway.rs:162-184`, OBSERVED-FAILING record in `crate_isolation.rs:74-82`). What stands: the
seedless walk generator still builds the in-process scenario tier — `realm_regions_for` =
`generate_walk_forest(walk_scale())` (`physics/worldgen.rs:807-824`), planted by
`tests/src/lib.rs:893` (`plant_seed_neighbourhood`) and the `_held` twin — and the false claims
"the EXACT geometry the production `shard.rs` boot computes" survive verbatim at
`tests/src/lib.rs:883` and `:901` (production boots `UniverseConfig::world` /
`generate_system_forest`). D-WORLD-5 stays honestly 🟧 (`DEFERRED.md:3932`); the new
"FIXTURE path — NOT a generated world" banner (`physics/worldgen.rs:853-855`) is a partial
honesty cure that does not reach the tests/src/lib.rs claims.

### [35] PARTIAL — SL5 (§856)
Both code halves closed (B1). Ledger residue: `docs/design/DEFERRED.md:2999-3003` still calls "the
live `--forest` window" the interim VISUAL proof and owes the GPU pixel-capture walk re-base "onto
the `--forest` node-per-realm cluster" — a cluster shape that now errors loudly and no longer
exists. The paragraph needs re-pointing at the `--demand` playground or striking.

### [38] HELD-OWNER — SL6 (§588)
The arm's mechanics were re-based by the lane batch — the lifted-set flatten is deleted (the payload
is the sender's own roster only, two levels per lane, D-LANE-3) and the sender is attested — but the
SL6 defect stands as filed: minor 11 (`crates/wire/src/version.rs:102-107`) still carries no
owner-approval citation while its siblings do, and no ask/approval is recorded in any design doc or
DEFERRED row. Blocked on: the owner's retroactive SL6 ruling on `InterShardFlow::RealmShapeObservation`
(approve, or order the D-LANE-4 authorship model to replace it).

### [40] HELD-OWNER — SL6 (§876)
Unchanged by both batches: `ShardRoster` (minor 9, `version.rs:94-97`) still has no approval marker
and no design/DEFERRED record of an ask. Blocked on: the owner's retroactive SL6 ruling on the arm.

## Tally

| Status | Count | Findings |
|---|---|---|
| CLOSED-B1 | 5 | 3, 12, 24, 30, 31 |
| DISSOLVED | 10 | 13, 15, 25, 26, 28, 29, 33, 34, 46, 47 |
| CLOSED-B2 | 25 | 0, 1, 2, 4, 5, 7, 8, 10, 11, 14, 17, 19, 20, 21, 36, 37, 39, 41, 42, 43, 44, 45, 48, 49, 50 |
| HELD-OWNER | 5 | 9 (coverage-artifact row), 22 (client-origin statement), 23 (client-origin / world-design ruling), 38 (wire-arm approval), 40 (wire-arm approval) |
| PARTIAL | 6 | 6, 16, 18, 27, 32, 35 |

**40 of 51 closed (5 B1 + 10 dissolved + 25 B2), 5 held on named owner decisions, 6 partial. Zero
findings fully OPEN with no fix and no ledger row.**

Claim-vs-code check on the two fix batches: every mechanism the lane batch and the standalone batch
claimed was found in the tree and does what was claimed (no deviations). None of the six PARTIALs
was claimed closed by either batch — their residues are pre-existing halves the batches did not
target, and all six are stated above with file:line. Two adjacent batch-review defects were also
fixed in the uncommitted work: the re-home same-entity collision (`stub.rs:3904-3925`) and the
vd-connection-plane SL4 carve-out (dev-dep move + gate re-coverage).

Weakening check on CLOSED findings: no named assert was weakened. #17's gate was inverted from
pinning the breach to asserting its absence plus stronger sibling-subtraction equalities; #44 kept
`max_absent_run == 0` as a labelled continuity floor and added the falsifiable mechanism counters;
#37 replaced the breach-pinning keep-alive test with an inward-lawful assert plus an outward
zero-demand + counted-refusal assert; #41 turned the printed rate into a ±1-beat assertion; #50
turned two tolerated lanes into decoded, asserted measurements.

UNMEASURED (read-only sweep, no cargo): that the rewritten gates run green. Every claim above is a
code/ledger reading, not a test run.
