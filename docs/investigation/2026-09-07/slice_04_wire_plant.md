# Slice 4 — The wire plant

What is planted, why now, what is NOT planted, how it is tested, and what the owner decides. Sources:
the proposal's slice 4 paragraph (`00_proposed_voxel_foundation.md` §2), the SL6 register (§5) and the
owner's answers to it (ruling V6, rows R-1 … R-21), `06_storage_diff_lane.md` §5.2, `03_generator_sl10.md`
§3.3 and §7, and THE CODE: `crates/wire/src/{intershard,channels,session_flow,version}.rs`,
`crates/sim/src/io/mod.rs`, `crates/io-prod/src/outbox.rs`, `crates/core/src/look.rs`. Simplified
Technical English, examples from the game.

---

## 1. What a plant is, and why it comes before the consumers

The wire is postcard. Postcard is positional: an enum variant is written as its INDEX, never its name.
So the ORDER of the variants is the contract. A variant appended at the end costs nothing; a variant
inserted in the middle renumbers every variant after it and breaks every peer that already speaks the
wire. Today no peer speaks a voxel wire, so every voxel variant is free to place. Tomorrow, when slice 10
ships the first chunk rows, the order is fixed forever.

A plant writes every voxel shape into the reviewed wire files NOW, in one commit, at the end of each
enum, with no producer and no consumer. Then ten slices each fill a shape in, and none of them touches
the order. That is HR1 (one reviewed file for inter-shard bytes) and the incremental-freeze rule (an
arm's SHAPE freezes with its first consumer; its PLACE freezes with the plant).

```text
   today                            slice 4                          slices 9..14
   +-----------------+              +-----------------+              +-----------------+
   | Ghost        =0 |              | Ghost        =0 |              | Ghost        =0 |
   | Transfer     =1 |   append     | ...             |   fill in     | ...             |
   | ...             |  --------->  | ChildFacts  =39 |  --------->   | ChildFacts  =39 |
   | ChildFacts  =39 |              | ChildFelt   =40 |  (no renumber)| ChildFelt   =40 | <- slice 13
   +-----------------+              +-----------------+              +-----------------+
```

**Example.** Slice 9 wants to ship the first chunk rows. If the bulk variant were added THEN, the mesh
minor bumps, every client rebuilds, and the next slice bumps it again for the world handshake. With the
plant, the minors bump ONCE, today, and slice 10 only turns a producer on.

---

## 2. What the owner said YES to, and where each YES lives

The owner answered every SL6 row by name (ruling V6). Slice 4 may plant exactly the YES rows and R-20
behind its trial flag, and nothing else. A planted arm with no recorded YES is a defect the closed-set
test names by R-number.

```text
                 client                gateway              realm shard          parent shard
                   |                      |                      |                     |
   R-9  HelloWorld |--------------------->|                      |                     |
   R-5  WorldAction|--------------------->|-- SessionAction ---->|                     |
   R-3  ChunkRows  |<-- BulkMsg ----------|<-- BulkFor ----------|                     |
   R-8  TAG_SURFACE|<-- (in SelfLook bag)-|<---------------------|                     |
   R-20 ChildFelt  |                      |                      |<-- felt accel ------|
                   |                      |                      |   (on trial)        |
   R-11, R-13, R-15: TAG NUMBERS inside bags that already exist or are planted above; their
                     payloads land with slices 13, 10 and 14 behind the skip-unknown rule.
   R-7:  a code ask inside the generator (slice 5). Not wire.
```

| Row | Owner | What slice 4 plants | Where | Carrier class |
|---|---|---|---|---|
| R-3 | YES | `MsgClass::Bulk` (byte 9, reliable, paced); `BulkMsg::ChunkRows { realm, rows }` and `BulkMsg::ChunkManifest { realm, coarse_key, digests }` appended after `Blob`; `ShardToGateway::BulkFor { realm_fence, recipients, window, bytes }` appended; the `ChunkRow` bag schema id and its seven tag numbers | `sim/io/mod.rs`, `io-prod/outbox.rs`, `wire/channels.rs`, `wire/session_flow.rs`, `core` (a codec module beside `look.rs`) | reliable, paced; the gateway forwards bytes it never decodes |
| R-5 | YES | `ClientControlMsg::WorldAction { seq, action }` appended after `Bye`; `WorldAction::BlockEdit { .. }` and a RESERVED `Fire`; `GatewayToShard::SessionAction { session, seq, action }` appended | `wire/channels.rs`, `wire/session_flow.rs` | reliable control lane |
| R-8 | YES | `TAG_SURFACE = 4` beside `TAG_LOOK/LUMA/EXTENT` in the window-body schema; payload = the realm's own `FrameRef` + the DECLARED generator tag (`u64`); encode/decode helpers; the 1 200-byte budget test | `core/look.rs` | rides `BodyStmt::SelfLook`, retained, on change |
| R-9 | YES | `ClientControlMsg::HelloWorld { declared, measured }` appended | `wire/channels.rs` | the login handshake, right after `Hello` |
| R-11 | YES | one TAG NUMBER (`TAG_BODIES = 5`) in the window-body schema, payload owned by slice 13 | `core/look.rs` | skip-unknown |
| R-13 | YES | the `ChunkRow` tags: key, rung, apply_tick, cell records, box records, attachment list, body id, object rows — numbered now, payloads by slices 10/13/14 | the chunk-row codec | skip-unknown |
| R-15 | YES | nothing on the wire: the canopy fold is a pyramid-entry family inside a chunk row (slice 9) | — | — |
| R-20 | YES on trial | `InterShardFlow::ChildFelt(ChildFelt { child, parent_fence, at, felt: [i64; 3] })` appended after `ChildFacts`: the proper acceleration the parent integrated, in the CHILD's own frame, in `DRIVE_UNITS_PER_MPS2`, stated ON CHANGE; plus the trial flag as a `ShardProfile` capability bit, default OFF | `wire/intershard.rs`, `sim/capability.rs` | `ProducerLessReliable` + `FireAndForget` like `ChildFacts` |
| on disk | (A-9) | schema ids 32 `chunk_delta` and 33 `chunk_pyramid` beside body 30 and berth 31; the `last_owner_fence` row key | `sim/stub/built_store.rs` | — |
| core | — | `ChunkKey`: the address the diff lane and the interest index share (realm, chunk, rung) — a type in `vd-core` beside the grid module, no wire arm | `core/grid/` | — |

**Example, R-3.** A miner on a moon breaks one cell. The moon's shard writes the diff, builds one
`ChunkRow` for that chunk at rung 0, and sends `BulkFor { recipients: [the 3 sessions that hold this
chunk], bytes }` to the gateway. The gateway looks at the recipients only and forwards the bytes on each
session's bulk lane. Nobody else receives a byte. No pose crossed anywhere.

**Example, R-20.** A pilot burns at 3 g out of a moon's well. The star system integrates the hull's push
plus the moon's gravity and gets a proper acceleration. When that number moves by more than the
quantum, it states `ChildFelt` to the hull once. The hull presses its crew to the deck. If the trial's
measurement says the lane costs too much with 600 hulls, the flag stays off and the hull derives the
felt acceleration from the forces it already states, accepting the error near a planet.

---

## 3. What is NOT planted, by name

| Item the proposal's slice 4 named | Why not |
|---|---|
| `BLOCK_STORE_FLUSH_STEP = 19`, `BLOCK_EDIT_FORWARD_STEP = 20`, `BlockEdit(BlockEditForward)`, `BlockEditAck` | R-1 and R-2 are NO for v1 (postponed by the owner). A planted arm with no YES is the defect the closed-set test names. Step ids 19 and 20 stay FREE; the disjointness test pins 7..18 as today |
| `rung_floor: u8` on `WindowOpen` and on the relay's downward request | R-4 WITHDRAWN; it collides with SL3 (a rung floor pushed down is a clamp) |
| `BlockStoreTuning` with its numbers | every default is a measurement slice 9 owes (M-1 … M-19). A number typed before its bench is a magic number. Slice 4 plants the struct with the field NAMES and the bench each field owes; slice 9 fills the values |
| `Fire` as a real action | reserved variant only (P11 combat); no payload |
| R-6, R-10, R-12, R-14, R-16, R-17, R-18, R-19, R-21 | each is NO or held in reserve; nothing appears in any enum, not even a reserved name |

---

## 4. The gates

| Gate | What it proves |
|---|---|
| The closed set | every new arm is in `every_arm()` and in the golden discriminant pin, in the SAME commit; the tombstone set is unchanged; `ChildFelt` is the ONLY new inter-shard arm |
| No placement crosses | `no_living_inter_shard_payload_carries_a_realm_placement_or_a_centre` still passes: `ChildFelt` carries an acceleration in the child's own frame, the same kind of datum as `ChildDrive`, never a pose or a centre |
| Roundtrip | every new variant encodes and decodes postcard byte for byte; the class-to-byte table is exhaustive and `Bulk` is 9 |
| Skip-unknown | a reader that knows only `TAG_LOOK` still decodes a bag that carries `TAG_SURFACE` and `TAG_BODIES`; the chunk-row tag numbers are distinct and never reused |
| The budget | a `TAG_SURFACE` bag for the largest `FrameRef` fits the 1 200-byte datagram budget with the look and the luma beside it |
| The version floor | `PROTO_MINOR` 30 → 31 and the mesh minor bump ONCE; the floor tests still pass; an old client's `Hello` without `HelloWorld` is refused by name once slice 5 turns the check on, and accepted until then |
| Coverage | Tier-A at 100 %; the plant's encoders and classifiers are branchless or fully driven |

**Measurement.** The encoded size of one `ChunkRow` header, one `BulkFor` envelope, one `TAG_SURFACE`
bag, one `ChildFelt`, in bytes, from a release example (the `registry_cost` pattern). No runtime cost
exists until a producer does; the numbers are recorded so slice 10's byte budgets start from facts.

---

## 5. Laws

- **SL6.** Every arm has a recorded YES by R-number; the doc line on each arm names it.
- **HR1.** One reviewed file for inter-shard bytes; `ChildFelt` is the one addition, in it.
- **SL1 / SL2.** No placement, no pose, no centre in any new payload. `ChildFelt` is an acceleration in
  the child's own frame: "what the parent DID with my push", the downward twin of "what I am doing".
- **SL3.** `TAG_SURFACE` is the realm stating HOW IT LOOKS about ITSELF; the parent's message shrinks.
- **SL8.** A style update refuses nothing (the registry); a content update refuses only the chunks that
  name unknown kinds, and the diff lane names them (slice 10's consumer of this plant).
- **SL10.** `HelloWorld` carries the declared and the measured halves apart, so a chip difference never
  refuses a durable file.

---

## 6. What you decide

| # | Question | Recommendation |
|---|---|---|
| S4-1 | Plant everything in ONE commit with the two minors bumped once | Yes |
| S4-2 | A refusal BACK to the client for a `WorldAction`: `ServerControlMsg::ActionRefused { seq, reason }` (a new client-facing arm; SL6 asks before any arm). Without it a refused placement is silent: the block never appears and the player never learns why | **YES, small.** Success needs no message: the player sees the block arrive on the diff lane |
| S4-3 | The world refusal after `HelloWorld`: a new `ServerControlMsg::WorldRefused { ours, theirs }` naming both values, or reuse the version-floor refusal | **a new arm.** Reusing `VersionFloor` for a world mismatch repurposes a message, which the no-hack rule refuses |
| S4-4 | `ChildFelt`'s shape: on change, quantised in `DRIVE_UNITS_PER_MPS2`, reliable like `ChildFacts`; the trial flag a `ShardProfile` capability, default OFF until the slice that measures it (slice 13, the bodies) | Yes |
| S4-5 | Number the seven `ChunkRow` tags and `TAG_BODIES` now, payloads landing with their slices | Yes: a number is the only thing that cannot be added later without care |
| S4-6 | `BlockStoreTuning`: field names now, values with slice 9's benches | Yes |
| S4-7 | The realm store's `last_owner_fence` row and schema ids 32/33 in this slice, or with slice 9 | this slice: an id is a plant like any other |

---

## 7. How it is built

I write it myself: the arms, the tags, the classes, the tests, the size example. Then one Opus 5
refuter attacks the closed set, the classes, the tag numbering and every "not planted" line, and I
answer every finding before the slice is called done. Estimated size: about 600 lines of code and 500
lines of tests, most of it in the two reviewed wire files.

---

## 8. Results (2026-09-07, after the refuter)

**What landed** (uncommitted until the owner's word), by carrier:

```text
  client ⇄ gateway   ClientControlMsg::WorldAction (6), ::HelloWorld (7)
                     ServerControlMsg::ActionRefused (17), ::WorldRefused (18)
                     BulkMsg::ChunkRows (1), ::ChunkManifest (2) on MsgClass::Bulk (byte 9)
                     window body: TAG_SURFACE = 4 (SurfaceStmt), TAG_BODIES = 5 (number only)
  gateway ⇄ shard    GatewayToShard::SessionAction (7); ShardToGateway::BulkFor (16) with a
                     BulkAudience (sessions OR window holders, by the type)
  shard ⇄ shard      InterShardFlow::ChildFelt (44) behind the `felt_down` capability, default OFF
  core               ChunkCoord and CellAddr now encode; a rung byte above 15 is REFUSED at the
                     decoder; chunk_row: CHUNK_ROW_SCHEMA = 21 and the storage report's seven tags
  on disk            CHUNK_DELTA_SCHEMA = 32, CHUNK_PYRAMID_SCHEMA = 33, the owner-fence key (3),
                     BlockStoreTuning with six named fields and no value
  counters           gateway: world_actions_unrouted, world_hello_stated, bulk_for_unrouted;
                     shard: session_actions_unrouted, bulk_unrouted — each planted arm that reaches a
                     receiver with no path yet is COUNTED, never silently dropped, and nothing is
                     sent back or forward (the tests pin both)
  PROTO_MINOR        30 → 31, one ledger entry, the floor unmoved
```

**Deviations from §2, each from the refuter's report (`verdicts/slice_04_refutation.md`, 21 findings,
0 law breaks, 8 stands, all answered):**

- The chunk row's seven tags are the STORAGE REPORT's seven (§4.2): sparse cells, dense cells,
  pyramid entries, a revert list, attachment presence, codec (reserved), feature state. §2's R-13 row
  named a different list; the report is the source. A sub-grid's boxes and a landscape object are cell
  RECORDS, not tags; the body id belongs to slice 13, which may append a tag 8.
- The row carries its instant on the wire struct (`ChunkRow { key, at, bag }`), as the report shapes it.
- ONE edit verb. Placing the Empty kind IS the removal (ruling V4), so `BlockEdit` is flat
  `{ at, kind, variant, scale, orientation, site }` and no second removal spelling exists.
- `BulkFor` names ONE audience by the shape of `BulkAudience`: the sessions that hold the chunk, or a
  window's holders. A row for nobody or for two audiences is unrepresentable.
- `SessionAction` carries the session's fence beside `{ session, seq, action }`.
- `ChunkKey` in §2 is `ChunkCoord`, which slice 2 landed; the plant added its serde derives.
- `Rung` decodes through its own constructor: byte 16 or 200 is an error, never a rung whose shift
  would panic. A whole `CellAddr` refuses with it (pinned).
- `Face` pins its six wire bytes; the five window-body tags pin their distinctness.
- The old `BulkKind::ChunkSnapshot` / `ChunkDelta` placeholders are marked ★TOMBSTONED: never
  produced, superseded by the typed rows, kept so `Catalog` keeps its index.
- Every ★ note that described a forward speaks in the future tense ("slice 10's gateway WILL hand
  it …") and says what happens today (counted, dropped). The ledger says the sender gates are OWED
  with the first producers: the client tracks no negotiated minor yet, so a planted variant a peer
  predates lands on that peer's `undecodable` counter today.
- `felt_down` sits at the END of `CapRequest`; `serde` is a dev-dependency of the bins crate; the
  self-look prologue has ONE builder (`self_look_writer`) under both bag functions; the budget
  constant core states is asserted equal to wire's datagram budget.

**Measured** (`just voxel-measures`, release; both examples assert their gates):

| Shape, with nothing in it but its header | Bytes |
|---|---|
| `ChunkRows`, one empty row | 46 |
| `BulkFor`, three recipients, no bytes | 71 |
| self-look bag with luma and surface (a moon) | 58 |
| the WIDEST self-look bag (Obb outline, area frame, luma), pinned in core | 124 of the 1 200 budget |
| `ChildFelt` | 60 |
| `WorldAction`, one placement | 38 |
| `SessionAction`, the same placement forwarded | 65 |
| `HelloWorld`, `WorldRefused` | 21 each |

Tests: core 395, wire 103 + the closed set, sim 650, connection-plane 253, io-prod 174 — all green;
`cargo clippy --workspace --all-targets -D warnings` clean. `just coverage` (clean first): Tier-A PASS
at 100 %; io-prod 95.14 / 95.31 % against the floor of 94.

**Pre-existing red, NOT this slice's:** `cargo test --workspace` stops at `crates/bins/tests/flight_table.rs`,
which expects 51 governed rows and gets the 3 500 645-region forest (the boot-plant symptom of Step
23; no file that test reads changed here). Two `frame_conversion_e2e` process tests and one release-only
`should_panic` in the sim stub also fail on this branch before and after the plant. They are reported,
not hidden, and they are owed to the branch, not to slice 4.
