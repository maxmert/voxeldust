export const meta = {
  name: 'r3-holistic-audit',
  description: 'Holistic whole-codebase /goal audit after R-3 lands: composition, scale, end-goal fit',
  phases: [
    { title: 'Audit', detail: '3 holistic lenses' },
    { title: 'Synthesize', detail: 'adjudicate + DONE_NO_CRITICAL verdict' },
  ],
}

const CONTEXT = `
Voxeldust is a transfer-first "Star Citizen x Minecraft" voxel MMO (greenfield rebuild). Just LANDED (commit
828f0ab): R-3' of the redelivering (at-least-once) MeshTransport — the ROOT CURE for the producer-less-flow
silent-loss class (DEFERRED.md D-6 #1). R-3' added, in crates/io-prod/src/mesh.rs + lib.rs:
 - a node-wide RecvLedger (Arc<Mutex<BTreeMap<(NodeId,MsgClass),RecvState>>>) locked on EVERY reliable inbound
   frame (classify_reliable verdict ladder) — the receiver dedup/contiguity state that SURVIVES connection teardown;
 - reverse cumulative acks on the SAME connection (serve_connection opens the ACK uni stream; peer_writer runs an
   accept_uni ack-reader → on_ack retires the sender's retry buffer);
 - MeshControl::drop_connections() blip lever; common_env exports a per-launch VD_PROCESS_INCARNATION.
The UNRELIABLE 20Hz hot path (snapshots/input/ghost-deltas) is a BARE DatagramFrame — it NEVER touches the ledger.
Reliable traffic is the control plane: Saga, Membership, Control, GhostReliable (and future Signal/BlockEdit/Coupling).

This is a HOLISTIC /goal audit — NOT a re-review of R-3' transport correctness (a focused adversarial review
wf_a909be64 already returned SHIP and verified the receiver ladder, inbox-drop rollback, and same-connection ack
topology sound). Your job is the BROADER lens: how does R-3' COMPOSE with the rest of the system, does it SCALE to
the end-goal, and does it advance (not endanger) the planned game. READ the real code + docs to ground every claim:
 - docs/design/PLAN.md (the 6 hard rules HR1-HR6, the roadmap), docs/design/DEFERRED.md (D-6, the R-4'/R-5'/R-6 owed items),
 - crates/io-prod/src/{mesh.rs,lib.rs}, crates/sim/src/saga.rs (the saga re-drive layer), crates/sim/src/io/mod.rs
   (the frozen Transport/Inbound/MsgClass seam), crates/connection-plane (gateway dispatch), crates/wire (InterShardFlow).

END-GOAL to keep in view: spherical voxel planets; player-built ships/stations/cities walked-inside while flying;
SIGNAL-HEAVY cross-shard functional-block gameplay (possibly cross-shard); multi-sharded meshes (ONE location spanning
shards); client simultaneous multi-mesh rendering; Newtonian physics; players physically COLLIDE cross-boundary (NO
client prediction); PvP; HUNDREDS of users in one location. Robust, scalable, DRY, elegant, cloud/k8s-ready, AAA MMO.
`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'verdict', 'findings'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['CLEAN', 'FINDINGS'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'area', 'problem', 'recommendation'],
        properties: {
          severity: { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] },
          area: { type: 'string' },
          problem: { type: 'string' },
          recommendation: { type: 'string' },
        },
      },
    },
  },
}

const LENSES = [
  {
    key: 'composition',
    prompt: `${CONTEXT}\n\nLENS: COMPOSITION & INTEGRATION. Does the R-3' transport-level redelivery compose CORRECTLY with the higher layers, or is there a double-cure / gap / conflict? Specifically: (1) the LAYERING claim (transport owns connection-loss + peer-restart redelivery; saga/store owns sender-CRASH re-drive) — is the boundary clean, or can the two redelivery mechanisms fight (e.g. the saga re-drives AND the transport replays the same flow → a dup the receiver must dedup — does the receiver's per-(peer,class) contiguity dedup actually cover saga-level dups, or only transport-level ones)? (2) The frozen Transport/Inbound/MsgClass seam: R-3' put acks BELOW the seam (never a MsgClass, never an Inbound) — verify sim/node truly never see acks and the seam is intact (HR1). (3) The gateway dispatch order-independence-across-classes contract (R-2b ledgered it): does any reliable consumer depend on cross-class ordering that the per-class lane split now breaks? (4) The retry buffer GROWS un-drained until R-4' for a stuck lane — before R-4' lands, is there any flow that could accumulate unbounded retry (OOM risk) in normal operation? (5) Does R-3' correctly RETIRE or leave dangling any of the per-flow band-aids (AwaitAdopt re-solicit, ghost staleness reaper, crossing re-emit) it was meant to eventually replace?`,
  },
  {
    key: 'scale-load',
    prompt: `${CONTEXT}\n\nLENS: SCALABILITY, HIGH-LOAD & CLOUD/AAA. Will R-3' hold up at HUNDREDS of users in one location across multiple shards? Probe: (1) the node-wide RecvLedger Mutex is locked on EVERY reliable inbound frame AND (nested) the inbox lock inside it — at high reliable-message rate (signal-heavy cross-shard functional blocks, PvP damage/death events, saga bursts), is this a contention hotspot / head-of-line stall across ALL reliable classes node-wide? Is a single node-wide Mutex the right granularity, or should it shard by peer? (2) Task growth: ack_egress + ack_reader + serve_data_stream tasks — count them per node as peers/streams scale; any unbounded task spawn? (3) The ack reverse stream is one-per-connection with per-frame notify (coalesced) — reverse-lane bandwidth at high reliable throughput? (4) The 20Hz UNRELIABLE hot path must stay ZERO-cost — confirm R-3' added nothing to it (no ledger, no ack) so PvP position sync is untouched at scale. (5) Cloud/k8s: does the per-launch wall-clock VD_PROCESS_INCARNATION + the dial-on-demand address-book model behave under pod restarts / rescheduling / horizontal gateway scaling? (6) Load-test coverage (feedback: land load tests with perf-bearing subsystems) — is there a reliable-transport load/volume test, or is that a gap?`,
  },
  {
    key: 'trajectory-dry',
    prompt: `${CONTEXT}\n\nLENS: END-GOAL TRAJECTORY, DRY & ELEGANCE. Does R-3' advance the planned game without painting into a corner? (1) The SIGNAL system (cross-shard functional blocks, possibly cross-shard, HMAC-granted, fence-stamped, idempotent-where-durable, Galaxy-Relay radio) will ride reliable cross-shard arms — does R-3''s at-least-once + dedup give Signals the loss-safety + exactly-once they need, or is there a mismatch (e.g. Signals need idempotency the transport dedup doesn't provide across a sender restart)? (2) Multi-sharded meshes + client multi-mesh rendering: R-3' is shard-to-shard; does it generalize to the mesh-spanning-shards case, or assume 1-location-1-owner? (3) DRY: is there ONE codec/framing home, ONE FSM, ONE Fence, no per-kind/per-shard fork (HR2/HR3)? Any duplication R-3' introduced (e.g. the ack codec vs the reliable codec, the two read_one_*_frame fns)? (4) The reliable_acked / never-silent counters — is the observability AAA-grade (Prometheus-ready, a stuck-ack-path alert)? (5) Any DRIFT from the 6 hard rules, or any place R-3' makes a FUTURE feature (ship compound transfer, warp provisioning, damage-across-transfer-barrier) harder? (6) Is the R-4'/R-5'/R-6 owed-work honestly ledgered and correctly sequenced, or is something load-bearing hidden?`,
  },
]

phase('Audit')
const audits = await parallel(
  LENSES.map((l) => () =>
    agent(l.prompt, { label: `audit:${l.key}`, phase: 'Audit', schema: SCHEMA, model: 'opus' })
  )
)

const valid = audits.filter(Boolean)
const all = valid.flatMap((a) => (a.findings || []).map((f) => ({ ...f, lens: a.lens })))
const blocking = all.filter((f) => f.severity === 'CRITICAL' || f.severity === 'HIGH')
log(`audits: ${valid.map((a) => `${a.lens}=${a.verdict}`).join(', ')}; ${blocking.length} CRITICAL/HIGH`)

phase('Synthesize')
const synthesis = await agent(
  `You are the synthesizer for the post-R-3' HOLISTIC /goal audit. Read the real code/docs yourself to ADJUDICATE each finding (drop false alarms with reasons; keep real ones). Reviewer findings (JSON):\n${JSON.stringify(valid, null, 2)}\n\nThe /goal bar: "how does it all work together — complementary or will it break? DRY? Robust? Scalable? Cloud-ready? Elegant? AAA MMO quality? Prepared for the planned features (signal-heavy cross-shard, multi-mesh, PvP, hundreds-in-one-location)?" Produce: (1) an adjudicated de-duplicated list of REAL findings by severity, each with a concrete recommendation and whether it is IN-SCOPE for now vs correctly-owed-to-a-later-slice (R-4'/R-5'/R-6/Px); (2) a verdict DONE_NO_CRITICAL (no CRITICAL and no in-scope-unaddressed HIGH) or HAS_CRITICAL; (3) a one-paragraph bottom line on whether the at-least-once transport composes soundly and advances the end-goal. Distinguish a genuine in-scope defect from a correctly-ledgered future-slice item.`,
  { label: 'synthesize', phase: 'Synthesize', model: 'opus' }
)

return { verdicts: valid.map((a) => ({ lens: a.lens, verdict: a.verdict })), blocking_count: blocking.length, blocking, synthesis }
