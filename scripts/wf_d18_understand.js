export const meta = {
  name: 'd18-understand',
  description: 'Map the snapshot-datagram-over-VXLAN transport + quinn MTU config to scope the D-18 fix',
  phases: [{ title: 'Map', detail: '4 parallel readers over snapshot path / quinn MTU / reliability / fix space' }],
}

const CTX = `
CONTEXT — Voxeldust greenfield rebuild, worktree /Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system.
D-18 / SPIKE-3a is now CONFIRMED LIVE (S5a agent-HR6 run, k3d cluster). SYMPTOM: an in-cluster agent client logs
in over real QUIC/mTLS, reaches Active + gets its own_entity (the RELIABLE control plane works end-to-end over the
flannel VXLAN overlay), but its DevState shows snapshots_applied=0, decode_errors=0, stale_frames_dropped=0 — it
receives ZERO frames. The gateway logs, ~every 20ms: "OutboundBox staging over cap: shed oldest UNRELIABLE frames
(latest-wins Snapshot/Input) toward sustained-congested peer(s) shed=1 cap=4096". So the 20 Hz snapshot fan-out
(shard -> gateway -> client) is UNRELIABLE QUIC datagrams, and those are LOST on the k3d flannel VXLAN overlay
(which reduces the pod-network MTU by the VXLAN header, ~1450 vs 1500), degrading the connection so the gateway
cannot drain -> sheds. The servers tolerate the overlay because their control plane RETRANSMITS (reliable QUIC
streams); the unreliable snapshot plane does not. We must design the FIX. This is a READ-ONLY mapping task — cite
file:line + verbatim names. If something does NOT exist, say so. This map is the sole input to the D-18 fix design.
`

const READERS = [
  {
    key: 'snapshot-path',
    prompt: `${CTX}
MAP the snapshot delivery path + the datagram-vs-stream decision + the shed/congestion logic. Answer:
- How does a snapshot travel shard -> gateway -> client? The MsgClass (Snapshot?), the frame type
  (ReliableFrame vs DatagramFrame), and EXACTLY where the send chooses an unreliable QUIC datagram vs a reliable
  stream. Cite crates/io-prod/src/mesh.rs (send path) + crates/wire/src/channels.rs (MsgClass / EffectClass).
- The snapshot SIZING: CONSERVATIVE_DATAGRAM_BUDGET (value + where), the GW-1 snapshot content-partitioning
  (partition so no datagram exceeds the budget), and VD_SNAPSHOT_BUDGET (=1100 in the cloud manifest). Is 1100 a
  byte budget? Does a snapshot datagram + QUIC/UDP/IP/VXLAN overhead stay under ~1450, or could it exceed the
  overlay MTU? Cite the partitioning + the datagram-too-large drop counter (datagrams_dropped_too_large).
- The gateway OutboundBox staging + shed: where "shed oldest UNRELIABLE frames toward sustained-congested peer"
  is emitted (vd_node::app), the cap (4096), and WHAT "sustained-congested" means — is it a per-peer send-queue
  depth, a QUIC congestion signal, or a datagram send-error backpressure? Cite the exact condition. Does the shed
  distinguish "the peer is genuinely gone" from "the datagram send keeps failing (MTU)"?
- The mesh honesty counters (datagrams_dropped_too_large, datagrams_dropped_send) — how are they incremented, and
  is there any way to read them for the gateway (only the orchestrator binds /metrics today — confirm)?
Search: crates/io-prod/src/mesh.rs, crates/wire/src/channels.rs, crates/node/src/app.rs (OutboundBox shed).`,
  },
  {
    key: 'quinn-mtu',
    prompt: `${CTX}
MAP the quinn / QUIC MTU + datagram configuration in the mesh transport. Answer:
- The quinn TransportConfig + EndpointConfig the mesh builds (crates/io-prod/src/mesh.rs ~line 970 't =
  quinn::TransportConfig::default()' + Endpoint::server). What is set vs left default: initial_mtu,
  min_mtu, mtu_discovery_config (PMTUD), datagram_receive_buffer_size, datagram_send_buffer_size,
  max_udp_payload_size? Quote the config block.
- quinn's DEFAULT MTU behaviour: initial_mtu (1200?), PMTUD probing UPWARD to the path max, and how an unreliable
  DATAGRAM's max size is derived (Connection::max_datagram_size, which tracks the discovered path MTU). If PMTUD
  probes above the ~1450 VXLAN MTU and the probe packets are black-holed, does quinn cap correctly or does it
  over-estimate max_datagram_size and then silently drop oversize datagrams? What quinn version (Cargo.lock)?
- What is the flannel/k3d pod-network MTU actually (VXLAN overhead ~50 bytes => ~1450)? Is it configurable at
  k3d cluster create (flannel-backend / MTU / --k3s-arg)? Cite any existing MTU handling in the codebase or
  justfile k3d-up.
- The KNOB that would make snapshot datagrams SAFE on the overlay: cap the QUIC UDP payload / max_datagram_size to
  <= overlay-MTU-minus-overhead, or disable PMTUD above a conservative floor, or set initial_mtu low + min_mtu
  low. Which of these is exposed by the quinn version in use? Cite.
Search: crates/io-prod/src/mesh.rs (TransportConfig/Endpoint), Cargo.lock (quinn version), justfile (k3d-up MTU).`,
  },
  {
    key: 'reliability-keying',
    prompt: `${CTX}
MAP the reliability keying (why snapshots are unreliable) + whether a reliable-snapshot fallback is design-legal.
Answer:
- EffectClass / MsgClass reliability: which classes ride RELIABLE QUIC streams vs UNRELIABLE datagrams, and the
  DESIGN RATIONALE (snapshots are latest-wins / lossy-tolerant at 20 Hz; control/transfer is reliable). Cite the
  reliability mapping (crates/wire/src/channels.rs, the frozen wire contract) + any DEFERRED/design note on why
  snapshots are unreliable (the interpolation-buffer tolerates loss; a lost snapshot is superseded next tick).
- The end-goal constraint (BINDING): NO client-side prediction; players PHYSICALLY collide; the client renders on a
  100-150ms interpolation buffer; HUNDREDS of players in one location; multisharding meshes; the client renders
  MULTIPLE meshes simultaneously. Given that, is snapshot loss ACCEPTABLE (interp buffer bridges a dropped 20Hz
  frame) or does the volume (hundreds of entities) mean snapshots are large + frequent and loss is visible? Does
  this argue for keeping snapshots unreliable-but-MTU-safe (cap datagram size) vs making them reliable (risking
  head-of-line blocking + unbounded buffering under loss)?
- A reliable-snapshot fallback: would routing snapshots over reliable streams break the frozen wire contract /
  HR1 seams / the latest-wins semantics (a reliable stream can't drop a superseded frame -> head-of-line stall)?
  Is there a per-peer or per-path reliability override, or is reliability keyed strictly by class? Cite.
Search: crates/wire/src/channels.rs (EffectClass/MsgClass), docs/design/*.md (transfer_protocol/connection_plane),
docs/design/DEFERRED.md, CLAUDE.md (no-prediction / interpolation rules).`,
  },
  {
    key: 'fix-options',
    prompt: `${CTX}
MAP + rank the FIX options for D-18 (unreliable snapshot datagrams lost on the k3d VXLAN overlay). For EACH,
state what it touches, the blast radius (whole mesh vs in-cluster-only), the AAA-correctness for the end-goal
(networked multi-mesh client, hundreds of players), and any risk:
- (A) CAP the QUIC datagram/UDP-payload size to the overlay MTU (quinn initial_mtu/min_mtu/PMTUD policy or a
  max_datagram_size clamp) so snapshot datagrams always fit ~1450 and are never black-holed. Does this need a
  code change in io-prod mesh (TransportConfig) + a tuning knob (TransportTuning), and does it interact with the
  GW-1 content-partitioning (which already sizes to CONSERVATIVE_DATAGRAM_BUDGET)? Is CONSERVATIVE_DATAGRAM_BUDGET
  already <= overlay MTU (=> the real issue is quinn's PMTUD over-estimating, not the payload)?
- (B) SET the pod-network MTU so the overlay path fits 1500 (k3d cluster create flannel MTU / a larger host MTU /
  jumbo frames). Is this a manifest/justfile-only change (no code), and is it robust across real clouds (not just
  k3d) or a k3d-local crutch?
- (C) RELIABLE-snapshot fallback for the in-cluster path (route snapshots over reliable streams). Blast radius +
  the latest-wins / head-of-line-stall risk (from the reliability-keying reader).
- (D) A PMTUD / DPLPMTUD correctness fix (make quinn discover the real path MTU + never send a datagram larger
  than the confirmed path MTU). Is this a quinn-config or a quinn-version issue?
Recommend the one (or combination) that is AAA-correct + minimal-blast-radius + robust for a REAL cloud (not a
k3d-only hack), and flag what must be a load-test (SPIKE-3a's 4MB-burst gate) to ratify it.
Search: crates/io-prod/src/mesh.rs, crates/io-prod/src/transport tuning, crates/wire/src/channels.rs, justfile,
docs/design/DEFERRED.md (D-18 entry + the GW-1 note).`,
  },
]

const MAP_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'findings', 'gaps', 'recommendation'],
  properties: {
    dimension: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['fact', 'evidence'],
        properties: {
          fact: { type: 'string' },
          evidence: { type: 'string', description: 'file:line or verbatim name/quote' },
        },
      },
    },
    gaps: { type: 'array', items: { type: 'string' } },
    recommendation: { type: 'string' },
  },
}

phase('Map')
const maps = await parallel(
  READERS.map((r) => () =>
    agent(r.prompt, {
      label: `map:${r.key}`,
      phase: 'Map',
      schema: MAP_SCHEMA,
      agentType: 'Explore',
      model: 'opus',
    }),
  ),
)

return { maps: maps.filter(Boolean) }
