export const meta = {
  name: 'd18-fix-design',
  description: 'Design the concrete D-18 fix (quinn MTU pin + observability, cloud-gated, plumbed)',
  phases: [
    { title: 'Design', detail: '2 independent implementation plans' },
    { title: 'Adjudicate', detail: 'synthesize + adversarially stress' },
  ],
}

const GROUND = `
GROUND TRUTH (from the D-18 understand workflow + a LIVE k3d run — BINDING):
- SYMPTOM (confirmed live): an in-cluster agent client reaches Active + own_entity (reliable control plane works)
  but receives ZERO snapshot frames (snapshots_applied=0, decode_errors=0, stale_frames_dropped=0); the gateway
  sheds UNRELIABLE Snapshot/Input frames toward a "sustained-congested peer" ~every 20ms.
- The pod-network MTU is 1450 (measured live: eth0/cni0/flannel.1 = 1450; the flannel VXLAN overlay reduces the
  1500 host MTU by ~50B). Any UDP packet quinn sends above 1450 is black-holed on the overlay.
- Snapshots are MsgClass::Snapshot -> Reliability::Unreliable -> a bare DatagramFrame -> conn.send_datagram()
  (crates/io-prod/src/mesh.rs:2278-2323), NO retransmit (latest-wins). The reliable classes ride QUIC streams
  (they retransmit, which is why the control plane survives the overlay). Reliability is keyed STRICTLY by class
  (crates/sim/src/runtime.rs:108-113 MsgClass::reliability); there is NO per-peer/per-path override, and a
  reliable-snapshot fallback is REJECTED (head-of-line stall breaks the load-bearing latest-wins seam).
- The app payload is already MTU-conservative: VD_SNAPSHOT_BUDGET=1100 bounds the SnapshotDatagram body;
  CONSERVATIVE_DATAGRAM_BUDGET=1200 (crates/wire/src/channels.rs:202); partition_entities() sizes each chunk
  under budget; the mesh gates on conn.max_datagram_size() (mesh.rs:2306) + counts datagrams_dropped_too_large
  (mesh.rs:2312) / datagrams_dropped_send (mesh.rs:2329) — BUT both return Ok(()) so the loss is SILENT to the
  shed (app.rs) which then mislabels it "sustained-congested peer" (a never-silent violation).
- The quinn TransportConfig is built at crates/io-prod/src/mesh.rs:970-977 (t = quinn::TransportConfig::default()
  ... then Endpoint::server with the same config, mesh.rs:978-985) — SHARED by server + client mesh. quinn 0.11.9
  exposes t.initial_mtu(u16), t.min_mtu(u16), t.mtu_discovery_config(Option<MtuDiscoveryConfig>). Default quinn
  does UPWARD DPLPMTUD probing; on the overlay the upward probes over-commit the path MTU -> oversize datagrams
  black-holed.
- The cloud profile is resolved by crates/io-prod/src/boot.rs (Profile::Cloud vs DevTest); D-3 + transport knobs
  are already plumbed (LivenessTuning::cloud, DirectoryTuning::cloud, resolve_d3). A new mesh-MTU knob should ride
  the SAME cloud-profile plumbing (unclamped in DevTest/native, overlay-safe in Cloud), NOT a raw hardcode.
- RATIFY: the SPIKE-3a 4MB-burst load-test gate (DEFERRED.md D-18) — snapshots_applied climbs, dropped_too_large
  stays 0, the OutboundBox unreliable-shed returns to ~0. Plus a LIVE agent re-run (the S5a boundary Job must
  cross). The k3d flannel-MTU bump (k3d --k3s-arg) is a LOAD-TEST CONTROL to prove the diagnosis, NOT the ship.
`

const BRIEF = `
Produce the CONCRETE, buildable D-18 fix plan:
1. THE quinn MTU PIN: the exact TransportConfig edit (crates/io-prod/src/mesh.rs:970-977) — initial_mtu value,
   min_mtu, whether to mtu_discovery_config(None) (disable upward probing) or set a discovery ceiling. Give the
   NUMBER (headroom below 1450 for QUIC/UDP/IP + the DatagramFrame envelope + real-cloud-CNI variance) and the
   reasoning (max_datagram_size at that MTU must exceed the ~1115B snapshot datagram with margin).
2. THE KNOB: how the MTU value is plumbed (a TransportTuning field? an env VD_MESH_MTU via resolve/boot? a
   MeshConfig field?), how the CLOUD profile selects the overlay-safe value while DevTest/native stays unclamped
   (discovery on / a high MTU), and where it is validated. Keep it DRY with the existing cloud-profile plumbing.
3. OBSERVABILITY (never-silent): expose datagrams_dropped_too_large / datagrams_dropped_send (+ the OutboundBox
   unreliable-shed count) on the gateway + shard so the fault is diagnosable in-cluster. Today only the
   orchestrator binds /metrics (VD_ADMIN_ADDR); the gateway/shard have only /healthz+/readyz. Decide: add a
   metrics surface to the probe server, or a minimal counters endpoint, or emit a periodic tracing line. Prefer
   the smallest change that makes the drop VISIBLE (a tracing::warn on first datagrams_dropped_too_large is a
   candidate — the mesh currently warns? verify mesh.rs:2312 context).
4. TESTS + RATIFICATION: the unit/coverage story (the MTU knob resolution is Tier-A/Tier-B testable); the SPIKE-3a
   load-test shape (4MB burst over the overlay, assert snapshots flow + 0 dropped_too_large + shed->0, with the
   raised-flannel-MTU positive control); and the LIVE agent re-run (the S5a boundary Job crosses).
5. BLAST RADIUS + RISK: the TransportConfig is shared server+client — confirm pinning the MTU does NOT harm the
   in-datacenter/native path (gate it to cloud), does NOT break the existing real-QUIC tests (mesh_load /
   mesh_under_loss / spike2a) which run on loopback (MTU 65535), and interacts correctly with GW-1 partitioning.
Flag every code change, and give the EXACT initial_mtu number + its derivation. Be concrete.
`

const EMPHASES = [
  { key: 'minimal-pin', lean: 'Lean MINIMAL: the smallest correct change — pin initial_mtu + disable upward discovery in TransportConfig, cloud-gated via the existing boot profile, plus make the silent datagram drop LOUD (a first-drop tracing::warn). Avoid new endpoints/tuning-structs if a simpler plumb suffices. But never leave the drop silent or the value hardcoded-uncloud-gated.' },
  { key: 'robust-plumbed', lean: 'Lean ROBUST + OBSERVABLE: a proper TransportTuning MTU field resolved from the cloud profile + an env override (VD_MESH_MTU), a real counters/metrics surface on the gateway+shard (so SPIKE-3a can assert programmatically), and a coverage story. Design for real-cloud CNI MTU variance (the value is a knob, not a k3d constant) and for the load-test to READ the counters, not infer from logs.' },
]

const DESIGN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['emphasis', 'mtu_pin', 'knob_plumbing', 'observability', 'tests_ratification', 'blast_radius', 'code_changes', 'risks'],
  properties: {
    emphasis: { type: 'string' },
    mtu_pin: { type: 'string', description: 'exact TransportConfig edit + the initial_mtu NUMBER + derivation' },
    knob_plumbing: { type: 'string', description: 'how the MTU value is plumbed + cloud-gated + validated' },
    observability: { type: 'string', description: 'how the datagram drop becomes visible in-cluster' },
    tests_ratification: { type: 'string', description: 'unit tests + SPIKE-3a load test + live agent re-run' },
    blast_radius: { type: 'string', description: 'shared server+client config; native path unharmed; loopback tests unaffected' },
    code_changes: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

phase('Design')
const designs = await parallel(
  EMPHASES.map((e) => () =>
    agent(`${GROUND}\n${BRIEF}\n\nYOUR EMPHASIS: ${e.lean}`, {
      label: `design:${e.key}`,
      phase: 'Design',
      schema: DESIGN_SCHEMA,
      agentType: 'Explore',
      model: 'opus',
    }).then((d) => (d ? { ...d, _key: e.key } : null)),
  ),
)
const valid = designs.filter(Boolean)

phase('Adjudicate')
const plan = await agent(
  `${GROUND}\n${BRIEF}\n\nYou are the ADJUDICATOR + ADVERSARY. Synthesize ONE vetted, buildable D-18 fix plan from
the ${valid.length} designs below, and adversarially STRESS it: is the initial_mtu number actually safe (does
max_datagram_size at that MTU exceed the ~1115B snapshot datagram WITH margin, and is it below 1450 minus real-CNI
variance)? Does disabling upward discovery hurt the native/loopback path (or is it cloud-gated)? Does the
observability change actually make the drop VISIBLE + testable? Does the SPIKE-3a load test truly ratify (reads
counters, has a positive control)? Produce a concrete plan an implementer can build directly: the exact
mesh.rs TransportConfig edit + the pinned number + derivation, the knob + cloud-gate wiring (exact files/fns), the
observability change, the test/coverage plan, the SPIKE-3a load-test shape, and the residual risks. Confirm it does
NOT regress the existing real-QUIC tests (mesh_load/mesh_under_loss/spike2a on loopback) and keeps snapshots UNRELIABLE.

DESIGNS:
${JSON.stringify(valid, null, 1)}`,
  { label: 'adjudicate', phase: 'Adjudicate', model: 'opus' },
)

return { plan, designs: valid }
