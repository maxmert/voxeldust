export const meta = {
  name: 'carbon-vs-voxeldust',
  description: 'Deep-analyze CCP Carbon/EVE engine repos vs our Rust voxel-MMO engine; produce reuse/adopt/ignore verdicts',
  phases: [
    { title: 'Recon-Ours', detail: 'map our multishard engine subsystems + game goals' },
    { title: 'Recon-Theirs', detail: 'deep-read each Carbon repo from cloned source' },
    { title: 'Compare', detail: 'per-domain reuse/adopt/ignore verdicts' },
    { title: 'Verify', detail: 'adversarially check reuse-feasibility claims' },
    { title: 'Critic', detail: 'completeness pass for missed opportunities' },
  ],
}

const CARBON = '/private/tmp/claude-501/-Users-maxim-Projects-my-voxeldust/3a45b713-5570-4ec7-9866-81578af56339/scratchpad/carbon'
const OURS = '/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system'
const ROOT_CLAUDE = '/Users/maxim/Projects/my/voxeldust/CLAUDE.md'

const VISION = `WHAT WE ARE BUILDING (voxeldust): an AAA-quality multiplayer voxel-planet MMO — "Star Citizen meets Minecraft".
- Procedurally generated universe: universe_seed -> system_seed -> planet_seed. Same seed = same world on client+server; only block-edit deltas networked.
- Spherical voxel planets (cubic-sphere, 6 faces, Nowell mapping), 62^3 chunks, binary-greedy-meshing, sphere-projected verts.
- Player-built ships from the same voxel grid; functional blocks (thrusters, cockpit, power); dual physics (exterior Newtonian rigid body + interior walkable KCC).
- Newtonian space physics; spherical gravity g=G*M/r^2; Rapier3D server-side physics.
- MULTISHARD architecture (the current greenfield rebuild in this worktree): system shard / planet shard(s) / ship shard(s); gateway-owned connection; SEAMLESS cross-shard transfers via a saga-based transfer protocol with redelivering transport, WAL/redb persistence, lease lifecycle, entity forward re-home.
- Rust everywhere: core (terrain/mesh/coord math), shard binaries on bevy_ecs 0.18, client on Bevy 0.18 + wgpu + bevy_egui, tokio async net, FlatBuffers wire, glam math, Rapier physics.
- Earth-sized planets eventually (~6.4M block radius) -> needs floating origin, tiered-i64 lattice coordinates (LatticePos), sparse shells.
- Quality bar: AAA MMO, no shortcuts, no magic numbers (all derived from seed / per-entity fields), deterministic, server-authoritative, seamless visual transitions.`

// ---------- schemas ----------
const OURS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['subsystem', 'purpose', 'capabilities', 'gaps', 'key_files'],
  properties: {
    subsystem: { type: 'string' },
    purpose: { type: 'string' },
    capabilities: { type: 'array', items: { type: 'string' } },
    gaps: { type: 'array', items: { type: 'string' }, description: 'known holes / TODOs / immature areas' },
    key_files: { type: 'array', items: { type: 'string' } },
    maturity: { type: 'string', enum: ['solid', 'in-progress', 'stub', 'planned'] },
    notes: { type: 'string' },
  },
}

const THEIRS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['repo', 'purpose', 'languages', 'license', 'key_architecture', 'reuse_class', 'reuse_rationale', 'relevance'],
  properties: {
    repo: { type: 'string' },
    purpose: { type: 'string' },
    languages: { type: 'array', items: { type: 'string' } },
    deps: { type: 'array', items: { type: 'string' }, description: 'notable external deps / platform ties (DirectX, Python, Wwise, etc.)' },
    license: { type: 'string' },
    key_architecture: { type: 'string', description: 'how it is built + notable algorithms/data-structures/patterns' },
    notable_algorithms: { type: 'array', items: { type: 'string' } },
    reuse_class: { type: 'string', enum: ['drop-in-link', 'ffi-wrap', 'algorithm-port', 'architecture-lesson', 'format-interop', 'ignore'] },
    reuse_rationale: { type: 'string' },
    specific_asset_or_lesson: { type: 'string', description: 'the concrete thing we could take (a file, an algorithm, a design pattern)' },
    relevance: { type: 'string', enum: ['high', 'medium', 'low'] },
    portability_blockers: { type: 'array', items: { type: 'string' }, description: 'DirectX-only, Python-C-API, Windows-only, Wwise-required, etc.' },
  },
}

const DIM_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dimension', 'reuse_candidates', 'adopt_practices', 'we_do_better_ignore', 'recommendation'],
  properties: {
    dimension: { type: 'string' },
    reuse_candidates: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['repo', 'what', 'feasibility', 'effort', 'license_ok'],
      properties: {
        repo: { type: 'string' }, what: { type: 'string' },
        feasibility: { type: 'string', enum: ['drop-in-link', 'ffi-wrap', 'algorithm-port', 'format-interop', 'not-worth-it'] },
        effort: { type: 'string', enum: ['low', 'medium', 'high', 'very-high'] },
        license_ok: { type: 'boolean' },
        vs_rust_alternative: { type: 'string', description: 'what mature Rust crate/approach we already have or would use instead' },
      },
    }},
    adopt_practices: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['practice', 'from_repo', 'why'],
      properties: { practice: { type: 'string' }, from_repo: { type: 'string' }, why: { type: 'string' }, how: { type: 'string' } },
    }},
    we_do_better_ignore: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['topic', 'why_ours_is_better'],
      properties: { topic: { type: 'string' }, why_ours_is_better: { type: 'string' } },
    }},
    recommendation: { type: 'string' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dimension', 'checks', 'corrected_recommendation', 'confidence'],
  properties: {
    dimension: { type: 'string' },
    checks: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['claim', 'verdict', 'reason'],
      properties: {
        claim: { type: 'string' },
        verdict: { type: 'string', enum: ['CONFIRMED', 'OVERSTATED', 'WRONG', 'UNDERSTATED'] },
        reason: { type: 'string' },
      },
    }},
    corrected_recommendation: { type: 'string' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

// ---------- Phase 1: recon ours ----------
phase('Recon-Ours')
const OURS_TASKS = [
  { key: 'transfer', focus: `The cross-shard TRANSFER system (the heart of the multishard rebuild). Read design docs ${OURS}/docs/design/{transfer_protocol.md,generic_transfer.md,d6_saga_wal.md,connection_plane.md,sealed_shards.md,slice_1d4_1d5_ordered_demote.md,identity_persistence.md,DEFERRED.md} and code in ${OURS}/crates/{wire,connection-plane,node}. Explain the saga model, lease lifecycle, CAS/authority handoff, persistence (redb/WAL), entity forward re-home, abort/robustness.` },
  { key: 'transport', focus: `The RELIABLE TRANSPORT / netcode layer (R-4 "redelivering transport"). Read ${OURS}/crates/io-prod (esp src/mesh.rs), ${OURS}/crates/wire, and docs ${OURS}/docs/design/{d7b_transient_ballistic.md,d7c_transient_burst.md,d7d_transient_crash.md}. Explain incarnation+seq+cumulative-ack+retransmit+reconnect-replay, reliability lanes, and where UDP/TCP/datagram fit.` },
  { key: 'sim', focus: `The SIMULATION + PHYSICS + COORDINATE core. Read ${OURS}/crates/{sim,core} and the coordinate/physics notes in ${ROOT_CLAUDE}. Explain: cubic-sphere mapping, 62^3 chunk meshing, terrain gen, Rapier usage, spherical gravity, KCC, and the tiered-i64 LatticePos floating-origin coordinate base. What space/newtonian physics exists vs planned?` },
  { key: 'client', focus: `The CLIENT + RENDERER. Read ${OURS}/crates/{client,client-render,client-harness} and ${OURS}/docs/design/slice_3_renderer.md plus spikes/bevy-readback. Explain the Bevy 0.18 + wgpu pipeline, dual-shard compositing, screenshot/readback harness, and current rendering maturity.` },
  { key: 'topology', focus: `SERVER TOPOLOGY / shard binaries / orchestration. Read ${OURS}/crates/{node,bins} and ${OURS}/docs/design/{PLAN.md,roadmap.json,integration.json}. Explain how shards are processes, how the gateway/orchestrator works, the 20Hz tick model, per-system schedules, and how a player/entity lives across shards.` },
  { key: 'goals', focus: `GAME GOALS + VISION + roadmap. Read ${ROOT_CLAUDE}, ${OURS}/CLAUDE.md, ${OURS}/docs/design/PLAN.md and ${OURS}/docs/design/roadmap.json. Summarize the product vision, the quality bar, the near-term roadmap slices, and what MMO-scale challenges are explicitly acknowledged (floating origin, LOD stance, single vs multi shard, etc.).` },
  { key: 'infra', focus: `BUILD / TEST / OBSERVABILITY / harness infra. Read ${OURS}/justfile, ${OURS}/crates/{harness,devproto}, ${OURS}/tests, ${OURS}/coverage-exemptions.toml and skim ${OURS}/docs/audit/*.json. Explain the test/gate/coverage strategy, dev-cluster/k3d tooling, and any observability (metrics/tracing) present or missing.` },
]
const oursThunks = OURS_TASKS.map(t => () => agent(
  `You are auditing OUR OWN engine (a Rust voxel-MMO in a greenfield multishard rebuild).\n${VISION}\n\nYOUR SUBSYSTEM: ${t.key}.\n${t.focus}\n\nRead the actual files (use Read/Grep/Glob on the absolute paths given). Be concrete and cite key files. Return a rigorous structured map: what exists and is solid, what is in-progress/stub/planned, and the real gaps. Do not invent; if a doc is a plan not code, say so.`,
  { label: `ours:${t.key}`, phase: 'Recon-Ours', schema: OURS_SCHEMA, effort: 'high' }
))

// ---------- Phase 2: recon theirs ----------
const REPOS_DEEP = [
  ['destiny', 'EVE\'s core game-world simulation engine — THE space sim. Focus: ballistic/newtonian movement integration, the "grid" spatial model, determinism, tick model, how objects/warp/orbit are simulated, server-side scale tricks (is this where time-dilation lives?).'],
  ['trinity', 'The rendering engine. Focus: renderer architecture, scene graph, material/shader system, LOD, large-scale space rendering, DirectX ties, how content is authored/loaded. It is 98MB — skim structure + READMEs + key headers, do NOT read every file.'],
  ['core', 'Low-level cross-platform syscall abstractions (33 stars, their most-starred lib). Focus: threading, memory, filesystem, containers, string, time — what fundamental utilities they standardize on and whether any concept beats std Rust.'],
  ['scheduler', 'Channels + scheduler for Greenlet coroutines (24 stars). THIS IS THEIR MMO CONCURRENCY MODEL. Focus: fiber/green-thread scheduling, channels, how EVE runs massive concurrency on few cores, cooperative vs preemptive. Compare conceptually to tokio async.'],
  ['blue', 'Glue between Python and C++: exposes C++ to Python AND handles persistence + resource loading (textures/geometry). Focus: their object model, persistence format, resource-loading pipeline, hot-reload. This is architecturally central to EVE.'],
  ['blueexposure', 'Python exposure for C++ + utilities. Focus: the binding mechanism and any codegen.'],
  ['io', 'Low-level networking (note: license is "Other" not MIT — CHECK THE LICENSE FILE carefully). Focus: socket abstraction, protocol handling, async model, reliability. Compare to our io-prod redelivering transport.'],
  ['pathfinder', 'C++ route-finding over EVE map (stargate graph). Focus: graph representation, algorithm (Dijkstra/A*/contraction?), how galaxy-scale routing is made fast. Relevant to our warp/navigation.'],
  ['math', 'Basic vector/plane/quaternion math. Focus: what primitives, SIMD, precision (f32/f64), determinism guarantees. Compare to glam.'],
  ['geo2', 'Python math built on Microsoft DirectXMath. Focus: what it adds over math, DirectX tie, precision.'],
  ['resources', 'Resource operations for Carbon projects (43MB). Focus: asset packaging, streaming, caching, the resource id/index scheme, hot-reload, CDN/patch delivery.'],
  ['db', 'Wrapper for game-server database access. Focus: the persistence model, connection pooling, ORM/query pattern, what DB, transactions. Compare to our redb/WAL identity persistence.'],
  ['prometheus', 'Native Prometheus client for "the monolith". Focus: metrics model, exposition, low-overhead collection. Compare to Rust prometheus/metrics crates. Note the word "monolith" — what does it imply about EVE server architecture?'],
  ['spatial-audio-clustering', 'Apache-2.0 Wwise plugin: dynamically groups spatial audio objects by proximity. Focus: the clustering algorithm, why it matters at scale (thousands of sound sources in a fleet fight). Is the algorithm portable independent of Wwise?'],
  ['mesh', '3D mesh manipulation, animation, storage. Focus: mesh data structures, animation/skinning, the storage format, LOD/compression. Compare to our binary-greedy-meshing chunk meshes.'],
  ['audio', 'The audio engine (200MB — skim, do not read binaries/assets). Focus: mixing, 3D spatialization, streaming, Wwise relationship, DSP.'],
  ['imageio', 'Bitmap image load/save + format serialization (C). Focus: which formats, the API. Compare to Rust image crate.'],
  ['imagetools', 'Image processing + compression (Python ext + C++). Focus: texture compression (BC/DXT?), mipmap gen, the pipeline.'],
  ['localization', 'Localization framework. Focus: string catalog format, plural/gender handling, runtime lookup, tooling. A real AAA-MMO need we have not touched.'],
  ['red-to-black-converter', 'Tool converting "red" files to "black" (Python). Focus: what red/black ARE (EVE\'s serialization formats) — infer the object-persistence/serialization design behind Blue. This reveals their data pipeline.'],
  ['exefile', 'Elements to build the final executable. Focus: app bootstrap, module composition, how the monolith is assembled from these libs. Reveals the overall engine composition.'],
  ['parser', 'Math expression parser. Focus: use case (data-driven formulas for ship/module stats?), grammar. Relevant to our "no magic numbers, derive from data" ethos.'],
  ['grpc', 'Base for project-specific Python gRPC modules. Focus: their service/RPC boundary, codegen, where gRPC sits vs the fast game protocol.'],
  ['videoplayer', 'Video player on Trinity. Focus: codec, integration. Low priority — quick pass.'],
  ['pdm', 'Platform Detection Module — OS-agnostic data collection. Focus: telemetry/hardware-survey, why. Quick pass.'],
]
const REPOS_TRIVIAL = ['ime', 'spacemouse', 'd3dinfo', 'exefileconsole', 'trinityaudioapi', 'pdm-proto-wrapper', 'vcpkg-registry', 'localization-tools']

const theirsThunks = REPOS_DEEP.map(([repo, focus]) => () => agent(
  `You are analyzing an open-sourced repo from CCP Games' Carbon engine (the tech behind EVE Online), cloned locally at ${CARBON}/${repo}.\n\nOUR context (what we might reuse it for):\n${VISION}\n\nREPO: ${repo}\nFOCUS: ${focus}\n\nRead the README, build files (CMakeLists.txt/vcpkg.json), public headers, and skim representative source. Determine license from the LICENSE file. Then judge, HONESTLY, how a Rust/Bevy engine could reuse this: 'drop-in-link' (link the C/C++ lib as-is via FFI with little glue), 'ffi-wrap' (usable but needs a real Rust binding layer), 'algorithm-port' (reimplement the algorithm in Rust — the value is the algorithm/design not the code), 'format-interop' (match a data format for interop), 'architecture-lesson' (no code reuse; the value is the design pattern), or 'ignore'. Be skeptical of C++->Rust drop-in claims; note DirectX/Windows/Python/Wwise portability blockers. Return the structured analysis.`,
  { label: `theirs:${repo}`, phase: 'Recon-Theirs', schema: THEIRS_SCHEMA, effort: 'high' }
))
theirsThunks.push(() => agent(
  `Batch-classify these small/peripheral Carbon repos (each cloned at ${CARBON}/<name>): ${REPOS_TRIVIAL.join(', ')} plus any of {ime, spacemouse, d3dinfo, exefileconsole, trinityaudioapi, pdm-proto-wrapper, vcpkg-registry} you have not covered. For each, one line: purpose + reuse_class + relevance to a Rust voxel MMO. Return ONE analysis object summarizing them collectively (repo="_trivial_batch", key_architecture = the per-repo one-liners).`,
  { label: 'theirs:_trivial', phase: 'Recon-Theirs', schema: THEIRS_SCHEMA, effort: 'medium' }
))

log(`Recon: ${OURS_TASKS.length} ours + ${REPOS_DEEP.length + 1} theirs agents`)
const [oursResults, theirsResults] = await Promise.all([
  parallel(oursThunks),
  parallel(theirsThunks),
])
const ours = oursResults.filter(Boolean)
const theirs = theirsResults.filter(Boolean)
log(`Recon done: ${ours.length} ours, ${theirs.length} theirs`)

const oursDigest = JSON.stringify(ours, null, 1)
const theirsDigest = JSON.stringify(theirs, null, 1)

// ---------- Phase 3+4: compare (pipeline: analyze -> adversarial verify) ----------
phase('Compare')
const DIMENSIONS = [
  { key: 'space-sim-physics', repos: ['destiny', 'pathfinder', 'math', 'geo2', 'parser'], q: 'Space simulation, newtonian/ballistic physics, orbits/warp, galaxy-scale navigation, and data-driven stat formulas.' },
  { key: 'networking-transport-concurrency', repos: ['io', 'scheduler', 'grpc', 'destiny'], q: 'Reliable transport, the concurrency model (fibers vs async), RPC boundaries, and how EVE sustains thousands of clients per node (single-shard, time dilation, node/proxy topology).' },
  { key: 'rendering-assets', repos: ['trinity', 'mesh', 'imageio', 'imagetools', 'resources', 'videoplayer'], q: 'Rendering architecture, mesh/animation storage, texture compression, asset streaming/packaging, large-scale space rendering & LOD.' },
  { key: 'scripting-persistence-serialization', repos: ['blue', 'blueexposure', 'red-to-black-converter', 'db', 'grpc'], q: 'Object model, persistence & serialization formats (red/black), the C++/scripting bridge, resource loading, DB access — vs our redb/WAL/saga + Rust/Lua plans.' },
  { key: 'math-coordinates', repos: ['math', 'geo2', 'parser'], q: 'Math primitives, precision/determinism, and coordinate/precision strategy at galaxy scale — vs our glam + tiered-i64 LatticePos floating origin.' },
  { key: 'audio', repos: ['audio', 'spatial-audio-clustering', 'trinityaudioapi'], q: 'Audio engine, 3D spatialization, and scaling to thousands of sources (fleet fights) — a domain we have not started.' },
  { key: 'topology-scale-observability', repos: ['exefile', 'core', 'prometheus', 'localization', 'db', 'pdm'], q: 'Overall engine composition (the monolith/exefile), server topology & MMO-scale operations, observability, localization, and platform abstractions.' },
]

const compareResults = await pipeline(
  DIMENSIONS,
  // stage 1: analyze the dimension
  dim => agent(
    `You are the lead architect deciding what our Rust voxel-MMO engine should REUSE, ADOPT, or IGNORE from CCP's Carbon/EVE engine, for the domain: "${dim.key}".\nDomain scope: ${dim.q}\nMost-relevant Carbon repos: ${dim.repos.join(', ')}.\n\nOUR ENGINE (structured recon of all subsystems):\n${oursDigest}\n\nCARBON REPOS (structured recon of all repos):\n${theirsDigest}\n\nOUR CONTEXT:\n${VISION}\n\nGround rules for your verdicts:\n- We are Rust; they are C++/Python/DirectX. A C++ lib is NOT free to adopt — weigh FFI cost, build/toolchain drag, determinism, and Windows/DirectX/Python/Wwise lock-in against the MATURE RUST ALTERNATIVE we already use or could use (glam, tokio, wgpu/Bevy, image, rapier, redb, petgraph, mlua). Our standing rules: prefer reusing existing deps, never adopt a new lib unilaterally (it is a joint decision), avoid dependency drift, no magic numbers, deterministic + seed-derived, AAA quality no shortcuts.\n- "Best practice to adopt" = an EVE-proven design pattern we should copy in Rust (e.g. single-shard + time dilation, the grid model, fiber-per-entity, data-driven stat formulas), independent of their code.\n- "We do better / ignore" = where our approach (voxel-native, deterministic seed gen, Rust safety, Bevy) is genuinely superior or where their thing is irrelevant to voxels.\nBe concrete, honest, and specific (name the repo, the algorithm/pattern, the effort). Return the structured verdict.`,
    { label: `cmp:${dim.key}`, phase: 'Compare', schema: DIM_SCHEMA, effort: 'high' }
  ),
  // stage 2: adversarial verification of the feasibility claims (returns merged {analysis, verdict})
  (analysis, dim) => agent(
    `Adversarially fact-check this reuse/adopt/ignore analysis for domain "${dim.key}". You are a skeptical Rust systems engineer whose job is to KILL over-optimistic reuse claims and license/portability errors.\n\nANALYSIS TO CHECK:\n${JSON.stringify(analysis, null, 1)}\n\nSUPPORTING RECON (theirs):\n${theirsDigest}\n\nFor each material claim (esp. every reuse_candidate feasibility/effort and every "license_ok"), verify against reality: Is a C++/DirectX/Python/Wwise lib really linkable into a Rust engine at that effort? Is the license actually compatible (recall io = "Other", spatial-audio-clustering = Apache-2.0, rest MIT)? Does a mature Rust crate already do this better (making reuse pointless / dep-drift)? Is any "adopt practice" actually inapplicable to a voxel game? Mark each check CONFIRMED / OVERSTATED / WRONG / UNDERSTATED with a reason, then give a corrected_recommendation. Default to skepticism on drop-in/ffi claims. Consult ${CARBON}/${dim.repos[0]} on disk if you need to confirm a detail. Return the structured verdict.`,
    { label: `vrf:${dim.key}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: 'high' }
  ).then(verdict => ({ dimension: dim.key, analysis, verdict })),
).then(rows => rows.filter(Boolean))

// ---------- Phase 5: completeness critic ----------
phase('Critic')
const critic = await agent(
  `You are a completeness critic for a "what can we reuse from CCP Carbon/EVE" analysis. Below is everything gathered. Identify what is MISSING or UNDER-EXPLORED: a Carbon repo whose high-value asset was overlooked, a reuse/interop opportunity not surfaced, an EVE-proven MMO-scale practice our roadmap ignores at its peril, or a place the analysis over-claimed. Also name the TOP 3 highest-leverage takeaways overall.\n\nOUR ENGINE:\n${oursDigest}\n\nCARBON REPOS:\n${theirsDigest}\n\nPER-DOMAIN VERDICTS (analysis + adversarial verify):\n${JSON.stringify(compareResults, null, 1)}\n\nReturn plain prose: (1) gaps/misses, (2) over-claims to walk back, (3) top-3 takeaways, (4) any repo we should clone-and-study further.`,
  { label: 'critic', phase: 'Critic', effort: 'high' }
)

return {
  ours,
  theirs,
  dimensions: compareResults,
  critic,
  counts: { ours: ours.length, theirs: theirs.length, dimensions: compareResults.length },
}
