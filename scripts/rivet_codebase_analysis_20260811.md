# Rivet codebase analysis — can we lift parts of it?

**Date** 2026-08-11 · **Subject** `rivet-dev/rivet` @ `750d0f2` · **License** Apache-2.0 (single top-level
LICENSE; only vendored exception is `engine/packages/term/LICENSE.md`) · **Method** shallow clone read
locally, all numbers measured from the tree, not from docs or marketing.

## Verdict

**The ideas transfer. The code almost nowhere — except one crate.**

Not for the usual reasons (license, language, death). Apache-2.0 means we *may* copy with attribution; it is
Rust; it is pushed daily. Three measured reasons:

1. **Every interesting part is a service, not a library.** Each candidate's dependency closure is the whole
   platform. Nothing worth taking has a small closure.
2. **The one seam that looked like our escape hatch is on the wrong side of the complexity.** Their
   pluggable-storage trait pushes all the workflow-history semantics *down* into the implementation.
3. **Their robustness is production mileage, not a test suite we can inherit.** We are 7–14× denser in
   tests and 100% region+branch on Tier-A. Copying their code imports the obligation to test it ourselves.

And the sharpest finding: **at the exact point where their design meets our hardest requirement — a thing
changing location — their own doc says "Future Work."** See §4.

## 1. Scale, measured

| | LOC | crates |
|---|---:|---:|
| Rivet engine workspace (Rust) | 244,439 | 70 |
| Our workspace (Rust, incl. tests) | 137,722 | 13 |

Same order of magnitude. This is not "adopt a giant, save years"; it is "merge a comparable codebase".

## 2. Dependency closures — the core measurement

Internal-crate closure per candidate, with distinct external crates.io deps across the closure:

| candidate | own LOC | closure crates | closure LOC | ext deps |
|---|---:|---:|---:|---:|
| `gasoline` (durable workflow engine) | 20,685 | 24 | **48,700** | 79 |
| `epoxy` (per-key Paxos KV) | 7,414 | 24 | **41,244** | 73 |
| `rivet-guard-core` (Rust proxy) | 8,365 | 23 | **40,701** | 82 |
| `pegboard` (lifecycle orchestrator) | 30,643 | 33 | **116,751** | 92 |
| `pegboard-gateway2` (conn. migration) | 3,478 | 35 | **128,594** | 100 |
| `pegboard-runner` (engine↔worker) | 2,763 | 35 | 127,879 | 99 |
| `rivet-runner-protocol` (wire schema) | 4,781 | **10** | **21,404** | 47 |
| `universaldb` (FDB-style KV) | 11,815 | **5** | **12,198** | 29 |

Every closure drags in `universaldb` + `universalpubsub` + `rivet-pools` (Postgres/NATS pools) +
`rivet-config` + `rivet-metrics-server` + `rivet-test-deps-docker`. `gasoline` additionally takes
**clickhouse**, **sentry**, and **opentelemetry** as direct dependencies. `epoxy` depends on `gasoline`,
`universaldb`, **axum** (HTTP server) and **reqwest** (HTTP client) — the Paxos implementation is a
networked service, not a consensus library.

## 3. The storage seam does not save us

`gasoline/src/db/mod.rs` (391 LOC) defines `trait Database` — at first glance the escape hatch: keep the
replay engine, implement the trait over redb. It is **40 methods**, and they are not storage primitives.
They are workflow semantics:

```
commit_workflow_activity_event      upsert_workflow_loop_event
commit_workflow_sleep_event         commit_workflow_branch_event
commit_workflow_version_check_event commit_workflow_removed_event
pull_workflows / pull_next_signals  clear_expired_leases / publish_metrics
async fn new(config: rivet_config::Config, pools: rivet_pools::Pools)
```

The constructor takes their config and pool types. The history-encoding logic lives **below** the trait, in
the 11k-LOC KV layer, not above it in the engine. Measured shape of `gasoline`:

| part | LOC | note |
|---|---:|---|
| `db/kv/**` (key layout + encoding) | ~11,000 | bound to `universaldb`, would have to be rewritten |
| `ctx/**` + `history/**` + `worker.rs` + builders | ~4,500 | the actual replay engine |
| metrics/debug/error/misc | ~5,000 | ops baggage |

So "keep the engine, swap the store" means reimplementing 11k LOC of semantics behind a 40-method trait we
did not design. The replay core we actually want is ~4,500 LOC.

## 4. ★ Their directory equivalent cannot move things yet

`docs-internal/engine/ACTOR_KEY_RESERVATION.md` describes the closest analogue to our directory CAS:
Epoxy per-key Paxos storing *which datacenter an actor key resolves to*. It buys local reads by making
values **immutable once set** — `kv_get_optimistic` "assumes a value does not change after being set."

Consequences they state themselves:
- They cannot store the actor ID (IDs are created/destroyed, values can't change) → they store an opaque
  reservation ID that encodes the datacenter.
- Moving an actor between datacenters is listed as **"Reservation Chains & Moving Reservation Datacenters
  (Future Work)"** — forwarding pointers from the old reservation to the new one.

**Our core operation is exactly the thing marked Future Work.** Every transfer changes the owner. Their
strong-consistency design traded relocation away for local reads; relocation is our whole product. Their
consensus code is therefore not a shortcut for our directory — the requirement it optimises for is the
opposite of ours.

Worth keeping, though: the *immutability insight* (§7 item 2).

## 5. Test posture — the "robust code" hypothesis, measured

| | test fns | LOC | per 1k LOC |
|---|---:|---:|---:|
| `gasoline` | 20 | 20,685 | 1.0 |
| `pegboard` | 58 | 30,643 | 1.9 |
| ours (`crates/` + `tests/`) | **1,971** | 137,722 | **14.3** |
| ours, `crates/core` | 338 | 13,749 | 24.6 |

Their tests boot Docker containers (`rivet-test-deps` → `rivet_test_deps_docker`, spins named containers,
sets `RIVET_TEST_RUNTIME`). Ours run in-process on a virtual clock with no sockets, in milliseconds.

Their code is genuinely robust — earned by production traffic at millions of MAU. But battle-testing is not
a transferable artifact. Copy the code and we inherit: HR5 obligation on every lifted line, a test suite
~1/10th our density that needs Docker, and no design intuition for code we did not write.

Also measured, per candidate: `gasoline` 37×`Instant::now`, 60×`tokio::time`, 24×`HashMap`; `universaldb`
25/34/19; `pegboard` 28 clock reads, 23 rand uses. All fine for a control plane, all illegal above our seam.

## 6. In-flight rewrites

`workflows/actor/` **and** `actor2/`; `runner.rs` **and** `runner2.rs`; `pegboard-gateway` **and**
`pegboard-gateway2`; `keys/` + `legacy_subspace`. Lifting means picking a side of a migration in progress
and inheriting whichever half they abandon.

## 7. What is genuinely worth taking (ideas, docs, one crate)

Ranked by value/effort. None of these require importing a dependency.

1. **★ Workflow-history location scheme** — `GASOLINE/WORKFLOW_HISTORY.md` + `history/cursor.rs` (879 LOC).
   Durable steps are addressed by *ordinate coordinates* (`{1}`, `{1,4}`, `{0.1}`, `{2,11,4.1}`) so a step
   can be **inserted into or removed from a saga that is already mid-flight**, gated by a step version, with
   `HistoryDiverged` when the rules are broken. This solves the #1 problem we will hit the first time we add
   a step to a shipped transfer saga. Highest-value idea in the repo for us. Study + reimplement small.
2. **★ Epoxy's immutability argument** — immutable values ⇒ no read quorums, local reads, idempotent
   replication, unordered changelogs, no merge logic. Directly applicable: if a directory entry is keyed
   *per fence* (a new fence writes a new key, never overwrites), we inherit idempotent replay and local
   reads. Their "reservation chain" sketch is the forwarding-pointer design we would need anyway.
3. **★ Sleep-sequence invariants** — `sleep-sequence.md`. Two distinct predicates (`can_arm_sleep_timer`
   idle vs `can_finalize_sleep` grace), an *enumerated* counter list in each, a grace deadline that logs
   **every non-drained counter**, and the stated lesson: a hand-paired boolean flag wedged actors awake; a
   scope-bound counter cannot leak. Maps 1:1 onto realm teardown and hysteresis, and onto our own
   "teardowns_reaped lies" debugging lesson. Adopt the pattern verbatim.
4. **Lifecycle authority rule** — "the engine owns lifecycle authority; `sleep()` is fire-and-forget intent;
   the local transition runs only when the engine replies `StopActor`." Independent restatement of our fence
   discipline, in the lifecycle layer. Free confirmation; worth a review rule.
5. **Connection-survives-the-move mechanism** — `HIBERNATING_WS.md` + `pegboard-gateway2` (3,444 LOC, 9
   files, the most readable crate in the tree). The trick in one line: the **proxy** holds the client socket,
   the **store** holds a keepalive liveness marker, and the **start command carries the list of inherited
   live connections** so the new host knows what it adopted. Direct input to our gateway route swap.
6. **Route-swap point-of-no-return rule** — `PEGBOARD_TUNNEL_RETRIES.md`. Transient failure ⇒ typed error
   ⇒ re-resolve route *ignoring cache* ⇒ backoff retry; and retries are only legal **before** the client
   socket is accepted — after that the handler must close gracefully. We need exactly this rule stated for
   our swap.
7. **Wire-versioning discipline** — `engine/CLAUDE.md`: never edit a published schema; add a version;
   convert **field-by-field even when versions look byte-identical**; ban `to_vec`+`from_slice` shortcuts
   and same-bytes macros; bump the protocol constant together with the schema. Five review rules for our
   frozen wire, at zero cost. (Their codec is BARE; ours stays postcard.)
8. **Test-fixture patterns** — checked-in RocksDB checkpoints as migration fixtures
   (`test-snapshot-gen`), and `UDB_SIMULATED_LATENCY_MS` injected below the storage seam for benchmarks.
   Both map onto our harness and redb store.
9. **Architecture confirmation** — `ACTOR_LIFECYCLE.md`'s sequence diagram is: per-entity durable workflow
   + per-worker durable workflow + gateway that awaits readiness then tunnels. That is our
   orchestrator/saga/gateway split, arrived at independently, at production scale. Diff our sequence against
   theirs as a checklist.

## 8. The only plausible *code* lift

`rivet-runner-protocol` — 4,781 LOC, closure of 10 crates / 21,404 LOC, the smallest closure in the tree,
and it is a generated versioned wire schema (BARE + their `vbare` versioning layer). It is also the part we
least need: our wire is frozen and postcard-based, and adopting theirs means swapping codec and codegen.
**Take the versioning discipline (§7.7), not the crate.**

## 9. Recommendation

- **Do not import any Rivet crate.** No candidate has a closure small enough, and the one with a small
  closure is the one we don't need.
- **Do harvest §7.1, §7.2, §7.3, §7.5, §7.6, §7.7** — six concrete, bounded pieces of work, all
  reimplementation-from-design, no dependency, no license obligation beyond attribution in comments where a
  design is directly derived.
- **Highest priority: §7.1 (mid-flight saga step insertion).** We have a saga runtime and no answer for
  changing it after ship. This is the one place their engineering is years ahead of ours and the fix is a
  small file, not a platform.
- Keep the clone for reference reading; do not vendor.
