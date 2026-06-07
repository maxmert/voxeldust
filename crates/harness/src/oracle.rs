//! The ControlOracle invariants live in P1: AUTHORITY-UNIQUE and
//! INPUT-CONSERVATION (`docs/design/test_harness.md` §oracles). Ground truth comes
//! from [`crate::topology::InspectReport`]s — the directory, node-reported
//! held-sets, shard input logs, and client sent-logs. Oracles AUDIT; they never
//! trust a single node's claim about another.

use std::collections::BTreeMap;

use vd_core::{EntityId, NodeId, SessionId};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey};

use crate::topology::InspectReport;

/// AUTHORITY-UNIQUE failed: an entity without EXACTLY one owner.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AuthorityViolation {
    #[error("entity {entity} is held by {holders:?} — exactly one holder required")]
    WrongHolderCount {
        entity: EntityId,
        holders: Vec<NodeId>,
    },
    #[error("entity {entity} held by {holder} but the directory records {recorded}")]
    DirectoryDisagrees {
        entity: EntityId,
        holder: NodeId,
        recorded: String,
    },
    #[error("entity {entity} held by {holder} has NO directory record (zero-owner)")]
    Unrecorded { entity: EntityId, holder: NodeId },
    #[error("directory entity {entity} (owner {recorded}) is held by no live node")]
    HeldNowhere { entity: EntityId, recorded: String },
    #[error("entity {entity} is still pending at {node} after the run settled")]
    UnsettledPending { entity: EntityId, node: NodeId },
}

/// AUTHORITY-UNIQUE (binding P1, checked every committed tick): every entity in
/// any held-set or directory record has EXACTLY one holder, and the directory
/// agrees with it. `len == 1` exactly — `len > 1` is split-brain, `len == 0` is a
/// zero-owner orphan; both are violations (`<= 1` would hide orphans).
///
/// In P1 `reports` come from one orchestrator + one shard; the audit is written
/// for ANY number of each (P2 adds shards, nothing here changes).
///
/// # Errors
/// The first [`AuthorityViolation`] found, in deterministic entity order.
pub fn verify_authority_unique(
    reports: &[(NodeId, InspectReport)],
) -> Result<(), AuthorityViolation> {
    // Who CLAIMS to hold each entity.
    let mut holders: BTreeMap<EntityId, Vec<NodeId>> = BTreeMap::new();
    // Who awaits a grant confirmation for each entity (the legal in-flight window).
    let mut pending: BTreeMap<EntityId, Vec<NodeId>> = BTreeMap::new();
    // What the directory RECORDS for each entity key.
    let mut recorded: BTreeMap<EntityId, AuthorityRef> = BTreeMap::new();
    let mut departing: BTreeMap<EntityId, Vec<NodeId>> = BTreeMap::new();
    for (node, report) in reports {
        for entity in &report.held_entities {
            holders.entry(*entity).or_default().push(*node);
        }
        for entity in &report.pending_entities {
            pending.entry(*entity).or_default().push(*node);
        }
        for entity in &report.departing_entities {
            departing.entry(*entity).or_default().push(*node);
        }
        for (key, record) in &report.directory {
            if let DirectoryKey::Entity(entity) = key {
                recorded.insert(*entity, record.authority);
            }
        }
    }

    // Every held entity: exactly one holder, and the directory names it.
    for (entity, holding_nodes) in &holders {
        if holding_nodes.len() != 1 {
            return Err(AuthorityViolation::WrongHolderCount {
                entity: *entity,
                holders: holding_nodes.clone(),
            });
        }
        let holder = holding_nodes[0];
        match recorded.get(entity) {
            None => {
                // Legal ONLY while the holder is releasing it (revoke recorded
                // at the directory, confirmation in flight back).
                let releasing = departing
                    .get(entity)
                    .is_some_and(|nodes| nodes.contains(&holder));
                if !releasing {
                    return Err(AuthorityViolation::Unrecorded {
                        entity: *entity,
                        holder,
                    });
                }
            }
            Some(AuthorityRef::Shard(node)) if *node == holder => {}
            Some(other) => {
                return Err(AuthorityViolation::DirectoryDisagrees {
                    entity: *entity,
                    holder,
                    recorded: format!("{other:?}"),
                });
            }
        }
    }

    // Every directory-recorded entity is held SOMEWHERE — or its grant
    // confirmation is still in flight TOWARD THE RECORDED OWNER (the directory
    // commit precedes the owner's knowledge by one delivery; an entity pending
    // anywhere else is NOT excused).
    for (entity, authority) in &recorded {
        if holders.contains_key(entity) {
            continue;
        }
        let in_flight_to_owner = pending.get(entity).is_some_and(|nodes| {
            nodes
                .iter()
                .any(|node| *authority == AuthorityRef::Shard(*node))
        });
        if !in_flight_to_owner {
            return Err(AuthorityViolation::HeldNowhere {
                entity: *entity,
                recorded: format!("{authority:?}"),
            });
        }
    }
    Ok(())
}

/// The settled form: after a quiesce window NOTHING may remain pending — every
/// requested grant has resolved, and AUTHORITY-UNIQUE holds exactly.
///
/// # Errors
/// [`AuthorityViolation::UnsettledPending`] for any lingering pending entity,
/// or whatever [`verify_authority_unique`] finds.
pub fn verify_authority_settled(
    reports: &[(NodeId, InspectReport)],
) -> Result<(), AuthorityViolation> {
    for (node, report) in reports {
        if let Some(entity) = report
            .pending_entities
            .first()
            .or(report.departing_entities.first())
        {
            return Err(AuthorityViolation::UnsettledPending {
                entity: *entity,
                node: *node,
            });
        }
    }
    verify_authority_unique(reports)
}

/// INPUT-CONSERVATION failed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConservationViolation {
    #[error("input (session {session}, seq {seq}) applied {count} times — exactly once allowed")]
    AppliedTwice {
        session: SessionId,
        seq: u64,
        count: usize,
    },
    #[error("applied seqs for session {session} are not strictly increasing: {seqs:?}")]
    NonMonotonicApply { session: SessionId, seqs: Vec<u64> },
    #[error(
        "sent input (session {session}, seq {seq}) is unaccounted: neither applied \
         nor discarded-with-reason"
    )]
    Unaccounted { session: SessionId, seq: u64 },
    #[error("applied input (session {session}, seq {seq}) was never sent by any client")]
    Phantom { session: SessionId, seq: u64 },
}

/// INPUT-CONSERVATION (binding P1, zero-fault form): every input a client SENT is
/// accounted for exactly once across all shards — applied once, or discarded with
/// a typed reason — and per-session applied seqs are strictly increasing. Phantom
/// applies (never sent) are violations too. Call after a quiesce window so no
/// input is still in flight.
///
/// # Errors
/// The first [`ConservationViolation`] found, in deterministic order.
pub fn verify_input_conservation(
    reports: &[(NodeId, InspectReport)],
) -> Result<(), ConservationViolation> {
    let mut sent: BTreeMap<(SessionId, u64), usize> = BTreeMap::new();
    let mut applied: BTreeMap<(SessionId, u64), usize> = BTreeMap::new();
    let mut discarded_with_seq: BTreeMap<(SessionId, u64), usize> = BTreeMap::new();
    let mut applied_order: BTreeMap<SessionId, Vec<u64>> = BTreeMap::new();

    for (_, report) in reports {
        for key in &report.sent_inputs {
            *sent.entry(*key).or_default() += 1;
        }
        for key in &report.applied_inputs {
            *applied.entry(*key).or_default() += 1;
            applied_order.entry(key.0).or_default().push(key.1);
        }
        for (session, seq, _) in &report.discarded_inputs {
            if let Some(seq) = seq {
                *discarded_with_seq.entry((*session, *seq)).or_default() += 1;
            }
        }
    }

    // Exactly-once application + strict per-session monotonicity.
    for ((session, seq), count) in &applied {
        if *count != 1 {
            return Err(ConservationViolation::AppliedTwice {
                session: *session,
                seq: *seq,
                count: *count,
            });
        }
        if !sent.contains_key(&(*session, *seq)) {
            return Err(ConservationViolation::Phantom {
                session: *session,
                seq: *seq,
            });
        }
    }
    for (session, seqs) in &applied_order {
        let strictly_increasing = seqs.windows(2).all(|w| w[0] < w[1]);
        if !strictly_increasing {
            return Err(ConservationViolation::NonMonotonicApply {
                session: *session,
                seqs: seqs.clone(),
            });
        }
    }
    // Every sent input is applied or discarded-with-its-seq. (Reasonless discards
    // — malformed/unknown-session — carry no seq and cannot account for a sent
    // input; they cover injected garbage, not script traffic.)
    for (session, seq) in sent.keys() {
        let accounted = applied.contains_key(&(*session, *seq))
            || discarded_with_seq.contains_key(&(*session, *seq));
        if !accounted {
            return Err(ConservationViolation::Unaccounted {
                session: *session,
                seq: *seq,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::{Fence, UniverseTick};
    use vd_sim::stub::DiscardReason;
    use vd_wire::seams::directory::OwnerRecord;

    const ORCH: NodeId = NodeId(1);
    const SHARD: NodeId = NodeId(2);
    const CLIENT: NodeId = NodeId(3);
    const SESSION: SessionId = SessionId(7);

    fn entity() -> EntityId {
        EntityId(42)
    }

    fn record(node: NodeId) -> OwnerRecord {
        OwnerRecord {
            authority: AuthorityRef::Shard(node),
            fence: Fence(1),
            lease_expires: UniverseTick(100),
            in_transfer: None,
        }
    }

    fn healthy() -> Vec<(NodeId, InspectReport)> {
        vec![
            (
                ORCH,
                InspectReport {
                    directory: vec![(DirectoryKey::Entity(entity()), record(SHARD))],
                    ..InspectReport::default()
                },
            ),
            (
                SHARD,
                InspectReport {
                    held_entities: vec![entity()],
                    applied_inputs: vec![(SESSION, 1), (SESSION, 2)],
                    discarded_inputs: vec![
                        (SESSION, Some(3), DiscardReason::DuplicateSeq),
                        // A seq-less discard (injected garbage): accounts for nothing.
                        (SESSION, None, DiscardReason::MalformedInput),
                    ],
                    ..InspectReport::default()
                },
            ),
            (
                CLIENT,
                InspectReport {
                    sent_inputs: vec![(SESSION, 1), (SESSION, 2), (SESSION, 3)],
                    ..InspectReport::default()
                },
            ),
        ]
    }

    #[test]
    fn healthy_topology_passes_both_oracles() {
        let reports = healthy();
        assert_eq!(verify_authority_unique(&reports), Ok(()));
        assert_eq!(verify_input_conservation(&reports), Ok(()));
    }

    #[test]
    fn split_brain_is_caught() {
        let mut reports = healthy();
        reports.push((
            NodeId(9),
            InspectReport {
                held_entities: vec![entity()],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::WrongHolderCount {
                entity: entity(),
                holders: vec![SHARD, NodeId(9)],
            })
        );
    }

    #[test]
    fn zero_owner_is_caught_in_both_directions() {
        // Held but unrecorded.
        let mut reports = healthy();
        reports[0].1.directory.clear();
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::Unrecorded {
                entity: entity(),
                holder: SHARD,
            })
        );
        // Recorded but held nowhere.
        let mut reports = healthy();
        reports[1].1.held_entities.clear();
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::HeldNowhere {
                entity: entity(),
                recorded: format!("{:?}", AuthorityRef::Shard(SHARD)),
            })
        );
    }

    #[test]
    fn in_flight_grants_are_legal_but_misdirected_or_settled_pending_is_not() {
        // Recorded in the directory, pending at the RECORDED owner: legal window.
        let mut reports = healthy();
        reports[1].1.held_entities.clear();
        reports[1].1.pending_entities = vec![entity()];
        assert_eq!(verify_authority_unique(&reports), Ok(()));
        // The settled check refuses the same state after quiesce.
        assert_eq!(
            verify_authority_settled(&reports),
            Err(AuthorityViolation::UnsettledPending {
                entity: entity(),
                node: SHARD,
            })
        );
        // Pending at a node that is NOT the recorded owner excuses nothing.
        let mut reports = healthy();
        reports[1].1.held_entities.clear();
        reports.push((
            NodeId(9),
            InspectReport {
                pending_entities: vec![entity()],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::HeldNowhere {
                entity: entity(),
                recorded: format!("{:?}", AuthorityRef::Shard(SHARD)),
            })
        );
        // And the settled check passes a fully-held topology.
        assert_eq!(verify_authority_settled(&healthy()), Ok(()));
    }

    #[test]
    fn the_release_window_is_legal_until_settled() {
        // Directory already cleared (revoke committed), holder still finishing:
        // held + departing at the SAME node, no record — legal in flight.
        let mut reports = healthy();
        reports[0].1.directory.clear();
        reports[1].1.departing_entities = vec![entity()];
        assert_eq!(verify_authority_unique(&reports), Ok(()));
        // But it may not survive the settle.
        assert_eq!(
            verify_authority_settled(&reports),
            Err(AuthorityViolation::UnsettledPending {
                entity: entity(),
                node: SHARD,
            })
        );
        // Departing at a DIFFERENT node excuses nothing.
        let mut reports = healthy();
        reports[0].1.directory.clear();
        reports.push((
            NodeId(9),
            InspectReport {
                departing_entities: vec![entity()],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::Unrecorded {
                entity: entity(),
                holder: SHARD,
            })
        );
    }

    #[test]
    fn directory_disagreement_is_caught() {
        let mut reports = healthy();
        reports[0].1.directory = vec![(DirectoryKey::Entity(entity()), record(NodeId(9)))];
        assert_eq!(
            verify_authority_unique(&reports),
            Err(AuthorityViolation::DirectoryDisagrees {
                entity: entity(),
                holder: SHARD,
                recorded: format!("{:?}", AuthorityRef::Shard(NodeId(9))),
            })
        );
        // A gateway-recorded ENTITY is equally a disagreement for a shard holder.
        let mut reports = healthy();
        reports[0].1.directory = vec![(
            DirectoryKey::Entity(entity()),
            OwnerRecord {
                authority: AuthorityRef::Gateway(NodeId(9)),
                ..record(NodeId(9))
            },
        )];
        let result = verify_authority_unique(&reports);
        assert_eq!(
            result,
            Err(AuthorityViolation::DirectoryDisagrees {
                entity: entity(),
                holder: SHARD,
                recorded: format!("{:?}", AuthorityRef::Gateway(NodeId(9))),
            })
        );
    }

    #[test]
    fn non_entity_directory_keys_are_ignored_by_authority_unique() {
        let mut reports = healthy();
        reports[0].1.directory.push((
            DirectoryKey::Session(SESSION),
            OwnerRecord {
                authority: AuthorityRef::Gateway(NodeId(5)),
                ..record(NodeId(5))
            },
        ));
        assert_eq!(verify_authority_unique(&reports), Ok(()));
    }

    #[test]
    fn double_apply_and_phantom_and_unaccounted_are_caught() {
        // Applied twice across two shards.
        let mut reports = healthy();
        reports.push((
            NodeId(9),
            InspectReport {
                applied_inputs: vec![(SESSION, 2)],
                ..InspectReport::default()
            },
        ));
        assert_eq!(
            verify_input_conservation(&reports),
            Err(ConservationViolation::AppliedTwice {
                session: SESSION,
                seq: 2,
                count: 2,
            })
        );
        // Phantom: applied but never sent.
        let mut reports = healthy();
        reports[2].1.sent_inputs = vec![(SESSION, 1), (SESSION, 3)];
        assert_eq!(
            verify_input_conservation(&reports),
            Err(ConservationViolation::Phantom {
                session: SESSION,
                seq: 2,
            })
        );
        // Unaccounted: sent but neither applied nor seq-discarded.
        let mut reports = healthy();
        reports[2].1.sent_inputs.push((SESSION, 4));
        assert_eq!(
            verify_input_conservation(&reports),
            Err(ConservationViolation::Unaccounted {
                session: SESSION,
                seq: 4,
            })
        );
    }

    #[test]
    fn non_monotonic_application_is_caught() {
        let mut reports = healthy();
        reports[1].1.applied_inputs = vec![(SESSION, 2), (SESSION, 1)];
        reports[2].1.sent_inputs = vec![(SESSION, 1), (SESSION, 2), (SESSION, 3)];
        reports[1].1.discarded_inputs = vec![(SESSION, Some(3), DiscardReason::DuplicateSeq)];
        assert_eq!(
            verify_input_conservation(&reports),
            Err(ConservationViolation::NonMonotonicApply {
                session: SESSION,
                seqs: vec![2, 1],
            })
        );
    }

    #[test]
    fn violations_display_for_failure_messages() {
        assert_eq!(
            AuthorityViolation::Unrecorded {
                entity: entity(),
                holder: SHARD,
            }
            .to_string(),
            format!(
                "entity {} held by {} has NO directory record (zero-owner)",
                entity(),
                SHARD
            )
        );
        assert_eq!(
            AuthorityViolation::WrongHolderCount {
                entity: entity(),
                holders: vec![SHARD, NodeId(9)],
            }
            .to_string(),
            format!(
                "entity {} is held by [NodeId(2), NodeId(9)] — exactly one holder required",
                entity()
            )
        );
        assert_eq!(
            AuthorityViolation::DirectoryDisagrees {
                entity: entity(),
                holder: SHARD,
                recorded: "X".to_owned(),
            }
            .to_string(),
            format!(
                "entity {} held by {} but the directory records X",
                entity(),
                SHARD
            )
        );
        assert_eq!(
            AuthorityViolation::HeldNowhere {
                entity: entity(),
                recorded: "X".to_owned(),
            }
            .to_string(),
            format!(
                "directory entity {} (owner X) is held by no live node",
                entity()
            )
        );
        assert_eq!(
            ConservationViolation::AppliedTwice {
                session: SESSION,
                seq: 2,
                count: 3,
            }
            .to_string(),
            format!("input (session {SESSION}, seq 2) applied 3 times — exactly once allowed")
        );
        assert_eq!(
            ConservationViolation::NonMonotonicApply {
                session: SESSION,
                seqs: vec![2, 1],
            }
            .to_string(),
            format!("applied seqs for session {SESSION} are not strictly increasing: [2, 1]")
        );
        assert_eq!(
            ConservationViolation::Phantom {
                session: SESSION,
                seq: 2,
            }
            .to_string(),
            format!("applied input (session {SESSION}, seq 2) was never sent by any client")
        );
        assert_eq!(
            AuthorityViolation::UnsettledPending {
                entity: entity(),
                node: SHARD,
            }
            .to_string(),
            format!(
                "entity {} is still pending at {} after the run settled",
                entity(),
                SHARD
            )
        );
        assert_eq!(
            ConservationViolation::Unaccounted {
                session: SESSION,
                seq: 4,
            }
            .to_string(),
            format!(
                "sent input (session {SESSION}, seq 4) is unaccounted: neither applied \
                 nor discarded-with-reason"
            )
        );
    }
}
