//! `vd-devproto` — the HR6 dev-control protocol leaf. Tiny, pure, Tier-A.
//!
//! Slice 0 scope: the dev-cluster [`DevPortScheme`] — the single source of truth
//! mapping a (worktree `slot`, client `agent`) to the deterministic localhost
//! ports the local-process dev cluster and `vdctl` use — plus the client-NodeId
//! convention ([`CLIENT_NODE_BASE`]) and the worktree→slot derivation
//! ([`slot_for_worktree`]). The `DevRequest`/`DevResponse` codec, the `wait-until`
//! predicate, and the `DevState` schema land in later P1.5 slices.
//!
//! ## Why a fixed scheme, not ephemeral ports
//! A dev cluster must use DETERMINISTIC ports so the agent's `vdctl` calls and
//! bash scenarios reach every node without a discovery step. Yet several
//! worktrees (`new-system`/`hud`/`rails`/…) and several client windows per
//! worktree (P2's *client2 screenshots client1 crossing a boundary*) run at once
//! and must never collide. The scheme gives each worktree `slot` a **disjoint
//! contiguous block** of ports; within a block the cluster nodes take fixed
//! offsets, then each dev-control client takes one port (`agent` in `0..K`), then
//! each client's QUIC link to the gateway takes one (`agent` in `0..K`). Disjoint
//! blocks make a cross-slot collision structurally impossible, and `K`
//! (`max_clients_per_worktree`) bounds the per-slot client windows.
//!
//! Per-slot block layout (offsets from the slot's base port):
//! ```text
//!   0 orchestrator (QUIC)   1 gateway (QUIC)   2 shard (QUIC)   3 admin (HTTP)
//!   4 .. 4+K   dev-control[agent]   (vdctl ↔ client, TCP)
//!   4+K .. 4+2K   client-quic[agent]  (client ↔ gateway, QUIC)
//! ```
//! All port math lives HERE (covered) and is surfaced to bash through the
//! `vd-slot` helper — never hand-computed in a shell script.
//!
//! ## Dev-control protocol (Slice 2)
//! The `vdctl` ↔ client command protocol — [`dispatch`] (DevRequest/DevResponse +
//! the pure `InputAction` seam), [`codec`] (the ONE JSON-lines codec), [`predicate`]
//! (the structured `wait-until`), and [`state`] (the `DevState` diagnosis substrate).
//! Pure data + serde; the actual socket lives in the Tier-B client-bin.

pub mod codec;
pub mod dispatch;
pub mod predicate;
pub mod state;

pub use codec::{decode_request, encode_response};
pub use dispatch::{DevError, DevRequest, DevResponse, InputAction, MAX_ACTION_INDEX, action_bit};
pub use predicate::{WaitField, WaitOp, WaitPredicate};
pub use state::{DevEntityRow, DevPhase, DevState, DevTransferView};

/// Ports reserved at the front of every slot block for the fixed cluster nodes
/// (orchestrator, gateway, shard, admin) ahead of the per-client port bands.
const RESERVED_NODE_PORTS: u16 = 4;

const ORCHESTRATOR_OFFSET: u16 = 0;
const GATEWAY_OFFSET: u16 = 1;
const SHARD_OFFSET: u16 = 2;
const ADMIN_OFFSET: u16 = 3;

/// The base `NodeId` for dev-control clients: client `agent` is `NodeId(BASE +
/// agent)`. ONE source of truth for the launcher (which seeds these into the
/// gateway's address book), the client, and `vd-slot`. Matches the 100/101
/// convention the process-tier parity test established.
pub const CLIENT_NODE_BASE: u64 = 100;

/// The worktree→slot ceiling: `slot_for_worktree` returns `0..CEILING`. Chosen so
/// even the highest derived slot's block stays below the range Docker grabs on
/// macOS (~10000+): base 7000 + 63·32 + 32 = 9048 < 10000.
pub const WORKTREE_SLOT_CEILING: u16 = 64;

/// The deterministic dev-cluster port scheme (HR6). ONE reviewed source of truth;
/// no port literal is ever inlined in a script or a binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevPortScheme {
    /// First port of slot 0's block.
    pub base: u16,
    /// Ports reserved per worktree slot. Must hold the [`RESERVED_NODE_PORTS`]
    /// node ports plus TWO per-client bands of `max_clients_per_worktree` ports.
    pub block_size: u16,
    /// `K` — the maximum simultaneous dev-control client windows per worktree slot.
    /// P2's two-client visual scenario needs `>= 2`; the default leaves headroom.
    pub max_clients_per_worktree: u16,
}

/// A typed, loud failure — the scheme never silently aliases two roles onto one
/// port or runs a client off the end of its slot block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PortSchemeError {
    /// `block_size` cannot hold the node ports plus the two `k`-wide client bands.
    #[error("block_size {block_size} cannot hold the node ports plus 2x{k} client ports")]
    BlockTooSmall { block_size: u16, k: u16 },
    /// `slot`'s block runs past the u16 port space.
    #[error("slot {slot} runs past the u16 port space for this scheme")]
    SlotOverflowsPortSpace { slot: u16 },
    /// `agent` is `>= K` — outside the slot's client bands.
    #[error("agent index {agent} is out of range for K={k}")]
    AgentOutOfRange { agent: u16, k: u16 },
}

/// The resolved ports for one worktree slot. Node ports are plain field reads;
/// per-client ports come from [`SlotPorts::dev_control`] / [`SlotPorts::client_quic`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotPorts {
    pub orchestrator: u16,
    pub gateway: u16,
    pub shard: u16,
    pub admin: u16,
    /// First dev-control port (`agent` 0); private — go through [`SlotPorts::dev_control`].
    dev_control_base: u16,
    /// First client-QUIC port (`agent` 0); private — go through [`SlotPorts::client_quic`].
    client_quic_base: u16,
    max_clients: u16,
}

impl DevPortScheme {
    /// The default dev scheme: 32-port blocks from 7000, `K = 4`.
    pub const DEFAULT: DevPortScheme = DevPortScheme {
        base: 7000,
        block_size: 32,
        max_clients_per_worktree: 4,
    };

    /// Internal consistency: the block must hold the node ports plus both client
    /// bands (`2·K`).
    pub fn validate(self) -> Result<(), PortSchemeError> {
        if self.block_size < RESERVED_NODE_PORTS + 2 * self.max_clients_per_worktree {
            return Err(PortSchemeError::BlockTooSmall {
                block_size: self.block_size,
                k: self.max_clients_per_worktree,
            });
        }
        Ok(())
    }

    /// First port of `slot`'s block. Computed in u32 (overflow-free for u16 inputs)
    /// then range-checked ONCE so the WHOLE block fits the u16 port space — after
    /// which every in-block offset add is overflow-free.
    fn slot_base(self, slot: u16) -> Result<u16, PortSchemeError> {
        let first = u32::from(self.base) + u32::from(slot) * u32::from(self.block_size);
        let last = first + u32::from(self.block_size); // exclusive end
        if last <= u32::from(u16::MAX) + 1 {
            Ok(first as u16)
        } else {
            Err(PortSchemeError::SlotOverflowsPortSpace { slot })
        }
    }

    /// Resolve every port for `slot`. Validates the scheme, then bounds the slot.
    pub fn slot_ports(self, slot: u16) -> Result<SlotPorts, PortSchemeError> {
        self.validate()?;
        let base = self.slot_base(slot)?;
        let k = self.max_clients_per_worktree;
        Ok(SlotPorts {
            orchestrator: base + ORCHESTRATOR_OFFSET,
            gateway: base + GATEWAY_OFFSET,
            shard: base + SHARD_OFFSET,
            admin: base + ADMIN_OFFSET,
            dev_control_base: base + RESERVED_NODE_PORTS,
            client_quic_base: base + RESERVED_NODE_PORTS + k,
            max_clients: k,
        })
    }
}

impl SlotPorts {
    /// The dev-control listen port (vdctl ↔ client) for client `agent` (`0..K`).
    pub fn dev_control(self, agent: u16) -> Result<u16, PortSchemeError> {
        Ok(self.dev_control_base + self.checked_agent(agent)?)
    }

    /// The QUIC port (client ↔ gateway) for client `agent` (`0..K`). The launcher
    /// seeds `(CLIENT_NODE_BASE + agent, 127.0.0.1:this)` into the gateway's
    /// address book so the gateway can route snapshots back to the client.
    pub fn client_quic(self, agent: u16) -> Result<u16, PortSchemeError> {
        Ok(self.client_quic_base + self.checked_agent(agent)?)
    }

    fn checked_agent(self, agent: u16) -> Result<u16, PortSchemeError> {
        if agent >= self.max_clients {
            return Err(PortSchemeError::AgentOutOfRange {
                agent,
                k: self.max_clients,
            });
        }
        Ok(agent)
    }
}

/// Derive a stable dev-cluster `slot` (`0..WORKTREE_SLOT_CEILING`) from a worktree
/// path, so two worktrees never default to the same slot and collide. A
/// deterministic FNV-1a hash (the std hashers are randomized and unusable here);
/// `--slot` always overrides if a derived collision is ever hit.
#[must_use]
pub fn slot_for_worktree(path: &str) -> u16 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for byte in path.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
    }
    (hash % u64::from(WORKTREE_SLOT_CEILING)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scheme_is_self_consistent() {
        assert_eq!(DevPortScheme::DEFAULT.base, 7000);
        assert_eq!(DevPortScheme::DEFAULT.block_size, 32);
        assert_eq!(DevPortScheme::DEFAULT.max_clients_per_worktree, 4);
        assert_eq!(DevPortScheme::DEFAULT.validate(), Ok(()));
    }

    #[test]
    fn slot_zero_resolves_nodes_then_dev_control_then_client_quic_bands() {
        let p = DevPortScheme::DEFAULT.slot_ports(0).expect("valid slot");
        assert_eq!(p.orchestrator, 7000);
        assert_eq!(p.gateway, 7001);
        assert_eq!(p.shard, 7002);
        assert_eq!(p.admin, 7003);
        // The dev-control band follows the 4 node ports…
        assert_eq!(p.dev_control(0), Ok(7004));
        assert_eq!(p.dev_control(3), Ok(7007));
        // …then the client-QUIC band follows the K dev-control ports.
        assert_eq!(p.client_quic(0), Ok(7008));
        assert_eq!(p.client_quic(3), Ok(7011));
    }

    #[test]
    fn slots_occupy_disjoint_blocks() {
        let s0 = DevPortScheme::DEFAULT.slot_ports(0).expect("s0");
        let s1 = DevPortScheme::DEFAULT.slot_ports(1).expect("s1");
        assert_eq!(s1.orchestrator, 7032, "slot 1 starts one block (32) later");
        // The highest port slot 0 hands out is below slot 1's first node port.
        assert_eq!(s0.client_quic(3), Ok(7011));
        assert!(s0.client_quic(3).expect("s0 last") < s1.orchestrator);
    }

    #[test]
    fn agent_at_or_above_k_is_a_typed_error_in_both_bands() {
        let p = DevPortScheme::DEFAULT.slot_ports(2).expect("valid");
        // Valid agents in both bands.
        assert_eq!(p.dev_control(0), Ok(7068)); // 7000 + 2*32 + 4
        assert_eq!(p.client_quic(0), Ok(7072)); // …+ K
        // Out of range in EACH band (covers checked_agent via both callers).
        assert_eq!(
            p.dev_control(4),
            Err(PortSchemeError::AgentOutOfRange { agent: 4, k: 4 })
        );
        assert_eq!(
            p.client_quic(9),
            Err(PortSchemeError::AgentOutOfRange { agent: 9, k: 4 })
        );
    }

    #[test]
    fn a_block_too_small_for_both_bands_is_rejected() {
        // 4 node ports + 2*4 client ports needs >= 12; block_size 11 fails.
        let scheme = DevPortScheme {
            base: 7000,
            block_size: 11,
            max_clients_per_worktree: 4,
        };
        assert_eq!(
            scheme.validate(),
            Err(PortSchemeError::BlockTooSmall {
                block_size: 11,
                k: 4
            })
        );
        // slot_ports propagates the validation failure before computing anything.
        assert_eq!(
            scheme.slot_ports(0),
            Err(PortSchemeError::BlockTooSmall {
                block_size: 11,
                k: 4
            })
        );
        // The exact-fit boundary (block_size == 4 + 2K) is valid.
        let exact = DevPortScheme {
            base: 7000,
            block_size: 12,
            max_clients_per_worktree: 4,
        };
        assert_eq!(exact.validate(), Ok(()));
    }

    #[test]
    fn a_slot_whose_block_overflows_the_port_space_is_rejected() {
        // base 7000, block 32: slot 1829 ends at 7000 + 1829*32 + 32 = 65560 > 65536.
        assert_eq!(
            DevPortScheme::DEFAULT.slot_ports(1829),
            Err(PortSchemeError::SlotOverflowsPortSpace { slot: 1829 })
        );
        // The highest slot that still fits resolves cleanly (boundary, not overflow).
        let last = DevPortScheme::DEFAULT
            .slot_ports(1828)
            .expect("last fitting slot");
        assert_eq!(last.orchestrator, 65496); // 7000 + 1828*32
        assert_eq!(last.client_quic(3), Ok(65507));
    }

    #[test]
    fn worktree_slot_is_stable_distinct_and_within_the_docker_safe_ceiling() {
        let a = slot_for_worktree("/Users/x/voxeldust/.claude/worktrees/new-system");
        let b = slot_for_worktree("/Users/x/voxeldust/.claude/worktrees/hud");
        // Stable: same path → same slot.
        assert_eq!(
            a,
            slot_for_worktree("/Users/x/voxeldust/.claude/worktrees/new-system")
        );
        // Distinct worktrees map to distinct slots (these two do).
        assert_ne!(a, b);
        // Always inside the Docker-safe ceiling (covers the empty-path edge too).
        assert!(a < WORKTREE_SLOT_CEILING);
        assert!(b < WORKTREE_SLOT_CEILING);
        assert!(slot_for_worktree("") < WORKTREE_SLOT_CEILING);
        // The highest derived slot's block stays clear of Docker's 10000+ range.
        let ceiling_block = DevPortScheme::DEFAULT
            .slot_ports(WORKTREE_SLOT_CEILING - 1)
            .expect("ceiling slot fits");
        assert!(ceiling_block.client_quic(3).expect("port") < 10000);
    }

    #[test]
    fn errors_render_for_the_operator() {
        let too_small = PortSchemeError::BlockTooSmall {
            block_size: 11,
            k: 4,
        };
        let overflow = PortSchemeError::SlotOverflowsPortSpace { slot: 1829 };
        let agent = PortSchemeError::AgentOutOfRange { agent: 4, k: 4 };
        assert_eq!(
            too_small.to_string(),
            "block_size 11 cannot hold the node ports plus 2x4 client ports"
        );
        assert_eq!(
            overflow.to_string(),
            "slot 1829 runs past the u16 port space for this scheme"
        );
        assert_eq!(agent.to_string(), "agent index 4 is out of range for K=4");
        assert_eq!(
            format!("{too_small:?}"),
            "BlockTooSmall { block_size: 11, k: 4 }"
        );
        assert_ne!(overflow, agent);
        assert_eq!(CLIENT_NODE_BASE, 100);
    }
}
