//! Core signal types — values, scopes, access policies, properties, merge strategies.

/// A signal value on a channel. Lightweight, copyable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SignalValue {
    /// Boolean state (on/off, open/closed).
    Bool(bool),
    /// Continuous value (throttle 0.0–1.0, pressure kPa, angle degrees).
    Float(f32),
    /// Discrete state index (junction branch 0/1/2, mode selector).
    State(u8),
}

impl Default for SignalValue {
    fn default() -> Self {
        Self::Float(0.0)
    }
}

impl SignalValue {
    /// Extract as f32 for arithmetic operations. Bool → 0.0/1.0, State → cast.
    pub fn as_f32(self) -> f32 {
        match self {
            Self::Bool(b) => if b { 1.0 } else { 0.0 },
            Self::Float(f) => f,
            Self::State(s) => s as f32,
        }
    }

    /// Extract as bool. Float > 0.5 → true, State > 0 → true.
    pub fn as_bool(self) -> bool {
        match self {
            Self::Bool(b) => b,
            Self::Float(f) => f > 0.5,
            Self::State(s) => s > 0,
        }
    }
}

/// Signal scope — determines transport and reach.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SignalScope {
    /// Ship/structure-internal only. Zero network cost.
    Local,
    /// Broadcasts to structures within proximity. QUIC to nearby shards.
    ShortRange { range_m: f64 },
    /// System-wide broadcast. Requires antenna block + power. System shard relay.
    LongRange,
    /// Galaxy-wide, topic-routed via Galaxy Relay Shard. Requires antenna block.
    /// Until the relay shard exists, travels the same path as LongRange but is
    /// tagged separately (scope code 3) so it can be routed differently later.
    Radio { frequency: u32 },
}

impl Default for SignalScope {
    fn default() -> Self {
        Self::Local
    }
}

/// Who can publish to or subscribe from a channel.
#[derive(Clone, Debug, PartialEq)]
pub enum AccessPolicy {
    /// Only the channel owner (ship owner / block placer).
    OwnerOnly,
    /// Specific player IDs allowed.
    AllowList(Vec<u64>),
    /// Anyone can access.
    Public,
}

impl Default for AccessPolicy {
    fn default() -> Self {
        Self::OwnerOnly
    }
}

impl AccessPolicy {
    /// Is `sender_id` allowed under this policy? `owner_id` is the
    /// channel owner (used by `OwnerOnly`).
    pub fn allows(&self, sender_id: u64, owner_id: u64) -> bool {
        match self {
            Self::OwnerOnly => sender_id == owner_id,
            Self::AllowList(list) => list.contains(&sender_id),
            Self::Public => true,
        }
    }
}

/// Which property of a functional block is read/written by a signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SignalProperty {
    /// Boolean: on/off state.
    Active,
    /// Float 0.0–1.0: power/throttle level.
    Throttle,
    /// Float: angle in degrees (rotor target).
    Angle,
    /// Float 0.0–1.0: piston extension.
    Extension,
    /// Float: pressure reading (kPa).
    Pressure,
    /// Float: speed reading (m/s).
    Speed,
    /// Float 0.0–1.0: fill/charge level (battery).
    Level,
    /// u8: discrete switch state (junction branch index).
    SwitchState,
    /// Float: thrust boost multiplier (1.0 = normal). Set by cruise drives.
    Boost,
    /// Float: mechanical status code (0=Idle, 1=Moving, 2=Blocked, 3=Error).
    Status,
    /// Text: server-authored display string (body names, warp
    /// targets, ship callsigns). Only valid on the wire inside
    /// `HudSignalValue::Text`; core simulation channels stay numeric.
    Text,
}

impl SignalProperty {
    /// Stable u8 ordinal used by the HudSignalEntry wire protocol.
    /// Adding variants MUST keep existing ordinals stable.
    pub fn as_ordinal(self) -> u8 {
        match self {
            Self::Active => 0,
            Self::Throttle => 1,
            Self::Angle => 2,
            Self::Extension => 3,
            Self::Pressure => 4,
            Self::Speed => 5,
            Self::Level => 6,
            Self::SwitchState => 7,
            Self::Boost => 8,
            Self::Status => 9,
            Self::Text => 10,
        }
    }

    pub fn from_ordinal(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Active,
            1 => Self::Throttle,
            2 => Self::Angle,
            3 => Self::Extension,
            4 => Self::Pressure,
            5 => Self::Speed,
            6 => Self::Level,
            7 => Self::SwitchState,
            8 => Self::Boost,
            9 => Self::Status,
            10 => Self::Text,
            _ => return None,
        })
    }
}

/// How multiple publishers to the same channel merge their values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelMergeStrategy {
    /// Last value written wins (default for single-publisher channels).
    LastWrite,
    /// Sum all published values (e.g., total thrust from multiple thrusters).
    Sum,
    /// Average of all published values.
    Average,
    /// Maximum of all published values.
    Max,
    /// Minimum of all published values.
    Min,
    /// Boolean OR: true if any publisher writes true.
    AnyTrue,
    /// Boolean AND: true only if ALL publishers write true.
    AllTrue,
}

impl Default for ChannelMergeStrategy {
    fn default() -> Self {
        Self::LastWrite
    }
}

impl ChannelMergeStrategy {
    /// Merge a list of pending values into a single result.
    pub fn merge(self, values: &[SignalValue]) -> SignalValue {
        if values.is_empty() {
            return SignalValue::default();
        }
        match self {
            Self::LastWrite => *values.last().unwrap(),
            Self::Sum => {
                let total: f32 = values.iter().map(|v| v.as_f32()).sum();
                SignalValue::Float(total)
            }
            Self::Average => {
                let total: f32 = values.iter().map(|v| v.as_f32()).sum();
                SignalValue::Float(total / values.len() as f32)
            }
            Self::Max => {
                let max = values.iter().map(|v| v.as_f32()).fold(f32::NEG_INFINITY, f32::max);
                SignalValue::Float(max)
            }
            Self::Min => {
                let min = values.iter().map(|v| v.as_f32()).fold(f32::INFINITY, f32::min);
                SignalValue::Float(min)
            }
            Self::AnyTrue => {
                SignalValue::Bool(values.iter().any(|v| v.as_bool()))
            }
            Self::AllTrue => {
                SignalValue::Bool(values.iter().all(|v| v.as_bool()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_value_conversions() {
        assert_eq!(SignalValue::Bool(true).as_f32(), 1.0);
        assert_eq!(SignalValue::Bool(false).as_f32(), 0.0);
        assert_eq!(SignalValue::Float(0.7).as_bool(), true);
        assert_eq!(SignalValue::Float(0.3).as_bool(), false);
        assert_eq!(SignalValue::State(3).as_f32(), 3.0);
        assert_eq!(SignalValue::State(0).as_bool(), false);
    }

    #[test]
    fn merge_sum() {
        let values = vec![SignalValue::Float(10.0), SignalValue::Float(20.0), SignalValue::Float(30.0)];
        assert_eq!(ChannelMergeStrategy::Sum.merge(&values), SignalValue::Float(60.0));
    }

    #[test]
    fn merge_average() {
        let values = vec![SignalValue::Float(10.0), SignalValue::Float(20.0)];
        assert_eq!(ChannelMergeStrategy::Average.merge(&values), SignalValue::Float(15.0));
    }

    #[test]
    fn merge_max_min() {
        let values = vec![SignalValue::Float(5.0), SignalValue::Float(15.0), SignalValue::Float(10.0)];
        assert_eq!(ChannelMergeStrategy::Max.merge(&values), SignalValue::Float(15.0));
        assert_eq!(ChannelMergeStrategy::Min.merge(&values), SignalValue::Float(5.0));
    }

    #[test]
    fn merge_any_true() {
        let values = vec![SignalValue::Bool(false), SignalValue::Bool(true), SignalValue::Bool(false)];
        assert_eq!(ChannelMergeStrategy::AnyTrue.merge(&values), SignalValue::Bool(true));

        let all_false = vec![SignalValue::Bool(false), SignalValue::Bool(false)];
        assert_eq!(ChannelMergeStrategy::AnyTrue.merge(&all_false), SignalValue::Bool(false));
    }

    #[test]
    fn merge_all_true() {
        let values = vec![SignalValue::Bool(true), SignalValue::Bool(true)];
        assert_eq!(ChannelMergeStrategy::AllTrue.merge(&values), SignalValue::Bool(true));

        let mixed = vec![SignalValue::Bool(true), SignalValue::Bool(false)];
        assert_eq!(ChannelMergeStrategy::AllTrue.merge(&mixed), SignalValue::Bool(false));
    }

    #[test]
    fn merge_last_write() {
        let values = vec![SignalValue::Float(1.0), SignalValue::Float(2.0), SignalValue::Float(3.0)];
        assert_eq!(ChannelMergeStrategy::LastWrite.merge(&values), SignalValue::Float(3.0));
    }

    #[test]
    fn merge_empty() {
        assert_eq!(ChannelMergeStrategy::Sum.merge(&[]), SignalValue::default());
    }
}
