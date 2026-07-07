//! Bin-shell support: typed environment configuration and the real-time tick
//! pacer. Lives HERE (not in the 4-line bins) so every line is testable — bins
//! stay thin shells excluded from coverage, this module does not.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use vd_core::NodeId;

/// Configuration errors are operator errors: typed, loud, at boot.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    #[error("missing required variable {0}")]
    Missing(String),
    #[error("variable {key} has unparseable value {value:?}")]
    Unparseable { key: String, value: String },
}

/// A typed view over environment-style key/value config.
pub struct EnvConfig {
    vars: BTreeMap<String, String>,
}

impl EnvConfig {
    #[must_use]
    pub fn new(vars: BTreeMap<String, String>) -> EnvConfig {
        EnvConfig { vars }
    }

    /// Snapshot the real process environment.
    #[must_use]
    pub fn from_process_env() -> EnvConfig {
        EnvConfig {
            vars: std::env::vars().collect(),
        }
    }

    fn raw(&self, key: &str) -> Result<&str, ConfigError> {
        self.vars
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| ConfigError::Missing(key.to_owned()))
    }

    /// A required string value.
    ///
    /// # Errors
    /// [`ConfigError::Missing`].
    pub fn string(&self, key: &str) -> Result<String, ConfigError> {
        self.raw(key).map(str::to_owned)
    }

    /// A required parseable value (u64, f64, SocketAddr, …).
    ///
    /// # Errors
    /// [`ConfigError`] on absence or parse failure.
    pub fn parse<T: std::str::FromStr>(&self, key: &str) -> Result<T, ConfigError> {
        let value = self.raw(key)?;
        value.parse().map_err(|_| ConfigError::Unparseable {
            key: key.to_owned(),
            value: value.to_owned(),
        })
    }

    /// A parseable value that DEFAULTS on absence but still ERRORS on a present-but-unparseable value
    /// (a typo is never silently defaulted). For operational params with a sane built-in default
    /// (Slice 2a saga deadlines) so a deployment works out-of-the-box yet a bad override fails loud.
    ///
    /// # Errors
    /// [`ConfigError::Unparseable`] when the key is present but does not parse.
    pub fn parse_or<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, ConfigError> {
        match self.raw(key) {
            Ok(value) => value.parse().map_err(|_| ConfigError::Unparseable {
                key: key.to_owned(),
                value: value.to_owned(),
            }),
            Err(_) => Ok(default),
        }
    }

    /// A `NodeId` (plain u64).
    ///
    /// # Errors
    /// [`ConfigError`] on absence or parse failure.
    pub fn node_id(&self, key: &str) -> Result<NodeId, ConfigError> {
        self.parse::<u64>(key).map(NodeId)
    }

    /// A boolean env flag: absent ⇒ `false`; `1`/`true`/`yes` ⇒ true; `0`/`false`/`no`/empty ⇒ false;
    /// anything else is a LOUD [`ConfigError::Unparseable`] (a typo like `VD_FOO=treu` is never silently
    /// treated as false). The ONE bool-parsing home (cloud-ready k3d Slice 2 — `vd_bins::parse_bool_env`
    /// delegates here, and the cloud footgun preflight reads the ephemeral escapes through it).
    ///
    /// # Errors
    /// [`ConfigError::Unparseable`] on a present-but-non-boolean value.
    pub fn bool(&self, key: &str) -> Result<bool, ConfigError> {
        match self.raw(key) {
            Err(_) => Ok(false),
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" => Ok(true),
                "0" | "false" | "no" | "" => Ok(false),
                _ => Err(ConfigError::Unparseable {
                    key: key.to_owned(),
                    value: v.to_owned(),
                }),
            },
        }
    }

    /// The peer address book: `"1=127.0.0.1:5001,2=127.0.0.1:5002"`.
    ///
    /// # Errors
    /// [`ConfigError`] on absence or any malformed entry.
    pub fn peer_book(&self, key: &str) -> Result<BTreeMap<NodeId, SocketAddr>, ConfigError> {
        let value = self.raw(key)?;
        let mut book = BTreeMap::new();
        for entry in value.split(',').filter(|e| !e.is_empty()) {
            let Some((id, addr)) = entry.split_once('=') else {
                return Err(ConfigError::Unparseable {
                    key: key.to_owned(),
                    value: entry.to_owned(),
                });
            };
            let id: u64 = id.parse().map_err(|_| ConfigError::Unparseable {
                key: key.to_owned(),
                value: entry.to_owned(),
            })?;
            let addr: SocketAddr = addr.parse().map_err(|_| ConfigError::Unparseable {
                key: key.to_owned(),
                value: entry.to_owned(),
            })?;
            book.insert(NodeId(id), addr);
        }
        Ok(book)
    }

    /// A comma-separated NodeId list: `"2,3,4"`.
    ///
    /// # Errors
    /// [`ConfigError`] on absence or any malformed entry.
    pub fn node_list(&self, key: &str) -> Result<Vec<NodeId>, ConfigError> {
        let value = self.raw(key)?;
        let mut out = Vec::new();
        for entry in value.split(',').filter(|e| !e.is_empty()) {
            let id: u64 = entry.parse().map_err(|_| ConfigError::Unparseable {
                key: key.to_owned(),
                value: entry.to_owned(),
            })?;
            out.push(NodeId(id));
        }
        Ok(out)
    }

    /// A 32-byte hex key (auth verifying keys, seeds).
    ///
    /// # Errors
    /// [`ConfigError`] on absence, non-hex content, or wrong length.
    pub fn hex32(&self, key: &str) -> Result<[u8; 32], ConfigError> {
        let value = self.raw(key)?;
        let err = || ConfigError::Unparseable {
            key: key.to_owned(),
            value: value.to_owned(),
        };
        if value.len() != 64 {
            return Err(err());
        }
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[2 * i..2 * i + 2], 16).map_err(|_| err())?;
        }
        Ok(out)
    }
}

/// Lowercase-hex-encode bytes — the exact inverse of [`EnvConfig::hex32`]. ONE
/// encoder for every site that renders a key/seed to env or the wire (the launcher
/// and the parity test both need it). Allocation-free over the byte slice.
#[must_use]
pub fn hex32_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        // Writing to a String is infallible.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// A drift-free fixed-rate tick pacer: deadlines advance by exact periods from
/// genesis (`next += period`), so a slow tick is followed by catch-up rather than
/// permanent phase drift. Production pacing lives HERE, behind the io seam — the
/// sim thread only ever calls a blocking `wait`.
pub struct TickPacer {
    period: Duration,
    next_deadline: Instant,
}

impl TickPacer {
    #[must_use]
    pub fn new(tick_hz: u32) -> TickPacer {
        let period = Duration::from_secs(1) / tick_hz.max(1);
        TickPacer {
            period,
            next_deadline: Instant::now() + period,
        }
    }

    /// Block until the next tick deadline; returns how many FULL periods were
    /// overrun (0 on schedule — overruns are counted, never silently absorbed).
    pub fn wait(&mut self) -> u32 {
        let now = Instant::now();
        let mut overruns = 0u32;
        while now >= self.next_deadline {
            self.next_deadline += self.period;
            overruns += 1;
        }
        std::thread::sleep(self.next_deadline - Instant::now());
        self.next_deadline += self.period;
        overruns.saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(pairs: &[(&str, &str)]) -> EnvConfig {
        EnvConfig::new(
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        )
    }

    #[test]
    fn typed_accessors_parse_and_reject() {
        let env = cfg(&[
            ("VD_NODE_ID", "7"),
            ("VD_BIND", "127.0.0.1:5000"),
            ("VD_TICK_HZ", "20"),
            ("VD_SPEED", "2.5"),
            ("BAD", "xyz"),
        ]);
        assert_eq!(env.node_id("VD_NODE_ID"), Ok(NodeId(7)));
        assert_eq!(
            env.parse::<SocketAddr>("VD_BIND"),
            Ok("127.0.0.1:5000".parse().expect("addr"))
        );
        assert_eq!(env.parse::<u32>("VD_TICK_HZ"), Ok(20));
        assert_eq!(env.parse::<f64>("VD_SPEED"), Ok(2.5));
        assert_eq!(env.string("VD_BIND").as_deref(), Ok("127.0.0.1:5000"));
        assert_eq!(
            env.parse::<u64>("MISSING"),
            Err(ConfigError::Missing("MISSING".to_owned()))
        );
        assert_eq!(
            env.parse::<u64>("BAD"),
            Err(ConfigError::Unparseable {
                key: "BAD".to_owned(),
                value: "xyz".to_owned(),
            })
        );
        // The process-env constructor exists and snapshots SOMETHING.
        let _ = EnvConfig::from_process_env();
    }

    #[test]
    fn parse_or_defaults_on_absence_but_errors_on_a_bad_value() {
        // Slice 2a: the saga deadlines default on absence (the orchestrator works out-of-the-box)
        // but a present-but-unparseable override fails LOUD (never silently defaulted).
        let env = cfg(&[("GOOD", "42"), ("BAD", "xyz")]);
        assert_eq!(
            env.parse_or::<u64>("GOOD", 7),
            Ok(42),
            "present + parseable → the value"
        );
        assert_eq!(
            env.parse_or::<u64>("MISSING", 7),
            Ok(7),
            "absent → the default"
        );
        assert_eq!(
            env.parse_or::<u64>("BAD", 7),
            Err(ConfigError::Unparseable {
                key: "BAD".to_owned(),
                value: "xyz".to_owned(),
            }),
            "present but unparseable → a loud error, not the default"
        );
    }

    #[test]
    fn peer_books_and_node_lists_parse_and_reject() {
        let env = cfg(&[
            ("PEERS", "1=127.0.0.1:5001,2=127.0.0.1:5002"),
            ("EMPTY", ""),
            ("NO_EQ", "1:127.0.0.1:5001"),
            ("BAD_ID", "x=127.0.0.1:5001"),
            ("BAD_ADDR", "1=nowhere"),
            ("LIST", "2,3,4"),
            ("BAD_LIST", "2,x"),
        ]);
        let book = env.peer_book("PEERS").expect("parses");
        assert_eq!(book.len(), 2);
        assert_eq!(
            book[&NodeId(1)],
            "127.0.0.1:5001".parse::<SocketAddr>().expect("addr")
        );
        assert_eq!(env.peer_book("EMPTY"), Ok(BTreeMap::new()));
        for bad in ["NO_EQ", "BAD_ID", "BAD_ADDR"] {
            let err = env.peer_book(bad).expect_err("rejected");
            assert!(
                matches!(err, ConfigError::Unparseable { .. }),
                "{bad}: {err}"
            );
        }
        assert_eq!(
            env.node_list("LIST"),
            Ok(vec![NodeId(2), NodeId(3), NodeId(4)])
        );
        assert!(env.node_list("BAD_LIST").is_err());
        assert_eq!(
            env.peer_book("MISSING"),
            Err(ConfigError::Missing("MISSING".to_owned()))
        );
    }

    #[test]
    fn hex32_parses_and_rejects() {
        let hex = "42".repeat(32);
        let env = cfg(&[
            ("KEY", hex.as_str()),
            ("SHORT", "abcd"),
            ("NOT_HEX", &"zz".repeat(32)),
        ]);
        assert_eq!(env.hex32("KEY"), Ok([0x42; 32]));
        assert!(env.hex32("SHORT").is_err());
        assert!(env.hex32("NOT_HEX").is_err());
    }

    #[test]
    fn hex32_encode_is_the_inverse_of_hex32() {
        let bytes = [0x42; 32];
        let hex = hex32_encode(&bytes);
        assert_eq!(hex, "42".repeat(32));
        let env = cfg(&[("KEY", hex.as_str())]);
        assert_eq!(env.hex32("KEY"), Ok(bytes));
        // A mixed payload round-trips byte-for-byte and is zero-padded per byte.
        let mixed = [0x00u8, 0x0f, 0xa0, 0xff];
        assert_eq!(hex32_encode(&mixed), "000fa0ff");
    }

    #[test]
    fn errors_display() {
        assert_eq!(
            ConfigError::Missing("X".to_owned()).to_string(),
            "missing required variable X"
        );
        assert_eq!(
            ConfigError::Unparseable {
                key: "K".to_owned(),
                value: "v".to_owned(),
            }
            .to_string(),
            "variable K has unparseable value \"v\""
        );
    }

    #[test]
    fn pacer_holds_the_rate_and_reports_overruns() {
        // 200 Hz: 10 ticks should take ~50ms (generous bounds; property not speed).
        let mut pacer = TickPacer::new(200);
        let started = Instant::now();
        let mut overruns = 0;
        for _ in 0..10 {
            overruns += pacer.wait();
        }
        let elapsed = started.elapsed();
        assert!(elapsed >= Duration::from_millis(45), "paced: {elapsed:?}");
        assert!(
            elapsed < Duration::from_millis(500),
            "not stuck: {elapsed:?}"
        );
        assert_eq!(overruns, 0, "no overruns on an idle loop");
        // A deliberately slow tick is reported as overrun, then recovered.
        std::thread::sleep(Duration::from_millis(30));
        let reported = pacer.wait();
        assert!(reported >= 1, "the stall was counted: {reported}");
    }
}
