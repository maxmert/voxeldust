use std::time::{Duration, Instant};

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Normal operation — requests pass through.
    Closed,
    /// Too many failures — requests are rejected immediately.
    Open,
    /// Cooldown expired — next request is a probe. Success → Closed, failure → Open.
    HalfOpen,
}

/// Per-peer circuit breaker for QUIC connections.
///
/// Tracks consecutive failures and opens the circuit after a threshold,
/// using exponential backoff for the cooldown period.
#[derive(Debug)]
pub struct CircuitBreaker {
    state: BreakerState,
    consecutive_failures: u32,
    failure_threshold: u32,
    last_failure_time: Option<Instant>,
    cooldown: Duration,
    min_cooldown: Duration,
    max_cooldown: Duration,
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self {
            state: BreakerState::Closed,
            consecutive_failures: 0,
            failure_threshold: 5,
            last_failure_time: None,
            cooldown: Duration::from_secs(5),
            min_cooldown: Duration::from_secs(5),
            max_cooldown: Duration::from_secs(60),
        }
    }

    /// Returns the current state, accounting for cooldown expiry.
    pub fn state(&self) -> BreakerState {
        match self.state {
            BreakerState::Open => {
                if let Some(last) = self.last_failure_time {
                    if last.elapsed() >= self.cooldown {
                        return BreakerState::HalfOpen;
                    }
                }
                BreakerState::Open
            }
            other => other,
        }
    }

    /// Check if a request should be allowed through.
    pub fn allow_request(&self) -> bool {
        match self.state() {
            BreakerState::Closed | BreakerState::HalfOpen => true,
            BreakerState::Open => false,
        }
    }

    /// Record a successful operation. Resets the breaker to Closed.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.state = BreakerState::Closed;
        self.cooldown = self.min_cooldown;
    }

    /// Record a failed operation. May trip the breaker to Open.
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.last_failure_time = Some(Instant::now());

        if self.consecutive_failures >= self.failure_threshold {
            self.state = BreakerState::Open;
            // Exponential backoff: double cooldown on each re-open, capped.
            self.cooldown = (self.cooldown * 2).min(self.max_cooldown);
        }
    }

    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_closed() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), BreakerState::Closed);
        assert!(cb.allow_request());
    }

    #[test]
    fn opens_after_threshold_failures() {
        let mut cb = CircuitBreaker::new();
        for _ in 0..4 {
            cb.record_failure();
            assert_eq!(cb.state(), BreakerState::Closed);
        }
        cb.record_failure(); // 5th failure
        assert_eq!(cb.state(), BreakerState::Open);
        assert!(!cb.allow_request());
    }

    #[test]
    fn success_resets() {
        let mut cb = CircuitBreaker::new();
        for _ in 0..5 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), BreakerState::Open);

        // Simulate cooldown expiry by creating a new breaker in half-open
        // (we can't easily fast-forward time, but we can test success reset)
        cb.state = BreakerState::HalfOpen;
        cb.record_success();
        assert_eq!(cb.state(), BreakerState::Closed);
        assert!(cb.allow_request());
        assert_eq!(cb.consecutive_failures(), 0);
    }

    #[test]
    fn exponential_backoff() {
        let mut cb = CircuitBreaker::new();
        // First trip
        for _ in 0..5 {
            cb.record_failure();
        }
        assert_eq!(cb.cooldown, Duration::from_secs(10)); // 5 * 2

        // Reset and trip again
        cb.state = BreakerState::HalfOpen;
        cb.consecutive_failures = 0;
        for _ in 0..5 {
            cb.record_failure();
        }
        assert_eq!(cb.cooldown, Duration::from_secs(20)); // 10 * 2
    }

    #[test]
    fn backoff_caps_at_max() {
        let mut cb = CircuitBreaker::new();
        // Trip many times
        for _ in 0..10 {
            for _ in 0..5 {
                cb.record_failure();
            }
            cb.consecutive_failures = 0;
        }
        assert!(cb.cooldown <= Duration::from_secs(60));
    }
}
