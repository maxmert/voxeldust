use std::time::{Duration, Instant};

use tracing::{debug, warn};

/// A named system function that runs each tick.
pub struct TickSystem {
    pub name: String,
    pub func: Box<dyn FnMut() + Send>,
}

/// Drives the 20Hz game loop, running registered systems each tick.
pub struct TickLoop {
    systems: Vec<TickSystem>,
    tick_interval: Duration,
    tick_count: u64,
    /// Last tick duration in milliseconds.
    pub last_tick_ms: f32,
    /// Rolling p99 tracker (stores last 100 tick durations).
    tick_history: Vec<f32>,
}

impl TickLoop {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            tick_interval: Duration::from_millis(50), // 20Hz
            tick_count: 0,
            last_tick_ms: 0.0,
            tick_history: Vec::with_capacity(100),
        }
    }

    /// Register a named system to run each tick.
    pub fn add_system(&mut self, name: impl Into<String>, func: impl FnMut() + Send + 'static) {
        self.systems.push(TickSystem {
            name: name.into(),
            func: Box::new(func),
        });
    }

    /// Run a single tick: execute all systems in order and track timing.
    pub fn tick(&mut self) {
        let tick_start = Instant::now();

        for system in &mut self.systems {
            let sys_start = Instant::now();
            (system.func)();
            let elapsed = sys_start.elapsed();
            debug!(system = %system.name, elapsed_us = elapsed.as_micros(), "system tick");
        }

        let elapsed = tick_start.elapsed();
        self.last_tick_ms = elapsed.as_secs_f32() * 1000.0;
        self.tick_count += 1;

        // Track tick history for p99.
        if self.tick_history.len() >= 100 {
            self.tick_history.remove(0);
        }
        self.tick_history.push(self.last_tick_ms);

        if self.last_tick_ms > self.tick_interval.as_secs_f32() * 1000.0 {
            warn!(
                tick = self.tick_count,
                elapsed_ms = self.last_tick_ms,
                budget_ms = self.tick_interval.as_secs_f32() * 1000.0,
                "tick exceeded budget"
            );
        }
    }

    /// Returns the tick interval (50ms for 20Hz).
    pub fn interval(&self) -> Duration {
        self.tick_interval
    }

    /// Current tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Compute p99 tick duration from recent history.
    pub fn p99_tick_ms(&self) -> f32 {
        if self.tick_history.is_empty() {
            return 0.0;
        }
        let mut sorted = self.tick_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((sorted.len() as f32) * 0.99).ceil() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn systems_run_each_tick() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut tick_loop = TickLoop::new();
        tick_loop.add_system("counter", move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        tick_loop.tick();
        tick_loop.tick();
        tick_loop.tick();

        assert_eq!(counter.load(Ordering::Relaxed), 3);
        assert_eq!(tick_loop.tick_count(), 3);
    }

    #[test]
    fn multiple_systems_run_in_order() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut tick_loop = TickLoop::new();

        let log1 = log.clone();
        tick_loop.add_system("first", move || {
            log1.lock().unwrap().push("first");
        });

        let log2 = log.clone();
        tick_loop.add_system("second", move || {
            log2.lock().unwrap().push("second");
        });

        tick_loop.tick();
        assert_eq!(*log.lock().unwrap(), vec!["first", "second"]);
    }

    #[test]
    fn p99_tracks_history() {
        let mut tick_loop = TickLoop::new();
        tick_loop.add_system("noop", || {});

        for _ in 0..10 {
            tick_loop.tick();
        }

        // p99 should be non-negative.
        assert!(tick_loop.p99_tick_ms() >= 0.0);
    }
}
