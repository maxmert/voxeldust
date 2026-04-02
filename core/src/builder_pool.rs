//! Thread-local pool for `FlatBufferBuilder` reuse.
//!
//! Avoids allocating a new builder (and its internal `Vec<u8>`) on every
//! `serialize()` call.  At 20 Hz × N messages this eliminates thousands of
//! heap allocations per second.

use std::cell::RefCell;

use flatbuffers::FlatBufferBuilder;

const MAX_POOL_SIZE: usize = 8;

thread_local! {
    static POOL: RefCell<Vec<FlatBufferBuilder<'static>>> = const { RefCell::new(Vec::new()) };
}

/// Acquire a builder from the thread-local pool (or create a new one).
pub fn acquire(capacity: usize) -> FlatBufferBuilder<'static> {
    POOL.with(|pool| {
        pool.borrow_mut()
            .pop()
            .unwrap_or_else(|| FlatBufferBuilder::with_capacity(capacity))
    })
}

/// Return a builder to the thread-local pool for reuse.
pub fn release(mut builder: FlatBufferBuilder<'static>) {
    builder.reset();
    POOL.with(|pool| {
        let mut p = pool.borrow_mut();
        if p.len() < MAX_POOL_SIZE {
            p.push(builder);
        }
        // else: drop — pool is full
    });
}

/// Returns the current number of builders in the thread-local pool (for testing).
#[cfg(test)]
fn pool_size() -> usize {
    POOL.with(|pool| pool.borrow().len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_creates_new_when_pool_empty() {
        // Pool starts empty on this thread.
        let _builder = acquire(128);
        // If we got here without panic, the builder was created successfully.
    }

    #[test]
    fn release_then_acquire_reuses_pooled_builder() {
        // Drain any existing pool entries.
        while pool_size() > 0 { let _ = acquire(64); }

        let b1 = acquire(512);
        release(b1);
        assert_eq!(pool_size(), 1);

        let _b2 = acquire(512);
        // Pool should now be empty (builder was taken from pool).
        assert_eq!(pool_size(), 0);
    }

    #[test]
    fn pool_caps_at_max_size() {
        // Drain existing pool.
        while pool_size() > 0 {
            let _ = acquire(64);
        }

        // Fill pool to max.
        let builders: Vec<_> = (0..MAX_POOL_SIZE + 4)
            .map(|_| acquire(64))
            .collect();
        for b in builders {
            release(b);
        }

        // Pool should be capped at MAX_POOL_SIZE.
        assert_eq!(pool_size(), MAX_POOL_SIZE);
    }

    #[test]
    fn roundtrip_serialization_after_pool() {
        // Verify a pooled builder still produces correct FlatBuffers output.
        let b = acquire(256);
        release(b);

        let mut b = acquire(256);
        let name = b.create_string("test");
        // Just verify it doesn't panic — builder state was properly reset.
        let _ = name;
        release(b);
    }
}
