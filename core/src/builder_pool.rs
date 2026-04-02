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
