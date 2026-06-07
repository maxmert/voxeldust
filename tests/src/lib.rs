//! # vd-tests — accumulated scenario suites
//!
//! Shared scenario builders/fixtures used by the integration tests in `tests/tests/`.
//! Standing rule: every phase ADDS scenarios; nothing is deleted. The accumulated
//! suite re-running green is the release gate for every later phase.
