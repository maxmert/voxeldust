//! ★ THE SOLVE OFF THE TICK (the landform arc, slice 8c stage C4; 03 §5.3: "the solve does NOT run on
//! the shard's tick thread"). A shard is a tick loop; a minute of solving inside a tick is a realm
//! that stops answering its lane, which the peer book reads as a lost connection. So the solve runs
//! on a worker the composition root injects — INLINE in a Tier-A test, a thread in the binary — and
//! the tick loop polls it, exactly the seam the client's chunk workers use (ruling S7-6).
//!
//! The sim never sees the worker: it lives in the composition root, and what it produces is handed
//! to the store as rows and to the lane as tiles by the root's own hands. The BYTES the worker
//! produces do not depend on when it finishes, so nothing here touches determinism.
//!
//! **Example.** The rocky planet's shard boots for the first time, finds no artifact in its store,
//! and hands the solve to the worker. It keeps ticking; a client watching from orbit sees the
//! realm's look bag. Twenty-seven seconds later the worker answers; the shard writes the rows,
//! states its surface, and the tiles begin to flow.

use vd_terrain::BodyDefinition;
use vd_terrain::artifact::Artifact;
use vd_terrain::par::Cores;
use vd_terrain::solve::{Schedule, SolveWords, solve_full_on};

/// What the worker is asked: a body and its charter's words.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveJob {
    pub body: BodyDefinition,
    pub words: SolveWords,
}

/// What the worker answers: the artifact, and the wall time it took in milliseconds (the number
/// the wake-ahead lead is measured against).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolveDone {
    pub artifact: Artifact,
    pub millis: u64,
}

/// THE SOLVE, as the worker runs it: the full schedule, the climate over the final relief, the
/// artifact. `None` where the body has no lattice.
#[must_use]
pub fn run_solve(job: &SolveJob) -> Option<Artifact> {
    run_solve_on(job, Cores::default())
}

/// [`run_solve`] ON `cores` THREADS (ruling B64): the same artifact, byte for byte, on any number of them.
#[must_use]
pub fn run_solve_on(job: &SolveJob, cores: Cores) -> Option<Artifact> {
    let lattice = job.body.macro_lattice()?;
    let (state, facies, _) = solve_full_on(
        &job.body,
        &job.words,
        Schedule::standard(job.words.age_yr),
        cores,
    )?;
    // ★ THE CLIMATE OVER THE CONNECTED OCEAN (2026-09-24, ruling B7 item 1): the same sea the
    // solve's own budget read, so the artifact's climate bytes and its lakes agree.
    let climate = vd_terrain::climate::climate_on(
        &job.body,
        &lattice,
        &job.words,
        &state.z,
        Some(state.sea_view()),
        cores,
    );
    Some(Artifact::of(
        &state,
        &facies,
        &climate,
        job.words.water_km3 > 0,
    ))
}

/// The seam: start a solve, poll for its answer. ONE job at a time per worker.
pub trait ArtifactWorker: Send {
    /// Begin solving `job`; a job already running is refused (`false`).
    fn start(&mut self, job: SolveJob) -> bool;
    /// The finished solve, once; `None` while it runs or when nothing was started. A body with no
    /// lattice answers `Some(None)`.
    fn poll(&mut self) -> Option<Option<SolveDone>>;
    /// Whether a job is running.
    fn busy(&self) -> bool;
}

/// The Tier-A worker: the solve runs at `start`, on the caller's thread.
#[derive(Default)]
pub struct InlineWorker {
    done: Option<Option<SolveDone>>,
    busy: bool,
}

impl ArtifactWorker for InlineWorker {
    fn start(&mut self, job: SolveJob) -> bool {
        if self.busy {
            return false;
        }
        self.busy = true;
        self.done = Some(run_solve(&job).map(|artifact| SolveDone {
            artifact,
            millis: 0,
        }));
        true
    }

    fn poll(&mut self) -> Option<Option<SolveDone>> {
        let done = self.done.take();
        if done.is_some() {
            self.busy = false;
        }
        done
    }

    fn busy(&self) -> bool {
        self.busy
    }
}

/// The binary's worker: one thread per solve, the answer on a channel the tick loop drains; the solve's steps run on
/// the worker's share of the host's cores ([`ThreadedWorker::with_cores`], ruling B64).
#[derive(Default)]
pub struct ThreadedWorker {
    receiver: Option<crossbeam_channel::Receiver<Option<SolveDone>>>,
    cores: Cores,
}

impl ThreadedWorker {
    /// A worker whose solves use `cores` threads.
    #[must_use]
    pub fn with_cores(cores: Cores) -> ThreadedWorker {
        ThreadedWorker {
            receiver: None,
            cores,
        }
    }
}

impl ArtifactWorker for ThreadedWorker {
    fn start(&mut self, job: SolveJob) -> bool {
        if self.receiver.is_some() {
            return false;
        }
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let cores = self.cores;
        std::thread::Builder::new()
            .name("artifact-solve".to_owned())
            .spawn(move || {
                // ★ BELOW THE TICK AND THE WINDOW (ruling B91): this thread and every helper of the solve's steps.
                vd_terrain::par::adopt_start_step(crate::thread_class::lower_to_solve_class);
                let started = std::time::Instant::now();
                let answer = run_solve_on(&job, cores).map(|artifact| SolveDone {
                    artifact,
                    millis: started.elapsed().as_millis() as u64,
                });
                // The receiver may be gone (the shard shut down mid-solve): nobody to answer.
                let _ = sender.send(answer);
            })
            .expect("a solve thread spawns");
        self.receiver = Some(receiver);
        true
    }

    fn poll(&mut self) -> Option<Option<SolveDone>> {
        let answer = match self.receiver.as_ref()?.try_recv() {
            Ok(answer) => answer,
            Err(crossbeam_channel::TryRecvError::Empty) => return None,
            // ★ A SOLVE THREAD THAT DIED (a panic in a step) dropped its sender: the solve ENDED with nothing, once.
            // Before ruling B91 this read "still running" for ever, which since B91 would keep the realm alive for
            // ever (a first solve that has not landed keeps its realm).
            Err(crossbeam_channel::TryRecvError::Disconnected) => None,
        };
        self.receiver = None;
        Some(answer)
    }

    fn busy(&self) -> bool {
        self.receiver.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_terrain::home::{home_moon, home_moon_solve_words};

    /// The inline worker solves the moon at `start`, answers once, and refuses a second job while
    /// the first stands unpolled; the threaded worker answers the same bytes on its own thread.
    #[test]
    fn both_workers_answer_the_moons_artifact() {
        let job = SolveJob {
            body: home_moon(),
            words: home_moon_solve_words(),
        };
        let mut inline = InlineWorker::default();
        assert!(!inline.busy());
        assert_eq!(inline.poll(), None);
        assert!(inline.start(job));
        assert!(inline.busy());
        assert!(!inline.start(job), "one job at a time");
        let first = inline.poll().expect("answered").expect("a lattice");
        assert_eq!(first.millis, 0);
        assert!(!inline.busy());
        assert_eq!(inline.poll(), None);
        let mut threaded = ThreadedWorker::default();
        assert_eq!(threaded.poll(), None);
        assert!(threaded.start(job));
        assert!(threaded.busy());
        assert!(!threaded.start(job));
        let mut answer = None;
        // A guard against a hung thread, never a speed gate: the moon's whole solve takes about 73 s in a debug build
        // on the 414-pass axis (ruling B56; MEASURED 2026-10-05 beside a loaded machine), twice the 214-pass axis's.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
        while answer.is_none() && std::time::Instant::now() < deadline {
            answer = threaded.poll();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let second = answer.expect("the thread answered").expect("a lattice");
        assert_eq!(second.artifact, first.artifact);
        assert!(!threaded.busy());
        assert_eq!(threaded.poll(), None);
    }

    /// ★ A SOLVE THREAD THAT DIED ENDS ITS SOLVE (ruling B91): its sender is gone, so the worker answers "nothing" once
    /// and is free — never "still running" for ever, which would keep the realm alive for ever.
    #[test]
    fn a_dead_solve_thread_ends_the_solve_with_nothing() {
        let (sender, receiver) = crossbeam_channel::bounded::<Option<SolveDone>>(1);
        drop(sender);
        let mut worker = ThreadedWorker {
            receiver: Some(receiver),
            cores: Cores::default(),
        };
        assert!(worker.busy());
        assert_eq!(worker.poll(), Some(None));
        assert!(!worker.busy());
        assert_eq!(worker.poll(), None);
    }
}
