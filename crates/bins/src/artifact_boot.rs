//! ★ THE ARTIFACT AT BOOT (the landform arc, slice 8c stage C4): what a shard does about its
//! planet's artifact when it starts — reads it from its own store, or solves it once on the worker
//! and writes it — and how the tick loop learns it is ready. The decisions live here, in the
//! composition root's library, so a test drives them on an in-memory store with the inline worker
//! and the shard binary is four lines.
//!
//! - A store with no head: the realm was never solved here — the worker starts.
//! - A head under ANOTHER WORLD TAG or another artifact version: another build's artifact — the
//!   worker starts and the rows are overwritten when it answers (the reason is logged).
//! - A head that agrees but rows that do not (a tile missing, a digest wrong): the store lies, and a
//!   store that lies REFUSES the boot — never a quiet re-solve over corrupt rows.
//! - No store at all (a test rig, a seeded-only shard): the worker solves and nothing is written.
//!
//! **Example.** The home planet's shard boots on a pod for the first time: no head, the worker
//! starts, the shard ticks and states its look; a minute later the worker answers, the rows are
//! written, and the surface is stated. It boots again after a crash: the head agrees, the tiles
//! are read, the surface is stated within the boot.

use std::sync::Arc;

use vd_sim::io::Store;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::Artifact;
use vd_terrain::solve::SolveWords;

use crate::artifact_store::{read_artifact, write_artifact};
use crate::artifact_worker::{ArtifactWorker, SolveJob};

/// Why the worker was started, for the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveReason {
    /// No head in the store: never solved here.
    NeverSolved,
    /// A head from another world tag or artifact version.
    AnotherBuild,
    /// No store to read.
    NoStore,
}

/// The artifact's state on this shard.
pub struct ArtifactBoot {
    world_tag: u64,
    artifact: Option<Arc<Artifact>>,
    worker: Box<dyn ArtifactWorker>,
    /// Why the worker runs, while it runs.
    pub solving: Option<SolveReason>,
}

impl ArtifactBoot {
    /// ★ BEGIN: read the store, or start the worker. Refuses (with the store's own words) when the
    /// store holds rows that do not agree with their head.
    ///
    /// # Errors
    /// The store's artifact rows are present and wrong.
    pub fn begin(
        store: Option<&dyn Store>,
        world_tag: u64,
        body: &BodyDefinition,
        words: &SolveWords,
        mut worker: Box<dyn ArtifactWorker>,
    ) -> Result<ArtifactBoot, String> {
        let lattice = body
            .macro_lattice()
            .ok_or("the body has no macro lattice")?;
        let reason = match store {
            None => SolveReason::NoStore,
            Some(store) => match read_artifact(store, world_tag, &lattice) {
                Ok(Some(artifact)) => {
                    return Ok(ArtifactBoot {
                        world_tag,
                        artifact: Some(Arc::new(artifact)),
                        worker,
                        solving: None,
                    });
                }
                Ok(None) => SolveReason::NeverSolved,
                Err(reason) if reason.contains("world tag") || reason.contains("version") => {
                    SolveReason::AnotherBuild
                }
                Err(reason) => return Err(reason),
            },
        };
        let started = worker.start(SolveJob {
            body: *body,
            words: *words,
        });
        debug_assert!(started, "a fresh worker starts");
        Ok(ArtifactBoot {
            world_tag,
            artifact: None,
            worker,
            solving: Some(reason),
        })
    }

    /// ★ POLL, once a tick: when the worker answers, write the rows (if a store is here) and hold
    /// the artifact. Returns the solve's wall time in milliseconds on the tick it lands.
    pub fn poll(&mut self, store: Option<&mut dyn Store>) -> Option<u64> {
        let done = self.worker.poll()?;
        self.solving = None;
        let done = done?;
        if let Some(store) = store {
            write_artifact(store, self.world_tag, &done.artifact);
        }
        let millis = done.millis;
        self.artifact = Some(Arc::new(done.artifact));
        Some(millis)
    }

    /// The artifact, once it is here.
    #[must_use]
    pub fn artifact(&self) -> Option<&Arc<Artifact>> {
        self.artifact.as_ref()
    }

    /// Whether the worker is still solving.
    #[must_use]
    pub fn solving(&self) -> bool {
        self.solving.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact_store::head_of;
    use crate::artifact_worker::InlineWorker;
    use vd_sim::io::mem::MemStore;
    use vd_sim::stub::built_store::{artifact_head_key, artifact_tile_key, encode_artifact_head};
    use vd_terrain::home::{home_moon, home_moon_solve_words};

    /// ★ THE FOUR BOOTS on the moon: a fresh store solves and writes; the next boot reads without a
    /// worker; another build's head is re-solved and overwritten; a corrupt store refuses; no store
    /// solves and writes nothing.
    #[test]
    fn a_shard_solves_once_then_reads_and_refuses_a_lying_store() {
        let moon = home_moon();
        let words = home_moon_solve_words();
        let mut store = MemStore::default();
        // 1. Fresh.
        let mut boot = ArtifactBoot::begin(
            Some(&store),
            7,
            &moon,
            &words,
            Box::new(InlineWorker::default()),
        )
        .expect("begins");
        assert_eq!(boot.solving, Some(SolveReason::NeverSolved));
        assert!(boot.solving());
        assert!(boot.artifact().is_none());
        let millis = boot
            .poll(Some(&mut store))
            .expect("the inline worker answered");
        assert_eq!(millis, 0);
        assert!(!boot.solving());
        let first = boot.artifact().expect("held").clone();
        assert_eq!(boot.poll(Some(&mut store)), None);
        // 2. The next boot reads.
        let again = ArtifactBoot::begin(
            Some(&store),
            7,
            &moon,
            &words,
            Box::new(InlineWorker::default()),
        )
        .expect("begins");
        assert!(!again.solving());
        assert_eq!(again.artifact().map(|a| a.as_ref()), Some(first.as_ref()));
        // 3. Another build's head: re-solved and overwritten.
        let mut other = ArtifactBoot::begin(
            Some(&store),
            8,
            &moon,
            &words,
            Box::new(InlineWorker::default()),
        )
        .expect("begins");
        assert_eq!(other.solving, Some(SolveReason::AnotherBuild));
        other.poll(Some(&mut store));
        assert_eq!(
            crate::artifact_store::read_artifact(
                &store,
                8,
                &moon.macro_lattice().expect("a lattice")
            )
            .expect("reads")
            .map(|a| a.digest()),
            Some(first.digest())
        );
        // A version from another build, likewise.
        let mut head = head_of(&first, 8);
        head.version += 1;
        store.put(&artifact_head_key(), &encode_artifact_head(&head).into());
        store.commit();
        let versioned = ArtifactBoot::begin(
            Some(&store),
            8,
            &moon,
            &words,
            Box::new(InlineWorker::default()),
        )
        .expect("begins");
        assert_eq!(versioned.solving, Some(SolveReason::AnotherBuild));
        // 4. A corrupt store: a tile removed under a good head refuses the boot.
        store.put(
            &artifact_head_key(),
            &encode_artifact_head(&head_of(&first, 8)).into(),
        );
        store.delete(&artifact_tile_key(0, 0, 0));
        store.commit();
        let refused = ArtifactBoot::begin(
            Some(&store),
            8,
            &moon,
            &words,
            Box::new(InlineWorker::default()),
        );
        assert!(refused.err().expect("refused").contains("missing the tile"));
        // 5. No store: solved, nothing written.
        let mut none =
            ArtifactBoot::begin(None, 7, &moon, &words, Box::new(InlineWorker::default()))
                .expect("begins");
        assert_eq!(none.solving, Some(SolveReason::NoStore));
        none.poll(None);
        assert!(none.artifact().is_some());
    }
}
