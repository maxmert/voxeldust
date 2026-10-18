//! ★ THE FINE TILES' PROVIDER (the fine ground's way in, B2; rulings B81 Q2 and B85: the planet's shard solves, stores
//! and ships its finished 128 m tiles): the composition root's side of the sim's fine ship
//! ([`vd_sim::stub::fine_ship`]).
//!
//! - **The reasons** are the realm's own occupants (SL7): each tick the sim names the finished tiles each one's view
//!   reads, nearest first ([`FineTiles::want`]); the next tick's work follows those lists, interleaved by rank so a fast
//!   flier never starves a walker (the design's §4.1).
//! - **The work** runs on worker threads on the solve's share of the cores (`VD_SOLVE_CORES`, ruling B64), one tile a
//!   thread, never more in flight than threads, so the order always follows the newest reasons. A finished tile wants
//!   its 3 × 3 RAW solves (B85: a tile depends on its eight neighbours only — across a face's edge too, and seven at a
//!   cube corner); a raw solve or a finished tile already in
//!   the realm's store is read back, never solved again (a tile is a pure function of its inputs, §3.7).
//! - **The store** is the realm's own (`RealmStore`): each raw solve and each finished record as its part rows, keyed by
//!   the input word (the artifact's digest and the fine version), so a world change never serves an old tile.
//! - **The shard's own book** holds the finished tiles of its reasons, decoded, for its own ground read; what no reason
//!   names leaves memory (it stays in the store).
//!
//! - **Two levels, one machinery** (ruling B95): every map is keyed by the level and the tile; a 16 m tile is solved from
//!   the finished 128 m tiles this provider holds (its book), and only once the 3 × 3 block of 128 m tiles around its
//!   node is finished, so its solve never reads a hole.
//!
//! **Example.** A hull lands on the home planet's belt: the planet's shard names the 128 m tiles within its reach, the
//! workers read the stored ones and solve the rest nearest first, each tile finishes once its eight neighbours are raw,
//! and the finished record goes to the pilot in about thirteen parts; the 16 m tiles under the hull follow from them.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_sim::io::{Bytes, Store};
use vd_sim::stub::built_store::{
    FINE_PART_BYTES, FineAt, FineForm, read_fine_record, write_fine_record,
};
use vd_sim::stub::fine_ship::{FineName, FineTiles, NONE_HELD};
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{Artifact, fine_needs};
use vd_terrain::fine::finish::block_keys;
use vd_terrain::fine::l1::{TileKey, solve_l1};
use vd_terrain::fine::l2::solve_l2;
use vd_terrain::fine::level::{FineLevel, LevelGeom};
use vd_terrain::fine::read::FineBook;
use vd_terrain::fine::stitch::Field;
use vd_terrain::fine::tile::{
    decode_levels, decode_raw, encode_levels, encode_raw, finish, input_word_of, range_record,
    raw_field,
};
use vd_terrain::macro_lattice::MacroLattice;
use vd_wire::channels::BulkMsg;

/// The 128 m level's number on the wire (`BulkMsg::FineTilePart::level`).
pub const LEVEL_L1: u8 = 1;

/// A tile of a level: the key every map of the work holds.
pub type LKey = (FineLevel, TileKey);

/// A finish's nine raw fields with their tiles (NW … SE; none past a cube corner).
type Block = Box<[Option<(TileKey, Arc<Field>)>; 9]>;

/// One piece of work.
enum Job {
    /// Solve a raw tile (a 16 m tile with the finished 128 m book it refines).
    Solve(LKey, Option<Arc<FineBook>>),
    /// Read a stored raw record back.
    LoadRaw(LKey, Vec<u8>),
    /// Finish a tile from its nine raw fields (NW … SE; none past a cube corner).
    Finish(LKey, Block),
    /// Read a stored finished record back.
    LoadFinished(LKey, Vec<u8>),
}

/// One answer: a raw field (and its record when freshly solved, to store), or a finished record and its levels (and
/// whether to store it).
enum Answer {
    Raw(LKey, Option<Arc<Field>>, Option<Vec<u8>>),
    Finished(LKey, Option<(Vec<u8>, Vec<Vec<i32>>)>, bool),
}

/// What a worker reads: the body, its lattice, its artifact, and each level's numbers.
struct Ctx {
    body: BodyDefinition,
    lattice: MacroLattice,
    artifact: Arc<Artifact>,
    /// The 128 m level's numbers and the 16 m level's.
    geoms: [LevelGeom; 2],
}

impl Ctx {
    fn geom(&self, level: FineLevel) -> LevelGeom {
        match level {
            FineLevel::L1 => self.geoms[0],
            FineLevel::L2 => self.geoms[1],
        }
    }
}

/// ★ ONE JOB'S WORK, the same on any thread and any host.
fn work(ctx: &Ctx, job: Job) -> Answer {
    match job {
        Job::Solve(lk @ (level, key), book) => {
            let geom = ctx.geom(level);
            let raster = match (level, book) {
                (FineLevel::L1, _) => solve_l1(&ctx.body, &ctx.lattice, &ctx.artifact, key)
                    .map(|t| (t.side, t.quarters)),
                (FineLevel::L2, Some(book)) => {
                    solve_l2(&ctx.body, &ctx.lattice, &ctx.artifact, &book, key)
                        .map(|t| (t.side, t.quarters))
                }
                (FineLevel::L2, None) => None,
            };
            let raw = raster.map(|(side, quarters)| raw_field(key, geom, side, quarters));
            let record = raw.as_ref().and_then(|f| encode_raw(f, geom.shift));
            Answer::Raw(lk, raw.map(Arc::new), record)
        }
        Job::LoadRaw(lk @ (level, key), bytes) => Answer::Raw(
            lk,
            decode_raw(&bytes, key, ctx.geom(level)).map(Arc::new),
            None,
        ),
        Job::Finish(lk @ (level, _), block) => {
            let geom = ctx.geom(level);
            let slots: [Option<(TileKey, &Field)>; 9] =
                std::array::from_fn(|k| block[k].as_ref().map(|(t, f)| (*t, &**f)));
            let levels = finish(slots, geom);
            let out = levels
                .and_then(|levels| encode_levels(&levels, geom.ts).map(|record| (record, levels)));
            Answer::Finished(lk, out, true)
        }
        Job::LoadFinished(lk @ (level, _), bytes) => {
            let ts = ctx.geom(level).ts;
            let levels = decode_levels(&bytes, level)
                .filter(|(t, _)| *t == ts)
                .map(|(_, levels)| levels);
            Answer::Finished(lk, levels.map(|levels| (bytes, levels)), false)
        }
    }
}

/// ★ THE FINE WORK'S METER (send-ahead's first step, 2026-10-13): what the work did since the shard last read it, and
/// where it stands now — so a fast flight shows whether its late tiles wait for a SOLVE (the backlog grows) or for the
/// WIRE (the backlog stays empty while the client builds on the rough ground). Shared, so the shard program reads it
/// after it hands the work to the sim; counters only, nothing reads them back into the work.
///
/// **Example.** A hull at 20 km/s over the belt: each pace line says how many named tiles have no finished record yet,
/// and how many tiles the workers solved or read back from the store in the last five seconds.
#[derive(Debug, Default)]
pub struct FineMeter {
    /// Raw tiles solved since the last read, and their wall nanoseconds summed.
    pub raw_solved: AtomicU64,
    pub raw_solve_nanos: AtomicU64,
    /// Raw records read back from the store since the last read.
    pub raw_loaded: AtomicU64,
    /// Tiles finished from their nine raw fields since the last read, and their wall nanoseconds summed.
    pub finished_made: AtomicU64,
    pub finish_nanos: AtomicU64,
    /// Finished records read back from the store since the last read.
    pub finished_loaded: AtomicU64,
    /// Now: the tiles the reasons name, those a view reads whole with no finished record yet (the backlog: a far copy
    /// never starts a solve, so it is never in it), the jobs in flight.
    pub reasons: AtomicU64,
    pub backlog: AtomicU64,
    pub in_flight: AtomicU64,
}

/// [`work`], timed into `meter` by the job's kind.
fn timed_work(ctx: &Ctx, job: Job, meter: &FineMeter) -> Answer {
    let (done, nanos) = match &job {
        Job::Solve(..) => (&meter.raw_solved, Some(&meter.raw_solve_nanos)),
        Job::LoadRaw(..) => (&meter.raw_loaded, None),
        Job::Finish(..) => (&meter.finished_made, Some(&meter.finish_nanos)),
        Job::LoadFinished(..) => (&meter.finished_loaded, None),
    };
    let started = std::time::Instant::now();
    let answer = work(ctx, job);
    if let Some(nanos) = nanos {
        nanos.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    done.fetch_add(1, Ordering::Relaxed);
    answer
}

/// The workers: threads on a channel, or none (the work runs at submission, on the tick's thread — the Tier-A rig's
/// shape, as the artifact's `InlineWorker`).
enum Pool {
    Threads {
        jobs: crossbeam_channel::Sender<Job>,
        answers: crossbeam_channel::Receiver<Answer>,
        threads: usize,
    },
    Inline {
        done: Vec<Answer>,
    },
}

/// The provider's moving parts.
struct State {
    pool: Pool,
    /// Held raw fields, finished records, their levels' wire parts as sessions asked for them (by the first level and the
    /// level past the last), and the shard's own book of finished levels.
    raw: BTreeMap<LKey, Arc<Field>>,
    finished: BTreeMap<LKey, Arc<Vec<u8>>>,
    wire: BTreeMap<(LKey, u8, u8), Arc<[Bytes]>>,
    /// The 128 m book and the 16 m book.
    books: [Arc<FineBook>; 2],
    /// The tiles that joined or left a book since the realm's ground last read it (ruling B81 Q6).
    book_changed: BTreeSet<LKey>,
    /// Work in flight: `(tile, finished?)`.
    in_flight: BTreeSet<(LKey, bool)>,
    /// Records the store does not hold: `(tile, finished?)`.
    unstored: BTreeSet<(LKey, bool)>,
    /// Tiles whose solve or finish answered nothing: `(tile, finished?)`.
    refused: BTreeSet<(LKey, bool)>,
    /// This tick's lists, by occupant, each tile with whether its view reads it WHOLE (level 0), and the merged order of
    /// the last with the tiles some view reads whole — served first; a far copy only when none of them waits (ruling B92).
    named: Vec<Vec<(LKey, bool)>>,
    reasons: Vec<LKey>,
    whole: BTreeSet<LKey>,
}

/// A book's slot in [`State::books`].
fn slot(level: FineLevel) -> usize {
    match level {
        FineLevel::L1 => 0,
        FineLevel::L2 => 1,
    }
}

/// ★ THE PROVIDER over one realm's artifact (see the module's head).
pub struct FineWork {
    realm: RealmId,
    ctx: Arc<Ctx>,
    /// Each level's input word (the 128 m level's, the 16 m level's).
    inputs: [u64; 2],
    /// The head's three numbers the reach reads — the drawn ground's lowest and highest metres and the sea (rulings
    /// B55, B87): the gateway reads the same off the head, so both name the same tiles at the same levels.
    head_band: (i32, i32, i16),
    state: Arc<Mutex<State>>,
    meter: Arc<FineMeter>,
}

/// ★ THE BOOK AS THE REALM'S GROUND READS IT (ruling B81 Q6): the provider's own book, shared, and the tiles that joined or
/// left it since the last read — what the realm's parked children stand on.
pub struct GroundBook(Arc<Mutex<State>>);

impl GroundBook {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The 128 m book as it stands (a pointer: no tile is copied).
    #[must_use]
    pub fn book(&self) -> Arc<FineBook> {
        Arc::clone(&self.lock().books[0])
    }

    /// The 16 m book as it stands.
    #[must_use]
    pub fn finer(&self) -> Arc<FineBook> {
        Arc::clone(&self.lock().books[1])
    }

    /// The tiles that joined or left a book since the last call.
    #[must_use]
    pub fn take_changed(&self) -> Vec<LKey> {
        std::mem::take(&mut self.lock().book_changed)
            .into_iter()
            .collect()
    }
}

impl FineWork {
    /// A provider over `artifact` with `threads` workers (none: the work runs on the tick's thread). `None` where the
    /// body has no lattice, its rung-7 cells do not nest in its nodes, or the level does not apply to it.
    #[must_use]
    pub fn new(
        realm: RealmId,
        body: BodyDefinition,
        artifact: Arc<Artifact>,
        threads: usize,
    ) -> Option<FineWork> {
        let lattice = body.macro_lattice()?;
        let geoms = [
            LevelGeom::of(FineLevel::L1, &body, &lattice)?,
            LevelGeom::of(FineLevel::L2, &body, &lattice)?,
        ];
        let book = FineBook::new(FineLevel::L1, &body, &lattice, artifact.sea_m)?;
        let finer = FineBook::new(FineLevel::L2, &body, &lattice, artifact.sea_m)?;
        if !vd_terrain::fine::read::applies(artifact.sea_m) {
            return None;
        }
        let drawn = artifact.drawn_range_m(&lattice);
        let head_band = (drawn.0, drawn.1, artifact.sea_m);
        let ctx = Arc::new(Ctx {
            body,
            lattice,
            artifact: Arc::clone(&artifact),
            geoms,
        });
        let meter = Arc::new(FineMeter::default());
        let pool = if threads == 0 {
            Pool::Inline { done: Vec::new() }
        } else {
            let (jobs, job_rx) = crossbeam_channel::unbounded::<Job>();
            let (answer_tx, answers) = crossbeam_channel::unbounded::<Answer>();
            for k in 0..threads {
                let (rx, tx, ctx) = (job_rx.clone(), answer_tx.clone(), Arc::clone(&ctx));
                let meter = Arc::clone(&meter);
                std::thread::Builder::new()
                    .name(format!("fine-solve-{k}"))
                    .spawn(move || {
                        // ★ BELOW THE TICK AND THE WINDOW (ruling B91): a fine tile's solve takes only the cores
                        // nobody else needs, and so would any helper a step of it starts.
                        vd_terrain::par::adopt_start_step(
                            crate::thread_class::lower_to_solve_class,
                        );
                        // The provider dropped (the shard shut down): the channel closes and the thread ends.
                        for job in rx.iter() {
                            if tx.send(timed_work(&ctx, job, &meter)).is_err() {
                                break;
                            }
                        }
                    })
                    .expect("a fine worker spawns");
            }
            Pool::Threads {
                jobs,
                answers,
                threads,
            }
        };
        Some(FineWork {
            realm,
            ctx,
            inputs: [
                input_word_of(FineLevel::L1, artifact.digest()),
                input_word_of(FineLevel::L2, artifact.digest()),
            ],
            head_band,
            state: Arc::new(Mutex::new(State {
                pool,
                raw: BTreeMap::new(),
                finished: BTreeMap::new(),
                wire: BTreeMap::new(),
                books: [Arc::new(book), Arc::new(finer)],
                book_changed: BTreeSet::new(),
                in_flight: BTreeSet::new(),
                unstored: BTreeSet::new(),
                refused: BTreeSet::new(),
                named: Vec::new(),
                reasons: Vec::new(),
                whole: BTreeSet::new(),
            })),
            meter,
        })
    }

    /// The work's meter, shared (see [`FineMeter`]).
    #[must_use]
    pub fn meter(&self) -> Arc<FineMeter> {
        Arc::clone(&self.meter)
    }

    /// How far the work stands: raw fields held, finished tiles held, jobs in flight — a reading for the log and the
    /// bench.
    #[must_use]
    pub fn progress(&self) -> (usize, usize, usize) {
        let state = self.lock();
        (state.raw.len(), state.finished.len(), state.in_flight.len())
    }

    /// The shard's own book of finished 128 m tiles (its ground read), as it stands.
    #[must_use]
    pub fn book(&self) -> FineBook {
        (*self.lock().books[0]).clone()
    }

    /// The book as the realm's ground reads it (ruling B81 Q6).
    #[must_use]
    pub fn ground_book(&self) -> GroundBook {
        GroundBook(Arc::clone(&self.state))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn at(&self, (level, key): LKey, finished: bool) -> FineAt {
        FineAt {
            form: if finished {
                FineForm::Finished
            } else {
                FineForm::Raw
            },
            input: self.inputs[slot(level)],
            level: level.id(),
            face: key.face.index(),
            tx: key.tx as u32,
            ty: key.ty as u32,
        }
    }

    /// The finished record's wire parts, cut at the store's own size.
    fn wire_parts(&self, (level, key): LKey, record: &[u8]) -> Arc<[Bytes]> {
        let parts = record.len().div_ceil(FINE_PART_BYTES).max(1);
        (0..parts)
            .map(|part| {
                let start = (part * FINE_PART_BYTES).min(record.len());
                let end = (start + FINE_PART_BYTES).min(record.len());
                let msg = BulkMsg::FineTilePart {
                    realm: self.realm,
                    level: level.id(),
                    face: key.face.index(),
                    tx: key.tx as u32,
                    ty: key.ty as u32,
                    part: part as u32,
                    parts: parts as u32,
                    bytes: record[start..end].to_vec(),
                };
                vd_sim::io::bytes(
                    postcard::to_allocvec(&msg).expect("a closed wire enum serializes"),
                )
            })
            .collect()
    }

    /// The tiles around `key` (NW … SE) on any face, at its level; none past a cube corner (B2's face-edge cut: every
    /// tile finishes).
    fn block_of(&self, (level, key): LKey) -> [Option<LKey>; 9] {
        block_keys(key, self.ctx.geom(level)).map(|k| k.map(|k| (level, k)))
    }

    /// ★ WHAT A 16 m TILE'S SOLVE READS (ruling B95): the 3 × 3 block of finished 128 m tiles around the 128 m tile that
    /// holds its node — its 128 m flow reaches a few nodes past the node, never past that block. Nothing for a 128 m
    /// tile.
    fn reads(&self, (level, key): LKey) -> Vec<LKey> {
        match level {
            FineLevel::L1 => Vec::new(),
            FineLevel::L2 => {
                let down = FineLevel::L1.tile_nodes_shift();
                let holder = TileKey {
                    face: key.face,
                    tx: key.tx >> down,
                    ty: key.ty >> down,
                };
                self.block_of((FineLevel::L1, holder))
                    .into_iter()
                    .flatten()
                    .collect()
            }
        }
    }

    /// A named tile the work can still make: not finished, its finish not refused, no raw field of its block refused,
    /// and every finished tile its solve reads held.
    fn workable(&self, state: &State, key: LKey) -> bool {
        !state.finished.contains_key(&key)
            && !state.refused.contains(&(key, true))
            && !self
                .block_of(key)
                .iter()
                .flatten()
                .any(|b| state.refused.contains(&(*b, false)))
            && self
                .reads(key)
                .iter()
                .all(|r| state.finished.contains_key(r))
    }

    /// Hands `job` to the workers (or runs it, inline).
    fn submit(state: &mut State, ctx: &Ctx, meter: &FineMeter, job: Job) {
        match &mut state.pool {
            Pool::Threads { jobs, .. } => {
                let _ = jobs.send(job);
            }
            Pool::Inline { done } => done.push(timed_work(ctx, job, meter)),
        }
    }

    /// The answers landed since the last call.
    fn answers(state: &mut State) -> Vec<Answer> {
        match &mut state.pool {
            Pool::Threads { answers, .. } => answers.try_iter().collect(),
            Pool::Inline { done } => std::mem::take(done),
        }
    }
}

impl FineTiles for FineWork {
    fn tick(&self, mut store: Option<&mut dyn Store>) {
        let mut state = self.lock();
        let state = &mut *state;
        // ★ THE ANSWERS LAND: stored (a fresh solve or finish), then held.
        let mut wrote = false;
        for answer in Self::answers(state) {
            match answer {
                Answer::Raw(key, field, record) => {
                    state.in_flight.remove(&(key, false));
                    match field {
                        Some(field) => {
                            if let (Some(store), Some(record)) = (store.as_deref_mut(), record) {
                                write_fine_record(store, &self.at(key, false), &record);
                                state.unstored.remove(&(key, false));
                                wrote = true;
                            }
                            state.raw.insert(key, field);
                        }
                        None => {
                            state.refused.insert((key, false));
                        }
                    }
                }
                Answer::Finished(key, out, store_it) => {
                    state.in_flight.remove(&(key, true));
                    match out {
                        Some((record, levels)) => {
                            if let Some(store) = store.as_deref_mut().filter(|_| store_it) {
                                write_fine_record(store, &self.at(key, true), &record);
                                state.unstored.remove(&(key, true));
                                wrote = true;
                            }
                            state.finished.insert(key, Arc::new(record));
                            Arc::make_mut(&mut state.books[slot(key.0)]).insert(key.1, levels);
                            state.book_changed.insert(key);
                        }
                        None => {
                            state.refused.insert((key, true));
                        }
                    }
                }
            }
        }
        if let Some(store) = store.as_deref_mut().filter(|_| wrote) {
            store.commit();
        }
        // ★ THE REASONS: the last tick's lists, interleaved by rank (round-robin across occupants), each tile once.
        let named = std::mem::take(&mut state.named);
        let longest = named.iter().map(Vec::len).max().unwrap_or(0);
        let mut seen = BTreeSet::new();
        state.reasons = (0..longest)
            .flat_map(|rank| {
                named
                    .iter()
                    .filter_map(move |list| list.get(rank).map(|(key, _)| *key))
            })
            .filter(|key| seen.insert(*key))
            .collect();
        state.whole = named
            .iter()
            .flatten()
            .filter_map(|&(key, whole)| whole.then_some(key))
            .collect();
        let threads = match &state.pool {
            Pool::Threads { threads, .. } => *threads,
            Pool::Inline { .. } => 1,
        };
        // ★ WHOLE TILES FIRST, FAR COPIES WHEN NONE WAITS (2026-10-13, ruling B92 with B81 Q4's idle work): the work serves
        // every tile a view reads whole that is not finished; only when none waits does it serve the nearest far copies,
        // as many as it has threads — so a fast flight's near ground always goes first, and a realm at rest fills its
        // middle distance (the fine version's change left every stored far copy unread). MEASURED before the rule: at
        // 1 000 km over the home planet the reach named 22 575 tiles for their coarse copies alone and queued a full solve
        // for each, in front of the near ground.
        let whole_waiting = state
            .reasons
            .iter()
            .any(|&key| state.whole.contains(&key) && self.workable(state, key));
        let serving: BTreeSet<LKey> = state
            .reasons
            .iter()
            .copied()
            .filter(|&key| {
                (state.whole.contains(&key) || !whole_waiting) && self.workable(state, key)
            })
            .take(if whole_waiting { usize::MAX } else { threads })
            .collect();
        // ★ WHAT NO REASON NAMES LEAVES MEMORY (it stays in the store): the finished tiles outside the reasons, and the
        // raw fields no served tile's block reads — the rule that serves a tile is the rule that keeps its block, so a
        // raw field is never dropped before its finish starts.
        let wanted: BTreeSet<LKey> = state.reasons.iter().copied().collect();
        let keep_raw: BTreeSet<LKey> = serving
            .iter()
            .flat_map(|&key| self.block_of(key).into_iter().flatten())
            .collect();
        state.finished.retain(|key, _| wanted.contains(key));
        state.wire.retain(|(key, _, _), _| wanted.contains(key));
        // A 128 m tile a wanted 16 m tile's solve reads stays in memory too.
        let wanted: BTreeSet<LKey> = wanted
            .iter()
            .flat_map(|&key| std::iter::once(key).chain(self.reads(key)))
            .collect();
        state.finished.retain(|key, _| wanted.contains(key));
        for level in FineLevel::ALL {
            let gone: Vec<TileKey> = state.books[slot(level)]
                .keys()
                .filter(|key| !wanted.contains(&(level, *key)))
                .collect();
            for key in gone {
                Arc::make_mut(&mut state.books[slot(level)]).remove(key);
                state.book_changed.insert((level, key));
            }
        }
        state.raw.retain(|key, _| keep_raw.contains(key));
        // ★ THE NEXT WORK, nearest first, never more in flight than threads (the order follows the newest reasons).
        let reasons = state.reasons.clone();
        for key in reasons {
            if state.in_flight.len() >= threads {
                break;
            }
            if state.finished.contains_key(&key)
                | state.in_flight.contains(&(key, true))
                | state.refused.contains(&(key, true))
            {
                continue;
            }
            // A stored finished record is read back, never made again.
            if !state.unstored.contains(&(key, true)) {
                match store
                    .as_deref()
                    .map(|s| read_fine_record(s, &self.at(key, true)))
                {
                    Some(Ok(Some(bytes))) => {
                        state.in_flight.insert((key, true));
                        Self::submit(state, &self.ctx, &self.meter, Job::LoadFinished(key, bytes));
                        continue;
                    }
                    Some(Err(reason)) => tracing::warn!(
                        ?key,
                        reason,
                        "a stored fine tile is refused; it is finished again"
                    ),
                    _ => {}
                }
                state.unstored.insert((key, true));
            }
            // A tile outside the served set (a far copy while a whole tile waits) is read back when stored, never made.
            if !serving.contains(&key) {
                continue;
            }
            let block = self.block_of(key);
            if block.iter().flatten().all(|b| state.raw.contains_key(b)) {
                let slots =
                    block.map(|b| b.and_then(|b| state.raw.get(&b).map(|f| (b.1, Arc::clone(f)))));
                state.in_flight.insert((key, true));
                Self::submit(
                    state,
                    &self.ctx,
                    &self.meter,
                    Job::Finish(key, Box::new(slots)),
                );
                continue;
            }
            for raw in block.into_iter().flatten() {
                if state.in_flight.len() >= threads {
                    break;
                }
                if state.raw.contains_key(&raw) | state.in_flight.contains(&(raw, false)) {
                    continue;
                }
                state.in_flight.insert((raw, false));
                let stored = if state.unstored.contains(&(raw, false)) {
                    None
                } else {
                    store
                        .as_deref()
                        .and_then(|s| read_fine_record(s, &self.at(raw, false)).ok().flatten())
                };
                match stored {
                    Some(bytes) => {
                        Self::submit(state, &self.ctx, &self.meter, Job::LoadRaw(raw, bytes))
                    }
                    None => {
                        state.unstored.insert((raw, false));
                        let book = (raw.0 == FineLevel::L2).then(|| Arc::clone(&state.books[0]));
                        Self::submit(state, &self.ctx, &self.meter, Job::Solve(raw, book));
                    }
                }
            }
        }
        // ★ THE METER'S GAUGES (send-ahead's first step): the tiles named, those not finished yet, the jobs in flight.
        let backlog = state
            .reasons
            .iter()
            .filter(|key| !state.finished.contains_key(key) && state.whole.contains(key))
            .count();
        self.meter
            .reasons
            .store(state.reasons.len() as u64, Ordering::Relaxed);
        self.meter.backlog.store(backlog as u64, Ordering::Relaxed);
        self.meter
            .in_flight
            .store(state.in_flight.len() as u64, Ordering::Relaxed);
    }

    fn want(&self, dir: [f64; 3], radial_m: f64) -> Vec<(FineName, u8)> {
        // ★ BY DISTANCE ALONE (rulings B86 item 3, B87), over the TRUE ground under the view (this realm holds it): the
        // tiles the client's own ladder reads, and no more to solve. The gateway, which reads no ground, serves its
        // sessions by the head's band (`fine_needs_of_head`), a reach that holds this one, so a session is never
        // served a tile this realm does not name.
        let ctx = &self.ctx;
        let pixel_rad = vd_core::geometry::drawable_theta_min_rad();
        // ★ THE GROUND THE CLIENT CAN DRAW (2026-10-18; the macro ring's own rule, `artifact_source`): the drawn 128 m and
        // 16 m ground stands up to the level's law off the 8 km field, and the client reads its horizon and its height
        // over it, so the reach reads the horizon over the field less that law and the height over the field plus it.
        let field_m = vd_terrain::height::height_field_m(&ctx.body, &*ctx.artifact, dir, 0)
            .unwrap_or(radial_m);
        let law_m = ctx.body.fine_law_m(ctx.artifact.sea_m);
        let banded = ctx
            .body
            .with_fine_band_m(self.head_band.0, self.head_band.1, self.head_band.2)
            .unwrap_or(ctx.body);
        let relief_m = banded.ground_range_m().1;
        let needs = |level: FineLevel| -> Vec<(FineName, u8)> {
            fine_needs(
                level,
                &banded,
                &ctx.lattice,
                dir,
                radial_m,
                (field_m - law_m, field_m + law_m),
                relief_m,
                pixel_rad,
            )
            .into_iter()
            .map(|((f, tx, ty), k)| ((level.id(), f, tx, ty), k))
            .collect()
        };
        // ★ THE ORDER (ruling B95): the 128 m tiles nearest first, then the 16 m tiles nearest first. The 128 m ground is
        // the ground in view (the login's loading screen waits for it alone, ruling B86 item 5), and every 16 m tile reads
        // the 128 m tiles around it. MEASURED (2026-10-14, hull 17's login): with the 16 m tiles before the outer 128 m
        // ring, the 16 m solves (19 s each on the solve's share) held every work slot, and the ring's STORED tiles — a
        // read of milliseconds — waited behind them with the loading screen.
        let named: Vec<(FineName, u8)> = needs(FineLevel::L1)
            .into_iter()
            .chain(needs(FineLevel::L2))
            .collect();
        self.hold(&named);
        named
    }

    fn hold(&self, tiles: &[(FineName, u8)]) {
        let keys: Vec<(LKey, bool)> = tiles
            .iter()
            .filter_map(|&((l, f, tx, ty), k)| {
                Some((
                    (
                        FineLevel::from_id(l)?,
                        TileKey {
                            face: Face::from_index(f)?,
                            tx: tx as i32,
                            ty: ty as i32,
                        },
                    ),
                    k == 0,
                ))
            })
            .collect();
        self.lock().named.push(keys);
    }

    fn levels(&self, level: u8) -> u8 {
        FineLevel::from_id(level).map_or(0, |l| (l.mean_levels() + 1) as u8)
    }

    fn parts(&self, tile: FineName, lo: u8, hi: u8) -> Option<Arc<[Bytes]>> {
        let level = FineLevel::from_id(tile.0)?;
        let key = (
            level,
            TileKey {
                face: Face::from_index(tile.1)?,
                tx: tile.2 as i32,
                ty: tile.3 as i32,
            },
        );
        let mut state = self.lock();
        if let Some(parts) = state.wire.get(&(key, lo, hi)) {
            return Some(Arc::clone(parts));
        }
        // ★ THE LEVELS A SESSION LACKS (ruling B86 item 3): `lo` up to the finest it holds, cut from the finished record.
        let record = Arc::clone(state.finished.get(&key)?);
        let top = level.mean_levels() + 1;
        let end = if hi == NONE_HELD {
            top
        } else {
            u32::from(hi).min(top)
        };
        let parts = self.wire_parts(key, &range_record(&record, u32::from(lo), end)?);
        state.wire.insert((key, lo, hi), Arc::clone(&parts));
        Some(parts)
    }
}
