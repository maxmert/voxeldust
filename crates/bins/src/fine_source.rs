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
    COAST_PART_BYTES, FINE_PART_BYTES, FineAt, FineForm, far_lakes_key, read_fine_record,
    write_fine_record,
};
use vd_sim::stub::fine_ship::{FineName, FineTiles, NONE_HELD};
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{Artifact, fine_needs};
use vd_terrain::fine::far::{FarNode, encode_far_nodes, tile_far_nodes};
use vd_terrain::fine::finish::block_keys;
use vd_terrain::fine::l1::{OutletRule, TileKey, Waters, solve_l1_levels};
use vd_terrain::fine::l2::solve_l2_with;
use vd_terrain::fine::level::{FineLevel, LevelGeom};
use vd_terrain::fine::read::FineBook;
use vd_terrain::fine::stitch::Field;
use vd_terrain::fine::tile::{
    WaterLevels, decode_raw_water, decode_water_levels, encode_raw_water, encode_water_levels,
    finish_water, input_word_of, raw_field, water_planes, water_range_record,
};
use vd_terrain::macro_lattice::MacroLattice;
use vd_wire::channels::BulkMsg;

/// The 128 m level's number on the wire (`BulkMsg::FineTilePart::level`).
pub const LEVEL_L1: u8 = 1;

/// A tile of a level: the key every map of the work holds.
pub type LKey = (FineLevel, TileKey);

/// ★ A RAW TILE WITH ITS WATER (slice W2): the raster's heights, its water's level and its wet share, each a field on
/// the level's global cells.
pub type RawTile = [Field; 3];

/// ★ THE OUTLET RULE THE GAME'S FINE TILES ARE SOLVED UNDER (slice W2; rulings B105 Q1 (b), B114, B118, B122): THE DELTAS —
/// the 128 m tiles are solved as the drowned basin (every lake's basin land that drains along the lake's own tree to its
/// deepest node) on THE BEDROCK ([`Ctx::bed`]), and each lake's laid rock is laid back as deltas from its rivers' mouths;
/// the water map floods that ground at the lake's level; the sea's points under its level are outlets, the rivers carved.
pub const GAME_RULE: OutletRule = OutletRule::Deltas;

/// A finish's nine raw tiles with their keys (NW … SE; none past a cube corner).
type Block = Box<[Option<(TileKey, Arc<RawTile>)>; 9]>;

/// ★ A LAKE'S FINE LEVEL IN THE STORE (ruling B115): it rides the fine family under level number 0 (no tile has it), the
/// lake's id in the tile index, keyed by the 128 m level's input word — so a world change never serves an old level.
const LAKE_LEVEL_AT: u8 = 0;
/// The stored lake level's form.
const LAKE_LEVEL_FORM: u8 = 1;
/// ★ A LAKE'S ROUNDS' BOUND (ruling B115, form (a′)): the rounds stop when the level holds; the bound only guarantees an
/// end. MEASURED (the bench, 2026-10-18): the basin lake held after one round.
const LAKE_ROUND_BOUND: u32 = 8;

/// ★ IDLE WORK'S SHARE OF THE SOLVE THREADS, as a divisor (2026-10-19): the far lakes' tiles no view names may fill at most
/// half the threads, so a tile a view reads starts at once. MEASURED before (the owner's flight over the basin, ruling
/// B116's first deploy): the lake tiles' solves held all seven threads for 10–30 s each, the tiles the flight needed rose to
/// 110 waiting, and the pieces over them stood on the 8 km ground. A far copy a view names is the ground that view draws
/// on the rough form until it lands, so it takes every thread when no whole tile waits (ruling B117: a solve about 5 s, so
/// a whole tile the flight then needs waits about that long for a thread; MEASURED at the 60 km stand over the basin: its
/// view named about 1 700 far copies, which the half share left on four threads while three stood idle).
const IDLE_THREADS_DIVISOR: usize = 2;

/// One lake's rounds (ruling B115, form (a′)): the level its tiles are solved at now, the round's number (round 0 at the
/// 8 km word reads the tiles' shared plain solves), and each tile's answer — its raster's spill for the lake, or none.
struct Round {
    level: i32,
    n: u32,
    got: BTreeMap<TileKey, Option<i32>>,
}

/// ★ A TILE'S PLAIN SOLVE (every lake at its 8 km word): the spills of every lake in its raster — round 0 of each of them
/// at once — and its raw field and record, which are the tile's final raw where every lake of its domain holds its 8 km
/// word.
struct Plain {
    spills: Vec<(u32, i32)>,
    raw: Option<(Arc<RawTile>, Option<Vec<u8>>)>,
}

/// One piece of work.
enum Job {
    /// Solve a raw tile (a 16 m tile with the finished 128 m book it refines), each named lake at its fine level.
    Solve(LKey, Option<Arc<FineBook>>, Arc<BTreeMap<u32, i32>>),
    /// ★ A LAKE'S ROUND (ruling B115): solve one of its 128 m tiles with the lake at the level and every other lake of the
    /// tile's domain at its settled level (the lakes together, 2026-10-19), for its spill.
    Round(u32, i32, TileKey, Arc<BTreeMap<u32, i32>>),
    /// ★ A 128 m TILE'S PLAIN SOLVE (ruling B115): every lake at its 8 km word — the first round of every lake in its raster.
    Plain(TileKey),
    /// Read a stored raw record back.
    LoadRaw(LKey, Vec<u8>),
    /// Finish a tile from its nine raw fields (NW … SE; none past a cube corner).
    Finish(LKey, Block),
    /// Read a stored finished record back.
    LoadFinished(LKey, Vec<u8>),
}

/// A 128 m final solve's check (the lakes together): the levels it named and every lake's spill in its raster.
type Checked = (Arc<BTreeMap<u32, i32>>, Vec<(u32, i32)>);

/// One answer: a raw field (and its record when freshly solved, to store), or a finished record and its levels (and
/// whether to store it).
enum Answer {
    /// A raw field (and its record when freshly solved), and for a 128 m tile's final solve the levels it named and every
    /// lake's spill in its raster (the lakes together: a lake passing under its level reopens).
    Raw(LKey, Option<Arc<RawTile>>, Option<Vec<u8>>, Option<Checked>),
    /// A lake's round: the lake, the level, the tile, and the tile's spill for the lake (none where it holds, or where the
    /// solve answered nothing).
    Round(u32, i32, TileKey, Option<i32>),
    /// A plain solve: the tile, its raw field and record, and every lake's spill in its raster.
    Plain(
        TileKey,
        Option<Arc<RawTile>>,
        Option<Vec<u8>>,
        Vec<(u32, i32)>,
    ),
    /// A finished tile: its record and levels (and whether to store it), and the far rows of its nodes under or beside a lake
    /// (ruling B116 part 2), folded on the worker so the tick only keeps them.
    Finished(LKey, Option<(Vec<u8>, WaterLevels)>, bool, Vec<FarNode>),
}

/// What a worker reads: the body, its lattice, its artifact, and each level's numbers.
struct Ctx {
    body: BodyDefinition,
    lattice: MacroLattice,
    artifact: Arc<Artifact>,
    /// ★ THE BEDROCK VIEW the 128 m solves and the lakes' bookkeeping read (ruling B122; `l1::bedrock_view`): the artifact
    /// with each lake record's laid rock taken off its row, made once per world version under [`OutletRule::Deltas`]; the
    /// artifact itself under every other rule. The reach and the 16 m level read the rows the client reads.
    bed: Arc<Artifact>,
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

/// A solved raster (its side, heights, water map and narrower rivers' shares) as a raw tile's three fields and its record.
fn raw_of(
    key: TileKey,
    geom: LevelGeom,
    (side, quarters, water, share): (usize, Vec<i32>, Vec<i32>, Vec<u8>),
) -> (RawTile, Option<Vec<u8>>) {
    let (level, wet) = water_planes(&water, &share);
    let raw = [
        raw_field(key, geom, side, quarters),
        raw_field(key, geom, side, level),
        raw_field(key, geom, side, wet),
    ];
    let record = encode_raw_water(&raw[0], &raw[1].v, &raw[2].v, geom.shift);
    (raw, record)
}

/// ★ A FINISHED 128 m TILE'S FAR ROWS (ruling B116 part 2): its nodes under or beside a lake, folded from its levels
/// (`vd_terrain::fine::far::tile_far_nodes`); none for a 16 m tile or a tile that did not finish.
fn far_rows_of(ctx: &Ctx, (level, key): LKey, levels: Option<&WaterLevels>) -> Vec<FarNode> {
    let Some(levels) = levels.filter(|_| level == FineLevel::L1) else {
        return Vec::new();
    };
    let none = BTreeMap::new();
    let waters = Waters::new(&ctx.lattice, &ctx.bed, &none);
    let keep = |n: u32| {
        waters.under_lake(n)
            || ctx
                .lattice
                .neighbours(n)
                .into_iter()
                .any(|m| m != vd_terrain::macro_lattice::NO_NODE && waters.under_lake(m))
    };
    tile_far_nodes(
        key,
        levels,
        ctx.geom(FineLevel::L1),
        &ctx.lattice,
        ctx.artifact.sea_m,
        keep,
    )
}

/// ★ ONE JOB'S WORK, the same on any thread and any host.
fn work(ctx: &Ctx, job: Job) -> Answer {
    match job {
        Job::Round(lake, level, key, others) => {
            let mut levels = (*others).clone();
            levels.insert(lake, level);
            let spill = solve_l1_levels(
                &ctx.body,
                &ctx.lattice,
                &ctx.bed,
                key,
                GAME_RULE,
                &levels,
            )
            .and_then(|t| t.spills.iter().find(|&&(l, _)| l == lake).map(|&(_, p)| p));
            Answer::Round(lake, level, key, spill)
        }
        Job::Plain(key) => {
            let geom = ctx.geom(FineLevel::L1);
            let none = BTreeMap::new();
            match solve_l1_levels(
                &ctx.body,
                &ctx.lattice,
                &ctx.bed,
                key,
                GAME_RULE,
                &none,
            ) {
                Some(t) => {
                    let spills = t.spills.clone();
                    let (raw, record) =
                        raw_of(key, geom, (t.side, t.quarters, t.water, t.river_share));
                    Answer::Plain(key, Some(Arc::new(raw)), record, spills)
                }
                None => Answer::Plain(key, None, None, Vec::new()),
            }
        }
        Job::Solve(lk @ (level, key), book, levels) => {
            let geom = ctx.geom(level);
            let mut checked = None;
            let raster = match (level, book) {
                (FineLevel::L1, _) => solve_l1_levels(
                    &ctx.body,
                    &ctx.lattice,
                    &ctx.bed,
                    key,
                    GAME_RULE,
                    &levels,
                )
                .map(|t| {
                    checked = Some((Arc::clone(&levels), t.spills));
                    (t.side, t.quarters, t.water, t.river_share)
                }),
                (FineLevel::L2, Some(book)) => solve_l2_with(
                    &ctx.body,
                    &ctx.lattice,
                    &ctx.artifact,
                    &book,
                    key,
                    GAME_RULE,
                )
                .map(|t| (t.side, t.quarters, t.water, t.river_share)),
                (FineLevel::L2, None) => None,
            };
            match raster.map(|r| raw_of(key, geom, r)) {
                Some((raw, record)) => Answer::Raw(lk, Some(Arc::new(raw)), record, checked),
                None => Answer::Raw(lk, None, None, None),
            }
        }
        Job::LoadRaw(lk @ (level, key), bytes) => Answer::Raw(
            lk,
            decode_raw_water(&bytes, key, ctx.geom(level)).map(Arc::new),
            None,
            None,
        ),
        Job::Finish(lk @ (level, _), block) => {
            let geom = ctx.geom(level);
            let slots: [Option<(TileKey, &Field)>; 9] =
                std::array::from_fn(|k| block[k].as_ref().map(|(t, f)| (*t, &f[0])));
            let plane = |p: usize| -> [Option<&Field>; 9] {
                std::array::from_fn(|k| block[k].as_ref().map(|(_, f)| &f[p]))
            };
            let levels = finish_water(slots, plane(1), plane(2), geom);
            let out = levels.and_then(|levels| {
                encode_water_levels(&levels, geom.ts).map(|record| (record, levels))
            });
            let far = far_rows_of(ctx, lk, out.as_ref().map(|(_, l)| l));
            Answer::Finished(lk, out, true, far)
        }
        Job::LoadFinished(lk @ (level, _), bytes) => {
            let ts = ctx.geom(level).ts;
            let levels = decode_water_levels(&bytes, level)
                .filter(|(t, _)| *t == ts)
                .map(|(_, levels)| levels);
            let far = far_rows_of(ctx, lk, levels.as_ref());
            Answer::Finished(lk, levels.map(|levels| (bytes, levels)), false, far)
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
        Job::Solve(..) | Job::Round(..) | Job::Plain(..) => {
            (&meter.raw_solved, Some(&meter.raw_solve_nanos))
        }
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
    raw: BTreeMap<LKey, Arc<RawTile>>,
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
    /// Every tile some view names (whole or for its coarse copies): the rest of the reasons are the far lakes' idle work.
    viewed: BTreeSet<LKey>,
    /// ★ THE LAKES (ruling B115): each lake's settled fine level, the lakes in their rounds, the round solves in flight,
    /// the lakes the store does not hold, and two caches — each 128 m tile's domain lakes, each lake's tiles.
    lake_level: BTreeMap<u32, i32>,
    rounds: BTreeMap<u32, Round>,
    round_flight: BTreeSet<(u32, TileKey)>,
    /// The tiles' plain solves (their spills, and their raw while a served block may still take it as final), those in
    /// flight, and the lakes settled since the store last took them.
    plain: BTreeMap<TileKey, Plain>,
    plain_flight: BTreeSet<TileKey>,
    settled: Vec<(u32, i32)>,
    /// ★ THE FAR LAKES (ruling B116 part 2): whether the work began (the store read, or the lake tiles listed), the lake
    /// tiles still to fold — the realm's idle work, after every tile a view reads — the rows folded so far (each node under
    /// or beside a lake, folded on the workers), and the list's wire parts once whole.
    far_started: bool,
    advance: BTreeSet<LKey>,
    /// Every lake tile of the far list, so a tile finished again after a lake reopens is folded again.
    far_tiles: BTreeSet<LKey>,
    far_rows: BTreeMap<u32, FarNode>,
    far_parts: Option<Arc<[Bytes]>>,
    lakes_unstored: BTreeSet<u32>,
    tile_lakes: BTreeMap<TileKey, Arc<BTreeSet<u32>>>,
    lake_tiles: BTreeMap<u32, Arc<BTreeSet<TileKey>>>,
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
        let bed = if GAME_RULE == OutletRule::Deltas {
            Arc::new(vd_terrain::fine::l1::bedrock_view(&artifact, &lattice))
        } else {
            Arc::clone(&artifact)
        };
        let ctx = Arc::new(Ctx {
            body,
            lattice,
            artifact: Arc::clone(&artifact),
            bed,
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
                viewed: BTreeSet::new(),
                lake_level: BTreeMap::new(),
                rounds: BTreeMap::new(),
                round_flight: BTreeSet::new(),
                plain: BTreeMap::new(),
                plain_flight: BTreeSet::new(),
                settled: Vec::new(),
                far_started: false,
                advance: BTreeSet::new(),
                far_tiles: BTreeSet::new(),
                far_rows: BTreeMap::new(),
                far_parts: None,
                lakes_unstored: BTreeSet::new(),
                tile_lakes: BTreeMap::new(),
                lake_tiles: BTreeMap::new(),
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

    /// A lake's fine level's place in the store (ruling B115; [`LAKE_LEVEL_AT`]).
    fn lake_at(&self, lake: u32) -> FineAt {
        FineAt {
            form: FineForm::Raw,
            input: self.inputs[0],
            level: LAKE_LEVEL_AT,
            face: 0,
            tx: lake,
            ty: 0,
        }
    }

    /// ★ THE LAKE LEVELS A 128 m TILE'S SOLVE NAMES (ruling B115): `Ok` with the lakes of the tile's domain whose fine level
    /// is not their 8 km word, once every lake there has settled; `Err` with the lakes still in their rounds. A level the
    /// store holds is read back; a lake it does not hold starts its rounds.
    fn tile_levels(
        &self,
        state: &mut State,
        store: Option<&dyn Store>,
        key: TileKey,
    ) -> Result<BTreeMap<u32, i32>, Vec<u32>> {
        let none = BTreeMap::new();
        let waters = Waters::new(&self.ctx.lattice, &self.ctx.bed, &none);
        let lakes = Arc::clone(
            state
                .tile_lakes
                .entry(key)
                .or_insert_with(|| Arc::new(waters.domain_lakes(key))),
        );
        let (mut levels, mut waiting) = (BTreeMap::new(), Vec::new());
        for &lake in lakes.iter() {
            if !state.lake_level.contains_key(&lake) && !state.lakes_unstored.contains(&lake) {
                match store.map(|s| read_fine_record(s, &self.lake_at(lake))) {
                    Some(Ok(Some(bytes))) => match bytes[..] {
                        [LAKE_LEVEL_FORM, a, b, c, d] => {
                            state
                                .lake_level
                                .insert(lake, i32::from_le_bytes([a, b, c, d]));
                        }
                        _ => {
                            state.lakes_unstored.insert(lake);
                        }
                    },
                    _ => {
                        state.lakes_unstored.insert(lake);
                    }
                }
            }
            let word = waters.lake_level_q(lake);
            match state.lake_level.get(&lake) {
                Some(&level) => {
                    if level != word {
                        levels.insert(lake, level);
                    }
                }
                None => {
                    state.rounds.entry(lake).or_insert_with(|| Round {
                        level: word,
                        n: 0,
                        got: BTreeMap::new(),
                    });
                    state
                        .lake_tiles
                        .entry(lake)
                        .or_insert_with(|| Arc::new(waters.lake_tiles(lake)));
                    waiting.push(lake);
                }
            }
        }
        if waiting.is_empty() {
            Ok(levels)
        } else {
            Err(waiting)
        }
    }

    /// The jobs in flight: raw and finished work, the lakes' rounds and the plain solves.
    fn busy(state: &State) -> usize {
        state.in_flight.len() + state.round_flight.len() + state.plain_flight.len()
    }

    /// ★ A LAKE'S ROUND IS IN (ruling B115, form (a′)): once every tile of the lake has answered, the lowest spill lowers
    /// the level and the tiles are solved again, until it holds (or the bound ends it at the lowest spill found); a settled
    /// level waits for the store in `settled`.
    fn advance(state: &mut State, lake: u32) {
        let all = state.lake_tiles.get(&lake).map_or(0, |t| t.len());
        let Some(round) = state.rounds.get_mut(&lake) else {
            return;
        };
        if round.got.len() < all {
            return;
        }
        let next = round
            .got
            .values()
            .flatten()
            .copied()
            .fold(round.level, i32::min);
        if next == round.level || round.n + 1 >= LAKE_ROUND_BOUND {
            state.rounds.remove(&lake);
            state.lake_level.insert(lake, next);
            state.settled.push((lake, next));
        } else {
            round.level = next;
            round.n += 1;
            round.got.clear();
        }
    }

    /// The lakes of a final raster that pass under the level its solve named (their 8 km word where it named none), with the
    /// lowest pass of each.
    fn passes_under(&self, levels: &BTreeMap<u32, i32>, spills: &[(u32, i32)]) -> Vec<(u32, i32)> {
        let none = BTreeMap::new();
        let waters = Waters::new(&self.ctx.lattice, &self.ctx.bed, &none);
        let mut out: BTreeMap<u32, i32> = BTreeMap::new();
        for &(lake, pass) in spills {
            let named = levels
                .get(&lake)
                .copied()
                .unwrap_or_else(|| waters.lake_level_q(lake));
            if pass < named {
                let e = out.entry(lake).or_insert(pass);
                *e = (*e).min(pass);
            }
        }
        out.into_iter().collect()
    }

    /// ★ THE OTHER LAKES A ROUND NAMES (the lakes together, 2026-10-19): every other lake of the tile's domain whose settled
    /// level is not its 8 km word. MEASURED before (the owner's window, a 595 m wall of water at a tile corner): a lake's
    /// rounds named that lake alone, so lake 4687251 found no pass at −136 m while its neighbour stood at its 8 km word;
    /// the neighbour's fine level (−731 m) eroded its former floor into valleys, and in the final solve the lake passed at
    /// −549 m — two tiles' solves then gave one valley two waters.
    fn others_of(&self, state: &mut State, tile: TileKey, lake: u32) -> BTreeMap<u32, i32> {
        let none = BTreeMap::new();
        let waters = Waters::new(&self.ctx.lattice, &self.ctx.bed, &none);
        let lakes = Arc::clone(
            state
                .tile_lakes
                .entry(tile)
                .or_insert_with(|| Arc::new(waters.domain_lakes(tile))),
        );
        lakes
            .iter()
            .filter(|&&m| m != lake)
            .filter_map(|&m| {
                let level = *state.lake_level.get(&m)?;
                (level != waters.lake_level_q(m)).then_some((m, level))
            })
            .collect()
    }

    /// ★ A LAKE REOPENS (the lakes together, 2026-10-19): a final raster shows it passing at `pass`, under the level its
    /// solve named, so it drains — its rounds start again at the pass, its level leaves the settled set (and the store is
    /// not read for it), and every 128 m raw that named its old level goes, with the finished tiles that read them and the
    /// 16 m tiles that read those; each is made again (and stored over the old) and a lake tile is folded again into the
    /// far list. Levels only fall, so the lakes settle.
    fn reopen(&self, state: &mut State, lake: u32, pass: i32) {
        let none = BTreeMap::new();
        let waters = Waters::new(&self.ctx.lattice, &self.ctx.bed, &none);
        let was = state.lake_level.remove(&lake);
        tracing::info!(
            lake,
            was_q = ?was,
            pass_q = pass,
            "a lake passes under its level in a final solve: it drains to the pass (the lakes together, B115)"
        );
        state.lakes_unstored.insert(lake);
        state.rounds.insert(
            lake,
            Round {
                level: pass,
                n: 1,
                got: BTreeMap::new(),
            },
        );
        state
            .lake_tiles
            .entry(lake)
            .or_insert_with(|| Arc::new(waters.lake_tiles(lake)));
        let raws: Vec<LKey> = state
            .tile_lakes
            .iter()
            .filter(|(_, lakes)| lakes.contains(&lake))
            .map(|(&t, _)| (FineLevel::L1, t))
            .collect();
        let mut finished: BTreeSet<LKey> = BTreeSet::new();
        for raw in &raws {
            state.raw.remove(raw);
            state.unstored.insert((*raw, false));
            if let Some(plain) = state.plain.get_mut(&raw.1) {
                plain.raw = None;
            }
            finished.extend(self.block_of(*raw).into_iter().flatten());
        }
        let finer: Vec<LKey> = state
            .raw
            .keys()
            .chain(state.finished.keys())
            .copied()
            .filter(|&k| k.0 == FineLevel::L2 && self.reads(k).iter().any(|r| finished.contains(r)))
            .collect();
        for k in finer {
            state.raw.remove(&k);
            state.unstored.insert((k, false));
            finished.insert(k);
        }
        for key in finished {
            if state.finished.remove(&key).is_some() || state.unstored.contains(&(key, true)) {
                state.wire.retain(|(k, _, _), _| *k != key);
                Arc::make_mut(&mut state.books[slot(key.0)]).remove(key.1);
                state.book_changed.insert(key);
            }
            state.unstored.insert((key, true));
            if state.far_tiles.contains(&key) {
                state.advance.insert(key);
                state.far_parts = None;
            }
        }
    }

    /// Hands out a lake's round while threads are free: round 0 reads each tile's plain solve (and asks once for each plain
    /// solve not made yet, for every lake at once) where no other lake of the tile's domain stands lowered, a later round
    /// (or a round 0 beside a lowered lake) solves the tile with the lake at its level and the others at theirs.
    fn round_work(&self, state: &mut State, lake: u32, threads: usize) {
        let (Some((level, n)), Some(tiles)) = (
            state.rounds.get(&lake).map(|r| (r.level, r.n)),
            state.lake_tiles.get(&lake).map(Arc::clone),
        ) else {
            return;
        };
        for &tile in tiles.iter() {
            if state
                .rounds
                .get(&lake)
                .is_none_or(|r| r.got.contains_key(&tile))
            {
                continue;
            }
            let others = self.others_of(state, tile, lake);
            if n == 0
                && others.is_empty()
                && let Some(plain) = state.plain.get(&tile)
            {
                let spill = plain
                    .spills
                    .iter()
                    .find(|&&(l, _)| l == lake)
                    .map(|&(_, p)| p);
                if let Some(round) = state.rounds.get_mut(&lake) {
                    round.got.insert(tile, spill);
                }
                continue;
            }
            if Self::busy(state) >= threads {
                continue;
            }
            if n == 0 && others.is_empty() {
                if state.plain_flight.insert(tile) {
                    Self::submit(state, &self.ctx, &self.meter, Job::Plain(tile));
                }
            } else if state.round_flight.insert((lake, tile)) {
                Self::submit(
                    state,
                    &self.ctx,
                    &self.meter,
                    Job::Round(lake, level, tile, Arc::new(others)),
                );
            }
        }
        Self::advance(state, lake);
    }

    /// The far lake list's wire parts: the record cut at the coast's size, each a `BulkMsg::ArtifactLakes`.
    fn far_parts_of(&self, record: &[u8]) -> Arc<[Bytes]> {
        let parts = record.len().div_ceil(COAST_PART_BYTES).max(1);
        (0..parts)
            .map(|part| {
                let start = (part * COAST_PART_BYTES).min(record.len());
                let end = (start + COAST_PART_BYTES).min(record.len());
                let msg = BulkMsg::ArtifactLakes {
                    realm: self.realm,
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

    /// ★ THE FAR LAKES BEGIN (ruling B116 part 2), once, on the first tick with the realm's store: the list the store keeps
    /// under this fine input word, or else every 128 m tile that holds a node under or beside a lake, listed as the
    /// realm's idle work.
    fn far_begin(&self, state: &mut State, store: Option<&dyn Store>) {
        let Some(store) = store.filter(|_| !state.far_started) else {
            return;
        };
        state.far_started = true;
        if let Some(record) = store.get(&far_lakes_key(self.inputs[0])) {
            state.far_parts = Some(self.far_parts_of(&record));
            tracing::info!(
                bytes = record.len(),
                "the far lakes read back from the store (ruling B116)"
            );
            return;
        }
        let (lattice, artifact) = (&self.ctx.lattice, &self.ctx.bed);
        let none = BTreeMap::new();
        let waters = Waters::new(lattice, artifact, &none);
        let mut keep = BTreeSet::new();
        for node in 0..artifact.rows.len() as u32 {
            if waters.under_lake(node) {
                keep.insert(node);
                for m in lattice.neighbours(node) {
                    if m != vd_terrain::macro_lattice::NO_NODE {
                        keep.insert(m);
                    }
                }
            }
        }
        let shift = FineLevel::L1.tile_nodes_shift();
        state.advance = keep
            .iter()
            .map(|&node| {
                let (face, ni, nj) = lattice.split(node);
                (
                    FineLevel::L1,
                    TileKey {
                        face,
                        tx: ni >> shift,
                        ty: nj >> shift,
                    },
                )
            })
            .collect();
        state.far_tiles = state.advance.clone();
        tracing::info!(
            nodes = keep.len(),
            tiles = state.advance.len(),
            "the far lakes begin: the lake tiles join the realm's idle work (ruling B116)"
        );
    }

    /// ★ ONE LAKE TILE FOLDED (ruling B116 part 2): its kept nodes' rows from its finished levels; the list is whole when the
    /// last lake tile is folded.
    fn far_fold(state: &mut State, key: LKey, rows: Vec<FarNode>) {
        if !state.advance.remove(&key) {
            return;
        }
        for row in rows {
            state.far_rows.insert(row.node, row);
        }
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
                Answer::Raw(key, field, record, checked) => {
                    state.in_flight.remove(&(key, false));
                    // ★ THE LAKES TOGETHER (2026-10-19): a lake passing under the level this final solve named drains
                    // there — the solve is not kept, the lake reopens, and the tile waits for its level again.
                    if let Some((levels, spills)) = &checked {
                        let lower = self.passes_under(levels, spills);
                        if !lower.is_empty() {
                            for (lake, pass) in lower {
                                self.reopen(state, lake, pass);
                            }
                            continue;
                        }
                    }
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
                Answer::Round(lake, level, tile, spill) => {
                    state.round_flight.remove(&(lake, tile));
                    if let Some(round) = state.rounds.get_mut(&lake).filter(|r| r.level == level) {
                        round.got.insert(tile, spill);
                        Self::advance(state, lake);
                    }
                }
                Answer::Plain(tile, raw, record, spills) => {
                    state.plain_flight.remove(&tile);
                    // Round 0 of every lake in the raster reads this one solve — where no other lake of the tile's
                    // domain stands lowered (else its round names them).
                    let candidates: Vec<u32> = state
                        .rounds
                        .iter()
                        .filter(|(lake, r)| {
                            r.n == 0
                                && state
                                    .lake_tiles
                                    .get(lake)
                                    .is_some_and(|t| t.contains(&tile))
                        })
                        .map(|(&lake, _)| lake)
                        .collect();
                    let waiting: Vec<u32> = candidates
                        .into_iter()
                        .filter(|&lake| self.others_of(state, tile, lake).is_empty())
                        .collect();
                    for lake in waiting {
                        let spill = spills.iter().find(|&&(l, _)| l == lake).map(|&(_, p)| p);
                        if let Some(round) = state.rounds.get_mut(&lake) {
                            round.got.insert(tile, spill);
                        }
                        Self::advance(state, lake);
                    }
                    state.plain.insert(
                        tile,
                        Plain {
                            spills,
                            raw: raw.map(|r| (r, record)),
                        },
                    );
                }
                Answer::Finished(key, out, store_it, far) => {
                    state.in_flight.remove(&(key, true));
                    match out {
                        Some((record, levels)) => {
                            if let Some(store) = store.as_deref_mut().filter(|_| store_it) {
                                write_fine_record(store, &self.at(key, true), &record);
                                state.unstored.remove(&(key, true));
                                wrote = true;
                            }
                            Self::far_fold(state, key, far);
                            state.finished.insert(key, Arc::new(record));
                            Arc::make_mut(&mut state.books[slot(key.0)])
                                .insert_water_from(key.1, 0, levels);
                            state.book_changed.insert(key);
                        }
                        None => {
                            state.refused.insert((key, true));
                            state.advance.remove(&key);
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
        // ★ THE FAR LAKES' TILES LAST (ruling B116 part 2): the list begins on the first tick with the store; a lake tile
        // whose finish or block was refused leaves the list; the rest follow every tile a view names, as idle work.
        self.far_begin(state, store.as_deref());
        let refused: Vec<LKey> = state
            .advance
            .iter()
            .copied()
            .filter(|&k| {
                state.refused.contains(&(k, true))
                    || self
                        .block_of(k)
                        .iter()
                        .flatten()
                        .any(|b| state.refused.contains(&(*b, false)))
            })
            .collect();
        for k in refused {
            state.advance.remove(&k);
        }
        let advance: Vec<LKey> = state
            .advance
            .iter()
            .copied()
            .filter(|k| seen.insert(*k))
            .collect();
        state.reasons.extend(advance);
        state.whole = named
            .iter()
            .flatten()
            .filter_map(|&(key, whole)| whole.then_some(key))
            .collect();
        state.viewed = named.iter().flatten().map(|&(key, _)| key).collect();
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
        for (tile, plain) in &mut state.plain {
            if !keep_raw.contains(&(FineLevel::L1, *tile)) {
                plain.raw = None;
            }
        }
        // ★ THE NEXT WORK, nearest first, never more in flight than threads (the order follows the newest reasons).
        let reasons = state.reasons.clone();
        for key in reasons {
            if Self::busy(state) >= threads {
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
            // Idle work (a far lake tile no view names) fills only its share of the threads.
            let limit = if state.viewed.contains(&key) {
                threads
            } else {
                threads.div_ceil(IDLE_THREADS_DIVISOR)
            };
            if Self::busy(state) >= limit {
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
                if Self::busy(state) >= limit {
                    break;
                }
                if state.raw.contains_key(&raw) | state.in_flight.contains(&(raw, false)) {
                    continue;
                }
                let stored = if state.unstored.contains(&(raw, false)) {
                    None
                } else {
                    store
                        .as_deref()
                        .and_then(|s| read_fine_record(s, &self.at(raw, false)).ok().flatten())
                };
                if let Some(bytes) = stored {
                    state.in_flight.insert((raw, false));
                    Self::submit(state, &self.ctx, &self.meter, Job::LoadRaw(raw, bytes));
                    continue;
                }
                state.unstored.insert((raw, false));
                // ★ THE LAKES FIRST (ruling B115): a 128 m tile solves once every lake in its domain has its fine level;
                // meanwhile those lakes' rounds take the free threads.
                let levels = if raw.0 == FineLevel::L1 {
                    match self.tile_levels(state, store.as_deref(), raw.1) {
                        Ok(levels) => levels,
                        Err(waiting) => {
                            for lake in waiting {
                                self.round_work(state, lake, limit);
                            }
                            continue;
                        }
                    }
                } else {
                    BTreeMap::new()
                };
                // ★ THE PLAIN SOLVE IS FINAL where no lake of the tile's domain stands lower than its 8 km word — and no
                // lake of its raster passes under its word there (the lakes together).
                if levels.is_empty() && raw.0 == FineLevel::L1 {
                    let lower = state
                        .plain
                        .get(&raw.1)
                        .map(|p| self.passes_under(&BTreeMap::new(), &p.spills))
                        .unwrap_or_default();
                    if !lower.is_empty() {
                        for (lake, pass) in lower {
                            self.reopen(state, lake, pass);
                        }
                        continue;
                    }
                }
                if levels.is_empty()
                    && raw.0 == FineLevel::L1
                    && let Some((field, record)) =
                        state.plain.get_mut(&raw.1).and_then(|p| p.raw.take())
                {
                    if let (Some(store), Some(record)) = (store.as_deref_mut(), record) {
                        write_fine_record(store, &self.at(raw, false), &record);
                        state.unstored.remove(&(raw, false));
                        store.commit();
                    }
                    state.raw.insert(raw, field);
                    continue;
                }
                state.in_flight.insert((raw, false));
                let book = (raw.0 == FineLevel::L2).then(|| Arc::clone(&state.books[0]));
                Self::submit(
                    state,
                    &self.ctx,
                    &self.meter,
                    Job::Solve(raw, book, Arc::new(levels)),
                );
            }
        }
        // ★ THE FAR LAKES WHOLE (ruling B116 part 2): the last lake tile folded, the list is coded, kept and offered.
        if state.far_started && state.far_parts.is_none() && state.advance.is_empty() {
            let rows: Vec<FarNode> = state.far_rows.values().copied().collect();
            let record = encode_far_nodes(&rows);
            if let Some(store) = store.as_deref_mut() {
                store.put(&far_lakes_key(self.inputs[0]), &record.clone().into());
                store.commit();
            }
            tracing::info!(
                nodes = rows.len(),
                bytes = record.len(),
                "the far lakes are whole: kept and offered to the gateways (ruling B116)"
            );
            state.far_parts = Some(self.far_parts_of(&record));
        }
        // ★ THE SETTLED LAKES TO THE STORE (ruling B115): each level once, so a later tile beside a known lake reads it back.
        if let Some(store) = store
            && !state.settled.is_empty()
        {
            for (lake, level) in std::mem::take(&mut state.settled) {
                let b = level.to_le_bytes();
                write_fine_record(
                    store,
                    &self.lake_at(lake),
                    &[LAKE_LEVEL_FORM, b[0], b[1], b[2], b[3]],
                );
                state.lakes_unstored.remove(&lake);
            }
            store.commit();
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
            .store(Self::busy(state) as u64, Ordering::Relaxed);
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

    fn far_lakes(&self) -> Option<Arc<[Bytes]>> {
        self.lock().far_parts.clone()
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
        let parts = self.wire_parts(key, &water_range_record(&record, u32::from(lo), end)?);
        state.wire.insert((key, lo, hi), Arc::clone(&parts));
        Some(parts)
    }
}
