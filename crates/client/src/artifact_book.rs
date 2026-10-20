//! ★ THE ARTIFACT BOOK — the client's side of the artifact ship (the landform arc, slice 8c stage
//! C4c; the owner's ruling of 2026-09-19: the solve runs once on the server, its artifact is saved
//! in the shard's db and shipped to every client).
//!
//! A realm's shard ships its artifact in three shapes, relayed by the gateway as
//! `ServerControlMsg::ArtifactPart`: the HEAD (what the artifact is), the PYRAMID's levels in parts
//! (the globe from orbit), and TILES of node rows (the valley under the boots). The book assembles
//! them per realm and hands the chunk builder a FIELD to read: a pyramid level for a coarse rung,
//! the tiles for a fine one. A chunk whose field is not whole yet WAITS while the coarser rung
//! stands (ruling F9); a part for a realm whose head has not arrived is refused and counted; a part
//! of the wrong shape is refused and counted; nothing here is guessed.
//!
//! The book is SHARED with the renderer by pointer (`Arc<ArtifactBook>` on the render snapshot) and
//! replaced by copy-on-write as parts land: a cache's levels and tiles sit behind their own `Arc`s,
//! so the copy is pointer bumps, never rows. The parts still assembling live beside the book, on
//! the receiver alone.
//!
//! **Example.** A pilot logs in aboard a hull in orbit of the home planet. The planet's head
//! arrives, then its six pyramid levels over a second, and the client states `ArtifactHeld`; the
//! renderer's coarse chunks read level 6 first (the coarsest, shipped first) and the finer levels as
//! they land. She lands: the tiles under her boots arrive four a tick, and each fine chunk builds
//! the moment every tile its stencil reads is here.

use std::collections::BTreeMap;
use std::sync::Arc;

use vd_core::pose::RealmId;
use vd_terrain::artifact::{
    CoastCounts, PyramidField, Tile, TileCache, ZField, tile_width, tiles_of_chunk,
};
use vd_terrain::chunk::ChunkKey;
use vd_terrain::fine::l1::TileKey;
use vd_terrain::fine::level::FineLevel;
use vd_terrain::fine::read::{FineBook, FineField, L1_TOP_RUNG, applies};
use vd_terrain::macro_lattice::MacroLattice;
use vd_wire::channels::BulkMsg;

/// What a realm's head stated about its artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactHead {
    pub world_tag: u64,
    pub version: u32,
    pub edge: u32,
    pub digest: [u64; 2],
    pub tiles_per_edge: u32,
    pub levels: u32,
    /// The sea's level, whole metres over the ladder radius, or `i16::MIN` for a dry body (C5).
    pub sea_m: i16,
    /// ★ HOW MANY COAST PARTS FOLLOW (2026-09-22, ruling W10): the parts of the realm's coast
    /// mask. The cache is whole only when every one is here.
    pub coast_parts: u32,
    /// ★ THE DRAWN GROUND'S RANGE (ruling B55): the lowest and highest height the realm's macro
    /// field can draw, whole metres over the ladder radius — what the body's band is sized from.
    pub low_m: i32,
    pub high_m: i32,
    /// ★ HOW MANY SEA-WIND PARTS FOLLOW (ruling B109): the parts of the realm's year surface wind; zero where it ships
    /// none. The cache is whole only when every one is here.
    pub wind_parts: u32,
}

impl ArtifactHead {
    /// The sea as the head states it: `None` for a dry body.
    #[must_use]
    pub fn sea(&self) -> Option<i32> {
        (self.sea_m != vd_terrain::artifact::DRY_M).then_some(i32::from(self.sea_m))
    }
}

/// A field a chunk build reads, shared with the builders.
pub type SharedField = Arc<dyn ZField + Send + Sync>;

/// What the book answers a chunk that asks for its field.
#[derive(Clone)]
pub enum FieldPick {
    /// The field, whole for this chunk: build.
    Ready(SharedField),
    /// ★ THE ROUGH GROUND (ruling B86 item 7, the safety net): the fine level serves this chunk but a fine tile it reads
    /// has not landed. Build now on the field under the fine ground (the macro rows and their octaves), on the cache's
    /// rough view ([`ArtifactCache::rough_view`]), and again when the tiles land: the rough ground stands only where no
    /// fine ground has arrived, never over it.
    Rough(SharedField),
    /// ★ THE 128 m GROUND UNDER THE 16 m LEVEL (ruling B95): the 16 m level serves this chunk but a 16 m tile it reads
    /// has not landed, while every 128 m tile it reads has. Build now on today's 128 m ground (the cache's coarse view,
    /// [`ArtifactCache::coarse_view`]) and again when the 16 m tiles land — never the 8 km ground, and never a reason for
    /// the login's loading screen to wait.
    Coarse(SharedField),
    /// The level is still assembling, or a tile the stencil reads has not arrived: the coarser
    /// rung stands.
    Waiting,
}

impl std::fmt::Debug for FieldPick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldPick::Ready(field) => write!(f, "Ready(level {})", field.level()),
            FieldPick::Rough(field) => write!(f, "Rough(level {})", field.level()),
            FieldPick::Coarse(field) => write!(f, "Coarse(level {})", field.level()),
            FieldPick::Waiting => write!(f, "Waiting"),
        }
    }
}

/// One realm's artifact as the client holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactCache {
    pub head: ArtifactHead,
    /// Level `k` at index `k − 1`; `None` while its parts assemble.
    pyramid: Vec<Option<Arc<PyramidField>>>,
    tiles: Arc<TileCache>,
    /// ★ THE COAST MASK (2026-09-22, ruling W10): one bit per FINE macro node, set where the node
    /// stands at or under the sea; `None` while its parts assemble. Every pyramid level shares it
    /// by pointer, so a far rung reads the water's side from the fine row's own word and the
    /// shoreline stands in one place at every rung.
    coast: Option<Arc<[u8]>>,
    /// ★ THE COAST COUNTS (2026-09-22, ruling W15): the wet-node counts folded from the mask the
    /// moment the mask is whole, shared by pointer with every level. They are DERIVED, so no byte
    /// of them crosses the wire: the client folds what the server folds, from one mask.
    counts: Option<Arc<CoastCounts>>,
    /// ★ THE SEA'S WIND (ruling B109): the year's surface wind of each climate cell, `None` while its parts assemble or
    /// where the head announces none.
    wind: Option<Arc<vd_terrain::sea_wind::SeaWind>>,
    /// ★ THE FINE TILES (the fine ground's way in, B2): the finished 128 m tiles the realm's shard shipped, `None` until
    /// the first lands (it states the tile's side). Shared by pointer; each tile's heights by pointer inside it.
    fine: Option<Arc<FineBook>>,
    /// ★ THE 16 m TILES (ruling B95): the finished 16 m tiles the realm's shard shipped, `None` until the first lands.
    finer: Option<Arc<FineBook>>,
    /// ★ A ROUGH VIEW (B86 item 7): the fine level is off — every rung reads the field under the fine ground. A chunk
    /// whose fine tiles have not landed is built on this view and built again on the whole cache when they land.
    rough: bool,
    /// ★ A COARSE VIEW (ruling B95): the 16 m level is off — every rung reads the 128 m ground at most. A chunk whose 16 m
    /// tiles have not landed is built on this view and built again on the whole cache when they land.
    coarse: bool,
    /// ★ THE FAR LAKES APPLIED (ruling B116 part 2): for each pyramid level `k` at index `k − 1`, the blocks the far lake
    /// list folded into it (their heights and water words now the list's majority class); `None` until the list lands.
    far: Option<Arc<Vec<std::collections::BTreeSet<u32>>>>,
    /// The far lake list's rows themselves, by node (each node under or beside a lake on the 128 m ground): a piece read
    /// on the 8 km rows takes its lakes from them.
    far_rows: Option<Arc<Vec<vd_terrain::fine::far::FarNode>>>,
    /// Bumped on every accepted change, so a reader tells a changed cache from the one it read.
    pub epoch: u64,
}

impl ArtifactCache {
    /// ★ Whether the far lake list is folded into the far levels (ruling B116 part 2).
    #[must_use]
    pub fn far_applied(&self) -> bool {
        self.far.is_some()
    }

    /// ★ The far lake list's row of an 8 km node (ruling B116 part 2): its 128 m wet share, water level and wet floor;
    /// `None` for a node no lake touches, or before the list lands.
    #[must_use]
    pub fn far_node(&self, node: u32) -> Option<vd_terrain::fine::far::FarNode> {
        let rows = self.far_rows.as_ref()?;
        rows.binary_search_by_key(&node, |r| r.node)
            .ok()
            .map(|i| rows[i])
    }

    /// ★ The far lake list's row count once folded (ruling B116 part 2), for the dev state.
    #[must_use]
    pub fn far_rows_count(&self) -> Option<usize> {
        self.far_rows.as_ref().map(|rows| rows.len())
    }

    /// ★ Whether pyramid level `k`'s block `block` stands by the far lake list (ruling B116 part 2): its water is the
    /// list's, whatever side the fine node under a place states.
    #[must_use]
    pub fn far_block(&self, k: u32, block: u32) -> bool {
        self.far.as_ref().is_some_and(|f| {
            k.checked_sub(1)
                .and_then(|i| f.get(i as usize))
                .is_some_and(|s| s.contains(&block))
        })
    }

    /// The fine tiles held, if any has landed.
    #[must_use]
    pub fn fine(&self) -> Option<&Arc<FineBook>> {
        self.fine.as_ref()
    }

    /// The 16 m tiles held, if any has landed.
    #[must_use]
    pub fn finer(&self) -> Option<&Arc<FineBook>> {
        self.finer.as_ref()
    }

    /// ★ THE FINE GROUND OVER A PICKED FIELD (B2; ruling B95): on a body where the fine levels apply, a chunk at rungs
    /// 0–12 reads its height from the 128 m tiles once every tile its stencils read is here, and until then is ROUGH
    /// (ruling B86 item 7: built on the field under the fine ground, and again when the tiles land); at the rungs the 16 m
    /// level serves (0–6) it reads the 16 m tiles once every one its stencils read is here, and until then is COARSE
    /// (built on the 128 m ground, and again when they land). Any other pick, and every pick of a rough view, stands.
    fn with_fine(&self, pick: FieldPick, key: ChunkKey) -> FieldPick {
        if self.rough || !applies(self.head.sea_m) || key.rung > L1_TOP_RUNG {
            return pick;
        }
        let FieldPick::Ready(inner) = pick else {
            return pick;
        };
        let holds = |book: &FineBook| {
            book.holds_all(
                &book.tiles_of_chunk(key.face, key.rung, key.x, key.y),
                key.rung,
            )
        };
        let Some(book) = self.fine.as_ref().filter(|b| holds(b)) else {
            return FieldPick::Rough(inner);
        };
        let field = |finer: Option<Arc<FineBook>>| {
            Arc::new(FineField {
                inner: Arc::clone(&inner),
                book: Arc::clone(book),
                finer,
            }) as SharedField
        };
        if self.coarse || key.rung > FineLevel::L2.top_rung() {
            return FieldPick::Ready(field(None));
        }
        match self.finer.as_ref().filter(|f| holds(f)) {
            Some(finer) => FieldPick::Ready(field(Some(Arc::clone(finer)))),
            None => FieldPick::Coarse(field(None)),
        }
    }

    /// ★ THE ROUGH VIEW of this cache (B86 item 7): the same head, levels and tiles with the fine level off — what a
    /// chunk whose fine tiles have not landed is built on, so its own field, its parents' and the fields its morph reads
    /// are all the one ground under the fine ground.
    #[must_use]
    pub fn rough_view(&self) -> ArtifactCache {
        ArtifactCache {
            fine: None,
            finer: None,
            rough: true,
            ..self.clone()
        }
    }

    /// ★ THE COARSE VIEW of this cache (ruling B95): the same cache with the 16 m level off — what a chunk whose 16 m
    /// tiles have not landed is built on, so its own field, its parents' and its morph's are the one 128 m ground.
    #[must_use]
    pub fn coarse_view(&self) -> ArtifactCache {
        ArtifactCache {
            finer: None,
            coarse: true,
            ..self.clone()
        }
    }

    /// Whether this cache is a coarse view.
    #[must_use]
    pub fn is_coarse_view(&self) -> bool {
        self.coarse
    }

    /// ★ THE 16 m GROUND IS NOT HERE YET (ruling B95): whether the 16 m level serves the column `(face, rung, x, y)` of a
    /// whole cache on a body where the fine levels apply, and one of the 16 m tiles its stencils read has not landed — a
    /// span read for it now stands on the 128 m ground, provisional, and is read again until they land.
    #[must_use]
    pub fn finer_waits(&self, face: vd_seed::bend::Face, rung: u8, x: i32, y: i32) -> bool {
        let serves = !self.rough
            & !self.coarse
            & applies(self.head.sea_m)
            & (rung <= FineLevel::L2.top_rung());
        let held = self
            .finer
            .as_ref()
            .is_some_and(|b| b.holds_all(&b.tiles_of_chunk(face, rung, x, y), rung));
        serves & !held
    }

    /// Whether a chunk built now on this cache would stand on the 128 m ground under the 16 m level: a coarse view, or
    /// the 16 m level serving the chunk with one of its 16 m tiles not landed.
    #[must_use]
    pub fn is_coarse(&self, lattice: &MacroLattice, key: ChunkKey) -> bool {
        self.coarse || matches!(self.field_for(lattice, key), FieldPick::Coarse(_))
    }

    /// Whether this cache is a rough view.
    #[must_use]
    pub fn is_rough_view(&self) -> bool {
        self.rough
    }

    /// Whether a chunk built now on this cache would stand on the rough ground: a rough view, or the fine level serving
    /// the chunk with one of its fine tiles not landed. `lattice` is the body's own macro lattice.
    #[must_use]
    pub fn is_rough(&self, lattice: &MacroLattice, key: ChunkKey) -> bool {
        self.rough || matches!(self.field_for(lattice, key), FieldPick::Rough(_))
    }
    /// The tiles this cache holds, by id (2026-09-22, the dev state's listing).
    pub fn tile_ids(&self) -> impl Iterator<Item = (u8, u32, u32)> + '_ {
        self.tiles.tile_ids()
    }

    fn new(head: ArtifactHead) -> ArtifactCache {
        ArtifactCache {
            head,
            pyramid: vec![None; head.levels as usize],
            // The head's own sea, so a held row names a pocket's floor as the mask does.
            tiles: Arc::new(TileCache::new(head.edge).with_sea(head.sea_m)),
            coast: None,
            counts: None,
            wind: None,
            fine: None,
            finer: None,
            rough: false,
            coarse: false,
            far: None,
            far_rows: None,
            epoch: 1,
        }
    }

    /// Whether every pyramid level AND the coast mask are here: the far view is whole, and the
    /// client says so. A level without the mask draws a shoreline the next rung does not share,
    /// which is the crawl the owner saw from 1 400 km, so the mask counts.
    #[must_use]
    pub fn whole(&self) -> bool {
        self.pyramid.iter().all(Option::is_some) && self.coast_held() && self.wind_held()
    }

    /// ★ Whether the sea's wind is here (ruling B109); a head that announces no part has none to wait for.
    #[must_use]
    pub fn wind_held(&self) -> bool {
        self.wind.is_some() || self.head.wind_parts == 0
    }

    /// The sea's wind, to share by pointer; `None` while its parts assemble or where none ships.
    #[must_use]
    pub fn wind(&self) -> Option<&Arc<vd_terrain::sea_wind::SeaWind>> {
        self.wind.as_ref()
    }

    /// Whether the coast mask is here. A head that announces NO part has no mask to wait for and
    /// says yes at once — which no solved body does (every lattice has nodes), so this arm is the
    /// stated answer for a head that states nothing, never a hole the cache hides.
    #[must_use]
    pub fn coast_held(&self) -> bool {
        self.coast.is_some() || self.head.coast_parts == 0
    }

    /// The coast mask, to share by pointer; `None` while its parts assemble.
    #[must_use]
    pub fn coast(&self) -> Option<&Arc<[u8]>> {
        self.coast.as_ref()
    }

    /// How many levels are here.
    #[must_use]
    pub fn levels_held(&self) -> usize {
        self.pyramid.iter().filter(|l| l.is_some()).count()
    }

    /// The tiles held.
    #[must_use]
    pub fn tiles(&self) -> &Arc<TileCache> {
        &self.tiles
    }

    /// Pyramid level `k` (from 1), if whole.
    #[must_use]
    pub fn level(&self, k: u32) -> Option<&Arc<PyramidField>> {
        self.pyramid.get(k.checked_sub(1)? as usize)?.as_ref()
    }

    /// ★ THE FIELD A CHUNK READS: the pyramid level its rung stands at (whole, or waiting), or the
    /// tiles when the rung reads the rows themselves — ready only when every tile the chunk's
    /// stencil reads is here. `lattice` is the body's own macro lattice (level zero).
    ///
    /// ★ A PYRAMID LEVEL WAITS FOR THE COAST MASK THE HEAD ANNOUNCES (2026-09-25, review finding 2
    /// of plan step 2; ruling W16: a rung boundary is invisible; ruling B7 item 1). While the shore
    /// law read the side, a level with no mask let the ground's own sign decide it: every dry
    /// pocket under the sea's level — the largest 1.57 million km² — was drawn as ocean until the
    /// mask landed, and then turned into a dry basin 11.9 km deep. [`ArtifactCache::whole`] states
    /// that the far view is not whole without the mask; the build obeys it, and the coarser rung
    /// stands (ruling F9). ★ Since ruling B22 (2026-09-28) no column reads the side and the shard
    /// announces no coast part, so a level is ready the moment its heights are whole.
    ///
    /// **Example.** A pilot logs in over the home planet. The head announces no mask, so the far
    /// chunks build as soon as the pyramid lands, and the salt pan in the dry interior is drawn as
    /// the dry ground it is.
    #[must_use]
    pub fn field_for(&self, lattice: &MacroLattice, key: ChunkKey) -> FieldPick {
        let level = PyramidField::level_for(lattice, self.head.levels, key.rung);
        let pick = if level == 0 {
            let needed = tiles_of_chunk(lattice, key);
            if self.tiles.holds_all(&needed) {
                FieldPick::Ready(self.tiles.clone() as SharedField)
            } else {
                FieldPick::Waiting
            }
        } else {
            match self.masked_level(level) {
                Some(field) => FieldPick::Ready(field.clone() as SharedField),
                None => FieldPick::Waiting,
            }
        };
        self.with_fine(pick, key)
    }

    /// Pyramid level `k` where it is whole AND the coast mask is held: the level a chunk and every
    /// reader of a rung's field may read (review finding 2 of plan step 2).
    fn masked_level(&self, k: u32) -> Option<&Arc<PyramidField>> {
        self.level(k).filter(|_| self.coast_held())
    }

    /// ★ THE TILES A CHUNK STILL WAITS FOR (the coast flight's hole instrument, 2026-09-20): the
    /// tiles of its stencil the cache does not hold, at a rung that reads the rows; none at a rung
    /// that reads a level. What a refused request is NAMED with on the stamp.
    #[must_use]
    pub fn missing_tiles(&self, lattice: &MacroLattice, key: ChunkKey) -> Vec<(u8, u32, u32)> {
        if PyramidField::level_for(lattice, self.head.levels, key.rung) != 0 {
            return Vec::new();
        }
        tiles_of_chunk(lattice, key)
            .into_iter()
            .filter(|&(f, tx, ty)| !self.tiles.holds(f, tx, ty))
            .collect()
    }

    /// The field a RUNG reads along any direction (the geomorph's parent fallback): the level,
    /// whole with its coast mask or not here (the rule [`ArtifactCache::field_for`] builds with); or
    /// the tiles, which answer nothing for a node they lack.
    #[must_use]
    pub fn field_at_rung(&self, lattice: &MacroLattice, rung: u8) -> Option<SharedField> {
        let field = self.rough_at_rung(lattice, rung)?;
        // ★ THE FINE GROUND (B2) under a rung it serves, where it applies: the fine field once a tile has landed (a tile not
        // held answers nothing for its cells, as the tiles do for a node they lack), and NOTHING before — the rung's ground
        // is not here yet, so every reader stands on a coarser level as a provisional answer and reads again. MEASURED
        // before (the first window stand, 2026-10-12): a column's span read on the coarse field before the first fine tile
        // was kept as final; where the fine ground stood 1.1 km lower, 3 843 of 3 989 chunks were empty and 1 480 columns
        // were holes.
        let serves = !self.rough & applies(self.head.sea_m) & (rung <= L1_TOP_RUNG);
        // The 16 m book under the rungs it serves, where it has landed and the view is not coarse (ruling B95).
        let finer = self
            .finer
            .as_ref()
            .filter(|_| !self.coarse & (rung <= FineLevel::L2.top_rung()))
            .map(Arc::clone);
        match (serves, self.fine.as_ref()) {
            (false, _) => Some(field),
            (true, Some(book)) => Some(Arc::new(FineField {
                inner: field,
                book: Arc::clone(book),
                finer,
            }) as SharedField),
            (true, None) => None,
        }
    }

    /// ★ THE ROUGH GROUND A RUNG READS (B86 item 7): the field under the fine ground — the level, whole with its coast
    /// mask, or the tiles — whatever fine tiles have landed. What a rough chunk is built on, and the span its column is
    /// read on until the fine tiles land.
    #[must_use]
    pub fn rough_at_rung(&self, lattice: &MacroLattice, rung: u8) -> Option<SharedField> {
        let level = PyramidField::level_for(lattice, self.head.levels, rung);
        if level == 0 {
            Some(self.tiles.clone() as SharedField)
        } else {
            self.masked_level(level).map(|f| f.clone() as SharedField)
        }
    }
}

/// The caches per realm, shared with the renderer by pointer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArtifactBook {
    realms: BTreeMap<RealmId, ArtifactCache>,
}

impl ArtifactBook {
    /// A realm's cache, if its head arrived.
    #[must_use]
    pub fn get(&self, realm: RealmId) -> Option<&ArtifactCache> {
        self.realms.get(&realm)
    }

    /// How many realms have a head.
    #[must_use]
    pub fn len(&self) -> usize {
        self.realms.len()
    }

    /// Whether no realm has a head.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.realms.is_empty()
    }

    /// The realms held.
    pub fn realms(&self) -> impl Iterator<Item = (RealmId, &ArtifactCache)> {
        self.realms.iter().map(|(r, c)| (*r, c))
    }
}

/// What one part did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactIngest {
    /// A head for a realm that had none, or a head with a NEW digest (the cache starts over).
    Head,
    /// A head equal to the one held: nothing changes.
    SameHead,
    /// A pyramid part kept; the level is not whole yet.
    Part,
    /// A pyramid part that made its level whole; other levels are still owed.
    LevelWhole,
    /// A pyramid part that made the LAST level whole: the far view is whole, and the client
    /// states `ArtifactHeld { realm, digest }`.
    PyramidWhole { realm: RealmId, digest: [u64; 2] },
    /// A tile kept.
    Tile,
    /// A coast part kept; the mask is not whole yet.
    CoastPart,
    /// A coast part that made the mask whole: every level held is rebuilt with it. The pyramid is
    /// still owed, so the client states nothing yet.
    CoastWhole,
    /// ★ A sea-wind part kept; the wind is not whole yet (ruling B109).
    WindPart,
    /// A sea-wind part that made the wind whole; the rest of the artifact is still owed, so the client states nothing yet.
    WindWhole,
    /// ★ A far lakes part kept; the list is not whole yet (ruling B116 part 2).
    LakesPart,
    /// The far lake list whole and folded into the far levels: the far pieces are built again with it.
    LakesWhole,
    /// ★ A coast part that made the WHOLE ARTIFACT whole — the pyramid was already here (2026-09-22,
    /// ruling W10). The client states `ArtifactHeld { realm, digest }`, exactly as it does when the
    /// last pyramid part lands after the mask. The statement waits for BOTH, because the gateway
    /// stops serving parts the moment it is made.
    ArtifactWhole { realm: RealmId, digest: [u64; 2] },
    /// A tile parked until its realm's head arrives.
    Parked,
    /// A part or a tile for a realm whose head has not arrived: refused.
    NoHead,
    /// A part or a tile of the wrong shape (a level past the head's, a part past its count, rows
    /// that are not the tile's, a level whose words are not the lattice's): refused.
    Shape,
    /// A part or a tile already held: nothing changes.
    Duplicate,
    /// A bulk message that is not an artifact's: not this receiver's, counted.
    NotArtifact,
    /// ★ A fine tile's part kept; the tile is not whole yet (B2).
    FinePart,
    /// A fine tile made whole and held: the chunks over it may build.
    FineTile,
    /// A whole fine tile parked until its realm's head arrives.
    FineParked,
}

/// The receiver's counters — every refusal by name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArtifactCounters {
    pub heads: u64,
    pub parts: u64,
    pub levels_whole: u64,
    pub pyramids_whole: u64,
    pub tiles: u64,
    /// Coast parts kept, and the masks made whole.
    pub coast_parts: u64,
    pub coasts_whole: u64,
    /// Sea-wind parts kept, and the winds made whole (ruling B109).
    pub wind_parts: u64,
    pub winds_whole: u64,
    /// Tiles parked before their head, and the parked tiles dropped past the bound.
    pub parked: u64,
    pub parked_dropped: u64,
    pub no_head: u64,
    pub shape: u64,
    pub duplicates: u64,
    pub not_artifact: u64,
    /// ★ Fine parts kept, fine tiles held, and fine tiles parked before their head (B2).
    pub fine_parts: u64,
    pub fine_tiles: u64,
    pub fine_parked: u64,
}

/// The coast mask's parts, assembling: each part's bits, `None` for one not here.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CoastParts {
    parts: Vec<Option<Vec<u8>>>,
}

/// One level's parts, assembling.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct LevelParts {
    /// Each part's heights and its water words (empty for a part that carries none).
    parts: Vec<Option<(Vec<i16>, Vec<i16>)>>,
}

/// ★ THE RECEIVER: the shared book, the parts still assembling, the counters.
#[derive(Clone, Debug, Default)]
pub struct ArtifactReceiver {
    book: Arc<ArtifactBook>,
    assembling: BTreeMap<(RealmId, u32), LevelParts>,
    /// The coast masks assembling, per realm (2026-09-22, ruling W10).
    coasts: BTreeMap<RealmId, CoastParts>,
    /// The sea winds assembling, per realm (ruling B109), in the coast's own part form.
    winds: BTreeMap<RealmId, CoastParts>,
    /// The far lake lists assembling, per realm (ruling B116 part 2), in the coast's own part form.
    lakes: BTreeMap<RealmId, CoastParts>,
    /// ★ TILES THAT ARRIVED BEFORE THEIR HEAD (the far-view ship): the tiles come from the
    /// realm's shard on their own lane and the head from the gateway's cache on another, so a
    /// tile may land first. It is PARKED, bounded per realm, and applied when the head lands —
    /// never dropped (each tile is shipped once) and never applied unchecked.
    parked: BTreeMap<RealmId, Vec<ParkedTile>>,
    /// ★ THE FINE TILES ASSEMBLING (B2): each tile's parts by realm and key, `None` for a part not here.
    fine_parts: BTreeMap<FinePartsKey, Vec<Option<Vec<u8>>>>,
    /// Whole fine tiles that arrived before their head: decoded, parked, bounded as the tiles are.
    fine_parked: BTreeMap<RealmId, Vec<ParkedFine>>,
    /// ★ THE FINE TILES HELD, NOT YET STATED (ruling B81 Q3): each tile's finest level once it is held, for the client
    /// to state to its gateway (`ClientControlMsg::FineHeld`), taken by [`ArtifactReceiver::take_fine_held`].
    fine_held: Vec<(RealmId, (u8, u32, u32, u8))>,
    pub counters: ArtifactCounters,
}

/// A fine tile's parts are kept by its realm, its level and its key.
type FinePartsKey = (RealmId, FineLevel, TileKey);

/// A parked tile: its face, its column and row in tiles, and its rows' bytes.
type ParkedTile = (u8, u32, u32, Vec<u8>);

/// A parked fine tile: its fine level, its key, its side's shift, its first level and its decoded levels from it.
type ParkedFine = (
    FineLevel,
    TileKey,
    u32,
    u32,
    vd_terrain::fine::tile::WaterLevels,
);

/// How many tiles a realm may park before its head arrives: the pace ships four a tick, so this
/// is over a minute of tiles; past it the oldest is dropped and counted.
pub const PARKED_TILES_MAX: usize = 256;

impl ArtifactReceiver {
    /// The book, to share: a pointer bump.
    #[must_use]
    pub fn book(&self) -> Arc<ArtifactBook> {
        Arc::clone(&self.book)
    }

    /// Forget a realm's artifact and its assembling parts (the realm left the drawn set).
    pub fn forget(&mut self, realm: RealmId) {
        if self.book.realms.contains_key(&realm) {
            Arc::make_mut(&mut self.book).realms.remove(&realm);
        }
        self.assembling.retain(|(r, _), _| *r != realm);
        self.coasts.remove(&realm);
        self.winds.remove(&realm);
        self.lakes.remove(&realm);
        self.parked.remove(&realm);
        self.fine_parts.retain(|(r, _, _), _| *r != realm);
        self.fine_parked.remove(&realm);
    }

    /// ★ ONE PART IN. The book is written copy-on-write, so the renderer's pointer stays whole.
    pub fn accept(&mut self, msg: BulkMsg) -> ArtifactIngest {
        let result = match msg {
            BulkMsg::ArtifactHead {
                realm,
                world_tag,
                version,
                edge,
                digest,
                tiles_per_edge,
                levels,
                sea_m,
                coast_parts,
                low_m,
                high_m,
                wind_parts,
            } => self.accept_head(
                realm,
                ArtifactHead {
                    world_tag,
                    version,
                    edge,
                    digest,
                    tiles_per_edge,
                    levels,
                    sea_m,
                    coast_parts,
                    low_m,
                    high_m,
                    wind_parts,
                },
            ),
            BulkMsg::ArtifactPyramid {
                realm,
                level,
                part,
                parts,
                z_m,
                water_m,
            } => self.accept_part(realm, level, part, parts, z_m, water_m),
            BulkMsg::ArtifactTile {
                realm,
                face,
                tx,
                ty,
                rows,
            } => self.accept_tile(realm, face, tx, ty, &rows),
            BulkMsg::ArtifactCoast {
                realm,
                part,
                parts,
                bits,
            } => self.accept_coast(realm, part, parts, bits),
            BulkMsg::ArtifactWind {
                realm,
                part,
                parts,
                bytes,
            } => self.accept_wind(realm, part, parts, bytes),
            BulkMsg::ArtifactLakes {
                realm,
                part,
                parts,
                bytes,
            } => self.accept_lakes(realm, part, parts, bytes),
            BulkMsg::FineTilePart {
                realm,
                level,
                face,
                tx,
                ty,
                part,
                parts,
                bytes,
            } => self.accept_fine(realm, level, (face, tx, ty), (part, parts), bytes),
            _ => ArtifactIngest::NotArtifact,
        };
        self.count(result);
        result
    }

    fn count(&mut self, result: ArtifactIngest) {
        let c = &mut self.counters;
        match result {
            ArtifactIngest::Head => c.heads += 1,
            ArtifactIngest::SameHead | ArtifactIngest::Duplicate => c.duplicates += 1,
            ArtifactIngest::Part => c.parts += 1,
            ArtifactIngest::LevelWhole => {
                c.parts += 1;
                c.levels_whole += 1;
            }
            ArtifactIngest::PyramidWhole { .. } => {
                c.parts += 1;
                c.levels_whole += 1;
                c.pyramids_whole += 1;
            }
            ArtifactIngest::Tile => c.tiles += 1,
            ArtifactIngest::CoastPart => c.coast_parts += 1,
            ArtifactIngest::CoastWhole | ArtifactIngest::ArtifactWhole { .. } => {
                c.coast_parts += 1;
                c.coasts_whole += 1;
            }
            ArtifactIngest::WindPart => c.wind_parts += 1,
            ArtifactIngest::LakesPart | ArtifactIngest::LakesWhole => c.parts += 1,
            ArtifactIngest::WindWhole => {
                c.wind_parts += 1;
                c.winds_whole += 1;
            }
            ArtifactIngest::Parked => c.parked += 1,
            ArtifactIngest::NoHead => c.no_head += 1,
            ArtifactIngest::Shape => c.shape += 1,
            ArtifactIngest::NotArtifact => c.not_artifact += 1,
            ArtifactIngest::FinePart => c.fine_parts += 1,
            ArtifactIngest::FineTile => {
                c.fine_parts += 1;
                c.fine_tiles += 1;
            }
            ArtifactIngest::FineParked => {
                c.fine_parts += 1;
                c.fine_parked += 1;
            }
        }
    }

    fn accept_head(&mut self, realm: RealmId, head: ArtifactHead) -> ArtifactIngest {
        if self.book.realms.get(&realm).map(|c| c.head) == Some(head) {
            return ArtifactIngest::SameHead;
        }
        if head.edge == 0 || head.levels as usize > MAX_LEVELS {
            return ArtifactIngest::Shape;
        }
        Arc::make_mut(&mut self.book)
            .realms
            .insert(realm, ArtifactCache::new(head));
        self.assembling.retain(|(r, _), _| *r != realm);
        self.coasts.remove(&realm);
        // The tiles that waited for this head land now, through the same checks.
        for (face, tx, ty, rows) in self.parked.remove(&realm).unwrap_or_default() {
            let landed = self.accept_tile(realm, face, tx, ty, &rows);
            self.count(landed);
        }
        for (level, key, ts, lo, levels) in self.fine_parked.remove(&realm).unwrap_or_default() {
            if self.hold_fine(realm, level, key, ts, lo, levels) {
                self.counters.fine_tiles += 1;
            } else {
                self.counters.shape += 1;
            }
        }
        ArtifactIngest::Head
    }

    fn accept_part(
        &mut self,
        realm: RealmId,
        level: u32,
        part: u32,
        parts: u32,
        z_m: Vec<i16>,
        water_m: Vec<i16>,
    ) -> ArtifactIngest {
        let Some(cache) = self.book.realms.get(&realm) else {
            return ArtifactIngest::NoHead;
        };
        let head = cache.head;
        if level == 0 || level > head.levels || parts == 0 || part >= parts {
            return ArtifactIngest::Shape;
        }
        // ★ The water words ride beside the heights, one each, or none at all for the whole part.
        if !water_m.is_empty() && water_m.len() != z_m.len() {
            return ArtifactIngest::Shape;
        }
        if cache.level(level).is_some() {
            return ArtifactIngest::Duplicate;
        }
        let slot = self.assembling.entry((realm, level)).or_default();
        if slot.parts.is_empty() {
            slot.parts = vec![None; parts as usize];
        }
        if slot.parts.len() != parts as usize {
            return ArtifactIngest::Shape;
        }
        if slot.parts[part as usize].is_some() {
            return ArtifactIngest::Duplicate;
        }
        slot.parts[part as usize] = Some((z_m, water_m));
        if !slot.parts.iter().all(Option::is_some) {
            return ArtifactIngest::Part;
        }
        // The level is whole: its words must be the coarser lattice's, and its water words all
        // there or all absent.
        let words: Vec<i16> = slot
            .parts
            .iter()
            .flatten()
            .flat_map(|(z, _)| z.iter().copied())
            .collect();
        let water: Vec<i16> = slot
            .parts
            .iter()
            .flatten()
            .flat_map(|(_, w)| w.iter().copied())
            .collect();
        self.assembling.remove(&(realm, level));
        let coarse_edge = head.edge >> level;
        if (coarse_edge << level) != head.edge || words.len() != 6 * (coarse_edge as usize).pow(2) {
            return ArtifactIngest::Shape;
        }
        if !water.is_empty() && water.len() != words.len() {
            return ArtifactIngest::Shape;
        }
        let book = Arc::make_mut(&mut self.book);
        let cache = book
            .realms
            .get_mut(&realm)
            .expect("the head was read above");
        // ★ A LEVEL ASSEMBLED AFTER THE MASK TAKES IT AT ASSEMBLY (2026-09-22, ruling W10); a
        // level assembled BEFORE it is rebuilt when the mask lands ([`ArtifactReceiver::accept_coast`]).
        let coast = cache.coast.clone();
        let counts = cache.counts.clone();
        cache.pyramid[level as usize - 1] = Some(Arc::new(PyramidField {
            level,
            z_m: words,
            water_m: water,
            coast,
            counts,
        }));
        cache.epoch += 1;
        if cache.whole() {
            ArtifactIngest::PyramidWhole {
                realm,
                digest: head.digest,
            }
        } else {
            ArtifactIngest::LevelWhole
        }
    }

    /// ★ ONE COAST PART IN (2026-09-22, ruling W10). The parts assemble into ONE mask of one bit
    /// per fine node; when it lands, every pyramid level already assembled is REBUILT with it (a
    /// level is behind an `Arc`, so the rebuild is one small clone of the level's own words and a
    /// pointer to the shared mask) and the cache's epoch bumps, so the builders see a changed
    /// cache. A part for a realm with no head is refused; a part of a count the head did not
    /// announce, or a whole mask that is not the lattice's size, is refused by shape.
    ///
    /// **Example.** A pilot in orbit holds the home planet's six levels and its mask arrives last.
    /// Every level takes the mask at once, the epoch bumps, and the chunks rebuilt after it draw
    /// the same shoreline the tiles under her boots will draw.
    fn accept_coast(
        &mut self,
        realm: RealmId,
        part: u32,
        parts: u32,
        bits: Vec<u8>,
    ) -> ArtifactIngest {
        let Some(cache) = self.book.realms.get(&realm) else {
            return ArtifactIngest::NoHead;
        };
        let head = cache.head;
        if parts == 0 || parts != head.coast_parts || part >= parts {
            return ArtifactIngest::Shape;
        }
        if cache.coast.is_some() {
            return ArtifactIngest::Duplicate;
        }
        let slot = self.coasts.entry(realm).or_default();
        if slot.parts.is_empty() {
            slot.parts = vec![None; parts as usize];
        }
        if slot.parts[part as usize].is_some() {
            return ArtifactIngest::Duplicate;
        }
        slot.parts[part as usize] = Some(bits);
        if !slot.parts.iter().all(Option::is_some) {
            return ArtifactIngest::CoastPart;
        }
        let mask: Vec<u8> = slot
            .parts
            .iter()
            .flatten()
            .flat_map(|b| b.iter().copied())
            .collect();
        self.coasts.remove(&realm);
        // ★ THE WHOLE MASK is a SIDE WORD per node of the body's own lattice, rounded up to bytes —
        // the generator's own `coast_bytes`, never a count this crate spells for itself. MEASURED
        // before it (2026-09-23, ruling W16 fault B): this line read `nodes.div_ceil(8)`, the mask
        // widened to two bits a node, and the client REFUSED every realm's mask — four refusals,
        // no coast held, and every far chunk drew the recipe's own relief instead of the solve's.
        let nodes = 6 * (head.edge as usize).pow(2);
        if mask.len() != vd_terrain::artifact::coast_bytes(nodes) {
            return ArtifactIngest::Shape;
        }
        let mask: Arc<[u8]> = mask.into();
        let book = Arc::make_mut(&mut self.book);
        let cache = book
            .realms
            .get_mut(&realm)
            .expect("the head was read above");
        // ★ THE COUNTS ARE FOLDED HERE, ONCE (2026-09-22, ruling W15): the client folds the wet
        // fractions from the very mask the server folded its own from, so a cell at rung 18 reads
        // one side on both hosts and no count crosses the wire.
        let counts = Arc::new(CoastCounts::of(&mask, head.edge));
        cache.coast = Some(Arc::clone(&mask));
        cache.counts = Some(Arc::clone(&counts));
        for level in cache.pyramid.iter_mut().flatten() {
            let mut rebuilt = PyramidField::clone(level);
            rebuilt.coast = Some(Arc::clone(&mask));
            rebuilt.counts = Some(Arc::clone(&counts));
            *level = Arc::new(rebuilt);
        }
        cache.epoch += 1;
        if cache.whole() {
            ArtifactIngest::ArtifactWhole {
                realm,
                digest: head.digest,
            }
        } else {
            ArtifactIngest::CoastWhole
        }
    }

    /// ★ ONE SEA-WIND PART IN (ruling B109): the parts assemble in the coast's own way into the generator's record
    /// (`vd_terrain::sea_wind::SeaWind::from_bytes`); a part for a realm with no head, of a count the head did not
    /// announce, or a whole record the generator does not read is refused. When the wind lands the cache's epoch bumps,
    /// so the sea's pieces built before it are built again with it.
    ///
    /// **Example.** The home planet's three parts land after its coast mask; the glint off the coast stand reads 1.9 m/s.
    fn accept_wind(
        &mut self,
        realm: RealmId,
        part: u32,
        parts: u32,
        bytes: Vec<u8>,
    ) -> ArtifactIngest {
        let Some(cache) = self.book.realms.get(&realm) else {
            return ArtifactIngest::NoHead;
        };
        let head = cache.head;
        if parts == 0 || parts != head.wind_parts || part >= parts {
            return ArtifactIngest::Shape;
        }
        if cache.wind.is_some() {
            return ArtifactIngest::Duplicate;
        }
        let slot = self.winds.entry(realm).or_default();
        if slot.parts.is_empty() {
            slot.parts = vec![None; parts as usize];
        }
        if slot.parts[part as usize].is_some() {
            return ArtifactIngest::Duplicate;
        }
        slot.parts[part as usize] = Some(bytes);
        if !slot.parts.iter().all(Option::is_some) {
            return ArtifactIngest::WindPart;
        }
        let record: Vec<u8> = slot
            .parts
            .iter()
            .flatten()
            .flat_map(|b| b.iter().copied())
            .collect();
        self.winds.remove(&realm);
        let Some(wind) = vd_terrain::sea_wind::SeaWind::from_bytes(&record) else {
            return ArtifactIngest::Shape;
        };
        let book = Arc::make_mut(&mut self.book);
        let cache = book
            .realms
            .get_mut(&realm)
            .expect("the head was read above");
        cache.wind = Some(Arc::new(wind));
        cache.epoch += 1;
        if cache.whole() {
            ArtifactIngest::ArtifactWhole {
                realm,
                digest: head.digest,
            }
        } else {
            ArtifactIngest::WindWhole
        }
    }

    /// ★ ONE FAR LAKES PART IN (ruling B116 part 2): the parts assemble in the coast's own way into the generator's list
    /// (`vd_terrain::fine::far::decode_far_nodes`); when it is whole, each held pyramid level takes the list's fold
    /// (`vd_terrain::fine::far::fold_level`) — the folded blocks' heights and water words replaced by their majority class
    /// — and the cache's epoch bumps, so the far pieces built before it are built again with it. A part for a realm with no
    /// head, a list already held, or a whole record the generator does not read is refused.
    ///
    /// **Example.** The home planet's ten parts land a while after the artifact: the basin lake's 16 km blocks now stand at
    /// its floor where most of their ground is under water, at the lake's 128 m level −2 721 m.
    fn accept_lakes(
        &mut self,
        realm: RealmId,
        part: u32,
        parts: u32,
        bytes: Vec<u8>,
    ) -> ArtifactIngest {
        let Some(cache) = self.book.realms.get(&realm) else {
            return ArtifactIngest::NoHead;
        };
        if parts == 0 || part >= parts {
            return ArtifactIngest::Shape;
        }
        if cache.far.is_some() {
            return ArtifactIngest::Duplicate;
        }
        let slot = self.lakes.entry(realm).or_default();
        if slot.parts.len() != parts as usize {
            slot.parts = vec![None; parts as usize];
        }
        if slot.parts[part as usize].is_some() {
            return ArtifactIngest::Duplicate;
        }
        slot.parts[part as usize] = Some(bytes);
        if !slot.parts.iter().all(Option::is_some) {
            return ArtifactIngest::LakesPart;
        }
        let record: Vec<u8> = slot
            .parts
            .iter()
            .flatten()
            .flat_map(|b| b.iter().copied())
            .collect();
        self.lakes.remove(&realm);
        let Some(rows) = vd_terrain::fine::far::decode_far_nodes(&record) else {
            return ArtifactIngest::Shape;
        };
        let book = Arc::make_mut(&mut self.book);
        let cache = book
            .realms
            .get_mut(&realm)
            .expect("the head was read above");
        let edge = cache.head.edge;
        let mut blocks = Vec::with_capacity(cache.pyramid.len());
        for (index, slot) in cache.pyramid.iter_mut().enumerate() {
            let k = index as u32 + 1;
            let Some(level) = slot.as_ref() else {
                blocks.push(std::collections::BTreeSet::new());
                continue;
            };
            let folded = vd_terrain::fine::far::fold_level(&rows, edge, k, |b| {
                level.z_m.get(b as usize).copied()
            });
            let mut field = (**level).clone();
            if field.water_m.is_empty() {
                field.water_m = vec![vd_terrain::artifact::DRY_M; field.z_m.len()];
            }
            for (&b, fb) in &folded {
                if let (Some(z), Some(w)) = (
                    field.z_m.get_mut(b as usize),
                    field.water_m.get_mut(b as usize),
                ) {
                    *z = fb.z_m;
                    *w = fb.level_m;
                }
            }
            blocks.push(folded.into_keys().collect());
            *slot = Some(Arc::new(field));
        }
        cache.far = Some(Arc::new(blocks));
        cache.far_rows = Some(Arc::new(rows));
        cache.epoch += 1;
        ArtifactIngest::LakesWhole
    }

    fn accept_tile(
        &mut self,
        realm: RealmId,
        face: u8,
        tx: u32,
        ty: u32,
        rows: &[u8],
    ) -> ArtifactIngest {
        let Some(cache) = self.book.realms.get(&realm) else {
            let parked = self.parked.entry(realm).or_default();
            if parked.len() >= PARKED_TILES_MAX {
                parked.remove(0);
                self.counters.parked_dropped += 1;
            }
            parked.push((face, tx, ty, rows.to_vec()));
            return ArtifactIngest::Parked;
        };
        let head = cache.head;
        if face >= 6 || tx >= head.tiles_per_edge || ty >= head.tiles_per_edge {
            return ArtifactIngest::Shape;
        }
        let Some(rows) = Tile::rows_from_bytes(rows) else {
            return ArtifactIngest::Shape;
        };
        let want = (tile_width(head.edge, tx) * tile_width(head.edge, ty)) as usize;
        if rows.len() != want {
            return ArtifactIngest::Shape;
        }
        if cache.tiles.holds(face, tx, ty) {
            return ArtifactIngest::Duplicate;
        }
        let book = Arc::make_mut(&mut self.book);
        let cache = book
            .realms
            .get_mut(&realm)
            .expect("the head was read above");
        Arc::make_mut(&mut cache.tiles).apply(Tile { face, tx, ty, rows });
        cache.epoch += 1;
        ArtifactIngest::Tile
    }

    /// ★ ONE FINE PART IN (B2; ruling B95): kept until the record's parts are all here, then the record decoded and its
    /// levels held in its fine level's book (ruling B86 item 3: a far tile's coarse levels first, its finer ones joining
    /// them later) — or parked until its realm's head. A fine level no book holds, a part past its count or a record that
    /// does not decode is refused by shape.
    fn accept_fine(
        &mut self,
        realm: RealmId,
        level: u8,
        (face, tx, ty): (u8, u32, u32),
        (part, parts): (u32, u32),
        bytes: Vec<u8>,
    ) -> ArtifactIngest {
        let (Some(face), Some(level)) = (
            vd_seed::bend::Face::from_index(face),
            FineLevel::from_id(level),
        ) else {
            return ArtifactIngest::Shape;
        };
        if (part >= parts)
            | (parts > MAX_FINE_PARTS)
            | (tx > i32::MAX as u32)
            | (ty > i32::MAX as u32)
        {
            return ArtifactIngest::Shape;
        }
        let key = TileKey {
            face,
            tx: tx as i32,
            ty: ty as i32,
        };
        let slots = self
            .fine_parts
            .entry((realm, level, key))
            .or_insert_with(|| vec![None; parts as usize]);
        if slots.len() != parts as usize {
            // A tile shipped again under another cut starts over.
            *slots = vec![None; parts as usize];
        }
        if slots[part as usize].is_some() {
            return ArtifactIngest::Duplicate;
        }
        slots[part as usize] = Some(bytes);
        if slots.iter().any(Option::is_none) {
            return ArtifactIngest::FinePart;
        }
        let record: Vec<u8> = self
            .fine_parts
            .remove(&(realm, level, key))
            .unwrap_or_default()
            .into_iter()
            .flatten()
            .flatten()
            .collect();
        let Some((ts, lo, levels)) = vd_terrain::fine::tile::decode_water_range(&record, level)
        else {
            return ArtifactIngest::Shape;
        };
        if !self.book.realms.contains_key(&realm) {
            let parked = self.fine_parked.entry(realm).or_default();
            if parked.len() >= PARKED_TILES_MAX {
                parked.remove(0);
                self.counters.parked_dropped += 1;
            }
            parked.push((level, key, ts, lo, levels));
            return ArtifactIngest::FineParked;
        }
        if self.hold_fine(realm, level, key, ts, lo, levels) {
            ArtifactIngest::FineTile
        } else {
            ArtifactIngest::Shape
        }
    }

    /// One fine tile's levels from `lo` into its realm's book of its fine level, the book made with the head's own numbers
    /// on the first tile; `false` where the tile's side is not the book's, its levels are not the side's, or they do not
    /// join the levels held. A 128 m tile is stated as held (`ClientControlMsg::FineHeld` names 128 m tiles; the 16 m
    /// tiles' statement is owed a wire step, SL6).
    fn hold_fine(
        &mut self,
        realm: RealmId,
        level: FineLevel,
        key: TileKey,
        ts: u32,
        lo: u32,
        levels: vd_terrain::fine::tile::WaterLevels,
    ) -> bool {
        let Some(cache) = Arc::make_mut(&mut self.book).realms.get_mut(&realm) else {
            return false;
        };
        let fresh = || FineBook::with_side(level, cache.head.edge, ts, cache.head.sea_m);
        let slot = match level {
            FineLevel::L1 => &mut cache.fine,
            FineLevel::L2 => &mut cache.finer,
        };
        let Some(mut book) = slot
            .as_deref()
            .cloned()
            .or_else(fresh)
            .filter(|b| b.side_shift() == ts)
        else {
            return false;
        };
        if !book.insert_water_from(key, lo, levels) {
            return false;
        }
        let held = book.held_from(key).unwrap_or(lo) as u8;
        *slot = Some(Arc::new(book));
        cache.epoch += 1;
        if level == FineLevel::L1 {
            self.fine_held.push((
                realm,
                (key.face.index(), key.tx as u32, key.ty as u32, held),
            ));
        }
        true
    }

    /// ★ THE FINE TILES HELD SINCE THE LAST CALL (ruling B81 Q3), by realm, for the client to state.
    pub fn take_fine_held(&mut self) -> Vec<(RealmId, (u8, u32, u32, u8))> {
        std::mem::take(&mut self.fine_held)
    }
}

/// How many parts one fine tile may be cut into: the coder's worst case (four bytes a sample) over a 512 × 512 tile and
/// its means is under 1.5 MB, 46 parts of 32 KiB; a count far past it is no tile.
pub const MAX_FINE_PARTS: u32 = 64;

/// A bound on the levels a head may state: an edge halves per level, and no lattice's edge halves
/// more times than this ([`vd_terrain::macro_lattice::MACRO_EDGE_CEILING`] is 2 048 = 2¹¹).
pub const MAX_LEVELS: usize = 11;

#[cfg(test)]
mod tests {
    use super::*;

    /// The drawn ground's range of a home-moon artifact, as its head states it (ruling B55).
    fn drawn_of(artifact: &vd_terrain::artifact::Artifact) -> (i32, i32) {
        artifact.drawn_range_m(
            &vd_terrain::home::home_moon()
                .macro_lattice()
                .expect("a lattice"),
        )
    }
    use vd_seed::bend::Face;
    use vd_terrain::artifact::{Artifact, TILE_EDGE};
    use vd_terrain::home::{HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words};
    use vd_terrain::solve::Schedule;

    fn moon() -> (vd_terrain::BodyDefinition, MacroLattice, Artifact) {
        let moon = home_moon();
        let words = home_moon_solve_words();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let (state, facies, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let climate =
            vd_terrain::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let artifact = Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        (moon, lattice, artifact)
    }

    fn realm() -> RealmId {
        RealmId::Planet(41)
    }

    /// A stated head over the moon's 68-node lattice, levels none, the sea at `sea_m` (no solve needed).
    fn stated_head(realm: RealmId, sea_m: i16) -> BulkMsg {
        BulkMsg::ArtifactHead {
            realm,
            world_tag: 1,
            version: vd_terrain::artifact::ARTIFACT_VERSION,
            edge: 68,
            digest: [5, 6],
            tiles_per_edge: 2,
            levels: 0,
            sea_m,
            coast_parts: 0,
            low_m: -100,
            high_m: 100,
            wind_parts: 0,
        }
    }

    /// A 128 m tile's record `2^ts` cells wide, every level one height.
    fn fine_record(ts: u32) -> Vec<u8> {
        fine_record_of(ts, vd_terrain::fine::read::MEAN_LEVELS)
    }

    /// A fine tile's record `2^ts` cells wide with `means` mean levels, level `k` the height `100 + k` (a 16 m tile's
    /// `200 + k`, so a read tells the two levels apart).
    fn fine_record_of(ts: u32, means: u32) -> Vec<u8> {
        let base = if means == vd_terrain::fine::read::MEAN_LEVELS {
            100
        } else {
            200
        };
        let heights: Vec<Vec<i32>> = (0..=means)
            .map(|k| vec![base + k as i32; (1usize << (ts - k)) * (1usize << (ts - k))])
            .collect();
        // The water's planes: every cell dry (slice W2's record form).
        let levels = vd_terrain::fine::tile::WaterLevels {
            level: heights
                .iter()
                .map(|l| vec![vd_terrain::fine::stitch::MISSING; l.len()])
                .collect(),
            wet: heights.iter().map(|l| vec![0; l.len()]).collect(),
            heights,
        };
        let whole = vd_terrain::fine::tile::encode_water_levels(&levels, ts).expect("codes");
        vd_terrain::fine::tile::water_range_record(&whole, 0, means + 1).expect("a range")
    }

    /// The field a pick hands a build on the fine ground, or none while it waits or stands on the rough ground.
    fn ready(pick: FieldPick) -> Option<SharedField> {
        match pick {
            FieldPick::Ready(field) => Some(field),
            FieldPick::Rough(_) | FieldPick::Coarse(_) | FieldPick::Waiting => None,
        }
    }

    /// The field a pick hands a build on the 128 m ground under the 16 m level.
    fn coarse(pick: FieldPick) -> Option<SharedField> {
        match pick {
            FieldPick::Coarse(field) => Some(field),
            FieldPick::Ready(_) | FieldPick::Rough(_) | FieldPick::Waiting => None,
        }
    }

    /// The field a pick hands a build on the rough ground.
    fn rough(pick: FieldPick) -> Option<SharedField> {
        match pick {
            FieldPick::Rough(field) => Some(field),
            FieldPick::Ready(_) | FieldPick::Coarse(_) | FieldPick::Waiting => None,
        }
    }

    /// ★ A FINE TILE IS ASSEMBLED, HELD AND READ (B2): its parts kept until whole, parked before its head and landed with
    /// it; a level, a count or a record that is not one refused by shape, a side the book does not hold too; a chunk at
    /// rungs 0–12 of a wet realm reads it through the fine field once every tile of its stencils is here and waits
    /// before; a dry realm, a rung over 12 and a waiting pick stand as they were; a realm forgotten forgets its parts.
    #[test]
    fn a_fine_tile_is_assembled_held_and_read_and_a_chunk_waits_for_it() {
        let wet = RealmId::Planet(77);
        let record = fine_record(5);
        let half = record.len() / 2;
        let part = |part: u32, parts: u32, bytes: &[u8]| BulkMsg::FineTilePart {
            realm: wet,
            level: 1,
            face: 2,
            tx: 1,
            ty: 1,
            part,
            parts,
            bytes: bytes.to_vec(),
        };
        let mut rx = ArtifactReceiver::default();
        assert_eq!(
            rx.accept(part(0, 2, &record[..half])),
            ArtifactIngest::FinePart
        );
        assert_eq!(
            rx.accept(part(0, 2, &record[..half])),
            ArtifactIngest::Duplicate
        );
        assert_eq!(
            rx.accept(part(1, 2, &record[half..])),
            ArtifactIngest::FineParked
        );
        // By shape: a level no level holds, a face past six, a part past its count, a count past the bound, a record
        // that is no record.
        let shaped =
            |level: u8, face: u8, part: u32, parts: u32, bytes: Vec<u8>| BulkMsg::FineTilePart {
                realm: wet,
                level,
                face,
                tx: 0,
                ty: 0,
                part,
                parts,
                bytes,
            };
        for msg in [
            shaped(3, 2, 0, 1, record.clone()),
            shaped(1, 6, 0, 1, record.clone()),
            shaped(1, 2, 1, 1, record.clone()),
            shaped(1, 2, 0, MAX_FINE_PARTS + 1, record.clone()),
            shaped(1, 2, 0, 1, vec![9, 9, 9]),
        ] {
            assert_eq!(rx.accept(msg), ArtifactIngest::Shape);
        }
        // A tile shipped again under another cut starts over.
        assert_eq!(
            rx.accept(shaped(1, 2, 0, 3, record[..10].to_vec())),
            ArtifactIngest::FinePart
        );
        assert_eq!(
            rx.accept(shaped(1, 2, 0, 1, record.clone())),
            ArtifactIngest::FineParked
        );
        // The head lands the two parked tiles.
        assert_eq!(rx.accept(stated_head(wet, 0)), ArtifactIngest::Head);
        assert_eq!((rx.counters.fine_tiles, rx.counters.fine_parked), (2, 2));
        let face = Face::from_index(2).expect("a face");
        let cache = rx.book().get(wet).cloned().expect("a cache");
        assert!(
            cache
                .fine()
                .is_some_and(|b| b.holds(TileKey { face, tx: 1, ty: 1 }))
        );
        // A tile of another side is refused; a whole tile after the head is held.
        let wider = fine_record(6);
        let other = |tx: u32, bytes: &[u8]| BulkMsg::FineTilePart {
            realm: wet,
            level: 1,
            face: 2,
            tx,
            ty: 1,
            part: 0,
            parts: 1,
            bytes: bytes.to_vec(),
        };
        assert_eq!(rx.accept(other(2, &wider)), ArtifactIngest::Shape);
        assert_eq!(rx.accept(other(3, &record)), ArtifactIngest::FineTile);
        // ★ FAR COPIES FIRST (B86 item 3): a tile's coarse levels alone are held; finer levels join them below; levels that
        // stop short of the coarsest a fresh tile needs are refused.
        // The levels in slice W2's water form (every cell dry), the only form the client reads.
        let levels = |lo: u32, hi: u32| {
            let heights: Vec<Vec<i32>> = (0..=vd_terrain::fine::read::MEAN_LEVELS)
                .map(|k| vec![100 + k as i32; (32usize >> k) * (32usize >> k)])
                .collect();
            let water = vd_terrain::fine::tile::WaterLevels {
                level: heights
                    .iter()
                    .map(|l| vec![vd_terrain::fine::stitch::MISSING; l.len()])
                    .collect(),
                wet: heights.iter().map(|l| vec![0; l.len()]).collect(),
                heights,
            };
            let whole = vd_terrain::fine::tile::encode_water_levels(&water, 5).expect("codes");
            vd_terrain::fine::tile::water_range_record(&whole, lo, hi).expect("a range")
        };
        let far = |tx: u32, ty: u32, bytes: &[u8]| BulkMsg::FineTilePart {
            realm: wet,
            level: 1,
            face: 2,
            tx,
            ty,
            part: 0,
            parts: 1,
            bytes: bytes.to_vec(),
        };
        let held_from = |rx: &ArtifactReceiver, tx: i32| {
            rx.book().get(wet).and_then(|c| {
                c.fine()
                    .and_then(|b| b.held_from(TileKey { face, tx, ty: 2 }))
            })
        };
        assert_eq!(
            rx.accept(far(4, 2, &levels(3, 6))),
            ArtifactIngest::FineTile
        );
        assert_eq!(held_from(&rx, 4), Some(3));
        assert_eq!(
            rx.accept(far(4, 2, &levels(1, 3))),
            ArtifactIngest::FineTile
        );
        assert_eq!(held_from(&rx, 4), Some(1));
        assert_eq!(rx.accept(far(5, 2, &levels(0, 2))), ArtifactIngest::Shape);
        assert_eq!(held_from(&rx, 5), None);
        // The chunk at rung 0 over cells 6 199 … 6 262 both ways reads samples 45 … 51: the tile (1, 1) alone.
        let cache = rx.book().get(wet).cloned().expect("a cache");
        let inner: SharedField = Arc::new(TileCache::new(68));
        let at = |rung: u8, x: i32| ChunkKey {
            face,
            rung,
            x,
            y: 100,
            z: 0,
        };
        // ★ THE 16 m LEVEL (ruling B95): with no 16 m tile here the chunk is COARSE — built on the 128 m ground, and the
        // cache's coarse view reads the same chunk on it as ready; a rung over 6 is ready on the 128 m ground alone.
        let wrapped = coarse(cache.with_fine(FieldPick::Ready(Arc::clone(&inner)), at(0, 100)))
            .expect("coarse");
        assert!(!Arc::ptr_eq(&wrapped, &inner));
        assert_eq!(
            wrapped.fine_z(face, 0, 0, 6_230, 6_230),
            cache.fine().expect("a book").z(face, 0, 0, 6_230, 6_230)
        );
        let lattice = MacroLattice::of(&home_moon()).expect("a lattice");
        // Its span waits for the 16 m ground; a rung over 6 and the rough and coarse views do not.
        assert!(cache.finer_waits(face, 0, 100, 100));
        assert!(!cache.finer_waits(face, 7, 1, 100));
        assert!(!cache.rough_view().finer_waits(face, 0, 100, 100));
        assert!(!cache.coarse_view().finer_waits(face, 0, 100, 100));
        let view = cache.coarse_view();
        assert_eq!(
            (view.is_coarse_view(), cache.is_coarse_view()),
            (true, false)
        );
        assert!(view.is_coarse(&lattice, at(0, 100)));
        assert!(ready(view.with_fine(FieldPick::Ready(Arc::clone(&inner)), at(0, 100))).is_some());
        // A rung over 6 is ready on the 128 m ground alone: the chunk (1, 1) at rung 7 reads the samples 59 … 127 both
        // ways, the 128 m tiles (1 … 3, 1 … 3), all sent now.
        for ty in 1..=3u32 {
            for tx in 1..=3u32 {
                rx.accept(BulkMsg::FineTilePart {
                    realm: wet,
                    level: 1,
                    face: 2,
                    tx,
                    ty,
                    part: 0,
                    parts: 1,
                    bytes: record.clone(),
                });
            }
        }
        let cache = rx.book().get(wet).cloned().expect("a cache");
        let r7 = ChunkKey {
            face,
            rung: 7,
            x: 1,
            y: 1,
            z: 0,
        };
        assert!(ready(cache.with_fine(FieldPick::Ready(Arc::clone(&inner)), r7)).is_some());
        assert!(!cache.is_coarse(&lattice, r7));
        // The 16 m tile under the chunk (samples 384 … 394 at rung 4: the tile (12, 12) alone) lands; the chunk is ready
        // on it, and reads the 16 m ground.
        let finer = |tx: u32, bytes: &[u8]| BulkMsg::FineTilePart {
            realm: wet,
            level: 2,
            face: 2,
            tx,
            ty: 12,
            part: 0,
            parts: 1,
            bytes: bytes.to_vec(),
        };
        assert_eq!(
            rx.accept(finer(12, &fine_record_of(5, 2))),
            ArtifactIngest::FineTile
        );
        // A 16 m record that carries the 128 m level's count of levels is refused by shape.
        assert_eq!(rx.accept(finer(13, &record)), ArtifactIngest::Shape);
        let cache = rx.book().get(wet).cloned().expect("a cache");
        assert!(!cache.finer_waits(face, 0, 100, 100));
        let wrapped = ready(cache.with_fine(FieldPick::Ready(Arc::clone(&inner)), at(0, 100)))
            .expect("ready");
        assert_eq!(
            wrapped.fine_z(face, 0, 0, 6_230, 6_230),
            cache.finer().expect("a book").z(face, 0, 0, 6_230, 6_230)
        );
        assert_ne!(
            wrapped.fine_z(face, 0, 0, 6_230, 6_230),
            cache.fine().expect("a book").z(face, 0, 0, 6_230, 6_230)
        );
        // A chunk over a tile not held stands on the rough ground (B86 item 7): the field under the fine ground, and the
        // cache's rough view reads the same chunk on it too; a rung over 12 and a waiting pick stand as they were.
        assert!(ready(cache.with_fine(FieldPick::Ready(Arc::clone(&inner)), at(0, 400))).is_none());
        let under = rough(cache.with_fine(FieldPick::Ready(Arc::clone(&inner)), at(0, 400)))
            .expect("rough");
        assert!(Arc::ptr_eq(&under, &inner));
        let view = cache.rough_view();
        assert_eq!((view.is_rough_view(), cache.is_rough_view()), (true, false));
        assert!(Arc::ptr_eq(
            &ready(view.with_fine(FieldPick::Ready(Arc::clone(&inner)), at(0, 100)))
                .expect("ready"),
            &inner
        ));
        assert!(rough(cache.with_fine(FieldPick::Ready(Arc::clone(&inner)), at(0, 100))).is_none());
        let kept =
            ready(cache.with_fine(FieldPick::Ready(Arc::clone(&inner)), at(13, 0))).expect("ready");
        assert!(Arc::ptr_eq(&kept, &inner));
        assert!(ready(cache.with_fine(FieldPick::Waiting, at(0, 100))).is_none());
        // The rung's field reads the fine ground too, at rungs it serves: the 16 m ground where its tile is held (the cell
        // 778 at rung 3 lies on the 16 m tile (12, 12)), the 128 m ground where it is not (the cell 1 200: the 16 m tile
        // (18, 18) is not here, the 128 m tile (2, 2) is).
        let near = cache.field_at_rung(&lattice, 3).expect("a field");
        assert_eq!(
            near.fine_z(face, 3, 3, 778, 778),
            cache.finer().expect("a 16 m book").z(face, 3, 3, 778, 778)
        );
        assert_eq!(
            near.fine_z(face, 3, 3, 1_200, 1_200),
            cache.fine().expect("a book").z(face, 3, 3, 1_200, 1_200)
        );
        // A dry realm's chunk stands on its pick, with or without a book.
        let dry = RealmId::Planet(78);
        assert_eq!(
            rx.accept(stated_head(dry, vd_terrain::artifact::DRY_M)),
            ArtifactIngest::Head
        );
        let dry_cache = rx.book().get(dry).cloned().expect("a cache");
        let kept = ready(dry_cache.with_fine(FieldPick::Ready(Arc::clone(&inner)), at(0, 100)))
            .expect("ready");
        assert!(Arc::ptr_eq(&kept, &inner));
        assert!(!dry_cache.finer_waits(face, 0, 100, 100));
        // A wet realm with no tile yet waits, and its fine rungs' field is not here yet (a reader stands on a coarser level
        // and reads again); over rung 12 the plain field answers.
        let bare = RealmId::Planet(79);
        assert_eq!(rx.accept(stated_head(bare, 0)), ArtifactIngest::Head);
        let bare_cache = rx.book().get(bare).cloned().expect("a cache");
        assert!(
            ready(bare_cache.with_fine(FieldPick::Ready(Arc::clone(&inner)), at(0, 100))).is_none()
        );
        assert!(bare_cache.field_at_rung(&lattice, 3).is_none());
        assert!(bare_cache.field_at_rung(&lattice, 13).is_some());
        // A dry realm's fine rung answers its plain field.
        assert!(dry_cache.field_at_rung(&lattice, 3).is_some());
        // A realm forgotten forgets its parts.
        assert_eq!(
            rx.accept(part(0, 2, &record[..half])),
            ArtifactIngest::FinePart
        );
        rx.forget(wet);
        assert_eq!(
            rx.accept(part(1, 2, &record[half..])),
            ArtifactIngest::FinePart
        );
    }

    fn head_of(artifact: &Artifact) -> BulkMsg {
        // ★ NO COAST MASK ON THIS HEAD (2026-09-22): the tests around it weigh the pyramid's own
        // path, and a head that announces no part has none to wait for. The mask's own statements
        // state a head of one or two parts and ship them.
        head_with_coast_parts(artifact, 0)
    }

    /// The moon's head, announcing its mask in `coast_parts` parts.
    fn head_with_coast_parts(artifact: &Artifact, coast_parts: u32) -> BulkMsg {
        BulkMsg::ArtifactHead {
            realm: realm(),
            world_tag: 9,
            version: artifact.version,
            edge: artifact.edge,
            digest: artifact.digest(),
            tiles_per_edge: artifact.tiles_per_edge(),
            levels: artifact.pyramid.len() as u32,
            sea_m: artifact.sea_m,
            coast_parts,
            low_m: drawn_of(artifact).0,
            high_m: drawn_of(artifact).1,
            wind_parts: 0,
        }
    }

    /// The same head, announcing the moon's mask: one part (27 744 nodes are 3 468 bytes).
    fn head_with_coast(artifact: &Artifact) -> BulkMsg {
        head_with_coast_parts(artifact, 1)
    }

    /// One coast part of the moon's mask, whole.
    fn coast_of(artifact: &Artifact) -> BulkMsg {
        BulkMsg::ArtifactCoast {
            realm: realm(),
            part: 0,
            parts: 1,
            bits: artifact.coast.clone(),
        }
    }

    fn part_of(level: u32, part: u32, parts: u32, z_m: Vec<i16>) -> BulkMsg {
        BulkMsg::ArtifactPyramid {
            realm: realm(),
            level,
            part,
            parts,
            z_m,
            water_m: vec![],
        }
    }

    /// ★ THE COAST MASK AT THE CLIENT (2026-09-22, ruling W10). Six statements, each of which
    /// could fail:
    ///
    /// 1. a coast part for a realm with no head is refused, and a part whose count is not the
    ///    head's, or past the count, or twice, is refused by shape or as a duplicate;
    /// 2. a mask whose whole is not one bit per fine node is refused by SHAPE, never kept;
    /// 3. the parts assemble into the artifact's own mask, bit for bit;
    /// 4. a level assembled BEFORE the mask is REBUILT with it, and the level's own reader then
    ///    answers the row's side;
    /// 5. a level assembled AFTER the mask takes it at assembly;
    /// 6. the cache is not whole until the mask is here, and the epoch bumps when it lands;
    /// 7. ★ a far chunk WAITS for the mask (review finding 2 of plan step 2): a level held without
    ///    it is no chunk's field and no rung's field, and the same level is both once it lands.
    ///
    /// **Example.** A pilot in orbit holds the moon's two levels; its mask lands last, both levels
    /// take it, and the globe she draws has the shoreline her boots will stand on — and nothing
    /// of the globe was drawn from a level that had no shoreline at all.
    #[test]
    fn the_coast_mask_assembles_rebuilds_the_levels_and_refuses_a_wrong_shape() {
        let (_moon, lattice, artifact) = moon();
        let far = ChunkKey {
            face: Face::PosX,
            rung: 13,
            x: 0,
            y: 0,
            z: 0,
        };
        let mut rx = ArtifactReceiver::default();
        // (1) no head yet.
        assert_eq!(rx.accept(coast_of(&artifact)), ArtifactIngest::NoHead);
        rx.accept(head_with_coast(&artifact));
        // The first level lands BEFORE the mask.
        assert_eq!(
            rx.accept(part_of(1, 0, 1, artifact.pyramid[0].clone())),
            ArtifactIngest::LevelWhole
        );
        let book = rx.book();
        let cache = book.get(realm()).expect("a cache");
        assert!(!cache.coast_held(), "the mask is not here yet");
        assert!(!cache.whole(), "a cache without its mask is not whole");
        assert_eq!(cache.level(1).expect("level 1").coast, None);
        // (7) the level is held, and still no chunk and no rung reads it.
        assert_eq!(format!("{:?}", cache.field_for(&lattice, far)), "Waiting");
        assert!(cache.field_at_rung(&lattice, 13).is_none());
        let epoch = cache.epoch;
        // (1) the shapes refused: a count the head did not announce, a part past the count.
        let bad = |part: u32, parts: u32, bits: Vec<u8>| BulkMsg::ArtifactCoast {
            realm: realm(),
            part,
            parts,
            bits,
        };
        assert_eq!(rx.accept(bad(0, 2, vec![0])), ArtifactIngest::Shape);
        assert_eq!(rx.accept(bad(1, 1, vec![0])), ArtifactIngest::Shape);
        assert_eq!(rx.accept(bad(0, 0, vec![0])), ArtifactIngest::Shape);
        // (2) a mask of the wrong size is refused by shape and nothing is kept.
        assert_eq!(rx.accept(bad(0, 1, vec![0; 7])), ArtifactIngest::Shape);
        assert!(!rx.book().get(realm()).expect("a cache").coast_held());
        // (3) (4) (6) the real mask lands: the level held is rebuilt and the cache is whole.
        assert_eq!(
            rx.accept(coast_of(&artifact)),
            ArtifactIngest::CoastWhole,
            "level 2 is still owed, so the artifact is not whole yet"
        );
        let book = rx.book();
        let cache = book.get(realm()).expect("a cache");
        assert!(cache.coast_held());
        assert_eq!(
            cache.coast().map(|m| m.to_vec()),
            Some(artifact.coast.clone())
        );
        assert!(cache.epoch > epoch, "the epoch bumps when the mask lands");
        let level_1 = cache.level(1).expect("level 1");
        assert_eq!(
            level_1.coast.as_deref().map(<[u8]>::to_vec),
            Some(artifact.coast.clone())
        );
        // (7) with the mask here, the far chunk builds from the level and the rung reads it.
        assert_eq!(
            format!("{:?}", cache.field_for(&lattice, far)),
            "Ready(level 1)"
        );
        assert_eq!(
            cache.field_at_rung(&lattice, 13).expect("level 1").level(),
            1
        );
        // The level's own reader now answers the row's side for a fine node.
        let node = (0..artifact.rows.len() as u32)
            .find(|&n| artifact.water_side(n) == Some(vd_terrain::artifact::Side::Sea))
            .unwrap_or(0);
        assert_eq!(level_1.water_side(node), artifact.water_side(node));
        // A second part of the same mask is a duplicate.
        assert_eq!(rx.accept(coast_of(&artifact)), ArtifactIngest::Duplicate);
        assert!(!cache.whole(), "level 2 is still owed");
        // (5) the level assembled AFTER the mask takes it at assembly, and finishes the artifact.
        // The LAST PYRAMID part says so here; the other order — the last COAST part finishing it —
        // is `ArtifactWhole`, which the statement below drives.
        assert_eq!(
            rx.accept(part_of(2, 0, 1, artifact.pyramid[1].clone())),
            ArtifactIngest::PyramidWhole {
                realm: realm(),
                digest: artifact.digest(),
            }
        );
        let book = rx.book();
        let cache = book.get(realm()).expect("a cache");
        assert!(cache.whole());
        assert_eq!(
            cache
                .level(2)
                .expect("level 2")
                .coast
                .as_deref()
                .map(<[u8]>::to_vec),
            Some(artifact.coast.clone())
        );
        assert_eq!(rx.counters.coast_parts, 1);
        assert_eq!(rx.counters.coasts_whole, 1);
        // A head at ANOTHER digest starts the realm over: the mask goes with it.
        let mut moved = artifact.clone();
        moved.sea_m += 1;
        rx.accept(head_with_coast(&moved));
        assert!(!rx.book().get(realm()).expect("a cache").coast_held());
        // ★ THE OTHER ORDER: the pyramid lands first and the LAST COAST PART finishes the
        // artifact, so the client states it holds the realm then and not before — the gateway
        // stops serving on that word, so it must wait for the mask.
        let mut other = ArtifactReceiver::default();
        other.accept(head_with_coast(&artifact));
        other.accept(part_of(1, 0, 1, artifact.pyramid[0].clone()));
        assert_eq!(
            other.accept(part_of(2, 0, 1, artifact.pyramid[1].clone())),
            ArtifactIngest::LevelWhole,
            "every level is here and the mask is not: nothing is stated yet"
        );
        assert_eq!(
            other.accept(coast_of(&artifact)),
            ArtifactIngest::ArtifactWhole {
                realm: realm(),
                digest: artifact.digest(),
            }
        );
        assert!(other.book().get(realm()).expect("a cache").whole());
        // And forgetting the realm drops the parts still assembling.
        rx.forget(realm());
        assert!(rx.book().get(realm()).is_none());
    }

    /// ★ A MASK IN TWO PARTS (2026-09-28): the first part waits for the second, the same part again
    /// is a duplicate, and the second part finishes the mask, whose bits are the two parts in
    /// order.
    ///
    /// **Example.** The gateway paces the home planet's mask in 68 parts; a pilot's client holds
    /// the first 67 apart, drops a part the gateway sent twice, and takes the mask whole only when
    /// the last one lands.
    #[test]
    fn a_mask_in_two_parts_waits_for_both_and_drops_a_part_sent_twice() {
        let (_moon, _lattice, artifact) = moon();
        let half = artifact.coast.len() / 2;
        let part = |part: u32, bits: &[u8]| BulkMsg::ArtifactCoast {
            realm: realm(),
            part,
            parts: 2,
            bits: bits.to_vec(),
        };
        let mut rx = ArtifactReceiver::default();
        rx.accept(head_with_coast_parts(&artifact, 2));
        assert_eq!(
            rx.accept(part(0, &artifact.coast[..half])),
            ArtifactIngest::CoastPart
        );
        assert_eq!(
            rx.accept(part(0, &artifact.coast[..half])),
            ArtifactIngest::Duplicate
        );
        assert!(!rx.book().get(realm()).expect("a cache").coast_held());
        assert_eq!(
            rx.accept(part(1, &artifact.coast[half..])),
            ArtifactIngest::CoastWhole
        );
        let book = rx.book();
        let cache = book.get(realm()).expect("a cache");
        assert_eq!(
            cache.coast().map(|m| m.to_vec()),
            Some(artifact.coast.clone())
        );
        assert_eq!(rx.counters.coast_parts, 2);
        assert_eq!(rx.counters.coasts_whole, 1);
    }

    /// ★ THE WATER WORDS' SHAPE (2026-09-21): a part whose water words are neither none nor one
    /// per height is refused, and a level whose parts assemble to water words of another count
    /// than its heights is refused too — never a level with a word for some nodes and not others.
    #[test]
    fn a_part_or_a_level_whose_water_words_do_not_match_its_heights_is_refused() {
        let mut rx = ArtifactReceiver::default();
        // A head of a tiny lattice: edge 2, one level of 6 · 1² words.
        rx.accept(BulkMsg::ArtifactHead {
            realm: realm(),
            world_tag: 9,
            version: 3,
            edge: 2,
            digest: [1, 2],
            tiles_per_edge: 1,
            levels: 1,
            sea_m: vd_terrain::artifact::DRY_M,
            coast_parts: 0,
            low_m: 0,
            high_m: 0,
            wind_parts: 0,
        });
        let wrong = BulkMsg::ArtifactPyramid {
            realm: realm(),
            level: 1,
            part: 0,
            parts: 1,
            z_m: vec![1, 2, 3, 4, 5, 6],
            water_m: vec![7, 8],
        };
        assert_eq!(rx.accept(wrong), ArtifactIngest::Shape);
        // Two parts: the first carries water words, the second none — the level's count is off.
        let first = BulkMsg::ArtifactPyramid {
            realm: realm(),
            level: 1,
            part: 0,
            parts: 2,
            z_m: vec![1, 2, 3],
            water_m: vec![7, 8, 9],
        };
        let second = BulkMsg::ArtifactPyramid {
            realm: realm(),
            level: 1,
            part: 1,
            parts: 2,
            z_m: vec![4, 5, 6],
            water_m: vec![],
        };
        assert_eq!(rx.accept(first), ArtifactIngest::Part);
        assert_eq!(rx.accept(second), ArtifactIngest::Shape);
        assert_eq!(rx.counters.shape, 2);
        assert_eq!(rx.book().get(realm()).expect("a head").levels_held(), 0);
    }

    fn tile_of(artifact: &Artifact, face: Face, tx: u32, ty: u32) -> BulkMsg {
        BulkMsg::ArtifactTile {
            realm: realm(),
            face: face.index(),
            tx,
            ty,
            rows: artifact.tile(face, tx, ty).to_bytes(),
        }
    }

    /// ★ THE MOON'S ARTIFACT RECEIVED: a part before the head is refused; the head opens the cache;
    /// level 2 in one part is whole at once, level 1 in two parts is whole on the second, and the
    /// second completion states the pyramid whole with the digest; a duplicate part and a duplicate
    /// head change nothing; a level past the head, a part past its count, a wrong part count and a
    /// level of the wrong length are refused by shape; a tile lands and reads back through the
    /// cache; a tile past the lattice or of the wrong length is refused; a chunk manifest is not an
    /// artifact; the book is shared copy-on-write so an older pointer keeps what it saw; a new head
    /// with another digest starts over; a forget drops the realm.
    #[test]
    fn the_moons_artifact_is_received_part_by_part() {
        let (_moon, lattice, artifact) = moon();
        let mut rx = ArtifactReceiver::default();
        assert_eq!(rx.accept(part_of(1, 0, 1, vec![])), ArtifactIngest::NoHead);
        // A tile before the head is PARKED and lands with the head; a torn one parked lands as a
        // shape refusal then; the bound drops the oldest.
        assert_eq!(
            rx.accept(tile_of(&artifact, Face::PosX, 1, 0)),
            ArtifactIngest::Parked
        );
        assert_eq!(
            rx.accept(BulkMsg::ArtifactTile {
                realm: realm(),
                face: 0,
                tx: 1,
                ty: 1,
                rows: vec![1, 2, 3],
            }),
            ArtifactIngest::Parked
        );
        for _ in 0..PARKED_TILES_MAX {
            assert_eq!(
                rx.accept(tile_of(&artifact, Face::PosX, 1, 1)),
                ArtifactIngest::Parked
            );
        }
        assert_eq!(rx.counters.parked_dropped, 2);
        assert_eq!(rx.accept(head_of(&artifact)), ArtifactIngest::Head);
        assert_eq!(rx.book().get(realm()).expect("held").tiles().len(), 1);
        assert!(rx.parked.is_empty());
        assert_eq!(rx.accept(head_of(&artifact)), ArtifactIngest::SameHead);
        let before = rx.book();
        let cache = rx.book().get(realm()).copied_head();
        assert_eq!(cache.levels, 2);
        assert_eq!(cache.edge, 68);
        assert_eq!(cache.sea(), None, "the airless moon states no sea");
        let wet = ArtifactHead {
            sea_m: -40,
            ..cache
        };
        assert_eq!(wet.sea(), Some(-40));
        // Level 2 whole in one part (6 · 17² = 1 734 words).
        let l2 = artifact.pyramid[1].clone();
        assert_eq!(l2.len(), 1_734);
        assert_eq!(
            rx.accept(part_of(2, 0, 1, l2.clone())),
            ArtifactIngest::LevelWhole
        );
        assert_eq!(rx.accept(part_of(2, 0, 1, l2)), ArtifactIngest::Duplicate);
        // Shapes refused.
        assert_eq!(rx.accept(part_of(3, 0, 1, vec![0])), ArtifactIngest::Shape);
        assert_eq!(rx.accept(part_of(0, 0, 1, vec![0])), ArtifactIngest::Shape);
        assert_eq!(rx.accept(part_of(1, 2, 2, vec![0])), ArtifactIngest::Shape);
        assert_eq!(rx.accept(part_of(1, 0, 0, vec![0])), ArtifactIngest::Shape);
        // Level 1 in two parts (6 · 34² = 6 936 words): the first is kept, a second copy of it is
        // a duplicate, a part with another count is a shape error, the second completes it.
        let l1 = artifact.pyramid[0].clone();
        assert_eq!(l1.len(), 6_936);
        let (a, b) = l1.split_at(4_000);
        assert_eq!(
            rx.accept(part_of(1, 0, 2, a.to_vec())),
            ArtifactIngest::Part
        );
        assert_eq!(
            rx.accept(part_of(1, 0, 2, a.to_vec())),
            ArtifactIngest::Duplicate
        );
        assert_eq!(
            rx.accept(part_of(1, 1, 3, b.to_vec())),
            ArtifactIngest::Shape
        );
        assert_eq!(
            rx.accept(part_of(1, 1, 2, b.to_vec())),
            ArtifactIngest::PyramidWhole {
                realm: realm(),
                digest: artifact.digest()
            }
        );
        let cache = rx.book().get(realm()).cloned().expect("a cache");
        assert!(cache.whole());
        assert_eq!(cache.levels_held(), 2);
        assert_eq!(cache.level(1).expect("level 1").z_m, artifact.pyramid[0]);
        assert_eq!(cache.level(2).expect("level 2").z_m, artifact.pyramid[1]);
        assert_eq!(cache.level(0), None);
        assert_eq!(cache.level(3), None);
        // The older pointer saw no level (copy-on-write).
        assert_eq!(before.get(realm()).expect("held").levels_held(), 0);
        // A rung-13 chunk reads level 1 and a rung-15 chunk level 2, ready; a rung-0 chunk reads
        // the tiles, waiting.
        let key = |rung: u8| ChunkKey {
            face: Face::PosX,
            rung,
            x: 0,
            y: 0,
            z: 0,
        };
        assert_eq!(
            format!("{:?}", cache.field_for(&lattice, key(13))),
            "Ready(level 1)"
        );
        assert_eq!(
            format!("{:?}", cache.field_for(&lattice, key(15))),
            "Ready(level 2)"
        );
        assert_eq!(
            format!("{:?}", cache.field_for(&lattice, key(0))),
            "Waiting"
        );
        assert_eq!(
            cache.field_at_rung(&lattice, 0).expect("the tiles").level(),
            0
        );
        assert_eq!(
            cache.field_at_rung(&lattice, 13).expect("level 1").level(),
            1
        );
        assert_eq!(
            cache.field_at_rung(&lattice, 15).expect("level 2").level(),
            2
        );
        // A level of the wrong length is refused by shape, and the level stays absent.
        let mut short = ArtifactReceiver::default();
        assert_eq!(short.accept(head_of(&artifact)), ArtifactIngest::Head);
        assert_eq!(
            short.accept(part_of(2, 0, 1, vec![0; 100])),
            ArtifactIngest::Shape
        );
        assert!(
            short
                .book()
                .get(realm())
                .expect("held")
                .field_at_rung(&lattice, 13)
                .is_none()
        );
        // The tiles: one lands and reads back; the same one again is a duplicate; a tile past the
        // lattice, a face past six, and rows of the wrong length are shape errors.
        let epoch = cache.epoch;
        assert_eq!(
            rx.accept(tile_of(&artifact, Face::PosX, 0, 0)),
            ArtifactIngest::Tile
        );
        assert_eq!(
            rx.accept(tile_of(&artifact, Face::PosX, 0, 0)),
            ArtifactIngest::Duplicate
        );
        let cache = rx.book().get(realm()).cloned().expect("a cache");
        assert_eq!(cache.epoch, epoch + 1);
        assert_eq!(cache.tiles().len(), 2);
        let node = lattice.index(Face::PosX, 3, 3);
        assert_eq!(cache.tiles().z_m(node), artifact.z_m(node));
        let bad_tile = |face: u8, tx: u32, ty: u32, rows: Vec<u8>| BulkMsg::ArtifactTile {
            realm: realm(),
            face,
            tx,
            ty,
            rows,
        };
        assert_eq!(rx.accept(bad_tile(0, 2, 0, vec![])), ArtifactIngest::Shape);
        assert_eq!(rx.accept(bad_tile(0, 0, 2, vec![])), ArtifactIngest::Shape);
        assert_eq!(rx.accept(bad_tile(6, 0, 0, vec![])), ArtifactIngest::Shape);
        assert_eq!(
            rx.accept(bad_tile(0, 0, 0, vec![0; 5])),
            ArtifactIngest::Shape
        );
        assert_eq!(
            rx.accept(bad_tile(
                0,
                0,
                0,
                vec![0; vd_terrain::artifact::ROW_BYTES * 3]
            )),
            ArtifactIngest::Shape
        );
        // ★ A ROW OF NINE BYTES IS REFUSED (slice 8d step 2): the row grew to ten when the rock map
        // landed, and a whole tile of the OLD row is exactly what an older shard would ship. Nine
        // and ten share no common multiple under a tile's own count, so the length test alone
        // catches it and the client keeps the coarser rung standing.
        let want = (tile_width(artifact.edge, 0) * tile_width(artifact.edge, 0)) as usize;
        assert_eq!(vd_terrain::artifact::ROW_BYTES, 10);
        assert_eq!(
            rx.accept(bad_tile(1, 0, 0, vec![0; want * 9])),
            ArtifactIngest::Shape
        );
        // The same tile at the row's real width is taken.
        assert_eq!(
            rx.accept(bad_tile(1, 0, 0, vec![0; want * 10])),
            ArtifactIngest::Tile
        );
        // A chunk over that tile alone is ready now.
        let mid = ChunkKey {
            face: Face::PosX,
            rung: 0,
            x: 4_000,
            y: 4_000,
            z: 0,
        };
        assert_eq!(
            format!("{:?}", cache.field_for(&lattice, mid)),
            "Ready(level 0)"
        );
        // Not an artifact.
        assert_eq!(
            rx.accept(BulkMsg::ChunkManifest {
                realm: realm(),
                coarse: vd_core::grid::ChunkCoord {
                    body: realm(),
                    face: vd_core::grid::Face::PosY,
                    rung: vd_core::grid::Rung::new(3).expect("rung 3"),
                    x: 0,
                    y: 0,
                    z: 0,
                },
                digests: vec![],
            }),
            ArtifactIngest::NotArtifact
        );
        assert_eq!(
            rx.counters,
            ArtifactCounters {
                heads: 1,
                parts: 3,
                levels_whole: 2,
                pyramids_whole: 1,
                // Three: the two the moon's own tiles gave, and the ten-byte tile the row-width
                // refusal above takes after refusing its nine-byte twin (slice 8d step 2).
                tiles: 3,
                parked: PARKED_TILES_MAX as u64 + 2,
                parked_dropped: 2,
                no_head: 1,
                shape: 11,
                duplicates: PARKED_TILES_MAX as u64 - 1 + 4,
                not_artifact: 1,
                // This statement ships no mask: its head announces none (`head_of`).
                coast_parts: 0,
                coasts_whole: 0,
                // Nor a wind: its head announces none.
                wind_parts: 0,
                winds_whole: 0,
                // No fine part rides this statement.
                fine_parts: 0,
                fine_tiles: 0,
                fine_parked: 0,
            }
        );
        // A new head with another digest starts the realm over — with a part still assembling,
        // which it drops (on a fresh receiver, whose level 2 is not whole).
        let mut fresh = ArtifactReceiver::default();
        assert_eq!(fresh.accept(head_of(&artifact)), ArtifactIngest::Head);
        assert_eq!(
            fresh.accept(part_of(2, 0, 2, artifact.pyramid[1][..800].to_vec())),
            ArtifactIngest::Part
        );
        let head_with =
            |realm: RealmId, edge: u32, levels: u32, digest: [u64; 2]| BulkMsg::ArtifactHead {
                realm,
                world_tag: 9,
                version: artifact.version,
                edge,
                digest,
                tiles_per_edge: artifact.tiles_per_edge(),
                levels,
                sea_m: artifact.sea_m,
                coast_parts: 0,
                low_m: 0,
                high_m: 0,
                wind_parts: 0,
            };
        let mut other_digest = artifact.digest();
        other_digest[0] ^= 1;
        assert_eq!(
            fresh.accept(head_with(realm(), 68, 2, other_digest)),
            ArtifactIngest::Head
        );
        assert!(fresh.assembling.is_empty(), "the head dropped the part");
        assert_eq!(
            rx.accept(head_with(realm(), 68, 2, other_digest)),
            ArtifactIngest::Head
        );
        assert_eq!(rx.book().get(realm()).expect("held").levels_held(), 0);
        assert!(rx.book().get(realm()).expect("held").tiles().is_empty());
        // Before any level is here a coarse chunk WAITS for its level.
        assert_eq!(
            format!(
                "{:?}",
                rx.book()
                    .get(realm())
                    .expect("held")
                    .field_for(&lattice, key(13))
            ),
            "Waiting"
        );
        // A head of no edge or too many levels is refused.
        assert_eq!(
            rx.accept(head_with(RealmId::Planet(42), 0, 2, artifact.digest())),
            ArtifactIngest::Shape
        );
        assert_eq!(
            rx.accept(head_with(RealmId::Planet(42), 68, 12, artifact.digest())),
            ArtifactIngest::Shape
        );
        // A head whose levels the edge does not halve to: the level's part is refused by shape
        // once whole (68 does not halve three times).
        assert_eq!(
            rx.accept(head_with(RealmId::Planet(43), 68, 3, artifact.digest())),
            ArtifactIngest::Head
        );
        assert_eq!(
            rx.accept(BulkMsg::ArtifactPyramid {
                realm: RealmId::Planet(43),
                level: 3,
                part: 0,
                parts: 1,
                z_m: vec![0; 6 * 8 * 8],
                water_m: vec![],
            }),
            ArtifactIngest::Shape
        );
        rx.forget(RealmId::Planet(43));
        assert_eq!(rx.book().len(), 1);
        assert_eq!(rx.book().realms().count(), 1);
        // The forget, with a part assembling: both go.
        assert_eq!(
            rx.accept(part_of(2, 0, 2, artifact.pyramid[1][..800].to_vec())),
            ArtifactIngest::Part
        );
        rx.forget(realm());
        assert!(rx.assembling.is_empty());
        rx.forget(realm());
        assert!(rx.book().is_empty());
        assert_eq!(rx.book().get(realm()), None);
        assert_eq!(TILE_EDGE, 64);
    }

    trait CopiedHead {
        fn copied_head(self) -> ArtifactHead;
    }

    impl CopiedHead for Option<&ArtifactCache> {
        fn copied_head(self) -> ArtifactHead {
            self.expect("a cache").head
        }
    }
}
