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
use vd_terrain::artifact::{PyramidField, Tile, TileCache, ZField, tile_width, tiles_of_chunk};
use vd_terrain::chunk::ChunkKey;
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
    /// The level is still assembling, or a tile the stencil reads has not arrived: the coarser
    /// rung stands.
    Waiting,
}

impl std::fmt::Debug for FieldPick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldPick::Ready(field) => write!(f, "Ready(level {})", field.level()),
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
    /// Bumped on every accepted change, so a reader tells a changed cache from the one it read.
    pub epoch: u64,
}

impl ArtifactCache {
    /// The tiles this cache holds, by id (2026-09-22, the dev state's listing).
    pub fn tile_ids(&self) -> impl Iterator<Item = (u8, u32, u32)> + '_ {
        self.tiles.tile_ids()
    }

    fn new(head: ArtifactHead) -> ArtifactCache {
        ArtifactCache {
            head,
            pyramid: vec![None; head.levels as usize],
            tiles: Arc::new(TileCache::new(head.edge)),
            coast: None,
            epoch: 1,
        }
    }

    /// Whether every pyramid level AND the coast mask are here: the far view is whole, and the
    /// client says so. A level without the mask draws a shoreline the next rung does not share,
    /// which is the crawl the owner saw from 1 400 km, so the mask counts.
    #[must_use]
    pub fn whole(&self) -> bool {
        self.pyramid.iter().all(Option::is_some) && self.coast_held()
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
    #[must_use]
    pub fn field_for(&self, lattice: &MacroLattice, key: ChunkKey) -> FieldPick {
        let level = PyramidField::level_for(lattice, self.head.levels, key.rung);
        if level == 0 {
            let needed = tiles_of_chunk(lattice, key);
            return if self.tiles.holds_all(&needed) {
                FieldPick::Ready(self.tiles.clone() as SharedField)
            } else {
                FieldPick::Waiting
            };
        }
        match self.level(level) {
            Some(field) => FieldPick::Ready(field.clone() as SharedField),
            None => FieldPick::Waiting,
        }
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
    /// whole or not here; or the tiles, which answer nothing for a node they lack.
    #[must_use]
    pub fn field_at_rung(&self, lattice: &MacroLattice, rung: u8) -> Option<SharedField> {
        let level = PyramidField::level_for(lattice, self.head.levels, rung);
        if level == 0 {
            return Some(self.tiles.clone() as SharedField);
        }
        self.level(level).map(|f| f.clone() as SharedField)
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
    /// Tiles parked before their head, and the parked tiles dropped past the bound.
    pub parked: u64,
    pub parked_dropped: u64,
    pub no_head: u64,
    pub shape: u64,
    pub duplicates: u64,
    pub not_artifact: u64,
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
    /// ★ TILES THAT ARRIVED BEFORE THEIR HEAD (the far-view ship): the tiles come from the
    /// realm's shard on their own lane and the head from the gateway's cache on another, so a
    /// tile may land first. It is PARKED, bounded per realm, and applied when the head lands —
    /// never dropped (each tile is shipped once) and never applied unchecked.
    parked: BTreeMap<RealmId, Vec<ParkedTile>>,
    pub counters: ArtifactCounters,
}

/// A parked tile: its face, its column and row in tiles, and its rows' bytes.
type ParkedTile = (u8, u32, u32, Vec<u8>);

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
        self.parked.remove(&realm);
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
            ArtifactIngest::Parked => c.parked += 1,
            ArtifactIngest::NoHead => c.no_head += 1,
            ArtifactIngest::Shape => c.shape += 1,
            ArtifactIngest::NotArtifact => c.not_artifact += 1,
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
        cache.pyramid[level as usize - 1] = Some(Arc::new(PyramidField {
            level,
            z_m: words,
            water_m: water,
            coast,
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
        // The whole mask is one bit per node of the body's own lattice, rounded up to bytes.
        let nodes = 6 * (head.edge as usize).pow(2);
        if mask.len() != nodes.div_ceil(8) {
            return ArtifactIngest::Shape;
        }
        let mask: Arc<[u8]> = mask.into();
        let book = Arc::make_mut(&mut self.book);
        let cache = book
            .realms
            .get_mut(&realm)
            .expect("the head was read above");
        cache.coast = Some(Arc::clone(&mask));
        for level in cache.pyramid.iter_mut().flatten() {
            let mut rebuilt = PyramidField::clone(level);
            rebuilt.coast = Some(Arc::clone(&mask));
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
}

/// A bound on the levels a head may state: an edge halves per level, and no lattice's edge halves
/// more times than this ([`vd_terrain::macro_lattice::MACRO_EDGE_CEILING`] is 2 048 = 2¹¹).
pub const MAX_LEVELS: usize = 11;

#[cfg(test)]
mod tests {
    use super::*;
    use vd_seed::bend::Face;
    use vd_terrain::artifact::{Artifact, TILE_EDGE};
    use vd_terrain::home::{HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words};
    use vd_terrain::solve::{Schedule, solve_full};

    fn moon() -> (vd_terrain::BodyDefinition, MacroLattice, Artifact) {
        let moon = home_moon();
        let words = home_moon_solve_words();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let (state, facies, _) =
            solve_full(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        let climate =
            vd_terrain::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_z));
        let artifact = Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        (moon, lattice, artifact)
    }

    fn realm() -> RealmId {
        RealmId::Planet(41)
    }

    fn head_of(artifact: &Artifact) -> BulkMsg {
        BulkMsg::ArtifactHead {
            realm: realm(),
            world_tag: 9,
            version: artifact.version,
            edge: artifact.edge,
            digest: artifact.digest(),
            tiles_per_edge: artifact.tiles_per_edge(),
            levels: artifact.pyramid.len() as u32,
            sea_m: artifact.sea_m,
            // ★ NO COAST MASK ON THIS HEAD (2026-09-22): the tests around it weigh the pyramid's
            // own path, and a head that announces no part has none to wait for. The mask's own
            // statement states a head of one part and ships it.
            coast_parts: 0,
        }
    }

    /// The same head, announcing the moon's mask: one part (27 744 nodes are 3 468 bytes).
    fn head_with_coast(artifact: &Artifact) -> BulkMsg {
        match head_of(artifact) {
            BulkMsg::ArtifactHead {
                realm,
                world_tag,
                version,
                edge,
                digest,
                tiles_per_edge,
                levels,
                sea_m,
                ..
            } => BulkMsg::ArtifactHead {
                realm,
                world_tag,
                version,
                edge,
                digest,
                tiles_per_edge,
                levels,
                sea_m,
                coast_parts: 1,
            },
            other => other,
        }
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
    /// 6. the cache is not whole until the mask is here, and the epoch bumps when it lands.
    ///
    /// **Example.** A pilot in orbit holds the moon's two levels; its mask lands last, both levels
    /// take it, and the globe she draws has the shoreline her boots will stand on.
    #[test]
    fn the_coast_mask_assembles_rebuilds_the_levels_and_refuses_a_wrong_shape() {
        let (_moon, _lattice, artifact) = moon();
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
        // The level's own reader now answers the row's side for a fine node.
        let node = (0..artifact.rows.len() as u32)
            .find(|&n| artifact.sea_side(n) == Some(true))
            .unwrap_or(0);
        assert_eq!(level_1.sea_side(node), artifact.sea_side(node));
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
