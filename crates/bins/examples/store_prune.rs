//! ★ THE CLEAN-OUT OF OLD FINE VERSIONS (2026-10-14, step 1 of the worktree plan; the owner: "Please proceed"). A realm
//! store keeps every fine tile it ever solved, keyed by the input word it was solved from, and nothing deletes the rows
//! of an older fine version or an older artifact, so the file only grows. MEASURED: the home planet's file held 128 m
//! tiles of fine versions 1, 2 and 3, and only version 3 is read.
//!
//! This tool copies every row the store still uses into a fresh file beside it — every row of every other family, and
//! the fine rows of the two words the stored artifact's digest gives with this build's fine versions (the 128 m and the
//! 16 m) — reads the fresh file back against the old one row by row, and only then swaps them: the old file moves into
//! the backup folder as it was.
//!
//! - A store with no artifact head names no live word, so nothing is dropped and nothing is written.
//! - The fresh file's label (the stamp) is written by its own open, exactly as a shard writes it.
//! - A fresh start is never allowed here: a store whose label refuses this build stops the tool and is never wiped.
//! - A dropped word's rows are skipped as one key range, never read row by row.
//!
//! **Example.** With the cluster down, the home planet's 12.9 GB file is copied without the 128 m tiles of fine
//! versions 1 and 2. The next login reads the artifact and the 128 m and 16 m tiles from the fresh file, as before.
//!
//! `cargo run --release -p vd-bins --example store_prune -- <realm store file> <backup folder>`

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};

use vd_core::pose::RealmId;
use vd_io_prod::runtime::EnvConfig;
use vd_io_prod::store::RedbStore;
use vd_sim::io::{Bytes, Store};
use vd_sim::stub::built_store::{
    FineForm, artifact_head_key, decode_artifact_head, fine_word_range, fine_words,
};
use vd_terrain::fine::level::FineLevel;
use vd_terrain::fine::tile::input_word_of;

/// Rows a read or a commit carries: at most 32 MiB of fine parts.
const BATCH: usize = 1024;

/// The key past every family tag in use (tag 0 is the file's own label; the families are append-only from 1).
const PAST_LAST_TAG: [u8; 1] = [u8::MAX];

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let (Some(file), Some(backup), None) = (args.next(), args.next(), args.next()) else {
        return Err("usage: store_prune <realm store file> <backup folder>".into());
    };
    let file = PathBuf::from(file);
    let name = file
        .file_name()
        .ok_or("the store path names no file")?
        .to_string_lossy()
        .into_owned();
    let backup_dir = PathBuf::from(backup);
    let kept = backup_dir.join(&name);
    if kept.exists() {
        return Err(format!("{} exists: a backup is never overwritten", kept.display()).into());
    }
    let fresh = file.with_file_name(format!("{name}.pruning"));
    if fresh.exists() {
        return Err(format!(
            "{} exists: an earlier run stopped half way; look at it and delete it first",
            fresh.display()
        )
        .into());
    }

    let old = open(&file)?;
    let Some(head) = old.get(&artifact_head_key()) else {
        println!("{name}: no artifact head, nothing to prune");
        return Ok(());
    };
    let head = decode_artifact_head(&head)?;
    let live = [
        input_word_of(FineLevel::L1, head.digest),
        input_word_of(FineLevel::L2, head.digest),
    ];
    let mut dropped = Vec::new();
    for form in [FineForm::Raw, FineForm::Finished] {
        for word in fine_words(&old, form)? {
            let is_live = live.contains(&word);
            println!(
                "{name}: {form:?} tiles under input word {word:016x}: {}",
                if is_live { "kept" } else { "DROPPED" }
            );
            if !is_live {
                dropped.push(fine_word_range(form, word));
            }
        }
    }
    if dropped.is_empty() {
        println!("{name}: nothing to prune");
        return Ok(());
    }
    // The walk reads from tag 1 to the last tag: the label is the only row under tag 0, and no row stands past the last.
    if old
        .range(&[0], &[1], 2)
        .iter()
        .any(|(key, _)| key.as_slice() != [0])
    {
        return Err("a row under tag 0 that is not the file's label".into());
    }
    if !old.scan(&PAST_LAST_TAG).is_empty() {
        return Err("a row under tag 255, which no family of this build holds".into());
    }

    let mut new = open(&fresh)?;
    let (mut copied, mut value_bytes, mut staged) = (0u64, 0u64, 0usize);
    for (key, value) in Walk::new(&old, &dropped) {
        new.put(&key, &value);
        copied += 1;
        value_bytes += value.len() as u64;
        staged += 1;
        if staged == BATCH {
            new.commit();
            staged = 0;
        }
    }
    new.commit();
    new.flush();
    drop(new);
    println!("{name}: copied {copied} rows ({value_bytes} bytes of values)");

    // The check: the fresh file, opened again, holds exactly the rows the walk kept, byte for byte, and no dropped word.
    let new = open(&fresh)?;
    let mut checked = 0u64;
    let (mut a, mut b) = (Walk::new(&old, &dropped), Walk::new(&new, &[]));
    loop {
        match (a.next(), b.next()) {
            (None, None) => break,
            (Some(x), Some(y)) if x.0 == y.0 && *x.1 == *y.1 => checked += 1,
            (x, y) => {
                return Err(format!(
                    "the fresh file differs after {checked} rows: {:02x?} against {:02x?}",
                    x.map(|r| r.0),
                    y.map(|r| r.0)
                )
                .into());
            }
        }
    }
    if checked != copied {
        return Err(format!("{checked} rows read back against {copied} copied").into());
    }
    for form in [FineForm::Raw, FineForm::Finished] {
        if let Some(word) = fine_words(&new, form)?
            .into_iter()
            .find(|w| !live.contains(w))
        {
            return Err(format!("the fresh file still holds input word {word:016x}").into());
        }
    }
    drop((new, old));
    println!("{name}: the fresh file read back equal, {checked} rows");

    std::fs::create_dir_all(&backup_dir)?;
    let before = std::fs::metadata(&file)?.len();
    let after = std::fs::metadata(&fresh)?.len();
    std::fs::rename(&file, &kept)?;
    std::fs::rename(&fresh, &file).map_err(|e| {
        format!(
            "the old file is at {}, the fresh one at {}: move the fresh one to {} by hand ({e})",
            kept.display(),
            fresh.display(),
            file.display()
        )
    })?;
    println!(
        "{name}: {:.2} GB -> {:.2} GB; the old file is kept at {}",
        before as f64 / 1e9,
        after as f64 / 1e9,
        kept.display()
    );
    Ok(())
}

/// Open one store file the way a shard opens it, with a fresh start never allowed.
fn open(path: &Path) -> Result<RedbStore, Error> {
    let mut vars: BTreeMap<String, String> = std::env::vars().collect();
    vars.remove("VD_STORE_ALLOW_GENESIS");
    vars.remove("VD_REALM_STORE_DIR");
    vars.insert("VD_REALM_STORE".to_owned(), path.display().to_string());
    vd_bins::open_realm_store(&EnvConfig::new(vars), RealmId::Planet(0))?
        .ok_or_else(|| format!("{} opened no store", path.display()).into())
}

/// A store's rows from tag 1 in key order, a batch a read, each dropped word's range skipped whole.
struct Walk<'a> {
    store: &'a dyn Store,
    skip: &'a [(Vec<u8>, Vec<u8>)],
    from: Vec<u8>,
    rows: VecDeque<(Vec<u8>, Bytes)>,
}

impl<'a> Walk<'a> {
    fn new(store: &'a dyn Store, skip: &'a [(Vec<u8>, Vec<u8>)]) -> Walk<'a> {
        Walk {
            store,
            skip,
            from: vec![1],
            rows: VecDeque::new(),
        }
    }
}

impl Iterator for Walk<'_> {
    type Item = (Vec<u8>, Bytes);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((key, value)) = self.rows.pop_front() {
                if let Some((_, to)) = self
                    .skip
                    .iter()
                    .find(|(from, to)| key >= *from && key < *to)
                {
                    self.from.clone_from(to);
                    self.rows.clear();
                    continue;
                }
                return Some((key, value));
            }
            let rows = self.store.range(&self.from, &PAST_LAST_TAG, BATCH);
            let mut after = rows.last()?.0.clone();
            after.push(0);
            self.from = after;
            self.rows.extend(rows);
        }
    }
}
