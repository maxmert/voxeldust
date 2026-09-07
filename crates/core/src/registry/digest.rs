//! THE IDENTITY DIGEST — the part of the registry that gives a SAVED record its meaning, folded into
//! one SHA-256 over rows `0..n` in table order (ruling B-12/B-13, V7 S3-4).
//!
//! IN: a kind's substance, form and function, each by its identity KEY; the form's geometry (class,
//! volume, full faces, collider); an attachment kind's key and whether it makes a body; the gap
//! convention (its step count AND its tag: radial, centre-sampled, negative inside solid). OUT: every
//! display name; a kind's variant count, its scale mask, its flags, its replacement pointer; a form's
//! legal-orientation mask and its provisional marks; an attachment's slots, faces and parameter
//! budget; every physical column. The test that decides: does this column change what a saved record
//! means?
//!
//! It is a PREFIX digest: a build that knows more rows than a store still matches the store's prefix.
//! A store carries `(kinds, attachments, digest)` in its own header row and opens if this build knows
//! at least that many rows and its prefix digest at those lengths matches.
//!
//! **Example.** The team renames "structural steel" to "mild steel" and ships a fluted style: neither
//! moves the digest, every store opens, every unpatched client connects. The team also changes row 2
//! from granite to chalk: the key "granite" is gone from the fold, and every store refuses to open,
//! naming both digests — as it must, because every granite cliff a player ever mined would otherwise
//! become chalk in silence.

use sha2::{Digest, Sha256};

use super::attachment::AttachmentDef;
use super::form::FormDef;
use super::function::FunctionDef;
use super::kind::KindDef;
use super::substance::SubstanceDef;
use super::{
    ATTACHMENTS, FORMS, FUNCTIONS, GAP_CONVENTION_TAG, GAP_STEPS_PER_CELL, KINDS, SUBSTANCES,
};

/// The five tables the digest reads, by reference, so a test can hand in a changed copy.
#[derive(Clone, Copy)]
pub struct Tables<'a> {
    pub substances: &'a [SubstanceDef],
    pub forms: &'a [FormDef],
    pub functions: &'a [FunctionDef],
    pub kinds: &'a [KindDef],
    pub attachments: &'a [AttachmentDef],
}

impl Tables<'static> {
    /// This build's tables.
    #[must_use]
    pub const fn current() -> Tables<'static> {
        Tables {
            substances: &SUBSTANCES,
            forms: &FORMS,
            functions: &FUNCTIONS,
            kinds: &KINDS,
            attachments: &ATTACHMENTS,
        }
    }
}

fn fold_key(h: &mut Sha256, key: &str) {
    h.update((key.len() as u16).to_le_bytes());
    h.update(key.as_bytes());
}

/// The identity prefix digest over the first `kinds` block kinds and the first `attachments`
/// attachment kinds of the given tables. Lengths past a table are clamped to it, so a stamp written by
/// a build that knew more rows can still be compared (and will not match). A row that names a table
/// entry that does not exist folds a marker instead, so the digest is total.
#[must_use]
pub fn identity_prefix_digest(t: Tables<'_>, kinds: usize, attachments: usize) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"voxeldust registry identity v2");
    h.update(GAP_STEPS_PER_CELL.to_le_bytes());
    h.update([GAP_CONVENTION_TAG]);
    for k in t.kinds.iter().take(kinds) {
        match t.substances.get(usize::from(k.substance.0)) {
            Some(s) => fold_key(&mut h, s.key),
            None => fold_key(&mut h, "?substance"),
        }
        match t.functions.get(usize::from(k.function.0)) {
            Some(f) => fold_key(&mut h, f.key),
            None => fold_key(&mut h, "?function"),
        }
        // The form's identity AND its geometry, as the row references it: a form change re-means
        // every kind on it.
        match t.forms.get(usize::from(k.form.0)) {
            Some(f) => {
                fold_key(&mut h, f.key);
                h.update([f.class as u8, f.collider as u8, f.full_faces]);
                h.update(f.volume_units.to_le_bytes());
            }
            None => fold_key(&mut h, "?form"),
        }
    }
    for a in t.attachments.iter().take(attachments) {
        fold_key(&mut h, a.key);
        h.update([u8::from(a.makes_body)]);
    }
    h.finalize().into()
}

/// What a block store's header row carries about the registry that wrote it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegistryStamp {
    pub kinds: u16,
    pub attachments: u16,
    pub digest: [u8; 32],
}

impl RegistryStamp {
    /// This build's stamp over its whole tables.
    #[must_use]
    pub fn current() -> RegistryStamp {
        let t = Tables::current();
        RegistryStamp {
            kinds: u16::try_from(KINDS.len())
                .expect("the kind table is capped at u16 by the record"),
            attachments: u16::try_from(ATTACHMENTS.len()).expect("the attachment table fits u16"),
            digest: identity_prefix_digest(t, KINDS.len(), ATTACHMENTS.len()),
        }
    }

    /// Whether this build may open a store written under `stored`: it knows at least as many rows,
    /// and its prefix digest at the stored lengths equals the stored digest.
    pub fn accepts(stored: &RegistryStamp) -> Result<(), RegistryMismatch> {
        let kinds = usize::from(stored.kinds);
        let attachments = usize::from(stored.attachments);
        if kinds > KINDS.len() || attachments > ATTACHMENTS.len() {
            return Err(RegistryMismatch::StoreKnowsMore {
                stored_kinds: stored.kinds,
                stored_attachments: stored.attachments,
                known_kinds: u16::try_from(KINDS.len()).expect("capped at u16"),
                known_attachments: u16::try_from(ATTACHMENTS.len()).expect("capped at u16"),
            });
        }
        let ours = identity_prefix_digest(Tables::current(), kinds, attachments);
        if ours != stored.digest {
            return Err(RegistryMismatch::MeaningChanged {
                stored: stored.digest,
                ours,
            });
        }
        Ok(())
    }
}

/// Why a store's registry stamp is refused. Named, never a silent open.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RegistryMismatch {
    #[error(
        "the store was written by a build that knew {stored_kinds} kinds and {stored_attachments} attachment kinds; this build knows {known_kinds} and {known_attachments}"
    )]
    StoreKnowsMore {
        stored_kinds: u16,
        stored_attachments: u16,
        known_kinds: u16,
        known_attachments: u16,
    },
    #[error(
        "the registry's meaning changed under the store's rows: stored {stored:02x?}, ours {ours:02x?}"
    )]
    MeaningChanged { stored: [u8; 32], ours: [u8; 32] },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::kind::KindFlags;
    use crate::registry::{BlockKindId, FunctionId, ShapeId, SubstanceId};

    fn digest_with(t: Tables<'_>) -> [u8; 32] {
        identity_prefix_digest(t, KINDS.len(), ATTACHMENTS.len())
    }

    #[test]
    fn a_rename_or_a_raised_bound_keeps_the_digest_and_a_meaning_swap_breaks_it() {
        // The digest-column test (the slice's gate).
        let stored = RegistryStamp::current();
        assert_eq!(
            RegistryStamp::accepts(&stored),
            Ok(()),
            "a store this build wrote opens"
        );
        let base = Tables::current();

        // Append a kind: the prefix at the OLD length is unchanged.
        let mut longer: Vec<KindDef> = KINDS.to_vec();
        longer.push(KindDef {
            substance: SubstanceId(2),
            form: ShapeId::CUBE,
            function: FunctionId::NONE,
            flags: KindFlags(0),
            variants: 1,
            sub_scales: 0,
            replaced_by: None,
        });
        assert_eq!(
            digest_with(Tables {
                kinds: &longer,
                ..base
            }),
            stored.digest,
            "an appended row never moves the prefix"
        );

        // Raise a variant count, widen a scale mask, retire, replace: all OUT.
        let mut restyled: Vec<KindDef> = KINDS.to_vec();
        restyled[5].variants = 4;
        restyled[5].sub_scales = 0b1111;
        restyled[5].flags = KindFlags(KindFlags::PLACEABLE.0 | KindFlags::RETIRED.0);
        restyled[5].replaced_by = Some(BlockKindId(6));
        assert_eq!(
            digest_with(Tables {
                kinds: &restyled,
                ..base
            }),
            stored.digest
        );

        // Rename a substance, a form, a function and an attachment (display names): all OUT.
        let mut substances = SUBSTANCES.to_vec();
        substances[2].name = "grey granite";
        substances[2].density_g_m3 += 1;
        let mut forms = FORMS.to_vec();
        forms[usize::from(ShapeId::CUBE.0)].name = "block";
        forms[usize::from(ShapeId::CUBE.0)].legal_orientations = 0b11;
        forms[usize::from(ShapeId::CUBE.0)].volume_provisional = true;
        let mut functions = FUNCTIONS.to_vec();
        functions[0].name = "plain";
        let mut att = ATTACHMENTS.to_vec();
        att[0].name = "HUD";
        att[0].slots = 9;
        att[0].legal_faces = 1;
        att[0].params_max = 1;
        assert_eq!(
            digest_with(Tables {
                substances: &substances,
                forms: &forms,
                functions: &functions,
                attachments: &att,
                ..base
            }),
            stored.digest,
            "a display rename or a raised bound refuses nothing"
        );

        // Swap a substance's meaning (its KEY): refused, naming both digests.
        let mut chalked = SUBSTANCES.to_vec();
        chalked[2].key = "chalk";
        let moved = digest_with(Tables {
            substances: &chalked,
            ..base
        });
        assert_ne!(moved, stored.digest);
        assert_eq!(
            RegistryStamp::accepts(&RegistryStamp {
                digest: moved,
                ..stored
            }),
            Err(RegistryMismatch::MeaningChanged {
                stored: moved,
                ours: stored.digest
            })
        );

        // Change one kind's triple, a form's key, a form's geometry, a function's key, an attachment's
        // key or its body-making: each moves the digest.
        let mut changed: Vec<KindDef> = KINDS.to_vec();
        changed[5].substance = SubstanceId(3);
        assert_ne!(
            digest_with(Tables {
                kinds: &changed,
                ..base
            }),
            stored.digest
        );
        let mut forms2 = FORMS.to_vec();
        forms2[usize::from(ShapeId::CUBE.0)].volume_units -= 512;
        assert_ne!(
            digest_with(Tables {
                forms: &forms2,
                ..base
            }),
            stored.digest,
            "a form's geometry is inside the digest"
        );
        let mut forms3 = FORMS.to_vec();
        forms3[usize::from(ShapeId::CUBE.0)].key = "block";
        assert_ne!(
            digest_with(Tables {
                forms: &forms3,
                ..base
            }),
            stored.digest
        );
        let mut functions2 = FUNCTIONS.to_vec();
        functions2[0].key = "plain";
        assert_ne!(
            digest_with(Tables {
                functions: &functions2,
                ..base
            }),
            stored.digest
        );
        let mut att2 = ATTACHMENTS.to_vec();
        att2[0].key = "HUD";
        assert_ne!(
            digest_with(Tables {
                attachments: &att2,
                ..base
            }),
            stored.digest
        );
        let mut att3 = ATTACHMENTS.to_vec();
        att3[1].makes_body = false;
        assert_ne!(
            digest_with(Tables {
                attachments: &att3,
                ..base
            }),
            stored.digest
        );
    }

    #[test]
    fn a_store_from_a_build_that_knew_more_rows_is_refused_by_name_and_a_short_prefix_is_accepted()
    {
        let stored = RegistryStamp::current();
        let future = RegistryStamp {
            kinds: stored.kinds + 1,
            ..stored
        };
        assert_eq!(
            RegistryStamp::accepts(&future),
            Err(RegistryMismatch::StoreKnowsMore {
                stored_kinds: stored.kinds + 1,
                stored_attachments: stored.attachments,
                known_kinds: stored.kinds,
                known_attachments: stored.attachments,
            })
        );
        let more_attachments = RegistryStamp {
            attachments: stored.attachments + 1,
            ..stored
        };
        assert_eq!(
            RegistryStamp::accepts(&more_attachments),
            Err(RegistryMismatch::StoreKnowsMore {
                stored_kinds: stored.kinds,
                stored_attachments: stored.attachments + 1,
                known_kinds: stored.kinds,
                known_attachments: stored.attachments,
            })
        );
        let short = RegistryStamp {
            kinds: 10,
            attachments: 2,
            digest: identity_prefix_digest(Tables::current(), 10, 2),
        };
        assert_eq!(
            RegistryStamp::accepts(&short),
            Ok(()),
            "an older, shorter store opens"
        );
        // A kind naming rows past the tables folds markers, so the digest is total and distinct.
        let odd = [KindDef {
            substance: SubstanceId(200),
            form: ShapeId(200),
            function: FunctionId(200),
            flags: KindFlags(0),
            variants: 1,
            sub_scales: 0,
            replaced_by: None,
        }];
        let t = Tables {
            kinds: &odd,
            ..Tables::current()
        };
        assert_ne!(identity_prefix_digest(t, 1, 0), stored.digest);
        let msg = format!(
            "{}",
            RegistryMismatch::StoreKnowsMore {
                stored_kinds: 2,
                stored_attachments: 1,
                known_kinds: 1,
                known_attachments: 1
            }
        );
        assert!(msg.contains("knew 2 kinds"), "{msg}");
    }
}
