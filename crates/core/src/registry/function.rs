//! THE FUNCTION TABLE — what a block does. One row today, "none": a plain block. The signal system
//! (P9) appends its rows — a thruster, a seat, a sensor — and nothing saved changes, because a kind's
//! function is part of its triple and the triple never moves.
//!
//! **Example.** A steel cube with function "none" is a wall. When P9 appends "thruster", a NEW kind
//! (steel, cube, thruster) is born beside it; the wall keeps its number.

/// A function's number: its index in [`FUNCTIONS`]. Dense, append-only, never reused. No `Default`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionId(pub u16);

/// One function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionDef {
    /// The identity, folded into the digest. Never changes once the row exists.
    pub key: &'static str,
    /// The display word. May change freely.
    pub name: &'static str,
}

/// THE FUNCTION TABLE. The index is the id. APPEND ONLY.
pub const FUNCTIONS: [FunctionDef; 1] = [FunctionDef {
    key: "none",
    name: "none",
}];

impl FunctionId {
    /// A plain block.
    pub const NONE: FunctionId = FunctionId(0);

    /// This function's row; `None` for a number the table does not hold.
    #[must_use]
    pub fn def(self) -> Option<&'static FunctionDef> {
        FUNCTIONS.get(usize::from(self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_dense_and_starts_with_none() {
        crate::registry::assert_dense(FUNCTIONS.len(), |id| FunctionId(id).def());
        assert_eq!(FunctionId::NONE.def().map(|f| f.key), Some("none"));
    }
}
