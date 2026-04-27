//! Determinism harness: scan lighting / celestial-physics source files for
//! bare numeric literals. Lighting code must reference values via named
//! constants (`core::physics_constants::*`) or rendering-fidelity fields
//! (`config::graphics::LightingFidelity`); inline literals are forbidden.
//!
//! Permitted file exceptions:
//!   * `core/src/physics_constants.rs` — the *only* place named physics
//!     constants are defined. Literals here have units + provenance.
//!   * `client/src/config/graphics.rs` — rendering-fidelity preset values
//!     (`LightingFidelity::low/medium/high/ultra`). These are visual-taste
//!     knobs with no physical meaning, configured by the user.
//!   * `client/src/lighting/quality.rs` — the apply system that translates
//!     fidelity fields to Bevy components; reads named fields, no literals.
//!
//! Per-literal exceptions (always allowed everywhere):
//!   * `0`, `0.0` — initialisation zero / null vectors / array indices.
//!   * `1`, `1.0` — multiplicative identity / cardinality of one.
//!   * `2`, `2.0` — common dimensionality (Vec2 / pair / index).
//!   * `3`, `3.0` — common dimensionality (Vec3 / RGB).
//!   * `4`, `4.0` — common dimensionality (Vec4 / RGBA / Quaternion).
//!
//! Anything else found in a scanned file is reported as a magic-number
//! violation. CI failure is the contract.

use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};

/// Files (relative to the workspace root) that are scanned for magic numbers.
///
/// Each path is appended to `CARGO_MANIFEST_DIR/..` (the workspace root) at
/// runtime.
const SCANNED_PATHS: &[&str] = &[
    "core/src/stellar.rs",
    "core/src/geophysics.rs",
    "core/src/planet_rotation.rs",
    "client/src/lighting/mod.rs",
    "client/src/lighting/quality.rs",
    "client/src/lighting/solar.rs",
    "client/src/lighting/camera.rs",
    "client/src/lighting/ibl.rs",
    "client/src/lighting/atmosphere.rs",
    "client/src/lighting/rotation.rs",
    "client/src/lighting/emitters/mod.rs",
    "client/src/lighting/emitters/throttle.rs",
    "client/src/lighting/emitters/full_block.rs",
    "client/src/lighting/emitters/sub_block.rs",
    // Phase 2 may also add: client/src/lighting/starfield_bake.rs.
    // Phase 4 adds: client/src/lighting/rotation.rs.
    // Phase 5 adds: client/src/lighting/volumetric.rs.
    // Phase 6 adds: client/src/lighting/eclipse.rs.
    // Phase 8 adds: client/src/lighting/local.rs.
];

/// Per-literal whitelist of small natural numbers permitted everywhere.
/// `0.5` is included as a universal math constant (midpoint / averaging).
const ALLOWED_LITERALS: &[&str] = &[
    "0", "1", "2", "3", "4",
    "0.0", "1.0", "2.0", "3.0", "4.0",
    "0.5",
];

#[test]
fn no_magic_numbers_in_lighting_or_celestial_physics() {
    let workspace_root = workspace_root();
    let mut findings: Vec<String> = Vec::new();
    for relative in SCANNED_PATHS {
        let path = workspace_root.join(relative);
        if !path.exists() {
            // Skip; subsequent phases will create the file. Reporting a
            // missing file as a violation here would block Phase 0.
            continue;
        }
        findings.extend(scan_file(&path));
    }
    if !findings.is_empty() {
        panic!(
            "magic-number violations found in lighting / celestial-physics modules:\n{}\n\
             — every numeric value must come from `core::physics_constants` (named) \
             or `config::graphics::LightingFidelity` (rendering knob).",
            findings.join("\n"),
        );
    }
}

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR"); // = client/
    PathBuf::from(manifest_dir).join("..").canonicalize().unwrap()
}

/// Walk the file content stripping comments + string literals + char literals,
/// blanking `#[cfg(test)] mod tests { ... }` blocks (test assertion literals
/// are legitimate), and then matching remaining numeric tokens against the
/// whitelist. Lines starting with `const NAME:` are also skipped — those are
/// named-constant declarations whose RHS is the constant's value.
fn scan_file(path: &Path) -> Vec<String> {
    let content = fs::read_to_string(path).expect("read scanned file");
    let stripped = strip_comments_and_strings(&content);
    let test_blanked = blank_cfg_test_blocks(&stripped);
    let literal_re = numeric_literal_regex();
    let mut violations = Vec::new();
    for (idx, line) in test_blanked.lines().enumerate() {
        // Allow `const NAME: TYPE = ...;` and `static NAME: TYPE = ...;`
        // declarations. The literal on the RHS *is* the constant's value;
        // the constant itself is the named entity.
        if is_named_constant_decl(line) {
            continue;
        }
        for m in literal_re.find_iter(line) {
            let token = m.as_str();
            if is_allowed(token) {
                continue;
            }
            violations.push(format!(
                "  {}:{}:  `{}`",
                path.display(),
                idx + 1,
                token,
            ));
        }
    }
    violations
}

/// Replace any `#[cfg(test)]`-tagged module body with whitespace of the same
/// length. Tests legitimately use literal assertion values (`assert!(x > 5.0)`);
/// they should not trigger the magic-number lint.
///
/// Heuristic: scan for `#[cfg(test)]` followed (within the next ~6 non-empty
/// lines) by `mod NAME {`. Then brace-count to find the matching closing `}`.
/// Everything between is replaced with whitespace (preserving newlines).
fn blank_cfg_test_blocks(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = bytes.to_vec();
    let marker = b"#[cfg(test)]";
    let mut i = 0;
    while i + marker.len() <= bytes.len() {
        if &bytes[i..i + marker.len()] == marker {
            // Look forward for the next `{` (the start of the mod body).
            let mut j = i + marker.len();
            while j < bytes.len() && bytes[j] != b'{' {
                j += 1;
            }
            if j >= bytes.len() {
                break;
            }
            // Brace-count from here to find the matching `}`.
            let mut depth: i32 = 0;
            let mut k = j;
            while k < bytes.len() {
                match bytes[k] {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            // Blank from the start of the marker through k.
                            for b in out.iter_mut().take(k + 1).skip(i) {
                                if *b != b'\n' {
                                    *b = b' ';
                                }
                            }
                            i = k + 1;
                            break;
                        }
                    }
                    _ => {}
                }
                k += 1;
            }
            if depth != 0 {
                // Unmatched braces — fall through.
                i += 1;
            }
            continue;
        }
        i += 1;
    }
    String::from_utf8(out).unwrap()
}

fn is_named_constant_decl(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("const ")
        || trimmed.starts_with("pub const ")
        || trimmed.starts_with("pub(crate) const ")
        || trimmed.starts_with("pub(super) const ")
        || trimmed.starts_with("static ")
        || trimmed.starts_with("pub static ")
}

/// Replace each comment / string / char literal with whitespace of the same
/// length so byte offsets are preserved. Attribute bodies (`#[...]`) are kept
/// — Bevy macros use them and we don't expect magic numbers there.
fn strip_comments_and_strings(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = vec![b' '; bytes.len()];
    // Preserve newlines so line numbering stays correct.
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            out[i] = b'\n';
        }
    }

    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let next = bytes.get(i + 1).copied();
        // Line comment
        if b == b'/' && next == Some(b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Block comment (handle nesting)
        if b == b'/' && next == Some(b'*') {
            let mut depth = 1u32;
            i += 2;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'\n' {
                    out[i] = b'\n';
                }
                if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                    continue;
                }
                if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                    continue;
                }
                i += 1;
            }
            continue;
        }
        // String literal
        if b == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 2;
                    continue;
                }
                if bytes[i] == b'\n' {
                    out[i] = b'\n';
                }
                i += 1;
            }
            i = i.saturating_add(1);
            continue;
        }
        // Char literal — be conservative because `'a` could also be a
        // lifetime. Heuristic: a char literal ends with `'` within 6 chars
        // and contains no whitespace. Lifetimes are followed by an
        // identifier char, not a closing quote.
        if b == b'\'' {
            // Look ahead for a closing quote within ~6 bytes, allowing
            // escape sequences.
            let mut j = i + 1;
            let mut closed = false;
            while j < bytes.len() && j < i + 8 {
                if bytes[j] == b'\\' && j + 1 < bytes.len() {
                    j += 2;
                    continue;
                }
                if bytes[j] == b'\'' {
                    closed = true;
                    break;
                }
                if bytes[j] == b'\n' {
                    break;
                }
                j += 1;
            }
            if closed {
                i = j + 1;
                continue;
            }
            // Treat as lifetime; emit the apostrophe as itself.
            out[i] = b;
            i += 1;
            continue;
        }
        // Default: keep the byte
        out[i] = b;
        i += 1;
    }
    String::from_utf8(out).unwrap()
}

/// Match decimal integer / float literals with optional underscores,
/// optional fractional part, optional exponent, and optional Rust type
/// suffix. Hex / octal / binary not allowed in scanned files (would also
/// flag here if present).
fn numeric_literal_regex() -> Regex {
    // Boundary: not preceded by an identifier char or `.` (which would
    // indicate a method call like `Vec3::ONE.0`). Followed by anything
    // that's not an identifier continuation.
    Regex::new(
        r"(?xm)
        (?P<lit>
            \b
            \d (?: [\d_] )*
            (?: \. \d (?: [\d_] )* )?
            (?: [eE] [+-]? \d+ )?
            (?: f32 | f64 | i32 | u32 | i64 | u64 | i16 | u16 | i8 | u8 | usize | isize )?
            \b
        )
        ",
    )
    .unwrap()
}

fn is_allowed(token: &str) -> bool {
    // Strip Rust type suffix for comparison with the whitelist.
    let core = strip_type_suffix(token);
    ALLOWED_LITERALS.contains(&core)
}

fn strip_type_suffix(token: &str) -> &str {
    for suffix in &[
        "f32", "f64", "i32", "u32", "i64", "u64", "i16", "u16", "i8", "u8", "usize", "isize",
    ] {
        if let Some(stripped) = token.strip_suffix(suffix) {
            return stripped;
        }
    }
    token
}
