//! Determinism harness: assert that no client-side module invokes a
//! `from_seed*` constructor from `core::stellar`, `core::geophysics`, or
//! `core::planet_rotation`. The client receives celestial physics state via
//! server-broadcast `WorldState.bodies.{stellar,planetary,rotation_params}`;
//! it must never re-derive these values from a seed locally.
//!
//! This is a static-text scan of `client/src/**/*.rs`; it walks the file tree
//! and flags any reference to the forbidden function names. Comments and
//! string literals are stripped before matching, so doc-comments referring
//! to the formula by name are fine.
//!
//! Failure indicates someone added a client-side derivation. The fix is to
//! delete it — the value should come from the broadcast state.

use std::fs;
use std::path::{Path, PathBuf};

/// Function-name patterns whose presence in client source is a violation.
/// Each entry is a substring; the match is case-sensitive.
const FORBIDDEN_PATTERNS: &[&str] = &[
    // core::stellar
    "StellarState::from_",
    // core::geophysics
    "PlanetGeophysicalState::from_",
    // core::planet_rotation — only the *constructor* is forbidden;
    // `rotation_at(params, t)` is the closed-form pure function the client
    // is *expected* to call each frame.
    "PlanetRotationParams::from_",
];

#[test]
fn client_does_not_invoke_seed_derivations() {
    let client_src = client_src_root();
    let mut findings: Vec<String> = Vec::new();
    walk_rs(&client_src, &mut |path| {
        let content = fs::read_to_string(path).expect("read client source");
        let stripped = strip_comments_and_strings(&content);
        for (idx, line) in stripped.lines().enumerate() {
            for pat in FORBIDDEN_PATTERNS {
                if line.contains(pat) {
                    findings.push(format!(
                        "  {}:{}:  `{}`",
                        path.display(),
                        idx + 1,
                        pat,
                    ));
                }
            }
        }
    });
    if !findings.is_empty() {
        panic!(
            "client-side seed derivation found:\n{}\n\
             — celestial physics state is broadcast by the server. \
             Replace each call with a read from `WorldState.bodies.{{stellar|planetary|rotation_params}}`.",
            findings.join("\n"),
        );
    }
}

fn client_src_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR"); // = client/
    PathBuf::from(manifest_dir).join("src").canonicalize().unwrap()
}

fn walk_rs(dir: &Path, visit: &mut dyn FnMut(&Path)) {
    let read_dir = match fs::read_dir(dir) {
        Ok(d) => d,
        Err(e) => panic!("cannot read {}: {}", dir.display(), e),
    };
    for entry in read_dir {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            walk_rs(&path, visit);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            visit(&path);
        }
    }
}

/// Same comment / string stripping as `no_magic_numbers.rs`. Duplicated for
/// test independence — these are small helpers, factoring isn't worth a
/// shared crate.
fn strip_comments_and_strings(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = vec![b' '; bytes.len()];
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            out[i] = b'\n';
        }
    }
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let next = bytes.get(i + 1).copied();
        if b == b'/' && next == Some(b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
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
        if b == b'\'' {
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
            out[i] = b;
            i += 1;
            continue;
        }
        out[i] = b;
        i += 1;
    }
    String::from_utf8(out).unwrap()
}
