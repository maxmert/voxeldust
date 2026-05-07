//! Platform-independent key name mapping.
//!
//! Key names use winit `KeyCode` variant names as canonical strings stored in configs
//! and sent over the network. Display names provide human-readable labels for the UI.
//!
//! Client-side conversion to/from winit `KeyCode` lives in `voxydust/src/input.rs`.

/// (key_name, display_name) pairs for all supported keys.
pub const KEY_NAMES: &[(&str, &str)] = &[
    // Letters
    ("KeyA", "A"), ("KeyB", "B"), ("KeyC", "C"), ("KeyD", "D"),
    ("KeyE", "E"), ("KeyF", "F"), ("KeyG", "G"), ("KeyH", "H"),
    ("KeyI", "I"), ("KeyJ", "J"), ("KeyK", "K"), ("KeyL", "L"),
    ("KeyM", "M"), ("KeyN", "N"), ("KeyO", "O"), ("KeyP", "P"),
    ("KeyQ", "Q"), ("KeyR", "R"), ("KeyS", "S"), ("KeyT", "T"),
    ("KeyU", "U"), ("KeyV", "V"), ("KeyW", "W"), ("KeyX", "X"),
    ("KeyY", "Y"), ("KeyZ", "Z"),
    // Digits
    ("Digit1", "1"), ("Digit2", "2"), ("Digit3", "3"), ("Digit4", "4"),
    ("Digit5", "5"), ("Digit6", "6"), ("Digit7", "7"), ("Digit8", "8"),
    ("Digit9", "9"), ("Digit0", "0"),
    // Function keys
    ("F1", "F1"), ("F2", "F2"), ("F3", "F3"), ("F4", "F4"),
    ("F5", "F5"), ("F6", "F6"), ("F7", "F7"), ("F8", "F8"),
    ("F9", "F9"), ("F10", "F10"), ("F11", "F11"), ("F12", "F12"),
    // Special
    ("Space", "Space"), ("Enter", "Enter"), ("Tab", "Tab"),
    ("Escape", "Esc"), ("Backspace", "Bksp"), ("Delete", "Del"),
    ("Insert", "Ins"), ("Home", "Home"), ("End", "End"),
    ("PageUp", "PgUp"), ("PageDown", "PgDn"),
    // Arrows
    ("ArrowUp", "\u{2191}"), ("ArrowDown", "\u{2193}"),
    ("ArrowLeft", "\u{2190}"), ("ArrowRight", "\u{2192}"),
    // Modifiers
    ("ShiftLeft", "L-Shift"), ("ShiftRight", "R-Shift"),
    ("ControlLeft", "L-Ctrl"), ("ControlRight", "R-Ctrl"),
    ("AltLeft", "L-Alt"), ("AltRight", "R-Alt"),
    // Punctuation
    ("Minus", "-"), ("Equal", "="),
    ("BracketLeft", "["), ("BracketRight", "]"),
    ("Backslash", "\\"), ("Semicolon", ";"), ("Quote", "'"),
    ("Comma", ","), ("Period", "."), ("Slash", "/"),
    ("Backquote", "`"),
    // Mouse buttons
    ("MouseLeft", "LMB"), ("MouseRight", "RMB"), ("MouseMiddle", "MMB"),
    ("MouseButton4", "M4"), ("MouseButton5", "M5"),
];

/// Get the human-readable display name for a key name.
/// Returns the key name itself if not found.
pub fn key_display_name(key_name: &str) -> &str {
    KEY_NAMES
        .iter()
        .find(|(name, _)| *name == key_name)
        .map(|(_, display)| *display)
        .unwrap_or(key_name)
}

/// Check if a key name is a mouse button.
pub fn is_mouse_button(key_name: &str) -> bool {
    key_name.starts_with("Mouse")
}

/// All supported key names and their display names.
pub fn all_key_names() -> &'static [(&'static str, &'static str)] {
    KEY_NAMES
}
