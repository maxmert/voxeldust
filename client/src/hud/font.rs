//! Minimal 5x7 bitmap font for HUD widgets.
//!
//! Covers uppercase A-Z, digits 0-9, space, and a small punctuation
//! set that suffices for numeric + status captions: `. , : ; / % -
//! + ! ? ( ) [ ]`. Lowercase falls through to uppercase (we render
//! uppercase only for HUD legibility; widget code may uppercase input
//! before calling).
//!
//! Glyphs are stored as 7 bytes per char (one byte per row). Bits 0-4
//! of each byte are used (bit 4 = leftmost pixel); higher bits are
//! ignored. `draw_text` walks the string and plots each glyph with a
//! 1-pixel advance gap.
//!
//! All glyph data is inline (no external file, no asset loader).

const GLYPH_W: usize = 5;
pub const GLYPH_H: usize = 7;
const ADVANCE: usize = GLYPH_W + 1;

fn glyph(c: char) -> Option<&'static [u8; 7]> {
    let c = c.to_ascii_uppercase();
    Some(match c {
        ' ' => &[0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        '0' => &[0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => &[0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => &[0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => &[0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => &[0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => &[0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => &[0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => &[0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => &[0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => &[0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' => &[0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => &[0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => &[0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => &[0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => &[0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => &[0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => &[0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => &[0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => &[0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => &[0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' => &[0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => &[0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => &[0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => &[0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => &[0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => &[0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => &[0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => &[0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => &[0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => &[0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => &[0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => &[0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => &[0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => &[0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => &[0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => &[0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '.' => &[0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        ',' => &[0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000],
        ':' => &[0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        ';' => &[0b00000, 0b01100, 0b01100, 0b00000, 0b00100, 0b00100, 0b01000],
        '/' => &[0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '\\' => &[0b10000, 0b01000, 0b01000, 0b00100, 0b00010, 0b00010, 0b00001],
        '%' => &[0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '-' => &[0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '+' => &[0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '!' => &[0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
        '?' => &[0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100],
        '(' => &[0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => &[0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '[' => &[0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => &[0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '_' => &[0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        '#' => &[0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010],
        '*' => &[0b00000, 0b01010, 0b00100, 0b11111, 0b00100, 0b01010, 0b00000],
        '=' => &[0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        '<' => &[0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        '>' => &[0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
        '°' => &[0b01110, 0b01010, 0b01110, 0b00000, 0b00000, 0b00000, 0b00000],
        _ => return None,
    })
}

/// Draw `text` onto `pixels` at `(x, y)` with the given integer
/// `scale` and RGBA color. Glyphs are plotted in their natural order;
/// characters without a glyph are skipped. Out-of-bounds pixels are
/// clipped. Returns the advance width in pixels.
pub fn draw_text(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    text: &str,
    mut x: i32,
    y: i32,
    scale: i32,
    color: [u8; 4],
) -> i32 {
    let start_x = x;
    for c in text.chars() {
        draw_char(pixels, width, height, c, x, y, scale, color);
        x += (ADVANCE as i32) * scale;
    }
    x - start_x
}

pub fn text_width(text: &str, scale: i32) -> i32 {
    (text.chars().count() as i32) * (ADVANCE as i32) * scale
}

fn draw_char(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    c: char,
    x: i32,
    y: i32,
    scale: i32,
    color: [u8; 4],
) {
    let Some(rows) = glyph(c) else { return };
    for (ry, row_byte) in rows.iter().enumerate() {
        for rx in 0..GLYPH_W {
            let bit = 1 << (GLYPH_W - 1 - rx);
            if (*row_byte & bit) == 0 {
                continue;
            }
            for dy in 0..scale {
                for dx in 0..scale {
                    let px = x + (rx as i32) * scale + dx;
                    let py = y + (ry as i32) * scale + dy;
                    if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                        continue;
                    }
                    let idx = (py as usize * width + px as usize) * 4;
                    if idx + 3 >= pixels.len() {
                        continue;
                    }
                    pixels[idx] = color[0];
                    pixels[idx + 1] = color[1];
                    pixels[idx + 2] = color[2];
                    pixels[idx + 3] = color[3];
                }
            }
        }
    }
}
