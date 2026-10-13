//! A diagnostic (not a gate): a piece of a picture, enlarged with no smoothing, for a close look at a defect.
//!
//! ```text
//! cargo run --release -p vd-bins --example crop_png -- <in.png> <x> <y> <w> <h> <scale> <out.png>
//! ```
//!
//! **Example.** `crop_png belt_stitched.png 0 0 256 256 3 corner.png` writes the map's north-west corner three times
//! larger, so a 128 m cell shows as 24 pixels.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        7,
        "usage: crop_png <in.png> <x> <y> <w> <h> <scale> <out.png>"
    );
    let n = |k: usize| -> u32 { args[k].parse().expect("a whole number") };
    let img = image::open(&args[0]).expect("the picture reads").to_rgb8();
    let piece = image::imageops::crop_imm(&img, n(1), n(2), n(3), n(4)).to_image();
    let s = n(5);
    let out = image::imageops::resize(
        &piece,
        piece.width() * s,
        piece.height() * s,
        image::imageops::FilterType::Nearest,
    );
    out.save(&args[6]).expect("the piece writes");
    println!("{}", args[6]);
}
