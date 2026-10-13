//! A diagnostic (not a gate): two pictures side by side for the owner's look, the left one first, a white gap between.
//!
//! ```text
//! cargo run --release -p vd-bins --example pair_png -- <left.png> <right.png> <out.png>
//! ```
//!
//! **Example.** `pair_png taiwan_lem_generated_tile.png taiwan_lem_earth_tile.png taiwan_pair.png` writes the made
//! ground of the Taiwan square beside Earth's real one.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        3,
        "usage: pair_png <left.png> <right.png> <out.png>"
    );
    let left = image::open(&args[0])
        .expect("the left picture reads")
        .to_rgb8();
    let right = image::open(&args[1])
        .expect("the right picture reads")
        .to_rgb8();
    // The right one scaled to the left one's height (Earth's 30 m maps are smaller than the made ones).
    let right = if right.height() == left.height() {
        right
    } else {
        let w = (u64::from(right.width()) * u64::from(left.height()) / u64::from(right.height()))
            as u32;
        image::imageops::resize(
            &right,
            w,
            left.height(),
            image::imageops::FilterType::Triangle,
        )
    };
    let gap = 16;
    let (w, h) = (
        left.width() + gap + right.width(),
        left.height().max(right.height()),
    );
    let mut out = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
    image::imageops::overlay(&mut out, &left, 0, 0);
    image::imageops::overlay(&mut out, &right, i64::from(left.width() + gap), 0);
    out.save(&args[2]).expect("the pair writes");
    println!("{}", args[2]);
}
