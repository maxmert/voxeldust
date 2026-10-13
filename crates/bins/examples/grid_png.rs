//! A diagnostic (not a gate): a node grid drawn over a picture — the node edges in red, the node centres' lines dotted in
//! blue — so a straight line in the ground can be matched to the 8 km lattice.
//!
//! ```text
//! cargo run --release -p vd-bins --example grid_png -- <in.png> <pixels a node> <out.png>
//! ```
//!
//! **Example.** `grid_png belt_tile_a.png 64 grid.png` marks every 8 km node of a 128 m tile picture (64 pixels a node).

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        3,
        "usage: grid_png <in.png> <pixels a node> <out.png>"
    );
    let step: u32 = args[1].parse().expect("a whole number");
    let mut img = image::open(&args[0]).expect("the picture reads").to_rgb8();
    let (w, h) = img.dimensions();
    for y in 0..h {
        for x in 0..w {
            if x % step == 0 || y % step == 0 {
                img.put_pixel(x, y, image::Rgb([220, 30, 30]));
            } else if (x % step == step / 2 && y % 4 < 2) || (y % step == step / 2 && x % 4 < 2) {
                img.put_pixel(x, y, image::Rgb([40, 80, 230]));
            }
        }
    }
    img.save(&args[2]).expect("the picture writes");
    println!("{}", args[2]);
}
