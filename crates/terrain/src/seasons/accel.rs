//! ★ ANDERSON'S ACCELERATION (Walker & Ni 2011) for the seasons' fixed-point solves: a round `G` maps a start `x` to
//! `G(x)`; the next start is `G(x)` corrected by the last few rounds' differences so that their combined change is least —
//! GMRES's answer for a linear `G` — so the slow modes a round leaves settle in a few rounds. The answer is still `G`'s own
//! fixed point: each caller's stop reads `G`'s change.
//!
//! **Example.** The mountains' rain on the home planet's warm passes: a call that needed 400 rounds of sweeps settles in
//! 37; the heated winds' drag, whose plain rounds crawl, settles in a handful.

use crate::gf::Gf;

/// `Σ a_i·b_i` in the index order.
fn dot(a: &[Gf], b: &[Gf]) -> Gf {
    a.iter().zip(b).fold(Gf::ZERO, |s, (u, v)| s + *u * *v)
}

/// Anderson's depth: the rounds whose differences the next start combines (Walker & Ni 2011 use 1–5).
pub const ANDERSON_DEPTH: usize = 4;

/// ★ ANDERSON'S ACCELERATION's words: the last rounds' changes of the residual `f = G(x) − x` and of `G(x)`, a ring of
/// [`ANDERSON_DEPTH`].
pub(crate) struct Anderson {
    df: Vec<Vec<Gf>>,
    dg: Vec<Vec<Gf>>,
    /// The differences' products `⟨Δf_p, Δf_q⟩`, kept: a round adds only its own row.
    gram: [[Gf; ANDERSON_DEPTH]; ANDERSON_DEPTH],
    f_last: Vec<Gf>,
    g_last: Vec<Gf>,
    count: usize,
    next: usize,
    last_change: Gf,
}

impl Anderson {
    pub(crate) fn new() -> Anderson {
        Anderson {
            df: Vec::new(),
            dg: Vec::new(),
            gram: [[Gf::ZERO; ANDERSON_DEPTH]; ANDERSON_DEPTH],
            f_last: Vec::new(),
            g_last: Vec::new(),
            count: 0,
            next: 0,
            last_change: Gf::ZERO,
        }
    }

    /// The next start from `x` and its round `g = G(x)` (`change` the largest `|g − x|`): `g − Σ γ_p·Δg_p`, the `γ` that
    /// make `f − Σ γ_p·Δf_p` least in its sum of squares (the normal equations, solved exactly). A round whose change
    /// more than doubles drops the history (the round alone, `x = g`, from there). The start is held in `range`, where the
    /// answer lies (the mountains' factor in `[0, 1]`; the drag's speed at least the gust), so no start leaves the law's
    /// own range — a bound on an iterate, never on the answer.
    #[allow(
        clippy::needless_range_loop,
        reason = "a small symmetric matrix filled by its row and column indices"
    )]
    pub(crate) fn step(&mut self, x: &mut [Gf], g: &[Gf], change: Gf, range: (Gf, Gf)) {
        let n = x.len();
        if self.f_last.len() != n {
            self.f_last = vec![Gf::ZERO; n];
            self.g_last = vec![Gf::ZERO; n];
            self.count = 0;
        } else if change > Gf::TWO * self.last_change {
            self.count = 0;
        } else {
            if self.df.len() < ANDERSON_DEPTH {
                self.df.push(vec![Gf::ZERO; n]);
                self.dg.push(vec![Gf::ZERO; n]);
            }
            let slot = if self.df.len() < ANDERSON_DEPTH || self.count < ANDERSON_DEPTH {
                self.count.min(self.df.len() - 1)
            } else {
                self.next
            };
            let (df, dg) = (&mut self.df[slot], &mut self.dg[slot]);
            for i in 0..n {
                let f = g[i] - x[i];
                df[i] = f - self.f_last[i];
                dg[i] = g[i] - self.g_last[i];
            }
            self.next = (slot + 1) % ANDERSON_DEPTH;
            self.count = (self.count + 1).min(ANDERSON_DEPTH);
            for q in 0..self.count {
                let v = dot(&self.df[slot], &self.df[q]);
                self.gram[slot][q] = v;
                self.gram[q][slot] = v;
            }
        }
        for i in 0..n {
            self.f_last[i] = g[i] - x[i];
            self.g_last[i] = g[i];
        }
        self.last_change = change;
        let m = self.count;
        if m == 0 {
            x.copy_from_slice(g);
            return;
        }
        let mut a = self.gram;
        let mut b = [Gf::ZERO; ANDERSON_DEPTH];
        for p in 0..m {
            b[p] = dot(&self.df[p], &self.f_last);
        }
        let Some(gamma) = solve_small(&mut a, &mut b, m) else {
            x.copy_from_slice(g);
            self.count = 0;
            return;
        };
        for i in 0..n {
            let mut v = g[i];
            for (p, &c) in gamma.iter().enumerate().take(m) {
                v -= c * self.dg[p][i];
            }
            x[i] = v.clamp(range.0, range.1);
        }
    }
}

/// `a·γ = b` for the first `m` rows, by elimination with partial pivoting; `None` where a pivot vanishes (the rounds'
/// differences have no independent direction left).
#[allow(
    clippy::needless_range_loop,
    reason = "elimination on a small matrix by its row and column indices"
)]
fn solve_small(
    a: &mut [[Gf; ANDERSON_DEPTH]; ANDERSON_DEPTH],
    b: &mut [Gf; ANDERSON_DEPTH],
    m: usize,
) -> Option<[Gf; ANDERSON_DEPTH]> {
    let scale = (0..m).fold(Gf::ZERO, |s, p| s.greater(a[p][p]));
    if scale <= Gf::ZERO {
        return None;
    }
    for col in 0..m {
        let p = (col..m).fold(col, |best, r| {
            if a[r][col].abs() > a[best][col].abs() {
                r
            } else {
                best
            }
        });
        if a[p][col].abs() <= scale * Gf::from_f64(1e-13) {
            return None;
        }
        a.swap(col, p);
        b.swap(col, p);
        for r in col + 1..m {
            let l = a[r][col] / a[col][col];
            for q in col..m {
                let v = a[col][q];
                a[r][q] -= l * v;
            }
            let v = b[col];
            b[r] -= l * v;
        }
    }
    let mut x = [Gf::ZERO; ANDERSON_DEPTH];
    for r in (0..m).rev() {
        let mut s = b[r];
        for q in r + 1..m {
            s -= a[r][q] * x[q];
        }
        x[r] = s / a[r][r];
    }
    Some(x)
}
