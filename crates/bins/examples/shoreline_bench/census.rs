//! ★ THE CENSUS WRITER: every line the bench prints goes to the terminal at once and into the census
//! file, and a stand's lines go into the file only when the stand ends, so a RED self-check on the
//! stand can turn every other verdict of that stand into VOID first (plan §4.1).
//!
//! **Example.** CHECK-R0 reads 3 differing columns at the plain's rung 9. Every line of the plain
//! then carries `VOID` in the file, and the file opens its plain section with a banner that says
//! why; the terminal showed the lines as they ran.

use std::fs::File;
use std::io::Write;

/// One census line: `CENSUS <GATE>.<stand>.<rule>.rNN <value> <band>[source] <verdict>`.
pub struct Census {
    file: Option<File>,
    /// The lines of the stand now running, held until it ends.
    held: Vec<String>,
    /// The stand now running, if any.
    stand: Option<String>,
    /// The stand's RED self-checks.
    stand_reds: Vec<String>,
    /// Every stand that was voided, with its first red check.
    pub voided: Vec<(String, String)>,
    /// Every CENSUS line, as (gate, value, band, verdict), for the summary tables.
    pub all: Vec<(String, String, String, String)>,
}

impl Census {
    /// A writer into `path` (or the terminal alone).
    #[must_use]
    pub fn new(path: Option<&str>) -> Census {
        let file = path.map(|p| {
            if let Some(dir) = std::path::Path::new(p).parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            File::create(p).expect("the census file")
        });
        Census {
            file,
            held: Vec::new(),
            stand: None,
            stand_reds: Vec::new(),
            voided: Vec::new(),
            all: Vec::new(),
        }
    }

    fn emit(&mut self, text: &str) {
        if self.stand.is_some() {
            self.held.push(text.to_string());
        } else if let Some(f) = self.file.as_mut() {
            let _ = writeln!(f, "{text}");
        }
    }

    /// A text line.
    pub fn say(&mut self, text: impl AsRef<str>) {
        let t = text.as_ref();
        println!("{t}");
        self.emit(t);
    }

    /// ★ ONE CENSUS LINE. A §4.1 self-check that reads RED voids the stand it stands in ([`voids`]).
    pub fn line(&mut self, gate: &str, value: &str, band: &str, verdict: &str) {
        let text = format!("CENSUS {gate} {value} {band} {verdict}");
        println!("{text}");
        if voids(gate) && verdict == "RED" && self.stand.is_some() {
            self.stand_reds.push(gate.to_string());
        }
        self.all.push((
            gate.to_string(),
            value.to_string(),
            band.to_string(),
            verdict.to_string(),
        ));
        self.emit(&text);
    }

    /// A line whose verdict is GREEN where `pass` holds, RED otherwise.
    pub fn judged(&mut self, gate: &str, value: &str, band: &str, pass: bool) {
        self.line(gate, value, band, if pass { "GREEN" } else { "RED" });
    }

    /// A reading with no band.
    pub fn reading(&mut self, gate: &str, value: &str, why: &str) {
        self.line(gate, value, &format!("no-band[{why}]"), "NO-BAND");
    }

    /// Opens a stand: its lines are held until [`Census::end_stand`].
    pub fn begin_stand(&mut self, name: &str) {
        self.stand = Some(name.to_string());
        self.held.clear();
        self.stand_reds.clear();
    }

    /// ★ CLOSES A STAND: a RED self-check turns every other verdict of the stand into VOID, and the
    /// section opens with a banner that names the check.
    pub fn end_stand(&mut self) {
        let Some(name) = self.stand.take() else {
            return;
        };
        let void = !self.stand_reds.is_empty();
        let mut out = Vec::new();
        if void {
            let first = self.stand_reds[0].clone();
            out.push(format!(
                "######## VOID: the stand `{name}` has {} RED self-check(s), the first {first}; every other line of this stand reads VOID ########",
                self.stand_reds.len()
            ));
            self.voided.push((name.clone(), first));
        }
        for text in std::mem::take(&mut self.held) {
            let gate = text.split(' ').nth(1).unwrap_or("");
            if void && text.starts_with("CENSUS ") && !voids(gate) {
                let mut words: Vec<&str> = text.split(' ').collect();
                if let Some(last) = words.last_mut() {
                    *last = "VOID";
                }
                out.push(words.join(" "));
            } else {
                out.push(text);
            }
        }
        if let Some(f) = self.file.as_mut() {
            for t in out {
                let _ = writeln!(f, "{t}");
            }
        }
        if void {
            for (gate, _, _, verdict) in &mut self.all {
                if gate.contains(&format!(".{name}.")) && !voids(gate) {
                    *verdict = "VOID".to_string();
                }
            }
        }
    }
}

/// ★ THE SELF-CHECKS THAT VOID A STAND (plan §4.1): the instrument's own proofs that it reads the
/// shipped path. A rule's own lines (CHECK-DRAWN-IS-SIDE, the candidates' cap, the rotation) are
/// results, never a proof of the instrument, and void nothing.
#[must_use]
pub fn voids(gate: &str) -> bool {
    let name = gate.split('.').next().unwrap_or("");
    matches!(
        name,
        "CHECK-R0"
            | "CHECK-CACHE-FIELD"
            | "CHECK-SHEET-COLUMNS"
            | "BASECHECK"
            | "CHECK-FINE-SPLIT"
            | "CHECK-R0-F64"
    )
}

/// A float as a census value with a unit, no space.
#[must_use]
pub fn val(x: f64, digits: usize, unit: &str) -> String {
    if x.is_finite() {
        format!("{x:.digits$}{unit}")
    } else {
        format!("none{unit}")
    }
}
