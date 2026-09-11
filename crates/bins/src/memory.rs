//! THE CLIENT'S MEMORY, READ FROM OUTSIDE (M8-2, ruling V17 item 1): what a capture client
//! costs the machine, from the operating system's own tools, so a gate can print it and a
//! flight can measure its growth.
//!
//! Three readings, each for a reason:
//! - the RESIDENT count (`ps`, RSS): what sits in RAM now. It FALLS when the machine is short of
//!   memory (the system compresses or swaps the process's pages), so a still-growing client can
//!   read smaller than it is. MEASURED on 2026-09-10 with 7.5 GB in swap: the hill stand read
//!   1 904 MB on one flight and 4 097 MB on the next, for the same client.
//! - the PHYSICAL FOOTPRINT (`footprint`): what the process holds in RAM PLUS what the system
//!   compressed or swapped on its behalf, by category — the graphics buffers on unified memory,
//!   the allocator's small and large blocks. It does not fall under pressure. This is the number
//!   a report quotes.
//! - the HEAP SUMMARY (`heap -s`): what the allocator holds IN USE, by size class, so a footprint
//!   category reads as held or as freed-and-retained (dirty pages the allocator kept).
//!
//! Example: a client at the ground stand draws 2 460 MB of chunks. Its footprint reads 6.6 GB:
//! 2 870 MB of graphics buffers (the chunks on the GPU, the slabs rounded up), 1 967 MB of small
//! malloc blocks of which the heap holds 712 MB in use, and 1 546 MB of unmapped memory the
//! process still owns. The resident count read 2 052 MB at that moment.

use std::process::Command;

/// One reading of a process's memory, in megabytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MemoryRead {
    /// The resident count (`ps` RSS).
    pub resident_mb: f64,
    /// The physical footprint's total.
    pub footprint_mb: f64,
    /// The footprint's graphics buffers (`Owned physical footprint (unmapped) (graphics)`).
    pub graphics_mb: f64,
    /// The footprint's owned-but-unmapped memory that is not graphics.
    pub unmapped_mb: f64,
    /// The footprint's small malloc blocks (`MALLOC_SMALL`).
    pub malloc_small_mb: f64,
    /// The footprint's large malloc blocks (`MALLOC_LARGE`).
    pub malloc_large_mb: f64,
}

impl MemoryRead {
    /// Read a live process. A tool that fails leaves its fields at zero.
    #[must_use]
    pub fn of(pid: u32) -> MemoryRead {
        let report = footprint_report(pid);
        MemoryRead {
            resident_mb: resident_mb(pid),
            ..MemoryRead::from_footprint(&report)
        }
    }

    /// The footprint fields from a `footprint` report; the resident count stays zero.
    #[must_use]
    pub fn from_footprint(report: &str) -> MemoryRead {
        MemoryRead {
            resident_mb: 0.0,
            footprint_mb: footprint_of(report).unwrap_or(0.0),
            graphics_mb: footprint_category(report, GRAPHICS_ROW).unwrap_or(0.0),
            unmapped_mb: footprint_category(report, UNMAPPED_ROW).unwrap_or(0.0),
            malloc_small_mb: footprint_category(report, "MALLOC_SMALL").unwrap_or(0.0),
            malloc_large_mb: footprint_category(report, "MALLOC_LARGE").unwrap_or(0.0),
        }
    }

    /// The growth from an earlier reading, field by field.
    #[must_use]
    pub fn since(self, earlier: MemoryRead) -> MemoryRead {
        MemoryRead {
            resident_mb: self.resident_mb - earlier.resident_mb,
            footprint_mb: self.footprint_mb - earlier.footprint_mb,
            graphics_mb: self.graphics_mb - earlier.graphics_mb,
            unmapped_mb: self.unmapped_mb - earlier.unmapped_mb,
            malloc_small_mb: self.malloc_small_mb - earlier.malloc_small_mb,
            malloc_large_mb: self.malloc_large_mb - earlier.malloc_large_mb,
        }
    }
}

impl std::fmt::Display for MemoryRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "footprint {:.0} MB (graphics {:.0}, unmapped {:.0}, malloc small {:.0}, malloc large \
             {:.0}), resident {:.0} MB",
            self.footprint_mb,
            self.graphics_mb,
            self.unmapped_mb,
            self.malloc_small_mb,
            self.malloc_large_mb,
            self.resident_mb
        )
    }
}

/// The footprint table's category for the graphics buffers on unified memory.
const GRAPHICS_ROW: &str = "Owned physical footprint (unmapped) (graphics)";
/// The footprint table's category for owned-but-unmapped memory that is not graphics.
const UNMAPPED_ROW: &str = "Owned physical footprint (unmapped)";

/// A process's resident memory in megabytes (`ps` RSS); zero when the reading fails.
#[must_use]
pub fn resident_mb(pid: u32) -> f64 {
    Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|out| {
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse::<f64>()
                .ok()
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

/// A process's physical footprint report (`footprint -p`), by category; empty when the tool
/// fails.
#[must_use]
pub fn footprint_report(pid: u32) -> String {
    tool_output("footprint", &["-p", &pid.to_string()])
}

/// A process's heap summary (`heap -s`): the malloc blocks in use, by zone and by size class,
/// largest first; empty when the tool fails.
#[must_use]
pub fn heap_summary(pid: u32) -> String {
    tool_output("heap", &["-s", &pid.to_string()])
}

/// A process's virtual memory summary (`vmmap --summary`): every region type with its virtual,
/// resident, dirty and swapped sizes — the table that names what the footprint's owned-but-
/// unmapped memory is (a graphics resource, a surface, an allocator region); empty when the tool
/// fails.
#[must_use]
pub fn vmmap_summary(pid: u32) -> String {
    tool_output("vmmap", &["--summary", &pid.to_string()])
}

/// The rows of a `vmmap --summary` worth printing: the region table from its column header to
/// its total, each row trimmed.
#[must_use]
pub fn vmmap_rows(summary: &str) -> Vec<String> {
    let lines: Vec<&str> = summary.lines().collect();
    let Some(at) = lines.iter().position(|row| row.starts_with("REGION TYPE")) else {
        return Vec::new();
    };
    // The column names take two lines; the first stands right above `REGION TYPE`.
    let from = at.saturating_sub(usize::from(at > 0 && !lines[at - 1].trim().is_empty()));
    lines[from..]
        .iter()
        .take_while(|row| !row.trim().is_empty())
        .map(|row| row.trim().to_owned())
        .collect()
}

/// THE GPU'S BUSY SHARE (D8-8's ablation): the device's and the renderer's utilisation in
/// percent, from the graphics driver's own statistics (`ioreg`, the accelerator's
/// `PerformanceStatistics`); `None` where the platform offers none.
#[must_use]
pub fn gpu_busy() -> Option<(f64, f64)> {
    let report = tool_output("ioreg", &["-r", "-d", "1", "-c", "AGXAccelerator"]);
    let pick = |key: &str| -> Option<f64> {
        let at = report.find(key)?;
        let rest = &report[at + key.len()..];
        let rest = rest.strip_prefix("\"=")?;
        let end = rest.find(|c: char| !c.is_ascii_digit())?;
        rest[..end].parse().ok()
    };
    Some((
        pick("Device Utilization %")?,
        pick("Renderer Utilization %")?,
    ))
}

/// The GPU's busy share averaged over `samples` readings `gap` apart: what the GPU does while a
/// stand holds still.
#[must_use]
pub fn gpu_busy_mean(samples: u32, gap: std::time::Duration) -> Option<(f64, f64)> {
    let mut sum = (0.0, 0.0);
    let mut n = 0.0;
    for i in 0..samples {
        if let Some((d, r)) = gpu_busy() {
            sum.0 += d;
            sum.1 += r;
            n += 1.0;
        }
        if i + 1 < samples {
            std::thread::sleep(gap);
        }
    }
    (n > 0.0).then(|| (sum.0 / n, sum.1 / n))
}

/// The rows of a report worth printing: the footprint's largest categories after its header, or
/// the heap summary's zone lines after ITS header.
#[must_use]
pub fn report_rows(report: &str, skip: usize, take: usize) -> Vec<String> {
    report
        .lines()
        .skip(skip)
        .take(take)
        .map(|row| row.trim().to_owned())
        .collect()
}

/// The megabytes on the report's `Footprint: <n> <unit>` line, in any of its units.
#[must_use]
pub fn footprint_of(report: &str) -> Option<f64> {
    let rest = report.split("Footprint: ").nth(1)?;
    let mut words = rest.split_whitespace();
    let n: f64 = words.next()?.parse().ok()?;
    let scale = unit_scale(words.next()?)?;
    Some(n * scale)
}

/// The megabytes of one category row of the footprint table: the dirty column of the row whose
/// category is exactly `name` (so the graphics row and the plain unmapped row stay apart).
#[must_use]
pub fn footprint_category(report: &str, name: &str) -> Option<f64> {
    report.lines().find_map(|line| {
        let mut words = line.split_whitespace();
        let n: f64 = words.next()?.parse().ok()?;
        let scale = unit_scale(words.next()?)?;
        // The category is every word after the counted columns: clean and reclaimable (a number
        // and a unit each) and the region count — so a category with a digit in its name (the
        // tool's `tag 22`) reads whole.
        let category = words.skip(5).collect::<Vec<_>>().join(" ");
        (category == name).then_some(n * scale)
    })
}

/// A unit word's scale to megabytes.
fn unit_scale(unit: &str) -> Option<f64> {
    match unit {
        "B" => Some(1.0 / (1024.0 * 1024.0)),
        "KB" => Some(1.0 / 1024.0),
        "MB" => Some(1.0),
        "GB" => Some(1024.0),
        _ => None,
    }
}

/// A tool's standard output as text, or nothing.
fn tool_output(tool: &str, args: &[&str]) -> String {
    Command::new(tool).args(args).output().map_or_else(
        |_| String::new(),
        |out| String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPORT: &str = "\
======================================================================
client [94413]: 64-bit    Footprint: 6611 MB (16384 bytes per page)
======================================================================

  Dirty      Clean  Reclaimable    Regions    Category
    ---        ---          ---        ---    ---
2870 MB        0 B          0 B        145    Owned physical footprint (unmapped) (graphics)
1967 MB        0 B        32 KB        521    MALLOC_SMALL
1546 MB        0 B          0 B      12371    Owned physical footprint (unmapped)
 161 MB        0 B          0 B         22    MALLOC_LARGE
  45 MB        0 B      1440 KB        682    IOAccelerator (graphics)
8992 KB        0 B        32 KB         26    MALLOC metadata
  16 KB        0 B          0 B          1    tag 22
";

    #[test]
    fn the_total_and_the_categories_read_from_the_table() {
        let read = MemoryRead::from_footprint(REPORT);
        assert_eq!(read.footprint_mb, 6611.0);
        assert_eq!(read.graphics_mb, 2870.0);
        assert_eq!(read.unmapped_mb, 1546.0);
        assert_eq!(read.malloc_small_mb, 1967.0);
        assert_eq!(read.malloc_large_mb, 161.0);
        assert_eq!(read.resident_mb, 0.0);
        assert_eq!(
            footprint_category(REPORT, "MALLOC metadata").map(|mb| (mb * 1024.0).round()),
            Some(8992.0)
        );
        assert_eq!(footprint_category(REPORT, "no such row"), None);
        assert_eq!(
            footprint_category(REPORT, "tag 22").map(|mb| (mb * 1024.0).round()),
            Some(16.0)
        );
        assert_eq!(footprint_category(REPORT, "tag"), None);
    }

    #[test]
    fn every_unit_scales_to_megabytes() {
        assert_eq!(footprint_of("x Footprint: 2 GB y"), Some(2048.0));
        assert_eq!(footprint_of("x Footprint: 512 KB y"), Some(0.5));
        assert_eq!(footprint_of("x Footprint: 1048576 B y"), Some(1.0));
        assert_eq!(footprint_of("x Footprint: 7 TB y"), None);
        assert_eq!(footprint_of("no line"), None);
    }

    #[test]
    fn the_growth_is_field_by_field() {
        let a = MemoryRead::from_footprint(REPORT);
        let b = MemoryRead {
            footprint_mb: a.footprint_mb + 100.0,
            unmapped_mb: a.unmapped_mb + 60.0,
            ..a
        };
        let grew = b.since(a);
        assert_eq!(grew.footprint_mb, 100.0);
        assert_eq!(grew.unmapped_mb, 60.0);
        assert_eq!(grew.graphics_mb, 0.0);
        assert_eq!(
            grew.to_string(),
            "footprint 100 MB (graphics 0, unmapped 60, malloc small 0, malloc large 0), \
             resident 0 MB"
        );
    }

    #[test]
    fn the_rows_after_a_header_are_trimmed() {
        assert_eq!(
            report_rows(REPORT, 6, 2),
            vec![
                "2870 MB        0 B          0 B        145    Owned physical footprint (unmapped) (graphics)",
                "1967 MB        0 B        32 KB        521    MALLOC_SMALL",
            ]
        );
    }

    /// The three tools are the platform's own; off it every reader answers empty or zero.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_live_reading_of_this_process_is_positive() {
        let read = MemoryRead::of(std::process::id());
        assert!(read.resident_mb > 0.0, "{read}");
        assert!(read.footprint_mb > 0.0, "{read}");
        assert!(!heap_summary(std::process::id()).is_empty());
        let rows = vmmap_rows(&vmmap_summary(std::process::id()));
        assert!(
            rows.first().is_some_and(|r| r.starts_with("VIRTUAL")),
            "{rows:?}"
        );
        assert!(
            rows.get(1).is_some_and(|r| r.starts_with("REGION TYPE")),
            "{rows:?}"
        );
        assert!(rows.iter().any(|r| r.starts_with("TOTAL")), "{rows:?}");
    }

    #[test]
    fn the_region_table_reads_from_header_to_total() {
        let summary = "\
Process:  x [1]

                               VIRTUAL RESIDENT    DIRTY  SWAPPED VOLATILE   NONVOL    EMPTY   REGION
REGION TYPE                       SIZE     SIZE     SIZE     SIZE     SIZE     SIZE     SIZE    COUNT
===========                    ======= ========    =====  ======= ========   ======    =====  =======
IOAccelerator                    45.0M    45.0M    45.0M       0K       0K       0K       0K      682
TOTAL                            10.0G     2.0G     6.5G       0K       0K       0K       0K    14000

Foot
";
        let rows = vmmap_rows(summary);
        assert_eq!(rows.len(), 5);
        assert!(rows[0].starts_with("VIRTUAL"));
        assert!(rows[3].starts_with("IOAccelerator"));
        assert!(rows[4].starts_with("TOTAL"));
        assert!(vmmap_rows("nothing here").is_empty());
        // A table with no first name line, and one whose `REGION TYPE` opens the text.
        assert_eq!(vmmap_rows("x\n\nREGION TYPE a\nrow\n\n").len(), 2);
        assert_eq!(vmmap_rows("REGION TYPE a\nrow").len(), 2);
    }

    /// The GPU's busy share reads on the platform that offers it, and averages.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_gpu_busy_share_reads() {
        let (d, r) = gpu_busy().expect("the accelerator's statistics");
        assert!(
            (0.0..=100.0).contains(&d) && (0.0..=100.0).contains(&r),
            "{d} {r}"
        );
        assert!(gpu_busy_mean(2, std::time::Duration::from_millis(10)).is_some());
    }

    #[test]
    fn a_missing_process_reads_zero() {
        let read = MemoryRead::of(u32::MAX);
        assert_eq!(read, MemoryRead::default());
        assert_eq!(tool_output("no-such-tool-anywhere", &[]), "");
    }
}
