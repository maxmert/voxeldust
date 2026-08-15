#!/usr/bin/env python3
"""The HR5 coverage-fast PASS/FAIL decision (owner ruling 2026-08-15, coverage option 3 —
docs/design/owner_decisions_2026-08-15.md item 8; the ledger row lives in docs/design/DEFERRED.md).

WHY THIS EXISTS. cargo-llvm-cov's summaries count coverage rows PER COMPILED RECORD. A Tier-A
workspace compiles the same source into many objects (each crate's own test binary, plus every
dependent crate's), so the report carries rows that own NO source-line miss in the MERGED view:

  * a `?` operator's never-taken early-return micro-region (a one-column span on a covered line);
  * a lazy closure argument's body (`map_or_else`'s default arm, an assert message) never evaluated;
  * a per-crate-hash duplicate instantiation of a span another record fully covers.

Those rows fail the raw `--fail-under-regions/functions` gate while the merged lcov shows ZERO
missed lines — the gate was red on rows no source line backs. The owner-picked rule is OBJECTIVE
(never a blessed location list):

  * REAL missed LINE    = a DA row of the merged lcov with zero hits          -> still FAILS
  * REAL missed BRANCH  = a BRDA row of the merged lcov with a never-taken side -> still FAILS
  * every other missed row (region/function/line counted per-record only)    -> DROPPED + PRINTED

The dropped count is printed EVERY run (shed-loud), with the never-executed micro-spans listed, so
growth in the artifact class stays visible in review even though it does not fail the build.

Usage: coverage_gate.py <report.json> <report.lcov>
Exit 0 iff no REAL miss remains; exit 1 otherwise, listing every real miss.
"""

import json
import sys
from collections import defaultdict


def parse_lcov(path):
    """The merged view: (missed DA lines, missed BRDA sides), each as (file, line[, detail])."""
    missed_da = []
    missed_brda = []
    current = None
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            row = raw.strip()
            if row.startswith("SF:"):
                current = row[3:]
            elif row.startswith("DA:"):
                line_no, count = row[3:].split(",")[:2]
                if int(count) == 0:
                    missed_da.append((current, int(line_no)))
            elif row.startswith("BRDA:"):
                line_no, block, branch, taken = row[5:].split(",")
                # "-" = the enclosing block never ran; "0" = reached but this side never taken.
                if taken in ("-", "0"):
                    missed_brda.append((current, int(line_no), block, branch))
    return missed_da, missed_brda


def json_missed_totals(data):
    """The tool's own per-record missed totals — what the raw gate used to fail on."""
    totals = data["totals"]
    return {
        name: totals[name]["count"] - totals[name]["covered"]
        for name in ("regions", "functions", "lines", "branches")
    }


def never_executed_spans(data):
    """Code-region spans with a summed count of ZERO across every compiled record — the
    never-executed micro-spans (`?` early-returns, lazy closure bodies). Diagnostic listing for
    the dropped rows; spans another record covers (duplicate instantiations) merge to nonzero
    and do not appear."""
    in_report = {f["filename"] for f in data["files"]}
    merged = defaultdict(int)
    for func in data.get("functions", []):
        for region in func.get("regions", []):
            # region = [line_start, col_start, line_end, col_end, count, file_id, expanded, kind]
            if region[7] != 0:  # code regions only
                continue
            filename = func["filenames"][region[5]]
            if filename not in in_report:
                continue
            merged[(filename, region[0], region[1], region[2], region[3])] += region[4]
    return sorted(key for key, total in merged.items() if total == 0)


def short(path):
    marker = "crates/"
    at = path.find(marker)
    return path[at:] if at != -1 else path


def main():
    if len(sys.argv) != 3:
        print("usage: coverage_gate.py <report.json> <report.lcov>", file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8") as handle:
        data = json.load(handle)["data"][0]
    missed_da, missed_brda = parse_lcov(sys.argv[2])

    tool = json_missed_totals(data)
    tool_total = sum(tool.values())
    real_total = len(missed_da) + len(missed_brda)
    dropped = tool_total - real_total
    if dropped < 0:
        # The merged lcov found MORE misses than the per-record totals — the two reports disagree
        # in a direction the model forbids; refuse loudly rather than gate on garbage.
        print(
            f"COVERAGE GATE: INCOHERENT reports (tool-missed {tool_total} < merged-real {real_total})"
        )
        return 1

    print(
        "COVERAGE GATE (option 3, owner 2026-08-15): "
        f"tool-missed rows {tool_total} "
        f"(regions {tool['regions']}, functions {tool['functions']}, "
        f"lines {tool['lines']}, branches {tool['branches']}) | "
        f"ARTIFACT ROWS DROPPED (no source-line miss in the merged report): {dropped} | "
        f"real misses: {real_total}"
    )
    for span in never_executed_spans(data):
        print(
            f"  dropped (never-executed micro-span, line covered): "
            f"{short(span[0])}:{span[1]}:{span[2]}-{span[3]}:{span[4]}"
        )

    if missed_da:
        print(f"REAL MISSED LINES ({len(missed_da)}):")
        for filename, line_no in missed_da:
            print(f"  {short(filename)}:{line_no}")
    if missed_brda:
        print(f"REAL MISSED BRANCHES ({len(missed_brda)}):")
        for filename, line_no, block, branch in missed_brda:
            print(f"  {short(filename)}:{line_no} (block {block}, branch {branch})")
    if real_total:
        print("COVERAGE GATE: FAIL — real misses remain (see above)")
        return 1
    print("COVERAGE GATE: PASS — 100% of merged source lines and branch sides covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
