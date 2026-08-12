"""Builds a self-contained HTML report comparing `output_face.csv` against
a real LiveLinkFace ground-truth capture, every channel, correlation and
magnitude, grouped and color-coded, meant to be regenerated after every
pipeline change that could move these numbers, not a one-off, and against
every ground-truth clip available, not just one (see `measure_arkit_error.
_find_ground_truth_csv`: any `<run_dir>` with its own paired `*_raw.csv`/
`*_neutral.csv`/`frame_log.csv` works).
Reuses `measure_arkit_error.aligned_channel_data` for the actual alignment/loading
so the two tools can never disagree on a number.

Usage: `pixi run ground-truth-report <run_dir> [out_path]`
`out_path` defaults to `<run_dir>/stage9_face/ground_truth_report.html`, alongside
the other face previews.
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_arkit_error import ALL_COMPARABLE_COLUMNS, aligned_channel_data  # noqa: E402

# Channel grouping for the report, purely presentational (which section a
# channel prints under), independent of ALL_COMPARABLE_COLUMNS' own CSV
# column order. Every name in ALL_COMPARABLE_COLUMNS must appear exactly
# once across these groups, checked at build time, not assumed.
GROUPS: list[tuple[str, list[str], str]] = [
    ("Jaw", ["JawOpen", "JawLeft", "JawRight", "JawForward"],
     "Group D, direct from jaw_pose rotation."),
    ("Eyes, look direction & blink",
     ["EyeBlinkLeft", "EyeBlinkRight", "EyeLookInLeft", "EyeLookInRight", "EyeLookOutLeft", "EyeLookOutRight",
      "EyeLookUpLeft", "EyeLookUpRight", "EyeLookDownLeft", "EyeLookDownRight"],
     "Group D (gaze) + geometric eyelid measurement."),
    ("Eyes, squint & wide", ["EyeSquintLeft", "EyeSquintRight", "EyeWideLeft", "EyeWideRight"],
     "EyeWide is geometric (Group E); EyeSquint is solved (Group S)."),
    ("Brow", ["BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft", "BrowOuterUpRight"],
     "Group S, landmark-space solve."),
    ("Mouth, shape",
     ["MouthPucker", "MouthFunnel", "MouthClose", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft",
      "MouthFrownRight"],
     "Group S."),
    ("Mouth, fine detail",
     ["MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight", "MouthRollLower",
      "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft", "MouthPressRight",
      "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft", "MouthUpperUpRight", "MouthLeft",
      "MouthRight"],
     "Group S, finer, lower-amplitude mouth channels."),
    ("Cheek / Nose / Tongue",
     ["CheekPuff", "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft", "NoseSneerRight", "TongueOut"],
     "CheekPuff/TongueOut are structural zeros (Group W/Z); CheekSquint/NoseSneer are solved."),
    ("Head / eye rotation (degrees)",
     ["LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll", "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
      "HeadYaw", "HeadPitch", "HeadRoll"],
     "HeadYaw/Pitch/Roll are deliberately zero by this project's own design (head rotation comes from the "
     "body track). Eye Yaw/Pitch are geometric; Roll is never predicted."),
]


def _validate_groups() -> None:
    covered = [name for _, names, _ in GROUPS for name in names]
    missing = set(ALL_COMPARABLE_COLUMNS) - set(covered)
    extra = set(covered) - set(ALL_COMPARABLE_COLUMNS)
    dupes = {name for name in covered if covered.count(name) > 1}
    if missing or extra or dupes:
        raise RuntimeError(f"GROUPS is out of sync with ALL_COMPARABLE_COLUMNS: missing={missing} extra={extra} dupes={dupes}")


def _classify(corr: float | None, ours_max: float, orig_max: float) -> tuple[str, str]:
    if corr is None or np.isnan(corr):
        if ours_max < 1e-4 and orig_max > 0.05:
            return "zero", "never activates"
        if orig_max < 1e-4:
            return "zero", "ground truth flat"
        return "zero", "flat signal"
    if corr < 0:
        return "wrong", "negative correlation"
    if corr < 0.3:
        return "weak", "weak correlation"
    if corr < 0.65:
        return "moderate", "moderate correlation"
    return "strong", "strong correlation"


def _ratio_note(ratio: float | None, corr: float | None) -> str:
    if ratio is None or corr is None or np.isnan(corr):
        return ""
    if ratio < 0.5:
        return "undershoots"
    if ratio > 1.6:
        return "overshoots"
    return "close"


def compute_channel_stats(run_dir: Path) -> dict[str, dict]:
    """`{channel_name: {corr, rms, ours_max, orig_max, ratio, orig_series,
    ours_series}}` for every channel in `ALL_COMPARABLE_COLUMNS`,
    `corr`/`ratio` are `None` when undefined (a flat signal on either
    side). `*_series` are the raw aligned per-frame arrays, kept
    alongside the scalar summary so the report's own per-channel chart
    (see `_sparkline_svg`) doesn't need a second, separately-aligned data
    load that could drift from the numbers next to it."""
    ours_aligned, orig_aligned = aligned_channel_data(run_dir)
    stats = {}
    for i, name in enumerate(ALL_COMPARABLE_COLUMNS):
        orig_col, ours_col = orig_aligned[:, i], ours_aligned[:, i]
        rms = float(np.sqrt(np.mean((orig_col - ours_col) ** 2)))
        corr = None if (orig_col.std() < 1e-6 or ours_col.std() < 1e-6) else float(np.corrcoef(orig_col, ours_col)[0, 1])
        orig_max = float(np.max(np.abs(orig_col)))
        ours_max = float(np.max(np.abs(ours_col)))
        ratio = (ours_max / orig_max) if orig_max > 1e-6 else None
        stats[name] = {
            "corr": corr, "rms": rms, "ours_max": ours_max, "orig_max": orig_max, "ratio": ratio,
            "orig_series": orig_col, "ours_series": ours_col,
        }
    return stats


# Fixed logical coordinate system for every chart's own viewBox, scaled to
# fit its actual rendered width by CSS (`width: 100%`), not by these numbers,
# so the SVG stays crisp at any container size instead of baking in a pixel
# width (see this project's own artifact-design habit of never assuming a
# fixed viewport). Left as generous headroom above/below the data range
# (10% each side) so a peak never touches the chart's own top/bottom edge.
_CHART_WIDTH = 600
_CHART_HEIGHT = 130
_CHART_PAD_FRAC = 0.1


def _sparkline_svg(orig_col: np.ndarray, ours_col: np.ndarray) -> str:
    """Self-contained SVG line chart, ground truth vs. our own output, over
    the aligned frame timeline, the same "does the shape/timing actually
    match, not just the correlation number" check this session did by hand
    (printing per-frame tables and eyeballing them) for `BrowInnerUp`'s own
    plateau bug. No JS, no chart library, two `<polyline>`s plus a zero
    baseline, matching every other self-contained report/preview this
    project ships."""
    n = len(orig_col)
    combined = np.concatenate([orig_col, ours_col])
    y_min, y_max = float(combined.min()), float(combined.max())
    if y_max - y_min < 1e-6:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    pad = (y_max - y_min) * _CHART_PAD_FRAC
    y_min, y_max = y_min - pad, y_max + pad

    def to_points(col: np.ndarray) -> str:
        xs = np.linspace(0, _CHART_WIDTH, n) if n > 1 else np.array([0.0])
        ys = _CHART_HEIGHT - (col - y_min) / (y_max - y_min) * _CHART_HEIGHT
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    zero_y = _CHART_HEIGHT - (0.0 - y_min) / (y_max - y_min) * _CHART_HEIGHT
    return f'''<svg viewBox="0 0 {_CHART_WIDTH} {_CHART_HEIGHT}" class="chart" preserveAspectRatio="none">
  <line x1="0" y1="{zero_y:.1f}" x2="{_CHART_WIDTH}" y2="{zero_y:.1f}" class="chart-zero" />
  <polyline points="{to_points(orig_col)}" class="chart-orig" />
  <polyline points="{to_points(ours_col)}" class="chart-ours" />
</svg>'''


def _rows_html(stats: dict[str, dict]) -> tuple[str, dict[str, int]]:
    rows = []
    counts = {"strong": 0, "moderate": 0, "weak": 0, "wrong": 0, "zero": 0}
    for group_name, names, group_note in GROUPS:
        rows.append(f'<tr class="group-row"><td colspan="6"><span class="group-name">{group_name}</span>'
                     f'<span class="group-note">{group_note}</span></td></tr>')
        for name in names:
            d = stats[name]
            status, _ = _classify(d["corr"], d["ours_max"], d["orig_max"])
            counts[status] += 1
            corr_disp = f"{d['corr']:+.2f}" if d["corr"] is not None else "—"
            ratio_disp = f"{d['ratio']:.2f}×" if (d["ratio"] is not None and d["corr"] is not None) else "—"
            rnote = _ratio_note(d["ratio"], d["corr"])
            chart_svg = _sparkline_svg(d["orig_series"], d["ours_series"])
            rows.append(f'''<tr>
  <td class="ch-name">{name}</td>
  <td class="num"><span class="pill pill-{status}">{corr_disp}</span></td>
  <td class="num">{ratio_disp}<span class="rnote">{rnote}</span></td>
  <td class="num">{d['ours_max']:.3f}</td>
  <td class="num">{d['orig_max']:.3f}</td>
  <td class="chart-cell"><details><summary>graph</summary><div class="chart-wrap">{chart_svg}<div class="chart-legend"><span class="legend-orig">&mdash; ground truth</span><span class="legend-ours">&mdash; ours</span></div></div></details></td>
</tr>''')
    return "\n".join(rows), counts


_TEMPLATE = r'''<title>output_face.csv vs. real ARKit ground truth &mdash; __CLIP_NAME__</title>
<style>
:root {
  --bg: #F7F5F1; --surface: #FFFFFF; --surface-2: #EFEBE3; --text: #26241F; --text-muted: #6B675C;
  --border: #E2DDD1; --accent: #2F6F6B;
  --strong: #3D7A4E; --strong-bg: #E5F0E6; --moderate: #8A6F1E; --moderate-bg: #F4EDD9;
  --weak: #A85A2E; --weak-bg: #F5E6DA; --wrong: #A23B3B; --wrong-bg: #F5DFDF; --zero: #8C877A; --zero-bg: #EEEBE3;
  --chart-orig: #8C877A; --chart-ours: #2F6F6B; --chart-zero: #D8D2C4;
}
:root[data-theme="dark"] {
  --bg: #16181A; --surface: #1E2123; --surface-2: #26292B; --text: #E8E5DC; --text-muted: #9C978A;
  --border: #34383A; --accent: #6BB0AA;
  --strong: #6FBE84; --strong-bg: #1E2E22; --moderate: #D3B75B; --moderate-bg: #332C18;
  --weak: #E0925C; --weak-bg: #362619; --wrong: #E27A7A; --wrong-bg: #3A2222; --zero: #8C877A; --zero-bg: #26292B;
  --chart-orig: #9C978A; --chart-ours: #6BB0AA; --chart-zero: #34383A;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #16181A; --surface: #1E2123; --surface-2: #26292B; --text: #E8E5DC; --text-muted: #9C978A;
    --border: #34383A; --accent: #6BB0AA;
    --strong: #6FBE84; --strong-bg: #1E2E22; --moderate: #D3B75B; --moderate-bg: #332C18;
    --weak: #E0925C; --weak-bg: #362619; --wrong: #E27A7A; --wrong-bg: #3A2222; --zero: #8C877A; --zero-bg: #26292B;
    --chart-orig: #9C978A; --chart-ours: #6BB0AA; --chart-zero: #34383A;
  }
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text);
  font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  line-height: 1.5; max-width: 980px; margin: 0 auto; padding: 48px 24px 96px;
}
h1, h2, h3 {
  font-family: "Iowan Old Style", "Palatino Linotype", "URW Palladio", Georgia, serif;
  font-weight: 600; text-wrap: balance; color: var(--text);
}
h1 { font-size: 2rem; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 1.3rem; margin: 48px 0 4px; }
.subtitle { color: var(--text-muted); font-size: 1.05rem; margin: 0 0 28px; }
.meta-line { color: var(--text-muted); font-size: 0.85rem; margin: 0 0 32px; }
.meta-line code, .callout code {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  background: var(--surface-2); padding: 1px 6px; border-radius: 4px; font-size: 0.82rem;
}
.summary-bar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.summary-chip {
  display: flex; align-items: baseline; gap: 7px; background: var(--surface);
  border: 1px solid var(--border); border-radius: 10px; padding: 10px 16px; flex: 1 1 140px;
}
.summary-chip .count {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-variant-numeric: tabular-nums; font-size: 1.4rem; font-weight: 600;
}
.summary-chip .label { color: var(--text-muted); font-size: 0.82rem; }
.summary-chip.strong .count { color: var(--strong); }
.summary-chip.moderate .count { color: var(--moderate); }
.summary-chip.weak .count { color: var(--weak); }
.summary-chip.wrong .count { color: var(--wrong); }
.summary-chip.zero .count { color: var(--zero); }
.callout {
  background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--accent);
  border-radius: 8px; padding: 16px 20px; margin: 14px 0; font-size: 0.94rem;
}
.callout h3 { margin: 0 0 6px; font-size: 1.02rem; }
.callout p { margin: 6px 0; color: var(--text); }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; min-width: 640px; }
thead th {
  text-align: left; padding: 10px 14px; color: var(--text-muted); font-weight: 600; font-size: 0.72rem;
  text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border);
  position: sticky; top: 0; background: var(--surface);
}
thead th.num, td.num { text-align: right; }
tbody tr:not(.group-row) { border-bottom: 1px solid var(--border); }
tbody tr:not(.group-row):last-child { border-bottom: none; }
tbody tr:not(.group-row):hover { background: var(--surface-2); }
td { padding: 8px 14px; vertical-align: middle; }
td.ch-name { font-family: "SF Mono", "Cascadia Code", Consolas, monospace; font-size: 0.83rem; }
td.num {
  font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
  font-variant-numeric: tabular-nums; color: var(--text-muted);
}
tr.group-row td { background: var(--surface-2); padding: 7px 14px; border-bottom: 1px solid var(--border); }
.group-name {
  font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text);
}
.group-note { color: var(--text-muted); font-size: 0.8rem; margin-left: 12px; }
.pill { display: inline-block; min-width: 3.4em; padding: 2px 9px; border-radius: 999px; font-weight: 600; font-size: 0.82rem; }
.pill-strong { color: var(--strong); background: var(--strong-bg); }
.pill-moderate { color: var(--moderate); background: var(--moderate-bg); }
.pill-weak { color: var(--weak); background: var(--weak-bg); }
.pill-wrong { color: var(--wrong); background: var(--wrong-bg); }
.pill-zero { color: var(--zero); background: var(--zero-bg); }
.rnote { color: var(--text-muted); font-size: 0.72rem; margin-left: 6px; font-family: -apple-system, "Segoe UI", sans-serif; }
td.chart-cell { min-width: 90px; }
td.chart-cell summary {
  cursor: pointer; color: var(--accent); font-size: 0.78rem; font-weight: 600; list-style: none;
  font-family: -apple-system, "Segoe UI", sans-serif; user-select: none;
}
td.chart-cell summary::-webkit-details-marker { display: none; }
td.chart-cell summary::before { content: "▸ "; }
td.chart-cell details[open] summary::before { content: "▾ "; }
.chart-wrap { width: min(600px, 60vw); margin-top: 8px; }
svg.chart { width: 100%; height: auto; display: block; }
.chart-zero { stroke: var(--chart-zero); stroke-width: 1; }
.chart-orig { fill: none; stroke: var(--chart-orig); stroke-width: 2; stroke-linejoin: round; }
.chart-ours { fill: none; stroke: var(--chart-ours); stroke-width: 2; stroke-linejoin: round; }
.chart-legend {
  display: flex; gap: 14px; margin-top: 4px; font-size: 0.72rem; color: var(--text-muted);
  font-family: -apple-system, "Segoe UI", sans-serif;
}
.chart-legend .legend-orig { color: var(--chart-orig); }
.chart-legend .legend-ours { color: var(--chart-ours); }
footer { margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.82rem; }
</style>

<h1>output_face.csv vs. real ARKit ground truth</h1>
<p class="subtitle">Every channel, correlation and magnitude, against <code>__CLIP_NAME__</code>'s paired LiveLinkFace capture</p>
<p class="meta-line">Clip: <code>__CLIP_NAME__</code> &middot; __N_FRAMES__ aligned frames &middot; generated by <code>scripts/build_ground_truth_report.py</code></p>

<div class="summary-bar">
  <div class="summary-chip strong"><span class="count">__N_STRONG__</span><span class="label">strong corr.<br>&ge; 0.65</span></div>
  <div class="summary-chip moderate"><span class="count">__N_MODERATE__</span><span class="label">moderate<br>0.30&ndash;0.65</span></div>
  <div class="summary-chip weak"><span class="count">__N_WEAK__</span><span class="label">weak<br>0&ndash;0.30</span></div>
  <div class="summary-chip wrong"><span class="count">__N_WRONG__</span><span class="label">negative corr.<br>wrong shape</span></div>
  <div class="summary-chip zero"><span class="count">__N_ZERO__</span><span class="label">never activates<br>or structural 0</span></div>
</div>

<div class="callout">
  <h3>Reading this table</h3>
  <p><strong>Corr.</strong> is Pearson correlation against ground truth &mdash; shape/timing fidelity, independent of scale. <strong>Ratio</strong> is our own peak magnitude divided by ground truth's peak (1.00&times; is a perfect match; &mdash; means no real signal on one side to compare). <strong>Our / Orig. Max</strong> are the raw peak values themselves, in each channel's own [0,1] range (degrees for the bottom group). <strong>Graph</strong> expands a real line chart, our output vs. ground truth, over the whole clip &mdash; a correlation number alone can hide a channel that's stuck at a plateau instead of tracking real motion (see <code>BrowInnerUp</code>'s own history in <code>docs/ARCHITECTURE.md</code>); this is the same shape check done by hand, in the report itself.</p>
</div>

<h2>All __N_TOTAL__ channels</h2>
<div class="table-wrap">
<table>
<thead><tr><th>Channel</th><th class="num">Corr.</th><th class="num">Ratio</th><th class="num">Our Max</th><th class="num">Orig. Max</th><th>Trend</th></tr></thead>
<tbody>
__ROWS__
</tbody>
</table>
</div>

<footer>
  Regenerate after any change that could move these numbers: <code>pixi run compare-ground-truth &lt;run_dir&gt;</code>. Full narrative for individual findings lives in <code>docs/ARCHITECTURE.md</code> and the plan file's own matching sections.
</footer>
'''


def build(run_dir: Path, out_path: Path | None = None) -> Path:
    _validate_groups()
    stats = compute_channel_stats(run_dir)
    rows_html, counts = _rows_html(stats)

    html = (_TEMPLATE
            .replace("__ROWS__", rows_html)
            .replace("__CLIP_NAME__", run_dir.name)
            .replace("__N_FRAMES__", str(len(aligned_channel_data(run_dir)[0])))
            .replace("__N_STRONG__", str(counts["strong"]))
            .replace("__N_MODERATE__", str(counts["moderate"]))
            .replace("__N_WEAK__", str(counts["weak"]))
            .replace("__N_WRONG__", str(counts["wrong"]))
            .replace("__N_ZERO__", str(counts["zero"]))
            .replace("__N_TOTAL__", str(sum(counts.values()))))

    # "stage9_face" duplicated rather than imported from stage_9_capture_face.
    # FACE_DIRNAME, that module pulls in torch at load time, which this
    # otherwise-light script has no other reason to need. Written alongside
    # the other face previews (FLAME/ARKit_face_preview.blend) rather than
    # the run root, at the user's own request.
    out_path = out_path or (run_dir / "stage9_face" / "ground_truth_report.html")
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} <run_dir> [out_path]")
        sys.exit(1)
    result_path = build(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) == 3 else None)
    print(f"Wrote {result_path}")
    webbrowser.open(result_path.as_uri())
