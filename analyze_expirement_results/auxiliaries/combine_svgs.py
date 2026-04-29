"""
Combine multiple SVG files into a single SVG with A, B, C... labels in the top-left of each panel.

Usage:
    python combine_svgs.py fig1.svg fig2.svg fig3.svg -o combined.svg [--ncols 2] [--label-size 24]
"""
import argparse
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import svgutils.compose as sc
import cairosvg

# Points-per-unit conversion to a common internal unit (pt)
_UNIT_TO_PT = {'pt': 1.0, 'px': 0.75, 'in': 72.0, 'cm': 28.3465, 'mm': 2.83465, '': 1.0}


def _svg_dimensions_pt(path):
    """Return (width, height) in pt by reading the viewBox or width/height attributes."""
    root = ET.parse(path).getroot()
    # Strip namespace prefix if present
    tag_ns = re.match(r'\{[^}]+\}', root.tag)
    ns_prefix = tag_ns.group(0) if tag_ns else ''

    vb = root.get('viewBox') or root.get(f'{ns_prefix}viewBox')
    if vb:
        parts = vb.split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])

    def parse(attr):
        s = (root.get(attr) or '').strip()
        m = re.match(r'([\d.]+)\s*(pt|px|in|cm|mm)?', s)
        if m:
            return float(m.group(1)) * _UNIT_TO_PT.get(m.group(2) or '', 1.0)
        return None

    w, h = parse('width'), parse('height')
    if w and h:
        return w, h
    raise ValueError(f"Cannot determine dimensions of {path}")


def _make_ascii_safe_tmp(path):
    """Write a temp SVG where every non-ASCII character is a decimal XML entity.

    svgutils/lxml on Windows reads files with the system encoding (CP1252) even
    when the XML declaration says UTF-8.  This corrupts characters like Δ and ≤
    at read time.  By escaping them to ASCII-safe entities first, lxml receives
    only ASCII bytes and round-trips them correctly to the combined output.

    The caller must delete the returned path when done.
    """
    content = Path(path).read_text(encoding='utf-8')
    # Remove the XML declaration and DOCTYPE — svgutils generates its own
    content = re.sub(r'<\?xml\b[^?]*\?>\s*', '', content)
    content = re.sub(r'<!DOCTYPE\b[^\[>]*(?:\[[^\]]*\])?\s*>\s*', '', content, flags=re.DOTALL)
    safe = ''.join(f'&#{ord(c)};' if ord(c) > 127 else c for c in content)
    fd, tmp_path = tempfile.mkstemp(suffix='.svg')
    with os.fdopen(fd, 'w', encoding='ascii') as f:
        f.write(safe)
    return tmp_path


def combine_svgs(input_paths, output_path, ncols, label_size, gap=0, span_last_row=False):
    if len(input_paths) > 26:
        raise ValueError("At most 26 figures are supported (A–Z).")

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = len(input_paths)
    nrows = math.ceil(n / ncols)

    # Read each figure's dimensions individually
    dims = [_svg_dimensions_pt(p) for p in input_paths]

    last_row_start = (nrows - 1) * ncols
    last_row_count = n - last_row_start
    last_row_spans = span_last_row and last_row_count < ncols

    # Max width per column and max height per row for alignment
    # Figures in a spanning last row don't contribute to per-column widths
    col_widths = [0.0] * ncols
    row_heights = [0.0] * nrows
    for i, (w, h) in enumerate(dims):
        row = i // ncols
        row_heights[row] = max(row_heights[row], h)
        if not (last_row_spans and i >= last_row_start):
            col_widths[i % ncols] = max(col_widths[i % ncols], w)

    col_x = [sum(col_widths[:c]) + c * gap for c in range(ncols)]
    row_y = [sum(row_heights[:r]) + r * gap for r in range(nrows)]
    total_w = sum(col_widths) + (ncols - 1) * gap
    total_h = sum(row_heights) + (nrows - 1) * gap

    tmp_paths = []
    try:
        tmp_paths = [_make_ascii_safe_tmp(p) for p in input_paths]
        svgs = [sc.SVG(p) for p in tmp_paths]

        panels = []
        for i, (svg, label) in enumerate(zip(svgs, labels)):
            col = i % ncols
            row = i // ncols
            w, h = dims[i]
            y = row_y[row]

            if last_row_spans and i >= last_row_start:
                # Distribute spanning figures evenly across the full canvas width
                slot_w = total_w / last_row_count
                slot_idx = i - last_row_start
                x_fig = slot_idx * slot_w + (slot_w - w) / 2
                x_label = slot_idx * slot_w
            else:
                x_fig = col_x[col] + (col_widths[col] - w) / 2
                x_label = col_x[col]

            panels.append(sc.Panel(
                svg.move(x_fig, y),
                sc.Text(label, x_label + 2, y + label_size * 0.7, size=label_size, weight='bold', font='Arial'),
            ))

        fig = sc.Figure(total_w, total_h, *panels)
        fig.save(output_path)
    finally:
        for tmp in tmp_paths:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    print(f"Saved combined SVG to {output_path}")

    png_path = output_path.with_suffix(".png")
    cairosvg.svg2png(url=str(output_path.resolve()), write_to=str(png_path), scale=600/96, background_color="white")
    print(f"Saved combined PNG to {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine SVG files into one with A/B/C labels.")
    parser.add_argument("inputs", nargs="+", help="Paths to input SVG files (in order)")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to output SVG file")
    parser.add_argument("--ncols", type=int, default=2, help="Number of columns in the grid (default: 2)")
    parser.add_argument("--label-size", type=int, default=30, help="Font size for panel labels (default: 28)")
    parser.add_argument("--gap", type=int, default=30, help="Gap in pixels between panels; use negative values to reduce whitespace (default: 0)")
    parser.add_argument("--span-last-row", action="store_true", help="Center incomplete last-row figures across the full canvas width")
    args = parser.parse_args()

    combine_svgs(args.inputs, args.output, args.ncols, args.label_size, args.gap, args.span_last_row)


if __name__ == "__main__":
    main()
