"""
Combine multiple SVG files into a single SVG with A, B, C... labels in the top-left of each panel.

Usage:
    python combine_svgs.py fig1.svg fig2.svg fig3.svg -o combined.svg [--ncols 2] [--label-size 24]
"""
import argparse
import math
import re
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


def combine_svgs(input_paths, output_path, ncols, label_size, gap=0):
    if len(input_paths) > 26:
        raise ValueError("At most 26 figures are supported (A–Z).")

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    svgs = [sc.SVG(p) for p in input_paths]

    # Read dimensions directly from the SVG XML to avoid svgutils unit-stripping
    cell_w, cell_h = _svg_dimensions_pt(input_paths[0])

    nrows = math.ceil(len(svgs) / ncols)
    panels = []

    for i, (svg, label) in enumerate(zip(svgs, labels)):
        col = i % ncols
        row = i // ncols
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        panels.append(sc.Panel(
            svg.move(x, y),
            sc.Text(label, x + 2, y + label_size * 0.7, size=label_size, weight='bold', font='Arial'),
        ))

    fig = sc.Figure(ncols * cell_w + (ncols - 1) * gap, nrows * cell_h + (nrows - 1) * gap, *panels)
    fig.save(output_path)
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
    args = parser.parse_args()

    combine_svgs(args.inputs, args.output, args.ncols, args.label_size, args.gap)


if __name__ == "__main__":
    main()
