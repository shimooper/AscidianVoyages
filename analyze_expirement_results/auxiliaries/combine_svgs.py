"""
Combine multiple SVG files into a single SVG with A, B, C... labels in the top-left of each panel.

Usage:
    python combine_svgs.py fig1.svg fig2.svg fig3.svg -o combined.svg [--ncols 2] [--label-size 24]
"""
import argparse
import math
import svgutils.compose as sc


def combine_svgs(input_paths, output_path, ncols, label_size):
    if len(input_paths) > 26:
        raise ValueError("At most 26 figures are supported (A–Z).")

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    svgs = [sc.SVG(p) for p in input_paths]

    # Use the first figure's dimensions as the cell size
    cell_w = float(svgs[0].width)
    cell_h = float(svgs[0].height)

    nrows = math.ceil(len(svgs) / ncols)
    panels = []

    for i, (svg, label) in enumerate(zip(svgs, labels)):
        col = i % ncols
        row = i // ncols
        x = col * cell_w
        y = row * cell_h
        panels.append(sc.Panel(
            svg.move(x, y),
            sc.Text(label, x + 2, y + label_size * 0.7, size=label_size, font='Arial'),
        ))

    fig = sc.Figure(ncols * cell_w, nrows * cell_h, *panels)
    fig.save(output_path)
    print(f"Saved combined SVG to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine SVG files into one with A/B/C labels.")
    parser.add_argument("inputs", nargs="+", help="Paths to input SVG files (in order)")
    parser.add_argument("-o", "--output", required=True, help="Path to output SVG file")
    parser.add_argument("--ncols", type=int, default=2, help="Number of columns in the grid (default: 2)")
    parser.add_argument("--label-size", type=int, default=28, help="Font size for panel labels (default: 28)")
    args = parser.parse_args()

    combine_svgs(args.inputs, args.output, args.ncols, args.label_size)


if __name__ == "__main__":
    main()
