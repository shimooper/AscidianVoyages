"""
Compare two CSV files for identical content regardless of row order.

Usage:
    python compare_csvs.py file1.csv file2.csv
"""
import argparse
import sys
import pandas as pd


def compare_csvs(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    if list(df1.columns) != list(df2.columns):
        print("DIFFERENT: columns do not match.")
        print(f"  {path1}: {list(df1.columns)}")
        print(f"  {path2}: {list(df2.columns)}")
        return False

    if len(df1) != len(df2):
        print(f"DIFFERENT: row counts differ ({len(df1)} vs {len(df2)}).")
        return False

    cols = list(df1.columns)
    sorted1 = df1.sort_values(by=cols).reset_index(drop=True)
    sorted2 = df2.sort_values(by=cols).reset_index(drop=True)

    diff = sorted1.compare(sorted2)
    if diff.empty:
        print("IDENTICAL: both files contain the same rows.")
        return True

    print(f"DIFFERENT: {len(diff)} row(s) differ after sorting.")
    print(diff.to_string())
    return False


def main():
    parser = argparse.ArgumentParser(description="Compare two CSV files regardless of row order.")
    parser.add_argument("file1", help="Path to first CSV file")
    parser.add_argument("file2", help="Path to second CSV file")
    args = parser.parse_args()

    identical = compare_csvs(args.file1, args.file2)
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
