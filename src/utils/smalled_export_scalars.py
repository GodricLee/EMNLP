import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def _parse_step(val: str):
    s = (val or "").strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            f = float(s)
            return int(f) if f.is_integer() else None
        except ValueError:
            return None

def filter_csv(input_path: Path, output_path: Path | None, interval: int, inplace: bool, encoding: str = "utf-8"):
    if interval <= 0:
        raise ValueError("interval must be a positive integer")
    input_path = Path(input_path)

    temp_path = None
    if inplace:
        fd, tmp_name = tempfile.mkstemp(prefix=input_path.stem + "_tmp_", suffix=input_path.suffix, dir=str(input_path.parent))
        os.close(fd)
        temp_path = Path(tmp_name)
        out_path = temp_path
    else:
        if output_path is None:
            out_path = input_path.with_name(input_path.stem + f"_every{interval}" + input_path.suffix)
        else:
            out_path = Path(output_path)

    in_rows = 0
    out_rows = 0

    with input_path.open("r", newline="", encoding=encoding) as fin, out_path.open("w", newline="", encoding=encoding) as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames or "step" not in reader.fieldnames:
            raise RuntimeError("CSV has no step")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            in_rows += 1
            step = _parse_step(row.get("step"))
            if step is None:
                continue
            if step % interval == 0:
                writer.writerow(row)
                out_rows += 1

    if inplace and temp_path is not None:
        os.replace(temp_path, input_path)
        out_path = input_path

    return in_rows, out_rows, out_path

def main():
    parser = argparse.ArgumentParser(description="Keep rows where step % N == 0 (default every 20 steps)")
    parser.add_argument("input", nargs="?", default=None, help="Input CSV path")
    parser.add_argument("-n", "--interval", type=int, default=20, help="Interval N (keep rows where step %% N == 0)")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path (auto-generated if omitted); mutually exclusive with --inplace")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file in-place")
    parser.add_argument("--encoding", default="utf-8", help="File encoding, default utf-8")
    args = parser.parse_args()

    if args.inplace and args.output is not None:
        print("Cannot use --inplace and --output together", file=sys.stderr)
        sys.exit(2)

    try:
        in_rows, out_rows, out_path = filter_csv(args.input, args.output, args.interval, args.inplace, args.encoding)
        print(f"Done: input {in_rows} rows, output {out_rows} rows -> {out_path}")
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()