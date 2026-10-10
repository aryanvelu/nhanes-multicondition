"""Download NHANES continuous-survey modules for the multi-condition project.

Pulls the raw ``.XPT`` (SAS transport) files straight from the CDC's public
NHANES site — no account or login required — for the cycles and modules used to
build the five diagnostic labels (diabetes, prediabetes, obesity, hypertension,
metabolic syndrome).

Files are laid out on disk as::

    data/raw/2013-2014/DEMO_H.XPT
    data/raw/2015-2016/DEMO_I.XPT
    data/raw/2017-2018/DEMO_J.XPT
    ...

Each downloaded file is opened with :mod:`pyreadstat` to confirm it is a valid
transport file, and its row/column counts are printed and written to a per-run
``_manifest.csv`` so the raw pull is auditable.

Usage
-----
    python -m src.data.download                 # all cycles + modules
    python -m src.data.download --force          # re-download even if present
    python -m src.data.download --cycles 2017-2018
    python -m src.data.download --list           # show plan, download nothing
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyreadstat
import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# NHANES data-file base URL. The public files live under a per-cycle
# "DataFiles" directory keyed by the *first* year of the cycle, e.g.
#   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.XPT
# File names carry a one-letter cycle suffix, so DEMO becomes DEMO_I for
# 2015-2016 and DEMO_J for 2017-2018. (The older /Nchs/Nhanes/<cycle>/ path
# now returns a 200 HTML "Page Not Found" page rather than a real 404.)
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

# cycle label -> file-name suffix used by the CDC.
CYCLES: dict[str, str] = {
    "2013-2014": "H",
    "2015-2016": "I",
    "2017-2018": "J",
}


@dataclass(frozen=True)
class Module:
    """One NHANES data module (maps to one .XPT file per cycle)."""

    base: str  # file base name, minus the cycle suffix (e.g. "DEMO")
    group: str  # which project input block this feeds
    label: str  # human-readable description


# The module set required by the ReadMe's label + feature plan. The requested
# "Cholesterol / Triglycerides" module is, in NHANES, split across three lab
# files — total cholesterol (TCHOL), HDL, and the LDL/triglycerides file
# (TRIGLY). All three are pulled because metabolic syndrome needs HDL + fasting
# triglycerides and the dyslipidemia signal needs total/LDL cholesterol.
MODULES: list[Module] = [
    Module("DEMO", "demographics", "Demographics"),
    Module("BPX", "exam", "Blood Pressure (exam)"),
    Module("BMX", "exam", "Body Measures"),
    Module("GLU", "labs", "Plasma Fasting Glucose"),
    Module("GHB", "labs", "Glycohemoglobin (HbA1c)"),
    Module("TCHOL", "labs", "Total Cholesterol"),
    Module("HDL", "labs", "HDL Cholesterol"),
    Module("TRIGLY", "labs", "LDL Cholesterol & Triglycerides"),
    Module("DIQ", "questionnaire", "Diabetes Questionnaire"),
    Module("BPQ", "questionnaire", "Blood Pressure & Cholesterol Questionnaire"),
    Module("SMQ", "questionnaire", "Smoking - Cigarette Use"),
    Module("MCQ", "questionnaire", "Medical Conditions (family history)"),
    Module("DR1TOT", "dietary", "Dietary Recall - Day 1 Total Nutrients"),
    Module("DR2TOT", "dietary", "Dietary Recall - Day 2 Total Nutrients"),
    Module("PAQ", "questionnaire", "Physical Activity Questionnaire"),
    Module("WHQ", "questionnaire", "Weight History Questionnaire"),
    Module("BIOPRO", "labs", "Standard Biochemistry Profile (kidney/liver function)"),
    Module("ALQ", "questionnaire", "Alcohol Use"),
    Module("SLQ", "questionnaire", "Sleep Disorders"),
    # RXQ_RX is structurally different from every other module: one row per
    # medication, not per person (SEQN is not unique). merge.py excludes it
    # from the generic per-module left-join and aggregates it separately —
    # see MEDICATION_MODULE / aggregate_medications() there.
    Module("RXQ_RX", "questionnaire", "Prescription Medications (aggregated separately)"),
    Module("HSQ", "questionnaire", "General Health Status"),
    Module("CBC", "labs", "Complete Blood Count (closest available substitute for CRP, which is unavailable for these cycles)"),
]

# CDC's server rejects requests without a normal browser-style User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}

RAW_ROOT = Path("data/raw")
REQUEST_TIMEOUT = 60  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 3  # seconds, multiplied by attempt number


# --------------------------------------------------------------------------- #
# Result bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class DownloadResult:
    cycle: str
    module: str
    filename: str
    url: str
    status: str  # "downloaded", "cached", or "failed"
    rows: int | None = None
    cols: int | None = None
    bytes: int | None = None
    error: str = ""


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #


def file_url(cycle: str, module: Module) -> tuple[str, str]:
    """Return (filename, full_url) for a module in a given cycle."""
    suffix = CYCLES[cycle]
    start_year = cycle.split("-")[0]
    filename = f"{module.base}_{suffix}.XPT"
    return filename, f"{BASE_URL}/{start_year}/DataFiles/{filename}"


def download_file(url: str, dest: Path) -> int:
    """Stream ``url`` to ``dest`` atomically. Returns bytes written."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(
                url, headers=HEADERS, stream=True, timeout=REQUEST_TIMEOUT
            ) as resp:
                resp.raise_for_status()
                # The CDC serves a 200 HTML "Page Not Found" page for bad
                # paths, so a clean HTTP status is not enough — reject anything
                # that isn't the expected SAS-transport payload.
                ctype = resp.headers.get("Content-Type", "")
                if "html" in ctype.lower():
                    raise IOError(f"expected .XPT but server returned {ctype!r}")
                written = 0
                first = True
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if not chunk:
                            continue
                        if first:
                            head = chunk.lstrip()[:14]
                            if head[:1] == b"<" or not head.startswith(b"HEADER RECORD"):
                                raise IOError("payload is not a SAS transport file")
                            first = False
                        fh.write(chunk)
                        written += len(chunk)
            if written == 0:
                raise IOError("empty response body")
            tmp.replace(dest)
            return written
        except (requests.RequestException, IOError) as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"      retry {attempt}/{MAX_RETRIES - 1} in {wait}s ({exc})")
                time.sleep(wait)

    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_error}")


def verify_xport(path: Path) -> tuple[int, int]:
    """Open an .XPT file with pyreadstat and return (rows, cols).

    Free-text fields (e.g. RXQ_RX's drug names) occasionally contain
    non-UTF-8 bytes (accented characters) that make pyreadstat's default
    UTF-8 decode raise; every other module here is numeric-coded and never
    hits this, but the fallback costs nothing when it's not needed.
    """
    try:
        df, _meta = pyreadstat.read_xport(str(path))
    except UnicodeDecodeError:
        df, _meta = pyreadstat.read_xport(str(path), encoding="latin1")
    return df.shape[0], df.shape[1]


def process_module(cycle: str, module: Module, force: bool) -> DownloadResult:
    filename, url = file_url(cycle, module)
    dest = RAW_ROOT / cycle / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    result = DownloadResult(cycle, module.base, filename, url, status="failed")

    try:
        if dest.exists() and not force:
            result.status = "cached"
            result.bytes = dest.stat().st_size
            print(f"    [cached]  {filename:<14} {module.label}")
        else:
            print(f"    [get]     {filename:<14} {module.label}")
            result.bytes = download_file(url, dest)
            result.status = "downloaded"

        result.rows, result.cols = verify_xport(dest)
        size_mb = (result.bytes or 0) / 1e6
        print(
            f"      -> loaded OK: {result.rows:>6,} rows x "
            f"{result.cols:>3} cols  ({size_mb:.2f} MB)"
        )
    except Exception as exc:  # noqa: BLE001 - report and continue to next file
        result.status = "failed"
        result.error = str(exc)
        print(f"      !! ERROR: {exc}", file=sys.stderr)

    return result


def write_manifest(results: list[DownloadResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["cycle", "module", "filename", "status", "rows", "cols", "bytes", "url", "error"]
        )
        for r in results:
            writer.writerow(
                [r.cycle, r.module, r.filename, r.status, r.rows, r.cols, r.bytes, r.url, r.error]
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cycles",
        nargs="+",
        choices=sorted(CYCLES),
        default=sorted(CYCLES),
        help="Which NHANES cycles to pull (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist on disk.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the download plan and exit without downloading.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"NHANES download plan: {len(args.cycles)} cycle(s) x {len(MODULES)} modules")
    for cycle in args.cycles:
        for module in MODULES:
            fname, url = file_url(cycle, module)
            print(f"  {cycle}  {fname:<14} {url}")

    if args.list:
        return 0

    results: list[DownloadResult] = []
    for cycle in args.cycles:
        print(f"\n=== Cycle {cycle} ===")
        for module in MODULES:
            results.append(process_module(cycle, module, args.force))

    # Summary
    ok = [r for r in results if r.status in ("downloaded", "cached")]
    failed = [r for r in results if r.status == "failed"]

    print("\n=== Summary ===")
    print(f"  files OK : {len(ok)}/{len(results)}")
    for r in ok:
        print(f"    {r.cycle}/{r.filename:<14} {r.rows:>6,} x {r.cols:>3}")
    if failed:
        print(f"  FAILED   : {len(failed)}")
        for r in failed:
            print(f"    {r.cycle}/{r.filename:<14} {r.error}")

    manifest_path = RAW_ROOT / "_manifest.csv"
    write_manifest(results, manifest_path)
    print(f"\n  manifest written to {manifest_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
