"""Verify and extract the submission archives for a local rehearsal."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def validate_members(archive):
    names = [item.filename for item in archive.infolist() if not item.is_dir()]
    if len(set(names)) != len(names):
        raise ValueError("Archive contains duplicate member names.")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError(f"Unsafe archive path: {name}")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("dashboard_archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    with ZipFile(args.archive) as archive:
        names = validate_members(archive)
        root, = {PurePosixPath(name).parts[0] for name in names}
        prefix = f"{root}/"
        manifest_path = f"{prefix}SHA256SUMS.txt"
        records = [line.split("  ", 1) for line in archive.read(manifest_path).decode("utf-8").splitlines()]
        expected = {name: digest for digest, name in records}
        if len(records) != len(expected) or {prefix + name for name in expected} != set(names) - {manifest_path}:
            raise ValueError("Submission manifest does not match archive membership.")
        archive.extractall(args.destination)
    extracted = args.destination / root
    for name, digest in expected.items():
        with (extracted / name).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"SHA-256 mismatch: {name}")
    with ZipFile(args.dashboard_archive) as dashboard:
        dashboard_names = validate_members(dashboard)
        if set(dashboard_names) != {name for name in expected if name.startswith("dashboard/")}:
            raise ValueError("Dashboard archive membership differs from the submission.")
        for name in dashboard_names:
            if sha256(dashboard.read(name)).hexdigest() != expected[name]:
                raise ValueError(f"Dashboard archive differs: {name}")
    result = {
        "submission_archive": args.archive.name,
        "submission_files": len(names),
        "manifest_entries_verified": len(expected),
        "dashboard_archive": args.dashboard_archive.name,
        "dashboard_files_verified": len(dashboard_names),
        "dashboard_archive_matches_submission": True,
        "extracted_project": root,
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
