#!/usr/bin/env python3
"""Check that the server package preserves every upstream runtime entry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=("windows", "linux"))
    args = parser.parse_args()
    runtime = args.runtime_root.resolve(strict=True)
    package = args.package_root.resolve(strict=True)
    if runtime == package:
        parser.error("The runtime and server package must be different directories")
    oga = runtime / ("bin/onnxruntime-genai.dll" if args.platform == "windows" else "lib/libonnxruntime-genai.so")
    if not oga.is_file():
        parser.error(f"Upstream OGA runtime missing: {oga}")

    failures = []
    entries = 0
    for directory, directories, files in os.walk(runtime, followlinks=False):
        for name in directories + files:
            source = Path(directory) / name
            relative = source.relative_to(runtime)
            destination = package / relative
            entries += 1
            if source.is_symlink():
                if not destination.is_symlink() or os.readlink(source) != os.readlink(destination):
                    failures.append(f"Symlink differs: {relative}")
            elif source.is_dir():
                if not destination.is_dir() or destination.is_symlink():
                    failures.append(f"Directory missing: {relative}")
            elif not destination.is_file() or destination.is_symlink():
                failures.append(f"File missing: {relative}")
            elif digest(source) != digest(destination):
                failures.append(f"File content differs: {relative}")
            elif args.platform == "linux":
                executable = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                if source.stat().st_mode & executable != destination.stat().st_mode & executable:
                    failures.append(f"Executable permissions differ: {relative}")

    server = package / "bin" / ("amdgpu-server.exe" if args.platform == "windows" else "amdgpu-server")
    for required in (server, package / "LICENSE", package / "AMD_LICENSE", package / "runtime-versions.json"):
        if not required.is_file():
            failures.append(f"Server package file missing: {required.relative_to(package)}")
    metadata_path = package / "runtime-versions.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {"hip_ep": "0.5.3", "onnxruntime": "1.31.0", "onnxruntime_genai": "0.19.0"}
        for key, version in expected.items():
            if metadata.get(key) != version:
                failures.append(f"Unexpected {key}: {metadata.get(key)!r}, expected {version}")
        headers = metadata.get("oga_headers", {})
        for key in ("ort_genai_h_sha256", "ort_genai_c_h_sha256"):
            value = headers.get(key, "")
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                failures.append(f"Missing or invalid OGA header SHA256: {key}")
        if not headers.get("source") or not headers.get("version") or not headers.get("runtime_compatibility"):
            failures.append("Missing OGA header provenance or compatibility status")
    if failures:
        parser.exit(1, "\n".join(failures) + "\n")
    print(f"Verified {entries} upstream entries and server package metadata in {package}")


if __name__ == "__main__":
    main()
