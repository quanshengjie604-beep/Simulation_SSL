#!/usr/bin/env python3
"""Small dependency-free parallel downloader for large Zenodo files.

It uses HTTP Range requests and keeps per-chunk .part files so interrupted
downloads can be resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("-j", "--jobs", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=128)
    parser.add_argument("--md5", default="")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--adopt-existing-parts",
        action="store_true",
        help="Adopt an existing .parts directory without a manifest. Only use when the same --chunk-mb was used.",
    )
    return parser.parse_args()


def request_headers(url: str, timeout: float) -> tuple[int, bool]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "codex-range-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        length = int(response.headers["Content-Length"])
        accept_ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    return length, accept_ranges


def download_range(
    url: str,
    target: Path,
    start: int,
    stop: int,
    timeout: float,
    retries: int = 1000,
) -> None:
    expected = stop - start
    existing = target.stat().st_size if target.exists() else 0
    if existing == expected:
        return
    if existing > expected:
        target.unlink()
        existing = 0

    for attempt in range(1, retries + 1):
        try:
            offset = start + existing
            req = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={offset}-{stop - 1}",
                    "User-Agent": "codex-range-downloader/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status != 206:
                    raise RuntimeError(f"server ignored Range request: HTTP {response.status}")
                with target.open("ab") as fh:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        fh.write(block)
            existing = target.stat().st_size
            if existing == expected:
                return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            wait = min(60.0, 2.0 + attempt * 0.5)
            print(f"[retry] {target.name}: {exc}; sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"failed to download {target} after {retries} retries")


def progress_thread(parts_dir: Path, total: int, done: threading.Event) -> None:
    last_bytes = 0
    last_time = time.time()
    while not done.wait(5.0):
        current = sum(p.stat().st_size for p in parts_dir.glob("*.part") if p.exists())
        now = time.time()
        speed = (current - last_bytes) / max(1e-6, now - last_time)
        pct = 100.0 * current / total
        print(
            f"[progress] {current / 1e9:.2f}/{total / 1e9:.2f} GB "
            f"({pct:.2f}%), {speed / 1e6:.2f} MB/s",
            flush=True,
        )
        last_bytes = current
        last_time = now


def merge_parts(output: Path, parts: list[Path]) -> None:
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as out:
        for part in parts:
            with part.open("rb") as src:
                while True:
                    block = src.read(16 * 1024 * 1024)
                    if not block:
                        break
                    out.write(block)
    tmp.replace(output)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            block = fh.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def check_or_write_manifest(parts_dir: Path, url: str, total: int, chunk: int, adopt_existing: bool) -> None:
    manifest_path = parts_dir / "manifest.json"
    manifest = {
        "url": url,
        "total_bytes": total,
        "chunk_bytes": chunk,
        "version": 1,
    }
    existing_parts = list(parts_dir.glob("*.part"))
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="ascii"))
        mismatches = [
            key
            for key in ("url", "total_bytes", "chunk_bytes")
            if old.get(key) != manifest[key]
        ]
        if mismatches:
            raise RuntimeError(
                "existing .parts manifest does not match this download "
                f"({', '.join(mismatches)} differ). Use the original options, "
                "or remove the .parts directory before changing --chunk-mb/URL."
            )
        return

    if existing_parts and not adopt_existing:
        raise RuntimeError(
            f"{parts_dir} already contains .part files but has no manifest. "
            "Re-run with the original --chunk-mb and --adopt-existing-parts if "
            "you are sure they belong to this exact download; otherwise remove "
            "the .parts directory and restart."
        )

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.chunk_mb < 1:
        raise ValueError("--chunk-mb must be >= 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total, accept_ranges = request_headers(args.url, args.timeout)
    if not accept_ranges:
        print("[warn] server did not advertise Accept-Ranges: bytes; trying anyway", flush=True)

    if args.output.exists() and args.output.stat().st_size == total:
        print(f"[done] output already complete: {args.output}", flush=True)
        if args.md5:
            got = md5sum(args.output)
            print(f"[md5] {got}", flush=True)
            return 0 if got.lower() == args.md5.lower() else 2
        return 0

    chunk = args.chunk_mb * 1024 * 1024
    ranges = [(start, min(start + chunk, total)) for start in range(0, total, chunk)]
    parts_dir = args.output.with_name(args.output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    check_or_write_manifest(parts_dir, args.url, total, chunk, args.adopt_existing_parts)
    parts = [parts_dir / f"{idx:06d}.part" for idx in range(len(ranges))]

    tasks: queue.Queue[int] = queue.Queue()
    for idx, (start, stop) in enumerate(ranges):
        if parts[idx].exists() and parts[idx].stat().st_size == stop - start:
            continue
        tasks.put(idx)

    done = threading.Event()
    reporter = threading.Thread(target=progress_thread, args=(parts_dir, total, done), daemon=True)
    reporter.start()

    errors: list[BaseException] = []

    def worker() -> None:
        while True:
            try:
                idx = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                start, stop = ranges[idx]
                download_range(args.url, parts[idx], start, stop, args.timeout)
            except BaseException as exc:  # noqa: BLE001 - surfaced after joining.
                errors.append(exc)
            finally:
                tasks.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    done.set()

    if errors:
        raise errors[0]

    print("[merge] combining parts", flush=True)
    merge_parts(args.output, parts)
    print(f"[done] wrote {args.output} ({args.output.stat().st_size / 1e9:.2f} GB)", flush=True)

    if args.md5:
        got = md5sum(args.output)
        print(f"[md5] {got}", flush=True)
        if got.lower() != args.md5.lower():
            print(f"[error] md5 mismatch, expected {args.md5}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
