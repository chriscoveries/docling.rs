#!/usr/bin/env python3
"""Bounded Linux CI smoke for opt-in ORT traces, not a performance benchmark.

Uses the already-built CLI and stock conformance models/real public PDF. No
downloads, builds, credentials, generated documents, or runtime substitutions.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time


FIXTURE = "tests/data/pdf/sources/2305.03393v1-pg9.pdf"
FIXTURE_BLOB = "d667217c2009127410fc15b8e9208381e555d5f0"
EXPECTED_SESSIONS = {"layout_heron": 1, "ocr_rec": 8, "ocr_det": 1}
TRACE_NAME = re.compile(r"(.+)-s([1-9][0-9]*)_.+\.json\Z")
GIB = 1024 ** 3


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def trace_inventory(directory):
    return {str(p.relative_to(directory)): sha256(p)
            for p in sorted(directory.rglob("*.json"))}


def validate_traces(directory, pid):
    runs = list(directory.glob(f"run-{pid}-*"))
    require(len(runs) == 1 and runs[0].is_dir(),
            f"PID {pid}: expected one reserved run directory, got {runs}")
    files = sorted(runs[0].iterdir())
    require(len(files) == sum(EXPECTED_SESSIONS.values()),
            f"PID {pid}: expected 10 closed-session traces, got {len(files)}")
    stems, sequences, rows = Counter(), set(), []
    for path in files:
        match = TRACE_NAME.fullmatch(path.name)
        require(path.is_file() and match, f"unexpected trace entry: {path}")
        require(path.stat().st_size <= 64 * 1024 ** 2, f"trace too large: {path}")
        stem, sequence = match.group(1), int(match.group(2))
        require(sequence not in sequences, f"reused session sequence: {path}")
        sequences.add(sequence)
        stems[stem] += 1
        events = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(events, list) and events
                and all(isinstance(e, dict) for e in events),
                f"invalid ORT Chrome-trace event array: {path}")
        require(any(e.get("name") == "session_initialization"
                    and e.get("cat") == "Session" and e.get("ph") == "X"
                    and isinstance(e.get("dur"), (int, float)) and e["dur"] >= 0
                    and isinstance(e.get("ts"), (int, float)) and e["ts"] >= 0
                    for e in events),
                f"no completed session initialization: {path}")
        model_runs = sum(e.get("name") == "model_run" for e in events)
        # An unused OCR lane is still a real session and may have zero runs.
        rows.append({"path": str(path.relative_to(directory)), "model": stem,
                     "sequence": sequence, "model_run_events": model_runs,
                     "bytes": path.stat().st_size, "sha256": sha256(path)})
    require(stems == EXPECTED_SESSIONS, f"PID {pid}: session inventory {dict(stems)}")
    require(sequences == set(range(1, len(files) + 1)),
            f"PID {pid}: missing reserved session sequence")
    require(sum(r["model_run_events"] for r in rows) > 0, "no inference events")
    require(sum(r["model_run_events"] for r in rows if r["model"] == "ocr_rec") > 0,
            "the real nonblank fixture did not exercise any recognizer session")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/docling-rs"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(sys.platform == "linux", "this bounded harness targets Linux CI only")
    root = Path.cwd().resolve()
    binary = args.binary.resolve(strict=True)
    fixture = (root / FIXTURE).resolve(strict=True)
    raw = fixture.read_bytes()
    require(hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            == FIXTURE_BLOB, "real upstream fixture does not match its pinned Git blob")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    profiles = out / "profiles"
    profiles.mkdir()
    affinity = sorted(os.sched_getaffinity(0))[:2]
    require(affinity, "no CPU affinity available")
    settings = {
        "DOCLING_RS_EP": "cpu", "DOCLING_RS_PDF_THREADS": "1",
        "DOCLING_RS_PDF_WORKERS": "1", "DOCLING_RS_TF_INTRA": "1",
        "DOCLING_RS_OCR_SESSIONS": "8", "DOCLING_RS_NO_ARENA": "1",
        "DOCLING_RS_NO_GRAPH_CACHE": "1", "DOCLING_RS_FP32": "1",
        "DOCLING_RS_SLOW_RESIZE": "1", "DOCLING_RS_RENDERER": "docling-parse",
        "DOCLING_PARSE_RENDER_LIB": str(root / ".docling-parse/lib"),
        "DOCLING_LAYOUT_ONNX": str(root / ".models/layout_heron.onnx"),
        "DOCLING_OCR_REC_ONNX": str(root / ".models/ocr_rec.onnx"),
        "DOCLING_OCR_DICT": str(root / ".models/ppocr_keys_v1.txt"),
        "DOCLING_OCR_DET_ONNX": str(root / ".models/ocr_det.onnx"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "2",
    }
    # Do not inherit a user's alternate model/provider/runtime configuration.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("DOCLING_", "ORT_"))}
    env.update(settings)
    assets = [Path(settings[k]) for k in (
        "DOCLING_LAYOUT_ONNX", "DOCLING_OCR_REC_ONNX", "DOCLING_OCR_DICT",
        "DOCLING_OCR_DET_ONNX")]
    assets += sorted((root / ".docling-parse/lib").glob("libdparse_render.*"))
    require(len(assets) > 4, "stock docling-parse renderer is missing")
    assets += [Path(str(p) + ".data") for p in list(assets)
               if Path(str(p) + ".data").is_file()]
    pins = {str(p.relative_to(root)): sha256(p) for p in assets}
    binary_hash = sha256(binary)
    command = [str(binary), str(fixture), "--to", "md", "--to", "json",
               "--jobs", "1", "--pages", "1", "--ocr-mode", "full_page",
               "--ocr-engine", "ppocr", "--ocr-lang", "ch",
               "--no-table-former", "--no-progress", "--images", "placeholder"]
    records = []
    receipt = {"schema": "docling-ort-profile-ci-smoke-v1", "status": "running",
               "github_checkout_sha": os.environ.get("GITHUB_SHA", "unreported"),
               "cargo_lock_sha256": sha256(root / "Cargo.lock"),
               "stock_downloader_sha256": sha256(root / "scripts/install/download_dependencies.sh"),
               "binary_sha256": binary_hash, "fixture": FIXTURE,
               "fixture_git_blob": FIXTURE_BLOB, "fixture_sha256": sha256(fixture),
               "assets_sha256": pins, "environment": settings,
               "conversion_argv": command[1:], "runs": records,
               "limits": {"cpu_affinity": affinity, "per_process_address_space_bytes": 4 * GIB,
                          "per_process_cpu_seconds": 240, "group_wall_seconds": 360,
                          "max_concurrent_processes": 2, "per_file_bytes": 64 * 1024 ** 2},
               "scope": "same static CI binary/runtime, FP32 scalar resize, forced OCR, no tables; not dynamic production 1.31 parity or performance evidence"}

    def limits():
        os.sched_setaffinity(0, affinity)
        resource.setrlimit(resource.RLIMIT_AS, (4 * GIB, 4 * GIB))
        resource.setrlimit(resource.RLIMIT_CPU, (240, 240))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 ** 2, 64 * 1024 ** 2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def run_group(names, profiling):
        jobs = []
        started = time.monotonic()
        try:
            for name in names:
                folder = out / name
                folder.mkdir()
                child_env = env.copy()
                if profiling:
                    child_env["DOCLING_RS_ORT_PROFILE"] = str(profiles)
                with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
                    proc = subprocess.Popen(command + ["--output", str(folder / "output")],
                                            cwd=root, env=child_env, stdout=stdout, stderr=stderr,
                                            start_new_session=True, preexec_fn=limits)
                row = {"name": name, "pid": proc.pid, "profiling": profiling,
                       "started_after_group_seconds": time.monotonic() - started}
                jobs.append((proc, folder, row))
                records.append(row)
            pending = list(jobs)
            while pending:
                require(time.monotonic() - started < 360, "bounded conversion group timed out")
                for proc, folder, row in pending[:]:
                    code = proc.poll()
                    if code is None:
                        continue
                    row["exit_code"] = code
                    row["completed_after_group_seconds"] = time.monotonic() - started
                    require(code == 0, f"{row['name']} failed ({code}); see {folder}/stderr.log")
                    row["outputs_sha256"] = {p.name: sha256(p) for p in sorted((folder / "output").iterdir()) if p.is_file()}
                    require(set(row["outputs_sha256"]) == {fixture.stem + ".md", fixture.stem + ".json"},
                            f"{row['name']}: missing/unexpected native output")
                    require((folder / "output" / (fixture.stem + ".md")).read_text().strip(),
                            f"{row['name']}: real nonblank fixture yielded empty Markdown")
                    require(isinstance(json.loads((folder / "output" / (fixture.stem + ".json")).read_text()), dict),
                            f"{row['name']}: native JSON must be an object")
                    if profiling:
                        row["traces"] = validate_traces(profiles, proc.pid)
                    pending.remove((proc, folder, row))
                if pending:
                    time.sleep(0.05)
            if len(jobs) == 2:
                require(max(row["started_after_group_seconds"] for _, _, row in jobs)
                        < min(row["completed_after_group_seconds"] for _, _, row in jobs),
                        "the two independent processes did not overlap")
        finally:
            # Normal success never signals the CLI: session Drop flushes traces.
            for proc, _, _ in jobs:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()

    try:
        version = subprocess.run([str(binary), "--version"], cwd=root, env=env,
                                 text=True, capture_output=True, check=True, timeout=30,
                                 preexec_fn=limits).stdout.strip()
        receipt["cli_version_output"] = version
        match = re.search(r"ONNX Runtime ([0-9]+\.[0-9]+)", version)
        require(match, "CLI did not report its actual runtime major.minor")
        receipt["onnx_runtime_reported_major_minor"] = match.group(1)
        receipt["onnx_runtime_patch_version"] = "not exposed by CLI; not assumed"
        run_group(["off"], False)
        require(not list(profiles.iterdir()), "profiling-off created trace entries")
        run_group(["on"], True)
        retained = trace_inventory(profiles)
        run_group(["parallel-a", "parallel-b"], True)
        final = trace_inventory(profiles)
        require(all(final.get(p) == digest for p, digest in retained.items()),
                "a later process overwrote the earlier completed run")
        for row in records[1:]:
            require(row["outputs_sha256"] == records[0]["outputs_sha256"],
                    f"{row['name']}: native Markdown/JSON differs from same-binary profiling-off")
            require(all(final[t["path"]] == t["sha256"] for t in row["traces"]),
                    f"{row['name']}: trace changed after normal process completion")
        require(len(final) == 30, "expected three independent ten-session trace sets")
        require(sha256(binary) == binary_hash and
                all(sha256(root / p) == digest for p, digest in pins.items()),
                "binary/model/renderer assets changed during comparison")
        receipt["status"] = "passed"
        print("ORT profiling smoke: off/on native outputs identical; 3 x 10 valid distinct traces")
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        raise
    finally:
        (out / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
