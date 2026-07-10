#!/usr/bin/env python3
"""Run downstream benchmark suites across CLIP/SigLIP model matrices."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = REPO_ROOT / "rebuttal" / "downstream_data"
DEFAULT_REFER = REPO_ROOT / "rebuttal" / "downstream_repos" / "refer" / "data"
DEFAULT_OUT = SCRIPT_DIR / "results" / "downstream"


@dataclass(frozen=True)
class ModelSpec:
    suite: str
    role: str
    name: str
    model_id: str
    processor_id: str


MODEL_SPECS = [
    ModelSpec("clip_l14_224", "baseline", "clip_l14_224_base", "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14"),
    ModelSpec("clip_l14_224", "rea", "clip_l14_224_rea", "fesvhtr/clip-r-rea-run1219-621", "openai/clip-vit-large-patch14"),
    ModelSpec("clip_l14_224", "des", "clip_l14_224_des", "fesvhtr/clip-r-des-run0131-949", "openai/clip-vit-large-patch14"),
    ModelSpec("clip_l14_224", "s1", "clip_l14_224_s1", "fesvhtr/clip-r-s1-run1207-1280", "openai/clip-vit-large-patch14"),
    ModelSpec("clip_l14_224", "s2", "clip_l14_224_s2", "fesvhtr/clip-r-s2-run1219-505", "openai/clip-vit-large-patch14"),
    ModelSpec("siglip_so400m", "baseline", "siglip_so400m_base", "google/siglip-so400m-patch14-384", "google/siglip-so400m-patch14-384"),
    ModelSpec("siglip_so400m", "rea", "siglip_so400m_rea", "fesvhtr/siglip-r-rea-run0126-1241", "google/siglip-so400m-patch14-384"),
    ModelSpec("siglip_so400m", "des", "siglip_so400m_des", "fesvhtr/siglip-r-des-run0131-1266", "google/siglip-so400m-patch14-384"),
    ModelSpec("siglip_so400m", "s1", "siglip_so400m_s1", "fesvhtr/siglip-r-s1-run0201-1280", "google/siglip-so400m-patch14-384"),
    ModelSpec("siglip_so400m", "s2", "siglip_so400m_s2", "fesvhtr/siglip-r-s2-run0203-673", "google/siglip-so400m-patch14-384"),
]


@dataclass(frozen=True)
class Job:
    name: str
    cmd: list[str]
    log_path: Path
    outputs: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream benchmark matrix")
    parser.add_argument("--suite", choices=["clip_l14_224", "siglip_so400m", "all"], default="all")
    parser.add_argument("--roles", default="baseline,rea,des,s1,s2")
    parser.add_argument(
        "--tasks",
        default="voc,ade20k,nyuv2_depth,nyuv2_normals,navi_depth,navi_normals,refcoco,refcocoplus",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--refer-root", type=Path, default=DEFAULT_REFER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--torch-dtype", default="bf16", choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seg-batch-size", type=int, default=16)
    parser.add_argument("--geom-batch-size", type=int, default=16)
    parser.add_argument("--ground-batch-size", type=int, default=64)
    parser.add_argument("--voc-epochs", type=int, default=20)
    parser.add_argument("--ade-epochs", type=int, default=5)
    parser.add_argument("--nyuv2-epochs", type=int, default=20)
    parser.add_argument("--navi-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cache-geometry-features", action="store_true")
    parser.add_argument("--hf-fallback", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_models(args: argparse.Namespace) -> list[ModelSpec]:
    roles = {item.strip() for item in args.roles.split(",") if item.strip()}
    return [
        spec
        for spec in MODEL_SPECS
        if (args.suite == "all" or spec.suite == args.suite) and spec.role in roles
    ]


def add_common_model_args(cmd: list[str], args: argparse.Namespace, spec: ModelSpec) -> None:
    cmd.extend(
        [
            "--model-id",
            spec.model_id,
            "--processor-id",
            spec.processor_id,
            "--model-name",
            spec.name,
            "--out-dir",
            str(args.out_dir),
            "--torch-dtype",
            args.torch_dtype,
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
        ]
    )
    if args.local_files_only:
        cmd.append("--local-files-only")


def coalesce_tasks(tasks: list[str]) -> list[str]:
    task_set = set(tasks)
    coalesced = []
    consumed = set()
    for task in tasks:
        if task in consumed:
            continue
        if task == "nyuv2_depth" and "nyuv2_normals" in task_set:
            coalesced.append("nyuv2_both")
            consumed.update({"nyuv2_depth", "nyuv2_normals"})
        elif task == "navi_depth" and "navi_normals" in task_set:
            coalesced.append("navi_both")
            consumed.update({"navi_depth", "navi_normals"})
        elif task in {"nyuv2_normals", "navi_normals"} and task.replace("normals", "depth") in consumed:
            continue
        else:
            coalesced.append(task)
            consumed.add(task)
    return coalesced


def build_jobs(args: argparse.Namespace) -> list[Job]:
    tasks = coalesce_tasks([item.strip() for item in args.tasks.split(",") if item.strip()])
    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for spec in selected_models(args):
        for task in tasks:
            cmd = [sys.executable]
            outputs: list[Path]
            if task in {"voc", "ade20k"}:
                outputs = [args.out_dir / "segmentation" / f"{task}_{spec.name}.json"]
                cmd.append(str(SCRIPT_DIR / "eval_downstream_segmentation.py"))
                cmd.extend(
                    [
                        "--dataset",
                        task,
                        "--data-root",
                        str(args.data_root),
                        "--epochs",
                        str(1 if args.smoke else (args.voc_epochs if task == "voc" else args.ade_epochs)),
                        "--batch-size",
                        str(args.seg_batch_size),
                    ]
                )
                if args.smoke:
                    cmd.extend(["--max-train", "8", "--max-val", "4"])
            elif task in {"nyuv2_depth", "nyuv2_normals", "nyuv2_both", "navi_depth", "navi_normals", "navi_both"}:
                dataset, geometry_task = task.split("_", 1)
                if geometry_task == "both":
                    outputs = [
                        args.out_dir / "geometry" / f"{dataset}_depth_{spec.name}.json",
                        args.out_dir / "geometry" / f"{dataset}_normals_{spec.name}.json",
                    ]
                else:
                    outputs = [args.out_dir / "geometry" / f"{dataset}_{geometry_task}_{spec.name}.json"]
                cmd.append(str(SCRIPT_DIR / "eval_downstream_geometry.py"))
                data_root = args.data_root / "NYUv2" if dataset == "nyuv2" else args.data_root / "NAVI" / "navi_v1.0"
                epochs = args.nyuv2_epochs if dataset == "nyuv2" else args.navi_epochs
                cmd.extend(
                    [
                        "--dataset",
                        dataset,
                        "--task",
                        geometry_task,
                        "--data-root",
                        str(data_root),
                        "--epochs",
                        str(1 if args.smoke else epochs),
                        "--batch-size",
                        str(args.geom_batch_size),
                    ]
                )
                if args.cache_geometry_features:
                    cmd.append("--cache-features")
                if args.smoke:
                    cmd.extend(["--max-train", "8", "--max-eval", "4"])
            elif task in {"refcoco", "refcocoplus"}:
                dataset = "refcoco+" if task == "refcocoplus" else "refcoco"
                safe_dataset = dataset.replace("+", "plus")
                outputs = [args.out_dir / "grounding" / f"{safe_dataset}_testA_{spec.name}.json"]
                cmd.append(str(SCRIPT_DIR / "eval_downstream_grounding.py"))
                cmd.extend(
                    [
                        "--dataset",
                        dataset,
                        "--split",
                        "testA",
                        "--split-by",
                        "unc",
                        "--data-root",
                        str(args.refer_root),
                        "--batch-size",
                        str(args.ground_batch_size),
                    ]
                )
                if args.hf_fallback:
                    cmd.append("--hf-fallback")
                if args.smoke:
                    cmd.extend(["--max-refs", "8"])
            else:
                raise ValueError(f"Unknown task: {task}")

            add_common_model_args(cmd, args, spec)
            log_path = logs_dir / f"{spec.name}_{task}.log"
            jobs.append(Job(name=f"{spec.name}:{task}", cmd=cmd, log_path=log_path, outputs=outputs))
    return jobs


def run_job(job: Job, dry_run: bool, gpu: str, skip_existing: bool) -> tuple[str, int]:
    if skip_existing and job.outputs and all(path.exists() for path in job.outputs):
        outputs = ", ".join(str(path) for path in job.outputs)
        print(f"[SKIP] {job.name} existing={outputs}", flush=True)
        return job.name, 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    command_line = " ".join(job.cmd)
    header = f"[GPU {gpu}] {command_line}\n"
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        job.log_path.write_text(header, encoding="utf-8")
        print(f"[DRY] {job.name} -> {job.log_path}", flush=True)
        return job.name, 0
    with job.log_path.open("w", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        proc = subprocess.run(job.cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"[DONE] {job.name} status={proc.returncode} log={job.log_path}", flush=True)
    return job.name, proc.returncode


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    if not jobs:
        raise RuntimeError("No jobs selected")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        gpus = ["0"]
    workers = max(1, min(args.parallel, len(gpus)))
    print(f"Selected {len(jobs)} jobs; workers={workers}; gpus={",".join(gpus)}; out_dir={args.out_dir}")
    if args.parallel <= 1:
        failures = [run_job(job, args.dry_run, gpus[idx % len(gpus)], args.skip_existing) for idx, job in enumerate(jobs)]
    else:
        failures = []
        gpu_pool: queue.Queue[str] = queue.Queue()
        for gpu in gpus[:workers]:
            gpu_pool.put(gpu)

        def run_with_gpu(job: Job) -> tuple[str, int]:
            gpu = gpu_pool.get()
            try:
                return run_job(job, args.dry_run, gpu, args.skip_existing)
            finally:
                gpu_pool.put(gpu)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_with_gpu, job) for job in jobs]
            for future in as_completed(futures):
                failures.append(future.result())
    failed = [(name, status) for name, status in failures if status != 0]
    if failed:
        print("Failures:")
        for name, status in failed:
            print(f"  {name}: {status}")
        raise SystemExit(1)
    print("All jobs completed")


if __name__ == "__main__":
    main()
