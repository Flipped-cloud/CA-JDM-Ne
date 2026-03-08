import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    params: Dict[str, str]


def build_base_cmd() -> List[str]:
    return [
        sys.executable,
        "train_ca_jdm_net.py",
        "--backbone", "dual_stream",
        "--model_type", "dual_stream_el",
        "--use_arcface",
        "--arc_s", "30",
        "--arc_m", "0.5",
        "--no_class_balance",
        "--lambda_align", "0.5",
        "--grad_clip", "5.0",
        "--lambda_lmk", "2.0",
        "--batch_size", "64",
        "--num_epoch", "60", # Reduce epochs for faster verification (originally 150)
        "--step_decay_epoch", "20",
        "--val_tta",
        "--save_dir", "runs_fusion_verify_final"
    ]


def build_experiments() -> List[ExperimentConfig]:
    # 设计了3组精简实验验证修复后的效果
    return [
        ExperimentConfig(
            name="fusion_fix_seed2026_baseline",
            seed=2027,
            params={
                "lr": "5e-5",
                "label_smoothing": "0.05",
                "lambda_exp": "0.1",
                "lambda_exp_after_freeze": "0.2",
                "lambda_lmk_after_freeze": "1.0",
                "freeze_fld_epoch": "15",
            },
        ),
        ExperimentConfig(
            name="fusion_fix_seed3407_late_freeze",
            seed=3704,
            params={
                "lr": "4e-5",
                "label_smoothing": "0.04",
                "lambda_exp": "0.15",
                "lambda_exp_after_freeze": "0.25",
                "lambda_lmk_after_freeze": "0.5",
                "freeze_fld_epoch": "18",
            },
        ),
        ExperimentConfig(
            name="fusion_fix_seed42_aggressive",
            seed=41,
            params={
                "lr": "6e-5", # Slightly higher LR to jump out of local minima
                "label_smoothing": "0.06",
                "lambda_exp": "0.2",
                "lambda_exp_after_freeze": "0.3",
                "lambda_lmk_after_freeze": "0.5",
                "freeze_fld_epoch": "20", # Delay freeze
                "grad_clip": "10.0",
            },
        ),
    ]


def config_to_cli(cfg: ExperimentConfig) -> List[str]:
    cmd = build_base_cmd()
    cmd.extend(["--seed", str(cfg.seed)])

    for key, value in cfg.params.items():
        if key == "disable_aggressive_decay" and value.lower() == "true":
            cmd.append("--disable_aggressive_decay")
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd


def run_one_experiment(cfg: ExperimentConfig, dry_run: bool = False) -> int:
    cmd = config_to_cli(cfg)
    print("=" * 100)
    print(f"[RUN] {cfg.name}")
    print("[CMD] " + " ".join(cmd))

    if dry_run:
        return 0

    start = time.time()
    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start
    print(f"[END] {cfg.name} | return_code={proc.returncode} | elapsed={elapsed / 60:.2f} min")
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-run script for Fusion CA-JDM-Net experiments")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands without executing")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop immediately if any run fails")
    args = parser.parse_args()

    experiments = build_experiments()

    summary = []
    for exp in experiments:
        code = run_one_experiment(exp, dry_run=args.dry_run)
        summary.append((exp.name, code))
        if code != 0 and args.stop_on_error and not args.dry_run:
            break

    print("\n" + "#" * 100)
    print("Experiment summary")
    for name, code in summary:
        status = "OK" if code == 0 else f"FAILED({code})"
        print(f"- {name}: {status}")


if __name__ == "__main__":
    main()
