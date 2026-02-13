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
        "--model_type", "ca_jdm",
        "--use_arcface",
        "--arc_s", "30",
        "--arc_m", "0.5",
        "--kl_start_epoch", "15",
        "--recon_start_epoch", "15",
        "--no_class_balance",
        "--lambda_align", "0.5",
        "--align_dim", "256",
        "--enable_dual_stream_diag",
        "--diag_output", "dual_stream_diagnostics",
        "--diag_max_batches", "5",
        "--diag_interval_steps", "500",
        "--grad_clip", "5.0",
        "--lambda_lmk", "2.0",
        "--batch_size", "32",
        "--num_epoch", "150",
        "--freeze_fld_epoch", "15",
        "--step_decay_epoch", "20",
        "--val_tta",
    ]


def build_experiments() -> List[ExperimentConfig]:
    # All lrs <= 5e-5 by design
    return [
        # ExperimentConfig(
        #     name="exp_a_seed1301_lr5e5_ferfocus",
        #     seed=1301,
        #     params={
        #         "lr": "5e-5",
        #         "fer_lr_mult": "0.5",
        #         "label_smoothing": "0.05",
        #         "lambda_exp": "0.12",
        #         "lambda_exp_after_freeze": "0.28",
        #         "lambda_lmk_after_freeze": "1.4",
        #         #"disable_aggressive_decay": "true",
        #     },
        # ),
        ExperimentConfig(
            name="exp_b_seed1403_lr4e5_balanced",
            seed=1403,
            params={
                "lr": "4e-5",
                "fer_lr_mult": "0.65",
                "label_smoothing": "0.03",
                "lambda_exp": "0.14",
                "lambda_exp_after_freeze": "0.30",
                "lambda_lmk_after_freeze": "1.3",
                "freeze_fld_epoch": "12",
                #"disable_aggressive_decay": "true",
            },
        ),
        ExperimentConfig(
            name="exp_c_seed1507_lr35e5_stable",
            seed=1507,
            params={
                "lr": "3.5e-5",
                "fer_lr_mult": "0.7",
                "label_smoothing": "0.02",
                "lambda_exp": "0.15",
                "lambda_exp_after_freeze": "0.32",
                "lambda_lmk_after_freeze": "1.2",
                "grad_clip": "3.0",
                #"disable_aggressive_decay": "true",
            },
        ),
        ExperimentConfig(
            name="exp_d_seed1613_lr3e5_latefreeze",
            seed=1613,
            params={
                "lr": "3e-5",
                "fer_lr_mult": "0.6",
                "label_smoothing": "0.07",
                "lambda_exp": "0.10",
                "lambda_exp_after_freeze": "0.26",
                "lambda_lmk_after_freeze": "1.6",
                "freeze_fld_epoch": "18",
                #"disable_aggressive_decay": "true",
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

    lr_value = float(cfg.params.get("lr", "5e-5"))
    if lr_value > 5e-5:
        raise ValueError(f"Experiment {cfg.name} violates lr<=5e-5: {lr_value}")

    if dry_run:
        return 0

    start = time.time()
    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start
    print(f"[END] {cfg.name} | return_code={proc.returncode} | elapsed={elapsed / 60:.2f} min")
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click multi-run script for CA-JDM-Net experiments")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands without executing")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop immediately if any run fails")
    args = parser.parse_args()

    experiments = build_experiments()
    if len(experiments) < 3:
        raise ValueError("At least three experiment configurations are required.")

    seeds = [exp.seed for exp in experiments]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Each experiment must use a different random seed.")

    # Ensure experiments differ by at least one sensitive hyperparameter
    signatures = []
    for exp in experiments:
        sig = (
            exp.params.get("lr"),
            exp.params.get("fer_lr_mult"),
            exp.params.get("label_smoothing"),
            exp.params.get("lambda_exp"),
            exp.params.get("lambda_exp_after_freeze"),
            exp.params.get("lambda_lmk_after_freeze"),
            exp.params.get("freeze_fld_epoch", "15"),
            exp.params.get("grad_clip", "5.0"),
        )
        signatures.append(sig)
    if len(set(signatures)) != len(signatures):
        raise ValueError("Each experiment must differ in at least one sensitive hyperparameter.")

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
