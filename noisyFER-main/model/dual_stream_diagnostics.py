import os
import json
from typing import Dict, List, Any

import torch


class _ActivationRecorder:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def hook(self, name: str):
        def _fn(module, inputs, output):
            with torch.no_grad():
                tensor = output
                if isinstance(output, (tuple, list)) and len(output) > 0:
                    tensor = output[0]
                if not torch.is_tensor(tensor):
                    return

                info = {
                    "name": name,
                    "shape": list(tensor.shape),
                    "mean": float(tensor.detach().float().mean().item()),
                    "std": float(tensor.detach().float().std(unbiased=False).item()),
                    "min": float(tensor.detach().float().min().item()),
                    "max": float(tensor.detach().float().max().item()),
                    "nan_count": int(torch.isnan(tensor).sum().item()),
                    "inf_count": int(torch.isinf(tensor).sum().item()),
                }
                self.records.append(info)

        return _fn


def _summarize_state_dict_loading(model) -> Dict[str, Any]:
    if not hasattr(model, "encoder"):
        return {"error": "model has no encoder"}

    encoder = model.encoder
    summary = {
        "has_dual_stream_encoder": hasattr(encoder, "fer_layer1") and hasattr(encoder, "fld_backbone"),
        "fer": {},
        "fld": {},
    }

    fer_prefixes = ("fer_input_layer", "fer_layer1", "fer_layer2", "fer_layer3", "fer_layer4")
    fld_prefixes = ("fld_backbone",)

    fer_total = fer_nonzero = 0
    fld_total = fld_nonzero = 0

    for name, param in encoder.named_parameters():
        if name.startswith(fer_prefixes):
            fer_total += 1
            if torch.count_nonzero(param.detach()).item() > 0:
                fer_nonzero += 1
        elif name.startswith(fld_prefixes):
            fld_total += 1
            if torch.count_nonzero(param.detach()).item() > 0:
                fld_nonzero += 1

    summary["fer"] = {
        "param_tensors": fer_total,
        "nonzero_param_tensors": fer_nonzero,
        "nonzero_ratio": float(fer_nonzero / fer_total) if fer_total > 0 else 0.0,
    }
    summary["fld"] = {
        "param_tensors": fld_total,
        "nonzero_param_tensors": fld_nonzero,
        "nonzero_ratio": float(fld_nonzero / fld_total) if fld_total > 0 else 0.0,
    }

    if hasattr(model, "mean_layer") and hasattr(model, "logvar_layer") and hasattr(model, "c_layer"):
        summary["heads"] = {
            "mean_layer_weight_norm": float(model.mean_layer.weight.detach().norm().item()),
            "logvar_layer_weight_norm": float(model.logvar_layer.weight.detach().norm().item()),
            "c_layer_weight_norm": float(model.c_layer.weight.detach().norm().item()),
        }

    return summary


def _grad_norms(model) -> Dict[str, float]:
    groups = {
        "fer": 0.0,
        "fld": 0.0,
        "attn": 0.0,
        "heads": 0.0,
    }

    if not hasattr(model, "encoder"):
        return groups

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().norm().item())

        if name.startswith("encoder.fer_input_layer") or name.startswith("encoder.fer_layer"):
            groups["fer"] += g
        elif name.startswith("encoder.fld_backbone"):
            groups["fld"] += g
        elif ".attn" in name or "encoder.attn" in name:
            groups["attn"] += g
        elif name.startswith("mean_layer") or name.startswith("logvar_layer") or name.startswith("c_layer"):
            groups["heads"] += g

    return groups


def _alignment_metrics(fer_x: torch.Tensor, fld_x: torch.Tensor) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "fer_shape": list(fer_x.shape),
        "fld_shape": list(fld_x.shape),
    }

    spatial_match = tuple(fer_x.shape[2:]) == tuple(fld_x.shape[2:])
    out["spatial_match"] = spatial_match

    if not spatial_match:
        fld_x = torch.nn.functional.interpolate(
            fld_x,
            size=fer_x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    if fld_x.shape[1] != fer_x.shape[1]:
        c = min(fld_x.shape[1], fer_x.shape[1])
        fld_slice = fld_x[:, :c]
        fer_slice = fer_x[:, :c]
    else:
        fld_slice = fld_x
        fer_slice = fer_x

    fer_flat = fer_slice.detach().flatten(1)
    fld_flat = fld_slice.detach().flatten(1)

    cos = torch.nn.functional.cosine_similarity(fer_flat, fld_flat, dim=1)
    out["cosine_mean"] = float(cos.mean().item())
    out["cosine_std"] = float(cos.std(unbiased=False).item())
    out["mean_abs_diff"] = float((fer_slice - fld_slice).abs().mean().item())

    return out


def run_dual_stream_diagnostics(model, train_loader, args, logger, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(model, "encoder"):
        logger.warning("[DIAG] Skip: model has no encoder.")
        return

    encoder = model.encoder
    required_attrs = ["fer_input_layer", "fer_layer1", "fer_layer2", "fer_layer3", "fer_layer4", "fld_backbone"]
    if not all(hasattr(encoder, a) for a in required_attrs):
        logger.warning("[DIAG] Skip: encoder is not dual_stream Encoder_Net.")
        return

    report: Dict[str, Any] = {
        "config": {
            "dataset": args.dataset,
            "model_type": args.model_type,
            "backbone": args.backbone,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
        },
        "weight_loading_summary": _summarize_state_dict_loading(model),
        "batches": [],
    }

    recorder = _ActivationRecorder()
    hooks = []
    hook_targets = {
        "fer_input": encoder.fer_input_layer,
        "fer_l1": encoder.fer_layer1,
        "fer_l2": encoder.fer_layer2,
        "fer_l3": encoder.fer_layer3,
        "fer_l4": encoder.fer_layer4,
        "fld_conv1": encoder.fld_backbone.conv1,
        "fld_conv23": encoder.fld_backbone.conv_23,
        "fld_conv34": encoder.fld_backbone.conv_34,
        "fld_conv45": encoder.fld_backbone.conv_45,
    }

    for n, m in hook_targets.items():
        hooks.append(m.register_forward_hook(recorder.hook(n)))

    attn_stats = []

    def _attn_hook(name):
        def _fn(module, inputs, outputs):
            with torch.no_grad():
                if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
                    return
                fer_o, fld_o = outputs
                if not (torch.is_tensor(fer_o) and torch.is_tensor(fld_o)):
                    return
                metrics = _alignment_metrics(fer_o, fld_o)
                metrics["name"] = name
                attn_stats.append(metrics)
        return _fn

    for i in range(1, 5):
        layer = getattr(encoder, f"attn{i}", None)
        if layer is not None:
            hooks.append(layer.register_forward_hook(_attn_hook(f"attn{i}")))

    model.train()
    max_batches = max(1, int(getattr(args, "diag_max_batches", 2)))

    iterator = iter(train_loader)
    for bidx in range(max_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        img = batch[0].to(model.device)
        lbl = batch[1].to(model.device)
        lmk = batch[2].to(model.device)

        model.set_input((img, lbl, lmk))
        model.optimize_params(epoch=0)

        batch_report = {
            "batch_index": bidx,
            "loss_class": float(model.loss_class.item()),
            "loss_lmk": float(model.loss_lmk.item()),
            "grad_norms": _grad_norms(model),
            "activations": recorder.records,
            "attention_alignment": attn_stats,
        }

        # Numerics sanity
        batch_report["numerics"] = {
            "z1_has_nan": bool(torch.isnan(model.z1_enc).any().item()),
            "z1_has_inf": bool(torch.isinf(model.z1_enc).any().item()),
            "lmk_has_nan": bool(torch.isnan(model.lmk_pred).any().item()),
            "lmk_has_inf": bool(torch.isinf(model.lmk_pred).any().item()),
        }

        report["batches"].append(batch_report)
        recorder.records = []
        attn_stats = []

    for h in hooks:
        h.remove()

    report_path = os.path.join(output_dir, "dual_stream_diag_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[DIAG] dual_stream report written: {report_path}")

    # Brief console summary
    if report["batches"]:
        b0 = report["batches"][0]
        logger.info(
            "[DIAG] batch0 class/lmk loss: %.4f / %.4f | grad(fer/fld/attn/heads)=%.3e/%.3e/%.3e/%.3e",
            b0["loss_class"],
            b0["loss_lmk"],
            b0["grad_norms"]["fer"],
            b0["grad_norms"]["fld"],
            b0["grad_norms"]["attn"],
            b0["grad_norms"]["heads"],
        )
