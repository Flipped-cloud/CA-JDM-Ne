import sys
import os

root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_path)

import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
from tensorboardX import SummaryWriter  # type: ignore

from loader.dataloader_raf import RAFDataset_Fusion
from loader.dataloader_affectnet_lmk import AffectNetDataset_Fusion
from utils import get_logger, save_config
from metrics import averageMeter
from model.nrmta_model import NRMTANetModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_model(model, test_loader, args):
    """Validation function for NRMTA-Net (dual_stream) model."""
    if hasattr(model, 'encoder'):
        encoder_was_training = model.encoder.training
        model.encoder.eval()
    else:
        encoder_was_training = None

    # ensure head eval too
    if hasattr(model, 'c_layer'):
        model.c_layer.eval()

    correct = 0
    total = 0
    use_tta = bool(getattr(args, "val_tta", False))
    
    with torch.inference_mode():
        for data_val in tqdm(test_loader):
            if args.channels_last:
                img = data_val[0].to(model.device if hasattr(model, 'device') else device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                img = data_val[0].to(model.device if hasattr(model, 'device') else device, non_blocking=True)
            clean_lbl = data_val[1].to(model.device if hasattr(model, 'device') else device, non_blocking=True)

            fer_embed, _ = model.encoder(img)
            if getattr(model, "use_arcface", False):
                z1_enc = model.c_layer(fer_embed, None)
            else:
                z1_enc = model.c_layer(fer_embed)

            if use_tta:
                img_flip = torch.flip(img, dims=[3])
                fer_embed_flip, _ = model.encoder(img_flip)
                if getattr(model, "use_arcface", False):
                    z1_flip = model.c_layer(fer_embed_flip, None)
                else:
                    z1_flip = model.c_layer(fer_embed_flip)
                z1_enc = 0.5 * (z1_enc + z1_flip)

            _, predicted = z1_enc.max(1)

            total += clean_lbl.size(0)
            correct += predicted.eq(clean_lbl).sum().item()
    
    # Restore training mode
    if encoder_was_training is not None and encoder_was_training:
        if hasattr(model, 'encoder'):
            model.encoder.train()
    
    return 100.0 * correct / total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--img_size", default=112, type=int)
    parser.add_argument("--num_epoch", default=150, type=int)
    parser.add_argument("--log_step", default=50, type=int)
    parser.add_argument("--val_step", default=100, type=int)
    parser.add_argument("--save_step", default=500, type=int)
    parser.add_argument("--early_stop_patience", default=20, type=int) # Increase patience from 10 to 20
    parser.add_argument("--isTrain", default=True, type=bool)
    parser.add_argument("--fc_layer", default=512, type=int)
    parser.add_argument("--noise_dim", default=100, type=int)

    parser.add_argument("--num_classes", default=7, type=int)
    parser.add_argument("--num_landmarks", default=68, type=int)
    parser.add_argument("--noise_ratio", default=0, type=float)
    parser.add_argument("--gan_start_epoch", default=999, type=int)#调参1 

    parser.add_argument("--lambda_exp", default=0.1, type=float)  # Adjusted default to 0.5 (was 1.0)
    parser.add_argument("--lambda_lmk", default=2.0, type=float)  # Adjusted default to 1.0 (was 0.5)
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_recon", default=1.0, type=float)
    parser.add_argument("--lambda_kl", default=0.001, type=float)
    parser.add_argument("--lambda_align", default=0.0, type=float)
    parser.add_argument("--align_dim", default=128, type=int)
    
    # New aggressiveness arguments
    parser.add_argument("--lambda_fer", default=0.5, type=float, help="Use this instead of lambda_exp for clarity if desired")
    parser.add_argument("--grad_clip", default=5.0, type=float, help="Max norm for gradient clipping")
    parser.add_argument("--step_decay_epoch", default=10, type=int, help="Epoch to aggressively decay LR by 0.1")
    parser.add_argument("--disable_aggressive_decay", action="store_true", help="Disable the one-shot 0.1 LR decay at step_decay_epoch")
    parser.add_argument("--freeze_fld_epoch", default=15, type=int, help="Epoch to freeze FLD branch (Delayed to allow alignment)")
    parser.add_argument("--lambda_exp_after_freeze", default=0.25, type=float,
                       help="FER loss weight after FLD freeze (dual_stream)")
    parser.add_argument("--lambda_lmk_after_freeze", default=1.5, type=float,
                       help="Landmark loss weight after FLD freeze (dual_stream)")
    parser.add_argument("--val_tta", action="store_true", help="Enable horizontal-flip TTA in validation")

    parser.add_argument("--lambda_sx", default=1.0, type=float)
    parser.add_argument("--lambda_sz0", default=1.0, type=float)
    parser.add_argument("--lambda_sz1", default=1.0, type=float)
    parser.add_argument("--lambda_slm", default=1.0, type=float)
    parser.add_argument("--lambda_sxz", default=1.0, type=float)

    # Co-attention params (align CMCNN paper)
    parser.add_argument("--e_ratio", default=0.2, type=float)
    parser.add_argument("--scam_kernel", default=7, type=int, choices=[3, 7])

    parser.add_argument("--lr", default=0.00005, type=float) # Reduced from 0.0001
    parser.add_argument("--iter_G", default=1, type=int)
    parser.add_argument("--iter_D", default=0, type=int)

    # Landmarks are normalized to [0,1] in RAF dataloader, so use small WingLoss params
    parser.add_argument("--wing_w", default=0.5, type=float)
    parser.add_argument("--wing_epsilon", default=0.1, type=float)

    parser.add_argument("--save_dir", default="runs_nrmta", type=str)
    parser.add_argument("--dataset", default="raf", type=str, choices=["raf", "affectnet"])
    parser.add_argument("--seed", default=15, type=int)

    # Regularization / augmentation
    parser.add_argument("--label_smoothing", default=0.05, type=float)
    parser.add_argument("--no_color_jitter", action="store_true")
    parser.add_argument("--no_random_erasing", action="store_true")
    parser.add_argument("--no_class_balance", action="store_true")

    parser.add_argument("--fer_pretrained_path", default="checkpoints/ms1mv3_arcface_r50.pth", type=str,
                       help="Path to IR50 pretrained weights for dual_stream")
    parser.add_argument("--fld_pretrained_path", default="checkpoints/mobilefacenet_model_best.pth", type=str,
                       help="Path to MobileFaceNet pretrained weights for dual_stream")
    parser.add_argument("--resume_seed", default=15, type=int,
                       help="If set, load best model from runs directory of this seed and resume training from its weights")
    parser.add_argument("--resume_path", default="runs_nrmta/15", type=str,
                       help="If set, load weights from this file (single file) or directory (will search for best files)")

    # RAF settings
    parser.add_argument(
        "--data_path",
        default="Dataset/RAF-DB/Image/aligned",
        type=str,
    )
    parser.add_argument("--raf_train_csv", default="Dataset/RAF-DB/train.csv", type=str)
    parser.add_argument("--raf_val_csv", default="Dataset/RAF-DB/test.csv", type=str)
    # AffectNet settings
    parser.add_argument("--affectnet_img_root", default="../datasets/affectnet", type=str)
    parser.add_argument("--affectnet_train_csv", default="../datasets/affectnet/training.csv", type=str)
    parser.add_argument("--affectnet_val_csv", default="../datasets/affectnet/validate.csv", type=str)

    # performance (deterministic-friendly)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--channels_last", action="store_true")

    # ---- ArcFace (classification head) ----
    parser.add_argument("--use_arcface", action="store_true", help="Use ArcFace margin head for FER classification (recommended)")
    parser.add_argument("--arc_s", default=30.0, type=float)
    parser.add_argument("--arc_m", default=0.5, type=float)

    # ---- warmup to avoid recon/KL hurting early FER ----
    parser.add_argument("--kl_start_epoch", default=10, type=int)
    parser.add_argument("--recon_start_epoch", default=10, type=int)

    # ---- FER backbone LR multiplier (dual_stream encoder) ----
    parser.add_argument("--fer_lr_mult", default=0.2, type=float, help="FER backbone lr = lr * fer_lr_mult (dual_stream)")

    return parser.parse_args()


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    # Ensure each worker has a deterministic, yet different, seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(writer, logger, args):
    if args.dataset == "raf":
        train_dataset = RAFDataset_Fusion(args, phase="train")
        test_dataset = RAFDataset_Fusion(args, phase="test")
    else:
        train_dataset = AffectNetDataset_Fusion(args, phase="train")
        test_dataset = AffectNetDataset_Fusion(args, phase="test")
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Calculate weights for sampling (Class Balancing)
    sampler = None
    shuffle = True
    if args.dataset == "raf" and not args.no_class_balance:
        try:
            targets = train_dataset.train_labels 
            class_counts = np.bincount(targets)
            class_weights = 1. / class_counts
            sample_weights = class_weights[targets]
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            shuffle = False
            logger.info("Using WeightedRandomSampler for class balancing.")
        except Exception as e:
            logger.warning(f"Could not init WeightedRandomSampler: {e}")

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle, # Use sampler instead of shuffle if available
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=args.pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    model = NRMTANetModel(args)
    model.setup()
    if args.channels_last:
        model.encoder = model.encoder.to(memory_format=torch.channels_last)
        model.decoder = model.decoder.to(memory_format=torch.channels_last)

    # Load pretrained weights if specified
    # Resume-from checkpoint (from another run's best) or load pretrained weights
    def _try_load_state_dict(module, path, desc):
        try:
            sd = torch.load(path, map_location="cpu")
            if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
                sd = sd['state_dict']
            module.load_state_dict(sd)
            logger.info(f"Loaded {desc} from {path}")
            return True
        except Exception as e:
            logger.debug(f"Could not load {desc} from {path}: {e}")
            return False

    resumed = False
    # If user provided a direct file path to resume from
    if args.resume_path:
        resume_path = args.resume_path
        if os.path.isfile(resume_path):
            # try to load as whole-model first
            if hasattr(model, 'load_state_dict'):
                try:
                    sd = torch.load(resume_path, map_location="cpu")
                    try:
                        model.load_state_dict(sd)
                        logger.info(f"Loaded full model state from {resume_path}")
                        resumed = True
                    except Exception:
                        # try nested fields
                        if isinstance(sd, dict) and 'state_dict' in sd:
                            model.load_state_dict(sd['state_dict'])
                            logger.info(f"Loaded model state['state_dict'] from {resume_path}")
                            resumed = True
                except Exception as e:
                    logger.debug(f"Failed to load resume_path as full model: {e}")
        elif os.path.isdir(resume_path):
            resume_dir = resume_path
        else:
            resume_dir = None
    else:
        resume_dir = None

    # If resume_seed provided, look under runs dir
    if not resumed and args.resume_seed is not None:
        cand_dir = os.path.join(args.save_dir, str(args.resume_seed))
        if os.path.isdir(cand_dir):
            resume_dir = cand_dir
        else:
            resume_dir = None

    def _load_nrmta_best_from_dir(resume_dir_path):
        loaded_any = False
        module_to_file = {
            "encoder": "best_net_encoder.pth",
            "decoder": "best_net_decoder.pth",
            "discriminator_x": "best_net_discriminator_x.pth",
            "discriminator_z0": "best_net_discriminator_z0.pth",
            "discriminator_emo": "best_net_discriminator_emo.pth",
            "discriminator_lmk": "best_net_discriminator_lmk.pth",
            "discriminator_joint_xz": "best_net_discriminator_joint_xz.pth",
            "c_layer": "best_net_c_layer.pth",
            "mean_layer": "best_net_mean_layer.pth",
            "logvar_layer": "best_net_logvar_layer.pth",
        }
        for module_name, filename in module_to_file.items():
            module = getattr(model, module_name, None)
            if module is None:
                continue
            ckpt_path = os.path.join(resume_dir_path, filename)
            if os.path.isfile(ckpt_path):
                ok = _try_load_state_dict(module, ckpt_path, module_name)
                loaded_any = loaded_any or ok
        return loaded_any

    # If we have a resume directory, try common filenames and model-specific loader
    if not resumed and resume_dir:
        resumed = _load_nrmta_best_from_dir(resume_dir)

        # Try common files
        if not resumed:
            enc_candidates = [
                os.path.join(resume_dir, 'best_model.pth'),
                os.path.join(resume_dir, 'best_encoder.pth'),
                os.path.join(resume_dir, 'best_net_encoder.pth'),
                os.path.join(resume_dir, 'best_encoder_state.pth'),
            ]
            for p in enc_candidates:
                if os.path.isfile(p) and hasattr(model, 'encoder'):
                    if _try_load_state_dict(model.encoder, p, 'encoder'):
                        resumed = True
                        break

        if not resumed and hasattr(model, "c_layer"):
            c_layer_candidates = [
                os.path.join(resume_dir, 'best_c_layer.pth'),
                os.path.join(resume_dir, 'best_classifier.pth'),
            ]
            for p in c_layer_candidates:
                if os.path.isfile(p) and _try_load_state_dict(model.c_layer, p, 'c_layer'):
                    resumed = True
                    break

    # Init schedulers
    scheduler_G = None
    scheduler_D = None
    if hasattr(model, 'optimizer_G'):
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer_G, T_max=args.num_epoch, eta_min=1e-6)
    elif hasattr(model, 'optimizer'):
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=args.num_epoch, eta_min=1e-6)
    
    if hasattr(model, 'optimizer_D'):
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer_D, T_max=args.num_epoch, eta_min=1e-6)

    time_meter = averageMeter()
    total_steps = 0
    best_acc = -1.0
    best_epoch = -1
    should_stop = False

    for epoch in range(args.num_epoch):
        for step, data_batch in tqdm(enumerate(train_loader)):
            start_ts = time.time()
            total_steps += 1

            if args.channels_last:
                data_batch = (
                    data_batch[0].to(model.device, non_blocking=True).to(memory_format=torch.channels_last),
                    data_batch[1].to(model.device, non_blocking=True),
                    data_batch[2].to(model.device, non_blocking=True),
                )
            else:
                data_batch = (
                    data_batch[0].to(model.device, non_blocking=True),
                    data_batch[1].to(model.device, non_blocking=True),
                    data_batch[2].to(model.device, non_blocking=True),
                )
            model.set_input(data_batch)
            model.optimize_params(epoch)
            time_meter.update(time.time() - start_ts)

            # --- Freeze FLD logic ---
            if epoch == args.freeze_fld_epoch and step == 0 and epoch > 0:
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'fld_backbone'):
                    logger.info("Freezing FLD branch (MobileFaceNet)...")
                    for param in model.encoder.fld_backbone.parameters():
                        param.requires_grad = False
                    # Make sure optimizer knows? 
                    # Actually, setting requires_grad=False is enough to stop gradients, 
                    # but optimizer will still track them unless we remove them.
                    # Usually fine to leave in optimizer as 0 grad, efficiency loss is minimal for small model.
            # ------------------------

            if total_steps % args.log_step == 0:
                writer.add_scalar("train/loss_exp", model.loss_class.item(), total_steps)
                writer.add_scalar("train/loss_lmk", model.loss_lmk.item(), total_steps)
                log_msg = (
                    f"Step {step}/ Epoch {epoch}]  loss_exp: {model.loss_class.item():.4f}  "
                    f"loss_lmk: {model.loss_lmk.item():.4f}  Time/Image: {time_meter.val / args.batch_size:.4f}"
                )
                logger.info(log_msg)
                print(log_msg)

                if epoch > args.gan_start_epoch:
                    writer.add_scalar("train/loss_gan", model.loss_gan.item(), total_steps)
                    writer.add_scalar("train/loss_recon", model.loss_recon.item(), total_steps)
                    writer.add_scalar("train/loss_G", model.loss_G.item(), total_steps)
                    writer.add_scalar("train/loss_D", model.loss_D_gan.item(), total_steps)
                time_meter.reset()

            if total_steps % args.val_step == 0:
                acc = validate_model(model, test_loader, args)
                logger.info(f"acc: {acc:.2f}")
                writer.add_scalar("val/acc", acc, total_steps)
                print("acc:", acc)

                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    model.save_networks("best", writer.file_writer.get_logdir())
                    logger.info(f"[*] best models saved. acc: {best_acc:.2f} @ epoch {best_epoch}")
                elif best_epoch >= 0 and (epoch - best_epoch) >= args.early_stop_patience:
                    logger.info(
                        f"Early stopping: acc not improved for {args.early_stop_patience} epochs."
                    )
                    should_stop = True

                if should_stop:
                    break

        if scheduler_G is not None:
            scheduler_G.step()
        if scheduler_D is not None:
            scheduler_D.step()

        # --- Aggressive Decay logic ---
        if (not args.disable_aggressive_decay) and epoch + 1 == args.step_decay_epoch:
            logger.info("Aggressive LR Decay: multiplying all LRs by 0.1")
            if hasattr(model, 'optimizer_G'):
                for param_group in model.optimizer_G.param_groups:
                    param_group['lr'] *= 0.1
            elif hasattr(model, 'optimizer'):
                 for param_group in model.optimizer.param_groups:
                    param_group['lr'] *= 0.1
        # ------------------------------
        
        if scheduler_G:
             logger.info(f"Epoch {epoch} LR: {[f'{lr:.2e}' for lr in scheduler_G.get_last_lr()]}")

        if should_stop:
            break


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    run_id = args.seed
    logdir = os.path.join(args.save_dir, str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    logger = get_logger(logdir)
    logger.info("NRMTA-Net training")
    save_config(logdir, args)

    train(writer, logger, args)
