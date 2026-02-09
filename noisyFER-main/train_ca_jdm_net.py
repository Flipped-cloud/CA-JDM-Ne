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
from tensorboardX import SummaryWriter

from loader.dataloader_raf import RAFDataset_Fusion
from loader.dataloader_affectnet_lmk import AffectNetDataset_Fusion
from utils import get_logger, save_config
from metrics import averageMeter
from model.ca_jdm_model import CAJDMNetModel
from model.multi_task_model import MultiTaskModel
from model.encoder_net import IR50_Encoder
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleTaskModel:
    """Simple single-task FER model using IR50_Encoder"""
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # Initialize IR50 encoder
        self.encoder = IR50_Encoder(
            img_size=args.img_size,
            fc_layer=args.fc_layer,
            latent_dim=args.num_classes,
            noise_dim=args.noise_dim,
            use_dual_stream=args.use_dual_stream,
            use_ca=getattr(args, "use_ca", False),
        ).to(self.device)
        
        # Classification head
        self.classifier = nn.Linear(args.fc_layer, args.num_classes).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            [
                {"params": [p for n, p in self.encoder.named_parameters() if p.requires_grad], "lr": args.lr * 0.1},
                {"params": [p for n, p in self.classifier.named_parameters() if p.requires_grad], "lr": args.lr},
            ],
            lr=args.lr,
        )
        
    def setup(self):
        """Setup for compatibility with CAJDMNetModel interface"""
        pass
        
    def set_input(self, data_batch):
        self.img, self.label, _ = data_batch  # Ignore landmarks for single task
        
    def optimize_params(self, epoch):
        self.encoder.train()
        self.classifier.train()
        
        self.optimizer.zero_grad()
        
        # Forward pass - only use emotion logits (z1) from encoder
        _, _, _, z1, _ = self.encoder(self.img)
        pred = self.classifier(z1)
        
        # Compute loss
        self.loss_class = self.criterion(pred, self.label)
        self.loss_lmk = torch.tensor(0.0).to(self.device)  # Dummy for logging compatibility
        
        # Backward pass
        self.loss_class.backward()
        self.optimizer.step()
    
    def eval(self):
        self.encoder.eval()
        self.classifier.eval()
        
    def train(self):
        self.encoder.train()
        self.classifier.train()


def validate_model(model, test_loader, args):
    """Unified validation function for all model types"""
    # Set model to eval mode
    if hasattr(model, 'encoder'):
        encoder_was_training = model.encoder.training
        model.encoder.eval()
    else:
        encoder_was_training = None
    
    if hasattr(model, 'classifier'):
        model.classifier.eval()
    
    correct = 0
    total = 0
    
    with torch.inference_mode():
        for data_val in tqdm(test_loader):
            if args.channels_last:
                img = data_val[0].to(model.device if hasattr(model, 'device') else device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                img = data_val[0].to(model.device if hasattr(model, 'device') else device, non_blocking=True)
            clean_lbl = data_val[1].to(model.device if hasattr(model, 'device') else device, non_blocking=True)

            # Different forward pass based on model type
            if args.model_type == "ca_jdm":
                if getattr(model, "backbone_type", None) == "dual_stream":
                    # dual_stream 下提取特征，然后手动经过分类头
                    fer_embed, _ = model.encoder(img)
                    z1_enc = model.c_layer(fer_embed) 
                else:
                    # 传统架构 encoder 直接输出 5 个值，取第 4 个
                    _, _, _, z1_enc, _ = model.encoder(img)
                _, predicted = z1_enc.max(1)
            elif args.model_type == "single_task":
                _, _, _, z1, _ = model.encoder(img)
                pred = model.classifier(z1)
                _, predicted = pred.max(1)
            elif args.model_type == "multi_task":
                # Assuming MultiTaskModel has similar interface
                outputs = model.forward(img)
                _, predicted = outputs['fer_pred'].max(1)  # Assuming outputs is a dict
            else:
                raise ValueError(f"Unknown model_type for validation: {args.model_type}")

            total += clean_lbl.size(0)
            correct += predicted.eq(clean_lbl).sum().item()
    
    # Restore training mode
    if encoder_was_training is not None and encoder_was_training:
        if hasattr(model, 'encoder'):
            model.encoder.train()
        if hasattr(model, 'classifier'):
            model.classifier.train()
    
    return 100.0 * correct / total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--img_size", default=112, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--log_step", default=50, type=int)
    parser.add_argument("--val_step", default=200, type=int)
    parser.add_argument("--save_step", default=500, type=int)
    parser.add_argument("--early_stop_patience", default=20, type=int) # Increase patience from 10 to 20
    parser.add_argument("--isTrain", default=True, type=bool)
    parser.add_argument("--fc_layer", default=512, type=int)
    parser.add_argument("--noise_dim", default=100, type=int)

    parser.add_argument("--num_classes", default=7, type=int)
    parser.add_argument("--num_landmarks", default=68, type=int)
    parser.add_argument("--noise_ratio", default=0, type=float)
    parser.add_argument("--gan_start_epoch", default=10, type=int)#调参1 

    parser.add_argument("--lambda_exp", default=1.0, type=float)
    parser.add_argument("--lambda_lmk", default=0.5, type=float)
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_recon", default=1.0, type=float)
    parser.add_argument("--lambda_kl", default=0.001, type=float)

    parser.add_argument("--lambda_sx", default=1.0, type=float)
    parser.add_argument("--lambda_sz0", default=1.0, type=float)
    parser.add_argument("--lambda_sz1", default=1.0, type=float)
    parser.add_argument("--lambda_slm", default=1.0, type=float)
    parser.add_argument("--lambda_sxz", default=1.0, type=float)

    # Co-attention params (align CMCNN paper)
    parser.add_argument("--e_ratio", default=0.2, type=float)
    parser.add_argument("--scam_kernel", default=7, type=int, choices=[3, 7])

    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--iter_G", default=1, type=int)
    parser.add_argument("--iter_D", default=2, type=int)

    # Landmarks are normalized to [0,1] in RAF dataloader, so use small WingLoss params
    parser.add_argument("--wing_w", default=0.5, type=float)
    parser.add_argument("--wing_epsilon", default=0.1, type=float)

    parser.add_argument("--save_dir", default="runs_ca_jdm", type=str)
    parser.add_argument("--dataset", default="raf", type=str, choices=["raf", "affectnet"])
    parser.add_argument("--seed", default=42, type=int)

    # Model selection
    parser.add_argument("--model_type", default="ca_jdm", type=str, choices=["ca_jdm", "single_task", "multi_task"], 
                       help="ca_jdm: CA-JDM-Net with GAN, single_task: FER only, multi_task: FER+FLD")
    parser.add_argument("--backbone", default="dual_stream", type=str, choices=["ir50", "vgg", "resnet18", "dual_stream"], 
                       help="Backbone architecture")
    parser.add_argument("--use_ca", action="store_true", help="Enable Coordinate Attention (CA) in IR50 encoder")
    # parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable pretrained weights")
    parser.add_argument("--pretrained_path", default="./pretrained/ms1mv3_arcface_r50.pth", type=str,
                       help="Path to pretrained weights")
    parser.add_argument("--fer_pretrained_path", default="checkpoints/ms1mv3_arcface_r50.pth", type=str,
                       help="Path to IR50 pretrained weights for dual_stream")
    parser.add_argument("--fld_pretrained_path", default="checkpoints/mobilefacenet_model_best.pth", type=str,
                       help="Path to MobileFaceNet pretrained weights for dual_stream")
    parser.add_argument("--resume_seed", default=42, type=int,
                       help="If set, load best model from runs directory of this seed and resume training from its weights")
    parser.add_argument("--resume_path", default="/root/autodl-tmp/CA-JDM-Ne/noisyFER-main/runs_ca_jdm/42", type=str,
                       help="If set, load weights from this file (single file) or directory (will search for best files)")
    # parser.add_argument("--use_dual_stream", action="store_true", help="Use dual-stream attention (default: single-stream CBAM)")
    parser.add_argument("--no_dual_stream", action="store_true", help="Disable dual-stream attention")

    # RAF settings
    parser.add_argument(
        "--data_path",
        default="/root/autodl-tmp/CA-JDM-Ne/noisyFER-main/Dataset/RAF-DB/Image/aligned",
        type=str,
    )
    parser.add_argument("--raf_train_csv", default="/root/autodl-tmp/CA-JDM-Ne/noisyFER-main/Dataset/RAF-DB/train.csv", type=str)
    parser.add_argument("--raf_val_csv", default="/root/autodl-tmp/CA-JDM-Ne/noisyFER-main/Dataset/RAF-DB/test.csv", type=str)
    # parser.add_argument(
    #     "--data_path",
    #     default="D:\\Python-Program\\CA-JDM-Ne-main\\noisyFER-main\\Dataset\\RAF-DB\\Image\\aligned",
    #     type=str,
    # )
    # parser.add_argument("--raf_train_csv", default="D:\\Python-Program\\CA-JDM-Ne-main\\noisyFER-main\\Dataset\\RAF-DB\\train.csv", type=str)
    # parser.add_argument("--raf_val_csv", default="D:\\Python-Program\\CA-JDM-Ne-main\\noisyFER-main\\Dataset\\RAF-DB\\test.csv", type=str)
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

    args = parser.parse_args()
    args.use_pretrained = not args.no_pretrained
    args.use_dual_stream = not args.no_dual_stream
    return args


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
    if args.dataset == "raf":
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

    # Model selection based on model_type
    if args.model_type == "ca_jdm":
        model = CAJDMNetModel(args)
        model.setup()
        if args.channels_last:
            model.encoder = model.encoder.to(memory_format=torch.channels_last)
            model.decoder = model.decoder.to(memory_format=torch.channels_last)
    elif args.model_type == "single_task":
        model = SingleTaskModel(args)
        model.setup()
    elif args.model_type == "multi_task":
        model = MultiTaskModel(args)
        model.setup()
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
        
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

    # If we have a resume directory, try common filenames and model-specific loader
    if not resumed and resume_dir:
        # Prefer model's own loader if available
        if hasattr(model, 'load_networks'):
            try:
                model.load_networks("best", resume_dir)
                logger.info(f"Loaded networks via model.load_networks from {resume_dir}")
                resumed = True
            except Exception as e:
                logger.warning(f"model.load_networks failed: {e}")

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

            # classifier
            cls_candidates = [
                os.path.join(resume_dir, 'best_classifier.pth'),
                os.path.join(resume_dir, 'best_net_classifier.pth'),
            ]
            for p in cls_candidates:
                if os.path.isfile(p) and hasattr(model, 'classifier'):
                    _try_load_state_dict(model.classifier, p, 'classifier')

    # Finally, if not resumed and user requested pretrained backbone, load pretrained weights
    if not resumed and args.use_pretrained and hasattr(model, 'encoder') and hasattr(model.encoder, 'load_pretrained_weights'):
        # dual_stream 已在 CAJDMNetModel.__init__ 内部加载，不再重复调用单路径接口
        if not (args.model_type == "ca_jdm" and getattr(model, "backbone_type", None) == "dual_stream"):
            model.encoder.load_pretrained_weights(args.pretrained_path)

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
                    if hasattr(model, 'save_networks'):
                        model.save_networks("best", writer.file_writer.get_logdir())
                    else:
                        torch.save(model.encoder.state_dict(), os.path.join(writer.file_writer.get_logdir(), "best_encoder.pth"))
                        if hasattr(model, 'classifier'):
                            torch.save(model.classifier.state_dict(), os.path.join(writer.file_writer.get_logdir(), "best_classifier.pth"))
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
    logger.info("CA-JDM-Net training")
    save_config(logdir, args)

    train(writer, logger, args)
