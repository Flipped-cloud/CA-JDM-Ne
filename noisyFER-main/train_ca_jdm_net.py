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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--img_size", default=128, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--log_step", default=50, type=int)
    parser.add_argument("--val_step", default=200, type=int)
    parser.add_argument("--save_step", default=500, type=int)
    parser.add_argument("--early_stop_patience", default=10, type=int)
    parser.add_argument("--isTrain", default=True, type=bool)
    parser.add_argument("--fc_layer", default=512, type=int)
    parser.add_argument("--noise_dim", default=100, type=int)

    parser.add_argument("--num_classes", default=7, type=int)
    parser.add_argument("--num_landmarks", default=68, type=int)
    parser.add_argument("--noise_ratio", default=0, type=float)
    parser.add_argument("--gan_start_epoch", default=10, type=int)

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
    parser.add_argument("--seed", default=3407, type=int)

    # RAF settings
    parser.add_argument(
        "--data_path",
        default="/root/autodl-tmp/CA-JDM-Ne/noisyFER-main/Dataset/RAF-DB/Image/aligned",
        type=str,
    )
    parser.add_argument("--raf_train_csv", default="/root/autodl-tmp/CA-JDM-Ne/noisyFER-main/Dataset/RAF-DB/train.csv", type=str)
    parser.add_argument("--raf_val_csv", default="/root/autodl-tmp/CA-JDM-Ne/noisyFER-main/Dataset/RAF-DB/test.csv", type=str)
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

    return parser.parse_args()


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    model = CAJDMNetModel(args)
    model.setup()
    if args.channels_last:
        model.encoder = model.encoder.to(memory_format=torch.channels_last)
        model.decoder = model.decoder.to(memory_format=torch.channels_last)

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
                # 记录当前状态，验证后恢复
                encoder_was_training = model.encoder.training
                model.encoder.eval()

                correct = 0
                total = 0
                with torch.inference_mode():
                    for data_val in tqdm(test_loader):
                        if args.channels_last:
                            img = data_val[0].to(model.device, non_blocking=True).to(memory_format=torch.channels_last)
                        else:
                            img = data_val[0].to(model.device, non_blocking=True)
                        clean_lbl = data_val[1].to(model.device, non_blocking=True)

                        _, _, _, z1_enc, _ = model.encoder(img)
                        _, predicted = z1_enc.max(1)

                        total += clean_lbl.size(0)
                        correct += predicted.eq(clean_lbl).sum().item()

                acc = 100.0 * correct / total
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

                # 恢复训练模式（关键）
                if encoder_was_training:
                    model.encoder.train()

                if should_stop:
                    break

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
