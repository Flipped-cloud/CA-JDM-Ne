import sys
import os

root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_path)

import time
import argparse
import random
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
    parser.add_argument("--img_size", default=64, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--log_step", default=50, type=int)
    parser.add_argument("--val_step", default=200, type=int)
    parser.add_argument("--save_step", default=500, type=int)
    parser.add_argument("--isTrain", default=True, type=bool)
    parser.add_argument("--fc_layer", default=512, type=int)
    parser.add_argument("--noise_dim", default=100, type=int)

    parser.add_argument("--num_classes", default=7, type=int)
    parser.add_argument("--num_landmarks", default=68, type=int)
    parser.add_argument("--noise_ratio", default=0.2, type=float)
    parser.add_argument("--gan_start_epoch", default=5, type=int)

    parser.add_argument("--lambda_exp", default=1.0, type=float)
    parser.add_argument("--lambda_lmk", default=1.0, type=float)
    parser.add_argument("--lambda_gan", default=1.0, type=float)
    parser.add_argument("--lambda_recon", default=1.0, type=float)
    parser.add_argument("--lambda_kl", default=0.001, type=float)

    parser.add_argument("--lambda_sx", default=1.0, type=float)
    parser.add_argument("--lambda_sz0", default=1.0, type=float)
    parser.add_argument("--lambda_sz1", default=1.0, type=float)
    parser.add_argument("--lambda_slm", default=1.0, type=float)
    parser.add_argument("--lambda_sxz", default=1.0, type=float)

    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--iter_G", default=1, type=int)
    parser.add_argument("--iter_D", default=2, type=int)

    parser.add_argument("--wing_w", default=10.0, type=float)
    parser.add_argument("--wing_epsilon", default=2.0, type=float)

    parser.add_argument("--save_dir", default="runs_ca_jdm", type=str)
    parser.add_argument("--dataset", default="raf", type=str, choices=["raf", "affectnet"])

    # RAF settings
    parser.add_argument("--data_path", default="../datasets/rafd/basic", type=str)

    # AffectNet settings
    parser.add_argument("--affectnet_img_root", default="../datasets/affectnet", type=str)
    parser.add_argument("--affectnet_train_csv", default="../datasets/affectnet/training.csv", type=str)
    parser.add_argument("--affectnet_val_csv", default="../datasets/affectnet/validate.csv", type=str)

    return parser.parse_args()


def train(writer, logger, args):
    if args.dataset == "raf":
        train_dataset = RAFDataset_Fusion(args, phase="train")
        test_dataset = RAFDataset_Fusion(args, phase="test")
    else:
        train_dataset = AffectNetDataset_Fusion(args, phase="train")
        test_dataset = AffectNetDataset_Fusion(args, phase="test")
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = CAJDMNetModel(args)
    model.setup()

    time_meter = averageMeter()
    total_steps = 0

    for epoch in range(args.num_epoch):
        for step, data_batch in tqdm(enumerate(train_loader)):
            start_ts = time.time()
            total_steps += 1

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
                model.encoder.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data_val in tqdm(test_loader):
                        img = data_val[0].to(model.device)
                        clean_lbl = data_val[1].to(model.device)
                        _, _, _, z1_enc, _ = model.encoder(img)
                        _, predicted = z1_enc.max(1)
                        total += clean_lbl.size(0)
                        correct += predicted.eq(clean_lbl).sum().item()

                acc = 100.0 * correct / total
                logger.info(f"acc: {acc:.2f}")
                writer.add_scalar("val/acc", acc, total_steps)
                print("acc:", acc)

            if total_steps % args.save_step == 0:
                model.save_networks("last", writer.file_writer.get_logdir())
                logger.info("[*] models saved.")
                print("[*] models saved.")


if __name__ == "__main__":
    args = parse_args()
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    logger = get_logger(logdir)
    logger.info("CA-JDM-Net training")
    save_config(logdir, args)

    train(writer, logger, args)
