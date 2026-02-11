import sys
import os

# 获取项目根目录（noisyFER-main的路径）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)  # 把根目录加入搜索路径

import torch
import time
import numpy as np
from tqdm import tqdm
import argparse
from loader.dataloader_cifar10_multi import DataloaderCifar10_MultiLabel
from utils import *
from torch.utils import data
from metrics import *
import random
from tensorboardX import SummaryWriter
from model.inconsistent_label_model import InconsistLabelModel


####### training settings #########
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--img_size", default=32, type=int)
parser.add_argument("--log_step", default=200, type=int)
parser.add_argument("--val_step", default=200, type=int)
parser.add_argument("--save_step", default=400, type=int)
parser.add_argument("--num_epoch", default=100000, type=int)
parser.add_argument("--isTrain", default=True, type=bool)
parser.add_argument("--early_stop_patience", default=7, type=int)
parser.add_argument("--seed", default=42, type=int)


parser.add_argument("--fc_layer", default=512, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--noise_dim", default=100, type=int, help='dimension of noise vector')
parser.add_argument("--gan_start_epoch", default=0, type=int, help='start gan loss from which epoch')

parser.add_argument("--lambda_class", default=1.0, type=float, help='weight for cross entropy loss')
parser.add_argument("--lambda_gan", default=0.8, type=float, help='weight for gan loss')

# weights for marginal scores in D
parser.add_argument("--lambda_sx", default=1.0, type=float, help='loss weight for marginal score of image')
parser.add_argument("--lambda_sz0", default=1.0, type=float, help='loss weight for mariginal score of noise vector')
parser.add_argument("--lambda_sz1", default=1.0, type=float, help='loss weight for mariginal score of noisy label set 1/2/3')
# weights for joint score in D
parser.add_argument("--lambda_sxz", default=1.0, type=float, help='loss weights for joint score')

# iteration number for G/D training
parser.add_argument("--iter_G", default=1, type=int)
parser.add_argument("--iter_D", default=2, type=int)

parser.add_argument("--save_dir", default='runs_cifar10', type=str)
parser.add_argument("--root", default='datasets/cifar10/cifar-10-batches-py',
                    type=str, help='path to dataset folder')
args = parser.parse_args()
###################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(writer, logger):
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = DataloaderCifar10_MultiLabel(img_size=args.img_size, is_transform=True, split='train',
                                                 noise_ratio_list=[0.2, 0.3, 0.4])
    train_dataset.load_data(args.root)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8, worker_init_fn=seed_worker, generator=g)

    val_dataset = DataloaderCifar10_MultiLabel(img_size=args.img_size, is_transform=False, split='test')
    val_dataset.load_data(args.root)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=2, worker_init_fn=seed_worker, generator=g)

    model = InconsistLabelModel(args)
    model.setup()
    time_meter = averageMeter()

    total_steps = 0
    best_acc = -1.0
    best_epoch = -1
    should_stop = False
    for epoch in range(args.num_epoch):
        for step, data in tqdm(enumerate(train_dataloader)):
            start_ts = time.time()
            total_steps += 1

            model.set_input(data)
            model.optimize_params(epoch)
            time_meter.update(time.time() - start_ts)

            if total_steps % args.log_step == 0:
                writer.add_scalar('train/class_loss', model.loss_class.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_class: {:.4f}  Time/Image: {:.4f}'. \
                    format(step, epoch, model.loss_class.item(), time_meter.val / args.batch_size)
                logger.info(print_str)
                print(print_str)

                if epoch > args.gan_start_epoch:
                    writer.add_scalar('train/loss_gan', model.loss_gan.item(), total_steps)
                    print_str = 'Step {:d}/ Epoch {:d}]  loss_gan: {:.4f}  Time/Image: {:.4f}'. \
                        format(step, epoch, model.loss_gan.item(), time_meter.val / args.batch_size)
                    logger.info(print_str)
                    print(print_str)

                    writer.add_scalar('train/G_total_loss', model.loss_G.item(), total_steps)
                    print_str = 'Step {:d}/ Epoch {:d}]  G_total_loss: {:.4f}  Time/Image: {:.4f}'. \
                        format(step, epoch, model.loss_G.item(), time_meter.val / args.batch_size)
                    logger.info(print_str)
                    print(print_str)
                    time_meter.reset()

                    # D loss
                    writer.add_scalar('train/D_total_loss', model.loss_D.item(), total_steps)
                    print_str = 'Step {:d}/ Epoch {:d}]  D_gan_loss: {:.4f}  Time/Image: {:.4f}'. \
                        format(step, epoch, model.loss_D.item(), time_meter.val / args.batch_size)
                    logger.info(print_str)
                    print(print_str)

            if total_steps % args.val_step == 0:
                model.encoder.eval()
                total, correct, correct_clean, correct_noise = 0, 0, 0, 0
                # if not args.noise:
                with torch.no_grad():
                    for step_val, data in tqdm(enumerate(val_dataloader)):
                        img = data[0].to(device)
                        clean_lbl = data[2].to(device)
                        total += clean_lbl.size(0)
                        _, _, _, z1_enc = model.encoder(img)
                        pred_1, pred_2, pred_3 = z1_enc[:, 0:10], z1_enc[:, 10:20], z1_enc[:, 20:]

                        _, predicted = pred_1.max(1)
                        correct += predicted.eq(clean_lbl).sum().item()

                    acc = 100.0 * correct / total
                    logger.info('acc:{}'.format(acc))
                    writer.add_scalar('val/acc', acc, total_steps)
                    print('acc:', acc)

                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    model.save_networks('best', writer.file_writer.get_logdir())
                    logger.info(f"[*] best models saved. acc: {best_acc:.2f} @ epoch {best_epoch}")
                elif best_epoch >= 0 and (epoch - best_epoch) >= args.early_stop_patience:
                    logger.info(
                        f"Early stopping: acc not improved for {args.early_stop_patience} epochs."
                    )
                    should_stop = True

                if total_steps % args.save_step == 0:
                    save_suffix = 'last'
                    model.save_networks(save_suffix, writer.file_writer.get_logdir())
                    print("[*] models saved.")
                    logger.info("[*] models saved.")

                if should_stop:
                    break

        if should_stop:
            break


        if epoch == args.num_epoch:
            break


if __name__ == '__main__':
    seed_everything(args.seed)
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))

    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    train(writer, logger)

