import os
import sys
import itertools

import torch
import torch.nn.functional as F

# 获取项目根目录（noisyFER-main的路径）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from model.base_model import BaseModel
from model.encoder_net import Encoder, IR50_Encoder, Encoder_Net
from model.decoder_net import Decoder_64, Decoder_112, Decoder_128, Decoder_224
from model.discriminator import (
    Discriminator_x,
    Discriminator_lbl,
    Discriminator_lmk,
    Discriminator_z0,
    Discriminator_joint,
)
from model.losses import WingLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hinge_d(real_scores, fake_scores):
    loss_real = torch.mean(F.relu(1.0 - real_scores))
    loss_fake = torch.mean(F.relu(1.0 + fake_scores))
    return loss_real + loss_fake


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def hinge_g(fake_scores):
    return -torch.mean(fake_scores)


class CAJDMNetModel(BaseModel):
    """CA-JDM-Net: Co-Attentive Generator + Reconstruction Decoder + Multi-modal Discriminator."""

    def __init__(self, args):
        super(CAJDMNetModel, self).initialize(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_classes = args.num_classes
        self.landmark_dim = args.num_landmarks * 2
        self.latent_dim = self.n_classes

        # Which backbone to use
        self.backbone_type = getattr(args, 'backbone', 'vgg')

        # Choose encoder based on backbone argument
        if self.backbone_type == 'dual_stream':
            # ============================================================
            # Heterogeneous Dual-Stream Encoder (IR50 + MobileFaceNet)
            # ============================================================
            fer_path = getattr(args, 'fer_pretrained_path', 'checkpoints/ms1mv3_arcface_r50.pth')
            fld_path = getattr(args, 'fld_pretrained_path', 'checkpoints/mobilefacenet_model_best.pth')

            self.encoder = Encoder_Net(
                img_size=self.args.img_size,
                fer_embedding_dim=self.args.fc_layer,  # 512
                fld_embedding_size=self.landmark_dim,   # 136
                use_se=getattr(self.args, 'use_se', False),
                e_ratio=getattr(self.args, "e_ratio", 0.2),
                scam_kernel=getattr(self.args, "scam_kernel", 7),
            ).to(self.device)

            # Load dual pretrained weights
            self.encoder.load_pretrained_weights(
                fer_path=fer_path,
                fld_path=fld_path,
            )

            # Separate output heads on top of fer_embedding (512-d)
            self.mean_layer = torch.nn.Linear(self.args.fc_layer, self.args.noise_dim).to(self.device)
            self.logvar_layer = torch.nn.Linear(self.args.fc_layer, self.args.noise_dim).to(self.device)
            self.c_layer = torch.nn.Linear(self.args.fc_layer, self.latent_dim).to(self.device)

        elif self.backbone_type == 'ir50':
            self.encoder = IR50_Encoder(
                img_size=self.args.img_size,
                fc_layer=self.args.fc_layer,
                latent_dim=self.latent_dim,
                noise_dim=self.args.noise_dim,
                use_dual_stream=getattr(self.args, "use_dual_stream", True),
                use_ca=getattr(self.args, "use_ca", False),
            ).to(self.device)
        else:
            # Legacy VGG-style encoder
            self.encoder = Encoder(
                img_size=self.args.img_size,
                fc_layer=self.args.fc_layer,
                latent_dim=self.latent_dim,
                noise_dim=self.args.noise_dim,
                e_ratio=getattr(self.args, "e_ratio", 0.2),
                scam_kernel=getattr(self.args, "scam_kernel", 7),
            ).to(self.device)

        if self.args.img_size == 64:
            self.decoder = Decoder_64(
                img_size=self.args.img_size,
                latent_dim=self.latent_dim + self.landmark_dim,
                noise_dim=self.args.noise_dim,
            ).to(self.device)
        elif self.args.img_size == 112:
            self.decoder = Decoder_112(
                img_size=self.args.img_size,
                latent_dim=self.latent_dim + self.landmark_dim,
                noise_dim=self.args.noise_dim,
            ).to(self.device)
        elif self.args.img_size == 128:
            self.decoder = Decoder_128(
                img_size=self.args.img_size,
                latent_dim=self.latent_dim + self.landmark_dim,
                noise_dim=self.args.noise_dim,
            ).to(self.device)
        elif self.args.img_size == 224:
            self.decoder = Decoder_224(
                img_size=self.args.img_size,
                latent_dim=self.latent_dim + self.landmark_dim,
                noise_dim=self.args.noise_dim,
            ).to(self.device)
        else:
            raise ValueError("Only img_size 64/112/128/224 supported for CA-JDM-Net decoder.")

        if self.args.isTrain:
            self.train_model_name = [
                "encoder",
                "decoder",
                "discriminator_x",
                "discriminator_z0",
                "discriminator_emo",
                "discriminator_lmk",
                "discriminator_joint_xz",
            ]

            # x / z0 / z1(emo) / lm / joint
            self.discriminator_x = Discriminator_x(
                img_size=self.args.img_size, out_channels=512, wide=True, with_sx=True, early_xfeat=False
            ).to(self.device)
            self.discriminator_z0 = Discriminator_z0(
                in_channels=self.args.noise_dim, out_channels=512
            ).to(self.device)
            self.discriminator_emo = Discriminator_lbl(
                in_channels=self.n_classes, out_channels=512, use_embedding=True
            ).to(self.device)
            self.discriminator_lmk = Discriminator_lmk(
                in_channels=self.landmark_dim, out_channels=512
            ).to(self.device)
            self.discriminator_joint_xz = Discriminator_joint(
                in_channels=512 * 3
            ).to(self.device)

            # Build G parameter groups
            g_param_groups = [
                {"params": [p for n, p in self.encoder.named_parameters() if p.requires_grad], "lr": self.args.lr * 0.1},
                {"params": [p for n, p in self.decoder.named_parameters() if p.requires_grad], "lr": self.args.lr},
            ]
            # dual_stream has separate output heads
            if self.backbone_type == 'dual_stream':
                g_param_groups.append(
                    {"params": list(self.mean_layer.parameters()) +
                               list(self.logvar_layer.parameters()) +
                               list(self.c_layer.parameters()),
                     "lr": self.args.lr}
                )

            self.optimizer_G = torch.optim.AdamW(
                g_param_groups,
                lr=self.args.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-4,
            )
            self.optimizer_D = torch.optim.AdamW(
                filter(
                    lambda p: p.requires_grad,
                    itertools.chain(
                        self.discriminator_x.parameters(),
                        self.discriminator_z0.parameters(),
                        self.discriminator_emo.parameters(),
                        self.discriminator_lmk.parameters(),
                        self.discriminator_joint_xz.parameters(),
                    ),
                ),
                lr=self.args.lr,
                betas=(0.5, 0.999),
                weight_decay=1e-4,
            )

            self.criterion_lmk = WingLoss(w=self.args.wing_w, epsilon=self.args.wing_epsilon).to(self.device)
            self.criterion_recon = torch.nn.L1Loss().to(self.device)
            smoothing = float(getattr(self.args, "label_smoothing", 0.0))
            self.criterion_class = LabelSmoothingCrossEntropy(smoothing=smoothing).to(self.device)

    def set_input(self, data):
        self.img = data[0].to(self.device)
        self.exp_lbl = data[1].to(self.device)
        self.landmarks = data[2].to(self.device)

        self.batch_size = self.img.shape[0]
        self.exp_lbl_onehot = torch.zeros(self.batch_size, self.n_classes, device=self.device)
        self.exp_lbl_onehot.scatter_(1, self.exp_lbl.view(-1, 1), 1.0)

    def _sample_z(self, z_mu, z_logvar):
        """Reparameterization trick for VAE."""
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        return z_mu + z_std * eps

    def forward_G(self, epoch):
        if self.backbone_type == 'dual_stream':
            # Dual-stream encoder returns (fer_embedding, fld_output)
            fer_embed, fld_pred = self.encoder(self.img)
            # fer_embed: (B, 512);  fld_pred: (B, 136)
            self.mean = self.mean_layer(fer_embed)
            self.logvar = self.logvar_layer(fer_embed)
            self.z0_enc = self._sample_z(self.mean, self.logvar)
            self.z1_enc = self.c_layer(fer_embed)  # emotion logits
            self.lmk_pred = fld_pred
        else:
            # IR50_Encoder / legacy Encoder: returns 5 values
            self.mean, self.logvar, self.z0_enc, self.z1_enc, self.lmk_enc = self.encoder(self.img)
            self.lmk_pred = self.lmk_enc                         # (B,136)

        self.emo_prob = F.softmax(self.z1_enc, dim=1)        # (B,C)
        self.dec_img = self.decoder(torch.cat([self.z0_enc, self.emo_prob, self.lmk_pred], dim=1))

    def backward_G(self, epoch):
        # supervised
        self.loss_class = self.criterion_class(self.z1_enc, self.exp_lbl)

        lmk_tgt = self.landmarks.view(self.batch_size, -1)
        self.loss_lmk = self.criterion_lmk(self.lmk_pred, lmk_tgt)

        # VAE KL
        self.loss_kl = -0.5 * torch.mean(1 + self.logvar - self.mean.pow(2) - self.logvar.exp())

        # reconstruction
        self.loss_recon = self.criterion_recon(self.dec_img, self.img)

        # adversarial (only after gan_start_epoch)
        self.loss_gan = torch.tensor(0.0, device=self.device)

        if epoch >= self.args.gan_start_epoch:
            # ---- marginal x: real img vs fake reconstructed img ----
            _, s_x_real = self.discriminator_x(self.img)
            _, s_x_fake = self.discriminator_x(self.dec_img)

            # ---- marginal z0: real prior vs fake encoder z0 ----
            z0_real = torch.randn_like(self.z0_enc)
            _, s_z0_real = self.discriminator_z0(z0_real)
            _, s_z0_fake = self.discriminator_z0(self.z0_enc)

            # ---- marginal emo: real onehot vs fake prob ----
            _, s_emo_real = self.discriminator_emo(self.exp_lbl_onehot)
            _, s_emo_fake = self.discriminator_emo(self.emo_prob)

            # ---- marginal lmk: real vs fake ----
            _, s_lmk_real = self.discriminator_lmk(lmk_tgt)
            _, s_lmk_fake = self.discriminator_lmk(self.lmk_pred)

            # ---- joint: use x_feat(img) as the x-side feature ----
            x_feat_joint, _ = self.discriminator_x(self.img)
            emo_feat_real, _ = self.discriminator_emo(self.exp_lbl_onehot)
            emo_feat_fake, _ = self.discriminator_emo(self.emo_prob)
            lmk_feat_real, _ = self.discriminator_lmk(lmk_tgt)
            lmk_feat_fake, _ = self.discriminator_lmk(self.lmk_pred)

            s_xz_real = self.discriminator_joint_xz(x_feat_joint, emo_feat_real, lmk_feat_real)
            s_xz_fake = self.discriminator_joint_xz(x_feat_joint, emo_feat_fake, lmk_feat_fake)

            # G wants fake to be classified as real => -E[s_fake]
            loss_g_sx = hinge_g(s_x_fake)
            loss_g_sz0 = hinge_g(s_z0_fake)
            loss_g_sz1 = hinge_g(s_emo_fake)
            loss_g_slm = hinge_g(s_lmk_fake)
            loss_g_sxz = hinge_g(s_xz_fake)

            self.loss_gan = (
                self.args.lambda_sx * loss_g_sx
                + self.args.lambda_sz0 * loss_g_sz0
                + self.args.lambda_sz1 * loss_g_sz1
                + self.args.lambda_slm * loss_g_slm
                + self.args.lambda_sxz * loss_g_sxz
            )

        # total G loss
        self.loss_G = (
            self.args.lambda_exp * self.loss_class
            + self.args.lambda_lmk * self.loss_lmk
            + self.args.lambda_kl * self.loss_kl
            + self.args.lambda_recon * self.loss_recon
            + self.args.lambda_gan * self.loss_gan
        )
        self.loss_G.backward()

    def backward_D(self):
        lmk_tgt = self.landmarks.view(self.batch_size, -1)

        # ---- marginal x ----
        _, s_x_real = self.discriminator_x(self.img.detach())
        _, s_x_fake = self.discriminator_x(self.dec_img.detach())

        # ---- marginal z0 ----
        z0_real = torch.randn_like(self.z0_enc).detach()
        _, s_z0_real = self.discriminator_z0(z0_real)
        _, s_z0_fake = self.discriminator_z0(self.z0_enc.detach())

        # ---- marginal emo ----
        _, s_emo_real = self.discriminator_emo(self.exp_lbl_onehot.detach())
        _, s_emo_fake = self.discriminator_emo(self.emo_prob.detach())

        # ---- marginal lmk ----
        _, s_lmk_real = self.discriminator_lmk(lmk_tgt.detach())
        _, s_lmk_fake = self.discriminator_lmk(self.lmk_pred.detach())

        # ---- joint ----
        x_feat_joint, _ = self.discriminator_x(self.img.detach())
        emo_feat_real, _ = self.discriminator_emo(self.exp_lbl_onehot.detach())
        emo_feat_fake, _ = self.discriminator_emo(self.emo_prob.detach())
        lmk_feat_real, _ = self.discriminator_lmk(lmk_tgt.detach())
        lmk_feat_fake, _ = self.discriminator_lmk(self.lmk_pred.detach())

        s_xz_real = self.discriminator_joint_xz(x_feat_joint, emo_feat_real, lmk_feat_real)
        s_xz_fake = self.discriminator_joint_xz(x_feat_joint, emo_feat_fake, lmk_feat_fake)

        loss_d_sx = hinge_d(s_x_real, s_x_fake)
        loss_d_sz0 = hinge_d(s_z0_real, s_z0_fake)
        loss_d_sz1 = hinge_d(s_emo_real, s_emo_fake)
        loss_d_slm = hinge_d(s_lmk_real, s_lmk_fake)
        loss_d_sxz = hinge_d(s_xz_real, s_xz_fake)

        self.loss_D_gan = (
            self.args.lambda_sx * loss_d_sx
            + self.args.lambda_sz0 * loss_d_sz0
            + self.args.lambda_sz1 * loss_d_sz1
            + self.args.lambda_slm * loss_d_slm
            + self.args.lambda_sxz * loss_d_sxz
        )

        self.loss_D = self.loss_D_gan
        self.loss_D.backward()

    def func_require_grad(self, model_, flag_):
        for mm in model_:
            for p in mm.parameters():
                p.requires_grad = flag_

    def func_zero_grad(self, model_):
        for mm in model_:
            mm.zero_grad(set_to_none=True)

    def optimize_params(self, epoch):
        self.encoder.train()
        self.decoder.train()
        self.forward_G(epoch)

        # 1) Update D
        if epoch >= self.args.gan_start_epoch:
            self.func_require_grad(
                [
                    self.discriminator_x,
                    self.discriminator_z0,
                    self.discriminator_emo,
                    self.discriminator_lmk,
                    self.discriminator_joint_xz,
                ],
                True,
            )
            self.func_require_grad([self.encoder, self.decoder], False)

            for _ in range(self.args.iter_D):
                self.optimizer_D.zero_grad(set_to_none=True)
                # ensure we have latest fake image from current G
                self.forward_G(epoch)
                self.backward_D()
                self.optimizer_D.step()

        # 2) Update G
        self.func_require_grad(
            [
                self.discriminator_x,
                self.discriminator_z0,
                self.discriminator_emo,
                self.discriminator_lmk,
                self.discriminator_joint_xz,
            ],
            False,
        )
        self.func_require_grad([self.encoder, self.decoder], True)

        for _ in range(self.args.iter_G):
            self.optimizer_G.zero_grad(set_to_none=True)
            self.forward_G(epoch)
            self.backward_G(epoch)
            self.optimizer_G.step()
