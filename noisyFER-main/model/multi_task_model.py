import sys
import os

# 获取项目根目录（noisyFER-main的路径）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)  # 把根目录加入搜索路径

from model.discriminator import Discriminator_x, Discriminator_lbl, Discriminator_joint
from model.encoder_net import Encoder, IR50_Encoder
from model.decoder_net import Decoder_64, Decoder_128
from model.losses import WingLoss, AdaptiveWingLoss, MultiTaskLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from loss import *
from model.base_model import BaseModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiTaskModel(BaseModel):
    def __init__(self, args):
        super(MultiTaskModel, self).initialize(args)
        self.latent_dim = 7 + 2  # 7 emotion categories, 2 for valence/arousal valudes
        self.n_classes = 7
        self.encoder = Encoder(img_size=self.args.img_size, fc_layer=self.args.fc_layer,
                               latent_dim=self.latent_dim, noise_dim=self.args.noise_dim).to(device)
        if self.args.img_size == 64:
            self.decoder = Decoder_64(img_size=self.args.img_size,
                                      latent_dim=self.latent_dim, noise_dim=self.args.noise_dim).to(device)
        elif self.args.img_size == 128:
            self.decoder = Decoder_128(img_size=self.args.img_size,
                                       latent_dim=self.latent_dim, noise_dim=self.args.noise_dim).to(device)

        if self.args.isTrain:
            self.train_model_name = ['encoder', 'decoder']
            self.train_model_name += ['discriminator_x', 'discriminator_z0',
                                      'discriminator_z1_exp', 'discriminator_z1_va', 'discriminator_joint_xz']

            self.discriminator_x = Discriminator_x(img_size=self.args.img_size, out_channels=512, wide=True,
                                                   with_sx=True, early_xfeat=False).to(device)
            self.discriminator_z0 = Discriminator_lbl(in_channels=self.args.noise_dim, out_channels=512).to(device)
            self.discriminator_z1_exp = Discriminator_lbl(in_channels=7, out_channels=512).to(device)
            self.discriminator_z1_va = Discriminator_lbl(in_channels=2, out_channels=512).to(device)

            self.discriminator_joint_xz = Discriminator_joint(in_channels=512*4).to(device)

            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                       itertools.chain(self.encoder.parameters(),
                                                                       self.decoder.parameters(),
                                                                       )),
                                                lr=self.args.lr)
            self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                       itertools.chain(self.discriminator_x.parameters(),
                                                                       self.discriminator_z0.parameters(),
                                                                       self.discriminator_z1_exp.parameters(),
                                                                       self.discriminator_z1_va.parameters(),
                                                                       self.discriminator_joint_xz.parameters(),
                                                                       )),
                                                lr=self.args.lr)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)


    def set_input(self, data):
        self.img = data[0].to(device)  # [batch_size, 3, 256, 256]
        self.exp_lbl = data[1].to(device)
        self.va_lbl = data[2].to(device)

        self.batch_size = self.img.shape[0]
        self.exp_lbl_onehot = torch.zeros(self.batch_size, self.n_classes).to(device)
        self.exp_lbl_onehot[range(self.batch_size), self.exp_lbl] = 1.0  # [bs, 7], float, require_grad=F


    def sample_z(self, z_mu, z_logvar):
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        return z_mu + z_std * eps

    def forward_G(self, epoch):
        # encoder
        mean, logvar, self.z0_enc, self.z1_enc = self.encoder(self.img)
        self.z1_exp_enc = self.z1_enc[:, 0:7]
        self.z1_va_enc = self.z1_enc[:, 7:]

        if epoch > self.args.gan_start_epoch:
            # decoder
            self.z0_dec = torch.randn(self.batch_size, self.args.noise_dim).to(device)
            self.z1_exp_dec = self.exp_lbl_onehot
            self.z1_va_dec = self.va_lbl

            self.dec_img = self.decoder(torch.cat([self.z0_dec, self.z1_exp_dec, self.z1_va_dec], dim=1))


    def backward_G(self, epoch):
        # exp loss + va loss
        self.loss_class = F.cross_entropy(input=self.z1_exp_enc, target=self.exp_lbl, reduction='mean')
        loss_v = ccc(self.z1_va_enc[:, 0], self.va_lbl[:, 0])
        loss_a = ccc(self.z1_va_enc[:, 1], self.va_lbl[:, 1])
        self.loss_va = 1 - (loss_v + loss_a) / 2

        if epoch > self.args.gan_start_epoch:
            # encoder gan loss
            # marginal scores
            x_feat_enc, s_x_enc = self.discriminator_x(self.img)
            z0_feat_enc, s_z0_enc = self.discriminator_z0(self.z0_enc)
            z1_exp_feat_enc, s_z1_exp_enc = self.discriminator_z1_exp(self.z1_exp_enc)
            z1_va_feat_enc, s_z1_va_enc = self.discriminator_z1_va(self.z1_va_enc)
            # joint score
            s_xz_enc = self.discriminator_joint_xz(x_feat_enc, torch.cat([z0_feat_enc, z1_exp_feat_enc, z1_va_feat_enc], dim=1))
            score_enc_xz = torch.cat([self.args.lambda_sx * s_x_enc,
                                      self.args.lambda_sz0 * s_z0_enc,
                                      self.args.lambda_sz1_exp * s_z1_exp_enc,
                                      self.args.lambda_sz1_va * s_z1_va_enc,
                                      self.args.lambda_sxz * s_xz_enc], 1)
            # decoder gan loss
            # marginal scores
            x_feat_dec, s_x_dec = self.discriminator_x(self.dec_img)
            z0_feat_dec, s_z0_dec = self.discriminator_z0(self.z0_dec)
            z1_exp_feat_dec, s_z1_exp_dec = self.discriminator_z1_exp(self.z1_exp_dec)
            z1_va_feat_dec, s_z1_va_dec = self.discriminator_z1_va(self.z1_va_dec)
            # joint score
            s_xz_dec = self.discriminator_joint_xz(x_feat_dec, torch.cat([z0_feat_dec, z1_exp_feat_dec, z1_va_feat_dec], dim=1))
            score_dec_xz = torch.cat([self.args.lambda_sx * s_x_dec,
                                      self.args.lambda_sz0 * s_z0_dec,
                                      self.args.lambda_sz1_exp * s_z1_exp_dec,
                                      self.args.lambda_sz1_va * s_z1_va_dec,
                                      self.args.lambda_sxz * s_xz_dec], 1)

            self.loss_gan = torch.mean(score_enc_xz) + torch.mean(-score_dec_xz)

            # self.G_s_x = torch.mean(s_z0_enc) + torch.mean(-s_z0_dec)
            # self.G_s_z0 = torch.mean(s_x_enc) + torch.mean(-s_x_dec)
            # self.G_s_z1_exp = torch.mean(s_z1_exp_enc) + torch.mean(-s_z1_exp_dec)
            # self.G_s_z1_va = torch.mean(s_z1_va_enc) + torch.mean(-s_z1_va_dec)
            # self.G_s_xz = torch.mean(s_xz_enc) + torch.mean(-s_xz_dec)

            self.loss_G = self.args.lambda_exp * self.loss_class + \
                          self.args.lambda_va * self.loss_va + \
                          self.args.lambda_gan * self.loss_gan
        else:
            self.loss_G = self.args.lambda_exp * self.loss_class + self.args.lambda_va * self.loss_va
        self.loss_G.backward()


    def backward_D(self):
        # marginal scores
        x_feat_enc, s_x_enc = self.discriminator_x(self.img)
        z0_feat_enc, s_z0_enc = self.discriminator_z0(self.z0_enc.detach())
        z1_exp_feat_enc, s_z1_exp_enc = self.discriminator_z1_exp(self.z1_exp_enc.detach())
        z1_va_feat_enc, s_z1_va_enc = self.discriminator_z1_va(self.z1_va_enc.detach())
        # joint score
        s_xz_enc = self.discriminator_joint_xz(x_feat_enc, torch.cat([z0_feat_enc, z1_exp_feat_enc, z1_va_feat_enc], dim=1))
        score_enc_xz = torch.cat([self.args.lambda_sx * s_x_enc,
                                  self.args.lambda_sz0 * s_z0_enc,
                                  self.args.lambda_sz1_exp * s_z1_exp_enc,
                                  self.args.lambda_sz1_va * s_z1_va_enc,
                                  self.args.lambda_sxz * s_xz_enc], 1)

        # marginal scores
        x_feat_dec, s_x_dec = self.discriminator_x(self.dec_img.detach())
        z0_feat_dec, s_z0_dec = self.discriminator_z0(self.z0_dec)
        z1_exp_feat_dec, s_z1_exp_dec = self.discriminator_z1_exp(self.z1_exp_dec)
        z1_va_feat_dec, s_z1_va_dec = self.discriminator_z1_va(self.z1_va_dec)
        # joint score
        s_xz_dec = self.discriminator_joint_xz(x_feat_dec, torch.cat([z0_feat_dec, z1_exp_feat_dec, z1_va_feat_dec], dim=1))
        score_dec_xz = torch.cat([self.args.lambda_sx * s_x_dec,
                                  self.args.lambda_sz0 * s_z0_dec,
                                  self.args.lambda_sz1_exp * s_z1_exp_dec,
                                  self.args.lambda_sz1_va * s_z1_va_dec,
                                  self.args.lambda_sxz * s_xz_dec], 1)

        loss_real_xz = torch.mean(F.relu(1. - score_enc_xz))
        loss_fake_xz = torch.mean(F.relu(1. + score_dec_xz))
        self.loss_D_gan = loss_real_xz + loss_fake_xz

        # self.D_s_x = torch.mean(F.relu(1. - s_x_enc)) + torch.mean(F.relu(1. + s_x_dec))
        # self.D_s_z0 = torch.mean(F.relu(1. - s_z0_enc)) + torch.mean(F.relu(1. + s_z0_dec))
        # self.D_s_z1_exp = torch.mean(F.relu(1. - s_z1_exp_enc)) + torch.mean(F.relu(1. + s_z1_exp_dec))
        # self.D_s_z1_va = torch.mean(F.relu(1. - s_z1_va_enc)) + torch.mean(F.relu(1. + s_z1_va_dec))
        # self.D_s_xz = torch.mean(F.relu(1. - s_xz_enc)) + torch.mean(F.relu(1. + s_xz_dec))

        self.loss_D = self.loss_D_gan
        self.loss_D.backward()


    def func_require_grad(self, model_, flag_):
        for mm in model_:
            self.set_requires_grad(mm, flag_)

    def func_zero_grad(self, model_):
        for mm in model_:
            mm.zero_grad()

    def optimize_params(self, epoch):
        self.encoder.train()
        self.decoder.train()
        self.forward_G(epoch)

        # D
        if epoch > self.args.gan_start_epoch:
            self.func_require_grad([self.discriminator_x,
                                    self.discriminator_z0, self.discriminator_z1_exp, self.discriminator_z1_va,
                                    self.discriminator_joint_xz], True)
            for i in range(self.args.iter_D):
                self.func_zero_grad([self.discriminator_x,
                                    self.discriminator_z0, self.discriminator_z1_exp, self.discriminator_z1_va,
                                    self.discriminator_joint_xz])
                self.backward_D()
                self.optimizer_D.step()

        # G
        self.func_require_grad([self.discriminator_x,
                                self.discriminator_z0, self.discriminator_z1_exp, self.discriminator_z1_va,
                                self.discriminator_joint_xz], False)
        for i in range(self.args.iter_G):
            self.optimizer_G.zero_grad()
            if i > 0:
                self.forward_G(epoch)
            self.backward_G(epoch)
            self.optimizer_G.step()


# ============================================================================
# Multi-Task Model with IR50_Encoder (Module 3)
# ============================================================================

class MultiTaskModel_IR50(nn.Module):
    """
    Multi-Task Model for FER + FLD using IR50_Encoder
    
    This model implements:
    - Facial Expression Recognition (FER): 7 or 8 emotion classes
    - Facial Landmark Detection (FLD): 68 landmarks (136 coordinates)
    
    Architecture:
        Input -> IR50_Encoder (with attention) -> FER Head + FLD Head
    
    Args:
        num_emotions (int): Number of emotion classes (7 for RAF-DB, 8 for AffectNet)
        num_landmarks (int): Number of facial landmarks (default: 68)
        img_size (int): Input image size (default: 112)
        use_dual_stream (bool): Use dual-stream attention (default: False)
        use_se (bool): Use SE modules in IR50 (default: False)
        pretrained_path (str): Path to pretrained IR-50 weights
        fer_weight (float): Weight for FER loss (default: 1.0)
        fld_weight (float): Weight for FLD loss (default: 0.1 to 0.5)
    """
    def __init__(self, num_emotions=7, num_landmarks=68, img_size=112,
                 use_dual_stream=False, use_se=False, pretrained_path=None,
                 fer_weight=1.0, fld_weight=0.1, use_wing_loss=True,
                 use_ca=False):
        super(MultiTaskModel_IR50, self).__init__()
        
        self.num_emotions = num_emotions
        self.num_landmarks = num_landmarks
        self.img_size = img_size
        
        # =====================================================================
        # Encoder (IR-50 with Attention)
        # =====================================================================
        self.encoder = IR50_Encoder(
            img_size=img_size,
            use_se=use_se,
            fc_layer=512,
            latent_dim=num_emotions,  # For emotion classification
            noise_dim=100,             # For VAE (if needed)
            e_ratio=0.2,
            scam_kernel=7,
            use_dual_stream=use_dual_stream,
            use_ca=use_ca,
        )
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.encoder.load_pretrained_weights(pretrained_path)
            print(f"[MultiTaskModel_IR50] Loaded pretrained weights from {pretrained_path}")
        
        # =====================================================================
        # Task-specific Output Heads
        # =====================================================================
        # Note: encoder.embedding outputs fc_layer (512) dimensional features
        
        # FER Head: Emotion classification
        self.fc_fer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_emotions)
        )
        
        # FLD Head: Facial landmark detection
        self.fc_fld = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_landmarks * 2)  # 68 landmarks * 2 (x, y)
        )
        
        # =====================================================================
        # Loss Functions
        # =====================================================================
        self.fer_criterion = nn.CrossEntropyLoss()
        
        if use_wing_loss:
            # Use Adaptive Wing Loss for better landmark detection
            self.fld_criterion = AdaptiveWingLoss(omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1)
            print("[MultiTaskModel_IR50] Using Adaptive Wing Loss for landmarks")
        else:
            # Fallback to standard Wing Loss
            self.fld_criterion = WingLoss(w=10.0, epsilon=2.0)
            print("[MultiTaskModel_IR50] Using Wing Loss for landmarks")
        
        # Combined multi-task loss
        self.multi_task_loss = MultiTaskLoss(
            fer_loss=self.fer_criterion,
            fld_loss=self.fld_criterion,
            fer_weight=fer_weight,
            fld_weight=fld_weight
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize task-specific heads"""
        for m in [self.fc_fer, self.fc_fld]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, H, W)
            return_features: If True, also return intermediate features
        
        Returns:
            fer_output: Emotion logits (B, num_emotions)
            fld_output: Landmark coordinates (B, num_landmarks * 2)
            (optional) features: Intermediate features from encoder
        """
        # Encoder forward pass
        # Returns: mean, logvar, z0, z1, pred_landmarks
        # We use the embedding features for our custom heads
        mean, logvar, z0, z1, encoder_landmarks = self.encoder(x)
        
        # Note: encoder already has emotion logits (z1) and landmarks (encoder_landmarks)
        # But we add custom heads for better control and modularity
        
        # Get features from encoder's embedding layer
        # We need to extract features before the final layers
        # For now, we can use z1 or create a separate feature extractor
        # Let's use the encoder's internal features
        
        # Since encoder.forward doesn't directly expose features,
        # we can use z1 as features (latent_dim dimensional)
        # But ideally we want 512-dim features from embedding layer
        
        # Alternative: modify encoder to return features or
        # use the mean/z0 as features
        
        # For simplicity, let's use mean (noise_dim=100) + z1 (latent_dim) as features
        # Or better: add a method to encoder to get features
        
        # Temporary solution: re-extract features
        # (In production, modify encoder.forward to return embedding features)
        
        # Get embedding features (this requires accessing encoder internals)
        with torch.no_grad():
            x_enc = self.encoder.input_layer(x)
            x_enc = self.encoder.stage1(x_enc)
            x_enc = self.encoder.stage2(x_enc)
            
            if self.encoder.use_dual_stream:
                f3 = self.encoder.stage3_f(x_enc)
                l3 = self.encoder.stage3_l(x_enc)
                f3, l3 = self.encoder.ccam3(f3, l3)
                f3, l3 = self.encoder.scam3(f3, l3)
                
                f4 = self.encoder.stage4_f(f3)
                l4 = self.encoder.stage4_l(l3)
                f4, l4 = self.encoder.ccam4(f4, l4)
                f4, l4 = self.encoder.scam4(f4, l4)
                
                f_flat = f4.view(f4.shape[0], -1)
            else:
                feat3 = self.encoder.stage3_f(x_enc)
                feat3_att = feat3 + self.encoder.cbam3(feat3)
                
                feat4 = self.encoder.stage4_f(feat3_att)
                feat4_att = feat4 + self.encoder.cbam4(feat4)
                
                f_flat = feat4_att.view(feat4_att.shape[0], -1)
        
        # Now get embedding features
        features = self.encoder.embedding(f_flat)
        
        # Task-specific heads
        fer_output = self.fc_fer(features)
        fld_output = self.fc_fld(features)
        
        if return_features:
            return fer_output, fld_output, features
        else:
            return fer_output, fld_output
    
    def compute_loss(self, fer_pred, fer_target, fld_pred, fld_target):
        """
        Compute multi-task loss
        
        Args:
            fer_pred: Emotion predictions (B, num_emotions)
            fer_target: Emotion labels (B,)
            fld_pred: Landmark predictions (B, num_landmarks * 2)
            fld_target: Landmark ground truth (B, num_landmarks * 2)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        return self.multi_task_loss(fer_pred, fer_target, fld_pred, fld_target)
    
    def freeze_encoder(self, freeze=True):
        """Freeze encoder parameters for fine-tuning heads only"""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        print(f"[MultiTaskModel_IR50] Encoder {'frozen' if freeze else 'unfrozen'}")
    
    def get_optimizer_params(self, lr_backbone=1e-4, lr_heads=1e-3):
        """
        Get parameter groups with different learning rates
        
        Args:
            lr_backbone: Learning rate for encoder (lower for fine-tuning)
            lr_heads: Learning rate for task heads (higher for training from scratch)
        
        Returns:
            List of parameter groups for optimizer
        """
        return [
            {'params': self.encoder.parameters(), 'lr': lr_backbone},
            {'params': self.fc_fer.parameters(), 'lr': lr_heads},
            {'params': self.fc_fld.parameters(), 'lr': lr_heads}
        ]

