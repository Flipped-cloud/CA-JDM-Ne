import torch
import torch.nn.functional as F

from model.base_model import BaseModel
from model.dual_stream_backbone import DualStreamBackbone
from model.heads import ArcMarginProduct
from model.losses import WingLoss


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(smoothing)
        self.confidence = 1.0 - float(smoothing)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class HCADSNet(BaseModel):
    """HCADSNet: Heterogeneous Co-Attention Dual-Stream Network.

    Backbone-only (no reconstruction branch, no GAN, no VAE).
    Dual streams: IR-50 (FER) + MobileFaceNet (FLD) with HeteroCoAttention at each stage.

    Losses:
      - emotion classification: CE (optionally ArcFace head)
      - landmark regression: WingLoss
      - optional attention-map alignment: MSE (when lambda_align > 0)
    """

    def __init__(self, args):
        super().initialize(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if int(getattr(args, "img_size", 112)) not in (112, 224):
            raise ValueError("Only img_size 112/224 supported.")

        self.n_classes = int(args.num_classes)
        self.landmark_dim = int(args.num_landmarks) * 2

        self.backbone = DualStreamBackbone(
            img_size=int(args.img_size),
            fer_embedding_dim=int(getattr(args, "fc_layer", 512)),
            fld_embedding_size=self.landmark_dim,
            use_se=bool(getattr(args, "use_se", False)),
            e_ratio=float(getattr(args, "e_ratio", 0.2)),
            scam_kernel=int(getattr(args, "scam_kernel", 7)),
        ).to(self.device)

        fer_path = getattr(args, "fer_pretrained_path", "")
        fld_path = getattr(args, "fld_pretrained_path", "")
        self.backbone.load_pretrained_weights(fer_path=fer_path, fld_path=fld_path)

        self.use_arcface = bool(getattr(args, "use_arcface", False))
        if self.use_arcface:
            self.c_layer = ArcMarginProduct(
                in_features=int(getattr(args, "fc_layer", 512)),
                out_features=self.n_classes,
                s=float(getattr(args, "arc_s", 30.0)),
                m=float(getattr(args, "arc_m", 0.5)),
                easy_margin=False,
            ).to(self.device)
        else:
            self.c_layer = torch.nn.Linear(int(getattr(args, "fc_layer", 512)), self.n_classes).to(self.device)

        if self.isTrain:
            self.train_model_name = ["backbone"]

            self.optimizer_G = torch.optim.AdamW(
                [
                    {
                        "params": [p for _, p in self.backbone.named_parameters() if p.requires_grad],
                        "lr": float(args.lr),
                    },
                    {
                        "params": [p for _, p in self.c_layer.named_parameters() if p.requires_grad],
                        "lr": float(args.lr) * 10.0,
                    },
                ],
                lr=float(args.lr),
                betas=(0.9, 0.999),
                weight_decay=1e-4,
            )

            self.criterion_lmk = WingLoss(w=float(args.wing_w), epsilon=float(args.wing_epsilon)).to(self.device)
            smoothing = float(getattr(args, "label_smoothing", 0.0))
            self.criterion_class = LabelSmoothingCrossEntropy(smoothing=smoothing).to(self.device)
            self.criterion_align = torch.nn.MSELoss().to(self.device)

    def set_input(self, data):
        self.img = data[0].to(self.device)
        self.exp_lbl = data[1].to(self.device)
        self.landmarks = data[2].to(self.device)
        self.batch_size = self.img.shape[0]

    def forward(self, return_feats: bool = False):
        lambda_align = float(getattr(self.args, "lambda_align", 0.0))
        if return_feats or lambda_align > 0:
            fer_embed, lmk_pred, align_feats = self.backbone(self.img, return_feats=True)
            self.align_feats = align_feats
        else:
            fer_embed, lmk_pred = self.backbone(self.img)
            self.align_feats = None

        self.fer_embed = fer_embed
        self.lmk_pred = lmk_pred

        if self.use_arcface:
            self.logits = self.c_layer(fer_embed, self.exp_lbl if self.isTrain else None)
        else:
            self.logits = self.c_layer(fer_embed)

        return self.logits

    def backward(self):
        self.loss_class = self.criterion_class(self.logits, self.exp_lbl)

        lmk_tgt = self.landmarks.view(self.batch_size, -1)
        self.loss_lmk = self.criterion_lmk(self.lmk_pred, lmk_tgt)

        lambda_align = float(getattr(self.args, "lambda_align", 0.0))
        if lambda_align > 0 and getattr(self, "align_feats", None):
            align_losses = []
            for fer_map, fld_map in self.align_feats:
                align_losses.append(self.criterion_align(fer_map, fld_map.detach()))
            self.loss_align = torch.stack(align_losses).mean()
        else:
            self.loss_align = torch.tensor(0.0, device=self.device)

        lambda_exp = float(getattr(self.args, "lambda_exp", 0.1))
        lambda_lmk = float(getattr(self.args, "lambda_lmk", 2.0))
        freeze_fld_epoch = int(getattr(self.args, "freeze_fld_epoch", 15))
        if int(getattr(self.args, "epoch", -1)) >= freeze_fld_epoch:
            lambda_exp = float(getattr(self.args, "lambda_exp_after_freeze", lambda_exp))
            lambda_lmk = float(getattr(self.args, "lambda_lmk_after_freeze", lambda_lmk))

        self.loss_G = lambda_exp * self.loss_class + lambda_lmk * self.loss_lmk + lambda_align * self.loss_align
        self.loss_G.backward()

    def optimize_params(self, epoch: int):
        self.args.epoch = epoch
        self.backbone.train()
        self.c_layer.train()

        for _ in range(int(getattr(self.args, "iter_G", 1))):
            self.optimizer_G.zero_grad(set_to_none=True)
            self.forward()
            self.backward()

            grad_clip = float(getattr(self.args, "grad_clip", 5.0))
            if grad_clip > 0:
                params_to_clip = [p for group in self.optimizer_G.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)

            self.optimizer_G.step()
