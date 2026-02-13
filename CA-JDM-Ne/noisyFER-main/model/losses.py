import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Wing Loss for Facial Landmark Detection
# ============================================================================

class WingLoss(nn.Module):
    """
    Wing Loss for robust facial landmark localisation.
    
    Reference: 
    'Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks'
    Feng et al., CVPR 2018
    
    Formula:
        loss = w * ln(1 + |x|/epsilon)   if |x| < w
        loss = |x| - C                   otherwise
    
    Where:
        x = predictions - targets
        C = w - w * ln(1 + w/epsilon)  (constant to ensure continuity)
    
    Args:
        w (float): Wing width parameter (default: 10.0)
                   Controls the transition point between log and linear regions
        epsilon (float): Curvature parameter (default: 2.0)
                        Controls the shape of the log curve
    
    Advantages:
        - More attention to small/medium errors (log region)
        - Robust to large outliers (linear region)
        - Smooth transition between regions
    """
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        # C = w - w * ln(1 + w/epsilon)
        # This constant ensures continuity at |x| = w
        self.C = w * (1.0 - math.log(1.0 + w / epsilon))

    def forward(self, predictions, targets):
        """
        Forward pass
        
        Args:
            predictions: (B, N) or (B, num_landmarks, 2)
                Predicted landmark coordinates
            targets: (B, N) or (B, num_landmarks, 2)
                Ground truth landmark coordinates
        
        Returns:
            loss: Scalar tensor
        """
        # Ensure same shape
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
        
        # Calculate absolute difference
        x = predictions - targets
        abs_x = torch.abs(x)
        
        # Determine which points are in the wing region (|x| < w)
        mask = abs_x < self.w
        
        # Calculate loss for each region
        # Wing region (small errors): logarithmic loss for higher gradient
        loss_wing = self.w * torch.log(1.0 + abs_x / self.epsilon)
        
        # Linear region (large errors): linear loss for robustness
        loss_linear = abs_x - self.C
        
        # Combine using mask
        loss = torch.where(mask, loss_wing, loss_linear)
        
        return torch.mean(loss)


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for robust facial landmark localisation.
    
    Reference:
    'Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression'
    Wang et al., ICCV 2019
    
    This is an improved version of Wing Loss that adapts to the target heatmap values.
    
    Formula:
        For |y - y_hat| < theta:
            loss = omega * ln(1 + |y - y_hat|^(alpha - y) / epsilon)
        For |y - y_hat| >= theta:
            loss = A * |y - y_hat| - C
    
    Where:
        A = omega * (1 / (1 + (theta/epsilon)^(alpha-y))) * (alpha-y) * (theta/epsilon)^(alpha-y-1) / epsilon
        C = theta * A - omega * ln(1 + (theta/epsilon)^(alpha-y))
    
    Args:
        omega (float): Wing width (default: 14.0)
        theta (float): Threshold parameter (default: 0.5)
        epsilon (float): Curvature parameter (default: 1.0)
        alpha (float): Adaptive parameter (default: 2.1)
    
    Advantages:
        - Adapts loss to target values
        - Better for heatmap-based landmark detection
        - More robust to occlusions and variations
    """
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, predictions, targets):
        """
        Forward pass
        
        Args:
            predictions: (B, N) or (B, C, H, W) for heatmaps
            targets: (B, N) or (B, C, H, W) for heatmaps
        
        Returns:
            loss: Scalar tensor
        """
        y = targets
        y_hat = predictions
        
        # Calculate absolute difference
        delta_y = torch.abs(y - y_hat)
        
        # Split into two regions based on theta
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        
        # Loss for small errors (|delta_y| < theta)
        if len(delta_y1) > 0:
            loss1 = self.omega * torch.log(
                1.0 + torch.pow(delta_y1 / self.epsilon, self.alpha - y1)
            )
        else:
            loss1 = torch.tensor(0.0, device=predictions.device)
        
        # Loss for large errors (|delta_y| >= theta)
        if len(delta_y2) > 0:
            A = self.omega * (1.0 / (1.0 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * \
                (self.alpha - y2) * torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1.0) / self.epsilon
            C = self.theta * A - self.omega * torch.log(
                1.0 + torch.pow(self.theta / self.epsilon, self.alpha - y2)
            )
            loss2 = A * delta_y2 - C
        else:
            loss2 = torch.tensor(0.0, device=predictions.device)
        
        # Combine losses
        total_elements = len(delta_y1) + len(delta_y2)
        if total_elements == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        if isinstance(loss1, torch.Tensor) and isinstance(loss2, torch.Tensor):
            return (loss1.sum() + loss2.sum()) / total_elements
        elif isinstance(loss1, torch.Tensor):
            return loss1.sum() / total_elements
        else:
            return loss2.sum() / total_elements


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss)
    
    A robust loss function that is less sensitive to outliers than MSE.
    Useful as a baseline for landmark regression.
    
    Formula:
        loss = 0.5 * x^2 / beta           if |x| < beta
        loss = |x| - 0.5 * beta           otherwise
    """
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
    
    def forward(self, predictions, targets):
        x = predictions - targets
        abs_x = torch.abs(x)
        
        mask = abs_x < self.beta
        loss = torch.where(
            mask,
            0.5 * x ** 2 / self.beta,
            abs_x - 0.5 * self.beta
        )
        return torch.mean(loss)


# ============================================================================
# Combined Multi-Task Loss
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning (FER + FLD)
    
    Args:
        fer_loss: Loss function for facial expression recognition (e.g., CrossEntropyLoss)
        fld_loss: Loss function for facial landmark detection (e.g., WingLoss)
        fer_weight: Weight for FER loss (default: 1.0)
        fld_weight: Weight for FLD loss (default: 0.1 to 0.5)
    """
    def __init__(self, fer_loss=None, fld_loss=None, fer_weight=1.0, fld_weight=0.1):
        super(MultiTaskLoss, self).__init__()
        self.fer_loss = fer_loss if fer_loss is not None else nn.CrossEntropyLoss()
        self.fld_loss = fld_loss if fld_loss is not None else WingLoss()
        self.fer_weight = fer_weight
        self.fld_weight = fld_weight
    
    def forward(self, fer_pred, fer_target, fld_pred, fld_target):
        """
        Calculate combined loss
        
        Args:
            fer_pred: Emotion predictions (B, num_emotions)
            fer_target: Emotion labels (B,)
            fld_pred: Landmark predictions (B, num_landmarks * 2)
            fld_target: Landmark ground truth (B, num_landmarks * 2)
        
        Returns:
            total_loss: Weighted sum of FER and FLD losses
            loss_dict: Dictionary containing individual losses
        """
        loss_fer = self.fer_loss(fer_pred, fer_target)
        loss_fld = self.fld_loss(fld_pred, fld_target)
        
        total_loss = self.fer_weight * loss_fer + self.fld_weight * loss_fld
        
        loss_dict = {
            'total': total_loss,
            'fer': loss_fer,
            'fld': loss_fld
        }
        
        return total_loss, loss_dict