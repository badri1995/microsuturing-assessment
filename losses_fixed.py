import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


class NIGLoss(nn.Module):
    """
    Normal-Inverse-Gamma evidential regression loss
    Based on: Amini et al., "Deep Evidential Regression", NeurIPS 2020
    """
    
    def __init__(self, lambda_reg=0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self, y_true, mu, v, alpha, beta):
        """
        Args:
            y_true: ground truth [B, 1]
            mu, v, alpha, beta: NIG parameters [B, 1]
        
        Returns:
            loss: scalar
        """
        # NLL term
        two_beta_lambda = 2 * beta * (1 + v)
        
        nll = 0.5 * torch.log(np.pi / v) \
              - alpha * torch.log(two_beta_lambda) \
              + (alpha + 0.5) * torch.log(v * (y_true - mu)**2 + two_beta_lambda) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)
        
        # Regularization term
        reg = torch.abs(y_true - mu) * (2 * v + alpha)
        
        return (nll + self.lambda_reg * reg).mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning
    - Indicator regression (NIG)
    - Safety classification (CE with class weighting)
    """
    
    def __init__(
        self,
        lambda_indicators=1.0,
        lambda_safety=1.0,
        class_weights=None,
        use_focal_loss=False,
        focal_gamma=2.0
    ):
        """
        Args:
            lambda_indicators: weight for indicator loss
            lambda_safety: weight for safety loss
            class_weights: [3] tensor of class weights (Unsafe, Moderate, Safe)
            use_focal_loss: whether to use focal loss instead of CE
            focal_gamma: focal loss gamma parameter
        """
        super().__init__()
        
        self.lambda_indicators = lambda_indicators
        self.lambda_safety = lambda_safety
        self.use_focal_loss = use_focal_loss
        
        self.nig_loss = NIGLoss(lambda_reg=0.01)
        
        # Class weights
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.class_weights = class_weights
        else:
            self.class_weights = None
        
        # Focal loss or standard CE
        if use_focal_loss:
            self.focal_gamma = focal_gamma
            print(f"Using Focal Loss (gamma={focal_gamma})")
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
    
    def focal_loss(self, inputs, targets):
        """Focal Loss implementation"""
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none', 
            weight=self.class_weights
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
        return focal_loss
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from model
            targets: dict with labels
        
        Returns:
            dict with losses
        """
        # Indicator loss (NIG)
        indicator_loss = 0.0
        for i, (mu, v, alpha, beta) in enumerate(outputs['indicator_predictions']):
            target = targets['indicator_scores'][:, i:i+1]
            indicator_loss += self.nig_loss(target, mu, v, alpha, beta)
        indicator_loss /= len(outputs['indicator_predictions'])
        
        # Safety loss (CE or Focal)
        if self.use_focal_loss:
            safety_loss = self.focal_loss(outputs['safety_logits'], targets['safety_label'])
        else:
            safety_loss = self.ce_loss(outputs['safety_logits'], targets['safety_label'])
        
        # Total loss
        total_loss = (
            self.lambda_indicators * indicator_loss +
            self.lambda_safety * safety_loss
        )
        
        return {
            'total': total_loss,
            'indicators': indicator_loss.item(),
            'safety': safety_loss.item()
        }


def compute_class_weights(dataset, method='custom', boost_rare=5.0):
    """
    CORRECTED: Efficiently compute class weights from dataset
    Uses dataset.get_class_counts() - NO IMAGE LOADING!
    
    Args:
        dataset: MicrosturingDataset
        method: 'inverse_freq' | 'balanced' | 'custom'
        boost_rare: extra boost for rare classes
    
    Returns:
        class_weights: [3] tensor
    """
    # CORRECTED: Get counts efficiently from DataFrame
    class_counts = dataset.get_class_counts()
    
    print(f"\nClass distribution:")
    print(f"  Unsafe (0):   {class_counts[0]:.0f} ({class_counts[0]/len(dataset)*100:.1f}%)")
    print(f"  Moderate (1): {class_counts[1]:.0f} ({class_counts[1]/len(dataset)*100:.1f}%)")
    print(f"  Safe (2):     {class_counts[2]:.0f} ({class_counts[2]/len(dataset)*100:.1f}%)")
    
    if method == 'inverse_freq':
        # Inverse frequency
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * 3  # Normalize
        
        # Extra boost for Safe class
        weights[2] *= boost_rare
        weights = weights / weights.sum() * 3
        
    elif method == 'balanced':
        # Balanced
        weights = len(dataset) / (3 * class_counts)
        
    elif method == 'custom':
        # Custom: manually set based on your data
        weights = torch.tensor([0.5, 1.0, 5.0])  # Boost Safe class
    
    print(f"\nComputed class weights ({method}):")
    print(f"  Unsafe:   {weights[0]:.2f}")
    print(f"  Moderate: {weights[1]:.2f}")
    print(f"  Safe:     {weights[2]:.2f}")
    
    return weights


class BaselineMSELoss(nn.Module):
    """Baseline loss using MSE instead of NIG"""
    
    def __init__(self, lambda_indicators=1.0, lambda_safety=1.0, class_weights=None):
        super().__init__()
        self.lambda_indicators = lambda_indicators
        self.lambda_safety = lambda_safety
        self.mse = nn.MSELoss()
        
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, outputs, targets):
        # Extract mu from NIG outputs
        indicator_preds = torch.cat([
            out[0] for out in outputs['indicator_predictions']
        ], dim=1)
        
        indicator_loss = self.mse(indicator_preds, targets['indicator_scores'])
        safety_loss = self.ce(outputs['safety_logits'], targets['safety_label'])
        
        total_loss = (
            self.lambda_indicators * indicator_loss +
            self.lambda_safety * safety_loss
        )
        
        return {
            'total': total_loss,
            'indicators': indicator_loss.item(),
            'safety': safety_loss.item()
        }
