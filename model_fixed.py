import torch


import torch.nn as nn


import torch.nn.functional as F


class NIGHead(nn.Module):
    """
    Normal-Inverse-Gamma regression head
    Outputs: mu (mean), v (precision), alpha (shape), beta (scale)
    
    CORRECTED UNCERTAINTY FORMULAS:
    - Aleatoric (data noise): beta / (alpha - 1)
    - Epistemic (model uncertainty): beta / (v * (alpha - 1))
    - Predictive variance: beta * (1 + v) / (v * (alpha - 1))
    """
    
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 4)  # mu, v, alpha, beta
        )
    
    def forward(self, h):
        out = self.net(h)
        mu = out[:, 0:1]
        v = F.softplus(out[:, 1:2]) + 1e-6
        alpha = F.softplus(out[:, 2:3]) + 1.0 + 1e-6
        beta = F.softplus(out[:, 3:4]) + 1e-6
        return mu, v, alpha, beta


class SurgicalAssessmentModel(nn.Module):
    """
    Complete surgical assessment model
    
    Architecture:
        Input Image
          ↓
        DINOv2 (frozen)
          ↓
        Features [B, 768]
          ↓
        ├─→ 5 NIG Heads (indicators)
        └─→ Safety Classifier (3-class)
    """
    
    def __init__(self, backbone, feat_dim=768, num_indicators=5, dropout=0.1):
        super().__init__()
        
        self.backbone = backbone
        self.num_indicators = num_indicators
        
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Indicator heads (5 NIG heads)
        self.indicator_heads = nn.ModuleList([
            NIGHead(feat_dim, dropout) for _ in range(num_indicators)
        ])
        
        # Safety classifier (simple 3-class)
        self.safety_classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # Unsafe, Moderate, Safe
        )
        
        # Temperature for calibration (learned post-hoc)
        # NOTE: This won't actually be trained during main training
        # It's only for post-hoc calibration if needed
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x, return_uncertainty=False, uncertainty_type='epistemic'):
        """
        Args:
            x: images [B, 3, H, W]
            return_uncertainty: whether to compute uncertainties
            uncertainty_type: 'epistemic' | 'aleatoric' | 'predictive'
                - epistemic: model uncertainty (for OOD detection)
                - aleatoric: data noise
                - predictive: total uncertainty
        
        Returns:
            dict with:
                - indicator_predictions: list of (mu, v, alpha, beta) tuples
                - indicator_scores: [B, 5] predicted means
                - safety_logits: [B, 3]
                - safety_probs: [B, 3] (with temperature scaling)
                - indicator_uncertainties: [B, 5] (if return_uncertainty)
                - epistemic_uncertainties: [B, 5] (if return_uncertainty)
                - aleatoric_uncertainties: [B, 5] (if return_uncertainty)
                - safety_confidence: [B, 1] (if return_uncertainty)
        """
        # Extract features
        features = self.backbone(x)  # [B, 768]
        
        # Indicator predictions
        indicator_outputs = []
        indicator_scores = []
        
        # Different types of uncertainty
        epistemic_uncertainties = []
        aleatoric_uncertainties = []
        predictive_uncertainties = []
        
        for head in self.indicator_heads:
            mu, v, alpha, beta = head(features)
            indicator_outputs.append((mu, v, alpha, beta))
            indicator_scores.append(mu)
            
            if return_uncertainty:
                # CORRECTED FORMULAS
                # Aleatoric (data noise)
                aleatoric = beta / (alpha - 1.0 + 1e-6)
                aleatoric_uncertainties.append(aleatoric)
                
                # Epistemic (model uncertainty) - FIXED!
                epistemic = beta / (v * (alpha - 1.0) + 1e-6)
                epistemic_uncertainties.append(epistemic)
                
                # Predictive variance (total)
                predictive = beta * (1 + v) / (v * (alpha - 1.0) + 1e-6)
                predictive_uncertainties.append(predictive)
        
        indicator_scores = torch.cat(indicator_scores, dim=1)  # [B, 5]
        
        # Select which uncertainty to return
        if return_uncertainty:
            epistemic_uncertainties = torch.cat(epistemic_uncertainties, dim=1)
            aleatoric_uncertainties = torch.cat(aleatoric_uncertainties, dim=1)
            predictive_uncertainties = torch.cat(predictive_uncertainties, dim=1)
            
            if uncertainty_type == 'epistemic':
                indicator_uncertainties = epistemic_uncertainties
            elif uncertainty_type == 'aleatoric':
                indicator_uncertainties = aleatoric_uncertainties
            elif uncertainty_type == 'predictive':
                indicator_uncertainties = predictive_uncertainties
            else:
                raise ValueError(f"Unknown uncertainty_type: {uncertainty_type}")
        
        # Safety classification
        safety_logits = self.safety_classifier(features)  # [B, 3]
        safety_probs = F.softmax(safety_logits / self.temperature, dim=1)
        
        # Prepare outputs
        outputs = {
            'indicator_predictions': indicator_outputs,
            'indicator_scores': indicator_scores,
            'safety_logits': safety_logits,
            'safety_probs': safety_probs,
        }
        
        if return_uncertainty:
            outputs['indicator_uncertainties'] = indicator_uncertainties
            outputs['epistemic_uncertainties'] = epistemic_uncertainties
            outputs['aleatoric_uncertainties'] = aleatoric_uncertainties
            outputs['predictive_uncertainties'] = predictive_uncertainties
            outputs['safety_confidence'] = torch.max(safety_probs, dim=1, keepdim=True)[0]
        
        return outputs
    
    def predict(self, x, uncertainty_type='epistemic'):
        """Convenience method for inference"""
        self.eval()
        with torch.no_grad():
            outputs = self(x, return_uncertainty=True, uncertainty_type=uncertainty_type)
        
        # Safety prediction
        safety_class = torch.argmax(outputs['safety_probs'], dim=1)
        safety_conf = outputs['safety_confidence'].squeeze(1)
        
        # Indicator predictions
        indicator_scores = outputs['indicator_scores']
        indicator_uncertainties = outputs['indicator_uncertainties']
        
        return {
            'safety_class': safety_class,
            'safety_confidence': safety_conf,
            'indicator_scores': indicator_scores,
            'indicator_uncertainties': indicator_uncertainties,
            'epistemic_uncertainties': outputs['epistemic_uncertainties'],
            'aleatoric_uncertainties': outputs['aleatoric_uncertainties']
        }
    
    def should_defer(self, x, tau_conf=0.7, tau_unc=1.0, uncertainty_type='epistemic'):
        """
        Deferral decision: defer if low confidence OR high uncertainty
        
        Args:
            uncertainty_type: Use 'epistemic' for OOD/deferral decisions
        
        Returns:
            defer_mask: [B] boolean
            reasons: list of strings
        """
        outputs = self(x, return_uncertainty=True, uncertainty_type=uncertainty_type)
        
        safety_conf = outputs['safety_confidence'].squeeze(1)  # [B]
        indicator_unc = outputs['indicator_uncertainties']  # [B, 5]
        max_unc = indicator_unc.max(dim=1)[0]  # [B]
        
        # Deferral conditions
        low_conf = safety_conf < tau_conf
        high_unc = max_unc > tau_unc
        defer_mask = low_conf | high_unc
        
        # Generate reasons
        indicator_names = ['ISD', 'Slack', 'Position', 'Angle', 'Width']
        reasons = []
        
        for i in range(len(defer_mask)):
            if defer_mask[i]:
                parts = []
                if low_conf[i]:
                    parts.append(f"Low conf: {safety_conf[i].item():.1%}")
                if high_unc[i]:
                    unc_idx = torch.argmax(indicator_unc[i]).item()
                    parts.append(f"High {indicator_names[unc_idx]} unc: {max_unc[i].item():.2f}")
                reasons.append(" | ".join(parts))
            else:
                reasons.append("Accepted")
        
        return defer_mask, reasons


def load_dinov2(device='cuda'):
    """Load DINOv2 ViT-B/14 from torch hub"""
    print("Loading DINOv2 ViT-B/14...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = model.to(device).eval()
    feat_dim = 768  # ViT-B has 768 dim
    print(f"✓ DINOv2 loaded (feature dim: {feat_dim})")
    return model, feat_dim


def create_model(device='cuda', dropout=0.1, unfreeze_last_n_blocks=0):
    """
    Create surgical assessment model with optional DINO unfreezing
    
    Args:
        device: Device to use
        dropout: Dropout rate
        unfreeze_last_n_blocks: Number of last DINO blocks to unfreeze (0-12)
    """
    # Load DINOv2 backbone
    backbone, feat_dim = load_dinov2(device)
    
    # Freeze all parameters first
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Create model - FIXED: Use correct constructor
    model = SurgicalAssessmentModel(
        backbone=backbone,
        feat_dim=feat_dim,
        dropout=dropout
    ).to(device)
    
    # Unfreeze last N blocks if requested
    trainable_params_backbone = 0
    if unfreeze_last_n_blocks > 0:
        n_blocks = len(backbone.blocks)
        blocks_to_unfreeze = backbone.blocks[-unfreeze_last_n_blocks:]
        
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
                trainable_params_backbone += param.numel()
        
        print(f"✓ Unfroze last {unfreeze_last_n_blocks} DINO blocks")
        print(f"  Backbone trainable params: {trainable_params_backbone:,}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"    - Backbone: {trainable_params_backbone:,}")
    print(f"    - Head: {trainable_params - trainable_params_backbone:,}")
    print(f"  Backbone frozen: {unfreeze_last_n_blocks == 0}")
    
    return model
