"""Exercise the real prediction heads without study images or DINOv2 downloads."""

import json

import pandas as pd
import torch
from torch import nn

from model_fixed import SurgicalAssessmentModel
from splits import make_folds


def main():
    torch.manual_seed(42)
    backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 16))
    model = SurgicalAssessmentModel(backbone, feat_dim=16, dropout=0).eval()
    with torch.no_grad():
        result = model(torch.rand(4, 3, 16, 16), return_uncertainty=True)
    for key in ("indicator_scores", "safety_probs", "epistemic_uncertainties", "aleatoric_uncertainties"):
        if not torch.isfinite(result[key]).all():
            raise RuntimeError(f"Nonfinite output: {key}")
    annotations = pd.DataFrame({
        "Name": [f"synthetic_{index}" for index in range(36)],
        "safety_label": [index % 3 for index in range(36)],
        "session": [f"session_{index // 3}" for index in range(36)],
    })
    folds = make_folds(annotations, folds=3, group_column="session")
    print(json.dumps({
        "mode": "synthetic wiring check, not research performance",
        "backbone": "small artificial backbone, not DINOv2",
        "indicator_shape": list(result["indicator_scores"].shape),
        "class_probability_shape": list(result["safety_probs"].shape),
        "group_disjoint_folds": len(folds),
        "downloads": False,
    }, indent=2))


if __name__ == "__main__":
    main()
