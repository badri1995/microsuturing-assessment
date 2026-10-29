import argparse
from copy import copy

import torch


import torch.nn as nn


from torch.optim import AdamW


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.utils.data import DataLoader, Subset


from splits import make_folds


from tqdm import tqdm


import numpy as np


from pathlib import Path


import pandas as pd


import json


from dataset_fixed import MicrosturingDataset, get_transforms


from model_fixed import create_model


from losses_fixed import MultiTaskLoss, compute_class_weights


def compute_regression_metrics(predictions, targets):
    """Compute L1, L2 and binned metrics"""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    l1 = np.abs(predictions - targets).mean()
    l2 = ((predictions - targets) ** 2).mean()
    
    bins = {
        '0-3': (targets >= 0) & (targets <= 3),
        '4-6': (targets >= 4) & (targets <= 6),
        '7-10': (targets >= 7) & (targets <= 10)
    }
    
    bin_metrics = {}
    for bin_name, mask in bins.items():
        if mask.sum() > 0:
            bin_metrics[bin_name] = {
                'l1': float(np.abs(predictions[mask] - targets[mask]).mean()),
                'l2': float(((predictions[mask] - targets[mask]) ** 2).mean()),
                'count': int(mask.sum())
            }
        else:
            bin_metrics[bin_name] = {'l1': 0.0, 'l2': 0.0, 'count': 0}
    
    return {
        'l1': float(l1),
        'l2': float(l2),
        'bins': bin_metrics
    }


def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    all_preds = []
    all_labels = []
    overall_preds = []
    overall_targets = []
    
    for batch in tqdm(loader, desc='  Training', leave=False):
        images = batch['image'].to(device)
        safety_labels = batch['safety_label'].to(device)
        overall_scores = batch['overall_score'].to(device)
        indicator_scores = batch['indicator_scores'].to(device)
        
        outputs = model(images, return_uncertainty=False)
        
        targets = {
            'indicator_scores': indicator_scores,
            'safety_label': safety_labels
        }
        
        losses = criterion(outputs, targets)
        loss = losses['total']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect metrics
        preds = torch.argmax(outputs['safety_probs'], dim=1)
        all_preds.append(preds.detach())
        all_labels.append(safety_labels.detach())
        
        ind_mus = [outputs['indicator_predictions'][i][0] for i in range(5)]
        overall_pred = torch.cat(ind_mus, dim=1).mean(dim=1)
        overall_preds.append(overall_pred.detach())
        overall_targets.append(overall_scores.detach())
    
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    overall_preds = torch.cat(overall_preds).cpu().numpy()
    overall_targets = torch.cat(overall_targets).cpu().numpy()
    
    loss = total_loss / len(loader)
    acc = (all_preds == all_labels).mean() * 100
    mae = np.abs(overall_preds - overall_targets).mean()
    
    return loss, acc, mae


def evaluate(model, loader, device):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    overall_preds = []
    overall_targets = []
    
    # Individual indicators
    ind_preds = [[] for _ in range(5)]
    ind_targets = [[] for _ in range(5)]
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='  Evaluating', leave=False):
            images = batch['image'].to(device)
            safety_labels = batch['safety_label'].to(device)
            overall_scores = batch['overall_score'].to(device)
            indicator_scores = batch['indicator_scores'].to(device)
            
            outputs = model(images, return_uncertainty=False)
            
            preds = torch.argmax(outputs['safety_probs'], dim=1)
            all_preds.append(preds)
            all_labels.append(safety_labels)
            
            # Extract indicator predictions
            ind_mus = [outputs['indicator_predictions'][i][0] for i in range(5)]
            overall_pred = torch.cat(ind_mus, dim=1).mean(dim=1)
            
            overall_preds.append(overall_pred)
            overall_targets.append(overall_scores)
            
            for i in range(5):
                ind_preds[i].append(ind_mus[i].squeeze(-1))
                ind_targets[i].append(indicator_scores[:, i])
    
    # Move to CPU
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    overall_preds = torch.cat(overall_preds).cpu().numpy()
    overall_targets = torch.cat(overall_targets).cpu().numpy()
    
    ind_preds = [torch.cat(x).cpu().numpy() for x in ind_preds]
    ind_targets = [torch.cat(x).cpu().numpy() for x in ind_targets]
    
    # Overall accuracy
    overall_acc = (all_preds == all_labels).mean() * 100
    
    # Per-class accuracy
    per_class_acc = []
    per_class_counts = []
    for c in range(3):
        mask = all_labels == c
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            per_class_acc.append(float(acc))
            per_class_counts.append(int(mask.sum()))
        else:
            per_class_acc.append(0.0)
            per_class_counts.append(0)
    
    balanced_acc = np.mean(per_class_acc)
    
    # Critical failure rate
    unsafe_mask = all_labels == 0
    n_unsafe = unsafe_mask.sum()
    if n_unsafe > 0:
        critical_failures = ((all_labels == 0) & (all_preds == 2)).sum()
        cfr = critical_failures / n_unsafe * 100
    else:
        critical_failures = 0
        cfr = 0.0
    
    # Regression metrics
    regression_metrics = {
        'overall': compute_regression_metrics(overall_preds, overall_targets),
        'isd': compute_regression_metrics(ind_preds[0], ind_targets[0]),
        'slack': compute_regression_metrics(ind_preds[1], ind_targets[1]),
        'position': compute_regression_metrics(ind_preds[2], ind_targets[2]),
        'angulation': compute_regression_metrics(ind_preds[3], ind_targets[3]),
        'width': compute_regression_metrics(ind_preds[4], ind_targets[4])
    }
    
    return {
        'overall_acc': float(overall_acc),
        'balanced_acc': float(balanced_acc),
        'per_class_acc': per_class_acc,
        'per_class_counts': per_class_counts,
        'unsafe_count': int(n_unsafe),
        'critical_failures': int(critical_failures),
        'cfr': float(cfr),
        'overall_mae': float(regression_metrics['overall']['l1']),
        'regression_metrics': regression_metrics,
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist()
    }


def train_fold(fold_num, train_idx, val_idx, full_dataset, device, save_dir, epochs=20):
    """Train one fold"""
    print(f"\n{'='*80}")
    print(f"FOLD {fold_num+1}")
    print("="*80)
    
    # Create fold datasets
    train_subset = Subset(full_dataset, train_idx)
    validation_dataset = copy(full_dataset)
    validation_dataset.transform = get_transforms(img_size=224, mode='val')
    val_subset = Subset(validation_dataset, val_idx)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nFold {fold_num+1} data:")
    print(f"  Train: {len(train_subset)} samples")
    print(f"  Val:   {len(val_subset)} samples")
    
    # Get labels for class weights
    fold_labels = [full_dataset.df.iloc[i]['safety_label'] for i in train_idx]
    fold_label_counts = pd.Series(fold_labels).value_counts().sort_index()
    print(f"\nFold {fold_num+1} class distribution:")
    for label, count in fold_label_counts.items():
        pct = count / len(fold_labels) * 100
        print(f"  Class {label}: {count} ({pct:.1f}%)")
    
    # Create model
    model = create_model(device=device, unfreeze_last_n_blocks=0)
    
    # Compute class weights from this fold's training data
    class_counts = torch.tensor([
        fold_label_counts.get(0, 0),
        fold_label_counts.get(1, 0),
        fold_label_counts.get(2, 0)
    ], dtype=torch.float32)
    
    # Custom weights (same strategy as main training)
    class_weights = torch.tensor([0.5, 1.0, 5.0], dtype=torch.float32).to(device)
    
    # Loss
    criterion = MultiTaskLoss(
        lambda_indicators=1.0,
        lambda_safety=1.0,
        class_weights=class_weights
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.01
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_balanced_acc = float('-inf')
    best_metrics = None
    
    print(f"\nTraining fold {fold_num+1} for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_mae = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%, MAE {train_mae:.3f}")
        print(f"  Val:   Bal Acc {val_metrics['balanced_acc']:.2f}%, "
              f"Safe {val_metrics['per_class_acc'][2]:.1f}%, CFR {val_metrics['cfr']:.2f}%")
        
        # Save best model based on balanced accuracy
        if val_metrics['balanced_acc'] > best_balanced_acc:
            best_balanced_acc = val_metrics['balanced_acc']
            best_metrics = val_metrics
            
            torch.save({
                'fold': fold_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics
            }, save_dir / f'fold_{fold_num+1}_best.pth')
        
        scheduler.step()
    
    print(f"\n✓ Fold {fold_num+1} complete. Best balanced acc: {best_balanced_acc:.2f}%")
    
    return best_metrics



def main():
    parser = argparse.ArgumentParser(description="Research cross-validation for microsuturing image assessment.")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/cross_validation"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-column", help="Workbook column identifying related subjects or sessions.")
    args = parser.parse_args()
    if args.folds < 2 or args.epochs < 1:
        parser.error("folds must be at least 2 and epochs must be positive")
    if not args.annotations.is_file() or not args.images.is_dir():
        parser.error("provide an existing annotation workbook and image directory")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = MicrosturingDataset(args.annotations, args.images, transform=get_transforms(224, mode="train"))
    folds = make_folds(dataset.df, args.folds, args.seed, args.group_column)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = []
    print("Research use only. Related subjects, sessions and near-duplicate images require grouped splits.")
    if args.group_column is None:
        print("WARNING: image-level splitting selected; use --group-column for related images.")
    for fold_number, (train_indices, validation_indices) in enumerate(folds):
        metrics.append(train_fold(fold_number, train_indices, validation_indices, dataset, device, args.output, epochs=args.epochs))
    summary = {
        "folds": args.folds,
        "samples": len(dataset),
        "seed": args.seed,
        "split_strategy": "stratified_group" if args.group_column else "stratified_image",
        "group_column": args.group_column,
        "balanced_accuracy_mean": float(np.mean([result["balanced_acc"] for result in metrics])),
        "balanced_accuracy_std": float(np.std([result["balanced_acc"] for result in metrics])),
        "fold_metrics": metrics,
        "note": "Validation-selected fold metrics are not independent held-out test estimates."
    }
    with (args.output / "cv_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
