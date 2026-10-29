import torch


from torch.utils.data import Dataset, DataLoader


import pandas as pd


import cv2


import numpy as np


from pathlib import Path


import albumentations as A


from albumentations.pytorch import ToTensorV2


class MicrosturingDataset(Dataset):
    """
    Dataset for surgical microsuturing assessment
    
    Excel columns expected:
    - Name: image filename
    - ISD, Slack, Position, Angulation, Width: 5 indicator scores (0-10)
    - Overall: overall score (0-10)
    """
    
    def __init__(self, excel_path, image_dir, transform=None, safety_margins=[4.0, 7.0]):
        """
        Args:
            excel_path: path to Excel file with annotations
            image_dir: directory containing images
            transform: albumentations transform
            safety_margins: [unsafe_threshold, safe_threshold]
        """
        self.df = pd.read_excel(excel_path)
        required_columns = {'Name', 'ISD', 'Slack', 'Position', 'Angulation', 'Width', 'Overall'}
        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            raise ValueError(f'Missing annotation columns: {sorted(missing_columns)}')
        score_columns = ['ISD', 'Slack', 'Position', 'Angulation', 'Width', 'Overall']
        if self.df.empty or self.df['Name'].isna().any():
            raise ValueError('Annotations must be nonempty with an image name for every row.')
        self.df[score_columns] = self.df[score_columns].apply(pd.to_numeric, errors='raise')
        scores = self.df[score_columns].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 10).any():
            raise ValueError('All annotation scores must be finite numbers between 0 and 10.')
        
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.safety_margins = safety_margins
        
        # Indicator columns (must match Excel)
        self.indicator_cols = ['ISD', 'Slack', 'Position', 'Angulation', 'Width']
        self.df['safety_label'] = self.df['Overall'].apply(self._get_safety_label)

        print(f"Loaded dataset: {len(self.df)} images")
        print(f"  From: {excel_path}")
        print(f"  Images dir: {image_dir}")
        
        # Check data integrity
        self._check_data()
    
    def _check_data(self):
        """Verify data integrity"""
        # Check all expected columns exist
        required_cols = ['Name'] + self.indicator_cols + ['Overall']
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in Excel: {missing}")
        
        # Check for missing images
        missing_images = 0
        for idx, row in self.df.iterrows():
            img_path = self.image_dir / row['Name']
            if not img_path.exists():
                missing_images += 1
        
        if missing_images > 0:
            print(f"⚠️  Warning: {missing_images}/{len(self.df)} images not found on disk")
        
        # Print score statistics
        print(f"\nScore statistics:")
        for col in self.indicator_cols + ['Overall']:
            mean = self.df[col].mean()
            std = self.df[col].std()
            print(f"  {col:12s}: {mean:.2f} ± {std:.2f}")
        
        # Print safety class distribution
        safety_labels = self.df['Overall'].apply(self._get_safety_label)
        class_counts = safety_labels.value_counts().sort_index()
        print(f"\nSafety class distribution:")
        class_names = ['Unsafe', 'Moderate', 'Safe']
        for cls, count in class_counts.items():
            print(f"  {class_names[cls]:10s} ({cls}): {count:4d} ({count/len(self.df)*100:5.1f}%)")
    
    def _get_safety_label(self, overall_score):
        """
        Convert overall score to safety class
        
        Args:
            overall_score: 0-10 scale
        
        Returns:
            0: Unsafe (score <= 4)
            1: Moderate (4 < score <= 7)
            2: Safe (score > 7)
        """
        if overall_score <= self.safety_margins[0]:
            return 0  # Unsafe
        elif overall_score <= self.safety_margins[1]:
            return 1  # Moderate
        else:
            return 2  # Safe
    
    def get_class_counts(self):
        """
        CORRECTED: Efficiently compute class counts from DataFrame
        Does NOT load images!
        
        Returns:
            torch.Tensor: [3] class counts
        """
        class_counts = torch.zeros(3)
        safety_labels = self.df['Overall'].apply(self._get_safety_label)
        
        for cls in [0, 1, 2]:
            class_counts[cls] = (safety_labels == cls).sum()
        
        return class_counts
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - image: [3, H, W] tensor
                - indicator_scores: [5] tensor (ISD, Slack, Position, Angle, Width)
                - overall_score: scalar
                - safety_label: 0/1/2 (Unsafe/Moderate/Safe)
                - image_name: filename
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.image_dir / row['Name']
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Get scores
        indicator_scores = torch.tensor([
            row[col] for col in self.indicator_cols
        ], dtype=torch.float32)
        
        overall_score = torch.tensor(row['Overall'], dtype=torch.float32)
        
        # Get safety label
        safety_label = self._get_safety_label(row['Overall'])
        
        return {
            'image': image,
            'indicator_scores': indicator_scores,
            'overall_score': overall_score,
            'safety_label': safety_label,
            'image_name': row['Name']
        }


def get_transforms(img_size=224, mode='train'):
    """
    Get albumentations transforms
    
    Args:
        img_size: target size (will resize to square)
        mode: 'train' or 'val'
    """
    if mode == 'train':
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:  # val
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def create_dataloaders(
    train_excel, train_images,
    val_excel, val_images,
    batch_size=32,
    img_size=224,
    num_workers=2
):
    """
    Create train and validation dataloaders
    
    Returns:
        train_loader, val_loader
    """
    print("Creating dataloaders...")
    print("="*60)
    
    # Train dataset
    print("\nTRAIN:")
    train_dataset = MicrosturingDataset(
        excel_path=train_excel,
        image_dir=train_images,
        transform=get_transforms(img_size, mode='train')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Val dataset
    print("\n" + "="*60)
    print("\nVALIDATION:")
    val_dataset = MicrosturingDataset(
        excel_path=val_excel,
        image_dir=val_images,
        transform=get_transforms(img_size, mode='val')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("\n" + "="*60)
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    
    return train_loader, val_loader
