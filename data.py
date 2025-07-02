import os
from torch.utils.data import Dataset, DataLoader, random_split
import shutil
import re 
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.notebook import tqdm  

def setup_kaggle(input_dir = "/kaggle/input/tyre-cv-data/augmented_renamed",output_dir = "/kaggle/working/classified_tires"):
    # Configuration
    copy_files = True  # Set to False to move files instead of copying

    # Define classes based on your balanced buckets image
    CLASSES = {
        '<=3': "3mm",  # Less than or equal to 3mm
        '4': "4mm",
        '5': "5mm",
        '6': "6mm",
        '7': "7mm",
        '8': "8mm"
    }

    # Regular expression to extract profile and side
    pattern = re.compile(r'.*?(?:LT=)?(\d+\.?\d*)_([lLrR])\.png$')

    # Create output directory structure
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # First pass to identify all unique profiles in the dataset
    found_profiles = set()
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            match = pattern.search(filename.lower())
            if match:
                profile = match.group(1)
                found_profiles.add(profile)

    print("Found profiles in dataset:", found_profiles)


    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process files
    for filename in tqdm(os.listdir(input_dir), desc="Processing files", unit=" file"):
        if filename.lower().endswith('.png'):
            match = pattern.search(filename.lower())
            if not match:
                print(f"Skipping: {filename}")
                continue
                
            profile = match.group(1)
            side = match.group(2).upper()  # Keep side info in filename
            
            # Determine class
            if float(profile) <= 3:
                class_name = CLASSES['<=3']
            elif profile in CLASSES:
                class_name = CLASSES[profile]
            else:
                print(f"Skipping unclassified profile: {filename}")
                continue
            
            # Create class directory
            class_dir = os.path.join(output_dir, class_name)
            Path(class_dir).mkdir(parents=True, exist_ok=True)
            
            # Copy/move while preserving original filename (with L/R info)
            src = os.path.join(input_dir, filename)
            dst = os.path.join(class_dir, filename)
            
            if copy_files:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)


class TirePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Group L/R pairs by their base name
            files = {}
            for fname in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}", unit=" file"):
                # Extract base name and side (L/R)
                match = re.match(r'(.*?)_([LR])\.png$', fname, re.IGNORECASE)
                if not match:
                    continue
                    
                base_name = match.group(1)
                side = match.group(2).upper()
                
                if base_name not in files:
                    files[base_name] = {}
                files[base_name][side] = os.path.join(class_dir, fname)
                
            # Create pairs
            for base_name, sides in tqdm(files.items(), desc=f"Creating pairs for {class_name}", unit=" pair"):
                if 'L' in sides and 'R' in sides:
                    samples.append({
                        'left': sides['L'],
                        'right': sides['R'],
                        'class': class_name,
                        'class_idx': self.class_to_idx[class_name]
                    })
                    
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        left_img = Image.open(sample['left'])
        right_img = Image.open(sample['right'])
        
        # Apply transforms
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        else:
            ToTensor=transforms.ToTensor()
            left_img = ToTensor(left_img)
            right_img = ToTensor(right_img)
            
        concat_img = torch.cat([left_img, right_img],2)   
        concat_img=concat_img.squeeze()
        return concat_img, sample['class']






def get_data_generators(split=0.8, batch_size=32, transform=None):
    if transform is None:
        # Default transform if none provided
        transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # v2.RandomResizedCrop(size=(1000, 1000), antialias=True),
        transforms.ToTensor(),
        ])
    dataset = TirePairDataset(
        root_dir='/kaggle/working/classified_tires',
        transform=transform
    )
    # Split into train/test
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )

    return train_loader, test_loader