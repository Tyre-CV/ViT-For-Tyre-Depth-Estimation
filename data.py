from collections import defaultdict
import os
import random
from torch.utils.data import Dataset, DataLoader, random_split
import shutil
import re 
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.notebook import tqdm  
from torchvision.transforms import v2 as T

def setup_kaggle(input_dir = "/kaggle/input/tyre-cv-data/augmented_renamed",output_dir = "/kaggle/working/classified_tires"):
    """
    Organizes and classifies tire images from the input directory into class-based folders in the output directory.
    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where classified images will be stored.
    """
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



def get_default_transform(
    p=0.2,
    posterize_bits=4,
    solarize_thresh=128,
    sharpness_factor=2.0,
    blur_kernel=5,
    blur_sigma=(0.1, 2.0),
):
    """
    Returns a torchvision transform pipeline with various augmentations for tire images.
    Args:
        p (float): Probability for each random transform.
        posterize_bits (int): Bits for posterization.
        solarize_thresh (int): Threshold for solarization.
        sharpness_factor (float): Factor for sharpness adjustment.
        blur_kernel (int): Kernel size for Gaussian blur.
        blur_sigma (tuple): Sigma range for Gaussian blur.
    Returns:
        torchvision.transforms.Compose: Composed transform.
    """
    return transforms.Compose([
        # v2 random effects
        T.RandomPosterize(bits=posterize_bits, p=p),
        T.RandomSolarize(threshold=solarize_thresh, p=p),
        T.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=p),
        T.RandomAutocontrast(p=p),
        T.RandomEqualize(p=p),
        # ColorJitter on grayscale->RGB->grayscale
        transforms.RandomApply([
            transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.0, hue=0.0),
                transforms.Grayscale(num_output_channels=1),
            ])
        ], p=p),
        # Gaussian blur on PIL
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)], p=p),
        # Convert to tensor
        transforms.ToTensor(),
    ])

class TirePairDataset(Dataset):
    """
    PyTorch Dataset for loading paired left and right tire images from a directory structure.
    Each sample is a concatenated (side-by-side) grayscale image and its class index.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing class folders with images.
            transform (callable, optional): Transform to apply to the concatenated image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir)) 
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        """
        Prepares paired left and right images for each class.
        Returns:
            list: List of sample dicts with left, right image paths and class index.
        """
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            files = {}
            for fname in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}", unit="file"):
                match = re.match(r'(.*?)_([LR])\.png$', fname, re.IGNORECASE)
                if not match:
                    continue
                base_name, side = match.group(1), match.group(2).upper()
                files.setdefault(base_name, {})[side] = os.path.join(class_dir, fname)
            for base_name, sides in tqdm(files.items(), desc=f"Creating pairs for {class_name}", unit="pair"):
                if 'L' in sides and 'R' in sides:
                    samples.append({
                        'left': sides['L'],
                        'right': sides['R'],
                        'class_idx': self.class_to_idx[class_name]
                    })
        return samples

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a sample consisting of a concatenated left-right image and its class index.
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (image tensor, class index)
        """
        sample = self.samples[idx]
        left_img = Image.open(sample['left']).convert('L')
        right_img = Image.open(sample['right']).convert('L')
        # Concatenate side-by-side in PIL
        w, h = left_img.size
        combined = Image.new('L', (w * 2, h))
        combined.paste(left_img, (0, 0))
        combined.paste(right_img, (w, 0))
        # Apply transforms
        if self.transform:
            img = self.transform(combined)
        else:
            img = transforms.ToTensor()(combined)

        return img, sample['class_idx']

def get_file_name(file_path):
    """
    Extracts the file name without extension from a file path.
    Args:
        file_path (str): Path to the file.
    Returns:
        str: File name without extension.
    """
    base_name = os.path.basename(file_path)
    name, _ = os.path.splitext(base_name)
    return name

def get_info(file_path):
    """
    Extracts id, label, and side information from a file path.
    Args:
        file_path (str): Path to the file.
    Returns:
        dict: Dictionary with keys 'id', 'side', and 'label'.
    """
    file_name = get_file_name(file_path)
    parts = file_name.split('_')
    
    if len(parts) < 3:
        raise ValueError(f"File name '{file_name}' does not contain enough parts to extract id, side, and label.")
    
    id_part = parts[0]
    label_part = parts[1]
    side_part = parts[2].upper()
    # label needs to be passed from string to float
    try:
        label_part = label_part
    except ValueError:
        raise ValueError(f"Label '{label_part}' in file name '{file_name}' is not a valid float.")
    
    return dict(
        id=id_part,
        side=side_part,
        label=label_part
    )

def group_stereo(img_paths):
    """
    Groups stereo image pairs (left and right) by label.
    Args:
        img_paths (str): Directory containing images.
    Returns:
        dict: Dictionary mapping label to list of stereo pairs.
    """
    file_names = os.listdir(img_paths)
    file_names = [os.path.join(img_paths, f) for f in file_names]

    groups = defaultdict(list)
    for path in file_names:
        info = get_info(path)
        id = info['id']
        label = info['label']
        side = info['side']
        # One side is sufficient
        if side == 'L':
            opposite_side = 'R'
            groups[label].append({
                'left': path, 
                'right': os.path.join(os.path.dirname(path), f"{id}_{label}_{opposite_side}.png"), 
                'label': label
            })
    
    return groups
        

class TyrePairDataset(Dataset):
    """
    PyTorch Dataset for loading stereo (left-right) tire image pairs from a directory.
    Each sample is a concatenated grayscale image and its label.
    """
    def __init__(self, data_dir, transform=None, sample=False, sampling_size=None):
        """
        Args:
            data_dir (str): Directory containing tire images.
            transform (callable, optional): Transform to apply to the concatenated image.
            sample (bool): Whether to sample randomly from the dataset.
            sampling_size (int, optional): Number of samples to draw if sampling.
        """
        super().__init__()
        self.transform = transform
        if sample and sampling_size is None:
            raise ValueError("If 'sample' is True, 'sampling_size' must be provided.")
        if sample and sampling_size is not None:
            self.sampling_size = sampling_size
        self.sample = sample
        self.grouped_paths = group_stereo(data_dir)  
        # For indexing
        self.samples = [item for values in self.grouped_paths.values() for item in values]
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        if self.sample:
            return self.sampling_size
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a sample consisting of a concatenated left-right image and its label.
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (image tensor, label)
        """
        if self.sample:
            sample = random.choice(self.samples)  # Randomly select a sample
        else:
            sample = self.samples[idx]

        left_img = Image.open(sample['left']).convert('L')
        right_img = Image.open(sample['right']).convert('L')
        # Concatenate side-by-side in PIL
        w, h = left_img.size
        combined = Image.new('L', (w * 2, h))
        combined.paste(left_img, (0, 0))
        combined.paste(right_img, (w, 0))
        # Apply transforms
        if self.transform:
            img = self.transform(combined)
        else:
            img = transforms.ToTensor()(combined)

        # Squeeze the image to remove the channel dimension
        img = img.squeeze(0)  # Assuming the image is grayscale, this will remove the channel dimension

        return img, sample['label']




def get_data_generators(
    train_dir,
    test_dir,
    batch_size=32,
    transform=None,
    shuffle_train=True,
    num_workers=4,
    p=0.05, # can either be one value for both or a tuple (p_train, p_test)
    sampling_size=None 
):
    """
    Creates PyTorch DataLoader generators for training and testing datasets.
    Args:
        train_dir (str): Directory for training data.
        test_dir (str): Directory for test data.
        batch_size (int): Batch size for DataLoader.
        transform (callable, optional): Transform to apply to images.
        shuffle_train (bool): Whether to shuffle the training data.
        num_workers (int): Number of worker processes for data loading.
        p (float or tuple): Probability for data augmentation transforms.
        sampling_size (int, optional): Number of samples to draw if sampling.
    Returns:
        tuple: (train_loader, test_loader)
    """
    if transform is None:
        if isinstance(p, tuple):
            transform= [get_default_transform(p[0]), get_default_transform(p[1])]
        else:
            transform = [get_default_transform(p=p) for _ in range(2)]
    
    if sampling_size is not None:
        # If sampling size is provided, set sample to True
        train_dataset = TyrePairDataset(data_dir=train_dir, transform=transform[0], sample=True, sampling_size=sampling_size)
        test_dataset = TyrePairDataset(data_dir=test_dir, transform=transform[1], sample=True, sampling_size=int(sampling_size * 0.2))  # Assuming you want to sample 20% of the training size for testing
    else:
        # If no sampling size is provided, use the full dataset
        train_dataset = TyrePairDataset(data_dir=train_dir, transform=transform[0])
        test_dataset = TyrePairDataset(data_dir=test_dir, transform=transform[1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
