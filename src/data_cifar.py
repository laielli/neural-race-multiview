"""
CIFAR-100 data loading with hierarchical (superclass) labels.

CIFAR-100 has natural hierarchical structure:
- 100 fine classes grouped into 20 superclasses
- Each superclass has 5 fine classes

This provides real-world validation of the hierarchical KD findings.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np


# CIFAR-100 superclass mapping (20 superclasses, 5 fine classes each)
CIFAR100_SUPERCLASS_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
    'trees', 'vehicles_1', 'vehicles_2'
]

# Mapping from fine class index to superclass index
# CIFAR-100 fine classes are organized such that fine_class // 5 = superclass
# But the actual mapping is more complex - we'll compute it from coarse labels
FINE_TO_COARSE = None  # Will be computed from dataset


def get_cifar100_mapping(trainset):
    """
    Extract the fine-to-coarse mapping from CIFAR-100 dataset.

    CIFAR-100's targets are fine labels, but it also has coarse_targets (superclasses).
    """
    fine_to_coarse = {}

    # CIFAR-100 stores coarse targets
    if hasattr(trainset, 'targets') and hasattr(trainset, 'coarse_targets'):
        for fine, coarse in zip(trainset.targets, trainset.coarse_targets):
            if fine not in fine_to_coarse:
                fine_to_coarse[fine] = coarse
    else:
        # Fallback: CIFAR-100 classes are ordered so that consecutive 5 belong to same superclass
        # This is approximate - the actual torchvision dataset has the true mapping
        for fine in range(100):
            fine_to_coarse[fine] = fine // 5

    return fine_to_coarse


class CIFAR100Hierarchical(Dataset):
    """
    CIFAR-100 dataset with both fine (100-class) and coarse (20-class) labels.

    Args:
        root: Data directory
        train: If True, use training set
        transform: Image transformations
        download: If True, download dataset
    """

    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        transform=None,
        download: bool = True
    ):
        # Load CIFAR-100
        self.cifar = torchvision.datasets.CIFAR100(
            root=root, train=train, download=download, transform=None
        )

        self.transform = transform
        self.train = train

        # Get labels
        self.fine_labels = np.array(self.cifar.targets)

        # CIFAR-100 has coarse_targets attribute for superclass labels
        if hasattr(self.cifar, 'coarse_targets'):
            self.coarse_labels = np.array(self.cifar.coarse_targets)
        else:
            # Compute from fine labels (they're ordered in groups of 5)
            self.coarse_labels = self.fine_labels // 5

        # Store images
        self.data = self.cifar.data

        # Class counts
        self.n_fine = 100
        self.n_coarse = 20
        self.n_per_superclass = 5

        # Build fine-to-coarse mapping
        self.fine_to_coarse = {}
        for fine, coarse in zip(self.fine_labels, self.coarse_labels):
            if fine not in self.fine_to_coarse:
                self.fine_to_coarse[fine] = coarse

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            fine_label: Fine class label (0-99)
            coarse_label: Superclass label (0-19)
        """
        img = self.data[idx]
        fine_label = self.fine_labels[idx]
        coarse_label = self.coarse_labels[idx]

        # Convert to PIL for transforms
        from PIL import Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, fine_label, coarse_label


def get_cifar100_transforms(train: bool = True, normalize: bool = True):
    """
    Get standard CIFAR-100 transforms.

    Args:
        train: If True, include data augmentation
        normalize: If True, normalize with CIFAR-100 mean/std
    """
    transform_list = []

    if train:
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    transform_list.append(transforms.ToTensor())

    if normalize:
        # CIFAR-100 normalization
        transform_list.append(
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        )

    return transforms.Compose(transform_list)


def get_cifar100_dataloaders(
    root: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True
):
    """
    Get CIFAR-100 dataloaders with hierarchical labels.

    Returns:
        train_loader: Training dataloader
        test_loader: Test dataloader
        dataset_info: Dict with class counts and mappings
    """
    train_transform = get_cifar100_transforms(train=augment)
    test_transform = get_cifar100_transforms(train=False)

    train_dataset = CIFAR100Hierarchical(
        root=root, train=True, transform=train_transform, download=True
    )
    test_dataset = CIFAR100Hierarchical(
        root=root, train=False, transform=test_transform, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    dataset_info = {
        'n_fine': train_dataset.n_fine,
        'n_coarse': train_dataset.n_coarse,
        'n_per_superclass': train_dataset.n_per_superclass,
        'fine_to_coarse': train_dataset.fine_to_coarse,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
    }

    return train_loader, test_loader, dataset_info


def collapse_soft_labels(soft_probs, fine_to_coarse, n_coarse=20):
    """
    Collapse fine-grained (100-class) soft labels to coarse (20-class).

    Args:
        soft_probs: Tensor of shape (batch, 100) - probabilities over fine classes
        fine_to_coarse: Dict mapping fine class index to coarse class index
        n_coarse: Number of coarse classes

    Returns:
        Tensor of shape (batch, 20) - probabilities over coarse classes
    """
    batch_size = soft_probs.shape[0]
    coarse_probs = torch.zeros(batch_size, n_coarse, device=soft_probs.device)

    for fine_idx in range(100):
        coarse_idx = fine_to_coarse[fine_idx]
        coarse_probs[:, coarse_idx] += soft_probs[:, fine_idx]

    return coarse_probs


def analyze_soft_label_structure(teacher, dataloader, temperature, fine_to_coarse, device='cuda'):
    """
    Analyze information content in teacher's soft labels.

    Similar to the synthetic experiment analysis - measures:
    - Entropy of collapsed soft labels
    - True superclass probability
    - Confusion within superclass (related fine classes)

    Args:
        teacher: Trained teacher model (100-class output)
        dataloader: Test dataloader
        temperature: Softmax temperature
        fine_to_coarse: Mapping from fine to coarse labels
        device: Device to use

    Returns:
        Dict with analysis metrics
    """
    import torch.nn.functional as F

    teacher.eval()

    all_entropies = []
    all_true_probs = []
    all_confusion_same_super = []

    with torch.no_grad():
        for images, fine_labels, coarse_labels in dataloader:
            images = images.to(device)
            fine_labels = fine_labels.to(device)
            coarse_labels = coarse_labels.to(device)

            # Get teacher logits and soft probs
            logits = teacher(images)
            soft_probs = F.softmax(logits / temperature, dim=-1)

            # Collapse to coarse
            coarse_probs = collapse_soft_labels(soft_probs, fine_to_coarse)

            # Entropy of coarse soft labels
            entropy = -torch.sum(coarse_probs * torch.log(coarse_probs + 1e-10), dim=-1)
            all_entropies.extend(entropy.cpu().numpy())

            # True superclass probability
            true_prob = coarse_probs[torch.arange(len(coarse_labels)), coarse_labels]
            all_true_probs.extend(true_prob.cpu().numpy())

            # Confusion within same superclass (other fine classes in same super)
            for i in range(len(fine_labels)):
                fine_true = fine_labels[i].item()
                coarse_true = coarse_labels[i].item()

                # Find other fine classes in same superclass
                same_super_mask = torch.zeros(100, dtype=torch.bool)
                for f, c in fine_to_coarse.items():
                    if c == coarse_true and f != fine_true:
                        same_super_mask[f] = True

                confusion = soft_probs[i, same_super_mask].sum().item()
                all_confusion_same_super.append(confusion)

    return {
        'avg_entropy': np.mean(all_entropies),
        'max_entropy': np.log(20),  # 20 superclasses
        'avg_true_superclass_prob': np.mean(all_true_probs),
        'avg_same_superclass_confusion': np.mean(all_confusion_same_super),
        'n_samples': len(all_entropies)
    }


if __name__ == '__main__':
    # Test data loading
    print("Testing CIFAR-100 hierarchical data loading...")

    train_loader, test_loader, info = get_cifar100_dataloaders(
        batch_size=128, num_workers=0
    )

    print(f"\nDataset info:")
    print(f"  Fine classes: {info['n_fine']}")
    print(f"  Coarse classes: {info['n_coarse']}")
    print(f"  Fine per coarse: {info['n_per_superclass']}")
    print(f"  Train size: {info['train_size']}")
    print(f"  Test size: {info['test_size']}")

    # Check a batch
    images, fine_labels, coarse_labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Fine labels: {fine_labels.shape}, range [{fine_labels.min()}, {fine_labels.max()}]")
    print(f"  Coarse labels: {coarse_labels.shape}, range [{coarse_labels.min()}, {coarse_labels.max()}]")

    # Verify mapping
    print(f"\nVerifying fine-to-coarse mapping (first 10):")
    for i in range(10):
        f, c = fine_labels[i].item(), coarse_labels[i].item()
        mapped_c = info['fine_to_coarse'][f]
        match = "✓" if c == mapped_c else "✗"
        print(f"  {match} fine={f} -> coarse={c} (mapped={mapped_c})")

    print("\nData loading test complete!")
