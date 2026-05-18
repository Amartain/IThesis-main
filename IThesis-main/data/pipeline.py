from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch import manual_seed, float32

from data.paths import get_dataset_dirs
from data.SkeView import SkeView
from configurations.conf import RANDOM_SEED



def get_transform(size_filter):

    return transforms.Compose(
    [   
        v2.Pad(padding=size_filter, padding_mode='edge'), # makes sure to have same image padding before uniform size crop (because shape images have white background while ground truth have black)
        transforms.CenterCrop(size_filter),
        v2.Compose([v2.ToImage(), v2.ToDtype(float32, scale=False)]) # binary images don't need scaling!
    ]
    )

def train_val_test_split(dataset, test_friction=0.15, val_friction=0.15):
    """
    IN: Dataset class, friction sizes of test and validation datasets
    OUT: read test, validation and train datasets!
    """

    val_len = int(len(dataset) * val_friction)
    test_len = int(len(dataset) * test_friction)
    train_len = len(dataset) - val_len - test_len

    print(f"Train/Val/Test Split: {train_len}/{val_len}/{test_len}")
    
    generator = manual_seed(RANDOM_SEED)
    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_len, val_len, test_len], generator=generator)

    return train_dataset, val_dataset, test_dataset



def get_train_val_test_loaders(dataset_choice, size_filter, batch_size, no_workers):
    print("Getting loaders", "."*70)

    original_dir, gt_dir, thumb_dir = get_dataset_dirs(dataset_choice)

    transform = get_transform(size_filter)


    dataset = SkeView(original_dir, gt_dir, thumb_dir, size_filter, transform)


    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset=dataset)


    generator = manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=no_workers, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=no_workers, persistent_workers=True, pin_memory=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)  

    print("DataLoaders ready", "."*70)

    return train_loader, val_loader, test_loader


def get_test_dataset_loader(dataset_choice, batch_size,size_filter):
    print("Getting loader", "."*70)

    original_dir, gt_dir, thumb_dir = get_dataset_dirs(dataset_choice)

    transform = get_transform(size_filter)

    test_dataset = SkeView(original_dir, gt_dir, thumb_dir, size_filter, transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)  

    return test_loader