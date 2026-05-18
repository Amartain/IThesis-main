"""
This module is responsible for the data loading and transformation pipeline.

It contains functions to:
    - compose transfomations
    - split dataset into train/test/validation sets
    - initialize dataloaders
"""
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torch import manual_seed, float32

from data.paths import get_dataset_dirs
from data.SkeView import SkeView
from configurations.conf import RANDOM_SEED



def get_transform(size_filter):
    """
    Creates and returns the image transform pipeline.

    Adds padding on the edges, then uses `CenterCrop` to ensure uniform sizes. Finally transforms the image into a float 32 tensor without scaling.

    **Args**:
        `size_filer` (`int`): The goal size of the image.
    
    **Returns**:
        `torchvision.transforms.Compose`: The object containing the transformation steps.
    
    """

    return transforms.Compose(
    [   
        v2.Pad(padding=size_filter, padding_mode='edge'), # makes sure to have same image padding before uniform size crop (because shape images have white background while ground truth have black)
        transforms.CenterCrop(size_filter),
        v2.Compose([v2.ToImage(), v2.ToDtype(float32, scale=False)]) # binary images don't need scaling!
    ]
    )

def train_val_test_split(dataset, test_friction=0.15, val_friction=0.15):
    """
    Devides dataset into training, validation and test sets.

    It devides using a fixed generator seed (from `configurations.conf` module) to ensure reproducibility.

    **Args**:
        `dataset` (`torch.utils.data.Dataset`): The dataset to be devided.
        `test_friction` (`float`, optional): The fraction of test dataset. Defaults to `0.15`.
        `val_friction` (`float`, optional): The fraction of validation dataset. Defaults to `0.15`.

    **Returns**: 3 item `tuple`
        - `train_dataset` (`torch.utils.data.Subset`): Training dataset.
        - `val_dataset` (`torch.utils.data.Subset`): Validation dataset.
        - `test_dataset` (`torch.utils.data.Subset`): Test dataset.
    """

    val_len = int(len(dataset) * val_friction)
    test_len = int(len(dataset) * test_friction)
    train_len = len(dataset) - val_len - test_len

    print(f"Train/Val/Test Split: {train_len}/{val_len}/{test_len}")
    
    generator = manual_seed(RANDOM_SEED)
    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_len, val_len, test_len], generator=generator)

    return train_dataset, val_dataset, test_dataset



def get_train_val_test_loaders(dataset_choice, size_filter, batch_size, no_workers):
    """
    Instantiates DataLoader objects for training, validation and testing.

    Gets the necc. directory paths, initializes the SkeView dataset, executes train/val/test split, then prepares dataloaders. 

    **Args**:
        `dataset_choice` (`DatasetSelection`): The chosen dataset's identifier.
        `size_filter` (`int`): The maximum dimension of images. The transformations will result in size_filter x size_filter images!
            **Warning**: *The dataset WILL BE FILTERED based on this dimension. If any dimension of a given images is larger than the given filter, than it will not be used.*
        `batch_size` (`int`): Batch size for loading the data.
        `no_workers` (`int`): The number of workers for the data loaders.

    **Returns**:
        `tuple`: 3 item `tuple` with `DataLoader` objects:
            - `train_loader` (`torch.utils.data.DataLoader`): Loader for training dataset.
            - `val_loader` (`torch.utils.data.DataLoader`): Loader for validation dataset.
            - `test_loader` (`torch.utils.data.DataLoader`): Loader for testing dataset.

    """

    print("Getting loaders", "."*70)

    original_dir, gt_dir, thumb_dir = get_dataset_dirs(dataset_choice)

    transform = get_transform(size_filter)


    dataset = SkeView(original_dir, gt_dir, thumb_dir, size_filter, transform)


    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset=dataset)


    generator = manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=no_workers, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=no_workers, persistent_workers=True, pin_memory=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=no_workers, persistent_workers=True, pin_memory=True)  

    print("DataLoaders ready", "."*70)

    return train_loader, val_loader, test_loader


def get_test_dataset_loader(dataset_choice, batch_size,size_filter):
    """
    Creates the DataLoader objects for testing with a whole dataset.

    **Warning**: Only use this function on datasets the model hasn't seen yet! This will not use the train/val/test split! Instead the whole dataset is given to the model for evaluation!
    
    **Args**:
    `dataset_choice` (`DatasetSelection`): The chosen dataset's identifier.
    `batch_size` (`int`): Batch size for loading the data.
    `size_filter` (`int`): The maximum dimension of images. The transformations will result in size_filter x size_filter images!
        **Warning**: *The dataset WILL BE FILTERED based on this dimension. If any dimension of a given images is larger than the given filter, than it will not be used.*
        **Warning**: **THIS MUST MATCH SIZE THE MODEL WAS TRAINED WITH!**  
    Returns:
        `torch.utils.data.DataLoader`: Loader for the test dataset.
    
    """


    print("Getting loader", "."*70)

    original_dir, gt_dir, thumb_dir = get_dataset_dirs(dataset_choice)

    transform = get_transform(size_filter)

    test_dataset = SkeView(original_dir, gt_dir, thumb_dir, size_filter, transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)  

    return test_loader