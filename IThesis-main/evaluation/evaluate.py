"""
The module responsible for loading saved PyTorch models and evaluating them it:
- loads previously saved PyTorch model (with state dicth - `.pth` file)
- initializes chosen dataset's test loaders or the test dataset loader based on given configuration
- starts evaluation process
"""

import torch

from data.paths import get_dataset_dirs
from configurations.selection import MODEL_MAP
from data.pipeline import get_train_val_test_loaders, get_test_dataset_loader
from evaluation.evaluation_proc import evaluate_model

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on {device}")


def setup_evaluation(model_path, model_choice, dataset_choice, test_dataset, size_filter, batch_size, no_workers, writer):
    """
    Prepares environment and executes model evaluation on loaded model.

    The function loads the model found the the model path given, sets up necc. DataLoaders depending on whether or not a test dataset is chosen, then gives control the `evaluate_model` function.

    **Args**:
        `model_path` (str): The path to which load the model's saved state dict. from. 
        `model_choice` (`Enum`): `ModelSelection` enum, that identifies the model's architecture.
        `dataset_choice` (`bool`): If `True` the whole dataset will be used for the evaluation, if `False` the evaluation will use the test portion of the dataset based on the standard train/set/val.
        `size_filter` (`int`): Maximum dimension for the images used during training.
        `batch_size` (`int`): Batch size for data loading.
        `no_workers` (`int`): The number of workers (parallel threads) used by the data loaders.
        `writer` (`SummaryWriter`): The TensorBoard writer object to log training metrics and results.

    **Returns**: 
        `None`
    """
    
    print("Evaluation Started With Parameters: ")
    print(dataset_choice, size_filter, batch_size, no_workers)

    print(f"Model To Be tested: {model_path}")


    print("0. INIT MODEL")
    print("_"*80)

    model = MODEL_MAP[model_choice]()
    model.to(device)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    

    print("2. Setting up DATALOADERs")
    print("_"*80)
    if test_dataset == 1:
        test_loader = get_test_dataset_loader(dataset_choice, batch_size, size_filter)
    else:
        *_, test_loader = get_train_val_test_loaders(dataset_choice=dataset_choice, batch_size=batch_size, size_filter=size_filter, no_workers=no_workers)
        print(f"Testing on {len(test_loader.dataset)} images.")
    

    print("3. STARTING TESTING")
    print("_"*80)
    evaluate_model(model=model, device=device, test_loader=test_loader, writer=writer)
    
    
    