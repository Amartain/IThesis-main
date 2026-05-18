"""
This module sets up training for the PyTorch models.

The module is **responsible** for:
    - loading the datasets,
    - initializing models and loss functions
    - setting up the optimizer (Adam)
    - running the training process
    - saving the trained model
"""

import torch

from data.pipeline import get_train_val_test_loaders
from configurations.conf import RANDOM_SEED
from configurations.selection import MODEL_MAP, LOSS_MAP
from training.training_proc import train_model

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")


def setup_training(dataset_choice, size_filter, batch_size, no_workers, model_choice, lr_choice, loss_choice, no_epochs, early_stop, writer, model_path):
    """
    Prepares environment and executes model training.

    This function:
        -  initializes `torch.manual_seed` with `RANDOM_SEED` configured in the `configurations.configure` script.
        -  sets up data loaders
        - instantiates chosen model and loss functions
        - sets up Adam optimizer for training
        - executes training
        - saves model weights

    **Args**: 
        `dataset_choice` (Enum): The `DatasetSelection` key for the dataset wished to be used.
        `size_filter` (int): Maximum dimension for the images used during training.
        `batch_size` (int): Batch size for data loading.
        `no_workers` (int): The number of workers (parallel threads) used by the data loaders.
        `model_choice` (Enum): The `ModelSelection` key for the model wished to be trained.
        `lr_choice` (float): The learning rate for the optimizer.
        `no_epochs` (int): Number of maximum training epochs.
        `early_stop` (int): Tolerance for early stop, counted in validation peridos. (Validation occurs every 5 epochs)    
        `writer` (SummaryWriter): The TensorBoard writer object to log training metrics and results.
        `model_path` (str): The path and the filename to which the saved model (state dict) will be saved.

    **Returns**:
        None
     """
    
    # Setting torch random state
    torch.manual_seed(RANDOM_SEED)

    print("Started in TRAINING MODE with the parameters: ")
    print(dataset_choice, size_filter, batch_size, no_workers, model_choice, loss_choice, no_epochs)
    print("_"*80)

    print("0. Setting up DataLoaders")
    print("_"*80)
    train_loader, val_loader, _ = get_train_val_test_loaders(dataset_choice, size_filter, batch_size, no_workers)

    print("1. Model Initialization")
    print("_"*80)
    model = MODEL_MAP[model_choice]()
    model.to(device)

    print("2. Setting up Loss & Optimizer")
    print("_"*80)
    loss_function = LOSS_MAP[loss_choice]()
    print(loss_function)

    optimizer = torch.optim.Adam(model.parameters(), lr_choice)

    print("3. STARTING TRAINING")
    print("_"*80)
    trained_model = train_model(model=model,device=device, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, loss_function=loss_function, writer=writer, no_epochs=no_epochs, early_stop=early_stop, batch_size=batch_size)
    
    print("4. SAVING MODEL")
    torch.save(trained_model.state_dict(), model_path)
