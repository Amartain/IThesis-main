import torch

from data.pipeline import get_train_val_test_loaders
from configurations.conf import RANDOM_SEED
from configurations.selection import MODEL_MAP, LOSS_MAP
from training.training_proc import train_model

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")


def setup_training(dataset_choice, size_filter, batch_size, no_workers, model_choice, lr_choice, loss_choice, no_epochs, early_stop, writer, model_path):
    torch.manual_seed(RANDOM_SEED)

    print("Started in TRAINING MODE with the parameters: ")
    print(dataset_choice, size_filter, batch_size, no_workers, model_choice, loss_choice, no_epochs)
    print("_"*80)

    # Initialize Dataset
    print("0. Setting up DataLoaders")
    print("_"*80)
    train_loader, val_loader, _ = get_train_val_test_loaders(dataset_choice, size_filter, batch_size, no_workers)

    # Initialize model
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
