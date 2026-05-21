"""
Module containing functions responsible for training and validating PyTorch models.

This module contains the functions for:
    - main training loop
    - training and validation epochs
    - image grid makers for TensorBoard logging
"""

import torch
from torchvision.utils import make_grid
from monai.metrics import DiceHelper
import monai.losses

from configurations.selection import LossSelection


def training_epoch(model, device, loss_function, optimizer, train_loader):
    """
    Executes single training epoch on the given model.

    The function iterates over the training loader (`train_loader`), executes the forward pass, calculates loss and Dice score, backpropagation and optimizer step. Collects the first batch's data for visualization and all the epoch's data for logging.

    **Args**: 
        `model` (`torch.nn.Module`): Neural network to be trained.
        `device` (`torch.device`): The device (CPU or Cuda (GPU)) on which to run the training.
        `loss_function` (callable): The function to calculate the loss between the target tensor and model output.
        `train_loader` (`torch.utils.data.DataLoader`): The dataloader loading the training data.

    **Returns**: 
        `epoch_loss` (`float`): Avarage epoch loss.
        `epoch_dice` (`float`): Avarage epoch Dice Score.
        `first_batch` (`tuple`): The first batch's data (original image, skeleton ground truth, model prediction) for visualization. 
    """
    
    epoch_loss = 0
    running_loss = 0.0
    running_dice = 0.0

    # prev. epoch might end wtih .eval() mode if it was a validation epoch  
    model.train()

    
    # used for passing up first batch for visualization!
    batch_count = 0
    first_batch = None

    for originals, skeletons, *_ in train_loader:
        # data setup
        x = originals.to(device)
        y = skeletons.to(device)

        # forward pass
        optimizer.zero_grad()

        output = model(x)

        # backward pass
        loss = loss_function(output, y)

        # metrics & for tensorboard
        if batch_count == 0:
            first_batch = (
               originals.cpu(),
               skeletons.cpu(),
               output.detach().cpu()
            )

        dice_score = DiceHelper(include_background=True, reduction='mean', get_not_nans=False, threshold=True)(output, y).item()

        running_dice += dice_score * len(x)

        loss.backward()

        optimizer.step()

        # unequal batch sizes
        running_loss += loss.item() * len(x) 

        batch_count += 1

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_dice = running_dice / len(train_loader.dataset)

    return epoch_loss, epoch_dice, first_batch

def val_epoch(model, device, loss_function, val_loader):
    """
    Executes single validation epoch on the given model.

    The function after setting the model into evaluation mode it iterates over the validation data loader (`val_loader`), and without calculating gradients calculates the validation loss and Dice Score. 
    The goal of the validation epoch is to measure the model's generalization ability during training. 
    
    **Args**: 
    `model` (`torch.nn.Module`): Model to be validated.
    `device` (`torch.device`): The device (CPU or cuda device (GPU)) to do the validation on.
    `loss_function` (`callable`): The function to calculate the loss between the model's outputs and the target.
    `val_loader` (`torch.utils.data.DataLoader`): The loader loading the validation data. 
    
    **Returns**: 3 element `tuple`:
        - `epoch_val_loss` (`float`): The epoch's average validation loss.
        - `epoch_dice` (`float`): The epoch's average validation Dice Score.
        - `first_batch` (`tuple`): The data of the first validation batch (original image, skeleton ground truth, prediction) for visualization.
    """
    running_loss = 0.0
    running_dice = 0.0
    batch_count = 0


    model.eval()


    with torch.no_grad():
        # During this phase we don't need labels, or thumbs
        for originals, skeletons, *_ in val_loader:
            x = originals.to(device)
            y = skeletons.to(device)

            output = model(x)


            if  isinstance(loss_function, (monai.losses.SoftclDiceLoss, monai.losses.SoftDiceclDiceLoss)):
                output_ohe = torch.cat((1-output, output), dim=1)
                y_ohe = torch.cat((1-y, y), dim=1)
                loss = loss_function(output_ohe, y_ohe)
            else:
                loss = loss_function(output, y)

            # for tensorboard
            if batch_count == 0:
                first_batch = (
                originals.cpu(),
                skeletons.cpu(),
                output.detach().cpu()
                )

            dice = DiceHelper(include_background=True, get_not_nans=False, threshold=True)(output, y).item()
            running_dice += dice * len(x)

            running_loss += loss.item() * len(x)

    epoch_val_loss = running_loss / len(val_loader.dataset)
    epoch_dice = running_dice / len(val_loader.dataset)


    return epoch_val_loss, epoch_dice, first_batch

def make_image_grid(image_batch):
    """
    Concatanotes and makes a grid of the images for TensorBoard visualization.

    Concatonates the original image, the skeleton ground truth and the model output into a single image channel through dim 3. 

    **Args**: 
        `image_batch` (`tuple`): A tuple composing of three tensors (original image, skeleton ground truth, model output).
        **Warning**: The tensors must already be on the CPU!
    
    **Returns**:
        `torch.Tensor`: Single tensor image organized into a grid using the `torchvision` `make_grid` function.

    """
    imgs, gt_imgs, ypreds = image_batch
    
    # concetonating images in the width dimension 
    cat_images = torch.cat((imgs, gt_imgs, ypreds), dim=3)
    
    return make_grid(cat_images, nrow=1)
    


def train_model(model, device,  optimizer, train_loader, val_loader, loss_function, writer,  no_epochs, early_stop, batch_size):
    """
    Executes full training loop with intra training validation on the given model.

    This function is responsible for managing the training process.
        -  Every epoch it runs the training epoch.
        - Every 5 epochs runs a validation epoch. 
        - Manages the early stopping conditions to avoid overfitting.
        - Logs metrics and image data to TensorBoard.

    **Args**:
        `model` (`torch.nn.Module`): Model to be validated.
        `device` (`torch.device`): The device (CPU or cuda device (GPU)) to do the validation on.
        `optimizer` (`torch.optim.Optimizer`): The optimizer responsible for updating model weights.
        `train_loader` (`torch.utils.data.DataLoader`): The dataloader loading the training data.
        `val_loader` (`torch.utils.data.DataLoader`): The loader loading the validation data. 
        `loss_function` (`callable`): The function to calculate the loss between the model's outputs and the target.
        `writer` (`SummaryWriter`): The TensorBoard writer object to log training metrics and results.
        `no_epochs` (`int`): Number of maximum training epochs.
        `early_stop` (`int`): Tolerance for early stop, counted in validation peridos. (Validation occurs every 5 epochs)
        `batch_size` (`int`): Batch size for data loading.

    **Returns**:
        `torch.nn.Module`: The trained model.
    """    
    
    step = 0
    stop = 0
    best_val_acc = 0


    image, *_ = next(iter(train_loader))
    image = image.to(device)

    writer.add_graph(model, image)

    for epoch in range(no_epochs):
        if stop == early_stop:
            print(f"Early Stop Triggered At {epoch}/{no_epochs} with validation acc of {best_val_acc}")
            
            break

        epoch_train_loss, epoch_train_dice, train_1st_batch = training_epoch(model, device, loss_function, optimizer, train_loader)

        if epoch % 5 == 0 or (epoch+1) == no_epochs:
            epoch_val_loss, epoch_val_dice, val_1st_batch = val_epoch(model,device, loss_function, val_loader)


            if epoch < 20:
                if epoch_val_dice > best_val_acc: best_val_acc = epoch_val_dice
            elif epoch_val_dice > best_val_acc:
                print(f"resetting early stop to 0 at {epoch}/{no_epochs}")
                best_val_acc = epoch_val_dice
                stop = 0
            else:
                stop += 1
                print(f"early stop +1 = {stop}")
        
            print("Epoch ", epoch, "/", no_epochs)
            print("Training Loss: ", epoch_train_loss)
            print("Validation Loss: ", epoch_val_loss)
            print("/"*80)
            print("Training Dice Score", epoch_train_dice)
            print("Validation Dice Score", epoch_val_dice)
            print("_"*80)

            # TensorBoard (visuals)
            writer.add_image("Training", make_image_grid(train_1st_batch), global_step=step)
            writer.add_image("Validation", make_image_grid(val_1st_batch), global_step=step)
            


        # TensorBoard (metrics)
        writer.add_scalars("Train vs Validation Loss", {"Training Loss": epoch_train_loss, "Validation Loss": epoch_val_loss}, global_step=step)
        writer.add_scalars("Training vs Validation DICE", {"Training Dice":epoch_train_dice, "Validation Dice":epoch_val_dice}, global_step=step)

        
        print(f"{epoch}/{no_epochs} done..................................")

        step += 1
    
    print("Training finished, logging results...")


    writer.add_hparams(
        {"batch_size":batch_size}, 
        {"Training loss":epoch_train_loss,
        "Validation Loss":epoch_val_loss, 
        "Training Accuracy F1":epoch_train_dice,
        "Validation Accuracy F1":epoch_val_dice
        })
    
    return model
