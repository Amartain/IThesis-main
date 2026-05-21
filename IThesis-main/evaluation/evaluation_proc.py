"""
The module contains functions responsible for evaluating PyTorch models.

This module contains functions for:
    - model evaluation on test dataset
    - calculating Dice Score on the test dataset and logging results and outputs to TensorBoard.
"""

import torch
from monai.metrics import DiceHelper

from training.training_proc import make_image_grid

def evaluate_model(model, device, test_loader, writer):
    """
    Evaluates model on the test dataset and logs metrics and visuals.

    Iterates over the `test_loader` and without gradients uses the model to make predictions on the dataset then evaluate them by calculating the Dice Score.
    Every batch's images are logged onto TensorBoard, and after the whole test dataset is iterated over the overall metrics (Dice Score) are saved to TensorBoard.

    **Args**: 
        `model` (`torch.nn.Module`): Model to be evaluated.
        `device` (`torch.device`): The device (CPU or cuda device (GPU)) to do the validation on.
        `test_loader` (`torch.utils.data.DataLoader`): The loader loading the test data. 
        `writer` (`SummaryWriter`): The TensorBoard writer object to log training metrics and results.

    **Returns**:
        `None`
    """

    running_dice = 0.0
    batch_count = 0
    total_samples = 0

    model.eval()


    with torch.no_grad():
        # No care for Thumbs and Labels
        for originals, skeletons, *_ in test_loader:
            x = originals.to(device)
            y = skeletons.to(device)

            output = model(x)

            # for tensorboard
            batch = (
                originals.cpu(),
                skeletons.cpu(),
                output.detach().cpu()
                )

            dice = DiceHelper(include_background=True, reduction='mean', get_not_nans=False, threshold=True)(output, y).item()

            running_dice += dice * len(x)
            total_samples += len(x)



            print("/"*80)
            print(f"Batch {batch_count}, Dice: {dice}")
            print("_"*80)

            # TensorBoard (visual)
            writer.add_image("Test Dataset", make_image_grid(batch), global_step=batch_count)

            batch_count += 1

    print("_"*80)
    print("TESTING FINISHED, logging results...")

    dice_score = running_dice / total_samples

    print(f"Final Dice For Dataset: {dice_score}")

    # TensorBoard (metrics)
    writer.add_scalar("Dice Score", dice_score)
    print("_"*80)

