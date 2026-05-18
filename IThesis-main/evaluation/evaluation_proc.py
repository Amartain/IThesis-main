import torch
from monai.metrics import DiceHelper

from training.training_proc import make_image_grid

def evaluate_model(model, device, test_loader, writer):
    running_dice = 0.0
    batch_count = 0
    total_samples = 0

    model.eval()


    with torch.no_grad():
        # I will only care about thumbs and labels when doing visual analysis...
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

            dice = DiceHelper(include_background=True, reduction='mean', get_not_nans=False, sigmoid=True)(output, y).item()

            running_dice += dice * len(x)
            total_samples += len(x)



            print("/"*80)
            print(f"Batch {batch_count}, Dice: {dice}")
            print("_"*80)

            # visual stuff to tensorboard
            writer.add_image("Test Dataset", make_image_grid(batch), global_step=batch_count)

            batch_count += 1

    print("_"*80)
    print("TESTING FINISHED, logging results...")

    dice_score = running_dice / total_samples

    print(f"Final Dice For Dataset: {dice_score}")

    writer.add_scalar("Dice Score", dice_score)
    print("_"*80)

