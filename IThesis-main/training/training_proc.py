import torch
from torchvision.utils import make_grid
from monai.metrics import DiceHelper




def training_epoch(model, device, loss_function, optimizer, train_loader):
    epoch_loss = 0
    running_loss = 0.0
    # running_f1 = 0.0
    running_dice = 0.0

    # prev. epoch ends w/ .eval() mode because of the validation epoch so reset
    model.train()

    


    # I return thumbs and labels too but I actually don't care about it during training!
     # later TODO: - make monitoring by labels see which label is accessed how many times! - maybe graph distributions?
    y_distr = []
    
    # used for passing up first batch for visualization!
    batch_count = 0
    first_batch = None

    for originals, skeletons, *_ in train_loader:
        # data setup
        x = originals.to(device)
        y = skeletons.to(device)
       # print("x min max", x.min(), x.max())
      #  print("y min max", y.min(), y.max())
        # 1.s vs. 0.s
        *_, val_counts = y.unique(return_counts=True) 
        
        count_0, count_1 = val_counts[0].item() / len(x), val_counts[1].item() / len(x)
       # print("VALCOUNTS: 0., 1.", count_0, count_1)
        distr = count_1 / count_0
        y_distr.append(distr)

        # forward pass
        optimizer.zero_grad()
      #  print("X shape", x.size())

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

        # calculating f1 score for the batch
        dice_score = DiceHelper(include_background=True, reduction='mean', get_not_nans=False, sigmoid=True)(output, y).item()
        # target = y.round().long()
        # tp, fp, fn, tn = metrics.get_stats(output, target, mode='binary', threshold=0.5)
        # f1 = metrics.f1_score(tp, fp, fn, tn, reduction='micro').item()
        # running_f1 += f1 * len(x)
        running_dice += dice_score * len(x)

        loss.backward()

        optimizer.step()

        # Accumulative LOSS
        running_loss += loss.item() * len(x) # avarage out w/ batch_size to ensure same weight for all

        batch_count += 1

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_dice = running_dice / len(train_loader.dataset)

    return epoch_loss, epoch_dice, first_batch

def val_epoch(model, device, loss_function, val_loader):
    running_loss = 0.0
    #running_f1 = 0.0
    running_dice = 0.0
    batch_count = 0
    # set model to eval!
    model.eval()


    # no need for grad.
    with torch.no_grad():
        # I will only care about thumbs and labels when doing visual analysis...
        for originals, skeletons, *_ in val_loader:
            x = originals.to(device)
            y = skeletons.to(device)

            output = model(x)
            
            loss = loss_function(output, y)

                # for tensorboard
            if batch_count == 0:
                first_batch = (
                originals.cpu(),
                skeletons.cpu(),
                output.detach().cpu()
                )

            # calculating batch f1 score
            dice = DiceHelper(include_background=True, get_not_nans=False, sigmoid=True)(output, y).item()

            # target = y.round().long()
            # tp, fp, fn, tn = metrics.get_stats(output, target, mode='binary', threshold=0.5)
            # f1 = metrics.f1_score(tp, fp, fn, tn, reduction='micro').item()
            # running_f1 += f1 * len(x)
            running_dice += dice * len(x)


            running_loss += loss.item() * len(x)


    epoch_val_loss = running_loss / len(val_loader.dataset)
    epoch_dice = running_dice / len(val_loader.dataset)


    return epoch_val_loss, epoch_dice, first_batch

def make_image_grid(image_batch):
    """
    image_batch - tuple (original image, ground_truth_image, model_output) ON CPU already!
    """
    imgs, gt_imgs, ypreds = image_batch
    
    cat_images = torch.cat((imgs, gt_imgs, ypreds), dim=3)
    
    return make_grid(cat_images, nrow=1)
    


def train_model(model, device,  optimizer, train_loader, val_loader, loss_function, writer,  no_epochs, early_stop, batch_size):
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

            # visual stuff to tensorboard
            writer.add_image("Training", make_image_grid(train_1st_batch), global_step=step)
            writer.add_image("Validation", make_image_grid(val_1st_batch), global_step=step)
            


        # TensorBoard
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
