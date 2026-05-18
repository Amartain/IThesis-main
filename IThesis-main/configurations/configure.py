

import argparse
from torch.utils.tensorboard import SummaryWriter

import configurations.selection as selection
from training.train import setup_training
from evaluation.evaluate import setup_evaluation

# data 
BATCH_SIZE = 4
NO_WORKERS = 12
SIZE_FILTER = 160
DATASET = selection.DatasetSelection.KIMIA99
# model
LOSS = selection.LossSelection.DICECE

# training
NO_EPOCHS = 200
EARLY_STOP = 3
LR_RATE = 1e-4

# logging
TRAINING_LOGS_PATH =  'runs/training_logs'
TEST_LOGS_PATH =  'runs/test_logs'


parser = argparse.ArgumentParser()

parser.add_argument('-e', '--evaluation', type=str, help='Set to evaluation mode with path to trained and saved model.')
parser.add_argument('-m', '--model', choices=['UNET', 'BUNET', 'RUNET', 'AUNET', 'RAUNET'], required=True, help='Define model to be used. MUST be defined.')
parser.add_argument('-ds', '--dataset', choices=['KIMIA99', 'KIMIA216', 'MPEG400', 'MPEG7', 'ANIMAL2000', 'SWLEAF'], default=DATASET.name, help='Define the dataset to be used. Defaults to Kimai99.')
parser.add_argument('--no_workers', type=int, default=NO_WORKERS ,help='Define max number of workers for the data loaders. Defaults to 4.')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Define batch size(s). Defaults to 4.')
parser.add_argument('--size_filter', type=int, default=SIZE_FILTER, help='Max image dimension on any side. Defaults to 160 for Kimia, for other datasets higher resolution is recommended.')
parser.add_argument('--loss', default=LOSS.name, choices=['DICE', 'FOCAL', 'DICECE','DICEFOCAL', 'CLDICE'], help='Define loss function to be used for training, defaults to DiceCE loss.')
parser.add_argument('--no_epochs', type=int, default=NO_EPOCHS ,help='Define max number epochs. Defaults to 20.')
parser.add_argument('--early_stop', type=int, default=EARLY_STOP ,help='Define max number epoch tolerance for no improvement in validation accuracy. Defaults to 3. Validation done every 5 epochs and thus 3 means 15 epoch tolerance.')
parser.add_argument('-lr', '--learning_rate', type=float, default=LR_RATE, help='Define learning rate for Adam optimizer. Default 1e-4')
parser.add_argument('--test_dataset', nargs=1, type=bool, default=False,help='Set dataset mode to test only, meaning the whole dataset will be used for evaluation! Use only for unseen datasets!')


args = parser.parse_args()

if args.loss == 'CLDICE' or args.loss == 'FOCAL':
    import sys
    print("Exit due to CLDICE / FOCAL loss use.")
    sys.exit(0)

def start():
    # training mode


    if args.evaluation is None:
        print("Training Mode")
        
        # Configuration
        # data
        dataset_choice = selection.DatasetSelection[args.dataset]
        size_filter = args.size_filter
        batch_size = args.batch_size
        no_workers = args.no_workers
        # model
        model_choice = selection.ModelSelection[args.model]
        loss_choice = selection.LossSelection[args.loss]
        # training 
        no_epochs = args.no_epochs
        early_stop = args.early_stop
        lr_choice = args.learning_rate

        writer = SummaryWriter(f"{TRAINING_LOGS_PATH}/{model_choice.name}/{dataset_choice.name}_{size_filter}/{loss_choice.name}")
        model_save_path = f'models/saved/{model_choice.name}_{loss_choice.name}_{dataset_choice.name}-{size_filter}_{no_epochs}epochs.pth'

        

        setup_training(dataset_choice, size_filter, batch_size, no_workers, model_choice,lr_choice, loss_choice, no_epochs, early_stop, writer, model_save_path)



    else:
        print("Evaluation mode")

        # Configuration
        # data
        dataset_choice = selection.DatasetSelection[args.dataset]
        test_dataset = args.test_dataset
        size_filter = args.size_filter
        batch_size = args.batch_size
        no_workers = args.no_workers
        
        # model
        model_choice = selection.ModelSelection[args.model]
        model_path = args.evaluation

        writer = SummaryWriter(f"{TEST_LOGS_PATH}/{model_choice.name}/{dataset_choice.name}_size_filter")

        setup_evaluation(model_path, model_choice, dataset_choice, test_dataset, size_filter, batch_size, no_workers, writer)



