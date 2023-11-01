from pathlib import Path
import argparse

import torch
from pytorch_lightning import Trainer

from src.data import CustomDataset, get_transform, split_data, subset_debug
import src.utils as utils
from src.utils.datasets import PennFundanDatasetLoader
from src.models.lit import VGCFModel
from prerun import prerun

# === training config === #
DATA_DIR = Path('./data')
if DATA_DIR.exists() is False: 
    DATA_DIR.mkdir(parents=True)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2
NUM_EPOCHS = 1
DEBUG = True

def train(num_epochs:int=NUM_EPOCHS, debug:bool=False): 
    """Run training pipeline."""
    num_epochs = int(num_epochs)
    print(f'{num_epochs=}')
    
    dataset_path = PennFundanDatasetLoader(location=DATA_DIR).get_path() 
    dataset = CustomDataset(dataset_path, get_transform(train=True))

    # split the dataset in train and test set
    if debug: 
        dataset = subset_debug(dataset)
        num_epochs = 1
    dataset_train, dataset_test = split_data(dataset, test_size=0.2)

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn
    )

    # get the model using our helper function
    model = VGCFModel(fiber_type='width')

    # get traininer
    trainer = Trainer(
        log_every_n_steps=1,
        max_epochs=num_epochs,
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )
    trainer.fit(model=model, train_dataloaders=data_loader_train)

if __name__ == '__main__':
    # get CLI tags
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", '--debug', action="store_true", default=False)
    parser.add_argument('-n', '--num-epochs')
    args = parser.parse_args()

    prerun()

    print(args.num_epochs)

    if args.debug: 
        print('\n--WARNING: training in `debug` mode--\n')

    train(num_epochs=args.num_epochs, debug=args.debug) 