from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader

from src.data import CustomDataset, get_transform, split_data, subset_debug
from src.models.models import format_MaskRCNN
from src.engine import train_one_epoch, evaluate
import src.utils as utils
from src.utils.datasets import PennFundanDatasetLoader
from prerun import prerun

# === training config === #
DATA_DIR = Path('./data')
if DATA_DIR.exists() is False: 
    DATA_DIR.mkdir(parents=True)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2
NUM_EPOCHS = 1

def train(debug:bool=False): 
    # use our dataset and defined transformations
    dataset_path = PennFundanDatasetLoader(location=DATA_DIR).get_path() 
    dataset = CustomDataset(dataset_path, get_transform(train=True))
    dataset_test = CustomDataset(dataset_path, get_transform(train=False))

    # split the dataset in train and test set
    if debug: 
        dataset = subset_debug(dataset)
        NUM_EPOCHS = 1 
    dataset_train, dataset_test = split_data(dataset, test_size=0.2)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn
    )

    # get the model using our helper function
    model = format_MaskRCNN(NUM_CLASSES)

    # move model to the right device
    model.to(DEVICE)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    for epoch in range(NUM_EPOCHS):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, DEVICE, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=DEVICE)

    # save model
    logs = Path('./logs'); logs.mkdir(parents=True, exist_ok=True)
    torch.save(model, logs / 'model.pth')



if __name__ == '__main__':
    # runline tags
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", '--debug', action="store_true", default=False)
    args = parser.parse_args()

    prerun()

    if args.debug: 
        print('\n--WARNING: training in `debug` mode--\n')

    train(debug=args.debug)