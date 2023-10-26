from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import CustomDataset, get_transform
from src.models.models import format_MaskRCNN
from src.engine import train_one_epoch, evaluate
import src.utils as utils
from src.utils.datasets import PennFundanDataset


# === training config === # 
DATA_DIR = Path('./data')
if DATA_DIR.exists() is False: 
    DATA_DIR.mkdir(parents=True)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2
NUM_EPOCHS = 5


# our dataset has two classes only - background and person
num_classes = NUM_CLASSES

# use our dataset and defined transformations
dataset_path = PennFundanDataset(location=DATA_DIR).get_path() 
dataset = CustomDataset(dataset_path, get_transform(train=True))
dataset_test = CustomDataset(dataset_path, get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
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
model = format_MaskRCNN(num_classes)

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
