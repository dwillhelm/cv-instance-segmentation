#%% 
from os import PathLike
from typing import Union

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from src.data import get_transform
from src.models.models import format_MaskRCNN


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def tensor_to_pil_img(tensor): 
    transform = T.ToPILImage()
    return transform(tensor)

def load_model(checkpoint_path:Union[str, PathLike]):
    """Load a serialized torch model from a PT/PTH file."""
    model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return model

image = read_image("./data/PennFudanPed/PNGImages/FudanPed00002.png")

eval_transform = get_transform(train=False)

x = eval_transform(image)


# model = load_model(checkpoint_path='./logs/model.pth')

try: 
    model = torch.load('./logs/model.pth')
    model.eval()

except:
    model = format_MaskRCNN(num_classes=2) 
    model.load_state_dict(torch.load('./logs/model_state_dict', map_location=device))
    model.eval()

with torch.no_grad():
    x = eval_transform(image)
    x = x[:3, ...].to(device) # convert RGBA -> RGB and move to device
    predictions = model([x, ])
    pred = predictions[0]

image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...] # convert RGBA -> RGB
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))
# %%
