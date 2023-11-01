#%% 
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from src.models.lit import VGCFModel
from src.data import get_transform

ckp_path = "./tmp/checkpoints/epoch=4-step=340.ckpt"
model = VGCFModel.load_from_checkpoint(ckp_path)

image = read_image("./data/PennFudanPed/PNGImages/FudanPed00001.png")
eval_transform = get_transform(train=False)
x = eval_transform(image)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with torch.no_grad():
    x = eval_transform(image)
    x = x[:3, ...].to(device) # convert RGBA -> RGB and move to device
    predictions = model.predict_step([x, ])
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