from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.models.models import format_MaskRCNN

class VGCFModel(pl.LightningModule):

    def __init__(self, fiber_type:str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if fiber_type == "lenght": 
            ... 
        elif fiber_type == "width": 
            ... 
        
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT",
            trainable_backbone_layers=0
        )
        num_classes = 2 # background and object (fibers, a person, etc.)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
        )
    
    def forward(self, images, targets) -> Any:
        self.model.eval()
        with torch.no_grad(): 
            out = self.model(images, targets)
        return out
        
    def shared_step(self, batch): 
        images, targets = batch
        out:dict = self.model(images, targets)
        loss = sum(_loss for _loss in out.values())
        return loss
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.shared_step(batch)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.shared_step(batch)
        return loss

    def predict_step(self, images):
        self.eval()
        with torch.no_grad(): 
            self.model.eval()
            prediction = self.model(images)
        return prediction

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer

