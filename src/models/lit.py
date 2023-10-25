from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch

from src.models import format_MaskRCNN

class VGCFModel(pl.LightningModule):

    def __init__(self, fiber_type:str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if fiber_type is "lenght": 
            ... 
        elif fiber_type is "width": 
            ... 
        
        self.model = format_MaskRCNN(num_classes=2)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.shared_step(batch)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.shared_step(batch)
        return loss

    def shared_step(self, batch): 
        images, targets = batch
        out:dict = self(images, targets)
        loss = sum(_loss for _loss in out.values())
        return loss
        
    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer
        

