from __future__ import annotations
from typing import TypeVar, Generic, Union, cast

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
import numpy as np
from numpy.typing import NDArray

from ..data import CustomDataset


# Little helper class, which is only used as a type.
DType = TypeVar("DType")
class NPArray(np.ndarray, Generic[DType]):
    def __getitem__(self, key) -> DType:
        return super().__getitem__(key) # type: ignore

def quickview(
        dataset:CustomDataset,
        idx:int, mask_idx:int=0,
        fig:Figure=None, 
        axs:Union[NPArray[Axes], Axes]=None
    ): 
    if fig is None and axs is None: 
        fig, axs = plt.subplots(1, 2)
    
    img, target = dataset[idx]
    masks = target['masks']

    for i, obj in enumerate([img, masks]): 
        ax = axs[i]
        ax.imshow(obj.numpy()[0])
    return fig, ax 