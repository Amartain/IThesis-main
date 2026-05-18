"""
Configuration module for choosing the parameters for deep learning experiments.

Ez a modul felsorolásokat (Enum) tartalmaz az adathalmazok, modellek és 
veszteségfüggvények kiválasztásához, valamint leképezéseket (dictionary), 
amelyek ezeket az enumerációs értékeket a megfelelő osztályok példányosítását 
végző paraméter nélküli (lambda) függvényekhez rendelik.

The module contains:
    - `Enum`s for datasets, models and loss function selection
    - maps (`dictionary`) that use these enums to map these to lambda functions that instantiate these classes.
"""

import monai.losses

from enum import Enum, auto
from models.UNet import UNet
from models.RAUNet import RAUNet



class DatasetSelection(Enum):
    """
    Enumeration of the available datasets.

    This class defines the identifiers for the different datasets that can be used for training and evaluation.
    Using `auto()` python automatically assigns unique value to these enums.
    """   
    KIMIA99 = auto()
    KIMIA216 = auto()
    MPEG400 = auto()
    MPEG7 = auto() # 1400 images
    ANIMAL2000 = auto()

class ModelSelection(Enum):
    """
    Enumeration of the available models.

    This class defines the identifiers for the different models that can be used for training and evaluation.
    """ 
    UNET = "U-Net"
    BUNET = "BatchNorm U-Net"
    RUNET = "Residual U-Net"
    AUNET = "Attention U-Net"
    RAUNET = "Residual Attention U-Net"

class LossSelection(Enum):
    """
    Enumeration of the available loss functions.

    This class defines the identifiers for the different loss functions that can be used for training and evaluation.
    """  
    DICE = "DiceLoss"
    DICECE = "DiceCELoss (1 x BCE + 1 x Dice)"
    FOCAL = "FocalLoss"
    DICEFOCAL = "DiceFocalLoss"
    CLDICE = "clDICE"


MODEL_MAP = {
    ModelSelection.UNET : lambda: UNet(),
    ModelSelection.BUNET : lambda: RAUNet(residual=False, attention=False),
    ModelSelection.AUNET : lambda: RAUNet(residual=False),
    ModelSelection.RUNET : lambda: RAUNet(attention=False),
    ModelSelection.RAUNET : lambda: RAUNet()
}

LOSS_MAP = {
    LossSelection.DICE : lambda: monai.losses.DiceLoss(),
    LossSelection.DICECE : lambda: monai.losses.DiceCELoss(),
    LossSelection.FOCAL : lambda: monai.losses.FocalLoss(),
    LossSelection.DICEFOCAL : lambda: monai.losses.DiceFocalLoss(),
    LossSelection.CLDICE : lambda: monai.losses.SoftDiceclDiceLoss()
}