import monai.losses

from enum import Enum, auto
from models.UNet import UNet
from models.RAUNet import RAUNet



class DatasetSelection(Enum):
    KIMIA99 = auto()
    KIMIA216 = auto()
    MPEG400 = auto()
    MPEG7 = auto() # 1400 images
    ANIMAL2000 = auto()

class ModelSelection(Enum):
    UNET = "U-Net"
    BUNET = "BatchNorm U-Net"
    RUNET = "Residual U-Net"
    AUNET = "Attention U-Net"
    RAUNET = "Residual Attention U-Net"

class LossSelection(Enum):
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
    LossSelection.FOCAL : lambda: monai.losses.FocalLoss(skip_this_function=True),
    LossSelection.DICEFOCAL : lambda: monai.losses.DiceFocalLoss(),
    LossSelection.CLDICE : lambda: monai.losses.SoftDiceclDiceLoss(alpha=0.5)
}