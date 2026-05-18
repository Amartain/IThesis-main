"""
Module for defining the  classic U-Net architecture.

    This modole contains the blocks needed to define a U-Net architecture, 
    including Encoder and Decoder Blocks.
    To solve segmentation problems like skeletonization. 

    **Important diversion from original U-Net paper**: `same` padding is used for easier tensor shape management!
"""
import torch
from torch import manual_seed, nn, cat


# MODEL SETUP
STRIDE = 2
POOL_TRANSPOSE_KERNEL_SIZE = (2,2)
KERNEL_SIZE = (3,3)
PADDING = "same"
OUT_CHANNELS = 32 # doubled w/ every down! this is where we start!!!!!


class CNNBlock(nn.Module):
    """
    Standard double convolutional block with two layers of consequtive convolutional blocks.

        Both conv blocks are followed by `BatchNorm2d` and `ReLU` for training stability.
    """
    
    def __init__(self, input_channels=1, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING):
        super().__init__()
    
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=kernel_size, 
                padding=padding
                ),
            nn.ReLU()      
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size, 
                padding=padding
                ),
            nn.ReLU()   
        )


    def forward(self, x):
        """
         Executes forward path through the convolutional block.
        **Args**:
            `x` (`torch.Tensor`): Input tensor.

        **Returns**:
            `torch.Tensor`: Output tensor.
        """
        x = self.conv_block(x)
        x = self.conv_block2(x)
        return x



class EncoderBlock(nn.Module):
    """ Block for the encoder path.

        Made out of double convolutional blocks (CNN Block)
        and a max pooling layer which cuts the dimensions in half.
    """
    def __init__(self, input_channels=1, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING):
        super().__init__()
        
        self.down_block = CNNBlock(input_channels,out_channels, kernel_size, padding)
        self.maxpool = nn.MaxPool2d(kernel_size=POOL_TRANSPOSE_KERNEL_SIZE, stride=STRIDE)

    def forward(self, x):
        feature_map = self.down_block(x)
        x = self.maxpool(feature_map)

        return  x, feature_map




# - Decoder - Upsampling - Mode for skeletons gotta be: nearest neighbour not bilinear!???
#     - UPSAMPLE (convtranspose)  & 1/2x channels  >> CONV BLOCK >> UPSAMPLE >> next conv block...
#     - Upsampling via: ConvTranspose 


class DecoderBlock(nn.Module):
    """
    The Block of Decoder path.

    Uses transposed convolution for the dubbling of dimensions, skip connections from the encoder path,
    and CNN Blocks.
    """
    def __init__(self, input_channels, out_channels, kernel_size=KERNEL_SIZE, padding=PADDING):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=input_channels, out_channels=out_channels, kernel_size=POOL_TRANSPOSE_KERNEL_SIZE, stride=STRIDE)
        self.conv_block = CNNBlock(input_channels, out_channels, kernel_size, padding)

# - Connecting paths
#      - concatanation that's it just cat... meow
#         - cat places convoluted image at that stage ALONGSIDE the decoded features!

    def forward(self, x, feature_map):
        """
        Executes forward pass throuhg decoder block.
            
            **Args**:
                `x` (`torch.Tensor`): Coarse resolution output tensor from deeper layers.
                `feature_map` (`torch.Tensor`): Higher resolution feature map from the encoder path.

            **Returns**:
                `torch.Tensor`: Output tensor.
        """

        x = self.up_conv(x)
        x = cat((feature_map,x),dim=1)
        x = self.conv_block(x) 

        return x



class UNet(nn.Module):
    """
    U-Net (RAUNet) for binary segmentation. 
        The architecture consist of 4 encoders, a bridge block, 4 decoders and a prediction layer with Sigmoid.

    **Important diversion from original U-Net paper**: `same` padding is used for easier tensor shape management!
    """

    def __init__(self, input_channels=1, kernel_size=KERNEL_SIZE, padding=PADDING):
        super().__init__()

        # Encoder / Down 
        # defaul start in: 1 -> 32 -> 64 -> 128 -> 256
        # DOUBLEs out_channels every block!
        self.encoder_block1 = EncoderBlock(input_channels,OUT_CHANNELS)
        self.encoder_block2 = EncoderBlock(input_channels=OUT_CHANNELS, out_channels=OUT_CHANNELS*2)
        self.encoder_block3 = EncoderBlock(input_channels=OUT_CHANNELS*2, out_channels=OUT_CHANNELS*4)
        self.encoder_block4 = EncoderBlock(input_channels=OUT_CHANNELS*4, out_channels=OUT_CHANNELS*8)

        # Bridge
        # - Bottleneck / Bridge - no Pool
        # in channels default -256 -> 512
        self.bridge = CNNBlock(input_channels=OUT_CHANNELS*8, out_channels=OUT_CHANNELS*16, kernel_size=kernel_size, padding=padding)

        # Decoder / Up
        # start channel default 512 -> 256 -> 128 -> 64 -> 32
        # HALVES start channel every block
        self.decoder_block1 = DecoderBlock(input_channels=OUT_CHANNELS*16, out_channels=OUT_CHANNELS*8)
        self.decoder_block2 = DecoderBlock(input_channels=OUT_CHANNELS*8, out_channels=OUT_CHANNELS*4)
        self.decoder_block3 = DecoderBlock(input_channels=OUT_CHANNELS*4, out_channels=OUT_CHANNELS*2)
        self.decoder_block4 = DecoderBlock(input_channels=OUT_CHANNELS*2, out_channels=OUT_CHANNELS)

        # Final Prediction layer
        self.prediction = nn.Sequential(
            nn.Conv2d(in_channels=OUT_CHANNELS, out_channels=1, kernel_size=(1,1),padding=PADDING),
            nn.Sigmoid()
        )

# let PyTorch __call__ handle the magic apperantly 
    def forward(self, x):
        """
        Executes forward pass through U-Net network.
            **Args**:
                `x` (`torch.Tensor`): Input image tensor (1 channel, binary image).

            **Returns**:
                `torch.Tensor`: The predicted binary segmentation map (probability values between 0-1!) because Sigmoid is applied!.
        """

        # Encoder / Down
        x, feature_map_1 = self.encoder_block1(x)
        x, feature_map_2 = self.encoder_block2(x)
        x, feature_map_3 = self.encoder_block3(x)
        x, feature_map_4 = self.encoder_block4(x)

        # Bridge
        x = self.bridge(x)

        # Decoder / Up
        x = self.decoder_block1(x, feature_map_4)
        x = self.decoder_block2(x, feature_map_3)
        x = self.decoder_block3(x, feature_map_2)
        x = self.decoder_block4(x, feature_map_1)

        # final prediction w/ sigmoid
        x = self.prediction(x)

        return x
