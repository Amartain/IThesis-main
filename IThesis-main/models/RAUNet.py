"""
Module for defining the RAU-Net (Residual Attention U-Net) architecture.

    This modole contains the blocks needed to define a U-Net based RAU-Net architecture, 
    including Attention Gate, Residual Blocks, Encoder and Decoder Blocks.
     To solve segmentation problems like skeletonization. 
"""
import torch
from torch import manual_seed, nn, cat
import torch.nn.functional as F


# MODEL SETUP
STRIDE = 2
POOL_TRANSPOSE_KERNEL_SIZE = (2,2)
KERNEL_SIZE = (3,3)
PADDING = "same"
OUT_CHANNELS = 32 # doubled w/ every down! this is where we start

class AttentionGate(nn.Module):
    """
    Attention Gate mechanism for the decoder path.

        This layer helps find the relevant regions from the skip connection feature maps
        by creating an attention map "alpha" to use as a filter before concatanation in the decoder blocks.
        """

    def __init__(self, input_channels=1, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING):
        super().__init__()

        self.conv_x = nn.Conv2d(in_channels=input_channels,out_channels=out_channels,kernel_size=1, padding=PADDING)      
        self.conv_fm = nn.Conv2d(in_channels=input_channels,out_channels=out_channels,kernel_size=1, padding=PADDING)      
        self.conv_psi = nn.Conv2d(in_channels=input_channels,out_channels=1,kernel_size=1, padding=PADDING)
        

    def forward(self, x, fm):
        """
            Executes forward path through the attention gate.
           
            **Args**:
                `x` (`torch.Tensor`): Course resolution output from deeper layers of the network.
                `fm` (`torch.Tensor`): Higher resolution feature maps from the encoder path.

            **Returns**:
                `torch.Tensor`: Alpha (attention map), which contains relevancy weights.
        """
        alpha = torch.add(self.conv_x(x), self.conv_fm(fm))
        alpha = F.relu(alpha)
        alpha = self.conv_psi(alpha)
        alpha = torch.sigmoid(alpha)

        return alpha
    


class ResidualBlock(nn.Module):
    """
        Residual convolutional block with 2D convolutions. 
            Contains 2 convolutional blocks with Batch Normalization
            and 1x1 convolution for the identiy / shortcut connection.
    """
    def __init__(self, input_channels=1, out_channels=OUT_CHANNELS,kernel_size=KERNEL_SIZE, padding=PADDING):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=kernel_size, 
                padding=padding
                ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()      
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size, 
                padding=padding
                ),
            nn.BatchNorm2d(out_channels),   
        )
        self.conv1x1 = nn.Conv2d(in_channels=input_channels,out_channels=out_channels,kernel_size=1, padding=PADDING)

    def forward(self, x):
        """
            Executes forward pass throught the residual block.
            **Args**:
                `x` (`torch.Tensor`): Input tensor.

            **Returns**:
                `torch.Tensor`: Output tensor after residual shortcut connection is added and ReLU is applied.
            """
        identity = self.conv1x1(x)
        x = self.conv_block(x)
        x = self.conv_block2(x)
        x = torch.add(x, identity)
        x = F.relu(x)
        
        return x


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
                nn.BatchNorm2d(out_channels),
                nn.ReLU()      
            )
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size, 
                    padding=padding
                    ),
                nn.BatchNorm2d(out_channels),
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
    """ 
    Block for the encoder path.

        Made out of a convolutional block - either standard CNN Block or a Residual Block
        and a max pooling layer which cuts the dimensions in half.
    """
    def __init__(self, input_channels=1, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, residual=False):
        super().__init__()
        
        if residual:
            self.down_block = ResidualBlock(input_channels,out_channels, kernel_size, padding)
        else:
            self.down_block = CNNBlock(input_channels,out_channels, kernel_size, padding)
        self.maxpool = nn.MaxPool2d(kernel_size=POOL_TRANSPOSE_KERNEL_SIZE, stride=STRIDE)

    def forward(self, x):
        """
        Executes forward pass through Encoder Block.

            **Args**:
                `x` (`torch.Tensor`): A bemeneti tenzor.

            **Returns**:
                `tuple`: 2 item tuple containing:
                    - `x` (`torch.Tensor`): Maxpooled input for the next layer.
                    - `feature_map` (`torch.Tensor`): Feature map before pooling for the skip connections.
            """
        feature_map = self.down_block(x)
        x = self.maxpool(feature_map)

        return  x, feature_map


class DecoderBlock(nn.Module):
    """
    The Block of Decoder path.

        Uses transposed convolution for the dubbling of dimensions, 
        based on it's initialization parameters uses Attention Gates on the skip connections from the encoder path,
        and uses either CNN or Residual Blocks based on initialization mode.
        """
    def __init__(self, input_channels, out_channels, kernel_size=KERNEL_SIZE, padding=PADDING, residual=False, attention=False):
        super().__init__()
                # out channels SAME cause - this output will be CONCAT w/ feature map >> double >> ...
        self.up_conv = nn.ConvTranspose2d(in_channels=input_channels, out_channels=out_channels, kernel_size=POOL_TRANSPOSE_KERNEL_SIZE, stride=STRIDE)
        # input channels remain the same cause CONCAT >> 2xout_channels => input_channels again!
        self.attention = attention
        if self.attention:
            self.attention_gate = AttentionGate(input_channels=out_channels, out_channels=out_channels)
        if residual:
            self.conv_block = ResidualBlock(input_channels, out_channels)
        else: 
            self.conv_block = CNNBlock(input_channels, out_channels)

# - Connecting paths
#      - concatanation that's it just cat... meow
#         - cat places convoluted image at that stage ALONGSIDE the decoded features!

    def forward(self, x, feature_map):
        """
        Executes forward pass throuhg decoder block. 
        If the model is inited as:
        - residual: residual blocks are used for the convolutional block 
        - attention: attention gates are used on the skip connection featuremap
            
            **Args**:
                `x` (`torch.Tensor`): Coarse resolution output tensor from deeper layers.
                `feature_map` (`torch.Tensor`): Higher resolution feature map from the encoder path.

            **Returns**:
                `torch.Tensor`: Output tensor.
        """
        x = self.up_conv(x)

        if self.attention:
            feature_map = torch.multiply(self.attention_gate(x, feature_map), feature_map)

        x = cat((feature_map,x),dim=1)
        x = self.conv_block(x) 

        return x



class RAUNet(nn.Module):
    """
    Residual Attention U-Net (RAUNet) for binary segmentation. 
        The architecture consist of 4 encoders, a bridge block, 4 decoders and a prediction layer with Sigmoid.

        Based on the configuration it uses:
         - residual shortcut connections - residual blocks
         - attention mechanism - attention gates 
        """
    def __init__(self, residual=True, attention=True):
        super().__init__()
        input_channels = 1
        kernel_size = KERNEL_SIZE
        padding = PADDING

        # Encoder / Down 
        # defaul start in: 1 -> 32 -> 64 -> 128 -> 256
        # DOUBLEs out_channels every block
        self.encoder_block1 = EncoderBlock(input_channels,OUT_CHANNELS, residual=residual)
        self.encoder_block2 = EncoderBlock(input_channels=OUT_CHANNELS, out_channels=OUT_CHANNELS*2, residual=residual)
        self.encoder_block3 = EncoderBlock(input_channels=OUT_CHANNELS*2, out_channels=OUT_CHANNELS*4, residual=residual)
        self.encoder_block4 = EncoderBlock(input_channels=OUT_CHANNELS*4, out_channels=OUT_CHANNELS*8, residual=residual)

        # Bridge
        # - Bottleneck / Bridge - no Pool
        # in channels default - 256 -> 512
        self.bridge = CNNBlock(input_channels=OUT_CHANNELS*8, out_channels=OUT_CHANNELS*16, kernel_size=kernel_size, padding=padding)

        # Decoder / Up
        # start channel default 512 -> 256 -> 128 -> 64 -> 32
        # HALVES start channel every block
        self.decoder_block1 = DecoderBlock(input_channels=OUT_CHANNELS*16, out_channels=OUT_CHANNELS*8,  residual=residual, attention=attention)
        self.decoder_block2 = DecoderBlock(input_channels=OUT_CHANNELS*8, out_channels=OUT_CHANNELS*4,  residual=residual, attention=attention)
        self.decoder_block3 = DecoderBlock(input_channels=OUT_CHANNELS*4, out_channels=OUT_CHANNELS*2,  residual=residual, attention=attention)
        self.decoder_block4 = DecoderBlock(input_channels=OUT_CHANNELS*2, out_channels=OUT_CHANNELS,  residual=residual, attention=attention)

        # Final Prediction layer
        self.prediction = nn.Sequential(
            nn.Conv2d(in_channels=OUT_CHANNELS, out_channels=1, kernel_size=(1,1),padding=PADDING),
            nn.Sigmoid()
        )

# let PyTorch __call__ handle the magic apperantly 
    def forward(self, x):
        """
        Executes forward pass through RAU-Net network.
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
