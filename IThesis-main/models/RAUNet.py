import torch
from torch import manual_seed, nn, cat
import torch.nn.functional as F


# MODEL SETUP
STRIDE = 2
POOL_TRANSPOSE_KERNEL_SIZE = (2,2)
KERNEL_SIZE = (3,3)
PADDING = "same"
OUT_CHANNELS = 32 # doubled w/ every down! this is where we start!!!!!

class AttentionGate(nn.Module):

    def __init__(self, input_channels=1, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING):
        super().__init__()

        self.conv_x = nn.Conv2d(in_channels=input_channels,out_channels=out_channels,kernel_size=1, padding=PADDING)      
        self.conv_fm = nn.Conv2d(in_channels=input_channels,out_channels=out_channels,kernel_size=1, padding=PADDING)      
        self.conv_psi = nn.Conv2d(in_channels=input_channels,out_channels=1,kernel_size=1, padding=PADDING)
        

    def forward(self, x, fm):
        """
        IN:
            x = coarse output of layers
            fm = feature_map from previous levels, high res, but already scaled down in decoder
        RETURNS: 
            alpha = attention map
        """
        alpha = torch.add(self.conv_x(x), self.conv_fm(fm))
        alpha = F.relu(alpha)
        alpha = self.conv_psi(alpha)
        alpha = torch.sigmoid(alpha)

        return alpha
    


class ResidualBlock(nn.Module):
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
        identity = self.conv1x1(x)
        x = self.conv_block(x)
        x = self.conv_block2(x)
        x = torch.add(x, identity)
        x = F.relu(x)
        
        return x






# ## Model Architecture [9]
# - Encoder - Downsampling
#     - CONV BLOCK >> DOWNSAMPLE (maxpool) & 2x channels >> next conv block >> down...
#     - saves convoluted output BEFORE maxpool every time because we will do skip connections with it !
#     - 2x CHANNELS after each conv block 
#     - PADDING: make so it remains the SAME SIZE! - saves headache

# TODO: add residual trigger

class CNNBlock(nn.Module):
    
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
            x = self.conv_block(x)
            x = self.conv_block2(x)
            return x



class EncoderBlock(nn.Module):
    def __init__(self, input_channels=1, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, residual=False):
        super().__init__()
        
        if residual:
            self.down_block = ResidualBlock(input_channels,out_channels, kernel_size, padding)
        else:
            self.down_block = CNNBlock(input_channels,out_channels, kernel_size, padding)
        self.maxpool = nn.MaxPool2d(kernel_size=POOL_TRANSPOSE_KERNEL_SIZE, stride=STRIDE)

    def forward(self, x):
        feature_map = self.down_block(x)
        x = self.maxpool(feature_map)

        return  x, feature_map




# - Decoder - Upsampling - Mode for skeletons gotta be: nearest neighbour not bilinear!???
#     - UPSAMPLE (convtranspose)  & 1/2x channels  >> CONV BLOCK >> UPSAMPLE >> next conv block...
#     - Upsampling via: ConvTranspose 

# TODO: add attention trigger
class DecoderBlock(nn.Module):
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
        x = self.up_conv(x)

        if self.attention:
            feature_map = torch.multiply(self.attention_gate(x, feature_map), feature_map)

        x = cat((feature_map,x),dim=1)
        x = self.conv_block(x) 

        return x



class RAUNet(nn.Module):
    def __init__(self, residual=True, attention=True):
        super().__init__()
        input_channels = 1
        kernel_size = KERNEL_SIZE
        padding = PADDING

        # Encoder / Down 
        # defaul start in: 1 then 16 > 32 > 64 > 128
        # DOUBLE out_channels every block!
        self.encoder_block1 = EncoderBlock(input_channels,OUT_CHANNELS, residual=residual)
        self.encoder_block2 = EncoderBlock(input_channels=OUT_CHANNELS, out_channels=OUT_CHANNELS*2, residual=residual)
        self.encoder_block3 = EncoderBlock(input_channels=OUT_CHANNELS*2, out_channels=OUT_CHANNELS*4, residual=residual)
        self.encoder_block4 = EncoderBlock(input_channels=OUT_CHANNELS*4, out_channels=OUT_CHANNELS*8, residual=residual)

        # Bridge
        # - Bottleneck / Bridge - no Pool
#     - so regular ass convolution w/o pool - so just use CNNBlock class
        # in channels default - 128 > 256
        self.bridge = CNNBlock(input_channels=OUT_CHANNELS*8, out_channels=OUT_CHANNELS*16, kernel_size=kernel_size, padding=padding)

        # Decoder / Up
        # start channel default 256 > 128 > 64 > 32 > 16
        # HALF start channel every block
        self.decoder_block1 = DecoderBlock(input_channels=OUT_CHANNELS*16, out_channels=OUT_CHANNELS*8,  residual=residual, attention=attention)
        self.decoder_block2 = DecoderBlock(input_channels=OUT_CHANNELS*8, out_channels=OUT_CHANNELS*4,  residual=residual, attention=attention)
        self.decoder_block3 = DecoderBlock(input_channels=OUT_CHANNELS*4, out_channels=OUT_CHANNELS*2,  residual=residual, attention=attention)
        self.decoder_block4 = DecoderBlock(input_channels=OUT_CHANNELS*2, out_channels=OUT_CHANNELS,  residual=residual, attention=attention)

        # Final Prediction layer
        # - OUT: convolution final time w/o w/ SIGMOID for binary image segmentation
        self.prediction = nn.Sequential(
            nn.Conv2d(in_channels=OUT_CHANNELS, out_channels=1, kernel_size=(1,1),padding=PADDING),
            nn.Sigmoid()
        )

# let PyTorch __call__ handle the magic apperantly 
    def forward(self, x):

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
