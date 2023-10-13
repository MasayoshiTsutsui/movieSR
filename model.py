import torch
import math

class SRCNN(torch.nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # Feature extraction layer.
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, (9, 9), (1, 1), (4, 4)),
            torch.nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            torch.nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = torch.nn.Conv2d(32, 3, (5, 5), (1, 1), (2, 2))

        # init model weights.
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initd by random extraction 0.001 (deviation is 0)
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                torch.nn.init.zeros_(module.bias.data)

        torch.nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        torch.nn.init.zeros_(self.reconstruction.bias.data)


class FSRCNN(torch.nn.Module):
    def __init__(self, scale_factor=2, num_channels=3):
        super(FSRCNN, self).__init__()
        
        # Feature extraction layers
        self.feature_extraction = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 64, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True)
        )
        
        # Shrinking layer
        self.shrinking = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=1),
            torch.nn.ReLU(inplace=True)
        )
        
        # Mapping layers
        self.mapping = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        
        # Expanding layer
        self.expanding = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=1),
            torch.nn.ReLU(inplace=True)
        )
        
        # Deconvolution layer (transposed convolution)
        self.deconvolution = torch.nn.ConvTranspose2d(64, num_channels, kernel_size=9, stride=scale_factor, padding=4, output_padding=scale_factor-1)
    
    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out

class DRRN(torch.nn.Module):
    def __init__(self, scale_factor=2, num_channels=3, num_blocks=9):
        super(DRRN, self).__init__()

        self.nun_blocks = num_blocks
        self.initial_conv = torch.nn.Conv2d(num_channels, 128, kernel_size=3, padding=1)
        self.res_block = self.make_res_block()
        self.reconstruction = torch.nn.Conv2d(128, num_channels, kernel_size=3, padding=1)

        self.initialize_weights()

    def make_res_block(self):
        layers = []
        for _ in range(2):
            layers.append(torch.nn.BatchNorm2d(128))  # BatchNorm added
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Conv2d(128, 128, kernel_size=3, padding=1))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        initial_out = self.initial_conv(x)
        next_in = initial_out

        for _ in range(self.nun_blocks):
            res_out = self.res_block(next_in)
            next_in = torch.add(initial_out, res_out)  # Residual connection

        reconstruction_out = self.reconstruction(next_in)
        sr_out = torch.add(reconstruction_out, x)  # Residual connection
        return sr_out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
