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
