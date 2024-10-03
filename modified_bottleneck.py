from typing import Callable, Optional, OrderedDict
from torch import nn, Tensor
from torchvision.models.resnet import conv3x3, conv1x1


class ModifiedBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.sequential_block = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv1x1(inplanes, width)),
                    ("bn1", norm_layer(width)),
                    ("relu1", self.relu),
                    ("conv2", conv3x3(width, width, stride, groups, dilation)),
                    ("bn2", norm_layer(width)),
                    ("relu2", self.relu),
                    ("conv3", conv1x1(width, planes * self.expansion)),
                    ("bn3", norm_layer(planes * self.expansion)),
                ]
            )
        )
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.sequential_block(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
