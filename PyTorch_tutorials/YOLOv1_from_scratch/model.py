import torch
import torch.nn as nn


# architecture without linear layers at the end (only convolutional layers)
# for representation
architecture_config = [
    (7, 64, 2, 3),
    "M",                                    # MaxPool layer 2x2 with stride 2
    (3, 192, 1, 1),                         # Convolutional layer: (kernel size, out channels, stride, padding)
    "M",                                    # MaxPool layer 2x2 with stride 2
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",                                    # MaxPool layer 2x2 with stride 2
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],    # List of Convolutional layers, last digit - how many times are repeated
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",                                    # MaxPool layer 2x2 with stride 2
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, 
                bias=False, **kwargs),      # bias is False because we are going \
                                            # to use batch normaliaztion after
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.block(x)


class YOLOv1(nn.Module):
    def __init__(self, in_channels: int = 3, **kwargs) -> None:
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.darknet(x)
        return self.fcs(x)

    def _create_conv_layers(self, architecture: list) -> nn.Module:
        darnet = nn.Sequential()
        in_channels = self.in_channels

        for layer in architecture:

            # convolutionl black
            if isinstance(layer, tuple):
                kernel_size, out_channels, stride, padding = layer
                darnet.append(CNNBlock(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding
                ))
                in_channels = out_channels

            # pooling layer
            elif isinstance(layer, str) and layer == "M":
                darnet.append(nn.MaxPool2d(kernel_size=2, stride=2))

            # convolutional block with repeats
            elif isinstance(layer, list):
                num_repeats = layer[-1]
                for _ in range(num_repeats):
                    for kernel_size, out_channels, stride, padding in layer[:-1]:
                        darnet.append(CNNBlock(
                            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding
                        ))
                        in_channels = out_channels
        return darnet

    def _create_fcs(self, split_size: int = 7, n_boxes: int = 2, n_classes: int = 20, 
                    hidden_size: int = 4096, p: float = 0.5):
        # split size means for how many grid cells should we split our feature map
        # the value of grid cells is s**2 (so s - n rows and n cols)
        s, b, c = split_size, n_boxes, n_classes
        return nn.Sequential(
            nn.Flatten(),  # flatten feature map
            nn.Linear(in_features=1024*s*s, out_features=hidden_size),  # original paper is 4096
            nn.Dropout(p=p),  # in original paper is 0.5
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=hidden_size, out_features=s*s*(b*5 + c)),
        )
