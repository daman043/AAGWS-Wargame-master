import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic residual block with 2 convolutions and a skip connection before the last
    ReLU activation.
    """
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state' described in the Feature Extractor model.
    """

    def __init__(self, inplanes, midplanes, outplanes, outdense, BLOCKS_num):
        super(ResNet, self).__init__()
        self.outplanes = outplanes
        self.outdense = outdense
        self.BLOCKS_num = BLOCKS_num
        self.conv1 = nn.Conv2d(inplanes, midplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)

        for block in range(self.BLOCKS_num):
            setattr(self, "res{}".format(block), BasicBlock(midplanes, midplanes))

        self.conv = nn.Conv2d(midplanes, outplanes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        for fc in range(outplanes):
            setattr(self, "fc{}".format(fc), nn.Linear(outplanes * outdense, outdense))

        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        """
        x : feature maps extracted from the state
        probas : a NxN + 1 matrix where N is the board size
                 Each value in this matrix represent the likelihood
                 of winning by playing this intersection
        """
        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(self.BLOCKS_num):
            x = getattr(self, "res{}".format(block))(x)

        x = F.relu(self.bn2(self.conv(x)))
        x = x.view(-1, self.outplanes*self.outdense)
        mm = [F.relu(getattr(self, "fc{}".format(fc))(x)) for fc in range(self.outplanes)]

        x = torch.stack(mm, dim=1)

        # x = self.fc(x)
        probas = self.logsoftmax(x).exp()
        return probas