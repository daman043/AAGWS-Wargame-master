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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ResNet_double(nn.Module):
    """
    This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state' described in the Feature Extractor model.
    """

    def __init__(self, inplanes, midplanes, outplanes, BLOCKS_num, in_vector, out_vector, outdense, size=[66, 51]):
        super(ResNet_double, self).__init__()
        self.BLOCKS_num = BLOCKS_num
        self.outplanes = outplanes
        self.outdense = outdense
        self.conv1 = nn.Conv2d(inplanes, midplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)

        for block in range(self.BLOCKS_num):
            setattr(self, "res{}".format(block), BasicBlock(midplanes, midplanes))

        self.conv2 = nn.Conv2d(midplanes, outplanes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        # self.fc = nn.Linear(self.plane_length, self.plane_length)
        self.linear_g = nn.Linear(in_vector, out_vector)

        for fc in range(outplanes):
            setattr(self, "fc{}".format(fc), nn.Linear(outplanes * outdense + out_vector, outdense))

        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, states_G, states_S, require_init):
        """
        x : feature maps extracted from the state
        probas : a NxN + 1 matrix where N is the board size
                 Each value in this matrix represent the likelihood
                 of winning by playing this intersection
        """
        states_G = states_G.squeeze(0)
        states_S = states_S.squeeze(0)
        x_s = F.relu(self.bn1(self.conv1(states_S)))
        for block in range(self.BLOCKS_num):
            x_s = getattr(self, "res{}".format(block))(x_s)
        x_s = F.relu(self.bn2(self.conv2(x_s)))
        x_s = x_s.view(-1, self.outplanes*self.outdense)

        x_g = F.relu(self.linear_g(states_G))

        x = torch.cat((x_s, x_g), 1)
        mm = [F.relu(getattr(self, "fc{}".format(fc))(x)) for fc in range(self.outplanes)]
        x = torch.stack(mm, dim=1)
        probas = self.logsoftmax(x).exp().unsqueeze(0)

        return probas