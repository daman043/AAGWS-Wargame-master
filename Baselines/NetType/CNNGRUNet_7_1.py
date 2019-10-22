
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class CNNGRUNet_7_1(torch.nn.Module):
    def __init__(self, inplanes, midplanes, outplanes, in_vector, out_vector, in_rnn, out_rnn, outdense):
        super(CNNGRUNet_7_1, self).__init__()
        self.midplanes = midplanes
        self.outplanes = outplanes
        self.out_vector = out_vector
        self.plane_length = outdense
        self.out_rnn = out_rnn
        self.conv1 = nn.Conv2d(inplanes, midplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)

        for num in range(2, 7):
            setattr(self, "conv{}".format(num), nn.Conv2d(midplanes, midplanes,
                                                          stride=1, kernel_size=3, padding=1, bias=False))
            setattr(self, "bn{}".format(num), nn.BatchNorm2d(midplanes))

        self.conv7 = nn.Conv2d(midplanes, outplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(outplanes)

        self.linear_g = nn.Linear(in_vector, out_vector)

        for num in range(6):
            setattr(self, "pre_fc{}".format(num), nn.Linear(outplanes * outdense + out_vector, in_rnn))
            setattr(self, "rnn{}".format(num), nn.GRUCell(input_size=in_rnn, hidden_size=out_rnn))
            setattr(self, "end_fc{}".format(num), nn.Linear(out_rnn, outdense))
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.h = None

    def forward(self, states_G, states_S, require_init):
        batch = states_S.size(1)
        self.h = [None for _ in range(6)]
        for num in range(6):
            if self.h[num] is None:
                self.h[num] = Variable(states_S.data.new().resize_((batch, self.out_rnn)).zero_())
            elif True in require_init:
                h = self.h[num].data
                for idx, init in enumerate(require_init):
                    if init:
                        h[idx].zero_()
                self.h[num] = Variable(h)
            else:
                pass

        values = []
        for idx, (state_S, state_G) in enumerate(zip(states_S, states_G)):
            x_s = F.relu(self.bn1(self.conv1(state_S)))
            for num in range(2, 8):
                x_s = F.relu(getattr(self, "bn{}".format(num))(getattr(self, "conv{}".format(num))(x_s)))

            x_s = x_s.view(-1, self.plane_length*self.outplanes)

            x_g = F.relu(self.linear_g(state_G))

            x = torch.cat((x_s, x_g), 1)

            x_list = [F.relu(getattr(self, "pre_fc{}".format(num))(x)) for num in range(6)]
            self.h = [getattr(self, "rnn{}".format(num))(x_list[num], self.h[num]) for num in range(6)]
            x_list = [F.relu(getattr(self, "end_fc{}".format(num))(self.h[num])) for num in range(6)]
            x = torch.stack(x_list, dim=1)
            probas = self.logsoftmax(x).exp()
            values.append(probas)
        return values

    def detach(self):
        for num in range(6):
            if self.h[num] is not None:
                self.h[num].detach_()
