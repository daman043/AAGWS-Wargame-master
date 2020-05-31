from __future__ import print_function

import os
import json
import time
import pickle
import argparse

import visdom
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

from Baselines.GlobalStateEvaluation.test import show_test_result

from data_loader.BatchEnvEnemy import *
from util.util import all_true

ArgsNet = {'midplanes1': 15,
                    'midplanes2': 5,
                 'outplanes': 1,
                 'out_vector': 100,
                 'in_rnn': 512,
                 'out_rnn': 128}
class StateEvaluationNet(nn.Module):

    def __init__(self, inplanes, midplanes1, midplanes2, outplanes, in_vector, out_vector, in_rnn, out_rnn, size = [66, 51]):
        super(StateEvaluationNet, self).__init__()
        self.outplanes = outplanes
        self.out_vector = out_vector
        self.out_rnn = out_rnn
        self.size = size
        self.conv1 = nn.Conv2d(inplanes, midplanes1, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes1)

        self.conv2 = nn.Conv2d(midplanes1, midplanes2, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes2)

        self.conv3 = nn.Conv2d(midplanes2, outplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        self.linear_g = nn.Linear(in_vector, out_vector)

        self.pre_fc = nn.Linear(outplanes * size[0] * size[1] + out_vector, in_rnn)
        self.rnn = nn.GRUCell(input_size=in_rnn, hidden_size=out_rnn)
        self.critic_linear = nn.Linear(out_rnn, 1)

        self.h = None

    def forward(self, states_S, states_G, require_init):
        batch = states_S.size(0)
        if self.h is None:
            self.h = Variable(states_S.data.new().resize_((batch, self.out_rnn)).zero_(), requires_grad = True)
        elif True in require_init:
            h= self.h.data
            for idx, init in enumerate(require_init):
                if init:
                    h[idx].zero_()
            self.h = Variable(h)
        else:
            pass

        x_s = F.relu(self.bn1(self.conv1(states_S)))
        x_s = F.relu(self.bn2(self.conv2(x_s)))
        x_s = F.relu(self.bn3(self.conv3(x_s)))
        x_s = x_s.view(-1, self.outplanes * self.size[0] * self.size[1])
        x_g = F.relu(self.linear_g(states_G))
        x = torch.cat((x_s, x_g), 1)
        x = F.relu(self.pre_fc(x))
        self.h = self.rnn(x, self.h)

        value = torch.sigmoid(self.critic_linear(self.h))

        return value

    def detach(self):
        if self.h is not None:
            self.h.detach_()

def train(model, env, args):
    #################################### PLOT ###################################################
    STEPS = 10
    LAMBDA = 0.97
    vis = visdom.Visdom(env=args.name+'[{}]'.format(args.phrase))
    pre_per_replay = [[] for _ in range(args.n_replays)]
    gt_per_replay = [[] for _ in range(args.n_replays)]
    acc = None
    win = vis.line(X=np.zeros(1), Y=np.zeros(1))
    loss_win = vis.line(X=np.zeros(1), Y=np.zeros(1))

    #################################### TRAIN ######################################################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    gpu_id = args.gpu_id
    with torch.cuda.device(gpu_id):
        model = model.cuda() if gpu_id >= 0 else model
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epoch = 0
    save = args.save_intervel
    env_return = env.n_step(all_init=True)
    if env_return is not None:
        (states_S, states_G, rewards), require_init, _ = env_return
        states_S = states_S.squeeze(0)
        states_G = states_G.squeeze(0)
        rewards = rewards.reshape(args.n_replays, -1)
        # rewards[rewards == 0] = 0.5
        rewards[rewards == -1] = 0
        # print(states_S.shape, states_G.shape, rewards.shape, )
    with torch.cuda.device(gpu_id):
        states_S = torch.from_numpy(states_S).float()
        states_G = torch.from_numpy(states_G).float()

        rewards = torch.from_numpy(rewards).float()
        if gpu_id >= 0:
            states_S = states_S.cuda()
            states_G = states_G.cuda()
            rewards = rewards.cuda()

    while True:
        values = model(Variable(states_S), Variable(states_G), require_init)
        value_loss = F.mse_loss(values, Variable(rewards))
        # print(value_loss)
        model.zero_grad()
        value_loss.backward(retain_graph=True)
        optimizer.step()
        if all_true(require_init):
            model.detach()

        if env.epoch > epoch:
            epoch = env.epoch
            for p in optimizer.param_groups:
                p['lr'] *= 0.6

        ############################ PLOT ##########################################
        vis.line(X=np.asarray([env.step_count()]),
                 Y=np.asarray([value_loss.data.cpu().numpy()]),
                        win=loss_win,
                        name='value',
                 update='append')

        values_np = values.data.cpu().numpy()
        rewards_np = rewards.cpu().numpy()

        for idx, (value, reward, init) in enumerate(zip(values_np, rewards_np, require_init)):
            if init and len(pre_per_replay[idx]) > 0:
                pre_per_replay[idx] = np.asarray(pre_per_replay[idx], dtype=np.uint8)
                gt_per_replay[idx] = np.asarray(gt_per_replay[idx], dtype=np.uint8)
                # print(pre_per_replay[idx], gt_per_replay[idx])
                step = len(pre_per_replay[idx]) // STEPS
                if step > 0:
                    acc_tmp = []
                    for s in range(STEPS):
                        value_pre = pre_per_replay[idx][s*step:(s+1)*step]
                        value_gt = gt_per_replay[idx][s*step:(s+1)*step]
                        acc_tmp.append(np.mean(value_pre == value_gt))

                    acc_tmp = np.asarray(acc_tmp)
                    if acc is None:
                        acc = acc_tmp
                    else:
                        acc = LAMBDA * acc + (1-LAMBDA) * acc_tmp

                    if acc is None:
                        continue
                    for s in range(STEPS):
                        vis.line(X=np.asarray([env.step_count()]),
                                        Y=np.asarray([acc[s]]),
                                        win=win,
                                        name='{}[{}%~{}%]'.format('value', s*10, (s+1)*10),
                                 update='append')
                    vis.line(X=np.asarray([env.step_count()]),
                                    Y=np.asarray([np.mean(acc)]),
                                    win=win,
                                    name='value[TOTAL]',
                             update='append')

                pre_per_replay[idx] = []
                gt_per_replay[idx] = []

            pre_per_replay[idx].append(int(value[-1] > 0.5))
            gt_per_replay[idx].append(int(reward))

        ####################### NEXT BATCH ###################################
        env_return = env.n_step(all_init=True)
        if env_return is not None:
            (raw_states_S, raw_states_G, raw_rewards), require_init, _ = env_return
            raw_states_S = raw_states_S.squeeze(0)
            raw_states_G = raw_states_G.squeeze(0)
            raw_rewards = raw_rewards.reshape(args.n_replays, -1)
            # raw_rewards[raw_rewards == 0] = 0.5
            raw_rewards[raw_rewards == -1] = 0
            states_S = states_S.copy_(torch.from_numpy(raw_states_S).float())
            states_G = states_G.copy_(torch.from_numpy(raw_states_G).float())
            rewards = rewards.copy_(torch.from_numpy(raw_rewards).float())

        if env.step_count() > save or env_return is None:
            save = env.step_count()+args.save_intervel
            torch.save(model.state_dict(),
                       os.path.join(args.model_path, 'model_iter_{}.pth'.format(env.step_count())))
            torch.save(model.state_dict(), os.path.join(args.model_path, 'model_latest.pth'))
        if env_return is None:
            env.close()
            break

def test(model, env, args):
    ######################### SAVE RESULT ############################
    value_pre_per_replay = [[]]
    value_gt_per_replay = [[]]

    ######################### TEST ###################################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    gpu_id = args.gpu_id
    with torch.cuda.device(gpu_id):
        model = model.cuda() if gpu_id >= 0 else model
    model.eval()

    env_return = env.n_step(test_mode=True, all_init=True)
    if env_return is not None:
        (states_S, states_G, rewards), require_init, _ = env_return
        states_S = states_S.squeeze(0)
        states_G = states_G.squeeze(0)
        rewards = rewards.reshape(args.n_replays, -1)
        # rewards[rewards == 0] = 0.5
        rewards[rewards == -1] = 0
    with torch.cuda.device(gpu_id):
        states_S = torch.from_numpy(states_S).float()
        states_G = torch.from_numpy(states_G).float()

        rewards = torch.from_numpy(rewards).float()
        if gpu_id >= 0:
            states_S = states_S.cuda()
            states_G = states_G.cuda()
            rewards = rewards.cuda()

    while True:
        values = model(Variable(states_S), Variable(states_G), require_init)
        ############################ PLOT ##########################################
        values_np = np.squeeze(values.data.cpu().numpy())
        rewards_np = np.squeeze(rewards.cpu().numpy())
        print(values_np, rewards_np)
        if require_init[-1] and len(value_gt_per_replay[-1]) > 0:
            value_pre_per_replay[-1] = np.ravel(np.hstack(value_pre_per_replay[-1]))
            value_gt_per_replay[-1] = np.ravel(np.hstack(value_gt_per_replay[-1]))

            value_pre_per_replay.append([])
            value_gt_per_replay.append([])

        value_pre_per_replay[-1].append(int(values_np > 0.5))
        value_gt_per_replay[-1].append(int(rewards_np))

        ########################### NEXT BATCH #############################################
        env_return = env.n_step(test_mode=True, all_init=True)
        if env_return is not None:
            (raw_states_S, raw_states_G, raw_rewards), require_init, _ = env_return
            raw_states_S = raw_states_S.squeeze(0)
            raw_states_G = raw_states_G.squeeze(0)
            raw_rewards = raw_rewards.reshape(args.n_replays, -1)
            # raw_rewards[raw_rewards == 0] = 0.5
            raw_rewards[raw_rewards == -1] = 0
            states_S = states_S.copy_(torch.from_numpy(raw_states_S).float())
            states_G = states_G.copy_(torch.from_numpy(raw_states_G).float())
            rewards = rewards.copy_(torch.from_numpy(raw_rewards).float())
        else:
            value_pre_per_replay[-1] = np.ravel(np.hstack(value_pre_per_replay[-1]))
            value_gt_per_replay[-1] = np.ravel(np.hstack(value_gt_per_replay[-1]))

            env.close()
            break

    return value_pre_per_replay, value_gt_per_replay

def next_path(model_folder, paths):
    models = {int(os.path.basename(model).split('.')[0].split('_')[-1])
                for model in os.listdir(model_folder) if 'latest' not in model}
    models_not_process = models - paths
    if len(models_not_process) == 0:
        return None
    models_not_process = sorted(models_not_process, reverse=True)
    paths.add(models_not_process[0])

    return os.path.join(model_folder, 'model_iter_{}.pth'.format(models_not_process[0]))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Global State Evaluation : Wargame')
    parser.add_argument('--name', type=str, default='raw_feature',
                        help='Experiment name. raw_feature or predect_feature')
    parser.add_argument('--net_name', type=str, default='CNN_GRU',
                        help='which type of the net is:raw_feature is feature_tensor_vector; RES_tensor or RES_tensor_map or RES_tensor_vector \
                            or CNN_GRU '),
    parser.add_argument('--predect_feature_from', default='CNN_GRU',
                        help='Which model the predected feature from, RES_tensor or RES_tensor_map or RES_tensor_vector \
                        or CNN_GRU ')
    parser.add_argument('--piece_name', type=str, default='all',
                        help='the name of the chess pieces: tank1, tank2, car1, car2, soldier1, soldier2, all'),
    # parser.add_argument('--replays_path', default='../../data/train_test/feature_tensor_vector',
    #                     help='Path for training, and test set')
    parser.add_argument('--race', default='red', help='Which race? (default: Red)')
    parser.add_argument('--enemy_race', default='blue', help='Which the enemy race? (default: Blue)')
    parser.add_argument('--phrase', type=str, default='train',
                        help='train|test (default: train)')
    parser.add_argument('--gpu_id', default=1, type=int, help='Which GPU to use [-1 indicate CPU] (default: 0)')

    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')

    parser.add_argument('--n_steps', type=int, default=1, help='# of forward steps (default: 20)')
    parser.add_argument('--load_steps', type=int, default=1,
                        help='# the intervel steps of the frame of data (default: 1)')
    parser.add_argument('--n_replays', type=int, default=32, help='# of replays (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=15, help='# of epoches (default: 10)')

    parser.add_argument('--save_intervel', type=int, default=50000,
                        help='Frequency of model saving (default: 50000)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume the training (default: False)')
    args = parser.parse_args()

    args.save_path = os.path.join('checkpoints', args.name)
    args.model_path = os.path.join(args.save_path, 'snapshots')
    piece_name_to_offset = {'all': 0, 'tank1': 0, 'tank2': 1, 'car1': 2, 'car2': 3, 'soldier1': 4, 'soldier2': 5}
    args.piece_offset = piece_name_to_offset[args.piece_name]

    if args.name == 'raw_feature':
        args.replays_path = '../../data/train_test/feature_tensor_vector'
    elif args.name == 'predect_feature':
        args.replays_path = os.path.join('../../data/train_test', args.predect_feature_from)
    else:
        raise Exception

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))
    print('-------------- End ----------------')

    if args.phrase == 'train':
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)
        with open(os.path.join(args.save_path, 'config'), 'w') as f:
            f.write(json.dumps(vars(args)))

        env = Env_Evaluation()
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format(args.phrase)))
        root = os.path.join('../')

        env.init(path_replays, root, args)

        model = StateEvaluationNet(env.n_channels,  ArgsNet['midplanes1'], ArgsNet['midplanes2'],
                                   ArgsNet['outplanes'], env.n_features, ArgsNet['out_vector'],
                                   ArgsNet['in_rnn'], ArgsNet['out_rnn'], env.frame_size)
        train(model, env, args)
    elif 'test' in args.phrase:
        test_result_path = os.path.join(args.save_path, args.phrase)
        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)

        paths = set()
        test_result = []
        path_list = []
        while True:
            path = next_path(args.model_path, paths)
            if path is not None:
                print('[{}]Testing {} ...'.format(len(paths), path))

                env = Env_Evaluation()
                path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format(args.phrase)))
                root = os.path.join('../')
                args.n_epochs = 1
                args.n_replays = 1
                env.init(path_replays, root, args)


                model = StateEvaluationNet(env.n_channels,  ArgsNet['midplanes1'], ArgsNet['midplanes2'],
                                   ArgsNet['outplanes'], env.n_features, ArgsNet['out_vector'],
                                   ArgsNet['in_rnn'], ArgsNet['out_rnn'], env.frame_size)
                model.load_state_dict(torch.load(path))

                result = test(model, env, args)

                with open(os.path.join(test_result_path, os.path.basename(path)), 'wb') as f:
                    pickle.dump(result, f)
                value_acc, value_acc_step = show_test_result(args.name, args.phrase, result, title=len(paths)-1)
                value_pre_per_replay, value_gt_per_replay = result
                dic = {'value_pre_per_replay': value_pre_per_replay,
                       'value_gt_per_replay': value_gt_per_replay,
                       'mean_acc': value_acc,
                       'n_stage_acc': value_acc_step}
                test_result.append(dic)
                path_list.append(path)
            else:
                with open(os.path.join(test_result_path, 'test_result'), 'wb') as f:
                    f.write(pickle.dumps(test_result))
                with open(os.path.join(test_result_path, 'path_list'), 'wb') as f:
                    f.write(pickle.dumps(path_list))
                print('Test ending')
                break

if __name__ == '__main__':
    main()
