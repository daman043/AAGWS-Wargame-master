from __future__ import print_function

import os
import json
import visdom
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from util.util import plot_position, cvtFlatten2Offset


class TrainGlobalSpacial(object):
    def __init__(self):
        pass

    @staticmethod
    def train(model, env, args):
        #################################### PLOT ###################################################
        save_pic_dir = os.path.join('../../data/result/{}'.format(args.name))
        if not os.path.isdir(save_pic_dir):
            os.makedirs(save_pic_dir)
        STEPS = 10
        LAMBDA = 0.98
        vis = visdom.Visdom(env=args.name+'[{}]'.format(args.phrase))
        pre_per_replay = [[] for _ in range(args.n_replays)]
        gt_per_replay = [[] for _ in range(args.n_replays)]
        acc = None
        record_loss, record_loss_step, record_acc, record_acc_step = [], [], [], []
        win = vis.line(X=np.zeros(1), Y=np.zeros(1), name='action',)
        loss_win = vis.line(X=np.zeros(1), Y=np.zeros(1), name='action[TOTAL]',)

        #################################### TRAIN ######################################################
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        gpu_id = args.gpu_id
        with torch.cuda.device(gpu_id):
            model = model.cuda() if gpu_id >= 0 else model
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

        epoch = 0
        save = args.save_intervel
        save_pic = args.save_intervel/10
        env_return = env.n_step()
        if env_return is not None:
            (states_G, states_S, actions_gt), require_init = env_return

        with torch.cuda.device(gpu_id):
            states_S = torch.from_numpy(states_S).float()
            states_G = torch.from_numpy(states_G).float()
            actions_gt = torch.from_numpy(actions_gt).float()  #.squeeze()

            if gpu_id >= 0:
                states_S = states_S.cuda()
                states_G = states_G.cuda()
                actions_gt = actions_gt.cuda()

        while True:
            actions = model(Variable(states_G), Variable(states_S), require_init)
            # print(actions.shape, actions_gt.shape)

            action_loss = 0
            for action, action_gt in zip(actions, actions_gt):
                action_loss += torch.sum((-action_gt.view(args.n_replays, 6, -1)
                                          * (1e-6 + action.view(args.n_replays, 6, -1)).log()), 2).mean()

            action_loss = action_loss / len(actions_gt)
            # print(action_loss)
            model.zero_grad()
            action_loss.backward()
            optimizer.step()

            if args.net_name != 'RES_tensor_vector':
                model.detach()

            if env.epoch > epoch:
                epoch = env.epoch
                for p in optimizer.param_groups:
                    p['lr'] *= 0.6
            ############################ PLOT ##########################################markers=True,
            vis.line(X=np.asarray([env.step_count()]),
                     Y=np.asarray([action_loss.data.cpu().numpy()]),
                     win=loss_win,
                     name='action',
                     update='append')
            record_loss.append(action_loss.data.cpu().numpy().tolist())
            record_loss_step.append(env.step_count())

            actions_np = np.swapaxes(np.asarray([np.argmax(action.data.cpu().numpy().reshape(args.n_replays, 6, -1),
                                                           axis=2) for action in actions]), 0, 1)
            actions_gt_np = np.swapaxes(np.asarray([np.argmax(action_gt.cpu().numpy().reshape(args.n_replays, 6, -1),
                                                              axis=2) for action_gt in actions_gt]), 0, 1)

            for idx, (action, action_gt, init) in enumerate(zip(actions_np, actions_gt_np, require_init)):

                if init and len(pre_per_replay[idx]) > 0:
                    pre_per_replay[idx] = np.asarray(pre_per_replay[idx], dtype=np.uint16)
                    gt_per_replay[idx] = np.asarray(gt_per_replay[idx], dtype=np.uint16)
                    step = len(pre_per_replay[idx]) // STEPS
                    # print(list(zip(action_pre, action_gt)))
                    if step > 0:
                        acc_tmp = []
                        for s in range(STEPS):
                            action_np = pre_per_replay[idx][s * step:(s + 1) * step, :]
                            action_gt_np = gt_per_replay[idx][s * step:(s + 1) * step, :]
                            acc_tmp.append(np.mean(action_np == action_gt_np))

                        acc_tmp = np.asarray(acc_tmp)
                        if acc is None:
                            acc = acc_tmp
                        else:
                            acc = LAMBDA * acc + (1-LAMBDA) * acc_tmp

                        if acc is None:
                            continue

                        step = env.step_count()
                        acc_mean = np.mean(acc)
                        for s in range(STEPS):
                            vis.line(X=np.asarray([step]),
                                     Y=np.asarray([acc[s]]),
                                     win=win,
                                     name='{}[{}%~{}%]'.format('action', s * 10, (s + 1) * 10),
                                     update='append')
                        vis.line(X=np.asarray([env.step_count()]),
                                 Y=np.asarray([acc_mean]),
                                 win=win,
                                 name='action[TOTAL]',
                                 update='append')

                        record_acc.append(acc_mean)
                        record_acc_step.append(step)


                        # print(true_postion, predict_position)
                        if env.step_count() > save_pic:
                            save_pic = env.step_count() + args.save_intervel
                            for num in range(6):
                                predict_position = [cvtFlatten2Offset(pos, env.frame_size[0], env.frame_size[1])
                                                    for pos in pre_per_replay[idx][:, num]]
                                true_position = [cvtFlatten2Offset(pos, env.frame_size[0], env.frame_size[1])
                                                 for pos in gt_per_replay[idx][:, num]]
                                plot_position(true_position, predict_position, save_pic_dir, str(env.step_count())+'-'+str(num))
                    pre_per_replay[idx] = []
                    gt_per_replay[idx] = []
                pre_per_replay[idx].append(action[-1])
                gt_per_replay[idx].append(action_gt[-1])
            ####################### NEXT BATCH ###################################
            env_return = env.n_step()
            if env_return is not None:
                (raw_states_G, raw_states_S, raw_rewards), require_init = env_return
                states_G = states_G.copy_(torch.from_numpy(raw_states_G).float())
                states_S = states_S.copy_(torch.from_numpy(raw_states_S).float())
                actions_gt = actions_gt.copy_(torch.from_numpy(raw_rewards).float())
                # .squeeze())
            if env.step_count() > save or env_return is None:
                # print(pre_per_replay)
                save = env.step_count()+args.save_intervel
                torch.save(model.state_dict(),
                           os.path.join(args.model_path, 'model_iter_{}.pth'.format(env.step_count())))
                torch.save(model.state_dict(), os.path.join(args.model_path, 'model_latest.pth'))

            if env_return is None:
                env.close()
                break
        dic = {'record_loss': record_loss,
               'record_loss_step': record_loss_step,
               'record_acc': record_acc,
               'record_acc_step': record_acc_step}
        with open(os.path.join(args.save_path, 'loss.json'), 'w', encoding='utf-8') as json_file:
            json.dump(dic, json_file, ensure_ascii=False)


    @staticmethod
    def test(model, env, args):
        ######################### SAVE RESULT ############################
        save_pic_dir = os.path.join('../../data/result/test/{}'.format(args.name))
        if not os.path.isdir(save_pic_dir):
            os.makedirs(save_pic_dir)
        save_pic = 10
        action_pre_per_replay = [[[]] for _ in range(6)]
        action_gt_per_replay = [[[]] for _ in range(6)]
        ######################### TEST ###################################
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        gpu_id = args.gpu_id
        with torch.cuda.device(gpu_id):
            model = model.cuda() if gpu_id >= 0 else model
        model.eval()

        env_return = env.n_step(test_mode=True)
        if env_return is not None:
            (states_G, states_S, actions_gt), require_init, see_n_piece = env_return
        with torch.cuda.device(gpu_id):
            states_G = torch.from_numpy(states_G).float()
            states_S = torch.from_numpy(states_S).float()
            actions_gt = torch.from_numpy(actions_gt).float()
            if gpu_id >= 0:
                states_G = states_G.cuda()
                states_S = states_S.cuda()
                actions_gt = actions_gt.cuda()
        while True:
            actions = model(Variable(states_G), Variable(states_S), require_init)
            ############################ PLOT ##########################################

            actions_np = np.asarray([np.argmax(action.data.cpu().numpy().reshape(args.n_replays, 6, -1), axis=2)
                                     for action in actions]).squeeze()
            actions_gt_np = np.asarray([np.argmax(action_gt.cpu().numpy().reshape(args.n_replays, 6, -1), axis=2)
                                        for action_gt in actions_gt]).squeeze()
            see_n_piece = see_n_piece.squeeze()

            if args.net_name != 'RES_tensor_vector':
                model.detach()

            if require_init[-1] and len(action_gt_per_replay[0][-1]) > 0:
                for piece in range(6):
                    action_pre_per_replay[piece][-1] = np.ravel(np.hstack(action_pre_per_replay[piece][-1]))
                    action_gt_per_replay[piece][-1] = np.ravel(np.hstack(action_gt_per_replay[piece][-1]))

                    action_pre_per_replay[piece].append([])
                    action_gt_per_replay[piece].append([])
                if env.step_count() > save_pic:
                    save_pic += save_pic + args.save_intervel/10000
                    for piece in range(6):
                        predict_position = [cvtFlatten2Offset(pos, env.frame_size[0], env.frame_size[1])
                                            for pos in action_pre_per_replay[piece][-2]]
                        true_position = [cvtFlatten2Offset(pos, env.frame_size[0], env.frame_size[1])
                                         for pos in action_gt_per_replay[piece][-2]]
                        plot_position(true_position, predict_position, save_pic_dir, str(env.step_count()) + '-' + str(piece))

            for piece in range(6):
                if not see_n_piece[piece]:
                    action_pre_per_replay[piece][-1].append(actions_np[piece])
                    action_gt_per_replay[piece][-1].append(actions_gt_np[piece])
            # print(list(zip(action_pre_per_replay[-1],action_gt_per_replay[-1])))
            ########################### NEXT BATCH #############################################
            env_return = env.n_step(test_mode=True)

            if env_return is not None:
                (raw_states_G, raw_states_S, raw_actions), require_init, see_n_piece = env_return
                states_G = states_G.copy_(torch.from_numpy(raw_states_G).float())
                states_S = states_S.copy_(torch.from_numpy(raw_states_S).float())
                actions_gt = actions_gt.copy_(torch.from_numpy(raw_actions).float())
            else:
                for piece in range(6):
                    action_pre_per_replay[piece][-1] = np.ravel(np.hstack(action_pre_per_replay[piece][-1]))
                    action_gt_per_replay[piece][-1] = np.ravel(np.hstack(action_gt_per_replay[piece][-1]))
                env.close()
                break

        return action_pre_per_replay, action_gt_per_replay






