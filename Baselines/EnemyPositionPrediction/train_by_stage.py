from __future__ import print_function
import json, pickle
import argparse
import torch
from Baselines.EnemyPositionPrediction.test import show_test_result
from data_loader.BatchEnvEnemy import *
from util.util import *
import pandas as pd


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Enemy position prediction by the features of tensor: Wargame')
    parser.add_argument('--net_name', type=str, default='stage', help='which type of the net is: stage'),

    parser.add_argument('--piece_name', type=str, default='all',
                        help='the name of the chess pieces: tank1, tank2, car1, car2, soldier1, soldier2, all'),
    parser.add_argument('--replays_path', default='../../data/train_val_test/feature_CNN_GRU',
                        help='Path for training, validation and test set (default: train_val_test)')
    parser.add_argument('--race', default='red', help='Which race? (default: Red)')
    parser.add_argument('--enemy_race', default='blue', help='Which the enemy race? (default: Blue)')
    parser.add_argument('--phrase', type=str, default='test',
                        help='train|val|test (default: train)')
    parser.add_argument('--n_steps', type=int, default=1, help='# of forward steps (default: 20)')
    parser.add_argument('--load_steps', type=int, default=1,
                        help='# the intervel steps of the frame of data (default: 1)')
    parser.add_argument('--seed', type=int, default=2, help='Random seed (default: 1)')
    parser.add_argument('--n_replays', type=int, default=32, help='# of replays (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=1, help='# of epoches (default: 10)')


    args = parser.parse_args()

    args.name = args.net_name + args.piece_name
    args.save_path = os.path.join('checkpoints', args.name, args.piece_name)
    args.model_path = os.path.join(args.save_path, 'snapshots')
    piece_name_to_offset = {'all': 0, 'tank1': 0, 'tank2': 1, 'car1': 2, 'car2': 3, 'soldier1': 4, 'soldier2': 5}

    args.piece_offset = piece_name_to_offset[args.piece_name]

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

        env = Env_CNN_GRU()
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format(args.phrase)))
        root = os.path.join('../')

        env.init(path_replays, root, args)

        train(env, args)

    elif 'val' in args.phrase or 'test' in args.phrase:
        test_result_path = os.path.join(args.save_path, args.phrase)
        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)

        args.n_epochs = 1
        args.n_replays = 1
        env = Env_CNN_GRU()
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format(args.phrase)))
        root = os.path.join('../')
        env.init(path_replays, root, args)

        position_count = pd.read_csv(os.path.join(args.save_path + "position_count.csv"))
        position_count.set_index(['stage', 'bopname'], inplace=True)
        action_pre_per_replay, action_gt_per_replay = test(position_count, env, args)



        mean_acc, n_stage_acc = [], []
        for piece in range(6):
            result = (action_pre_per_replay[piece], action_gt_per_replay[piece])
            mean_acc_piece, n_stage_acc_piece = show_test_result(args.name, args.phrase, result, title=str(piece))
            mean_acc.append(mean_acc_piece)
            n_stage_acc.append(n_stage_acc_piece)
        test_result = {'action_pre_per_replay': action_pre_per_replay,
                       'action_gt_per_replay': action_gt_per_replay,
                       'mean_acc': mean_acc,
                       'n_stage_acc': n_stage_acc}

        with open(os.path.join(test_result_path, 'test_result'), 'wb') as f:
            f.write(pickle.dumps(test_result))
        print('Test ending')



def train(env, args):
    #################################### TRAIN ######################################################
    row_size = env.frame_size[0]
    col_size = env.frame_size[1]
    index_array = np.zeros((row_size, col_size), dtype=int)
    for i in range(row_size):
        for j in range(col_size):
            # look_grid_Int6loc = cvtHexOffset2Int6loc(i, j)
            index_array[i, j] = i * col_size + j #int(look_grid_Int6loc)
    index_array.reshape(row_size * col_size)
    zero_array = np.zeros_like(index_array)
    columns = np.append(['stage', 'bopname'], index_array)
    position_count = pd.DataFrame()
    for stage in range(1, 21):
        for bopname in range(6):
            row_data = np.append([stage, bopname], zero_array).reshape((1, row_size * col_size + 2))
            row_pd = pd.DataFrame(data=row_data, columns=columns, dtype=int)
            position_count = position_count.append(row_pd)
    position_count.set_index(['stage', 'bopname'], inplace=True)
    print(position_count)
    while True:
        env_return = env.n_step()
        if env_return is not None:
            (states_G, states_S, raw_rewards), require_init = env_return
            n_replays = states_G.shape[0]
            n_steps = states_G.shape[1]
            for replay in range(n_replays):
                for step in range(n_steps):
                    enemy_position = np.argmax(raw_rewards[replay, step, :, :, :].reshape(-1, row_size * col_size), axis=1)
                    # print(enemy_position)
                    stage_one_hot = states_G[replay, step, :20]
                    if stage_one_hot.sum() == 1:
                        stage_index = np.argmax(stage_one_hot, axis=0) + 1
                        # print(stage_index)
                    else:
                        raise Exception('error')
                    for bop in range(6):
                        if enemy_position[bop] != 0:
                            position_count.loc[(stage_index, bop), [str(enemy_position[bop])]] += 1

        if env_return is None:
            env.close()
            break
    print(position_count)
    position_count = position_count.div(position_count.sum(axis=1), axis=0)
    print(position_count.max(axis=1))

    position_count.to_csv(os.path.join(args.save_path + "position_count.csv"))


def test(position_count, env, args):
    ######################### SAVE RESULT ############################
    save_pic_dir = os.path.join('../../data/result/test/{}'.format(args.name))
    if not os.path.isdir(save_pic_dir):
        os.makedirs(save_pic_dir)
    save_pic = 100
    action_pre_per_replay = [[[]] for _ in range(6)]
    action_gt_per_replay = [[[]] for _ in range(6)]
    ######################### TEST ###################################
    row_size = env.frame_size[0]
    col_size = env.frame_size[1]



    while True:
        env_return = env.n_step(test_mode=True)
        if env_return is not None:
            (states_G, states_S, actions_gt), require_init, see_n_piece = env_return
        states_G = states_G.squeeze()
        states_S = states_S.squeeze()
        actions_gt = actions_gt.squeeze()
        see_n_piece = see_n_piece.squeeze()

        stage_one_hot = states_G[:20]
        if stage_one_hot.sum() == 1:
            stage_index = np.argmax(stage_one_hot, axis=0) + 1
        else:
            raise Exception('error')
        actions_gt_np = np.argmax(actions_gt.reshape(-1, row_size * col_size), axis=1)
        actions_np = position_count.loc[(stage_index, slice(None)), :].idxmax(axis=1).values.astype(int)

        # print(actions_gt_np, actions_np)
        if require_init[-1] and len(action_gt_per_replay[0][-1]) > 0:
            for piece in range(6):
                action_pre_per_replay[piece][-1] = np.ravel(np.hstack(action_pre_per_replay[piece][-1]))
                action_gt_per_replay[piece][-1] = np.ravel(np.hstack(action_gt_per_replay[piece][-1]))

                action_pre_per_replay[piece].append([])
                action_gt_per_replay[piece].append([])
            if env.step_count() > save_pic:
                save_pic += save_pic + 1000
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


        if env_return is None:
            for piece in range(6):
                action_pre_per_replay[piece][-1] = np.ravel(np.hstack(action_pre_per_replay[piece][-1]))
                action_gt_per_replay[piece][-1] = np.ravel(np.hstack(action_gt_per_replay[piece][-1]))
            env.close()
            break

    return action_pre_per_replay, action_gt_per_replay


if __name__ == '__main__':
    main()