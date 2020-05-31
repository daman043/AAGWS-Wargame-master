from __future__ import print_function
import json, pickle
import argparse
import torch, sys
sys.path.append("../..")

from Baselines.EnemyPositionPrediction.test import show_test_result, show_test_pro_result
from data_loader.BatchEnvEnemy import *
from Baselines.NetType.ResNet import ResNet
from Baselines.NetType.CNNGRUNet import CNNGRUNet

from Baselines.NetType.ResNet_double import ResNet_double
from Baselines.NetType.CNNGRUNet_5_1 import CNNGRUNet_5_1
from Baselines.NetType.CNNGRUNet_5_2 import CNNGRUNet_5_2
from Baselines.NetType.CNNGRUNet_7_1 import CNNGRUNet_7_1
from Baselines.NetType.CNNGRUNet_7_2 import CNNGRUNet_7_2
from Baselines.NetType.CNNGRUNet_9_1 import CNNGRUNet_9_1
from Baselines.NetType.CNNGRUNet_9_2 import CNNGRUNet_9_2
from Baselines.EnemyPositionPrediction.train_spatial import TrainSpatial
from Baselines.EnemyPositionPrediction.train_global_spatial import TrainGlobalSpacial


ArgsResNet = {'midplanes': 20,
              'outplanes': 6,
              'BLOCKS_num': 8}

ArgsResNet_double = {'midplanes': 20,
                     'outplanes': 6,
                     'BLOCKS_num': 8,
                     'out_vector': 100}

ArgsCNNGRUNet = {'midplanes': 20,
                 'outplanes': 6,
                 'out_vector': 100,
                 'in_rnn': 200,
                 'out_rnn': 200}


torch.cuda.empty_cache()

def get_env(name):
    if name == 'RES_tensor':
        env = Env_RES_tensor()
    elif name == 'RES_tensor_map':
        env = Env_RES_tensor_map()
    elif name == 'RES_tensor_vector':
        env = Env_RES_tensor_vector()
    elif name == 'CNN_GRU':
        env = Env_CNN_GRU()

    else:
        raise Exception('error')
    return env


def resume_training(model, args):
    model_latest_file = os.path.join(args.model_path, 'model_latest.pth')
    if os.path.isfile(model_latest_file):
        model.load_state_dict(torch.load(model_latest_file))
        print('Resume the training from {}'.format(model_latest_file))
    else:
        print('Resume missing, there are not files!')


def model_train(env, args):
    if args.net_name in ['RES_tensor', 'RES_tensor_map']:
        model = ResNet(env.n_channels, ArgsResNet['midplanes'], ArgsResNet['outplanes'],
                       env.frame_size[0] * env.frame_size[1], ArgsResNet['BLOCKS_num'])
        if args.resume:
            resume_training(model, args)
        TrainSpatial.train(model, env, args)
    elif args.net_name in ['RES_tensor_vector']:
        model = ResNet_double(env.n_channels,  ArgsResNet_double['midplanes'], ArgsResNet_double['outplanes'],
                              ArgsResNet_double['BLOCKS_num'], env.n_features, ArgsResNet_double['out_vector'],
                              env.frame_size[0] * env.frame_size[1])
        if args.resume:
            resume_training(model, args)
        TrainGlobalSpacial.train(model, env, args)
    elif args.net_name in ['CNN_GRU', 'CNN_GRU_stage']:
        if not args.deep_of_CNN_GRU:
            model = CNNGRUNet(env.n_channels, ArgsCNNGRUNet['midplanes'], ArgsCNNGRUNet['outplanes'], env.n_features,
                              ArgsCNNGRUNet['out_vector'], ArgsCNNGRUNet['in_rnn'], ArgsCNNGRUNet['out_rnn'],
                              env.frame_size[0] * env.frame_size[1])
            if args.resume:
                resume_training(model, args)
            TrainGlobalSpacial.train(model, env, args)
        else:
            cnn_num = args.deep_of_CNN_GRU[0]
            gru_num = args.deep_of_CNN_GRU[1]
            model_name = 'CNNGRUNet_{}_{}'.format(cnn_num, gru_num)
            model = globals()[model_name]\
                (env.n_channels, ArgsCNNGRUNet['midplanes'], ArgsCNNGRUNet['outplanes'], env.n_features, \
                 ArgsCNNGRUNet['out_vector'], ArgsCNNGRUNet['in_rnn'], ArgsCNNGRUNet['out_rnn'], env.frame_size[0] * env.frame_size[1])
            if args.resume:
                resume_training(model, args)
            TrainGlobalSpacial.train(model, env, args)
    else:
        raise Exception('error')


def model_test(env, path, args):
    if args.net_name in ['RES_tensor', 'RES_tensor_map']:
        model = ResNet(env.n_channels, ArgsResNet['midplanes'], ArgsResNet['outplanes'],
                       env.frame_size[0] * env.frame_size[1], ArgsResNet['BLOCKS_num'])
        model.load_state_dict(torch.load(path))
        if args.phrase == 'test':
            result = TrainSpatial.test(model, env, args)
        elif args.phrase == 'test_pro':
            result = TrainSpatial.test_pro(model, env, args)
        elif args.phrase == 'produe_feature_data':
            result = TrainSpatial.predect(model, env, args)

    elif args.net_name in ['RES_tensor_vector']:
        model = ResNet_double(env.n_channels,  ArgsResNet_double['midplanes'], ArgsResNet_double['outplanes'],
                              ArgsResNet_double['BLOCKS_num'], env.n_features, ArgsResNet_double['out_vector'],
                              env.frame_size[0] * env.frame_size[1])
        model.load_state_dict(torch.load(path))
        if args.phrase == 'test':
            result = TrainGlobalSpacial.test(model, env, args)
        elif args.phrase == 'test_pro':
            result = TrainGlobalSpacial.test_pro(model, env, args)
        elif args.phrase == 'produe_feature_data':
            result = TrainGlobalSpacial.predect(model, env, args)
    elif args.net_name in ['CNN_GRU']:
        if not args.deep_of_CNN_GRU:
            model = CNNGRUNet(env.n_channels, ArgsCNNGRUNet['midplanes'], ArgsCNNGRUNet['outplanes'], env.n_features,
                              ArgsCNNGRUNet['out_vector'], ArgsCNNGRUNet['in_rnn'], ArgsCNNGRUNet['out_rnn'],
                              env.frame_size[0] * env.frame_size[1])
            model.load_state_dict(torch.load(path))
            if args.phrase == 'test':
                result = TrainGlobalSpacial.test(model, env, args)
            elif args.phrase == 'test_pro':
                result = TrainGlobalSpacial.test_pro(model, env, args)
            elif args.phrase == 'produe_feature_data':
                result = TrainGlobalSpacial.predect(model, env, args)

        else:
            cnn_num = args.deep_of_CNN_GRU[0]
            gru_num = args.deep_of_CNN_GRU[1]
            model_name = 'CNNGRUNet_{}_{}'.format(cnn_num, gru_num)
            model = globals()[model_name]\
                (env.n_channels, ArgsCNNGRUNet['midplanes'], ArgsCNNGRUNet['outplanes'], env.n_features,
                              ArgsCNNGRUNet['out_vector'], ArgsCNNGRUNet['in_rnn'], ArgsCNNGRUNet['out_rnn'],
                              env.frame_size[0] * env.frame_size[1])
            model.load_state_dict(torch.load(path))
            if args.phrase == 'test':
                result = TrainGlobalSpacial.test(model, env, args)
            elif args.phrase == 'test_pro':
                result = TrainGlobalSpacial.test_pro(model, env, args)
            elif args.phrase == 'produe_feature_data':
                result = TrainGlobalSpacial.predect(model, env, args)
    else:
        raise Exception('error')
    return result


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
    parser = argparse.ArgumentParser(description='Enemy position prediction by the features of tensor: Wargame')
    parser.add_argument('--net_name', type=str, default='RES_tensor_vector',
                        help='which type of the net is: RES_tensor or RES_tensor_map or RES_tensor_vector \
                        or CNN_GRU'),
    parser.add_argument('--deep_of_CNN_GRU', type=list, default=None,
                        help='experiment about the depth of CNN_GRU, default=None, means not experiment, else,[5,1],\
                        or [5,2],or [7,1],or [7,2],or [9,1],or [9,2]'),
    parser.add_argument('--piece_name', type=str, default='all',
                        help='the name of the chess pieces: tank1, tank2, car1, car2, soldier1, soldier2, all'),
    parser.add_argument('--replays_path', default='../../data/train_test/feature_tensor_vector',
                        help='Path for training, and test set')
    parser.add_argument('--race', default='red', help='Which race? (default: red)')
    parser.add_argument('--enemy_race', default='blue', help='Which the enemy race? (default: blue)')
    parser.add_argument('--phrase', type=str, default='train',
                        help='train|test_pro|test|produe_feature_data(default: train)')
    parser.add_argument('--gpu_id', default=0, type=int, help='Which GPU to use [-1 indicate CPU] (default: 0)')

    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--n_steps', type=int, default=10, help='# of forward steps (default: 10)')
    parser.add_argument('--load_steps', type=int, default=1, help='# the intervel steps of the frame of data (default: 1)')
    parser.add_argument('--n_replays', type=int, default=32, help='# of replays (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=15, help='# of epoches (default: 10)')

    parser.add_argument('--save_intervel', type=int, default=50000,
                        help='Frequency of model saving (default: 50000)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume the training (default: False)')
    args = parser.parse_args()
    if args.deep_of_CNN_GRU:
        args.name = args.net_name + '_' + str(args.deep_of_CNN_GRU[0]) + \
                    '_' + str(args.deep_of_CNN_GRU[1])
    else:
        args.name = args.net_name
    args.save_path = os.path.join('checkpoints', args.name, args.piece_name)
    args.model_path = os.path.join(args.save_path, 'snapshots')
    piece_name_to_offset = {'all': 0, 'tank1': 0, 'tank2': 1, 'car1': 2, 'car2': 3, 'soldier1': 4, 'soldier2': 5}

    args.piece_offset = piece_name_to_offset[args.piece_name]
    if args.net_name == 'RES_tensor_vector':
        args.n_steps = 1
        args.load_steps = 1

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

        env = get_env(args.net_name)
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format(args.phrase)))
        root = os.path.join('../')

        env.init(path_replays, root, args)

        model_train(env, args)

    elif 'test' == args.phrase:
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
                env = get_env(args.net_name)
                path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format(args.phrase)))
                root = os.path.join('../')
                args.n_epochs = 1
                args.n_replays = 1
                env.init(path_replays, root, args)

                action_pre_per_replay, action_gt_per_replay = model_test(env, path, args)
                mean_acc, n_stage_acc = [], []
                for piece in range(6):
                    result = (action_pre_per_replay[piece], action_gt_per_replay[piece])
                    mean_acc_piece, n_stage_acc_piece = show_test_result(args.name, args.phrase, result, title=str(len(paths)-1) + '-' + str(piece))
                    mean_acc.append(mean_acc_piece)
                    n_stage_acc.append(n_stage_acc_piece)
                dic = {'action_pre_per_replay': action_pre_per_replay,
                       'action_gt_per_replay': action_gt_per_replay,
                       'mean_acc': mean_acc,
                       'n_stage_acc': n_stage_acc}
                test_result.append(dic)
                path_list.append(path)
            else:
                with open(os.path.join(test_result_path, 'test_result'), 'wb') as f:
                    f.write(pickle.dumps(test_result))
                with open(os.path.join(test_result_path, 'path_list'), 'wb') as f:
                    f.write(pickle.dumps(path_list))
                print('Test ending')
                break
    elif 'test_pro' == args.phrase:
        test_result_path = os.path.join(args.save_path, args.phrase)
        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)
        #  res  1921472
        #  res_map 1921152
        #  res_vector 1800576
        #  GRU 1800519

        if args.net_name == 'RES_tensor' or args.net_name == 'RES_tensor_map':
            number = 1921472
        elif args.net_name == 'RES_tensor_map':
            number = 1921152
        elif args.net_name == 'RES_tensor_vector' :
            number = 1800576
        elif args.net_name == 'CNN_GRU':
            number = 1800519

        path = os.path.join(args.model_path, 'model_iter_{}.pth'.format(number))

        print('Testing {} ...'.format(path))
        env = get_env(args.net_name)
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format('test')))
        root = os.path.join('../')
        args.n_epochs = 1
        args.n_replays = 1
        env.init(path_replays, root, args)

        action_pre_per_replay, action_gt_per_replay = model_test(env, path, args)

        dic1 = {'action_pre_per_replay': action_pre_per_replay,
               'action_gt_per_replay': action_gt_per_replay}
        # 'mean_acc': mean_acc,
        # 'n_stage_acc': n_stage_acc}

        with open(os.path.join(test_result_path, 'test_result1'), 'wb') as f:
            f.write(pickle.dumps(dic1))

        mean_acc, n_stage_acc = [], []
        for piece in range(6):
            result = (action_pre_per_replay[piece], action_gt_per_replay[piece])
            mean_acc_piece, n_stage_acc_piece = show_test_pro_result(args.name, args.phrase, result)
            mean_acc.append(mean_acc_piece)
            n_stage_acc.append(n_stage_acc_piece)


        dic = {'action_pre_per_replay': action_pre_per_replay,
               'action_gt_per_replay': action_gt_per_replay,
               'mean_acc': mean_acc,
               'n_stage_acc': n_stage_acc}


        with open(os.path.join(test_result_path, 'test_result'), 'wb') as f:
            f.write(pickle.dumps(dic))
        print('Test_pro ending')

    elif 'produe_feature_data' == args.phrase:
        save_data_path = os.path.join('../../data/predected_feature_data', args.net_name)
        if not os.path.isdir(save_data_path):
            os.makedirs(save_data_path)
        args.save_data_path = save_data_path
        #  res  1921472
        #  res_map 1921152
        #  res_vector 1800576
        #  GRU 1800519
        if args.net_name == 'RES_tensor' or args.net_name == 'RES_tensor_map':
            number = 1921472
        elif args.net_name == 'RES_tensor_map':
            number = 1921152
        elif args.net_name == 'RES_tensor_vector':
            number = 1800576
        elif args.net_name == 'CNN_GRU':
            number = 1800519
        path = os.path.join(args.model_path, 'model_iter_{}.pth'.format(number))

        print('Testing data')
        env = get_env(args.net_name)
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format('test')))
        root = os.path.join('../')
        args.n_epochs = 1
        args.n_replays = 1
        env.init(path_replays, root, args)

        result = model_test(env, path, args)

        print('Training data')
        env = get_env(args.net_name)
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format('train')))
        root = os.path.join('../')
        args.n_epochs = 1
        args.n_replays = 1
        env.init(path_replays, root, args)

        result = model_test(env, path, args)



if __name__ == '__main__':
    main()