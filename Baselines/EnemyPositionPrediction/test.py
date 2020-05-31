import os
import visdom
import pickle
import argparse

import numpy as np

LAMBDA = 0.99

def calc_action_acc(action_pre, action_gt):

    return np.mean(action_pre == action_gt)

# def calc_weighted_action_acc(action_pre, action_gt, weight):
#     return np.sum((action_pre == action_gt) * np.abs(weight)) / np.sum(np.abs(weight))

def show_test_result(name, phrase, result, steps=10, title=''):
    action_pres, action_gts = result

    ################################## Calc Acc #########################################

    action_pres_np = np.hstack(action_pres)
    action_gts_np = np.hstack(action_gts)

    action_acc = calc_action_acc(action_pres_np, action_gts_np)
    print('\tAction Accuracy: {}%\t'.format(action_acc*100))
    ################################### Plot ###################################################
    vis = visdom.Visdom(env=name + '[{}]'.format(phrase))

    action_pre_result, action_gt_result = [[] for _ in range(steps)], [[] for _ in range(steps)]
    for action_pre, action_gt in zip(action_pres, action_gts):
        if len(action_pre) < steps:
            continue

        step = len(action_pre) // steps
        for s in range(steps):
            action_pre_result[s].append(action_pre[s * step:(s + 1) * step])
            action_gt_result[s].append(action_gt[s * step:(s + 1) * step])

    legend = ['Action']
    X = np.arange(steps).reshape(steps, -1)
    Y = np.zeros((steps, 1))
    # print(action_pre_result, action_gt_result)
    for idx, (action_stage_pres, action_stage_gts) in enumerate(zip(action_pre_result, action_gt_result)):

        action_stage_pres_np = np.hstack(action_stage_pres)
        action_stage_gts_np = np.hstack(action_stage_gts)

        Y[idx, 0] = calc_action_acc(action_stage_pres_np, action_stage_gts_np)

    vis.line(X=X, Y=Y,
             opts=dict(title='Acc[{}]'.format(title), legend=legend), win=title)
    return action_acc, Y[idx, 0]

def show_test_pro_result(name, phrase, result, steps=10, title=''):
    action_pres, action_gts = result
    # print(action_pres)
    ################################## Calc Acc #########################################

    action_pres_np = np.vstack(action_pres)
    action_gts_np = np.hstack(action_gts)
    print(action_pres_np.shape, action_gts_np.shape)
    acc_type = [1,3,5,10,15,50]
    action_acc = []
    for i, num in enumerate(acc_type):
        # acc = []
        # for j in range(len(action_gts_np)):

            # if action_gts_np[j] in action_pres_np[j, -num:]:
            #     acc.append(1)
            # else:
            #     acc.append(0)
        action_acc.append(np.mean([pos in action_pres_np[j, -num:]
                                   for j, pos in enumerate(action_gts_np)]))

    print('\tAction Accuracy: {}\t'.format(action_acc))
    ################################### Plot ###################################################
    vis = visdom.Visdom(env=name + '[{}]'.format(phrase))

    action_pre_result, action_gt_result = [[] for _ in range(steps)], [[] for _ in range(steps)]
    for action_pre, action_gt in zip(action_pres, action_gts):
        if len(action_pre) < steps:
            continue

        step = len(action_pre) // steps
        for s in range(steps):
            action_pre_result[s].append(action_pre[s * step:(s + 1) * step, :])
            action_gt_result[s].append(action_gt[s * step:(s + 1) * step])

    legend = ['Action']
    acc_type = [1, 3, 5, 10, 15, 50]
    X = np.arange(steps)
    Y = np.zeros((steps, len(acc_type)))
    # print(action_pre_result, action_gt_result)
    for idx, (action_stage_pres, action_stage_gts) in enumerate(zip(action_pre_result, action_gt_result)):

        action_stage_pres_np = np.vstack(action_stage_pres)
        action_stage_gts_np = np.hstack(action_stage_gts)

        for i, num in enumerate(acc_type):
            acc = []
            for j in range(len(action_stage_gts_np)):
                if action_stage_gts_np[j] in action_stage_pres_np[j, -num:]:
                    acc.append(1)
                else:
                    acc.append(0)
            Y[idx, i] = np.mean(acc)

    print(Y, Y[:,0].shape, X.shape)
    vis.line(X=X, Y=Y[:,0],
             opts=dict(title='Acc[{}]'.format(title), legend=legend), win=title)
    return action_acc, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wargame : the Enemy position predection for Wargame')
    parser.add_argument('--name', type=str, default='RES_tensor',
                        help='Experiment name. RES_tensor or RES_tensor_map or RES_tensor_vector \
                        or CNN_GRU, All outputs will be stored in checkpoints/[name]/')
    parser.add_argument('--phrase', type=str, default='test_pro',
                        help='test, test_pro')
    parser.add_argument('--piece_name', type=str, default='all',
                        help='the name of the chess pieces: tank1, tank2, car1, car2, soldier1, soldier2, all'),
    args = parser.parse_args()


    if args.phrase == 'test':
        test_path = os.path.join('checkpoints', args.name, args.piece_name, args.phrase)

        result_path = os.path.join(test_path, 'test_result')

        print('reading:' + result_path)

        with open(result_path, 'rb') as f:
            results = pickle.load(f)
        for j, result in enumerate(results):
            for i in range(6):
                data = (result['action_pre_per_replay'][i],
                        result['action_gt_per_replay'][i])
                show_test_result(args.name, args.phrase, data, title='test_{}_{}'.format(j, i))
    elif args.phrase == 'test_pro':
        test_path = os.path.join('checkpoints', args.name, args.piece_name, args.phrase)

        result_path = os.path.join(test_path, 'test_result1')

        print('reading:' + result_path)

        with open(result_path, 'rb') as f:
            result = pickle.load(f)
        for i in range(6):
            data = (result['action_pre_per_replay'][i],
                    result['action_gt_per_replay'][i])
            show_test_pro_result(args.name, args.phrase, data, title='test_pro')


