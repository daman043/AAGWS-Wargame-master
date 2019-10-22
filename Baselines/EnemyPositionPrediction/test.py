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
    for idx, (action_pres, action_gts) in enumerate(zip(action_pre_result, action_gt_result)):

        action_pres_np = np.hstack(action_pres)
        action_gts_np = np.hstack(action_gts)

        Y[idx, 0] = calc_action_acc(action_pres_np, action_gts_np)

    vis.line(X=X, Y=Y,
             opts=dict(title='Acc[{}]'.format(title), legend=legend), win=title)
    return action_acc, Y[idx, 0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2C : Starcraft II')
    parser.add_argument('--name', type=str, default='StarCraft II:TvT[BuildOrder]',
                        help='Experiment name. All outputs will be stored in checkpoints/[name]/')
    parser.add_argument('--phrase', type=str, default='test',
                        help='val|test (default: test)')
    args = parser.parse_args()

    test_path = os.path.join('checkpoints', args.name, args.phrase)
    test_results = sorted([int(os.path.basename(result).split('.')[0].split('_')[-1])
                        for result in os.listdir(test_path)])

    for idx, name_id in enumerate(test_results):
        result_path = os.path.join(test_path, 'model_iter_{}.pth'.format(name_id))
        print('Processing {}/{} ...'.format(idx+1, len(test_results)))
        print('\t'+result_path)

        with open(result_path, 'rb') as f:
            result = pickle.load(f)

        show_test_result(args.name, args.phrase, result, title=idx)