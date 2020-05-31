import os
import json
import glob
import numpy as np


# FLAGS = {'states_direct': 'feature_tensor_vector',
#          'root': '../data/feature_data/',
#          'save_path': '../data/train_test/',
#          'ratio': '8.5:1.5',
#          'seed': 1}
FLAGS_list = [
        {'states_direct': 'feature_tensor_vector',
         'root': '../data/feature_data/',
         'save_path': '../data/train_test/',
         'ratio': '8.5:1.5',
         'seed': 1}
        {'states_direct': 'RES_tensor',
         'root': '../data/predected_feature_data/',
         'save_path': '../data/train_test/',
         'ratio': '8.5:1.5',
         'seed': 1},
        {'states_direct': 'RES_tensor_map',
         'root': '../data/predected_feature_data/',
         'save_path': '../data/train_test/',
         'ratio': '8.5:1.5',
         'seed': 1},
        {'states_direct': 'RES_tensor_vector',
         'root': '../data/predected_feature_data/',
         'save_path': '../data/train_test/',
         'ratio': '8.5:1.5',
         'seed': 1},
        {'states_direct': 'CNN_GRU',
         'root': '../data/predected_feature_data/',
         'save_path': '../data/train_test/',
         'ratio': '8.5:1.5',
         'seed': 1}
        ]




def save(replays, prefix, folder):
    print('{}/{}: {}'.format(folder, prefix, len(replays)))
    with open(os.path.join(folder, prefix + '.json'), 'w') as f:
        json.dump(replays, f)


def split(ratio, feature_directory, save_directory):
    result_redwin = []
    result_bluewin = []
    file_recorded = []
    for file in glob.glob(os.path.join(feature_directory, '*@S.npz')):
        print(file)
        file_name = os.path.split(file)[-1]
        # skip the file name which have recorded

        pre_file = file_name.split('_')[0]
        if pre_file in file_recorded:
            # print('skip')  # skip the file have recorded
            continue
        who_win = file_name.split('_')[1]
        vs = file_name.split('_')[2][:8]

        if who_win == 'redwin':
            replay_path_dict = {}
            parsed_replays_info_red = {}
            parsed_replays_info_blue = {}
            if vs == 'red2blue':
                parsed_replays_info_red['state_S'] = os.path.join(feature_directory, file_name)
                parsed_replays_info_red['state_G'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'red2blue@G.npz')
                parsed_replays_info_red['state_A'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'red2blue@A.npz')
                parsed_replays_info_blue['state_S'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'blue2red@S.npz')
                parsed_replays_info_blue['state_G'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'blue2red@G.npz')
                parsed_replays_info_blue['state_A'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'blue2red@A.npz')

                file_recorded.append(pre_file)
                replay_path_dict['red'] = parsed_replays_info_red
                replay_path_dict['blue'] = parsed_replays_info_blue
            elif vs == 'blue2red':
                parsed_replays_info_blue['state_S'] = os.path.join(feature_directory, file_name)
                parsed_replays_info_blue['state_G'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'blue2red@G.npz')
                parsed_replays_info_blue['state_A'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'blue2red@A.npz')

                parsed_replays_info_red['state_S'] = os.path.join(feature_directory,
                                                                  pre_file + '_' + who_win + '_' + 'red2blue@S.npz')
                parsed_replays_info_red['state_G'] = os.path.join(feature_directory,
                                                                  pre_file + '_' + who_win + '_' + 'red2blue@G.npz')
                parsed_replays_info_red['state_A'] = os.path.join(feature_directory,
                                                                  pre_file + '_' + who_win + '_' + 'red2blue@A.npz')


                file_recorded.append(pre_file)
                replay_path_dict['blue'] = parsed_replays_info_blue
                replay_path_dict['red'] = parsed_replays_info_red
            else:
                Exception('error')
            result_redwin.append(replay_path_dict)
        elif who_win == 'bluewin':
            replay_path_dict = {}
            parsed_replays_info_red = {}
            parsed_replays_info_blue = {}
            if vs == 'red2blue':
                parsed_replays_info_red['state_S'] = os.path.join(feature_directory, file_name)
                parsed_replays_info_red['state_G'] = os.path.join(feature_directory,
                                                                  pre_file + '_' + who_win + '_' + 'red2blue@G.npz')
                parsed_replays_info_red['state_A'] = os.path.join(feature_directory,
                                                                  pre_file + '_' + who_win + '_' + 'red2blue@A.npz')
                parsed_replays_info_blue['state_S'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'blue2red@S.npz')
                parsed_replays_info_blue['state_G'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'blue2red@G.npz')
                parsed_replays_info_blue['state_A'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'blue2red@A.npz')

                file_recorded.append(pre_file)

                replay_path_dict['red'] = parsed_replays_info_red
                replay_path_dict['blue'] = parsed_replays_info_blue
            elif vs == 'blue2red':
                parsed_replays_info_blue['state_S'] = os.path.join(feature_directory, file_name)
                parsed_replays_info_blue['state_G'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'blue2red@G.npz')
                parsed_replays_info_blue['state_A'] = os.path.join(feature_directory,
                                                                   pre_file + '_' + who_win + '_' + 'blue2red@A.npz')

                parsed_replays_info_red['state_S'] = os.path.join(feature_directory,
                                                                  pre_file + '_' + who_win + '_' + 'red2blue@S.npz')
                parsed_replays_info_red['state_G'] = os.path.join(feature_directory,
                                                                  pre_file + '_' + who_win + '_' + 'red2blue@G.npz')
                parsed_replays_info_red['state_A'] = os.path.join(feature_directory,
                                                                  pre_file + '_' + who_win + '_' + 'red2blue@A.npz')

                file_recorded.append(pre_file)
                replay_path_dict['blue'] = parsed_replays_info_blue
                replay_path_dict['red'] = parsed_replays_info_red
            else:
                Exception('error')
            result_bluewin.append(replay_path_dict)
        else:
            raise Exception('error')

    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)
    np.random.seed(1)
    train_end_redwin = int(len(result_redwin) * ratio[0])

    train_end_bluewin = int(len(result_bluewin) * ratio[0])

    np.random.shuffle(result_redwin)
    np.random.shuffle(result_bluewin)
    result_train = result_redwin[:train_end_redwin] + result_bluewin[:train_end_bluewin]
    result_test = result_redwin[train_end_redwin:] + result_bluewin[train_end_bluewin:]
    # np.random.shuffle(result_train)
    # np.random.shuffle(result_test)
    save(result_train, 'train', save_directory)
    save(result_test, 'test', save_directory)


def main():

    for FLAGS in FLAGS_list:
        np.random.seed(FLAGS['seed'])
        ratio = np.asarray([float(i) for i in FLAGS['ratio'].split(':')])
        ratio /= np.sum(ratio)

        feature_directory = os.path.join(FLAGS['root'], FLAGS['states_direct'])
        save_directory = os.path.join(FLAGS['save_path'], FLAGS['states_direct'])
        split(ratio, feature_directory, save_directory)


if __name__ == '__main__':
    main()