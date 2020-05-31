import os
import json
from collections import namedtuple
import numpy as np
from scipy import sparse
from tqdm import tqdm
from util.util import all_true

class BatchEnv(object):
    def __init__(self):
        pass

    def init(self, path, root, args):
        # race, enemy_race, n_replays = 5, n_steps = 1, epochs = 10, seed = None, piece_offset = 0
        np.random.seed(args.seed)
        if 'GRU' in args.net_name.split('_'):
            self.shuffle = False
            print('the net is {}, the relay featrues will not be shuffled'.format(args.net_name))
        elif args.phrase == 'test' or args.phrase == 'test_pro':
            self.shuffle = False
            print('the net is {}, the relay featrues will not be shuffled when testing'.format(args.net_name))
        else:
            self.shuffle = True
            print('the net is {}, the relay featrues will be shuffled'.format(args.net_name))

        with open(path) as f:
            replays = json.load(f)
        self.replays = self.__generate_replay_list__(replays, root, args.race, args.enemy_race)
        self.race = args.race
        self.enemy_race = args.enemy_race
        self.piece_offset = args.piece_offset
        self.n_replays = args.n_replays
        self.n_steps = args.n_steps
        self.load_steps = args.load_steps
        self.epochs = args.n_epochs
        self.phrase = args.phrase

        self.epoch = -1
        self.steps = 0
        self.replay_idx = -1
        self.replay_list = [None for _ in range(self.n_replays)]
        ## Display Progress Bar
        self.epoch_pbar = tqdm(total=self.epochs, desc='Epoch')
        self.replay_pbar = None

        self.Feature = namedtuple('Feature', ['S', 'G', 'A'])
        self.car_see_range = 13
        self.soldier_see_range = 14
        self.enemy_position = 19

    def __generate_replay_list__(self, replays, root, race, enemy_race):
        result = []
        for path_dict in replays:
            result.append([os.path.join(root, path_dict[race]['state_S']),
                           os.path.join(root, path_dict[race]['state_G']),
                           os.path.join(root, path_dict[race]['state_A'])])
            result.append([os.path.join(root, path_dict[enemy_race]['state_S']),
                           os.path.join(root, path_dict[enemy_race]['state_G']),
                           os.path.join(root, path_dict[enemy_race]['state_A'])])
        return result

    def __init_epoch__(self):
        self.epoch += 1
        if self.epoch > 0:
            self.epoch_pbar.update(1)
        if self.epoch == self.epochs:
            return False

        np.random.shuffle(self.replays)
        ## Display Progress Bar
        if self.replay_pbar is not None:
            self.replay_pbar.close()
        self.replay_pbar = tqdm(total=len(self.replays), desc='Replays')
        return True

    def __reset__(self):
        self.replay_idx += 1
        if self.replay_idx % len(self.replays) == 0:
            has_more = self.__init_epoch__()
            if not has_more:
                return None

        path = self.replays[self.replay_idx%len(self.replays)]
        self.path = path
        return self.__load_replay__(path)

    def __load_replay__(self, path):
        replay_dict = {}
        replay_dict['ptr'] = 0
        replay_dict['done'] = False
        replay_dict['states_S'] = np.asarray(sparse.load_npz(path[0]).todense()).reshape([-1, self.all_channels,
                                                                                          self.frame_size[0], self.frame_size[1]])
        replay_dict['states_G'] = np.asarray(sparse.load_npz(path[1]).todense())
        replay_dict['states_A'] = np.asarray(sparse.load_npz(path[2]).todense())

        if self.shuffle:
            c = list(zip(replay_dict['states_S'], replay_dict['states_G'], replay_dict['states_A']))
            np.random.shuffle(c)
            replay_dict['states_S'][:], replay_dict['states_G'][:], replay_dict['states_A'][:] = zip(*c)

        return replay_dict

    def step(self, test_mode=False, **kwargs):
        require_init = [False for _ in range(self.n_replays)]
        for i in range(self.n_replays):
            if self.replay_list[i] is None or self.replay_list[i]['done']:
                if self.replay_list[i] is not None:
                    keys = set(self.replay_list[i].keys())
                    for k in keys:
                        del self.replay_list[i][k]
                self.replay_list[i] = self.__reset__()
                require_init[i] = True
                if self.replay_list[i] is None:
                    return None

        result_n_replays = []
        for i in range(self.n_replays):
            replay_dict = self.replay_list[i]
            features = self.__one_step__(replay_dict, replay_dict['done'])
            result_n_replays.append(features)
        # if test_mode:
        see_n_piece = self.__get_see__(result_n_replays)  # see_n_piece means the ability of observe of all pieces
        return self.__post_process__(result_n_replays), require_init, see_n_piece
        # return self.__post_process__(result_n_replays), require_init

    def n_step(self, test_mode=False, **kwargs):
        require_init = [False for _ in range(self.n_replays)]
        if kwargs['all_init']:
            for i in range(self.n_replays):
                if self.replay_list[i] is None:
                    self.replay_list[i] = self.__reset__()
            all_done = all_true([self.replay_list[j]['done'] for j in range(i+1)])
            if all_done:
                for i in range(self.n_replays):
                    if self.replay_list[i] is not None:
                        keys = set(self.replay_list[i].keys())
                        for k in keys:
                            del self.replay_list[i][k]
                    self.replay_list[i] = self.__reset__()
                    require_init[i] = True
                    if self.replay_list[i] is None:
                        return None
        else:
            for i in range(self.n_replays):
                if self.replay_list[i] is None or self.replay_list[i]['done']:
                    if self.replay_list[i] is not None:
                        keys = set(self.replay_list[i].keys())
                        for k in keys:
                            del self.replay_list[i][k]
                    self.replay_list[i] = self.__reset__()
                    require_init[i] = True
                if self.replay_list[i] is None:
                    return None
        result = []
        for step in range(self.n_steps):
            result_per_step = []
            for i in range(self.n_replays):
                replay_dict = self.replay_list[i]
                features = self.__multi_step__(step, replay_dict, replay_dict['done'])
                result_per_step.append(features)
            result.append(result_per_step)
        # if test_mode:
        see_n_piece = self.__get_see__(result[-1])

        return self.__post_process__(result), require_init, see_n_piece

        # return self.__post_process__(result), require_init

    def __one_step__(self, replay_dict, done):
        states_S = replay_dict['states_S']
        states_G = replay_dict['states_G']
        states_A = replay_dict['states_A']
        feature_shape_S = states_S.shape[1:]
        feature_shape_G = states_G.shape[1:]
        feature_shape_A = states_A.shape[1:]
        if done:
            return self.Feature(np.zeros(feature_shape_S), np.zeros(feature_shape_G), np.zeros(feature_shape_A))

        self.steps += 1
        state_S = states_S[replay_dict['ptr']]
        state_G = states_G[replay_dict['ptr']]
        state_A = states_A[replay_dict['ptr']]

        replay_dict['ptr'] += 1
        if replay_dict['ptr'] == states_S.shape[0]:
            self.replay_pbar.update(1)
            replay_dict['done'] = True
        return self.Feature(state_S, state_G, state_A)

    def __multi_step__(self, step, replay_dict, done):
        states_S = replay_dict['states_S']
        states_G = replay_dict['states_G']
        states_A = replay_dict['states_A']
        feature_shape_S = states_S.shape[1:]
        feature_shape_G = states_G.shape[1:]
        feature_shape_A = states_A.shape[1:]
        if done:
            return self.Feature(np.zeros(feature_shape_S), np.zeros(feature_shape_G), np.zeros(feature_shape_A))

        state_S = states_S[replay_dict['ptr'] + step]
        state_G = states_G[replay_dict['ptr'] + step]
        state_A = states_A[replay_dict['ptr'] + step]
        if step == (self.n_steps - 1):
            self.steps += 1
            replay_dict['ptr'] += self.load_steps
        if (replay_dict['ptr'] + step) == states_S.shape[0]:
            self.replay_pbar.update(1)
            replay_dict['done'] = True
        return self.Feature(state_S, state_G, state_A)

    def __post_process__(self, result, **kwargs):
        raise NotImplementedError

    def __get_see__(self, result_n_replays):
        return self.__see__(self.car_see_range, self.soldier_see_range, self.enemy_position, result_n_replays)

    @staticmethod
    def __see__(car_see_range, soldier_see_range, enemy_position, result_n_replays):
        see_n_replays = []
        for result in result_n_replays:
            if not isinstance(result, np.ndarray):
                result = result.S
            all_see = []
            for piece in range(6):
                if piece <= 3:
                    friendly_see = result[car_see_range, :, :]
                else:
                    friendly_see = result[soldier_see_range, :, :]  # 找到观察范围
                if (result[enemy_position + piece, :, :] == 1).any() and (result[enemy_position + piece, 0, 0] != 1):
                    all_see.append(int(friendly_see[result[enemy_position + piece, :, :] == 1]) == 0)  # 为0，表示能够看见
                else:
                    all_see.append(True)    # if the piece is dead,then can_see = True
            see_n_replays.append(all_see)
        return np.array(see_n_replays, dtype=bool)

    def step_count(self):
        return self.steps

    def close(self):
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
        if self.replay_pbar is not None:
            self.replay_pbar.close()


class Env_RES_tensor(BatchEnv):

    n_channels = 15
    all_channels = 25
    frame_size = [66, 51]

    enemy_position = 19
    map_position = 15

    def __post_process__(self, result, all_data=False):
        result = self.Feature(*zip(*result))
        S = np.asarray(result.S)
        # G = np.asarray(result.G)
        # A = np.asarray(result.A)
        result_return = [S[:, :self.map_position, :, :], S[:, self.enemy_position:, :, :]]
        return result_return


class Env_RES_tensor_map(BatchEnv):
    n_channels = 19
    all_channels = 25
    frame_size = [66, 51]


    def __post_process__(self, result, all_data=False):
        result = self.Feature(*zip(*result))
        S = np.asarray(result.S)
        # G = np.asarray(result.G)
        # A = np.asarray(result.A)
        result_return = [S[:, :self.enemy_position, :, :], S[:, self.enemy_position:, :, :]]
        return result_return

class Env_RES_tensor_vector(BatchEnv):
    """
    # spatial feature:player's colour 1, stage 1, friendly 6, enemy 12, observe 2, map 4, end_pos 6, all is 32*66*51
    # global feature:city 4, friendly 22*6, enemy 22*6,  win 1, all is 269
    """
    n_features = 300
    all_features = 301
    n_channels = 19
    all_channels = 25
    frame_size = [66, 51]


    def __post_process__(self, result, all_data=False):
        result = self.Feature(*zip(*[self.Feature(*zip(*result_per_step)) for result_per_step in result]))
        S = np.asarray(result.S)
        G = np.asarray(result.G)
        # A = np.asarray(result.A)
        result_return = [S[:, :, :self.enemy_position, :, :], G[:, :, :-1], S[:, :, self.enemy_position:, :, :]]
        return result_return


class Env_CNN_GRU(BatchEnv):
    """
    # spatial feature:player's colour 1, friendly 6, enemy 12, observe 2, map 4, end_pos 6, all is 31*66*51
    # global feature:stage 20, city 4, friendly 22*6, enemy 22*6,  win 1, all is 289
    """
    n_features = 300
    all_features = 301
    n_channels = 19
    all_channels = 25
    frame_size = [66, 51]


    def __post_process__(self, result, all_data=False):
        result = self.Feature(*zip(*[self.Feature(*zip(*result_per_step)) for result_per_step in result]))
        S = np.asarray(result.S)
        G = np.asarray(result.G)
        # A = np.asarray(result.A)
        result_return = [S[:, :, :self.enemy_position, :, :], G[:, :, :-1], S[:, :, self.enemy_position:, :, :]]
        return result_return



class Env_Evaluation(BatchEnv):
    """
    # spatial feature:player's colour 1, friendly 6, enemy 12, observe 2, map 4, end_pos 6, all is 31*66*51
    # global feature:stage 20, city 4, friendly 22*6, enemy 22*6,  win 1, all is 289
    """
    n_features = 300
    all_features = 301
    n_channels = 19
    all_channels = 25
    frame_size = [66, 51]


    def __post_process__(self, result, all_data=False):
        result = self.Feature(*zip(*[self.Feature(*zip(*result_per_step)) for result_per_step in result]))
        S = np.asarray(result.S)
        G = np.asarray(result.G)
        # A = np.asarray(result.A)
        result_return = [S[:, :, :self.enemy_position, :, :], G[:, :, :-1], G[:, :, -1]]
        return result_return
