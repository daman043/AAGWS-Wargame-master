import pprint
import numpy as np
from util.util import cvtHexOffset2Int6loc, to_categorical


class StateToFeature(object):
    # state_keys = ['living', 'ObjPass', 'ObjHide', 'ObjKeep', 'ObjTire', 'ObjRound',
    #               'ObjAttack', 'ObjSon', 'ObjBlood', 'ObjStep']

    def __init__(self, LOS, enemy_color=1, size=(66, 51)):
        # Reward
        self.LOS = LOS
        self.enemy_color = enemy_color
        self.size = size
        self.stage = None
        self.city = None
        self.red_win = None

        # the range of friendly units seeing
        self.see_soldier = np.zeros(self.size)
        self.see_car = np.zeros(self.size)
        self.enemy_see = {}
        # Units
        self.friendly_units = {}
        self.enemy_units = {}

        # present position
        self.friendly_position = None
        self.enemy_position = None

        # end position
        self.enemy_end_position = None

    def update(self, state):
        # for k in self.int_vars:
        #     setattr(self, k, state[k])
        # Reward
        self.board = state[0]
        self.score = state[1]
        self.acts = state[2]

        self.stage = self.board['stage']
        self.city = self.board['city']
        self.red_win = self.score['red_win']

        # 求通视范围
        self.see_soldier, self.see_car, self.enemy_see = self.__get_see_range__(self.board)

        # Units
        if self.enemy_color:
            self.friendly_units = {'tank1': self.__dict_to_array__(self.board['red_tank1']['state']),
                                   'tank2': self.__dict_to_array__(self.board['red_tank2']['state']),
                                   'car1': self.__dict_to_array__(self.board['red_car1']['state']),
                                   'car2': self.__dict_to_array__(self.board['red_car2']['state']),
                                   'soldier1': self.__dict_to_array__(self.board['red_soldier1']['state']),
                                   'soldier2': self.__dict_to_array__(self.board['red_soldier2']['state'])}

            self.enemy_units = {'tank1': self.__dict_to_array__(self.board['blue_tank1']['state']),
                                'tank2': self.__dict_to_array__(self.board['blue_tank2']['state']),
                                'car1': self.__dict_to_array__(self.board['blue_car1']['state']),
                                'car2': self.__dict_to_array__(self.board['blue_car2']['state']),
                                'soldier1': self.__dict_to_array__(self.board['blue_soldier1']['state']),
                                'soldier2': self.__dict_to_array__(self.board['blue_soldier2']['state'])}
        else:
            self.friendly_units = {'tank1': self.__dict_to_array__(self.board['blue_tank1']['state']),
                                   'tank2': self.__dict_to_array__(self.board['blue_tank2']['state']),
                                   'car1': self.__dict_to_array__(self.board['blue_car1']['state']),
                                   'car2': self.__dict_to_array__(self.board['blue_car2']['state']),
                                   'soldier1': self.__dict_to_array__(self.board['blue_soldier1']['state']),
                                   'soldier2': self.__dict_to_array__(self.board['blue_soldier2']['state'])}
            self.enemy_units = {'tank1': self.__dict_to_array__(self.board['red_tank1']['state']),
                                'tank2': self.__dict_to_array__(self.board['red_tank2']['state']),
                                'car1': self.__dict_to_array__(self.board['red_car1']['state']),
                                'car2': self.__dict_to_array__(self.board['red_car2']['state']),
                                'soldier1': self.__dict_to_array__(self.board['red_soldier1']['state']),
                                'soldier2': self.__dict_to_array__(self.board['red_soldier2']['state'])}
        # present position
        if self.enemy_color:
            self.friendly_position = {'tank1': self.board['red_tank1']['Pos'],
                                      'tank2': self.board['red_tank2']['Pos'],
                                      'car1': self.board['red_car1']['Pos'],
                                      'car2': self.board['red_car2']['Pos'],
                                      'soldier1': self.board['red_soldier1']['Pos'],
                                      'soldier2': self.board['red_soldier2']['Pos']}
            self.enemy_position = {'tank1': self.board['blue_tank1']['Pos'],
                                   'tank2': self.board['blue_tank2']['Pos'],
                                   'car1': self.board['blue_car1']['Pos'],
                                   'car2': self.board['blue_car2']['Pos'],
                                   'soldier1': self.board['blue_soldier1']['Pos'],
                                   'soldier2': self.board['blue_soldier2']['Pos']}
        else:
            self.friendly_position = {'tank1': self.board['blue_tank1']['Pos'],
                                      'tank2': self.board['blue_tank2']['Pos'],
                                      'car1': self.board['blue_car1']['Pos'],
                                      'car2': self.board['blue_car2']['Pos'],
                                      'soldier1': self.board['blue_soldier1']['Pos'],
                                      'soldier2': self.board['blue_soldier2']['Pos']}
            self.enemy_position = {'tank1': self.board['red_tank1']['Pos'],
                                   'tank2': self.board['red_tank2']['Pos'],
                                   'car1': self.board['red_car1']['Pos'],
                                   'car2': self.board['red_car2']['Pos'],
                                   'soldier1': self.board['red_soldier1']['Pos'],
                                   'soldier2': self.board['red_soldier2']['Pos']}


    def __get_see_range__(self, state):
        see_soldier = np.ones(self.size)
        see_car = np.ones(self.size)
        enemy_see = {}
        if self.enemy_color:
            friendly_position = [state['red_tank1']['Pos'], state['red_tank2']['Pos'],
                                 state['red_car1']['Pos'], state['red_car2']['Pos'],
                                 state['red_soldier1']['Pos'], state['red_soldier2']['Pos']]
            int6_array = [cvtHexOffset2Int6loc(row, col) for (row, col) in friendly_position if (row != 0 and col != 0)]
            LOS_array_soldier = np.array(self.LOS.loc[(1, int6_array), :]).reshape(-1, self.size[0], self.size[1])
            LOS_array_car = np.array(self.LOS.loc[(2, int6_array), :]).reshape(-1, self.size[0], self.size[1])

            see_soldier[np.sum(LOS_array_soldier, axis=0) != 0] = 0   # 能看见的位置为0，不能看见的地方为1
            see_car[np.sum(LOS_array_car, axis=0) != 0] = 0

            enemy_position = {'tank1': state['blue_tank1']['Pos'],
                              'tank2': state['blue_tank2']['Pos'],
                              'car1': state['blue_car1']['Pos'],
                              'car2': state['blue_car2']['Pos'],
                              'soldier1': state['blue_soldier1']['Pos'],
                              'soldier2': state['blue_soldier2']['Pos']}
            for key, value in enemy_position.items():
                if (key != 'soldier1') or (key != 'soldier2'):
                    if value[0] != 0 and value[1] != 0:
                        if see_car[value[0], value[1]] == 0:  #为0表示不能看见
                            enemy_see[key] = True
                        else:
                            enemy_see[key] = False
                    else:
                        enemy_see[key] = False
                else:
                    if value[0] != 0 and value[1] != 0:
                        if see_soldier[value[0], value[1]] == 0:
                            enemy_see[key] = True
                        else:
                            enemy_see[key] = False
                    else:
                        enemy_see[key] = False
        else:
            friendly_position = [state['blue_tank1']['Pos'], state['blue_tank2']['Pos'],
                                 state['blue_car1']['Pos'], state['blue_car2']['Pos'],
                                 state['blue_soldier1']['Pos'], state['blue_soldier2']['Pos']]
            int6_array = [cvtHexOffset2Int6loc(row, col) for (row, col) in friendly_position if (row != 0 and col != 0)]
            LOS_array_soldier = np.array(self.LOS.loc[(1, int6_array), :]).reshape(-1, self.size[0], self.size[1])
            LOS_array_car = np.array(self.LOS.loc[(2, int6_array), :]).reshape(-1, self.size[0], self.size[1])

            see_soldier[np.sum(LOS_array_soldier, axis=0) != 0] = 0
            see_car[np.sum(LOS_array_car, axis=0) != 0] = 0

            enemy_position = {'tank1': state['red_tank1']['Pos'], 'tank2': state['red_tank2']['Pos'],
                              'car1': state['red_car1']['Pos'], 'car2': state['red_car2']['Pos'],
                              'soldier1': state['red_soldier1']['Pos'], 'soldier2': state['red_soldier2']['Pos']}
            enemy_see = {}
            for key, value in enemy_position.items():
                if (key != 'soldier1') or (key != 'soldier2'):
                    if value[0] != 0 and value[1] != 0:
                        if see_car[value[0], value[1]] == 0:
                            enemy_see[key] = True
                        else:
                            enemy_see[key] = False
                    else:
                        enemy_see[key] = False
                else:
                    if value[0] != 0 and value[1] != 0:
                        if see_soldier[value[0], value[1]] == 0:
                            enemy_see[key] = True
                        else:
                            enemy_see[key] = False
                    else:
                        enemy_see[key] = False

        return see_soldier, see_car, enemy_see

    @staticmethod
    def __set_to_array__(set_var, key2id):
        result = np.zeros(len(key2id))
        for key in set_var:
            result[key2id[key]] = 1
        return result

    def __set_to_tensor__(self, set_var, enemy=False):
        result = []
        for key, value in set_var.items():
            temp = np.zeros(self.size)
            if enemy:
                if self.enemy_see[key]:
                    temp[value[0], value[1]] = 1
                elif value[0] != 0 and value[1] != 0:  ##棋子活着，但看不见
                    temp[:,:] = 1.0/(self.size[0] * self.size[1])
                else:   ##棋子死亡
                    pass
            else:
                # if value[0] != 0 and value[1] != 0:  # 活着的棋子显示位置
                temp[value[0], value[1]] = 1
            result.append(temp)
        return np.stack(result)

    def __dict_to_array__(self, dict_var, enemy=False):
        result = []
        for key, value in dict_var.items():
            if enemy:
                if not self.enemy_see[key]:
                    value[1:] = 0
            result.append(value)
        return np.hstack(result)

    def __dict_to_np__(self, dict_var, enemy=False):
        result = []
        for key, value in dict_var.items():
            if enemy:
                if not self.enemy_see[key]:
                    value[1:] = 0
            result.append(value)
        return np.array(result)

    def to_vector(self):
        result = [to_categorical(self.stage - 1, 20),
                  self.city,
                  self.__dict_to_array__(self.friendly_units),
                  self.__dict_to_array__(self.enemy_units, enemy=True),
                  self.red_win
                  ]
        return np.hstack(result)

    def enemy_to_np(self):
        return self.__dict_to_np__(self.enemy_units, enemy=True)

    def friendly_to_tensor(self):
        return self.__set_to_tensor__(self.friendly_position)

    def enemy_to_tensor(self):
        return self.__set_to_tensor__(self.enemy_position, enemy=True)

    def to_tensor(self, *map_args):
        result = []
        result.append(self.__set_to_tensor__(self.friendly_position))
        result.append(self.__set_to_tensor__(self.enemy_position, enemy=True))
        result.append(np.expand_dims(self.see_car, axis=0))
        result.append(np.expand_dims(self.see_soldier, axis=0))

        if map_args:
            for map_feature in map_args:
                map_feature = np.expand_dims(map_feature, axis=0)
                result.append(map_feature)
        result.append(self.__set_to_tensor__(self.enemy_position))

        return np.concatenate(result)

    def acts_to_np(self):
        result = []
        for key, value in self.acts.items():
            if self.enemy_color:
                if key.split("_")[0] == 'red':      # record red action
                    one_piece_act = self.__dict_to_array__(value)
                    result.append(one_piece_act)
            else:
                if key.split("_")[0] == 'blue':      # record blue action
                    one_piece_act = self.__dict_to_array__(value)
                    result.append(one_piece_act)
        result.append(value)
        return np.hstack(result)

    def __str__(self):
        return pprint.pformat(self.__dict__)
