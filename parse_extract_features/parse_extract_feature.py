import pandas as pd
import numpy as np
import os
import copy
from scipy import sparse
from game_state import GameState
from state_to_feature import StateToFeature
from util.util import cvtHexOffset2Int6loc

MapHeight,MapLength = 66, 51


def load_files():
    preprocess_file_head = '../data/preprocess_data/'
    replay_file_head = '../data/replay_data/urban_terrain/'
    map_file_head = '../data/map/'
    csv_roomrecord= pd.read_csv(os.path.join(preprocess_file_head + "select_preprocess.csv"))
    csv_object = pd.read_csv(os.path.join(replay_file_head + "object.csv"))
    LOS = pd.read_csv(map_file_head + 'LOS.csv').set_index(['bop_type', 'grid_pos'])
    game_map = pd.read_csv(map_file_head + 'map83.csv')
    return csv_roomrecord, csv_object, LOS, game_map


def extract_map_feature(game_map):
    map_height = np.zeros((MapHeight, MapLength))
    map_urban = np.zeros_like(map_height)
    map_road = np.zeros_like(map_height)
    map_normal = np.zeros_like(map_height)
    for row in range(MapHeight):
        for col in range(MapLength):
            Int6loc = cvtHexOffset2Int6loc(row, col)
            map_height[row, col] = int(game_map[game_map['MapID'] == Int6loc]['GroundID'])
            cond = int(game_map[game_map['MapID'] == Int6loc]['Cond'])
            grid_type = int(game_map[game_map['MapID'] == Int6loc]['GridType'])
            if cond == 7 or (cond == 0 and grid_type == 3):
                map_urban[row, col] = 1
            if grid_type == 2:
                map_road[row, col] = 1
            if cond == 0 and grid_type == 0:
                map_normal[row, col] = 1
    map_height = (map_height - map_height.min()) / (map_height.max() - map_height.min())
    return map_height, map_urban, map_road, map_normal


def Update_enemy_position(new_position, old_position):
    for piece in range(6):
        if (new_position[piece, :, :] == 1).any():
            old_position[2*piece + 1, np.where(old_position[2*piece, :, :] == 1)] = 1
            old_position[2*piece, :, :] = new_position[piece, :, :]


def Update_enemy_state(new_state, old_state):
    for piece in range(6):
        if (new_state[piece, :].sum() > 1) and (new_state[piece, :] != old_state[piece, :]).any():
            old_state[piece, :] = new_state[piece, :]


def StateFeature_two_stage(bops_odd, bops_even, bops_last_pos, LOS, enemy_color=1, *map_args):
    # enemy_color = 1   敌方棋子颜色为蓝色
    spatial_states_np = []
    flag = False
    if enemy_color:
        bops_this_stage = bops_odd
        bops_last_stage = bops_even
    else:
        bops_this_stage = bops_even
        bops_last_stage = bops_odd

    game_state = StateToFeature(LOS, enemy_color)
    states_enemy_last_stage = []
    for state in bops_last_stage:
        game_state.update(state, bops_last_pos)
        states_enemy_last_stage.append(game_state.enemy_to_tensor())

    states_enemy_tensor = np.zeros((12, MapHeight, MapLength))  # 初始化提取的敌方位置特征
    if states_enemy_last_stage:  # 有敌方位置数据
        states_enemy_last_stage = np.asarray(states_enemy_last_stage)
        shape = states_enemy_last_stage.shape
        for bop_i in range(shape[1]):
            for index in range(1, shape[0] + 1):
                if (states_enemy_last_stage[-index, bop_i, :, :] != 0).any():
                    states_enemy_tensor[bop_i * 2, :, :] = states_enemy_last_stage[-index, bop_i, :, :]  # 写入最近一次见到这个棋子的位置
                    states_enemy_sum = states_enemy_last_stage[:-index - 1, bop_i, :, :].sum(axis=0)
                    states_enemy_sum[states_enemy_sum != 0] = 1  # 记录倒数第二次之前见到棋子的轨迹
                    states_enemy_tensor[bop_i * 2 + 1, :, :] = states_enemy_sum  # 写入敌方位置特征
                    break

    states_our_this_stage = np.zeros((6, MapHeight, MapLength))
    states_enemy_this_stage = states_enemy_tensor[(0, 2, 4, 6, 8, 10), :, :]
    for state in bops_this_stage:
        game_state.update(state, bops_last_pos)
        game_state_temp = game_state.to_tensor(*map_args)
        result_step = []
        if not (states_our_this_stage == game_state_temp[1:7, :, :]).all():
            if enemy_color:
                result_step.append(np.zeros((1, MapHeight, MapLength)))
            else:
                result_step.append(np.ones((1, MapHeight, MapLength)))
            result_step.append(game_state_temp[:7, :, :])
            if not (states_enemy_this_stage == game_state_temp[7:13, :, :]).all() and (states_enemy_this_stage == 0).all():
                # update enemy position in our stage
                Update_enemy_position(game_state_temp[7:13, :, :], states_enemy_tensor)
            result_step.append(copy.deepcopy(states_enemy_tensor))
            if map_args:
                result_step.append(game_state_temp[-12:, :, :])
            else:
                result_step.append(game_state_temp[-8:, :, :])
            result_step = np.concatenate(result_step)
            spatial_states_np.append(result_step)
            flag = True

        states_our_this_stage = game_state_temp[1:7, :, :]
        states_enemy_this_stage = game_state_temp[7:13, :, :]
    spatial_states_np = np.asarray(spatial_states_np)
    return flag, spatial_states_np


def StateFeature_vec_ten(bops_odd, bops_even, bops_last_pos, LOS, enemy_color=1, *map_args):
    # enemy_color = 1   敌方棋子颜色为蓝色
    spatial_states_np = []
    global_states_np = []
    flag = False
    if enemy_color:
        bops_this_stage = bops_odd
        bops_last_stage = bops_even
    else:
        bops_this_stage = bops_even
        bops_last_stage = bops_odd

    game_state = StateToFeature(LOS, enemy_color)
    states_enemy_last_stage = []
    global_states_enmemy_last_stage = []
    for state in bops_last_stage:
        game_state.update(state, bops_last_pos)
        states_enemy_last_stage.append(game_state.enemy_to_tensor())
        global_states_enmemy_last_stage.append(game_state.enemy_to_np())

    states_enemy_tensor = np.zeros((12, MapHeight, MapLength))  # 初始化提取的敌方位置特征
    global_states_enemy_np = np.zeros((6, 22))
    global_states_enemy_np[:, 0] = 1
    if states_enemy_last_stage:  # 有敌方位置数据
        states_enemy_last_stage = np.asarray(states_enemy_last_stage)
        global_states_enmemy_last_stage = np.asarray(global_states_enmemy_last_stage)
        shape = states_enemy_last_stage.shape
        for bop_i in range(shape[1]):
            global_states_enemy_np[bop_i, :] = global_states_enmemy_last_stage[-1, bop_i, :]
            for index in range(1, shape[0] + 1):
                if (states_enemy_last_stage[-index, bop_i, :, :] != 0).any():
                    global_states_enemy_np[bop_i, :] = global_states_enmemy_last_stage[-index, bop_i, :]
                    states_enemy_tensor[bop_i * 2, :, :] = states_enemy_last_stage[-index, bop_i, :, :]  # 写入最近一次见到这个棋子的位置
                    states_enemy_sum = states_enemy_last_stage[:-index - 1, bop_i, :, :].sum(axis=0)
                    states_enemy_sum[states_enemy_sum != 0] = 1  # 记录倒数第二次之前见到棋子的轨迹
                    states_enemy_tensor[bop_i * 2 + 1, :, :] = states_enemy_sum  # 写入敌方位置特征
                    break

    states_our_this_stage = np.zeros((6, MapHeight, MapLength))
    states_enemy_this_stage = states_enemy_tensor[(0, 2, 4, 6, 8, 10), :, :]
    for state in bops_this_stage:
        game_state.update(state, bops_last_pos)
        game_state_temp = game_state.to_tensor(*map_args)
        global_state_temp = game_state.to_vector()
        enemy_state_np_temp = game_state.enemy_to_np()
        result_step = []
        global_step = []
        if not (states_our_this_stage == game_state_temp[1:7, :, :]).all():
            if enemy_color:
                result_step.append(np.zeros((1, MapHeight, MapLength)))
            else:
                result_step.append(np.ones((1, MapHeight, MapLength)))
            result_step.append(game_state_temp[1:7, :, :])
            if not (states_enemy_this_stage == game_state_temp[7:13, :, :]).all():
                # update enemy position in our stage
                Update_enemy_position(game_state_temp[7:13, :, :], states_enemy_tensor)
                Update_enemy_state(enemy_state_np_temp, global_states_enemy_np)
            result_step.append(copy.deepcopy(states_enemy_tensor)) ## 'states_enemy_tensor' maybe change in iteral, so append its copy
            if map_args:
                result_step.append(game_state_temp[-12:, :, :])
            else:
                result_step.append(game_state_temp[-8:, :, :])
            result_step = np.concatenate(result_step)
            spatial_states_np.append(result_step)
            flag = True

            global_step.append(global_state_temp[:156])
            global_step.append(copy.deepcopy(global_states_enemy_np).flatten())
            global_step.append([global_state_temp[-1]])
            global_step = np.hstack(global_step)
            global_states_np.append(global_step)
        states_our_this_stage = game_state_temp[1:7, :, :]
        states_enemy_this_stage = game_state_temp[7:13, :, :]
    spatial_states_np = np.asarray(spatial_states_np)
    global_states_np = np.asarray(global_states_np)
    return flag, spatial_states_np, global_states_np


def StateFeature_vec_ten_stage(bops_odd, bops_even, bops_last_pos, LOS, enemy_color=1, *map_args):
    # enemy_color = 1   敌方棋子颜色为蓝色
    spatial_states_np = []
    global_states_np = []
    flag = False
    if enemy_color:
        bops_this_stage = bops_odd
        bops_last_stage = bops_even
    else:
        bops_this_stage = bops_even
        bops_last_stage = bops_odd

    game_state = StateToFeature(LOS, enemy_color)
    states_enemy_last_stage = []
    global_states_enemy_last_stage = []
    for state in bops_last_stage:
        game_state.update(state, bops_last_pos)
        states_enemy_last_stage.append(game_state.enemy_to_tensor())
        global_states_enemy_last_stage.append(game_state.enemy_to_np())

    states_enemy_tensor = np.zeros((12, MapHeight, MapLength))  # 初始化提取的敌方位置特征
    global_states_enemy_np = np.zeros((6, 22))
    global_states_enemy_np[:, 0] = 1
    if states_enemy_last_stage:  # 有敌方位置数据
        states_enemy_last_stage = np.asarray(states_enemy_last_stage)
        global_states_enemy_last_stage = np.asarray(global_states_enemy_last_stage)
        shape = states_enemy_last_stage.shape
        for bop_i in range(shape[1]):
            global_states_enemy_np[bop_i, :]= global_states_enemy_last_stage[-1, bop_i, :]
            for index in range(1, shape[0] + 1):
                if (states_enemy_last_stage[-index, bop_i, :, :] != 0).any():
                    global_states_enemy_np[bop_i, :] = global_states_enemy_last_stage[-index, bop_i, :]
                    states_enemy_tensor[bop_i * 2, :, :] = states_enemy_last_stage[-index, bop_i, :, :]  # 写入最近一次见到这个棋子的位置
                    states_enemy_sum = states_enemy_last_stage[:-index - 1, bop_i, :, :].sum(axis=0)
                    states_enemy_sum[states_enemy_sum != 0] = 1  # 记录倒数第二次之前见到棋子的轨迹
                    states_enemy_tensor[bop_i * 2 + 1, :, :] = states_enemy_sum  # 写入敌方位置特征
                    break

    states_our_this_stage = np.zeros((6, MapHeight, MapLength))
    states_enemy_this_stage = states_enemy_tensor[(0, 2, 4, 6, 8, 10), :, :]
    for state in bops_this_stage:
        game_state.update(state, bops_last_pos)
        game_state_temp = game_state.to_tensor(*map_args)
        global_state_temp = game_state.to_vector()
        enemy_state_np_temp = game_state.enemy_to_np()
        result_step = []
        global_step = []
        if not (states_our_this_stage == game_state_temp[1:7, :, :]).all():
            if enemy_color:
                result_step.append(np.zeros((1, MapHeight, MapLength)))
            else:
                result_step.append(np.ones((1, MapHeight, MapLength)))
            result_step.append(game_state_temp[:7, :, :])
            if not (states_enemy_this_stage == game_state_temp[7:13, :, :]).all():
                # update enemy position in our stage
                Update_enemy_position(game_state_temp[7:13, :, :], states_enemy_tensor)
                Update_enemy_state(enemy_state_np_temp, global_states_enemy_np)

            result_step.append(copy.deepcopy(states_enemy_tensor)) ## 'states_enemy_tensor' maybe change in iteral, so append its copy
            if map_args:
                result_step.append(game_state_temp[-12:, :, :])
            else:
                result_step.append(game_state_temp[-8:, :, :])
            result_step = np.concatenate(result_step)
            spatial_states_np.append(result_step)
            flag = True

            global_step.append(global_state_temp[20:156])
            global_step.append(copy.deepcopy(global_states_enemy_np).flatten())
            global_step.append([global_state_temp[-1]])
            global_step = np.hstack(global_step)
            global_states_np.append(global_step)
        states_our_this_stage = game_state_temp[1:7, :, :]
        states_enemy_this_stage = game_state_temp[7:13, :, :]
    spatial_states_np = np.asarray(spatial_states_np)
    global_states_np = np.asarray(global_states_np)
    return flag, spatial_states_np, global_states_np


def feature_CNN_GRU(csv_roomrecord_groupby, csv_object, LOS, *map_args):
    for (Filename), group in csv_roomrecord_groupby:
        if group.iloc[0]["JmResult"] >= 0:
            game_result = 'redwin'
        else:
            game_result = 'bluewin'
        save_file_name = Filename + '_' + game_result
        print(save_file_name)
        game_state = GameState(Filename, csv_object)
        bops, bops_last_pos = game_state.bops, game_state.bops_last_pos
        stage_groupby = group.groupby(["StageID"])
        states_vector_red = []
        states_tensor_red = []
        states_vector_blue = []
        states_tensor_blue = []
        bops_odd = []
        bops_even = []
        for (stageid), group in stage_groupby:
            group1 = group.sort_values(by=["TimeID", "ObjStep"], axis=0, ascending=[False, False])

            game_state.initCancelKeep()
            bops["stage"] = stageid
            bops["red_win"] = group.iloc[0]["JmResult"]

            bops_one_stage = []
            for indexs, row in group1.iterrows():
                bop_name = game_state.Get_bop_name(row.ObjID)
                if row.ObjNewPos == row.ObjPos:
                    if row.ObjInto != 0:
                        game_state.UpdateOn(row, bop_name)
                    if row.ObjOut != 0:
                        game_state.UpdateOut(row, bop_name)

                    if row.AttackID != 0:
                        game_state.Attack(row, bop_name)
                    if row.CityTake != 0:
                        game_state.Occupy(row)
                    if row.ObjHide != 0:
                        game_state.Hide(row, bop_name)
                    if row.ObjPass == 1:
                        game_state.Pass(row, bop_name)
                    if row.ObjKeepCancel != 0:
                        game_state.KeepCancel(row, bop_name)
                else:
                    game_state.Move(row, bop_name)
                bops_one_stage.append(copy.deepcopy(bops))

            if stageid % 2 == 1:  # 奇数阶段,采集红方预测蓝方棋子信息
                bops_odd = bops_one_stage
                flag, tensor_one_stage_red, vector_one_stage_red = StateFeature_vec_ten(bops_odd, bops_even, bops_last_pos, LOS, 1, *map_args)
                if flag:
                    states_tensor_red.append(tensor_one_stage_red)
                    states_vector_red.append(vector_one_stage_red)
            else:  # 偶数阶段,采集蓝方预测红方棋子信息
                bops_even = bops_one_stage
                flag, tensor_one_stage_blue, vector_one_stage_blue = StateFeature_vec_ten(bops_odd, bops_even, bops_last_pos, LOS, 0, *map_args)
                if flag:
                    states_tensor_blue.append(tensor_one_stage_blue)
                    states_vector_blue.append(vector_one_stage_blue)
        spatial_states_np_red = np.concatenate(states_tensor_red)
        spatial_states_np_blue = np.concatenate(states_tensor_blue)
        global_states_np_red = np.concatenate(states_vector_red)
        global_states_np_blue = np.concatenate(states_vector_blue)
        print(spatial_states_np_blue.shape)
        print(spatial_states_np_red.shape, "--------")
        spatial_states_np_red = spatial_states_np_red.reshape([spatial_states_np_red.shape[0], -1])
        spatial_states_np_blue = spatial_states_np_blue.reshape([spatial_states_np_blue.shape[0], -1])

        print(global_states_np_red.shape)
        print(global_states_np_blue.shape, "--------")
        vs1 = 'red2blue'
        vs2 = 'blue2red'

        path = os.path.join('../data/feature_data/feature_CNN_GRU/')
        if not os.path.isdir(path):
            os.makedirs(path)

        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs1 + '@S'), sparse.csc_matrix(spatial_states_np_red))
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs1 + '@G'), sparse.csc_matrix(global_states_np_red))
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs2 + '@S'), sparse.csc_matrix(spatial_states_np_blue))
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs2 + '@G'), sparse.csc_matrix(global_states_np_blue))


def feature_CNN_GRU_stage(csv_roomrecord_groupby, csv_object, LOS, *map_args):
    for (Filename), group in csv_roomrecord_groupby:
        if group.iloc[0]["JmResult"] >= 0:
            game_result = 'redwin'
        else:
            game_result = 'bluewin'
        save_file_name = Filename + '_' + game_result
        print(save_file_name)
        game_state = GameState(Filename, csv_object)
        bops, bops_last_pos = game_state.bops, game_state.bops_last_pos
        stage_groupby = group.groupby(["StageID"])
        states_vector_red = []
        states_tensor_red = []
        states_vector_blue = []
        states_tensor_blue = []
        bops_odd = []
        bops_even = []
        for (stageid), group in stage_groupby:
            group1 = group.sort_values(by=["TimeID", "ObjStep"], axis=0, ascending=[False, False])

            game_state.initCancelKeep()
            bops["stage"] = stageid
            bops["red_win"] = group.iloc[0]["JmResult"]

            bops_one_stage = []
            for indexs, row in group1.iterrows():
                bop_name = game_state.Get_bop_name(row.ObjID)
                if row.ObjNewPos == row.ObjPos:
                    if row.ObjInto != 0:
                        game_state.UpdateOn(row, bop_name)
                    if row.ObjOut != 0:
                        game_state.UpdateOut(row, bop_name)

                    if row.AttackID != 0:
                        game_state.Attack(row, bop_name)
                    if row.CityTake != 0:
                        game_state.Occupy(row)
                    if row.ObjHide != 0:
                        game_state.Hide(row, bop_name)
                    if row.ObjPass == 1:
                        game_state.Pass(row, bop_name)
                    if row.ObjKeepCancel != 0:
                        game_state.KeepCancel(row, bop_name)
                else:
                    game_state.Move(row, bop_name)
                bops_one_stage.append(copy.deepcopy(bops))

            if stageid % 2 == 1:  # 奇数阶段,采集红方预测蓝方棋子信息
                bops_odd = bops_one_stage
                flag, tensor_one_stage_red, vector_one_stage_red = StateFeature_vec_ten_stage(bops_odd, bops_even, bops_last_pos, LOS, 1, *map_args)
                if flag:
                    states_tensor_red.append(tensor_one_stage_red)
                    states_vector_red.append(vector_one_stage_red)
            else:  # 偶数阶段,采集蓝方预测红方棋子信息
                bops_even = bops_one_stage
                flag, tensor_one_stage_blue, vector_one_stage_blue = StateFeature_vec_ten_stage(bops_odd, bops_even, bops_last_pos, LOS, 0, *map_args)
                if flag:
                    states_tensor_blue.append(tensor_one_stage_blue)
                    states_vector_blue.append(vector_one_stage_blue)
        spatial_states_np_red = np.concatenate(states_tensor_red)
        spatial_states_np_blue = np.concatenate(states_tensor_blue)
        global_states_np_red = np.concatenate(states_vector_red)
        global_states_np_blue = np.concatenate(states_vector_blue)
        print(spatial_states_np_blue.shape)
        print(spatial_states_np_red.shape, "--------")
        spatial_states_np_red = spatial_states_np_red.reshape([spatial_states_np_red.shape[0], -1])
        spatial_states_np_blue = spatial_states_np_blue.reshape([spatial_states_np_blue.shape[0], -1])
        print(global_states_np_red.shape)
        print(global_states_np_blue.shape, "--------")
        vs1 = 'red2blue'
        vs2 = 'blue2red'

        path = os.path.join('../data/feature_data/feature_CNN_GRU_stage/')
        if not os.path.isdir(path):
            os.makedirs(path)
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs1 + '@S'), sparse.csc_matrix(spatial_states_np_red))
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs1 + '@G'), sparse.csc_matrix(global_states_np_red))
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs2 + '@S'), sparse.csc_matrix(spatial_states_np_blue))
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs2 + '@G'), sparse.csc_matrix(global_states_np_blue))


def feature_RES_tensor(csv_roomrecord_groupby, csv_object, LOS):
    for (Filename), group in csv_roomrecord_groupby:
        if group.iloc[0]["JmResult"] >= 0:
            game_result = 'redwin'
        else:
            game_result = 'bluewin'
        save_file_name = Filename + '_' + game_result
        print(save_file_name)
        game_state = GameState(Filename, csv_object)
        bops, bops_last_pos = game_state.bops, game_state.bops_last_pos
        stage_groupby = group.groupby(["StageID"])
        states_tensor_red = []
        states_tensor_blue = []
        bops_odd = []
        bops_even = []
        for (stageid), group in stage_groupby:
            group1 = group.sort_values(by=["TimeID", "ObjStep"], axis=0, ascending=[False, False])

            game_state.initCancelKeep()
            bops["stage"] = stageid
            bops["red_win"] = group.iloc[0]["JmResult"]

            bops_one_stage = []
            for indexs, row in group1.iterrows():
                bop_name = game_state.Get_bop_name(row.ObjID)
                if row.ObjNewPos == row.ObjPos:
                    if row.ObjInto != 0:
                        game_state.UpdateOn(row, bop_name)
                    if row.ObjOut != 0:
                        game_state.UpdateOut(row, bop_name)

                    if row.AttackID != 0:
                        game_state.Attack(row, bop_name)
                    if row.CityTake != 0:
                        game_state.Occupy(row)
                    if row.ObjHide != 0:
                        game_state.Hide(row, bop_name)
                    if row.ObjPass == 1:
                        game_state.Pass(row, bop_name)
                    if row.ObjKeepCancel != 0:
                        game_state.KeepCancel(row, bop_name)
                else:
                    game_state.Move(row, bop_name)
                bops_one_stage.append(copy.deepcopy(bops))

            if stageid % 2 == 1:  # 奇数阶段,采集红方预测蓝方棋子信息
                bops_odd = bops_one_stage
                flag, tensor_one_stage_red, = StateFeature_two_stage(bops_odd, bops_even, bops_last_pos, LOS, 1)
                if flag:
                    states_tensor_red.append(tensor_one_stage_red)
            else:  # 偶数阶段,采集蓝方预测红方棋子信息
                bops_even = bops_one_stage
                flag, tensor_one_stage_blue, = StateFeature_two_stage(bops_odd, bops_even, bops_last_pos, LOS, 0)
                if flag:
                    states_tensor_blue.append(tensor_one_stage_blue)

        spatial_states_np_red = np.concatenate(states_tensor_red)
        spatial_states_np_blue = np.concatenate(states_tensor_blue)
        print(spatial_states_np_blue.shape)
        print(spatial_states_np_red.shape, "--------")
        spatial_states_np_red = spatial_states_np_red.reshape([spatial_states_np_red.shape[0], -1])
        spatial_states_np_blue = spatial_states_np_blue.reshape([spatial_states_np_blue.shape[0], -1])

        vs1 = 'red2blue'
        vs2 = 'blue2red'

        path = os.path.join('../data/feature_data/feature_RES_tensor/')
        if not os.path.isdir(path):
            os.makedirs(path)
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs1 + '@S'), sparse.csc_matrix(spatial_states_np_red))

        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs2 + '@S'), sparse.csc_matrix(spatial_states_np_blue))


def feature_RES_tensor_map(csv_roomrecord_groupby, csv_object, LOS, *map_args):
    for (Filename), group in csv_roomrecord_groupby:
        if group.iloc[0]["JmResult"] >= 0:
            game_result = 'redwin'
        else:
            game_result = 'bluewin'
        save_file_name = Filename + '_' + game_result
        print(save_file_name)
        game_state = GameState(Filename, csv_object)
        bops, bops_last_pos = game_state.bops, game_state.bops_last_pos
        stage_groupby = group.groupby(["StageID"])
        states_tensor_red = []
        states_tensor_blue = []
        bops_odd = []
        bops_even = []
        for (stageid), group in stage_groupby:
            group1 = group.sort_values(by=["TimeID", "ObjStep"], axis=0, ascending=[False, False])

            game_state.initCancelKeep()
            bops["stage"] = stageid
            bops["red_win"] = group.iloc[0]["JmResult"]

            bops_one_stage = []
            for indexs, row in group1.iterrows():
                bop_name = game_state.Get_bop_name(row.ObjID)
                if row.ObjNewPos == row.ObjPos:
                    if row.ObjInto != 0:
                        game_state.UpdateOn(row, bop_name)
                    if row.ObjOut != 0:
                        game_state.UpdateOut(row, bop_name)

                    if row.AttackID != 0:
                        game_state.Attack(row, bop_name)
                    if row.CityTake != 0:
                        game_state.Occupy(row)
                    if row.ObjHide != 0:
                        game_state.Hide(row, bop_name)
                    if row.ObjPass == 1:
                        game_state.Pass(row, bop_name)
                    if row.ObjKeepCancel != 0:
                        game_state.KeepCancel(row, bop_name)
                else:
                    game_state.Move(row, bop_name)
                bops_one_stage.append(copy.deepcopy(bops))

            if stageid % 2 == 1:  # 奇数阶段,采集红方预测蓝方棋子信息
                bops_odd = bops_one_stage
                flag, tensor_one_stage_red = StateFeature_two_stage(bops_odd, bops_even, bops_last_pos, LOS, 1, *map_args)
                if flag:
                    states_tensor_red.append(tensor_one_stage_red)
            else:  # 偶数阶段,采集蓝方预测红方棋子信息
                bops_even = bops_one_stage
                flag, tensor_one_stage_blue = StateFeature_two_stage(bops_odd, bops_even, bops_last_pos, LOS, 0, *map_args)
                if flag:
                    states_tensor_blue.append(tensor_one_stage_blue)

        spatial_states_np_red = np.concatenate(states_tensor_red)
        spatial_states_np_blue = np.concatenate(states_tensor_blue)
        print(spatial_states_np_blue.shape)
        print(spatial_states_np_red.shape, "--------")
        spatial_states_np_red = spatial_states_np_red.reshape([spatial_states_np_red.shape[0], -1])
        spatial_states_np_blue = spatial_states_np_blue.reshape([spatial_states_np_blue.shape[0], -1])

        vs1 = 'red2blue'
        vs2 = 'blue2red'

        path = os.path.join('../data/feature_data/feature_RES_tensor_map/')
        if not os.path.isdir(path):
            os.makedirs(path)
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs1 + '@S'), sparse.csc_matrix(spatial_states_np_red))

        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs2 + '@S'), sparse.csc_matrix(spatial_states_np_blue))


def feature_RES_tensor_vector(csv_roomrecord_groupby, csv_object, LOS, *map_args):
    for (Filename), group in csv_roomrecord_groupby:
        if group.iloc[0]["JmResult"] >= 0:
            game_result = 'redwin'
        else:
            game_result = 'bluewin'
        save_file_name = Filename + '_' + game_result
        print(save_file_name)
        game_state = GameState(Filename, csv_object)
        bops, bops_last_pos = game_state.bops, game_state.bops_last_pos
        stage_groupby = group.groupby(["StageID"])
        states_vector_red = []
        states_tensor_red = []
        states_vector_blue = []
        states_tensor_blue = []
        bops_odd = []
        bops_even = []
        for (stageid), group in stage_groupby:
            group1 = group.sort_values(by=["TimeID", "ObjStep"], axis=0, ascending=[False, False])

            game_state.initCancelKeep()
            bops["stage"] = stageid
            bops["red_win"] = group.iloc[0]["JmResult"]

            bops_one_stage = []
            for indexs, row in group1.iterrows():
                bop_name = game_state.Get_bop_name(row.ObjID)
                if row.ObjNewPos == row.ObjPos:
                    if row.ObjInto != 0:
                        game_state.UpdateOn(row, bop_name)
                    if row.ObjOut != 0:
                        game_state.UpdateOut(row, bop_name)

                    if row.AttackID != 0:
                        game_state.Attack(row, bop_name)
                    if row.CityTake != 0:
                        game_state.Occupy(row)
                    if row.ObjHide != 0:
                        game_state.Hide(row, bop_name)
                    if row.ObjPass == 1:
                        game_state.Pass(row, bop_name)
                    if row.ObjKeepCancel != 0:
                        game_state.KeepCancel(row, bop_name)
                else:
                    game_state.Move(row, bop_name)
                bops_one_stage.append(copy.deepcopy(bops))

            if stageid % 2 == 1:  # 奇数阶段,采集红方预测蓝方棋子信息
                bops_odd = bops_one_stage
                flag, tensor_one_stage_red, vector_one_stage_red = StateFeature_vec_ten_stage(bops_odd, bops_even, bops_last_pos, LOS, 1, *map_args)
                if flag:
                    states_tensor_red.append(tensor_one_stage_red)
                    states_vector_red.append(vector_one_stage_red)
            else:  # 偶数阶段,采集蓝方预测红方棋子信息
                bops_even = bops_one_stage
                flag, tensor_one_stage_blue, vector_one_stage_blue = StateFeature_vec_ten_stage(bops_odd, bops_even, bops_last_pos, LOS, 0, *map_args)
                if flag:
                    states_tensor_blue.append(tensor_one_stage_blue)
                    states_vector_blue.append(vector_one_stage_blue)
        spatial_states_np_red = np.concatenate(states_tensor_red)
        spatial_states_np_blue = np.concatenate(states_tensor_blue)
        global_states_np_red = np.concatenate(states_vector_red)
        global_states_np_blue = np.concatenate(states_vector_blue)
        print(spatial_states_np_blue.shape)
        print(spatial_states_np_red.shape, "--------")
        spatial_states_np_red = spatial_states_np_red.reshape([spatial_states_np_red.shape[0], -1])
        spatial_states_np_blue = spatial_states_np_blue.reshape([spatial_states_np_blue.shape[0], -1])
        print(global_states_np_red.shape)
        print(global_states_np_blue.shape, "--------")
        vs1 = 'red2blue'
        vs2 = 'blue2red'

        path = os.path.join('../data/feature_data/feature_RES_tensor_vector/')
        if not os.path.isdir(path):
            os.makedirs(path)

        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs1 + '@S'), sparse.csc_matrix(spatial_states_np_red))
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs1 + '@G'), sparse.csc_matrix(global_states_np_red))
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs2 + '@S'), sparse.csc_matrix(spatial_states_np_blue))
        sparse.save_npz(os.path.join(path,
                                     save_file_name + '_' + vs2 + '@G'), sparse.csc_matrix(global_states_np_blue))


if __name__ == '__main__':
    csv_roomrecord, csv_object, LOS, game_map = load_files()
    csv_roomrecord_groupby = csv_roomrecord.groupby(["Filename"])
    map_height, map_urban, map_road, map_normal = extract_map_feature(game_map)

    # spatial feature:play's colour 1, stage 1, friendly 6, enemy 12, observe 2, end_pos 6, all is 28*66*51
    # global feature:0
    # feature_RES_tensor(csv_roomrecord_groupby, csv_object, LOS)

    # spatial feature:play's colour 1, stage 1, friendly 6, enemy 12, observe 2, map 4, end_pos 6, all is 32*66*51
    # global feature:0
    # feature_RES_tensor_map(csv_roomrecord_groupby, csv_object, LOS, map_height, map_urban, map_road, map_normal)

    # spatial feature:play's colour 1, stage 1, friendly 6, enemy 12, observe 2, map 4, end_pos 6, all is 32*66*51
    # global feature:city 4, friendly 22*6, enemy 22*6, win 1, all is 269
    # feature_RES_tensor_vector(csv_roomrecord_groupby, csv_object, LOS, map_height, map_urban, map_road, map_normal)

    # spatial feature:play's colour 1, friendly 6, enemy 12, observe 2, map 4, end_pos 6, all is 31*66*51
    # global feature:stage 20, city 4, friendly 22*6, enemy 22*6,  win 1, all is 289
    # feature_CNN_GRU(csv_roomrecord_groupby, csv_object, LOS, map_height, map_urban, map_road, map_normal)

    # spatial feature:play's colour 1, stage 1, friendly 6, enemy 12, observe 2, map 4, end_pos 6, all is 32*66*51
    # global feature:city 4, friendly 22*6, enemy 22*6,  win 1, all is 269
    feature_CNN_GRU_stage(csv_roomrecord_groupby, csv_object, LOS, map_height, map_urban, map_road, map_normal)
