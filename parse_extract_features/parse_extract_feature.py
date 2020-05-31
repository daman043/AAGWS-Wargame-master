import pandas as pd
import numpy as np
import os, json
import copy
from multiprocessing import Pool
from scipy import sparse
from game_state import GameState
from state_to_feature import StateToFeature
from util.util import cvtHexOffset2Int6loc
from plot.plot_3d import plot_3d, plot_2d
MapHeight, MapLength = 66, 51

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



def StateFeature_vec_ten(bops_state, LOS, enemy_color=1, *map_args):
    # enemy_color = 1   敌方棋子颜色为蓝色
    flag = True
    if not bops_state:
        flag = False
    spatial_states_np = []
    global_states_np = []
    acts_np = []

    game_state = StateToFeature(LOS, enemy_color)
    for state in bops_state:
        game_state.update(state)
        spatial_state_temp = game_state.to_tensor(*map_args)
        global_state_temp = game_state.to_vector()
        acts_temp = game_state.acts_to_np()
        result_step = []
        if enemy_color:
            result_step.append(np.zeros((1, MapHeight, MapLength)))
        else:
            result_step.append(np.ones((1, MapHeight, MapLength)))
        result_step.append(spatial_state_temp)
        spatial_state_temp = np.concatenate(result_step)
        spatial_states_np.append(spatial_state_temp)
        global_states_np.append(global_state_temp)
        acts_np.append(acts_temp)

    spatial_states_np = np.asarray(spatial_states_np)
    global_states_np = np.asarray(global_states_np)
    acts_np = np.asarray(acts_np)
    # plot_3d(spatial_states_np[0,:,:,:])
    return flag, spatial_states_np, global_states_np, acts_np


def get_para(csv_roomrecord, csv_object, LOS, map_height, map_urban, map_road, map_normal):
    csv_roomrecord_groupby = csv_roomrecord.groupby(["Filename"])
    para = [(Filename, one_game_group, csv_object, LOS, map_height, map_urban, map_road, map_normal) for (Filename), one_game_group in csv_roomrecord_groupby]
    print(len(para))
    return para


def multi_para(z):
	return parse_one_game(z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7])


def parse_one_game(Filename, one_game_group, csv_object, LOS, *map_args):
    if one_game_group.iloc[0]["JmResult"] >= 0:
        game_result = 'redwin'
    else:
        game_result = 'bluewin'
    save_file_name = Filename + '_' + game_result
    print(save_file_name)
    game_state = GameState(Filename, csv_object)
    bops, score, acts = game_state.board, game_state.score, game_state.acts
    one_game_group = one_game_group.sort_values(by=["DateAndTime"], axis=0, ascending=[True])
    stage_groupby = one_game_group.groupby(["StageID"])
    # last_bops = copy.deepcopy(bops)
    states_tensor_red = []
    states_vector_red = []
    states_act_red = []
    states_tensor_blue = []
    states_vector_blue = []
    states_act_blue = []

    for (stageid), group in stage_groupby:
        group1 = group.sort_values(by=["TimeID", "ObjStep"], axis=0, ascending=[False, False])

        game_state.initCancelKeep()
        bops["stage"] = stageid

        game_state.score["red_win"] = group.iloc[0]["JmResult"]

        board_one_stage = []
        score_one_stage = []
        acts_one_stage = []
        for indexs, row in group1.iterrows():
            board_one_stage.append(copy.deepcopy(game_state.board))
            score_one_stage.append(copy.deepcopy(game_state.score))
            row_to_state(game_state, row, stageid, one_game_group)
            acts_one_stage.append(copy.deepcopy(game_state.acts))
            game_state.fill_no_act()
        board_one_stage.append(copy.deepcopy(game_state.board))
        score_one_stage.append(copy.deepcopy(game_state.score))
        acts_one_stage.append(copy.deepcopy(game_state.acts))
        if stageid % 2 == 1:  # 奇数阶段,采集红方预测蓝方棋子信息
            bops_odd = zip(board_one_stage, score_one_stage, acts_one_stage)
            flag, tensor_one_stage_red, vector_one_stage_red, acts_one_stage_red = StateFeature_vec_ten(bops_odd, LOS,
                                                                                                        1, *map_args)
            if flag:
                states_tensor_red.append(tensor_one_stage_red)
                states_vector_red.append(vector_one_stage_red)
                states_act_red.append(acts_one_stage_red)
        else:  # 偶数阶段,采集蓝方预测红方棋子信息
            bops_even = zip(board_one_stage, score_one_stage, acts_one_stage)
            flag, tensor_one_stage_blue, vector_one_stage_blue, acts_one_stage_blue = StateFeature_vec_ten(bops_even,
                                                                                                           LOS, 0, *map_args)
            if flag:
                states_tensor_blue.append(tensor_one_stage_blue)
                states_vector_blue.append(vector_one_stage_blue)
                states_act_blue.append(acts_one_stage_blue)

    spatial_states_np_red = np.concatenate(states_tensor_red)
    spatial_states_np_blue = np.concatenate(states_tensor_blue)
    spatial_states_np_red = spatial_states_np_red.reshape([spatial_states_np_red.shape[0], -1])
    spatial_states_np_blue = spatial_states_np_blue.reshape([spatial_states_np_blue.shape[0], -1])

    global_states_np_red = np.concatenate(states_vector_red)
    global_states_np_blue = np.concatenate(states_vector_blue)

    acts_np_red = np.concatenate(states_act_red)
    acts_np_blue = np.concatenate(states_act_blue)
    print('spatial_states_np_red', spatial_states_np_red.shape, "--------")
    print('spatial_states_np_blue', spatial_states_np_blue.shape)
    print('global_states_np_red', global_states_np_red.shape)
    print('global_states_np_blue', global_states_np_blue.shape)
    print('acts_np_red', acts_np_red.shape)
    print('acts_np_blue', acts_np_blue.shape)

    vs1 = 'red2blue'
    vs2 = 'blue2red'

    path = os.path.join('../data/feature_data/feature_tensor_vector/')
    if not os.path.isdir(path):
        os.makedirs(path)

    sparse.save_npz(os.path.join(path,
                                 save_file_name + '_' + vs1 + '@G'), sparse.csr_matrix(global_states_np_red))
    sparse.save_npz(os.path.join(path,
                                 save_file_name + '_' + vs1 + '@S'), sparse.csr_matrix(spatial_states_np_red))
    sparse.save_npz(os.path.join(path,
                                 save_file_name + '_' + vs1 + '@A'), sparse.csr_matrix(acts_np_red))

    sparse.save_npz(os.path.join(path,
                                 save_file_name + '_' + vs2 + '@G'), sparse.csr_matrix(global_states_np_blue))
    sparse.save_npz(os.path.join(path,
                                 save_file_name + '_' + vs2 + '@S'), sparse.csr_matrix(spatial_states_np_blue))
    sparse.save_npz(os.path.join(path,
                                 save_file_name + '_' + vs2 + '@A'), sparse.csr_matrix(acts_np_blue))


def parse_feature(csv_roomrecord, csv_object, LOS, *args):
    para = get_para(csv_roomrecord, csv_object, LOS, *args)

    pool = Pool(20)
    pool.map(multi_para, para)
    pool.close()
    pool.join()


def row_to_state(game_state, row, stageid, one_game_group):
    bop_name = game_state.Get_bop_name(row.ObjID)
    if row.ObjNewPos == row.ObjPos:
        if row.ObjInto != 0:
            game_state.UpdateOn(row, bop_name)

        if row.ObjOut != 0:
            game_state.UpdateOut(row, bop_name)

        if row.AttackID != 0:
            game_state.Attack(row, bop_name, stageid, one_game_group)

        if row.CityTake != 0:
            game_state.Occupy(row, bop_name)

        if row.ObjHide != 0:
            game_state.to_Hide(row, bop_name)

        if row.ObjPass == 1 and row.objname != '步兵小队':
            game_state.to_Pass(row, bop_name)

        if row.ObjPass == 0 and row.ObjHide == 0 and row.ObjKeepCancel == 0 and row.ObjRound > 1:
            game_state.to_move(row, bop_name)

        if row.ObjKeepCancel != 0:
            game_state.KeepCancel(row, bop_name)

    elif row.ObjNewPos != row.ObjPos:
        game_state.Move(row, bop_name)

    else:
        raise Exception


if __name__ == '__main__':
    csv_roomrecord, csv_object, LOS, game_map = load_files()

    map_height, map_urban, map_road, map_normal = extract_map_feature(game_map)

    # spatial feature:play's colour 1, friendly 6, enemy 6, observe 2, map 4, end_pos 6, all is 25*66*51
    # global feature:stage 20, city 4, friendly 23*6, enemy 23*6,  win 1, all is 301
    # action 14*6, no_act 1  all is 85
    parse_feature(csv_roomrecord, csv_object, LOS, map_height, map_urban, map_road, map_normal)
    print('success')


