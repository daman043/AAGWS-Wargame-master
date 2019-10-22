import pandas as pd
import os

# %读入8张原始数据表格,并且csv_roomrecord做了重复性筛选
file_head = '../data/replay_data/urban_terrain/'
csv_roomrecord = pd.read_csv(os.path.join(file_head + "actionrecord.csv"))
csv_roomrecord.drop_duplicates(keep='first', inplace=True)
csv_caijue = pd.read_csv(os.path.join(file_head + "caijue.csv"))
csv_caijue.drop_duplicates(keep='first', inplace=True)
csv_object = pd.read_csv(os.path.join(file_head + "object.csv"))
csv_object.drop_duplicates(keep='first', inplace=True)
csv_object1 = pd.read_csv(os.path.join(file_head + "object1.csv"))
csv_object1.drop_duplicates(keep='first', inplace=True)
csv_object2 = pd.read_csv(os.path.join(file_head + "object2.csv"))
csv_object2.drop_duplicates(keep='first', inplace=True)
csv_city1 = pd.read_csv(os.path.join(file_head + "city1.csv"))
csv_city1.drop_duplicates(keep='first', inplace=True)

def get_score(warid):
    Red_ObjBlood = csv_object1.loc[(csv_object1['WARID']==warid) & (csv_object1["GameColor"]=="RED")]["ObjBlood"]
    Red_ObjBlood2 = csv_object1.loc[(csv_object1['WARID']==warid) & (csv_object1["GameColor"]=="RED")]["ObjBlood2"]
    Red_ObjValue = csv_object1.loc[(csv_object1['WARID']==warid) & (csv_object1["GameColor"]=="RED")]["ObjValue"]
    Red_ObjBlood_incar = csv_object2.loc[(csv_object1['WARID'] == warid) & (csv_object1["GameColor"] == "RED")]["ObjBlood"]
    Red_ObjBlood2_incar = csv_object2.loc[(csv_object1['WARID'] == warid) & (csv_object1["GameColor"] == "RED")]["ObjBlood2"]
    Red_ObjValue_incar = csv_object2.loc[(csv_object1['WARID'] == warid) & (csv_object1["GameColor"] == "RED")]["ObjValue"]

    Red_Last_Obj_score = Red_ObjBlood.mul(Red_ObjValue).sum() + Red_ObjBlood_incar.mul(Red_ObjValue_incar).sum()
    Red_Total_Obj_score = Red_ObjBlood2.mul(Red_ObjValue).sum() + Red_ObjBlood2_incar.mul(Red_ObjValue_incar).sum()
    Red_Lost_Obj_score = Red_Total_Obj_score - Red_Last_Obj_score

    Blue_ObjBlood = csv_object1.loc[(csv_object1['WARID']==warid) & (csv_object1["GameColor"]=="BLUE")]["ObjBlood"] #
    Blue_ObjBlood2 = csv_object1.loc[(csv_object1['WARID']==warid) & (csv_object1["GameColor"]=="BLUE")]["ObjBlood2"]
    Blue_ObjValue = csv_object1.loc[(csv_object1['WARID']==warid) & (csv_object1["GameColor"]=="BLUE")]["ObjValue"]
    Blue_ObjBlood_incar = csv_object2.loc[(csv_object1['WARID'] == warid) & (csv_object1["GameColor"] == "BLUE")][
        "ObjBlood"]
    Blue_ObjBlood2_incar = csv_object2.loc[(csv_object1['WARID'] == warid) & (csv_object1["GameColor"] == "BLUE")][
        "ObjBlood2"]
    Blue_ObjValue_incar = csv_object2.loc[(csv_object1['WARID'] == warid) & (csv_object1["GameColor"] == "BLUE")]["ObjValue"]

    Blue_Last_Obj_score = Blue_ObjBlood.mul(Blue_ObjValue).sum() + Blue_ObjBlood_incar.mul(Blue_ObjValue_incar).sum()
    Blue_Total_Obj_score = Blue_ObjBlood2.mul(Blue_ObjValue).sum() + Blue_ObjBlood2_incar.mul(Blue_ObjValue_incar).sum()
    Blue_Lost_Obj_score = Blue_Total_Obj_score - Blue_Last_Obj_score

    city = csv_city1.loc[(csv_city1['WARID']==warid)] #
    red_city = 0
    blue_city = 0
    for indexs, row in city.iterrows():
        if row.CityIco == 'RED':
            red_city += row.C1
        if row.CityIco == 'BLUE':
            blue_city += row.C1
    red_win_all = Red_Last_Obj_score + Blue_Lost_Obj_score + red_city
    blue_win_all = Blue_Last_Obj_score + Red_Lost_Obj_score + blue_city
    Red_Lost_score = Red_Lost_Obj_score + blue_city
    Blue_Lost_score = Blue_Lost_Obj_score + red_city
    if (red_win_all - blue_win_all)>0:
        win_red = 1
    elif red_win_all == blue_win_all:
        win_red = 0
    else:
        win_red = -1
    return Red_Lost_score, Blue_Lost_score, win_red


if __name__ == '__main__':

    # 对比赛数据进行过滤筛选,有3条原则:1.少于18个阶段的;2.第1,2阶段存在未走子的;3.双方损伤值都较小.
    csv_roomrecord_groupby = csv_roomrecord.groupby(["warid"])
    total_war_num = 0
    last_war_num = 0
    principle_1 = 0
    principle_2 = 0
    principle_3 = 0

    for (warid), group in csv_roomrecord_groupby:
        total_war_num += 1
        # group = group.sort_values(by="StageID", axis=0, ascending=True)
        if group['StageID'].max() < 10:
            csv_roomrecord = csv_roomrecord.drop(csv_roomrecord[csv_roomrecord['warid'] == warid].index)
            principle_1 += 1
        group_select = group[(group['TimeID'] >= 39) & (group['TimeID'] < 80)]
        if group_select.loc[group_select['StageID']==1]["ObjID"].count() < 15:
            csv_roomrecord = csv_roomrecord.drop(csv_roomrecord[csv_roomrecord['warid'] == warid].index)
            principle_2 += 1
        if group_select.loc[group_select['StageID']==2]["ObjID"].count() < 15:
            csv_roomrecord = csv_roomrecord.drop(csv_roomrecord[csv_roomrecord['warid'] == warid].index)
            principle_2 += 1
        Red_Lost_Obj_score, Blue_Lost_Obj_score, win = get_score(warid)
        if Red_Lost_Obj_score < 10 and Blue_Lost_Obj_score < 10:
            csv_roomrecord = csv_roomrecord.drop(csv_roomrecord[csv_roomrecord['warid'] == warid].index)
            principle_3 += 1
        csv_roomrecord.loc[csv_roomrecord['warid'] == warid, ["JmResult"]] = win
    last_war_num = total_war_num - principle_1 - principle_2 - principle_3
    print('total_war_num:', total_war_num)
    print('filtering number:', principle_1, principle_2, principle_3)
    print('last_war_num:', last_war_num)
    save_file = '../data/preprocess_data/'
    if not os.path.isdir(save_file):
        os.makedirs(save_file)
    csv_roomrecord.to_csv(os.path.join(save_file + "select_preprocess.csv"), index=False)