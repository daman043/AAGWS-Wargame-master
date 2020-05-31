
import numpy as np
from util.util import cvtInt6loc2HexOffset, cvtHexOffset2Int6loc, to_categorical, from_categorical
import copy, math


bops_dic = ["red_tank1", "red_tank2", "red_car1", "red_car2", "red_soldier1", "red_soldier2",
             "blue_tank1", "blue_tank2", "blue_car1", "blue_car2", "blue_soldier1", "blue_soldier2"]
score_dic = ["red_shoot", "red_lost", "red_occupy", "red_total",
             "blue_shoot", "blue_lost", "blue_occupy", "blue_total",
             "red_win"]
act_dic = ["north_east",
           "east",
           "south_east",
            "south_west",
            "west",
            "north_west",
            "occupy",
            "on_car",
            "off_car",
            "to_pass",
            "to_move",
            "to_hide",
            "cancel_keep",
            "shoot"]

class GameState(object):
    def __init__(self, Filename, csv_object):
        self.csv_object = csv_object

        self.board = {"stage": 1,    # There are total 20 stages in a game, so the stage is 1 to 20
                      "city": [0, 0, 0, 0]}
        for bop_name in bops_dic:
            self.board[bop_name] = {"ID": None, "state": None, "Pos": None}

        self.score = {score_name: 0 for score_name in score_dic}
        self.act = {act_name: 0 for act_name in act_dic}

        self.acts = {bop_name: copy.deepcopy(self.act) for bop_name in bops_dic}
        self.acts['no_act'] = 1
        csv_object_red = csv_object.loc[
            (csv_object["Filename"] == Filename) & (csv_object["GameColor"] == "RED"), ['ID', 'ObjName', "ObjPos"]]
        csv_object_red = csv_object_red.sort_values(by="ObjPos")
        if csv_object_red.empty or len(csv_object_red) != 6:
            raise Exception('program error')

        for indexs, row in csv_object_red.iterrows():
            if row.ObjName == "坦克" or row.ObjName == "重型坦克":
                if not self.board["red_tank1"]["ID"]:
                    self.board["red_tank1"]["ID"] = row.ID
                    self.board["red_tank1"]["state"], self.board["red_tank1"]["Pos"] = self.get_bop_init(row.ObjPos)

                else:
                    self.board["red_tank2"]["ID"] = row.ID
                    self.board["red_tank2"]["state"], self.board["red_tank2"]["Pos"] = self.get_bop_init(row.ObjPos)
            if row.ObjName == "战车" or row.ObjName == "中型战车" or row.ObjName == "重型战车":
                if not self.board["red_car1"]["ID"]:
                    self.board["red_car1"]["ID"] = row.ID
                    self.board["red_car1"]["state"], self.board["red_car1"]["Pos"] = self.get_bop_init(row.ObjPos)
                else:
                    self.board["red_car2"]["ID"] = row.ID
                    self.board["red_car2"]["state"], self.board["red_car2"]["Pos"] = self.get_bop_init(row.ObjPos)
            if row.ObjName == "步兵小队":
                if not self.board["red_soldier1"]["ID"]:
                    self.board["red_soldier1"]["ID"] = row.ID
                    self.board["red_soldier1"]["state"], self.board["red_soldier1"]["Pos"] = self.get_bop_init(row.ObjPos)
                else:
                    self.board["red_soldier2"]["ID"] = row.ID
                    self.board["red_soldier2"]["state"], self.board["red_soldier2"]["Pos"] = self.get_bop_init(row.ObjPos)

        csv_object_blue = csv_object.loc[
            (csv_object["Filename"] == Filename) & (csv_object["GameColor"] == "BLUE"), ['ID', 'ObjName', "ObjPos"]]
        csv_object_blue = csv_object_blue.sort_values(by="ObjPos")
        for indexs, row in csv_object_blue.iterrows():
            if row.ObjName == "坦克" or row.ObjName == "重型坦克":
                if not self.board["blue_tank1"]["ID"]:
                    self.board["blue_tank1"]["ID"] = row.ID
                    self.board["blue_tank1"]["state"], self.board["blue_tank1"]["Pos"] = self.get_bop_init(row.ObjPos)
                else:
                    self.board["blue_tank2"]["ID"] = row.ID
                    self.board["blue_tank2"]["state"], self.board["blue_tank2"]["Pos"] = self.get_bop_init(row.ObjPos)
            if row.ObjName == "战车" or row.ObjName == "中型战车" or row.ObjName == "重型战车":
                if not self.board["blue_car1"]["ID"]:
                    self.board["blue_car1"]["ID"] = row.ID
                    self.board["blue_car1"]["state"], self.board["blue_car1"]["Pos"] = self.get_bop_init(row.ObjPos)
                else:
                    self.board["blue_car2"]["ID"] = row.ID
                    self.board["blue_car2"]["state"], self.board["blue_car2"]["Pos"] = self.get_bop_init(row.ObjPos)
            if row.ObjName == "步兵小队":
                if not self.board["blue_soldier1"]["ID"]:
                    self.board["blue_soldier1"]["ID"] = row.ID
                    self.board["blue_soldier1"]["state"], self.board["blue_soldier1"]["Pos"] = self.get_bop_init(row.ObjPos)
                else:
                    self.board["blue_soldier2"]["ID"] = row.ID
                    self.board["blue_soldier2"]["state"], self.board["blue_soldier2"]["Pos"] = self.get_bop_init(row.ObjPos)

    def get_bop_init(self, Pos):
        bop_state = {}
        bop_state["living"] = 1
        bop_state["ObjPass"] = 0
        bop_state["ObjHide"] = 0
        bop_state["ObjKeep"] = 0
        bop_state["ObjTire"] = to_categorical(0, 3)
        bop_state["ObjRound"] = 0
        bop_state["ObjAttack"] = 0
        bop_state["ObjSon"] = 0
        bop_state["ObjInto"] = 0

        bop_state["ObjBlood"] = to_categorical(3, 4)
        bop_state["ObjStep"] = to_categorical(6, 8)
        bop_Pos = cvtInt6loc2HexOffset(Pos)
        return bop_state, bop_Pos

    def load_bop_state(self, bop_name, row):
        self.board[bop_name]["state"]["living"] = 1
        self.board[bop_name]["state"]["ObjPass"] = 0 if row.ObjPass == 0 else 1
        self.board[bop_name]["state"]["ObjHide"] = 0 if row.ObjHide == 0 else 1
        self.board[bop_name]["state"]["ObjKeep"] = 0 if row.ObjKeep == 0 else 1
        if row.ObjTire <= 2:
            Tire = row.ObjTire
        else:
            Tire = 2
        self.board[bop_name]["state"]["ObjTire"] = to_categorical(Tire, 3)
        if row.ObjRound <= 1:
            Round = row.ObjRound
        else:
            Round = 1
        self.board[bop_name]["state"]["ObjRound"] = Round
        if row.ObjAttack <= 1:
            Attack = row.ObjAttack
        else:
            Attack = 1
        self.board[bop_name]["state"]["ObjAttack"] = Attack
        self.board[bop_name]["state"]["ObjSon"] = 0  if row.ObjSon == 0 else 1
        self.board[bop_name]["state"]["ObjInto"] = 0 if row.ObjInto == 0 else 1
        self.board[bop_name]["state"]["ObjBlood"] = to_categorical(row.ObjBlood, 4)
        self.board[bop_name]["state"]["ObjStep"] = to_categorical(row.ObjStep, 8)
        self.board[bop_name]["Pos"] = cvtInt6loc2HexOffset(row.ObjNewPos)

    def get_bop_death(self):
        bop_state = {}
        bop_state["living"] = 0
        bop_state["ObjPass"] = 0
        bop_state["ObjHide"] = 0
        bop_state["ObjKeep"] = 0
        bop_state["ObjTire"] = np.zeros(3)
        bop_state["ObjRound"] = 0
        bop_state["ObjAttack"] = 0
        bop_state["ObjSon"] = 0
        bop_state["ObjInto"] = 0
        bop_state["ObjBlood"] = np.zeros(4)
        bop_state["ObjStep"] = np.zeros(8)
        bop_Pos = (0, 0)
        return bop_state, bop_Pos

    def init_acts(self):
        for key1 in self.acts.keys():
            if key1 != 'no_act':
                for key2 in act_dic:
                    self.acts[key1][key2] = 0
            else:
                self.acts['no_act'] = 0

    def fill_no_act(self):
        for key1 in self.acts.keys():
            if key1 != 'no_act':
                for key2 in act_dic:
                    self.acts[key1][key2] = 0
            else:
                self.acts["no_act"] = 1

    def initCancelKeep(self):
        if self.board["stage"] % 2 == 0:
            self.board["red_tank1"]["state"]["ObjKeep"] = 0
            self.board["red_tank2"]["state"]["ObjKeep"] = 0
            self.board["red_car1"]["state"]["ObjKeep"] = 0
            self.board["red_car2"]["state"]["ObjKeep"] = 0
            self.board["red_soldier1"]["state"]["ObjKeep"] = 0
            self.board["red_soldier2"]["state"]["ObjKeep"] = 0
        else:
            self.board["blue_tank1"]["state"]["ObjKeep"] = 0
            self.board["blue_tank2"]["state"]["ObjKeep"] = 0
            self.board["blue_car1"]["state"]["ObjKeep"] = 0
            self.board["blue_car2"]["state"]["ObjKeep"] = 0
            self.board["blue_soldier1"]["state"]["ObjKeep"] = 0
            self.board["blue_soldier2"]["state"]["ObjKeep"] = 0

    def UpdateOn(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        on_bop_name = self.Get_bop_name(row.ObjInto)
        self.board[on_bop_name]["state"]["ObjInto"] = 1
        # update the action
        self.init_acts()
        self.acts[bop_name]["on_car"] = 1


    def UpdateOut(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        out_bop_name = self.Get_bop_name(row.ObjOut)
        self.board[out_bop_name]["state"]["ObjSon"] = 0
        # update the action
        self.init_acts()
        self.acts[bop_name]["off_car"] = 1

    def Attack(self, row, bop_name, stageid, group):
        if not row.AttackID:
            raise Exception('This row is not attacking happened')
        # update the attribution of the bops
        self.load_bop_state(bop_name, row)
        target_bop_name = self.Get_bop_name(row.TarID)
        TarBop_select = group.loc[(group["StageID"] <= stageid) & (group["ObjID"] == row.TarID)]
        if not TarBop_select.empty:
            SoldierID = int(TarBop_select.sort_values(by="DateAndTime", axis=0, ascending=True).iloc[-1]['ObjSon'])
        else:
            raise Exception

        if row.TarBlood == 0:
            self.board[target_bop_name]["state"], self.board[target_bop_name]["Pos"] = self.get_bop_death()
            if SoldierID:
                soldier_name = self.Get_bop_name(SoldierID)
                self.board[soldier_name]["state"], self.board[soldier_name]["Pos"] = self.get_bop_death()
        else:
            self.board[target_bop_name]["state"]["ObjBlood"] = to_categorical(row.TarBlood, 4)
            if row.TarKeep != 0:
                self.board[target_bop_name]["state"]["ObjKeep"] = 1
            if SoldierID:
                soldier_name = self.Get_bop_name(SoldierID)
                soldier_blood = from_categorical(self.board[soldier_name]["state"]["ObjBlood"]) - row.TarLost
                if soldier_blood == 0:
                    self.board[soldier_name]["state"], self.board[soldier_name]["Pos"] = self.get_bop_death()
                else:
                    self.board[soldier_name]["state"]["ObjBlood"] = to_categorical(soldier_blood, 4)

        # update the score
        self.get_score(row, SoldierID)

        # update the action
        self.init_acts()
        self.acts[bop_name]["shoot"] = 1


    def Occupy(self, row, bop_name):
        if not row.CityTake:
            raise Exception('This row is not occupying happened')
        # update the total attribute and the score
        if row.ObjPos == 80048:
            if row.StageID % 2 == 1:
                if self.board["city"][0:2] == [0, 1]:
                    self.score["red_occupy"] += 80
                    self.score["red_total"] += 80
                    self.score["blue_occupy"] -= 80
                    self.score["blue_total"] -= 80
                else:
                    self.score["red_occupy"] += 80
                    self.score["red_total"] += 80
                self.board["city"][0:2] = [1, 0]

            else:
                if self.board["city"][0:2] == [1, 0]:
                    self.score["red_occupy"] -= 80
                    self.score["red_total"] -= 80
                    self.score["blue_occupy"] += 80
                    self.score["blue_total"] += 80
                else:
                    self.score["blue_occupy"] += 80
                    self.score["blue_total"] += 80
                self.board["city"][0:2] = [0, 1]

        elif row.ObjPos == 100049:
            if row.StageID % 2 == 1:
                if self.board["city"][2:4] == [0, 1]:
                    self.score["red_occupy"] += 50
                    self.score["red_total"] += 50
                    self.score["blue_occupy"] -= 50
                    self.score["blue_total"] -= 50
                else:
                    self.score["red_occupy"] += 50
                    self.score["red_total"] += 50
                self.board["city"][2:4] = [1, 0]

            else:
                if self.board["city"][2:4] == [1, 0]:
                    self.score["red_occupy"] -= 50
                    self.score["red_total"] -= 50
                    self.score["blue_occupy"] += 50
                    self.score["blue_total"] += 50
                else:
                    self.score["blue_occupy"] += 50
                    self.score["blue_total"] += 50
                self.board["city"][2:4] = [0, 1]
        else:
            raise Exception('program error')

        # update the action
        self.init_acts()
        self.acts[bop_name]["occupy"] = 1

    def to_Hide(self, row, bop_name):
        if not row.ObjHide:
            raise Exception('This row is not hiding happened')
        self.load_bop_state(bop_name, row)
        # update the action
        self.init_acts()
        self.acts[bop_name]["to_hide"] = 1

    def to_Pass(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        # update the action
        self.init_acts()
        self.acts[bop_name]["to_pass"] = 1

    def to_move(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        # update the action
        self.init_acts()
        self.acts[bop_name]["to_move"] = 1

    def KeepCancel(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        if not row.ObjBlood:
            self.board[bop_name]["state"], self.board[bop_name]["Pos"] = self.get_bop_death()
        # update the action
        self.init_acts()
        self.acts[bop_name]["cancel_keep"] = 1

    def Move(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        if row.ObjSon:
            ObjSon_name = self.Get_bop_name(row.ObjSon)
            self.board[ObjSon_name]["Pos"] = cvtInt6loc2HexOffset(row.ObjNewPos)

        # update the action
        old_pos = cvtInt6loc2HexOffset(row.ObjPos)
        new_pos = cvtInt6loc2HexOffset(row.ObjNewPos)
        direction = self.which_side(old_pos, new_pos)

        self.init_acts()
        self.acts[bop_name][direction] = 1

    def Get_bop_name(self, ObjID):
        bop_name = None
        for key in bops_dic:
            if self.board[key]["ID"] == ObjID:
                bop_name = key
        if not bop_name:
            raise Exception('program error')

        return bop_name

    def get_score(self, row, SoldierID):
        TarID = row.TarID
        csv_object_ObjValue = self.csv_object.loc[(self.csv_object["Filename"] == row.Filename) & (self.csv_object["ID"] == TarID)]
        if not csv_object_ObjValue.empty:
            ObjValue = int(csv_object_ObjValue.ObjValue)
        else:
            raise Exception
        if SoldierID != 0:
            SoldierValue = int(self.csv_object.loc[
                (self.csv_object["Filename"] == row.Filename) & (self.csv_object["ID"] == SoldierID)].ObjValue)
        else:
            SoldierValue = 0
        kill_num = row.TarLost
        result = kill_num * (ObjValue + SoldierValue)

        Tar_name = self.Get_bop_name(TarID)
        Tar_color =Tar_name.split('_')[0]
        if Tar_color == 'red':
            self.score["red_lost"] += result
            self.score["blue_shoot"] += result
            self.score["blue_total"] += result
        else:
            self.score["red_shoot"] += result
            self.score["red_total"] += result
            self.score["blue_lost"] += result

    def which_side(self, old_hex, to_hex):
        """Tuple of 1, 2, or 6(same hex) hexsides passed through on way to_hex.
        Accurate for distances up to about 10000 hexes.
        :return: Tuple of integer hexsides.
        """

        if old_hex == to_hex:
            raise Exception
        d_x = old_hex[0] - to_hex[0]
        d_y = old_hex[1] - to_hex[1]
        # Same hex column.
        if d_x == 0:
            if d_y < 0:
                return 'east'
            else:
                return 'west'
        # Even / odd columns need a nudge.
        if (old_hex[0] % 2) and not (to_hex[0] % 2):
            d_y += .5
        if not (old_hex[0] % 2) and (to_hex[0] % 2):
            d_y -= .5
        # Hexes are wider than tall.
        d_y *= -20.0  # Negative cause math origin is lower left not upper left.
        d_x *= 17.32
        angle = round(math.degrees(math.atan2(d_y, d_x)), 2)
        if angle > 0 and angle < 60:
            answer = 'north_east'
        elif angle > 60 and angle < 120:
            answer = 'east'
        elif angle > 120 and angle < 180:
            answer = 'south_east'
        elif angle < 0 and angle > -60:
            answer = 'north_west'
        elif angle < -60 and angle > -120:
            answer = 'west'
        elif angle < -120 and angle > -180:
            answer = 'south_west'
        elif angle == 60:
            answer = 'east'
        elif angle == 0:
            answer = 'north_east'
        elif angle == -60:
            answer = 'north_west'
        elif angle == -120:
            answer = 'west'
        elif abs(angle) == 180:
            answer = 'south_east'
        elif angle == 120:
            answer = 'west'
        else:

            raise Exception
        return answer
