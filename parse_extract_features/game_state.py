
import numpy as np
from util.util import cvtInt6loc2HexOffset, to_categorical


class GameState(object):
    def __init__(self, Filename, csv_object):
        self.bops_last_pos = {"red_tank1": None,
                              "red_tank2": None,
                              "red_car1": None,
                              "red_car2": None,
                              "red_soldier1": None,
                              "red_soldier2": None,
                              "blue_tank1": None,
                              "blue_tank2": None,
                              "blue_car1": None,
                              "blue_car2": None,
                              "blue_soldier1": None,
                              "blue_soldier2": None}

        self.bops = {"stage": 1, "city": [0, 0, 0, 0],
                     "red_tank1": {"ID": None, "state": None, "Pos": None},
                     "red_tank2": {"ID": None, "state": None, "Pos": None},
                     "red_car1": {"ID": None, "state": None, "Pos": None},
                     "red_car2": {"ID": None, "state": None, "Pos": None},
                     "red_soldier1": {"ID": None, "state": None, "Pos": None},
                     "red_soldier2": {"ID": None, "state": None, "Pos": None},
                     "blue_tank1": {"ID": None, "state": None, "Pos": None},
                     "blue_tank2": {"ID": None, "state": None, "Pos": None},
                     "blue_car1": {"ID": None, "state": None, "Pos": None},
                     "blue_car2": {"ID": None, "state": None, "Pos": None},
                     "blue_soldier1": {"ID": None, "state": None, "Pos": None},
                     "blue_soldier2": {"ID": None, "state": None, "Pos": None},
                     "red_win": 0}

        csv_object_red = csv_object.loc[
            (csv_object["Filename"] == Filename) & (csv_object["GameColor"] == "RED"), ['ID', 'ObjName', "ObjPos"]]
        csv_object_red = csv_object_red.sort_values(by="ObjPos")
        if csv_object_red.empty or len(csv_object_red) != 6:
            raise Exception('program error')

        for indexs, row in csv_object_red.iterrows():
            if row.ObjName == "坦克" or row.ObjName == "重型坦克":
                if not self.bops["red_tank1"]["ID"]:
                    self.bops["red_tank1"]["ID"] = row.ID
                    self.bops["red_tank1"]["state"], self.bops["red_tank1"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["red_tank1"] = cvtInt6loc2HexOffset(row.ObjPos)

                else:
                    self.bops["red_tank2"]["ID"] = row.ID
                    self.bops["red_tank2"]["state"], self.bops["red_tank2"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["red_tank2"] = cvtInt6loc2HexOffset(row.ObjPos)
            if row.ObjName == "战车" or row.ObjName == "中型战车" or row.ObjName == "重型战车":
                if not self.bops["red_car1"]["ID"]:
                    self.bops["red_car1"]["ID"] = row.ID
                    self.bops["red_car1"]["state"], self.bops["red_car1"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["red_car1"] = cvtInt6loc2HexOffset(row.ObjPos)
                else:
                    self.bops["red_car2"]["ID"] = row.ID
                    self.bops["red_car2"]["state"], self.bops["red_car2"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["red_car2"] = cvtInt6loc2HexOffset(row.ObjPos)
            if row.ObjName == "步兵小队":
                if not self.bops["red_soldier1"]["ID"]:
                    self.bops["red_soldier1"]["ID"] = row.ID
                    self.bops["red_soldier1"]["state"], self.bops["red_soldier1"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["red_soldier1"] = cvtInt6loc2HexOffset(row.ObjPos)
                else:
                    self.bops["red_soldier2"]["ID"] = row.ID
                    self.bops["red_soldier2"]["state"], self.bops["red_soldier2"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["red_soldier2"] = cvtInt6loc2HexOffset(row.ObjPos)

        csv_object_blue = csv_object.loc[
            (csv_object["Filename"] == Filename) & (csv_object["GameColor"] == "BLUE"), ['ID', 'ObjName', "ObjPos"]]
        csv_object_blue = csv_object_blue.sort_values(by="ObjPos")
        for indexs, row in csv_object_blue.iterrows():
            if row.ObjName == "坦克" or row.ObjName == "重型坦克":
                if not self.bops["blue_tank1"]["ID"]:
                    self.bops["blue_tank1"]["ID"] = row.ID
                    self.bops["blue_tank1"]["state"], self.bops["blue_tank1"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["blue_tank1"] = cvtInt6loc2HexOffset(row.ObjPos)
                else:
                    self.bops["blue_tank2"]["ID"] = row.ID
                    self.bops["blue_tank2"]["state"], self.bops["blue_tank2"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["blue_tank2"] = cvtInt6loc2HexOffset(row.ObjPos)
            if row.ObjName == "战车" or row.ObjName == "中型战车" or row.ObjName == "重型战车":
                if not self.bops["blue_car1"]["ID"]:
                    self.bops["blue_car1"]["ID"] = row.ID
                    self.bops["blue_car1"]["state"], self.bops["blue_car1"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["blue_car1"] = cvtInt6loc2HexOffset(row.ObjPos)
                else:
                    self.bops["blue_car2"]["ID"] = row.ID
                    self.bops["blue_car2"]["state"], self.bops["blue_car2"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["blue_car2"] = cvtInt6loc2HexOffset(row.ObjPos)
            if row.ObjName == "步兵小队":
                if not self.bops["blue_soldier1"]["ID"]:
                    self.bops["blue_soldier1"]["ID"] = row.ID
                    self.bops["blue_soldier1"]["state"], self.bops["blue_soldier1"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["blue_soldier1"] = cvtInt6loc2HexOffset(row.ObjPos)
                else:
                    self.bops["blue_soldier2"]["ID"] = row.ID
                    self.bops["blue_soldier2"]["state"], self.bops["blue_soldier2"]["Pos"] = self.get_bop_init(row.ObjPos)
                    self.bops_last_pos["blue_soldier2"] = cvtInt6loc2HexOffset(row.ObjPos)

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
        bop_state["ObjBlood"] = to_categorical(3, 4)
        bop_state["ObjStep"] = to_categorical(0, 8)
        bop_Pos = cvtInt6loc2HexOffset(Pos)
        return bop_state, bop_Pos

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
        bop_state["ObjBlood"] = np.zeros(4)
        bop_state["ObjStep"] = np.zeros(8)
        bop_Pos = (0, 0)
        return bop_state, bop_Pos

    def initCancelKeep(self):
        if self.bops["stage"] % 2 == 0:
            self.bops["red_tank1"]["state"]["ObjKeep"] = 0
            self.bops["red_tank2"]["state"]["ObjKeep"] = 0
            self.bops["red_car1"]["state"]["ObjKeep"] = 0
            self.bops["red_car2"]["state"]["ObjKeep"] = 0
            self.bops["red_soldier1"]["state"]["ObjKeep"] = 0
            self.bops["red_soldier2"]["state"]["ObjKeep"] = 0
        else:
            self.bops["blue_tank1"]["state"]["ObjKeep"] = 0
            self.bops["blue_tank2"]["state"]["ObjKeep"] = 0
            self.bops["blue_car1"]["state"]["ObjKeep"] = 0
            self.bops["blue_car2"]["state"]["ObjKeep"] = 0
            self.bops["blue_soldier1"]["state"]["ObjKeep"] = 0
            self.bops["blue_soldier2"]["state"]["ObjKeep"] = 0

    def UpdateOn(self, row, bop_name):
        self.bops[bop_name]["state"]["ObjSon"] = 1
        on_bop_name = self.Get_bop_name(row.ObjInto)
        self.bops[on_bop_name]["state"]["ObjSon"] = 1

    def UpdateOut(self, row, bop_name):
        self.bops[bop_name]["state"]["ObjSon"] = 0
        out_bop_name = self.Get_bop_name(row.ObjOut)
        self.bops[out_bop_name]["state"]["ObjSon"] = 0

    def Attack(self, row, bop_name):
        self.bops[bop_name]["state"]["ObjAttack"] = 1
        self.bops[bop_name]["state"]["ObjHide"] = 0
        target_bop_name = self.Get_bop_name( row.TarID)
        if row.TarBlood == 0:
            self.bops[target_bop_name]["state"], self.bops[target_bop_name]["Pos"] = self.get_bop_death()
            self.bops_last_pos[bop_name] = (0, 0)
        else:
            self.bops[target_bop_name]["state"]["ObjBlood"] = to_categorical(row.ObjBlood, 4)
            if row.TarKeep != 0:
                self.bops[target_bop_name]["state"]["ObjKeep"] = 1

    def Occupy(self, row):
        if row.ObjPos == 80048:
            if row.StageID % 2 == 1:
                self.bops["city"][0:2] = [1, 0]
            else:
                self.bops["city"][0:2] = [0, 1]
        elif row.ObjPos == 100049:
            if row.StageID % 2 == 1:
                self.bops["city"][2:4] = [1, 0]
            else:
                self.bops["city"][2:4] = [0, 1]
        else:
            raise Exception('program error')

    def Hide(self, row, bop_name):
        self.bops[bop_name]["state"]["ObjHide"] = 1

    def Pass(self, row, bop_name):
        self.bops[bop_name]["state"]["ObjPass"] = 1

    def KeepCancel(self, row, bop_name):
        self.bops[bop_name]["state"]["ObjKeep"] = 0
        self.bops[bop_name]["state"]["ObjBlood"] = to_categorical(row.ObjBlood, 4)
        if not row.ObjBlood:
            self.bops[bop_name]["state"], self.bops[bop_name]["Pos"] = self.get_bop_death()

    def Move(self, row, bop_name):
        self.bops[bop_name]["state"], self.bops[bop_name]["Pos"] = self.get_bop_move(row)
        if row.ObjSon:
            ObjSon_name = self.Get_bop_name(row.ObjSon)
            self.bops[ObjSon_name]["Pos"] = cvtInt6loc2HexOffset(row.ObjNewPos)
            self.bops_last_pos[ObjSon_name] = cvtInt6loc2HexOffset(row.ObjNewPos)
        self.bops_last_pos[bop_name] = cvtInt6loc2HexOffset(row.ObjNewPos)

    @staticmethod
    def get_bop_move(row):
        bop_state = {}
        bop_state["living"] = 1
        bop_state["ObjPass"] = 1 if row.ObjPass == 1 else 0
        bop_state["ObjHide"] = 0 if row.ObjHide == 0 else 1
        bop_state["ObjKeep"] = 0 if row.ObjKeep == 0 else 1
        if row.ObjTire >= 3:
            row.ObjTire = 2
        bop_state["ObjTire"] = to_categorical(row.ObjTire, 3)
        bop_state["ObjRound"] = 0 if row.ObjRound == 0 else 1
        bop_state["ObjAttack"] = 0 if row.ObjAttack == 0 else 1
        bop_state["ObjSon"] = 0 if row.ObjSon == 0 else 1
        bop_state["ObjBlood"] = to_categorical(row.ObjBlood, 4)
        bop_state["ObjStep"] = to_categorical(row.ObjStep, 8)
        bop_Pos = cvtInt6loc2HexOffset(row.ObjNewPos)

        return bop_state, bop_Pos

    def Get_bop_name(self, ObjID):
        bop_name = None
        for key in self.bops:
            if key != "stage" and key != "city" and key != "red_win":
                if self.bops[key]["ID"] == ObjID:
                    bop_name = key
        if not bop_name:
            raise Exception('program error')
        else:
            return bop_name
