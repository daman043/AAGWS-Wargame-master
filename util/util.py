import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import seaborn as sns

def cvtInt6loc2HexOffset(int6loc):
    '''转换6位整形坐标int6loc转换为偏移坐标（y,x）2元组'''
    try:
        str6loc = str(int6loc)
        len_oristr6loc = len(str6loc)
        assert (len_oristr6loc <= 6)
        if len_oristr6loc < 6:
            str6loc = '0' * (6 - len_oristr6loc) + str6loc

        int_first_2, int_last_3 = int(str6loc[0:2]), int(str6loc[3:])
        if int_last_3 % 2 == 1:
            row , col = int_first_2 * 2 + 1 , (int_last_3 - 1) // 2
        else:
            row , col = int_first_2 * 2 , int_last_3 // 2
        return (row,col)
    except Exception as e:
        echosentence_color('comnon > cvtInt6loc2HexOffset():{}'.format(str(e)))
        raise


def cvtHexOffset2Int6loc(row, col):
    '''转换（row,col）到6位整型坐标'''
    try:
        if row % 2 == 1:
            tmpfirst = (row - 1) // 2
            tmplast = col * 2 + 1
        else:
            tmpfirst = row // 2
            tmplast = col * 2
        assert (tmpfirst >= 0 and tmplast >= 0)
        return int(tmpfirst * 10000 + tmplast)
    except Exception as e:
        echosentence_color('common > cvtHexOffset2Int6():{}'.format(str(e)))
        raise


def cvtFlatten2Offset(flatten_num, ROW, COL):
    '''转换（row,col）到6位整型坐标'''
    try:
        if isinstance(flatten_num,list):
            flatten_num = int(flatten_num)
        col = flatten_num % COL
        row = flatten_num // COL

        assert (flatten_num >= 0 and ROW >= row >= 0 and COL >= col >= 0)
        return (row,col)
    except Exception as e:
        echosentence_color('common > cvtHexOffset2Int6():{}'.format(str(e)))
        raise

def cvtOffset2Flatten(Offset, ROW, COL):
    '''转换（row,col）到6位整型坐标'''
    try:
        assert len(Offset) == 2
        flatten_num = Offset[0] * COL + Offset[1]

        assert (flatten_num >= 0 and ROW >= Offset[0] >= 0 and COL >= Offset[1] >= 0)
        return flatten_num
    except Exception as e:
        echosentence_color('common > cvtOffset2Flatten6():{}'.format(str(e)))
        raise


def echosentence_color(str_sentence=None, color=None, flag_newline=True):
    try:
        if color is not None:
            list_str_colors = ['darkbrown', 'red', 'green', 'yellow', 'blue', 'purple', 'yank', 'white']
            assert str_sentence is not None and color in list_str_colors
            id_color = 30 + list_str_colors.index(color)
            print('\33[1;35;{}m'.format(id_color) + str_sentence + '\033[0m')
        else:
            if flag_newline:
                print(str_sentence)
            else:
                print(str_sentence, end=" ")
    except Exception as e:
        print('error in echosentence_color {}'.format(str(e)))
        raise


def xiangsu(q):
    """
    # 六角格像素范围确定函数
    :param q: 推演数据中记录的6位整数坐标
    :return: 对应地图六角格六个顶点的六元数组，用于填充对应六角格
    """
    a = q % 10
    b=(q//10)%10
    c=(q//100)%10
    d=(q//1000)%10
    e=(q//10000)%10
    f=(q//100000)%10
    x=a+b*10+c*100
    y=f*10+e
    if (x%2)==0:
        y=2*y*45+13
    else:
        y=(2*y)*45+58
    x=x*26
    #得到x、y，六角格中心坐标
    x1=x-25
    x2=x
    x3=x+25+1
    x4=x+25+1
    x5=x
    x6=x-25
    y1=y-16
    y2=y-28
    y3=y-16
    y4=y+16
    y5=y+28+1
    y6=y+16
    center = (x,y)
    w=[(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6)]
    return center, w


# 画图函数
def plot_position(true_postion, predict_position, map_dir, map_size, result_dir, name, blend_level=0.9, text=None):
    # map_pic = os.path.join('../../data/map/城镇居民地.jpg')  # 地图背景图片目录
    im = Image.open(map_dir)  # 作为背景
    img = im.convert('RGBA')  # 转换为RGBA模式,添加透明度通道,便于透明图像融合
    hex_draw = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(hex_draw)  # 引入画笔
    if len(true_postion) != len(predict_position):
        raise Exception('预测位置与真实位置不对应')
    for i, pos in enumerate(true_postion):
        if pos == predict_position[i]:
            (row, col) = cvtFlatten2Offset(pos, map_size[0], map_size[1])
            Int6loc = cvtHexOffset2Int6loc(row, col)
            center, x = xiangsu(Int6loc)
            draw.polygon(x, fill=(0, 255, 0, int(blend_level * 100)))
        else:
            (row, col) = cvtFlatten2Offset(pos, map_size[0], map_size[1])
            Int6loc = cvtHexOffset2Int6loc(row, col)
            center, x = xiangsu(Int6loc)
            draw.polygon(x, fill=(255, 0, 0, int(blend_level * 100)))

            (row, col) = cvtFlatten2Offset(predict_position[i], map_size[0], map_size[1])
            Int6loc = cvtHexOffset2Int6loc(row, col)
            center, x = xiangsu(Int6loc)
            draw.polygon(x, fill=(0, 0, 255, int(blend_level * 100)))
    img = Image.alpha_composite(img, hex_draw)
    del draw
    img.save(os.path.join(result_dir, name + ".png"))  # 保存图像

def plot_position_Flatten(postion_list, map_dir, map_size, result_dir, name, blend_level=0.9, text=None):
    # map_pic = os.path.join('../../data/map/城镇居民地.jpg')  # 地图背景图片目录
    im = Image.open(map_dir)  # 作为背景
    img = im.convert('RGBA')  # 转换为RGBA模式,添加透明度通道,便于透明图像融合
    hex_draw = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(hex_draw)  # 引入画笔

    for i, pos in enumerate(postion_list):
        (row, col) = cvtFlatten2Offset(pos, map_size[0], map_size[1])
        Int6loc = cvtHexOffset2Int6loc(row, col)
        center, x = xiangsu(Int6loc)
        draw.polygon(x, fill=(255, 0, 0, int(blend_level * 100)))

    img = Image.alpha_composite(img, hex_draw)
    del draw
    img.save(os.path.join(result_dir, name + ".png"))  # 保存图像

def get_name_from_labels(labels):
    text_labels = ['tank_1', 'tank_2', 'armored_vehicles_1', 'armored_vehicles_2', 'soldier_1', 'soldier_2']
    return [text_labels[int(i)] for i in labels]


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def from_categorical(c):
    label = (np.argmax(c, axis=0)).tolist()
    if isinstance(label, int):
        return label
    else:
        raise Exception

def all_true(logic_list):
    if not isinstance(logic_list, list):
        raise Exception
    if not logic_list:
        return False
    for i in logic_list:
        if (i is False) or i == 0 or i == [] or (i is None):
            return False
    return True


def plot_hot_map(data, name:str):
    # fig = plt.figure()
    ax = sns.heatmap(data, annot=False)
    ax.set_title(name)
    plt.show()



def plot_predict_heatmap(predict_data, true_pos, map_size, result_dir, name, blend_level=0):
    tank_dir = '../../plot/Tank2.png'

    map_dir = os.path.join('../../data/map/城镇居民地.jpg')  # 地图背景图片目录
    im = Image.open(map_dir)  # 作为背景
    img = im.convert('RGBA')  # 转换为RGBA模式,添加透明度通道,便于透明图像融合
    hex_draw = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(hex_draw)  # 引入画笔


    data = normalization(predict_data)
    for index in range(len(data)):
        (row, col) = cvtFlatten2Offset(index, map_size[0], map_size[1])
        Int6loc = cvtHexOffset2Int6loc(row, col)
        center, x = xiangsu(Int6loc)
        color = get_heat_color(data[index], blend_level)
        # print(color)
        draw.polygon(x, fill=color)

    img = Image.alpha_composite(img, hex_draw)

    im = Image.open(tank_dir)
    ## rawimg的size和im的size要相同，不然不能匹配
    # paste(用来粘贴的图片，(位置坐标))，可以通过设置位置坐标来确定粘贴图片的位置
    # 该方法没有返回值，直接作用于原图片
    (row, col) = cvtFlatten2Offset(true_pos, map_size[0], map_size[1])
    Int6loc = cvtHexOffset2Int6loc(row, col)
    center, x = xiangsu(Int6loc)
    img.paste(im, (center[0]-23, center[1]-25, center[0] + 48-23, center[1] + 48-25), mask=im)
    # img.show()
    del draw
    img.save(os.path.join(result_dir, name + ".png"))  # 保存图像


def get_heat_color(data, blend_level):
    bg_1 = (255, 165, 0)
    bg_2 = (0, 0, 128)

    # 设置步长
    step_r = (bg_1[0] - bg_2[0])
    step_g = (bg_1[1] - bg_2[1])
    step_b = (bg_1[2] - bg_2[2])

    bg_r = int(round(bg_2[0] + step_r * data))
    bg_g = int(round(bg_2[1] + step_g * data))
    bg_b = int(round(bg_2[2] + step_b * data))
    blend_level = int(blend_level * 255)
    return (bg_r, bg_g, bg_b, blend_level)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range



if __name__ == '__main__':
    plot_hot_map(np.ones((5,5)), str(1))