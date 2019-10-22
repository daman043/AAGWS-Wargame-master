import numpy as np
from PIL import Image, ImageDraw
import os


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
def plot_position(true_postion, predict_position, result_dir, name):
    map_pic = os.path.join('../../data/map/城镇居民地.jpg')  # 地图背景图片目录
    im = Image.open(map_pic)  # 作为背景
    img = im.convert('RGBA')  # 转换为RGBA模式,添加透明度通道,便于透明图像融合
    im2 = Image.new("RGB", img.size, )  # 建立前景图像
    img2 = im2.convert('RGBA')  # 添加透明度通道
    im3 = Image.new("RGB", img.size, )  # 建立前景图像
    img3 = im3.convert('RGBA')  # 添加透明度通道
    draw2 = ImageDraw.Draw(img2)  # 引入画笔
    draw3 = ImageDraw.Draw(img3)  # 引入画笔

    for (row, col) in true_postion:
        Int6loc = cvtHexOffset2Int6loc(row, col)
        center, x = xiangsu(Int6loc)
        draw2.polygon(x, fill=(255, 0, 0))

    for (row, col) in predict_position:
        Int6loc = cvtHexOffset2Int6loc(row, col)
        center, x = xiangsu(Int6loc)
        draw3.polygon(x, fill=(0, 0, 255))

    blend2 = Image.blend(img, img2, 0.5)  # 背景与所画图像进行融合
    blend3 = Image.blend(blend2, img3, 0.5)  # 背景与所画图像进行融合
    blend3.save(os.path.join(result_dir, name + ".png"))  # 保存图像


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
