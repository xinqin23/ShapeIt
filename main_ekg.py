import argparse

from shapeit.shape_it import ShapeIt
from options import Options
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_alphabet():
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Create a Rectangle patch
    rect1 = patches.Rectangle((-1.1, 0.04), 2.5, 0.25, linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((-0.4, 0.32), 0.59, 0.23, linewidth=1, edgecolor='r', facecolor='none')
    rect3 = patches.Rectangle((-69, 0.015), 40, 0.01, linewidth=1, edgecolor='r', facecolor='none')
    rect4 = patches.Rectangle((5.3, 0.008), 28.7, 0.017, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)

    plt.show()

# param aone in [-1.1,1.4];
# param bone in [-1.8,2.4];
# dparam lone in [0.04,0.29];
#
# param atwo in [-0.45,0.14];
# param btwo in [-1.9,0.4];
# dparam ltwo in [0.320,0.550];
#
# param athree in [-69,-29];
# param bthree in [-158,-63];
# dparam lthree in [0.015,0.025];
#
# param afour in [5.3,44];
# param bfour in [13,92];
# dparam lfour in [0.008,0.025];


def infer_shape(args):
    max_mse = args.max_mse[0]  # todo:  max_mse is a list?
    max_delta_wcss = args.max_delta_wcss[0]
    sources = args.input
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()

    shapeit.get_times()


def gen_table1(args):
    tabel1 = {'seg': [], 'cluster': [], 'learn': [], 'total': []}

    trace_num = 10
    sig_length = 100

    for trace_num in [1, 10, 100]:
        for sig_length in [10, 100, 1000]:
            print(trace_num, sig_length)

            file_list = []
            for i in range(trace_num):
                filerootname = '/home/xin/Desktop/generator/experiments/pulse_nb_samples_100000_id_{}.csv'.format(trace_num)
                file_list.append(filerootname)

            print("list created")
            sources = file_list
            max_mse = args.max_mse[0]  # todo:  max_mse is a list?
            max_delta_wcss = args.max_delta_wcss[0]
            shapeit = ShapeIt(sources, max_mse, max_delta_wcss, sig_length)
            shapeit.mine_shape()

            print("The time for {} trace is".format(trace_num))
            ts, tc, tl, tt = shapeit.get_times()
            tabel1['seg'].append(ts)
            tabel1['cluster'].append(tc)
            tabel1['learn'].append(tl)
            tabel1['total'].append(tt)
    df = pd.DataFrame.from_dict(tabel1)
    df.to_csv('table1_2.csv')


def case_ekg(args):
    name1 = ['ekg2_1.csv', 'ekg2_2.csv', 'ekg2_3.csv']
    name2 = ['ekg_1.csv', 'ekg_2.csv', 'ekg_3.csv']
    max_mse = 0.001   # todo:  max_mse is a list?
    # max_mse = 0.05

    file_list = []
    for n in name1:
        file_list.append(os.path.join('ekg_data', n))
    print("list created")
    sources = file_list

    max_delta_wcss = args.max_delta_wcss[0]
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()

    file_list = []
    for n in name2:
        file_list.append(os.path.join('ekg_data', n))
    print("list created")
    sources = file_list
    max_delta_wcss = args.max_delta_wcss[0]
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()


def case_ekg_2(args):
    name1 = ['ekg2_1.csv', 'ekg2_2.csv', 'ekg2_3.csv', 'ekg_1.csv', 'ekg_2.csv', 'ekg_3.csv']
    max_mse = 0.0001  # 0.0008 also works. But error threshold too small do not work

    file_list = []
    for n in name1:
        file_list.append(os.path.join('ekg_data', n))
    print("list created")
    sources = file_list

    max_delta_wcss = args.max_delta_wcss[0]
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()


def case_sony(args):
    name1 = ['SonyAIBORobotSurface1_TEST_1_class_1', 'SonyAIBORobotSurface1_TEST_4_class_1',
             'SonyAIBORobotSurface1_TEST_5_class_1']
    name2 = ['SonyAIBORobotSurface1_TEST_2_class_2', 'SonyAIBORobotSurface1_TEST_3_class_2',
             'SonyAIBORobotSurface1_TEST_8_class_2']
    folder = 'sony'

    max_mse = 0.1


    file_list = []
    for n in name1:
        file_list.append(os.path.join(folder, n + '.csv'))
    print("list created")
    sources = file_list

    max_delta_wcss = args.max_delta_wcss[0]
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()

    file_list = []
    for n in name2:
        file_list.append(os.path.join(folder, n + '.csv'))
    print("list created")
    sources = file_list
    max_delta_wcss = args.max_delta_wcss[0]
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()

def case_sony_2(args):
    name1 = ['SonyAIBORobotSurface1_TEST_1_class_1', 'SonyAIBORobotSurface1_TEST_4_class_1',
             'SonyAIBORobotSurface1_TEST_5_class_1', 'SonyAIBORobotSurface1_TEST_2_class_2',
             'SonyAIBORobotSurface1_TEST_3_class_2',
             'SonyAIBORobotSurface1_TEST_8_class_2']
    folder = 'sony'

    max_mse = 0.5


    file_list = []
    for n in name1:
        file_list.append(os.path.join(folder, n + '.csv'))
    print("list created")
    sources = file_list

    max_delta_wcss = args.max_delta_wcss[0]
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()


def case_ekg_dejan(args):
    file_list_1 = []
    file_list_2 = []
    file_list_3 = []
    for filename in glob.glob(os.path.join('patient007/patient007/shifted_xy', "*.csv")):
    #for filename in glob.glob(os.path.join('ekg_data', "ekg_*.csv")):
        file_list_1.append(filename)
        file_list_3.append(filename)
    for filename in glob.glob(os.path.join('patient024/patient024/shifted_xy', "*.csv")):
    #for filename in glob.glob(os.path.join('ekg_data', "ekg2_*.csv")):
        file_list_2.append(filename)
        #file_list_3.append(filename)
    max_mse = 0.5

    max_delta_wcss = args.max_delta_wcss[0]
    #shapeit_1 = ShapeIt(file_list_1, 0.001, 0.1)
    #shapeit_1.load()
    #shapeit_1.segment()
    #shapeit_1.abstract()
    #shapeit_1.learn()
    #
    shapeit_2 = ShapeIt(file_list_2, 0.001, 1)
    shapeit_2.load()
    shapeit_2.segment()
    shapeit_2.abstract()
    shapeit_2.learn()

    #shapeit_3 = ShapeIt(file_list_3, 0.001, 5)
    #shapeit_3.load()
    #shapeit_3.segment()
    #shapeit_3.abstract()
    #shapeit_3.learn()
    #shapeit_3.learn_ekg(0,125)
    #shapeit_3.learn_ekg(125,250)

    # fig, axs = plt.subplots(1, 2)
    # for trace in shapeit_1.raw_traces:
    #     axs[0].plot(trace['Time'], trace['Value'], color='lightgrey')
    #
    # for segmented_trace in shapeit_1.segmented_traces:
    #     for segment in segmented_trace:
    #         t1 = segment[1]
    #         t2 = segment[2]
    #         slope = segment[3]
    #         offset = segment[4]
    #         x0, y0 = t1, slope * t1 + offset
    #         x1, y1 = t2, slope * t2 + offset
    #         #axs[0].plot([x0, x1], [y0, y1], linewidth=1, color='grey')
    #
    # for trace in shapeit_2.raw_traces:
    #     axs[1].plot(trace['Time'], trace['Value'], color='lightgrey')
    #     for segmented_trace in shapeit_2.segmented_traces:
    #         for segment in segmented_trace:
    #             t1 = segment[1]
    #             t2 = segment[2]
    #             slope = segment[3]
    #             offset = segment[4]
    #             x0, y0 = t1, slope * t1 + offset
    #             x1, y1 = t2, slope * t2 + offset
    #             #axs[1].plot([x0, x1], [y0, y1], linewidth=1, color='grey')
    #
    # plt.show()

    # print('Shape 1 statistics')
    # print(shapeit_1.abstract_traces)
    # print(shapeit_1.alphabet_box_dict)
    #
    # print('----------------')
    # print('Shape 2 statistics')
    # print(shapeit_2.abstract_traces)
    # print(shapeit_2.alphabet_box_dict)
    #
    # print('----------------')
    print('Shape 3 statistics')
    print(shapeit_3.abstract_traces)
    print(shapeit_3.alphabet_box_dict)

    #shapeit.mine_shape()


def case_kleene_star(args):
    file_list = ["data/pulse1-1.csv", "data/pulse1-2.csv", "data/pulse1-3.csv"]
    file_list =  ["data/pulse2-1.csv", "data/pulse2-2.csv", "data/pulse2-3.csv"]
    file_list = ["data/pulse1-2.csv","data/pulse1-1.csv", "star_data/pulse1-1-repeat.csv"]
    file_list = ["star_data/pulse1-1-repeat.csv"]
    # file_list = ["data/pulse1-2.csv","data/pulse1-1.csv", "star_data/pulse1-1-2.csv"]

    # file_list = ["data/pulse1-2.csv", "star_data/pulse1-2-2.csv"]


    max_mse = 0.1

    sources = file_list

    max_delta_wcss = args.max_delta_wcss[0]
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()


def main():
    args = Options().parse()
    # infer_shape(args)

    # gen_table1(args)

    #case_ekg(args)
    #case_ekg_2(args)
    # case_sony(args)
    # case_sony_2(args)
    # case_sony(args)

    case_ekg_dejan(args)

    #case_kleene_star(args)

if __name__ == '__main__':
    main()


