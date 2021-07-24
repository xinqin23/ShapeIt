import argparse

from shapeit.shape_it import ShapeIt
from options import Options
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt


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


def case_sony_3(args):
    file_list_1 = []
    file_list_2 = []
    for filename in glob.glob(os.path.join('sony_2/class_1', "*.csv")):
        file_list_1.append(filename)
    for filename in glob.glob(os.path.join('sony_2/class_2', "*.csv")):
        file_list_2.append(filename)
    max_mse = 0.5

    max_delta_wcss = args.max_delta_wcss[0]
    shapeit_2 = ShapeIt(file_list_2, 0.5, 5)
    shapeit_2.load()
    shapeit_2.segment()
    shapeit_2.abstract()
    shapeit_2.learn()

    shapeit_1 = ShapeIt(file_list_1, 0.5, 5)
    shapeit_1.load()
    shapeit_1.segment()
    shapeit_1.abstract()
    shapeit_1.learn()
    #shapeit_1.kmeans = shapeit_2.kmeans
    #shapeit_1.abstract_custom()

    print(shapeit_1.abstract_traces)
    print(shapeit_1.alphabet_box_dict)

    print(shapeit_2.abstract_traces)
    print(shapeit_2.alphabet_box_dict)

    fig, axs = plt.subplots(1)
    for trace in shapeit_1.raw_traces:
        axs.plot(trace['time'], trace['value'], color='grey')

    for segmented_trace in shapeit_1.segmented_traces:
        for segment in segmented_trace:
            t1 = segment[1]
            t2 = segment[2]
            slope = segment[3]
            offset = segment[4]
            x0, y0 = t1, slope * t1 + offset
            x1, y1 = t2, slope * t2 + offset
            #axs.plot([x0, x1], [y0, y1], linewidth=1, color='blue')

    for trace in shapeit_2.raw_traces:
        axs.plot(trace['time'], trace['value'], color='red')
        #axs.plot(shapeit_1.raw_traces[15]['time'], shapeit_1.raw_traces[4]['value'], color='red')
        # for segmented_trace in shapeit_2.segmented_traces:
        #     for segment in segmented_trace:
        #         t1 = segment[1]
        #         t2 = segment[2]
        #         slope = segment[3]
        #         offset = segment[4]
        #         x0, y0 = t1, slope * t1 + offset
        #         x1, y1 = t2, slope * t2 + offset
        #         #axs[1].plot([x0, x1], [y0, y1], linewidth=1, color='grey')

    plt.show()

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

    case_sony_3(args)


    #case_kleene_star(args)

if __name__ == '__main__':
    main()

