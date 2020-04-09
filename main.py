import argparse

from shapeit.shape_it import ShapeIt
from options import Options


def infer_shape(args):
    max_mse = args.max_mse[0]  # todo:  max_mse is a list?
    max_delta_wcss = args.max_delta_wcss[0]
    sources = args.input
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()

    shapeit.get_times()

def gen_table1(args):
    trace_num = 1
    sig_length = 10
    file_list = []
    for i in range(trace_num):
        filerootname = '/home/xin/Desktop/generator/experiments/pulse_nb_samples_100000_id_{}.csv'.format(trace_num)
        file_list.append(filerootname)

    sources = file_list
    max_mse = args.max_mse[0]  # todo:  max_mse is a list?
    max_delta_wcss = args.max_delta_wcss[0]
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss, sig_length)
    shapeit.mine_shape()
    t1, t2, t3 = shapeit.get_times()
    total_time_consumed = t1 + t2+ t3


    print("The time for {} trace is {}".format(trace_num, total_time_consumed))


def main():
    args = Options().parse()
    # infer_shape(args)

    gen_table1(args)


if __name__ == '__main__':
    main()


