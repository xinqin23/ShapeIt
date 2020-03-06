<<<<<<< HEAD
import argparse
from shapeit.shape_it import ShapeIt
=======
from shapeit.ShapeIt import ShapeIt
from options import Options

>>>>>>> 3a7dd3db0194e377129b1ef5c7b494641c13ca6d

def infer_shape(args):
    max_mse = args.max_mse[0]  # todo:  max_mse is a list?
    max_delta_wcss = args.max_delta_wcss[0]
    sources = args.input
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()


def main():
    args = Options().parse()
    infer_shape(args)

if __name__ == '__main__':
    main()


