from shapeit.ShapeIt import ShapeIt
from options import Options


def infer_shape(args):
    max_mse = args.max_mse[0]
    max_delta_wcss = args.max_delta_wcss[0]
    sources = args.input
    shapeit = ShapeIt(sources, max_mse, max_delta_wcss)
    shapeit.mine_shape()

if __name__ == '__main__':
    args = Options().parse()
    # infer_shape(args)

