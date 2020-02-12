import argparse
from shapeit.ShapeIt import ShapeIt

def infer_shape(args):
    max_mse = args.max_mse[0]
    sources = args.input
    shapeit = ShapeIt(sources, max_mse)
    shapeit.mine_shape()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn a shape expression from positive examples.')
    parser.add_argument('--input', metavar='filename.csv', nargs='+',
                        help='List of input files in CSV format.')
    parser.add_argument('--max-mse', type=float, nargs=1,
                        help='Maximum mean squared error (MSE).')
    args = parser.parse_args()

    infer_shape(args)

