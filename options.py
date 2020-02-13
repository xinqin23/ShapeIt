import argparse

class Options():

    def __init__(self):
        self.initialized = False
        self.args = None
        self.parser = argparse.ArgumentParser(description='Learn a shape expression from positive examples.',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        self.parser.add_argument('--input', metavar='filename.csv', nargs='+',
                                 help='List of input files in CSV format.')
        self.parser.add_argument('--max-mse', type=float, nargs=1,
                                 help='Maximum mean squared error (MSE).')
        self.parser.add_argument('--max-delta-wcss', type=float, nargs=1,
                                 help='Maximum difference between two consecutive WCSS.')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.args = self.parser.parse_args()
        return self.args


