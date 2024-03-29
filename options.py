import argparse

class Options():

    def __init__(self):
        self.initialized = False
        self.args = None
        self.parser = argparse.ArgumentParser(description='Learn a shape expression from positive examples.',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        # https://docs.python.org/3/library/argparse.html#nargs
        self.parser.add_argument('--input', default=["data/pulse1-1.csv", "data/pulse1-2.csv", "data/pulse1-3.csv",
                                                     "data/pulse2-1.csv", "data/pulse2-2.csv", "data/pulse2-3.csv"],
                                 metavar='filename.csv',
                                 nargs='+',
                                 help='List of input files in CSV format.')
        self.parser.add_argument('--max-mse', default=[0.5], type=float, nargs=1, #todo: was 0.3
                                 help='Maximum mean squared error (MSE).')
        self.parser.add_argument('--max-delta-wcss', default=[10], type=float, nargs=1,
                                 help='Maximum difference between two consecutive WCSS.') # todo: default value update

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.args = self.parser.parse_args()
        return self.args



