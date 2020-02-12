import pandas as pd
from segmenter import *

class ShapeIt:
    """A class used as a container for the ShapeIt algorithm and its associated data structures

        Attributes:
            alphabet : Dictionary of (letter, semantics) pairs

            sources : list of .csv files containing raw time series
            raw_traces : list of raw traces
            segmented_traces : list of segmented traces
            abstract_finite_traces : list of abstract finite traces

            learned_automaton = automaton structure containing the inferred automaton
            learned_expression = data structure containing the inferred expression

        Methods
            get_spec_from_file - create and populate specification object from the text file
            parse - parse the specification
            update - update the specification
        """

    def __init__(self, sources, max_mse):
        self.alphabet = dict()

        self.max_mse = max_mse

        self.sources = sources
        self.raw_traces = []
        self.segmented_traces = []
        self.abstract_finite_traces = []

        self.learned_automaton = None
        self.learned_expression = None

    def mine_shape(self):
        self.load()
        self.segment()
        self.abstract()
        self.learn()

    def load(self):
        for source in self.sources:
            raw_trace = pd.read_csv(source)
            self.raw_traces.append(raw_trace)

    def segment(self):
        for raw_trace in self.raw_traces:
            x = raw_trace["Time"].values
            y = raw_trace["Value"].values
            segment = compute_optimal_splits(x, y, self.max_mse, False)
            self.segmented_traces.append(segment)


    def abstract(self):
        pass

    def learn(self):
        pass

    @property
    def alphabet(self):
        return self._alphabet

    @alphabet.setter
    def alphabet(self, alphabet):
        self._alphabet = alphabet

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, sources):
        self._sources = sources

    @property
    def raw_traces(self):
        return self._raw_traces

    @raw_traces.setter
    def raw_traces(self, raw_traces):
        self._raw_traces = raw_traces

    @property
    def segmented_traces(self):
        return self._segmented_traces

    @segmented_traces.setter
    def segmented_traces(self, segmented_traces):
        self._segmented_traces = segmented_traces

    @property
    def abstract_finite_traces(self):
        return self._abstract_finite_traces

    @abstract_finite_traces.setter
    def abstract_finite_traces(self, abstract_finite_traces):
        self._abstract_finite_traces = abstract_finite_traces
