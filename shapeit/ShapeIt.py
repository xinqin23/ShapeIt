import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np

from shapeit.segmenter import compute_optimal_splits, plot_splits


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

            max_mse - Maximum Mean Squared Error (MSE)
            max_delta_wcss - Maximum difference between two consicutive WCSS

        Methods
            get_spec_from_file - create and populate specification object from the text file
            parse - parse the specification
            update - update the specification
        """

    def __init__(self, sources, max_mse, max_delta_wcss):
        self.alphabet = dict()

        self.max_mse = max_mse
        self.max_delta_wcss = max_delta_wcss

        self.sources = sources
        self.raw_traces = []
        self.segmented_traces = []
        self.abstract_finite_traces = []

        self.segments = []
        self.segments_dimension = 3  # todo: pass this as parameter in

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
        # total_time_consumed = 0
        for raw_trace in self.raw_traces:
            x = raw_trace["Time"].values
            y = raw_trace["Value"].values

            # todo: time for computing each seg to add around line 66
            # total_time_consumed += time
            segmented_trace = compute_optimal_splits(x, y, self.max_mse, False)
            self.segmented_traces.append(segmented_trace)

            # todo: for different csv index will change
            for segment in segmented_trace:
                start = segment[1]
                slope = segment[3]
                offset = segment[4]
                relative_offset = slope * start + offset
                duration = segment[6]
                seg = [slope, relative_offset, duration]
                self.segments.append(seg)

    def abstract(self):
        wcss = []
        nb_clusters = float("inf")
        for i in range(1, 5):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            pred_y = kmeans.fit_predict(self.segments)
            wcss.append(kmeans.inertia_)
        print(pred_y)

        # todo: map the pred_y result back to each trace

    def learn(self):
        pass

    def normalize(self, segments):
        sum_vec = np.sum(segments, axis=0)  # [a,b]
        normalized_seg = segments / sum_vec  # todo: add a flag to use only normalized term

        # todo: necessary to scale different dim to same order, for clustering to work.
        max_value = np.max(normalized_seg, axis=0)
        scale = max_value[0] / max_value[1]

        if self.segments_dimension == 3:
            scale2 = max_value[0] / max_value[2]
            return normalized_seg * [1, scale, scale2]
        else:
            return normalized_seg * [1, scale]

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
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = segments

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
