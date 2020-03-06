import os
import jpype
import jpype.imports
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering


from segmenter import *

class ShapeIt:
    """A class used as a container for the ShapeIt algorithm and its associated data structures

        Attributes:
            alphabet : Dictionary of (letter, semantics) pairs

            sources : list of .csv files containing raw time series
            raw_traces : list of raw traces
            segmented_traces : list of segmented traces
            n_segmented_traces : list of normalized segmented traces
            abstract_traces : list of abstract finite traces

            segments : list of all segments from all traces
            n_segments : list of all segments from all traces with normalized parameters


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
        self.n_segmented_traces = []
        self.abstract_traces = []

        self.segments = []
        self.n_segments = []

        self.learned_automaton = None
        self.learned_expression = None

    def mine_shape(self):
        self.load()
        self.segment()
        self.abstract()
        #self.learn()

    def load(self):
        for source in self.sources:
            raw_trace = pd.read_csv(source)
            self.raw_traces.append(raw_trace)

    def segment(self):
        for raw_trace in self.raw_traces:
            x = raw_trace["Time"].values
            y = raw_trace["Value"].values
            segmented_trace = compute_optimal_splits(x, y, self.max_mse, False)
            self.segmented_traces.append(segmented_trace)

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

        self.normalize()

        nb_clusters = 1
        kmeans = KMeans(n_clusters=nb_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        letters = kmeans.fit_predict(self.n_segments)
        current_wcss = kmeans.inertia_
        past_wcss = float("inf")
        delta_wcss = past_wcss - current_wcss
        wcss.append(current_wcss)

        while nb_clusters < len(self.segments) and delta_wcss > self.max_delta_wcss:
            nb_clusters = nb_clusters + 1
            kmeans = KMeans(n_clusters=nb_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
            letters = kmeans.fit_predict(self.n_segments)

            past_wcss = current_wcss
            current_wcss = kmeans.inertia_
            delta_wcss =  past_wcss - current_wcss

            wcss.append(current_wcss)

        letters = set(letters)
        let_seg_dict = dict()
        for letter in letters:
           let_seg_dict[letter] = []

        for n_segmented_trace in self.n_segmented_traces:
            abstract_trace = []
            for segment in n_segmented_trace:
                slope = segment[0]
                offset = segment[1]
                duration = segment[2]
                letter = kmeans.predict([[slope, offset, duration]])
                abstract_trace.append(letter[0])
                let_seg_dict[letter[0]].append([slope, offset, duration])
            self.abstract_traces.append(abstract_trace)

        for letter in letters:
           self.alphabet[letter] = [min(let_seg_dict[letter]), max(let_seg_dict[letter])]

        print(self.alphabet)

    def learn(self):
        # Set up CLASSPATH and start the Java Virtual Machine
        learnlib_folder = os.path.join("lib", "learnlib-distribution-0.14.0-dependencies-bundle.jar")
        jpype.addClassPath(learnlib_folder)
        startJVM(getDefaultJVMPath(), "-ea")

        # Load LearnLib classes needed by the tool
        BlueFringeMDLDFA = jpype.JClass("de.learnlib.algorithms.rpni.BlueFringeMDLDFA")
        PassiveDFALearner = jpype.JClass("de.learnlib.api.algorithm.PassiveLearningAlgorithm.PassiveDFALearner")
        DFA = jpype.JClass("net.automatalib.automata.fsa.DFA")
        Visualization = jpype.JClass("net.automatalib.visualization.Visualization")
        Alphabet = jpype.JClass("net.automatalib.words.Alphabet")
        Word = jpype.JClass("net.automatalib.words.Word")
        Alphabets = jpype.JClass("net.automatalib.words.impl.Alphabets")

        from java.util import Arrays
        from java.util import Collection

        # Close the Java Virtual Machine
        shutdownJVM()

    def normalize(self):
        slopes = [row[0] for row in self.segments]
        relative_offsets = [row[1] for row in self.segments]
        durations = [row[2] for row in self.segments]

        min_slope = min(slopes)
        max_slope = max(slopes)
        min_relative_offset = min(relative_offsets)
        max_relative_offset = max(relative_offsets)
        min_duration = min(durations)
        max_duration = max(durations)

        range_slope = max_slope - min_slope
        range_relative_offset = max_relative_offset - min_relative_offset
        range_duration = max_duration - min_duration

        normalized_segments = []

        for segment in self.segments:
            if range_slope > 0:
                normalized_slope = (segment[0] - min_slope) / (max_slope - min_slope)
            else:
                normalized_slope = segment[0]
            if range_relative_offset > 0:
                normalized_relative_offset = (segment[1] - min_relative_offset) / (
                            max_relative_offset - min_relative_offset)
            else:
                normalized_relative_offset = segment[1]
            if range_duration > 0:
                normalized_duration = (segment[2] - min_duration) / (max_duration - min_duration)
            else:
                normalized_duration = segment[2]
            normalized_segment = [normalized_slope, normalized_relative_offset, normalized_duration]
            normalized_segments.append(normalized_segment)

        self.n_segments = normalized_segments

        for segmented_trace in self.segmented_traces:
            normalized_segmented_trace = []
            for segment in segmented_trace:
                normalized_segment = []
                start = segment[1]
                slope = segment[3]
                offset = segment[4]
                duration = segment[6]
                relative_offset = slope*start + offset
                if range_slope > 0:
                    normalized_slope = (slope - min_slope) / (max_slope - min_slope)
                else:
                    normalized_slope = slope
                if range_relative_offset > 0:
                    normalized_relative_offset = (relative_offset - min_relative_offset) / (
                            max_relative_offset - min_relative_offset)
                else:
                    normalized_relative_offset = relative_offset
                if range_duration > 0:
                    normalized_duration = (duration - min_duration) / (max_duration - min_duration)
                else:
                    normalized_duration = duration
                normalized_segment = [normalized_slope, normalized_relative_offset, normalized_duration]
                normalized_segmented_trace.append(normalized_segment)
            self.n_segmented_traces.append(normalized_segmented_trace)



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