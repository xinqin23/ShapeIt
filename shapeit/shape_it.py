import os
import logging
import jpype
import jpype.imports
from jpype.types import *
import pandas as pd
from sklearn.cluster import KMeans
from timeit import default_timer as timer
import random
import pickle
import numpy as np


from .segmenter import *

class ShapeIt(object):
    """A class used as a container for the ShapeIt algorithm and its associated data structures

        Attributes:
            alphabet : set of alphabet letters
            alphabet_box_dict: Dictionary of (letter, bounding box) pairs

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

    def __init__(self, sources, max_mse, max_delta_wcss, sig_length=None, plog_seg=True):
        self.alphabet = set()
        self.alphabet_box_dict = dict()

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

        self.total_segment_time = 0
        self.total_cluster_time = 0
        self.learning_time = 0

        self.sig_length = sig_length
        self.plot_seg = plog_seg

        random.seed(2)

    def mine_shape(self):
        self.load()
        print("Load finish")

        self.segment()
        self.abstract()
        self.learn()

    def get_times(self):
        total_time = self.total_segment_time + self.total_cluster_time + self.learning_time
        print("Seg: {}, Cluster: {}, Learn: {}".format(self.total_segment_time, self.total_cluster_time,
                                                       self.learning_time, total_time))

        return self.total_segment_time, self.total_cluster_time, self.learning_time, total_time
        # depends on how many seg is called

    def load(self):
        for source in self.sources:
            raw_trace = pd.read_csv(source, sep=r'\s*,\s*', nrows=self.sig_length)  # nrows is needed to reduce no
            # need data
            self.raw_traces.append(raw_trace)

    def segment(self):
        for raw_trace in self.raw_traces:
            # x = raw_trace["Time"].values
            # y = raw_trace["Value"].values

            # for sony data
            x = raw_trace["time"].values
            y = raw_trace["value"].values

            # plt.plot(x, y)
            # plt.show()

            start_time = timer()
            segmented_trace = compute_optimal_splits(x, y, self.max_mse, False)
            end_time = timer()
            time_consumed = end_time - start_time
            self.total_segment_time += time_consumed
            # print("Total Elapsed Seg Computation Time: {} sec.".format(time_consumed))

            if self.plot_seg:
                df = pd.DataFrame(segmented_trace,
                                  columns=["Line Nr", "Start Idx", "End Idx", "Slope", "Offset", "Error", "Duration"])
                df[["Line Nr", "Start Idx", "End Idx"]] = df[["Line Nr", "Start Idx", "End Idx"]].astype(int)
                print(tabulate(df, headers='keys', showindex=False, tablefmt='psql'))

                _ = plot_splits(x, y, segmented_trace, plotLegend=False)
                plt.show()

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

        self.normalize()

        nb_clusters = 1

        kmeans = KMeans(n_clusters=nb_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        letters = kmeans.fit_predict(self.n_segments)
        current_wcss = kmeans.inertia_
        past_wcss = float("inf")
        delta_wcss = past_wcss - current_wcss
        wcss.append(current_wcss)

        start_time = timer()
        while nb_clusters < len(self.segments) and delta_wcss > self.max_delta_wcss:
            nb_clusters = nb_clusters + 1

            nb_clusters = 5  # for ekg
            nb_clusters = 6 # for sony try
            kmeans = KMeans(n_clusters=nb_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
            letters = kmeans.fit_predict(self.n_segments)

            past_wcss = current_wcss
            current_wcss = kmeans.inertia_
            delta_wcss = past_wcss - current_wcss

            wcss.append(current_wcss)

        end_time = timer()
        self.total_cluster_time += (end_time - start_time)


        letters = set(letters)
        self.alphabet = letters
        let_seg_dict = dict()
        for letter in letters:
           let_seg_dict[letter] = []

        #for n_segmented_trace in self.n_segmented_traces:
        for i in range(len(self.n_segmented_traces)):
            n_segmented_trace = self.n_segmented_traces[i]
            segmented_trace = self.segmented_traces[i]
            abstract_trace = []
            #for segment in n_segmented_trace:
            for j in range(len(n_segmented_trace)):
                n_segment = n_segmented_trace[j]
                segment = segmented_trace[j]

                n_slope = n_segment[0]
                n_offset = n_segment[1]
                n_duration = n_segment[2]

                start = segment[1]*0.01 # at least for ekg
                slope = segment[3]
                offset = segment[4]
                duration = segment[6]
                relative_offset = slope * start + offset

                letter = kmeans.predict([[n_slope, n_offset, n_duration]])
                abstract_trace.append(letter[0])
                let_seg_dict[letter[0]].append([slope, relative_offset, duration])
            self.abstract_traces.append(abstract_trace)

        print("Abstract traces", self.abstract_traces)

        for letter in letters:
            let_seg_list = np.array(let_seg_dict[letter])
            lower_bound = np.min(let_seg_list, axis=0)
            upper_bound = np.max(let_seg_list, axis=0)

            self.alphabet_box_dict[letter] = [lower_bound, upper_bound]
        print(self.alphabet_box_dict)

        with open('automaton_to_regex/abstract_traces.p', 'wb') as f:
            pickle.dump(self.abstract_traces, f)

        with open('automaton_to_regex/alphabet_box.p', 'wb') as f:
            pickle.dump(self.alphabet_box_dict, f)

    def learn(self):
        # Set up CLASSPATH and start the Java Virtual Machine
        learnlib_folder = os.path.join("lib", "learnlib-distribution-0.14.0-dependencies-bundle.jar")
        jpype.addClassPath(learnlib_folder)

        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea")

        # Load LearnLib classes needed by the tool
        BlueFringeMDLDFA = jpype.JClass("de.learnlib.algorithms.rpni.BlueFringeMDLDFA")
        Visualization = jpype.JClass("net.automatalib.visualization.Visualization")
        Word = jpype.JClass("net.automatalib.words.Word")
        Alphabets = jpype.JClass("net.automatalib.words.impl.Alphabets")
        GraphDOT = jpype.JClass("net.automatalib.serialization.dot.GraphDOT")

        from java.util import ArrayList
        from java.io import BufferedWriter
        from java.io import FileWriter
        from java.io import Writer
        from java.io import IOException;

        alphabet_list = ArrayList()
        for letter in self.alphabet:
            alphabet_list.add(JInt(letter))

        alphabet = Alphabets.fromList(alphabet_list)
        learner = BlueFringeMDLDFA(alphabet)

        words_list = ArrayList()
        for trace in self.abstract_traces:
            word_list = ArrayList()
            for letter in trace:
                word_list.add(JInt(letter))
                word = Word.fromList(word_list)
            words_list.add(word)


        start_time = timer()
        learner.addPositiveSamples(words_list)
        model = learner.computeModel()

        end_time = timer()
        time_consumed = end_time - start_time
        self.learning_time = time_consumed

        Visualization.visualize(model, alphabet);

        f = FileWriter("automaton_to_regex/automaton.dot")
        wf = BufferedWriter(f)
        GraphDOT.write(model,  wf)


        # when not using for loop
        # Close the Java Virtual Machine
        # jpype.shutdownJVM()

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
