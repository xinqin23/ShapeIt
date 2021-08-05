import os
from datetime import datetime
import jpype
import jpype.imports
from jpype.types import *
import pandas as pd
from sklearn.cluster import KMeans
from timeit import default_timer as timer
from tabulate import tabulate
import networkx as nx

from shapeit.segmenter import *

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

    def __init__(self, sources=[], max_mse=0.1, max_delta_wcss=0.01, sig_length=None, plog_seg=True, sampling_period=0.01, time_header="timestamp", value_header="value"):
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

        self.learned_automaton = nx.MultiDiGraph()
        self.learned_expression = ""

        self.sampling_period = sampling_period

        self.time_header = time_header
        self.value_header = value_header

        self.total_segment_time = 0
        self.total_cluster_time = 0
        self.learning_time = 0

        self.sig_length = sig_length
        self.plot_seg = plog_seg

        self.add_noise = False
        self.fix_noise_tune_threshold = False

        random.seed(datetime.now())

    def mine_shape(self):
        self.load()
        print("Load finish")

        self.segment()
        self.abstract()
        self.learn()

    def segment_only(self):
        self.load()
        print("Load Finish")
        self.segment()
        return self.segments   # each [] from one trace

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
            x = raw_trace[self.time_header]
            y = raw_trace[self.value_header]

            start_time = timer()
            segmented_trace = compute_optimal_splits(x, y, self.max_mse, False)
            end_time = timer()
            time_consumed = end_time - start_time
            self.total_segment_time += time_consumed

            number_of_shape_found = segmented_trace.shape[0]
            print("Number of segments found: {}".format(number_of_shape_found))

            if self.plot_seg:
                df = pd.DataFrame(segmented_trace,
                                  columns=["Line Nr", "Start Idx", "End Idx", "Slope", "Offset", "Error", "Duration"])
                df[["Line Nr", "Start Idx", "End Idx"]] = df[["Line Nr", "Start Idx", "End Idx"]].astype(int)
                print(tabulate(df, headers='keys', showindex=False, tablefmt='psql'))

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

        for i in range(len(self.n_segmented_traces)):
            n_segmented_trace = self.n_segmented_traces[i]
            segmented_trace = self.segmented_traces[i]
            abstract_trace = []
            for j in range(len(n_segmented_trace)):
                n_segment = n_segmented_trace[j]
                segment = segmented_trace[j]

                n_slope = n_segment[0]
                n_offset = n_segment[1]
                n_duration = n_segment[2]

                start = segment[1]*self.sampling_period
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

    def get_alphabet_box_dict(self):
        return self.alphabet_box_dict

    def learn(self):  # Calling Java
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


        learner.addPositiveSamples(words_list)
        model = learner.computeModel()

        self.learned_automaton = self.learnlib2dfa(model)

        jpype.shutdownJVM()

        self.learned_expression = self.dfa2re(self.learned_automaton)

    def learnlib2dfa(self, model):
        aut = nx.MultiDiGraph()
        states = model.getStates()
        init_state = model.getInitialState()

        for state in states:
            accepting = False
            initial = False

            if model.isAccepting(state):
                accepting = True

            if state == init_state:
                initial = True

            aut.add_node(state, initial=initial, accepting=accepting)

            for letter in self.alphabet:
                transitions = model.getTransitions(state, letter)
                if transitions.size() > 0:
                    aut.add_edge(state, transitions.toArray()[0], label=str(letter))
        return aut

    def dfa2re(self, aut, init_node):
        INIT_STATE = -1
        ACC_STATE = -2
        EPSILON = 'eps'

        # Collect all states, initial states, and accepting states
        states_initial = []
        states_accepting = []
        states_to_process = set()
        for state, state_data in aut.nodes(data=True):
            if state_data['initial']:
                states_initial.append(state)
            if state_data['accepting']:
                states_accepting.append(state)
            states_to_process.add(state)

        # We expect a single initial state
        if len(states_initial) != 1:
            raise Exception("dfa2re: a single initial state is expected")

        # We expect at least one accepting state
        if len(states_accepting) <= 0:
            raise Exception("dfa2re: at least one accepting state is expected")

        state_initial = states_initial[0]
        init_incoming = aut.in_edges(state_initial)

        # Add a new initial state only if there are incoming edges to the old initial state
        if len(init_incoming) > 0:
            is_accepting = aut.nodes[state_initial]['accepting']
            aut.add_node(INIT_STATE, initial=True, accepting=False)
            aut.add_node(init_node, initial=False, accepting=is_accepting)
            aut.add_edge(INIT_STATE, state_initial, label=EPSILON)
            state_initial = INIT_STATE
        else:
            states_to_process.remove(state_initial)


        # Add a new accepting state
        # We skip this state if (1) there is only one final state, and
        # (2) it has no outgoing edges
        if len(states_accepting) > 1:
            aut.add_node(ACC_STATE, initial=False, accepting=True)
            for state_accepting in states_accepting:
                aut.add_node(state_accepting, accepting=False)
                aut.add_edge(state_accepting, ACC_STATE, label=EPSILON)
        elif len(states_accepting) == 1:
            # Check if there is an accepting state with outgoing edge
            has_out = False
            state_accepting = states_accepting[0]
            accepting_outgoing = aut.out_edges(state_accepting)
            if len(accepting_outgoing) > 0:
                aut.add_node(ACC_STATE, initial=False, accepting=True)
                aut.add_node(state_accepting, accepting=False)
                aut.add_edge(state_accepting, ACC_STATE, label=EPSILON)
            else:
                states_to_process.remove(state_accepting)

        # Merge multiple edges between two states 0 and 1
        edges_to_remove = []
        for source in aut.nodes:
            for target in aut.nodes:
                if aut.number_of_edges(source, target) > 1:
                    edge_labels = aut.get_edge_data(source, target)
                    counter = 0
                    new_label = edge_labels[0]['label']
                    for label in edge_labels:
                        if counter >= 1:
                            new_label = new_label + ' + ' + edge_labels[label]['label']
                        counter = counter + 1
                        edges_to_remove.append((source, target, label))
                    new_label = '(' + new_label + ')'
                    aut.add_edge(source, target, label=new_label)

        for edge in edges_to_remove:
            aut.remove_edge(edge[0], edge[1], edge[2])


        # We now remove state by state
        for state in states_to_process:
            edges_to_add = []
            edges_to_remove = []

            # Create self-loop label for states with self-loop
            self_loop_label = "eps"
            if aut.has_edge(state, state):
                self_loop_label = aut.get_edge_data(state, state)[0]['label']

            # Consider all pairs of incoming/outgoing edges to/from state
            for in_edge in aut.in_edges(state):
                edges_to_remove.append(in_edge)
                # Ignore the self loop
                if in_edge[0] == in_edge[1]:
                    continue

                # Ignore self loop
                for out_edge in aut.out_edges(state):
                    edges_to_remove.append(out_edge)
                    if out_edge[0] == out_edge[1]:
                        continue

                    in_label = self.label_union(aut, in_edge)
                    out_label = self.label_union(aut, out_edge)

                    new_label = self.label_concat(in_label, self_loop_label, out_label)
                    edges_to_add.append((in_edge[0], out_edge[1], new_label))

            # Add edges
            for edge in edges_to_add:
#                if aut.has_edge(edge[0], edge[1]):
#                    aut.remove_edge(edge[0], edge[1])
                aut.add_edge(edge[0], edge[1], label=edge[2])

            # Remove edges
            for edge in edges_to_remove:
                if aut.has_edge(edge[0], edge[1]):
                    aut.remove_edge(edge[0], edge[1])

        # Remove states
        for state in states_to_process:
            aut.remove_node(state)

        # Merge multiple edges between two states 0 and 1
        edges_to_remove = []
        for source in aut.nodes:
            for target in aut.nodes:
                if aut.number_of_edges(source, target) > 1:
                    edge_labels = aut.get_edge_data(source, target)
                    counter = 0
                    new_label = edge_labels[0]['label']
                    for label in edge_labels:
                        if counter >= 1:
                            new_label = new_label + ' + ' + edge_labels[label]['label']
                        counter = counter + 1
                        edges_to_remove.append((source, target, label))
                    new_label = '(' + new_label + ')'
                    aut.add_edge(source, target, label=new_label)

        for edge in edges_to_remove:
            aut.remove_edge(edge[0], edge[1], edge[2])

        if not aut.number_of_edges(INIT_STATE, ACC_STATE) == 1:
            raise Exception("Bug in dfa2re: only one transition should remain.")

        edge_labels = aut.get_edge_data(INIT_STATE, ACC_STATE)
        for label in edge_labels:
            out = aut.get_edge_data(INIT_STATE, ACC_STATE)[label]['label']

        return out


    def label_union(self, aut, edge):
        new_label = ""

        counter = 0
        for edge_data in aut.get_edge_data(edge[0], edge[1]):
            label = aut.get_edge_data(edge[0], edge[1])[edge_data]['label']
            if counter == 0:
                new_label = label
            else:
                new_label = new_label + ' + ' + label

        new_label = '(' + new_label + ')'
        return new_label

    def label_concat(self, left, middle, right):
        out = "eps"
        left_empty = left == "eps" or left == "(eps)"
        right_empty = right == "eps" or right == "(eps)"
        middle_empty = middle == "eps" or middle == "(eps)"

        if left_empty and middle_empty and not right_empty:
            out = right
        elif left_empty and not middle_empty and right_empty:
            out = '(' + middle + ')*'
        elif left_empty and not middle_empty and not right_empty:
            out = '(' + middle + ')*.' + right
        elif not left_empty and middle_empty and right_empty:
            out = left
        elif not left_empty and not middle_empty and right_empty:
            out = left + '.(' + middle + ')*'
        elif not left_empty and middle_empty and not right_empty:
            out = left + '.' + right
        elif not left_empty and not middle_empty and not right_empty:
            out = left + '.(' + middle + ')*.' + right

        return out







        #             in_data = aut.get_edge_data(in_edge[0], in_edge[1])
        #             out_data = aut.get_edge_data(out_edge[0], out_edge[1])
        #
        #             label_left = ""
        #             for in_datum in in_data.values():
        #                 if not label_left and not in_datum['label'] == '-1':
        #                     label_left = in_datum['label']
        #                 elif label_left and not in_datum['label'] == '-1':
        #                     label_left = '(' + label_left + ' + ' + in_datum['label'] + ')'
        #
        #             label_right = ""
        #             for out_datum in out_data.values():
        #                 if not label_right and not out_datum['label'] == '-1':
        #                     label_right = out_datum['label']
        #                 elif label_right and not out_datum['label'] == '-1':
        #                     label_right = '(' + label_right + ' + ' + out_datum['label'] + ')'
        #
        #             if (aut.has_edge(node, node)):
        #                 if not label_left and not label_right:
        #                     new_label = self_loop_label + '*'
        #                 elif not label_left and label_right:
        #                     new_label = self_loop_label + '*.' + label_right
        #                 elif label_left and not label_right:
        #                     new_label = label_left + '.' + self_loop_label + '*'
        #                 else:
        #                     new_label = label_left + '.' + self_loop_label + '*.' + label_right
        #             else:
        #                 if not label_left and label_right:
        #                     new_label = label_right
        #                 elif label_left and not label_right:
        #                     new_label = label_left
        #                 elif label_left and label_right:
        #                     new_label = label_left + '.' + label_right
        #
        #             edges_to_remove.append(in_edge)
        #             edges_to_remove.append(out_edge)
        #             edges_to_add.append((in_edge[0], out_edge[1], new_label))
        #
        #     for edge in edges_to_remove:
        #         if aut.has_edge(edge[0], edge[1]):
        #             aut.remove_edge(edge[0], edge[1])
        #
        #     aut.remove_node(node)
        #
        #     for edge in edges_to_add:
        #         aut.add_edge(edge[0], edge[1], label=edge[2])
        #
        # data = aut.get_edge_data(-2, -1)
        # return data[0]['label']

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
