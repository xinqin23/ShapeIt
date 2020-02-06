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

    def __init__(self):
        alphabet = dict()

        sources = []
        raw_traces = []
        segmented_traces = []
        abstract_finite_traces = []

        learned_automaton = None
        learned_expression = None

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

