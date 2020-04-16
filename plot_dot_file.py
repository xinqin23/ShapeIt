from graphviz import Source
# path = 'automaton_to_regex/sony_automaton_better.dot' # added redundent info into the .dot file so not working
# s = Source.from_file(path)
# s.view()



temp = """
digraph g {

	s0 [shape="circle" label="0"];
	s1 [shape="doublecircle" label="1"];
	s2 [shape="circle" label="2"];
	s3 [shape="circle" label="3"];
	s4 [shape="circle" label="4"];
	s5 [shape="circle" label="5"];
	s6 [shape="circle" label="6"];
	s0 -> s1 [label="3"];
	s2 -> s0 [label="3"];
	s3 -> s2 [label="1"];
	s4 -> s3 [label="3"];
	s4 -> s6 [label="4"];
	s5 -> s5 [label="0"];
	s5 -> s4 [label="2"];
	s5 -> s4 [label="3"];
	s5 -> s6 [label="4"];
	s5 -> s5 [label="5"];
	s6 -> s1 [label="1"];

__start0 [label="" shape="none" width="0" height="0"];
__start0 -> s5;

}
"""
s = Source(temp, filename="SonyBest", format="png")
s.view()