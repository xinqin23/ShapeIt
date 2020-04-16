from graphviz import Source
# path = 'automaton_to_regex/sony_automaton_better.dot' # added redundent info into the .dot file so not working
# s = Source.from_file(path)
# s.view()



temp = """
digraph g {

	s0 [shape="circle" label="0"];
	s1 [shape="doublecircle" label="1"];
	s2 [shape="circle" label="2"];
	s3 [shape="doublecircle" label="3"];
	s0 -> s1 [label="1"];
	s2 -> s2 [label="0"];
	s2 -> s3 [label="2"];
	s2 -> s3 [label="3"];
	s2 -> s3 [label="5"];
	s3 -> s3 [label="1"];
	s3 -> s3 [label="3"];
	s3 -> s0 [label="4"];

__start0 [label="" shape="none" width="0" height="0"];
__start0 -> s2;

}
"""
s = Source(temp, filename="test.gv", format="png")
s.view()