from graphviz import Source


def plot_by_txt():
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


def plot_by_file():
    path = 'automaton_to_regex/automaton.dot'  # added redundent info into the .dot file so not working
    s = Source.from_file(path, format="png")
    s.view()


def plot_by_txt_sony():
    temp = """
    digraph g {

        s0 [shape="circle" label="s0"];
        s1 [shape="doublecircle" label="s1"];
        s2 [shape="circle" label="s2"];
        s3 [shape="circle" label="s3"];
        s4 [shape="circle" label="s4"];
        s5 [shape="circle" label="s5"];
        s6 [shape="circle" label="s6"];
        s0 -> s1 [label="D"];
        s2 -> s0 [label="D"];
        s3 -> s2 [label="B"];
        s4 -> s3 [label="D"];
        s4 -> s6 [label="E"];
        s5 -> s5 [label="A"];
        s5 -> s4 [label="C"];
        s5 -> s4 [label="D"];
        s5 -> s6 [label="E"];
        s5 -> s5 [label="F"];
        s6 -> s1 [label="B"];

    __start0 [label="" shape="none" width="0" height="0"];
    __start0 -> s5;

    }
    """
    s = Source(temp, filename="SonyBest", format="png")
    s.view()

def plot_by_txt_sony_cluster5():
    temp = """
   digraph g {

	s0 [shape="circle" label="s0"];
	s1 [shape="doublecircle" label="s1"];
	s2 [shape="circle" label="s2"];
	s3 [shape="circle" label="s3"];
	s0 -> s1 [label="C"];
	s1 -> s1 [label="B"];
	s2 -> s0 [label="B"];
	s2 -> s0 [label="E"];
	s3 -> s3 [label="A"];
	s3 -> s2 [label="B"];
	s3 -> s2 [label="D"];

__start0 [label="" shape="none" width="0" height="0"];
__start0 -> s3;

}
    """
    s = Source(temp, filename="SonyBest5", format="png")
    s.view()

plot_by_txt_sony_cluster5()



