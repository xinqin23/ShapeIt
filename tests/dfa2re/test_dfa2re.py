import pytest
import networkx as nx
from shapeit.shape_it import ShapeIt

def test_dfa2re_1():
	aut = nx.MultiDiGraph()
	aut.add_node(0, initial=True, accepting=False)
	aut.add_node(1, initial=False, accepting=True)
	aut.add_edge(0, 1, label='0')
	aut.add_edge(1, 0, label='0')

	shapeit = ShapeIt()
	re = shapeit.dfa2re(aut, 0)

	assert re == '0.(0.0)*'

def test_dfa2re_2():
	aut = nx.MultiDiGraph()
	aut.add_node(0, initial=True, accepting=True)
	aut.add_node(1, initial=False, accepting=False)
	aut.add_edge(0, 1, label='0')
	aut.add_edge(1, 0, label='0')

	shapeit = ShapeIt()
	re = shapeit.dfa2re(aut, 0)

	assert re == '(0.0)*'