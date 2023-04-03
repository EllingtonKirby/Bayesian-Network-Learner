from typing import Set
from time import time
from random import seed, random
from graphviz import Digraph

class Node:
    """
    A class to represent each node of the graph
    """
    def __init__(self, prev=[], value=0, variable='None') -> None:
        self.prev = prev
        self.variable=variable
        self.value = value

class DAG:
    """
    A class to reprensent a Directed Acycic Graph
    """
    def __init__(self, nodes: Set[Node], randomize=False, density=0.5, random_seed=time()) -> None:
        if not randomize:
            self.nodes = nodes
        else:
            seed(random_seed)
            self.nodes = set()
            for node in nodes:  # Add a node with no parent
                self.nodes.add(Node(prev=[], variable=node.variable, value=node.value))
            colors = {node.variable: {node.variable} for node in self.nodes}
            for current_node in self.nodes:
                other_nodes = {node for node in self.nodes if node != current_node}
                for candidate_parent in other_nodes:    # Add or not the candidate as a parent of the current node
                    if random() < density:   # Probability test
                        if candidate_parent.variable not in colors[current_node.variable]:  # No cycle created
                            current_node.prev.append(candidate_parent)
                            colors[candidate_parent.variable] = colors[current_node.variable].union(
                                colors[candidate_parent.variable])

    def trace_dag(self):
        nodes, edges = set(), set()
        for v in self.nodes:
            if v not in nodes:
                nodes.add(v)
                for child in v.prev:
                    edges.add((child, v))
        return nodes, edges

    def draw_dot_from_digraph(self):
        dot = Digraph(format='svg', graph_attr={'rankdir':'LR'})

        nodes, edges = self.trace_dag()
        for n in nodes:
            uid = str(id(n))
            dot.node(name=uid, label="{ %s | value: %d }"%(n.variable, n.value), shape='record')
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)))

        return dot

    def print(self):
        for node in self.nodes:
            print(f"Node: {node.variable}, parents: {[parent.variable for parent in node.prev]}")

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | value: %d }"%(n.variable, n.value), shape='record')
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))

    return dot

# Test zone
if __name__ == "__main__":
    node_set = {Node(variable=i) for i in range(0, 8)}
    test = DAG(node_set, randomize=True, density=0.4, random_seed=time())
    test.print()
