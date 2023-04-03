from typing import Set
from time import time
from random import seed, random
from graphviz import Digraph
from itertools import product
from scipy.special import loggamma
import pandas as pd

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
    
    def bayesian_dirichilet_score(self, data: pd.DataFrame):
        total_sum = 0
        variable_unique_counts = {variable: data[variable].unique() for variable in data.columns}

        # For each node we need to iterate over all the possible configurations of the parents
        for node in self.nodes:
            possible_values = variable_unique_counts[node.variable]
            
            parents = node.prev
            parent_variables = [parent.variable for parent in parents]
            parent_configurations = self.generate_parent_configurations(node, variable_unique_counts)
            
            r_i = len(possible_values)
            
            left_sum = 0
            # The possible configurations are generated via the cartesian product
            for config in parent_configurations:
                m_ij0 = 0

                # Build the DataFrame query that matches our parent configuration
                base_string = ""
                for (key,value) in zip(parent_variables, config):
                    base_string += f"{key} == {value} & "

                right_sum = 0
                for k in possible_values:
                    query_string = base_string + f"{node.variable} == {k}"
                    # Find all observations that have our parent configuration and a value of k for this node
                    m_ijk = len(data.query(query_string))
                    # We keep track of m_ij0 here to reduce the DataFrame queries
                    # Count all instances of this parent configuration across all values of Xi
                    m_ij0 += m_ijk
                    right_sum += loggamma(1 + m_ijk)

                left_sum += (loggamma(r_i) - loggamma(r_i + m_ij0)) + right_sum

            total_sum += left_sum + right_sum
        return total_sum
    
    def generate_parent_configurations(self, node: Node, unique_counts: dict):
            parents = node.prev
            possible_values = [unique_counts[parent.variable] for parent in parents]
            return list(product(*possible_values))

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

# Test zone
if __name__ == "__main__":
    node_set = {Node(variable=i) for i in range(0, 8)}
    test = DAG(node_set, randomize=True, density=0.4, random_seed=time())
    test.print()
