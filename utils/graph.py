from typing import Set
from time import time
from random import seed, random
from graphviz import Digraph
from itertools import product
from scipy.special import loggamma
import pandas as pd
import numpy as np

class Node:
    """
    A class to represent each node of the graph
    """
    def __init__(self, prev=[], value=0, variable='None') -> None:
        self.prev = prev
        self.variable = variable
        self.value = value

    def add_prev(self, node):
        if node not in self.prev:
            self.prev.append(node)

    def remove_prev(self, node):
        self.prev.remove(node)

class DAG:
    """
    A class to reprensent a Directed Acycic Graph
    """
    def __init__(self, nodes: Set[Node], randomize=False, density=0.5, random_seed=time()) -> None:
        if not randomize:
            # self.nodes = deepcopy(nodes)
            # Stupid copy
            node_list = list()
            for node in nodes:  # Build new Nodes
                new_node = Node(variable=node.variable, value=node.value)
                node_list.append(new_node)
            for node in nodes:  # Fill the prev
                current_node = find_node(node_list, node.variable)
                current_node.prev = []
                for parent in node.prev:    # find the parent in new DAG
                    new_parent = find_node(node_list, parent.variable)
                    current_node.add_prev(new_parent)
            self.nodes = set(node_list)
            self.colors = self.coloring()
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
                            for parent in candidate_parent.prev:
                                colors[parent.variable] = colors[parent.variable].union(colors[candidate_parent.variable])
            self.colors = self.coloring()
    
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

            total_sum += left_sum
        return total_sum
    
    def bayesian_dirichilet_score_fast(self, data: pd.DataFrame):
        total_sum = 0
        
        def collect_state_names(variable):
            states = sorted(list(data.loc[:, variable].dropna().unique()))
            return states

        state_names = {
            var: collect_state_names(var) for var in data.columns
        }

        for node in self.nodes:
            variable = node.variable
            parents = [parent.variable for parent in node.prev]
            
            configurations = data.groupby([variable] + parents).size().unstack(parents)

            parents_states = [state_names[parent] for parent in parents]
            
            if len(parents_states) == 0:
                state_count_data = data.loc[:, variable].value_counts()

                state_counts = (
                    state_count_data.reindex(state_names[variable])
                    .fillna(0)
                    .to_frame()
                )
            else:
                row_index = state_names[variable]
                column_index = pd.MultiIndex.from_product(parents_states, names=parents)
                state_counts = configurations.reindex(
                    index=row_index, columns=column_index
                ).fillna(0)

            r_i = len(state_names[variable])
            
            counts = np.asarray(state_counts)
            
            log_counts = loggamma(counts + 1)
            loggamma_sum_per_configuration = np.sum(log_counts, axis=0, dtype=float)
            sum_per_configuration = np.sum(counts, axis=0)

            left_sum = 0
            for (sum, loggamma_sum) in zip(sum_per_configuration, loggamma_sum_per_configuration):
                left_sum += loggamma(r_i) - loggamma(r_i + sum) + loggamma_sum

            total_sum += left_sum
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
            dot.node(name=uid, label="{ %s }"%(n.variable), shape='record')
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)))

        return dot

    def print(self):
        for node in self.nodes:
            print(f"Node: {node.variable}, parents: {[parent.variable for parent in node.prev]}")

    def neighbor_generation(self):
        """
        Creates all the possible neighbor of the current DAG
        """
        forbiden_edges = set()  # Edges that would create cycles
        for node, descendants in self.colors.items():
            for descendant in descendants:
                forbiden_edges.add((descendant, node))

        already_present = set()  # Edges already in the DAG
        for node in self.nodes:
            for parent in node.prev:
                already_present.add((parent, node))

        def added_edge_neigbhor():
            """
            Build the list of all possible neighbor with an addition of 1 egde
            """
            all_possibilities = set()  # All possible edges in the DAG
            for node_i in self.nodes:
                for node_j in self.nodes:
                    if node_j != node_i:
                        all_possibilities.add((node_i, node_j))

            possible_addition = all_possibilities - already_present - forbiden_edges
            result = list()
            for edge in possible_addition:
                new_dag = DAG(self.nodes)
                new_dag.add_edge(edge)
                result.append(new_dag)   # build the new DAG
            return result

        def remove_edge_neighbor():
            """
           Build the list of all possible neighbor with a deletion of 1 egde
           """
            result = list()
            for edge in already_present:
                new_dag = DAG(self.nodes)
                new_dag.remove_edge(edge)
                result.append(new_dag)
            return result

        a = added_edge_neigbhor()
        b = remove_edge_neighbor()
        return a + b

    def coloring(self):
        """
        The colors of a Node represent the nodes that would create a cycle if they were parents of the considered Node
        """
        def color_parent(coloring, current_node):
            for parent in current_node.prev:
                try:
                    coloring[parent] = coloring[parent].union(coloring[current_node])
                except KeyError:
                    coloring[parent] = {parent}.union(coloring[current_node])
                color_parent(coloring, parent)

        result = dict()
        for node in self.topologic_sort()[::-1]:  # we start from the well
            if node not in result:
                result[node] = {node}
                color_parent(result, node)

        return result

    def topologic_sort(self):
        """
        Give a topologic sort of the Nodes
        """
        def explore(adjency_list, current_node, visited, stack):
            """
            Explore a connected part of the graph (with respect of the orientation of the edges)
            """
            visited[current_node] = True
            for neighbor in adjency_list[current_node]:
                if not visited[neighbor]:
                    explore(adjency_list, neighbor, visited, stack)
            stack.insert(0, current_node)

        visited = {node: False for node in self.nodes}
        stack = list()

        adjency_list = {node: set() for node in self.nodes}
        for node in adjency_list:
            for parent in node.prev:
                adjency_list[parent].add(node)

        for node in self.nodes:
            if not visited[node]:
                explore(adjency_list, node, visited, stack)
        return stack

    def add_edge(self, edge):
        old_parent, old_child = edge
        new_child = find_node(self.nodes, old_child.variable)
        target = find_node(self.nodes, old_parent.variable)
        new_child.add_prev(target)
        self.colors = self.coloring()

    def remove_edge(self, edge):
        old_parent, old_child = edge
        new_child = find_node(self.nodes, old_child.variable)
        target = find_node(self.nodes, old_parent.variable)
        new_child.remove_prev(target)
        self.colors = self.coloring()

    def reverse_edge(self, edge):
        self.remove_edge(edge)
        u, v = edge
        self.add_edge((v, u))


# Test zone
def print_edge_set(edge_set):
    for edge in edge_set:
        parent, child = edge
        print(f"{parent.variable} -> {child.variable}", end='\t')
    print()


def find_node(node_list, variable):
    """
    Important, find a node by its variable's name
    """
    for node in node_list:
        if node.variable == variable:
            return node
    return None


class HillClimbing:
    def __init__(self, base_graph : DAG, data_frame, max_iteration=1000):
        self.base_graph = base_graph
        self.max_iteration = max_iteration
        self.data_frame = data_frame

    def solve(self):
        best_graph = self.base_graph
        best_score = best_graph.bayesian_dirichilet_score_fast(self.data_frame)
        for i in range(1, self.max_iteration):
            change = False
            all_neighbours = best_graph.neighbor_generation()
            for neighbor in all_neighbours:
                current_score = neighbor.bayesian_dirichilet_score_fast(self.data_frame)
                if current_score > best_score:
                    best_graph = neighbor
                    best_score = current_score
                    change = True
            if not change:  # No better graph found
                return best_graph, best_score
        return best_graph, best_score

if __name__ == "__main__":
    node_set = {Node(variable=i) for i in range(0, 5)}
    test = DAG(node_set, randomize=True, density=0.4, random_seed=time())
    test.print()
    for node, a in test.colors.items():
        print(f"{node.variable} -> {[b.variable for b in a]}")
    print([node.variable for node in test.topologic_sort()])
    all_neighbor = test.neighbor_generation()
