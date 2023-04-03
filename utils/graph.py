from typing import Set
from time import time
from random import seed, random


class Node:
    """
    A class to represent each node of the graph
    """
    def __init__(self, prev=[], data=0) -> None:
        self.prev = prev
        self.label = data


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
                self.nodes.add(Node(prev=[], data=node.label))
            colors = {node.label: {node.label} for node in self.nodes}
            for current_node in self.nodes:
                other_nodes = {node for node in self.nodes if node != current_node}
                for candidate_parent in other_nodes:    # Add or not the candidate as a parent of the current node
                    if random() < density:   # Probability test
                        if candidate_parent.label not in colors[current_node.label]:  # No cycle created
                            current_node.prev.append(candidate_parent)
                            colors[candidate_parent.label] = colors[current_node.label].union(
                                colors[candidate_parent.label])

    def print(self):
        for node in self.nodes:
            print(f"Node: {node.label}, parents: {[parent.label for parent in node.prev]}")


# Test zone
if __name__ == "__main__":
    node_set = {Node(data=i) for i in range(0, 8)}
    test = DAG(node_set, randomize=True, density=0.4, random_seed=time())
    test.print()
