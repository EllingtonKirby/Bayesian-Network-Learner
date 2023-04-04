from graph import *


class HillClimbing:
    def __init__(self, base_graph : DAG, data_frame, max_iteration=1000):
        self.base_graph = base_graph
        self.max_iteration = max_iteration
        self.data_frame = data_frame

    def solve(self):
        best_graph = self.base_graph
        best_score = best_graph.bayesian_dirichilet_score(self.data_frame)
        for _ in range(1, self.max_iteration):
            change = False
            all_neighbours = current_score.neighbor_generation()
            for neighbor in all_neighbours:
                current_score = neighbor.bayesian_dirichilet_score(self.data_frame)
                if current_score > best_score:
                    best_graph = neighbor
                    best_score = current_score
                    change = True
            if not change:  # No better graph found
                return best_graph, best_score
        return best_graph, best_score
