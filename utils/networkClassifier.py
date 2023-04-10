import pandas as pd
from math import ceil
from graph import Node, DAG, find_node
import itertools


class ProbaTable:
    def __init__(self, parents: set, data_frame: pd.DataFrame, target):
        self.parents = parents
        self.target = target
        self.content = pd.DataFrame(columns=list(parents) + [target, "result"])
        self.data_frame = data_frame

        target_min = self.data_frame[self.target].min()
        target_max = self.data_frame[self.target].max()
        self.target_domain = range(target_min, target_max + 1)

        self.init_table(list(parents), dict())

    def init_table(self, parents, assignment: dict):
        def domain(var):
            var_min = self.data_frame[var].min()
            var_max = self.data_frame[var].max()
            var_domain = range(var_min, var_max + 1)
            return list(var_domain)

        cartesian_product = domain(parents[0])
        for parent in parents[1:]:
            cartesian_product = itertools.product(cartesian_product, domain(parent))

        for elt in cartesian_product:
            for target_value in self.target_domain:
                result = pd.DataFrame(columns=self.content.columns)
                result.at[0, self.target] = target_value
                assignment = dict()
                for index, variable in enumerate(parents):
                    assignment[variable] = elt[index]
                    if variable in {self.target, "result"}:
                        continue
                    result.at[0, variable] = elt[index]
                result.at[0, "result"] = self.estimate(assignment, target_value)
                self.content = pd.concat([self.content, result], ignore_index=True)

        return None

    def give_proba(self, assignment, target_value):
        mask = self.content[self.target] == target_value
        for variable in assignment:
            mask &= self.content[variable] == assignment[variable]
        line = self.content.loc[mask]
        return line['result'].iloc[0]

    def print(self):
        print(self.content)

    def estimate(self, assignment: dict, target_value):
        """
        Estimate P[target | a, b, ...] given a assignment for a, b, ...
        """
        mask = True
        for parent in assignment:
            mask &= (self.data_frame[parent] == assignment[parent])
        tmp_df = self.data_frame.loc[mask]
        assignment_proportion = len(tmp_df) / len(self.data_frame)

        mask = True
        for parent in assignment:
            mask &= (self.data_frame[parent] == assignment[parent]) & (self.data_frame[self.target] == target_value)
        tmp_df = self.data_frame.loc[mask]
        variable_value_prob_with_target_value = len(tmp_df) / len(self.data_frame)

        if assignment_proportion == 0:
            return 0

        return variable_value_prob_with_target_value / assignment_proportion


class Network_Classifier:
    def __init__(self, dag: DAG, data_frame: pd.DataFrame, target_variable: str, training_split=0.8):
        self.data_frame = data_frame
        self.target_variable = target_variable
        self.training_split = training_split
        self.training_set, self.test_set = self.split()
        self.markov_blanket_estimation_table = None
        self.dag = dag
        self.central_node = find_node(self.dag.nodes, self.target_variable)

        target_min = self.data_frame[self.target_variable].min()
        target_max = self.data_frame[self.target_variable].max()
        self.target_domain = range(target_min, target_max + 1)

    def split(self):
        """
        Split the set into training and testing
        """
        self.data_frame = self.data_frame.sample(frac=1)
        training_size = ceil(self.training_split * len(self.data_frame))
        training_set = self.data_frame[:training_size]
        test_set = self.data_frame[training_size:]
        return training_set, test_set

    def train_markov(self):
        self.markov_blanket_estimation_table = ProbaTable(self.find_markov_blanket(),
                                                          self.training_set, self.target_variable)

    def predict(self, x: pd.Series):
        assignment = dict()
        for variable in {parent.variable for parent in self.central_node.prev}:
            assignment[variable] = x[variable]
        best_guess = None
        best_score = -1
        for target_value in self.target_domain:
            score = self.markov_blanket_estimation_table.give_proba(assignment, target_value)
            if score > best_score:
                best_guess = target_value
                best_score = score

        return best_guess

    def predict_all(self, x_vector: pd.DataFrame):
        return pd.Series([self.predict(x) for index, x in x_vector.iterrows()], name=self.target_variable)

    def find_markov_blanket(self):
        """
        Find the Markov blanket of the node we want to predict
        """
        result = set()
        result = result.union({parent.variable for parent in self.central_node.prev})  # add the parents
        children_nodes = {node for node in self.dag.nodes if self.central_node in node.prev}
        for child in children_nodes:
            result = result.add(child.variable)  # add the children
            result = result.union({parent.variable for parent in child.prev})

        return result

    def test(self):
        x_test = self.test_set.iloc[:, :-1]
        y_test = self.test_set.iloc[:, -1]
        predictions = self.predict_all(x_test)
        count, total = 0, 0
        for y, prediction in zip(y_test.tolist(), predictions.tolist()):
            if y == prediction:
                count += 1
            total += 1
        return count / total


if __name__ == "__main__":
    print("TITANIC")
    best_titanic_score = 0
    titanic_df = pd.read_csv("../small.csv")
    node_dict = {variable: Node(variable=variable, prev=[]) for variable in titanic_df.keys()}
    node_dict["survived"].prev += [node_dict["sex"], node_dict["passengerclass"]]
    node_dict["sex"].prev += [node_dict["portembarked"], node_dict["passengerclass"], node_dict["age"],
                              node_dict["numsiblings"], node_dict["numparentschildren"]]
    node_dict["age"].prev += [node_dict["numsiblings"], node_dict["passengerclass"], node_dict["numparentschildren"]]
    node_dict["passengerclass"].prev += [node_dict["fare"], node_dict["numsiblings"], node_dict["portembarked"]]
    node_dict["numparentschildren"].prev += [node_dict["numsiblings"]]
    node_dict["portembarked"].prev += [node_dict["fare"], node_dict["numsiblings"]]
    titanic_dag = DAG({node for _, node in node_dict.items()})
    for _ in range(10):
        titanic_classifier = Network_Classifier(titanic_dag, titanic_df, "survived")
        titanic_classifier.train_markov()
        if titanic_classifier.test() > best_titanic_score:
            best_titanic_score = titanic_classifier.test()
    print(f"Best accuracy of {best_titanic_score * 100}% on test set")

    print("WINE")
    best_wine_score = 0
    wine_df = pd.read_csv("../medium.csv")
    node_dict = {variable: Node(variable=variable, prev=[]) for variable in wine_df.keys()}
    node_dict["freesulfurdioxide"].prev += [node_dict["totalsulfurdioxide"]]
    node_dict["totalsulfurdioxide"].prev += [node_dict["alcohol"], node_dict["density"]]
    node_dict["sulphates"].prev += [node_dict["alcohol"], node_dict["density"]]
    node_dict["ph"].prev += [node_dict["alcohol"], node_dict["fixedacidity"]]
    node_dict["quality"].prev += [node_dict["alcohol"], node_dict["volatileacidity"]]
    node_dict["alcohol"].prev += [node_dict["fixedacidity"], node_dict["chlorides"], node_dict["volatileacidity"],
                                  node_dict["density"]]
    node_dict["fixedacidity"].prev += [node_dict["citricacid"]]
    node_dict["citricacid"].prev += [node_dict["density"], node_dict["chlorides"]]
    node_dict["density"].prev += [node_dict["residualsugar"]]
    node_dict["volatileacidity"].prev += [node_dict["residualsugar"]]
    wine_dag = DAG({node for _, node in node_dict.items()})

    for i in range(100):
        print(i)
        wine_classifier = Network_Classifier(wine_dag, wine_df, "quality")
        wine_classifier.train_markov()
        if wine_classifier.test() > best_wine_score:
            best_wine_score = wine_classifier.test()
    print(f"Best accuracy of {best_wine_score * 100}% on test set")


