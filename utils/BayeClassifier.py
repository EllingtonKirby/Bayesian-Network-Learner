import pandas as pd
from math import ceil


class Bayesian_Classifier:
    def __init__(self, data_frame: pd.DataFrame, target_variable: str, training_split=0.8):
        self.data_frame = data_frame
        self.target_variable = target_variable
        self.training_split = training_split
        self.training_set, self.test_set = self.split()
        self.probability_estimation_table = dict()

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

    def estimate_proba_given_target(self, variable, variable_value, target_value):
        """
        Estimate P[variable = variable_value | y = target_value]
        """
        mask = self.data_frame[self.target_variable] == target_value
        tmp_df = self.training_set.loc[mask]
        target_value_proportion = len(tmp_df) / len(self.training_set)

        mask = (self.data_frame[variable] == variable_value) & (self.data_frame[self.target_variable] == target_value)
        tmp_df = self.training_set.loc[mask]
        variable_value_prob_with_target_value = len(tmp_df) / len(self.training_set)

        return variable_value_prob_with_target_value / target_value_proportion

    def train(self):
        for variable in self.training_set.keys():
            self.probability_estimation_table[variable] = dict()
            if variable == self.target_variable:
                continue
            min_value = self.data_frame[variable].min()
            max_value = self.data_frame[variable].max()
            for variable_value in range(min_value, max_value + 1):
                self.probability_estimation_table[variable][variable_value] = dict()
                for target_value in self.target_domain:
                    self.probability_estimation_table[variable][variable_value][target_value]\
                        = self.estimate_proba_given_target(variable, variable_value, target_value)

    def predict(self, x: pd.Series):
        best_guess = None
        best_score = -1
        for target_value in self.target_domain:

            mask = self.data_frame[self.target_variable] == target_value
            tmp_df = self.training_set.loc[mask]
            target_value_proportion = len(tmp_df) / len(self.training_set)

            product = target_value_proportion
            for variable in x.keys():
                product *= self.probability_estimation_table[variable][x[variable]][target_value]
            if product >= best_score:
                best_score = product
                best_guess = target_value
        return best_guess

    def predict_all(self, x_vector: pd.DataFrame):
        return pd.Series([self.predict(x) for index, x in x_vector.iterrows()], name=self.target_variable)

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
    for _ in range(10):
        titanic_classifier = Bayesian_Classifier(titanic_df, "survived")
        titanic_classifier.train()
        if titanic_classifier.test() > best_titanic_score:
            best_titanic_score = titanic_classifier.test()
    print(f"Best accuracy of {best_titanic_score * 100}% on test set")

    print("WINE")
    best_wine_score = 0
    wine_df = pd.read_csv("../medium.csv")
    for _ in range(10):
        wine_classifier = Bayesian_Classifier(wine_df, "quality")
        wine_classifier.train()
        if wine_classifier.test() > best_wine_score:
            best_wine_score = wine_classifier.test()
    print(f"Best accuracy of {best_wine_score * 100}% on test set")


