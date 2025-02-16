"""
Random Forest Lab

Name
Section
Date
"""

from platform import uname
import graphviz
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from uuid import uuid4
import random
from time import time


class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        return sample[self.column] >= self.value

    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(float(self.value)))


def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    # Split the data into two arrays based on the question using the match method
    left = data[np.array([question.match(row) for row in data])]
    right = data[np.array([not question.match(row) for row in data])]

    # Checking to see if either side is empty. If so, make sure the array is still 2D
    if len(left) == 0:
        left = left.reshape(-1, data.shape[1])
    if len(right) == 0:
        right = right.reshape(-1, data.shape[1])

    return left, right


animals = np.loadtxt("./Data/animals.csv", delimiter=",")
features = np.loadtxt("./Data/animal_features.csv", delimiter=",", dtype=str, comments=None)
names = np.loadtxt("./Data/animal_names.csv", delimiter=",", dtype=str)
# question = Question(column=1, value=3, feature_names=features)
# question = Question(column=1, value=75, feature_names=features)
# left, right = partition(animals, question)
# print(len(left), len(right))


# Helper function
def num_rows(array):
    """Returns the number of rows in a given array"""
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]


# Helper function
def class_counts(data):
    """Returns a dictionary with the number of samples under each class label
    formatted {label : number_of_samples}"""
    if len(data.shape) == 1:  # If there's only one row
        return {data[-1]: 1}

    # Create a dictionary  with unique values as keys and the counts as values
    unique, counts = np.unique(data[:, -1], return_counts=True)
    return dict(zip(unique, counts))


# Helper function
def info_gain(data, left, right):
    """Return the info gain of a partition of data.
    Parameters:
        data (ndarray): the unsplit data
        left (ndarray): left split of data
        right (ndarray): right split of data
    Returns:
        (float): info gain of the data"""

    def gini(data):
        """Return the Gini impurity of given array of data.
        Parameters:
            data (ndarray): data to examine
        Returns:
            (float): Gini impurity of the data"""
        counts = class_counts(data)
        N = num_rows(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / N
            impurity -= prob_of_lbl**2
        return impurity

    p = num_rows(right) / (num_rows(left) + num_rows(right))
    return gini(data) - p * gini(right) - (1 - p) * gini(left)


def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 6
    Returns:
        (float): Best info gain
        (Question): Best question"""
    best_gain = 0
    best_question = None
    n_features = data.shape[1] - 1
    # If random_subset is True, we will randomly select a sqrt(n_features) number of features
    if random_subset:
        # Randomly selecting a sqrt(n_features) number of features
        n_features = int(np.sqrt(n_features))
        random_features = random.sample(range(data.shape[1] - 1), n_features)
        # Selecting only the random features
        data = data[:, random_features]
        feature_names = feature_names[random_features]
    for col in range(n_features):
        values = np.unique(data[:, col])
        for val in values:
            # Create a question
            question = Question(col, val, feature_names)
            true_rows, false_rows = partition(data, question)
            if (
                num_rows(true_rows) < min_samples_leaf
                or num_rows(false_rows) < min_samples_leaf
            ):
                continue
            gain = info_gain(data, true_rows, false_rows)
            # Update the best gain and question if the current gain is better
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question


# Testing problem 2
# print(find_best_split(animals, features, min_samples_leaf=5))

class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""

    def __init__(self, data):
        self.prediction = class_counts(data)


class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""

    def __init__(self, question, left_branch, right_branch):
        self.question = question
        self.left = left_branch
        self.right = right_branch

def build_tree(
    data,
    feature_names,
    min_samples_leaf=5,
    max_depth=4,
    current_depth=0,
    random_subset=False,
):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    if num_rows(data) < 2 * min_samples_leaf:
        return Leaf(data)

    # Finding the optimal gain and corresponding question
    gain, question = find_best_split(data, feature_names, min_samples_leaf)
    if gain == 0 or current_depth >= max_depth:
        # If the gain is 0 or the max depth has been reached, return a leaf
        return Leaf(data)

    # Partition the data based on the question
    true_rows, false_rows = partition(data, question)
    left_branch = build_tree(
        true_rows, feature_names, min_samples_leaf, max_depth, current_depth + 1
    )
    right_branch = build_tree(
        false_rows, feature_names, min_samples_leaf, max_depth, current_depth + 1
    )
    return Decision_Node(question, left_branch, right_branch)

def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    # Check if the tree is a leaf
    if isinstance(my_tree, Leaf):
        # Return the most common label
        return max(my_tree.prediction, key=my_tree.prediction.get)
    elif my_tree.question.match(sample):
        # If the sample answers the question True, go left
        return predict_tree(sample, my_tree.left)
    else:
        # If the sample answers the question False, go right
        return predict_tree(sample, my_tree.right)


def analyze_tree(dataset, my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    correct = 0
    for row in dataset:
        # Remove the label from the row
        predict_row = row[:-1]
        if predict_tree(predict_row, my_tree) == row[-1]:
            correct += 1
    return correct / len(dataset)

def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    # Create a dictionary to store the counts of each label
    counts = {}
    # For each tree in the forest, predict the label
    for tree in forest:
        label = predict_tree(sample, tree)
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return max(counts, key=counts.get)


def analyze_forest(dataset, forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    correct = 0
    for row in dataset:
        # Remove the label from the row
        predict_row = row[:-1]
        if predict_forest(predict_row, forest) == row[-1]:
            # If the prediction is correct, increment the count
            correct += 1
    return correct / len(dataset)

def prob7():
    """Using the file parkinsons.csv, return three tuples of floats. For tuples 1 and 2,
    randomly select 130 samples; use 100 for training and 30 for testing.
    For tuple 3, use the entire dataset with an 80-20 train-test split.
    Tuple 1:
        a) Your accuracy in a 5-tree forest with min_samples_leaf=15
            and max_depth=4
        b) The time it took to run your 5-tree forest
    Tuple 2:
        a) Scikit-Learn's accuracy in a 5-tree forest with
            min_samples_leaf=15 and max_depth=4
        b) The time it took to run that 5-tree forest
    Tuple 3:
        a) Scikit-Learn's accuracy in a forest with default parameters
        b) The time it took to run that forest with default parameters
    """
    # Load the data
    data = np.loadtxt("parkinsons.csv", delimiter=",", skiprows=1)
    # Loading the feature names
    features = np.loadtxt("parkinsons_features.csv", delimiter=",", dtype=str)

    # Removing the first column of the data
    data = data[:, 1:]

    # Organizing the data
    np.random.shuffle(data)
    train1 = data[:100]
    test1 = data[100:130]
    train2 = data[:100]
    test2 = data[100:130]

    # For the third forest we will use the entire dataset
    # Using 80% for training and 20% for testing
    np.random.shuffle(data)
    train3 = data[: int(0.8 * len(data))]
    test3 = data[int(0.8 * len(data)) :]

    # Tuple 1
    start = time()
    forest1 = [
        build_tree(train1, features, min_samples_leaf=15, max_depth=4) for _ in range(5)
    ]
    time1 = time() - start
    accuracy1 = analyze_forest(test1, forest1)

    # Tuple 2
    start = time()
    forest2 = RandomForestClassifier(n_estimators=5, min_samples_leaf=15, max_depth=4)
    forest2.fit(train2[:, :-1], train2[:, -1])
    time2 = time() - start
    accuracy2 = forest2.score(test2[:, :-1], test2[:, -1])

    # Tuple 3
    start = time()
    forest3 = RandomForestClassifier()
    forest3.fit(train3[:, :-1], train3[:, -1])
    time3 = time() - start
    accuracy3 = forest3.score(test3[:, :-1], test3[:, -1])

    return (accuracy1, time1), (accuracy2, time2), (accuracy3, time3)


# Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    # If it's a leaf, draw an oval and label with the prediction
    if hasattr(my_tree, "prediction"):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else:  # If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id


def draw_tree(my_tree, filename="Digraph"):
    """Draws a tree"""
    # Remove the files if they already exist
    for file in [f"{filename}.gv", f"{filename}.gv.pdf"]:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    # graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf
    in_wsl = False
    in_wsl = "microsoft-standard" in uname().release
    if in_wsl:
        graph.render(f"{filename}.gv", view=False)
        os.system(f"cmd.exe /C start {filename}.gv.pdf")
    else:
        graph.render(view=True)


# Testing problem 4
my_tree = build_tree(animals, features)
draw_tree(my_tree)

# Testing problem 5
# Using np.random.shuffle() to use 80 samples for training and 20 for testing
# np.random.shuffle(animals)
# train = animals[:80]
# test = animals[80:] # 20 samples for testing
# my_tree = build_tree(train, features)
# print(analyze_tree(test, my_tree)) # Should be around 0.8

# Testing problem 6
# forest = [build_tree(train, features) for _ in range(5)]
# print(analyze_forest(test, forest)) # Should be around 0.8

# Displaying the trees to ensure they are different
# for tree in forest:
#     draw_tree(tree)

# Testing problem 7
# print(prob7())
