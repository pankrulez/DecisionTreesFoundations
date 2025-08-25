

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as pp
import math
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('breast_cancer_data.csv')
df.head()

df.info()

df.describe()

# check null values
df.isnull().sum()

# check data types
df.dtypes

df.columns

print("Unique values in 'diagnosis' column:")
df['diagnosis'].unique()

px.pie(
    df, 'diagnosis',
    color = 'diagnosis',
    title = 'Distribution of Diagnosis',
    color_discrete_map={'M': 'red', 'B': 'blue'},
    width=500, height=500
    )


# Inferences :

# Data is clean, no null values
# Data types are appropriate for analysis
# 'diagnosis' column has two unique values: 'M' (Malignant) and 'B' (Benign)
# The dataset contains 569 samples with 30 features
# Dataset is imbalanced (M : B = 63:37)
# There are more cases of benign tumors than malignant tumors
# For imbalanced datasets, accuracy can be a misleading metric
#   for example, if 90% of the cases are benign, the model will always predict "benign"
#   in such cases, we need "Balanced accuracy"

# %pip install nbformat>=4.2.0

# visually compare the distribution of each feature

# for malignant tumours versus benign .

# for a given feature, do its values tend to be different for malignant vs benign cases

for column in df.drop("diagnosis",axis=1).columns[:5]:
    # for loop auto iterates through the first five feature columns in the dataframe
    fig = px.box(data_frame =df ,
                 x='diagnosis',
                 color = 'diagnosis',
                 y = column,
                 color_discrete_sequence = ['#007500','#5CFF5C'],
                 orientation = 'v')

    fig.show()

for column in df.drop("diagnosis",axis=1).columns[5:10]:
    # for loop auto iterates through the first five feature columns in the dataframe
    fig = px.scatter(data_frame =df ,
                 x=column,
                 color = 'diagnosis',
                 color_discrete_sequence = ['#007500','#5CFF5C'],
                 orientation = 'v')

    fig.show()

# diagnosis : M or B :categorical

# encode : 1 or 0 :categorical

df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)

# this line converts the categorical feature into numerical

# setting M = 1  then B = 0

# take the correlation
corr = df.corr()

plt.figure(figsize = (20,20))

# heatmap
sns.heatmap(corr , cmap='viridis_r' , annot=True)

plt.show()

# correlation : -1 to 1

"""Feature Selection"""

# We should now choose which features are good enough predictors to be used to train the model
# get the absoulte correlation

import pprint

cor_target = abs(corr['diagnosis'])

# select better correlated features
# this is the filtering step
# it creates a new list of relevant features
relevant_features = cor_target[cor_target>0.25]

# 0.25 is user defined. It is the hyper-parameter value
# collect the names of features
# list comprehension
names = [index for index,value in relevant_features.items()]

# Drop the target vairable from the results
names.remove("diagnosis")

pprint.pprint(names)

"""Assigning Training data and labels

"""

X = df[names].values
y = df['diagnosis'].values.reshape(-1, 1)

print("Features ",X.shape, "Labels ",y.shape)

# we need to scale
# Standardize / Z-score normalization
# apply on X

import numpy as np

def scale(X):
    '''
    Parameters : numpy.ndarray)
    Returns : numpy.ndarray
    '''
    # Calculate the mean and standard deviation of each feature
    # along the columns (axis=0)
    # mean and std are numpy arrays of shape (n_features,)
    mean = np.mean(X, axis=0)

    std = np.std(X, axis=0)

    # Standardize this data
    X = (X - mean) / std

    # This will scale the data to have mean 0 and standard deviation 1
    # This is useful for many machine learning algorithms
    # that assume the data is centered around 0 and has unit variance
    return X

X = scale(X)
# X is now standardized
display(X)

"""### Model Implementation

Steps
* We will start with all the examples at the Root Nodes
* Then we will calculate the Information Gain for each feature / Gini Index for each feature

* then we will pick the feature with the highest Information Gain / Gini Index

* then we will split the data according to selected feature

* we will repeat this process until we reach the stopping criteria
"""

# Node Class

class Node:
    '''
    A class representing a Node in a Decision Tree.
    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None,gain = None, value=None):
        '''
        Initializes a Node.

        Parameters:
        - feature: The index of the feature to split on.
        - threshold: The threshold value for the split. Defaults to None
        - left: The left child Node. Defaults to None
        - right: The right child Node. Defaults to None
        - value: The class label if it's a leaf node.

        '''
        # Initialize the Node with the given parameters
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

'''
Explanation :

self.threshold = threshold
self.feature = feature

The above two are used by Decision Nodes.
They store the question being asked at this node .
For example , "Is the radius_mean < 15.5 ? "


self.left = left and self.right = right

Used by decision nodes to point to the left and right child nodes.
They are also called pointer nodes.


self.value = value

used  by leaf nodes to store the class label.
If a node is a final endpoint . it does not ask any questions
it holds predicted class label or prediciton for each branch

self.value will be 0(Benign)   or 1(Malignant) for leaf nodes.


self.gain = gain

Used by Decision Nodes to store the Information Gain or Gini Index of the split.

'''

class DesicionTree:
    '''
    This is a decision tree classifier.
    It builds a decision tree from the training data.
    It can predict the class label for new data points.
    '''

    def __init__(self, min_samples = 2, max_depth = 3):
        self.min_samples = min_samples
        self.max_depth = max_depth


    def split(self, dataset, feature, threshold):
        '''
        Splits the dataset into two subsets based on the feature and threshold.

        Parameters:
        - dataset: The dataset to split
        - feature: The index of the feature to split on
        - threshold: The threshold value for the split

        Returns:
        - left: The left subset of the dataset with values less than the threshold
        - right: The right subset of the dataset with values greater than or equal to the threshold
        '''

        left_dataset = []
        right_dataset = []
        # Iterate through each row in the dataset
        for row in dataset:
            # If the value of the feature is less than the threshold, add it to the left subset
            if row[feature] < threshold:
                left_dataset.append(row)
            # Otherwise, add it to the right subset
            else:
                right_dataset.append(row)

        # Convert the left dataset and right dataset to numpy arrays
        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)

        return left_dataset, right_dataset


# write function to calculate Entropy
    def entropy(self, y):
        '''
        Calculates the entropy of the class labels.
        Entropy is a measure of the uncertainty in the class labels.
        Entropy suggests impurity or disorder in the dataset.

        Parameters:
        - y: The class labels of the dataset

        Returns:
        - entropy: The entropy of the class labels
        '''

        entropy = 0.0

        # use numpy's unique function to get the unique class labels and their counts
        labels = np.unique(y)

        for label in labels:

            # find examples in y that match the current label
            # this will give us the examples that belong to the current label
            # we will use this to calculate the probability of the current label
            label_examples = y[y == label]

            # calculate the ratio of current labe in y
            pl = len(label_examples) / len(y)

            # calculate the entropy for the current label and ratio
            entropy += -pl * np.log2(pl) if pl > 0 else 0
            # (pl > 0) this is to avoid log(0) which is undefined

            return entropy

# write function to calculate Gini Index / Information Gain
    def gini_index(self, parent, left, right):
        '''
        Calculates the Gini Index of the class labels.
        Gini Index is a measure of impurity in the class labels.
        Computes the information gain from splitting the parent dataset into two subsets.

        Parameters:
        parent(ndarray): Input parent dataset
        left : left subset of the dataset
        right : right subset of the dataset

        Returns:
        - gini: The Gini Index of the class labels
        '''

        # Initialize the information gain to 0
        information_gain = 0.0

        # compute the entropy of the parent dataset
        parent_entropy = self.entropy(y)

        # weights for the left and right subsets
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)

        # calculate the entropy of the left and right datasets/nodes
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)

        # calculate weighted entropy
        # weighted entropy = post split entropy
        # weighted entropy is the sum of the weighted entropies of the left and right subsets
        # where the weights are the proportion of examples in each subset
        weighted_entropy = (weight_left * entropy_left) + (weight_right * entropy_right)

        # calculate the information gain
        # information gain = pre split entropy - post split entropy
        information_gain = parent_entropy - weighted_entropy

        return information_gain

    # write function to get the best split
def best_split(self, dataset, num_samples, num_features):
    '''
    Finds the best feature and threshold to split the dataset.

    Parameters:
    - dataset: The dataset to split
    - num_samples: The number of samples in the dataset
    - num_features: The number of features in the dataset

    Returns:
    dict: A dictionary containing the best feature index, threshold value, and information gain
    '''
    best_split = {'gain': -1, 'feature': None, 'threshold': None}

    # loop over all features
    for feature_index in range(num_features):
        # get all unique values of the feature
        thresholds = np.unique(dataset[:, feature_index])

        for threshold in thresholds:
            # split the dataset based on the current feature and threshold
            left_dataset, right_dataset = self.split_data(dataset, feature_index, threshold)

            # if either subset is empty, skip this split
            if len(left_dataset) == 0 and len(right_dataset) == 0:
            # get y values of the parent and right, left nodes
                y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                # compute information gain based on the y values
                information_gain = self.information_gain(y, left_y, right_y)
                # update the best split if conditions are met

            # if this gain is better than the best gain found so far, update the best gain and corresponding feature and threshold
            if information_gain > best_split['gain']:
                best_split['gain'] = information_gain
                best_split['feature'] = feature_index
                best_split['threshold'] = threshold
                best_split['left'] = left_dataset
                best_split['right'] = right_dataset
    return best_split

def calculate_leaf_value(self, y):
        """
        Calculates the most occurring value in the given list of y values.

        Args:
            y (list): The list of y values.

        Returns:
            The most occurring value in the list.
        """
        y = list(y)
        #get the highest present class in the array
        most_occuring_value = max(y, key = y.count)
        return most_occuring_value
def build_tree(self, dataset, current_depth=0):
        """
        Recursively builds a decision tree from the given dataset.

        Args:
        dataset (ndarray): The dataset to build the tree from.
        current_depth (int): The current depth of the tree.

        Returns:
        Node: The root node of the built decision tree.
        """
        # split the dataset into X, y values
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
        # keeps spliting until stopping conditions are met
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            # Get the best split
            best_split = self.best_split(dataset, n_samples, n_features)
            # Check if gain isn't zero
            if best_split["gain"]:
                # continue splitting the left and the right child. Increment current depth
                left_node = self.build_tree(best_split["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth + 1)
                # return decision node
                return Node(best_split["feature"], best_split["threshold"],
                            left_node, right_node, best_split["gain"])

        # compute leaf node value
        leaf_value = self.calculate_leaf_value(y)
        # return leaf node value
        return Node(value=leaf_value)
def fit(self, X, y):
        """
        Builds and fits the decision tree to the given X and y values.

        Args:
        X (ndarray): The feature matrix.
        y (ndarray): The target values.
        """
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)

def predict(self, X):
        """
        Predicts the class labels for each instance in the feature matrix X.

        Args:
        X (ndarray): The feature matrix to make predictions for.

        Returns:
        list: A list of predicted class labels.
        """
        # Create an empty list to store the predictions
        predictions = []
        # For each instance in X, make a prediction by traversing the tree
        for x in X:
            prediction = self.make_prediction(x, self.root)
            # Append the prediction to the list of predictions
            predictions.append(prediction)
        # Convert the list to a numpy array and return it
        np.array(predictions)
        return predictions
def make_prediction(self, x, node):
        """
        Traverses the decision tree to predict the target value for the given feature vector.

        Args:
        x (ndarray): The feature vector to predict the target value for.
        node (Node): The current node being evaluated.

        Returns:
        The predicted target value for the given feature vector.
        """
        # if the node has value i.e it's a leaf node extract it's value
        if node.value != None:
            return node.value
        else:
            #if it's node a leaf node we'll get it's feature and traverse through the tree accordingly
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)

# Evaluation

# X_train,y_train, X_test,y_test = train_test_split(X, y, random_state=41, test_size=0.2)

def train_test_split(X, y, random_state=41, test_size=0.2):
    """
    Splits the data into training and testing sets.

    Parameters:
        X (numpy.ndarray): Features array of shape (n_samples, n_features).
        y (numpy.ndarray): Target array of shape (n_samples,).
        random_state (int): Seed for the random number generator. Default is 42.
        test_size (float): Proportion of samples to include in the test set. Default is 0.2.

    Returns:
        Tuple[numpy.ndarray]: A tuple containing X_train, X_test, y_train, y_test.
    """
    # Get number of samples
    n_samples = X.shape[0] # rows are samples

    # Set the seed for the random number generator
    np.random.seed(random_state)

    # Shuffle the indices
    shuffled_indices = np.random.permutation(np.arange(n_samples))

    # Determine the size of the test set
    test_size = int(n_samples * test_size)

    # Split the indices into test and train
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    # Split the features and target arrays into test and train
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def accuracy(self, y_true, y_pred):
        """
        Calculates the accuracy of the predictions.

        Args:
            y_true (list): The true class labels.
            y_pred (list): The predicted class labels.

        Returns:
            float: The accuracy of the predictions as a percentage.
        """
        y_true = y_true.flatten()
        total_samples = len(y_true)

        # Calculate the number of correct predictions
        correct_predictions = np.sum(y_true == y_pred)
        return (correct_predictions / total_samples)

def balanced_accuracy(y_true, y_pred):
    """Calculate the balanced accuracy for a multi-class classification problem.

    Parameters
    ----------
        y_true (numpy array): A numpy array of true labels for each data point.
        y_pred (numpy array): A numpy array of predicted labels for each data point.

    Returns
    -------
        balanced_acc : The balanced accuracyof the model

    """
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    # Get the number of classes
    n_classes = len(np.unique(y_true))

    # Initialize an array to store the sensitivity and specificity for each class
    sen = []
    spec = []
    # Loop over each class
    for i in range(n_classes):
        # Create a mask for the true and predicted values for class i
        mask_true = y_true == i
        mask_pred = y_pred == i

        # Calculate the true positive, true negative, false positive, and false negative values
        TP = np.sum(mask_true & mask_pred)
        TN = np.sum((mask_true != True) & (mask_pred != True))
        FP = np.sum((mask_true != True) & mask_pred)
        FN = np.sum(mask_true & (mask_pred != True))

        # Calculate the sensitivity (true positive rate) and specificity (true negative rate)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        # Store the sensitivity and specificity for class i
        sen.append(sensitivity)
        spec.append(specificity)
    # Calculate the balanced accuracy as the average of the sensitivity and specificity for each class
    average_sen =  np.mean(sen)
    average_spec =  np.mean(spec)
    balanced_acc = (average_sen + average_spec) / n_classes

    return balanced_acc

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

# sklearn implementation

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a decision tree classifier model object.
decision_tree_classifier = DecisionTreeClassifier()

# Train the decision tree classifier model using the training data.
decision_tree_classifier.fit(X_train, y_train)

# Use the trained model to make predictions on the test data.
predictions = decision_tree_classifier.predict(X_test)

# Calculate evaluating metrics
print(f" Model's Accuracy: {accuracy_score(y_test, predictions)}")
print(f"Model's Balanced Accuracy: {balanced_accuracy(y_test, predictions)}")

"""### Classification Report"""

from sklearn.metrics import classification_report

# Generate and print the classification report for the predictions
print(classification_report(y_test, predictions, target_names=['Benign', 'Malignant']))

"""### Confusion Matrix"""

from sklearn.metrics import confusion_matrix

# Generate the confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

# Extract values from confusion matrix
TN, FP, FN, TP = cm.ravel()

print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# Plot bar chart for TP, TN, FP, FN
labels = ['TP', 'TN', 'FP', 'FN']
values = [TP, TN, FP, FN]

plt.figure(figsize=(6,4))
ax= sns.barplot(x = labels, y = values, palette = 'pastel')
plt.title('Confusion Matrix Counts')
plt.ylabel('Count')
plt.xlabel('Category')
for i, v in enumerate(values):
    ax.text(i, v + 0.5, str(v), ha='center')
plt.show()

"""### Precision & Recall"""

# Precision: Out of all the samples predicted as positive (Malignant, label=1), how many are actually positive?
# Recall (Sensitivity): Out of all actual positive samples, how many did the model correctly identify?

# For binary classification:
# Precision = TP / (TP + FP)
# Recall    = TP / (TP + FN)

# Extract values from confusion matrix
TN, FP, FN, TP = cm.ravel()

precision = TP / (TP + FP)
recall = TP / (TP + FN)


print(f"Precision: {precision:.2f} (Out of all predicted Malignant, how many are truly Malignant)")
print(f"Recall: {recall:.2f} (Out of all actual Malignant, how many did we correctly identify)")

# Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# 1. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# 2. Train Decision Tree
# ===============================
clf = DecisionTreeClassifier(
    max_depth=5,           # limit depth to avoid overfitting
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# ===============================
# 3. Predictions
# ===============================
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # for ROC AUC

# ===============================
# 4. Evaluation Metrics
# ===============================
print("Model Evaluation Metrics")
print(f"Accuracy:            {accuracy_score(y_test, y_pred):.4f}")
print(f"Balanced Accuracy:   {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:           {precision_score(y_test, y_pred):.4f}")
print(f"Recall (Sensitivity):{recall_score(y_test, y_pred):.4f}")
print(f"F1-score:            {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:             {roc_auc_score(y_test, y_proba):.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ===============================
# 5. Confusion Matrix Visualization
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Benign','Pred Malignant'],
            yticklabels=['Actual Benign','Actual Malignant'])
plt.title("Confusion Matrix")
plt.show()

# ===============================
# 6. Visualize Decision Tree
# ===============================
plt.figure(figsize=(30,12))
plot_tree(
    clf,
    filled=True,
    feature_names=names,
    class_names=['Benign','Malignant'],
    fontsize=8
    )
plt.title("Decision Tree (max_depth=5)")
plt.show()

# ===============================
# 7. Feature Importances
# ===============================
importances = pd.Series(clf.feature_importances_, index=names)
importances = importances.sort_values(ascending=False)[:10]  # top 10

plt.figure(figsize=(7,4))
sns.barplot(x=importances, y=importances.index, palette="viridis")
plt.title("Top 10 Important Features")
plt.show()

"""### Code summary for best_split:
* Purpose: Finds the best feature and threshold to split the dataset so that the information gain is maximized.
* Steps:
    * Initialize best_split with placeholders (gain = -1, no feature, no threshold).
    * Loop through each feature (column in dataset).
    * For each feature:
        * Get all unique values → possible thresholds.
        * For each threshold:
            * Split dataset into left_dataset and right_dataset using self.split_data.
            * Skip invalid splits (where both sides are empty).
            * Extract target values y for parent, left, and right.
            * Compute information gain using self.information_gain.
            * If this gain is better than the best so far:
                * Update best_split with current feature, threshold, gain, and subsets.
            * Return the dictionary containing the best split.


### Code Summary for calculate_leaf_value:
* Purpose: Determines the value to assign to a leaf node (final prediction) in the decision tree.
* Steps:
    * Convert y (array of target values) into a Python list.
    * Find the most frequently occurring value in y
    * Return that value as the leaf node’s prediction.


### Code Summary for build_tree:

* The function recursively builds a decision tree from a dataset.
* At each step:
    1. Split the dataset into features (X) and target (y).
    2. Check stopping conditions (enough samples & depth limit not exceeded).
    3. If conditions allow:
        * Find the best feature and threshold to split the data.
        * Recursively build left and right child nodes.
        * Return a decision node with split info.
    4. If no further splitting is possible:
        * Compute a leaf value (like majority class/mean).
        * Return a leaf node.


### Code Summary for fit:
* Purpose: Fits (trains) the decision tree using the input features (X) and target values (y).
* Steps:
    * Combines X (features) and y (target) into a single dataset using np.concatenate.
        * X: shape (n_samples, n_features)
        * y: shape (n_samples, 1) (must be 2D to concatenate).
    * Calls self.build_tree(dataset) to construct the decision tree recursively.
    * Stores the root node of the built tree in self.root.


### Code summary for predict:
* Purpose: Predicts class labels for new input data using the trained decision tree.
* Steps:
    * Initialize an empty list predictions.
    * For each sample x in X:
        * Call self.make_prediction(x, self.root) to traverse the tree and get a prediction
        * Append the prediction to the list.
    * Convert predictions to a NumPy array (though the current code does not store it).
    * Return the list of predictions.


### Code summary for make_prediction:
* Purpose: Traverses the decision tree to predict the output for a single data point (x).
* Steps:
    * Check if current node is a leaf:
        * If node.value is not None, return it as the prediction.
    * If not a leaf (decision node):
        * Look at the feature index stored in the node (node.feature).
        * Compare the feature value (x[node.feature]) with the threshold (node.threshold).
        * If the value is less than or equal to the threshold → recurse into the left child
        * Otherwise → recurse into the right child.
"""