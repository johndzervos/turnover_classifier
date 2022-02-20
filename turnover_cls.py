"""
pip install
pandas
sklearn
graphviz
pydotplus
ipython
"""
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from six import StringIO  
from pydotplus import graph_from_dot_data
from IPython.display import Image 
import random
import numpy as np

LABELS = ['aa', 'bb', 'cc']
COLUMNS = [
    'f1',
    'f2',
    'f3',
    'f4',
    'f5',
    'f6',
    'Outcome',
]
CLASS_COLUMN = 'Outcome'
CATEGORICAL_COLUMNS = ['f4', 'f5']

CATEGORIES = ['zero', 'one', 'two', 'three']
CATEGORIES_OR_NONE = ['four', 'five', 'six', 'seven', None]
BOOL = [True, False]


def generate_entries(cnt, low=1, high=10, label=None):
  """
  Generate entries for the dataset.
  """
  return [
    [
        random.randint(low, high),  # f1 Int
        random_int_or_None(low, high),  # f2 Int or None
        random.randint(low, high*100)/100,  # f3 Float
        random.choice(CATEGORIES),  # f4 Categorical
        random.choice(CATEGORIES_OR_NONE),  # f5 Categorical or None
        random.choice(BOOL),  # f6 Boolean
        label
    ]
    for _ in range(cnt)
  ]

def random_int_or_None(low, high):
  """
  Return a random int between low and high or None
  """
  number = random.randint(-1, 5)
  if number < 0:
    return None
  else:
    return random.randint(low, high)


def generate_dataset():
  """
  Generate dataset with specific classes
  """
  df1 = pd.DataFrame(generate_entries(100, 1, 10, LABELS[0]), columns=COLUMNS)
  df2 = pd.DataFrame(generate_entries(100, 5, 10, LABELS[1]), columns=COLUMNS)
  df3 = pd.DataFrame(generate_entries(100, 1, 10, LABELS[2]), columns=COLUMNS)
  
  return pd.concat([df1, df2, df3]).fillna(0)

def get_feature_data_class_data_and_columns(data):
  """
  Return the feature data, the class column and the list of the feature columns
  In case of categorical data, a new column is created for each of the provided category
  """
  print(data)
  # Cope with categorical data
  data = pd.get_dummies(data, columns=CATEGORICAL_COLUMNS)
  print(data)
  feature_data = data.loc[:, data.columns!=CLASS_COLUMN]
  return feature_data, data[CLASS_COLUMN], list(feature_data.columns)

def calculate_feature_importance(clf, feature_cols):
  # feat_importance = clf.tree_.compute_feature_importances(normalize=False)
  feat_imp_dict = dict(zip(feature_cols, clf.feature_importances_))
  feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
  feat_imp.rename(columns = {0:'FeatureImportance'}, inplace = True)
  feat_imp = feat_imp.sort_values(by=['FeatureImportance'], ascending=False).head()
  return feat_imp

def visualize_classifier(clf, feature_cols):
  dot_data = StringIO()
  export_graphviz(
      clf,
      out_file=dot_data,  
      filled=True,
      rounded=True,
      special_characters=True,
      feature_names=feature_cols,
      class_names=LABELS
  )
  graph = graph_from_dot_data(dot_data.getvalue())  
  graph.write_png('decision_tree_classifier.png')
  Image(graph.create_png())

data = generate_dataset()
X, y, feature_cols = get_feature_data_class_data_and_columns(data)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

random_entries = pd.DataFrame(generate_entries(5, 1, 10), columns=COLUMNS).fillna(0)
random_data, _, _, = get_feature_data_class_data_and_columns(random_entries)

# Fill empty columns with zeros
for col in feature_cols:
  if col not in random_data.columns:
    random_data[col] = 0
# Reorder columns and remove columns that do not exist in the feature_cols
random_data = random_data[feature_cols]

print(random_data)
random_pred = clf.predict(random_data)
print(random_pred)

# Calculating feature importance
feat_imp = calculate_feature_importance(clf, feature_cols)
print(feat_imp)

# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

visualize_classifier(clf, feature_cols)