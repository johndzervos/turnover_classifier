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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from six import StringIO
from pydotplus import graph_from_dot_data
from IPython.display import Image 
import random
import numpy as np

LABELS = ['Class1', 'Class2', 'Class3']
COLUMNS = [
    'ID',
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
        f"id_{i}", # Unique id
        random.randint(low, high),  # f1 Int
        random_int_or_None(low, high),  # f2 Int or None
        random.randint(low, high*100)/100,  # f3 Float
        random.choice(CATEGORIES),  # f4 Categorical
        random.choice(CATEGORIES_OR_NONE),  # f5 Categorical or None
        random.choice(BOOL),  # f6 Boolean
        label
    ]
    for i in range(cnt)
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
  # df0 = pd.DataFrame(generate_entries(200, 1, 10, None), columns=COLUMNS)
  df1 = pd.DataFrame(generate_entries(400, 1, 5, LABELS[0]), columns=COLUMNS)
  df2 = pd.DataFrame(generate_entries(200, 5, 10, LABELS[1]), columns=COLUMNS)
  df3 = pd.DataFrame(generate_entries(300, 1, 10, LABELS[2]), columns=COLUMNS)
  
  # return pd.concat([df0, df1])
  return pd.concat([df1, df2, df3])

def clean_data(data):
  """
  Replace None values and vectorize categorical columns
  """
  return pd.get_dummies(data.fillna(0), columns=CATEGORICAL_COLUMNS)

def get_feature_data_class_data_and_columns(data):
  """
  Return the feature data, the class column and the list of the feature columns
  In case of categorical data, a new column is created for each of the provided category
  """
  feature_data = data.loc[:, ~data.columns.isin([CLASS_COLUMN, 'ID'])]
  return feature_data, data[CLASS_COLUMN], list(feature_data.columns)

def calculate_feature_importance(clf, feature_cols):
  # feat_importance = clf.tree_.compute_feature_importances(normalize=False)
  feat_imp_dict = dict(zip(feature_cols, clf.feature_importances_))
  feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
  feat_imp.rename(columns = {0:'FeatureImportance'}, inplace = True)
  feat_imp = feat_imp.sort_values(by=['FeatureImportance'], ascending=False)
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

# Clean data
data = clean_data(data)

X, y, feature_cols = get_feature_data_class_data_and_columns(data)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
dt_clf = DecisionTreeClassifier()
# Create Random Forest Classifier object
rf_clf = RandomForestClassifier(n_estimators=100)

# Train Decision Tree Classifer
dt_clf = dt_clf.fit(X_train, y_train)
# Train Random Forest Classifer
rf_clf = rf_clf.fit(X_train, y_train)

# Predict the response for test dataset (Decision Tree)
dt_y_pred = dt_clf.predict(X_test)
# Predict the response for test dataset (Random Forest)
rf_y_pred = rf_clf.predict(X_test)

random_entries = pd.DataFrame(generate_entries(5, 1, 10), columns=COLUMNS)
random_data = clean_data(random_entries)
random_data, _, _, = get_feature_data_class_data_and_columns(random_data)

# Fill empty columns with zeros
for col in feature_cols:
  if col not in random_data.columns:
    random_data[col] = 0
# Reorder columns and remove columns that do not exist in the feature_cols
random_data = random_data[feature_cols]

print(random_data)

dt_random_pred = dt_clf.predict(random_data)
dt_prob = dt_clf.predict_proba(random_data)

rf_random_pred = rf_clf.predict(random_data)
rf_prob = rf_clf.predict_proba(random_data)

print("Decision tree predictions: ", dt_random_pred)
print("Random forest predictions: ", rf_random_pred)
print(dt_prob)
print(rf_prob)

# Calculating feature importance
dt_feat_imp = calculate_feature_importance(dt_clf, feature_cols)
print(dt_feat_imp)
rf_feat_imp = calculate_feature_importance(rf_clf, feature_cols)
print(rf_feat_imp)

# Model Accuracy, how often is the classifier correct?
print("Decision Tree Accuracy: ", metrics.accuracy_score(y_test, dt_y_pred))
print("Random Forest Accuracy: ", metrics.accuracy_score(y_test, rf_y_pred))

# Visualize Decision Tree classifier
visualize_classifier(dt_clf, feature_cols)
