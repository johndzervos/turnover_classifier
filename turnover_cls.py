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
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from six import StringIO  
from pydotplus import graph_from_dot_data
from IPython.display import Image 
import random

LABELS = ['a', 'b', 'c']
COLUMNS = [
    'f1',
    'f2',
    'f3',
    'f4',
    'Outcome'
]

CATEGORIES = ['one', 'two', 'three']


def generate_entries(cnt, low=1, high=10, label=None):
  """

  """
  return [
    [
        random.randint(low, high),  # f1
        random.randint(low, high),  # f2
        random.randint(low, high),  # f3
        random.choice(CATEGORIES),  # f4
        label
    ]
    for _ in range(cnt)
  ]

def generate_dataset():
  """

  """
  feature_cols = COLUMNS[:-1]
  class_col = COLUMNS[-1]
  df1 = pd.DataFrame(generate_entries(500, 1, 10, LABELS[0]), columns=COLUMNS)
  df2 = pd.DataFrame(generate_entries(100, 100, 200, LABELS[1]), columns=COLUMNS)
  df3 = pd.DataFrame(generate_entries(50, 1, 200, LABELS[2]), columns=COLUMNS)
  
  result = pd.concat([df1, df2, df3])
  print(result)
  print(len(result))
  return result[feature_cols], result[class_col], feature_cols

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

X, y, feature_cols = generate_dataset()

# Cope with categorical data
print(X)

X_feature = X[feature_cols]
X_dict = X_feature.T.to_dict().values()

# turn list of dicts into a numpy array
vect = DictVectorizer(sparse=False)
X_vector = vect.fit_transform(X_dict)

# Used to vectorize the class label
le = LabelEncoder()
y = le.fit_transform(y)
print(X)


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# print(X_train)
# print(X_test)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

random_entries = generate_entries(5, 1, 200)
random_pred = clf.predict(X_test[feature_cols])
print(random_entries)
print(random_pred)

# Calculating feature importance
feat_imp = calculate_feature_importance(clf, feature_cols)
print(feat_imp)

# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

visualize_classifier(clf, feature_cols)