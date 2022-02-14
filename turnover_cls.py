"""
pip install
pandas
sklearn
graphviz
pydotplus

"""

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import random

LABELS = ['a', 'b', 'c']
COLUMNS = ['f1', 'f2', 'f3', 'Outcome']

def load_dataset():
  """
  pima.head()
      Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
  0   6            148      72             35             0         33.6 0.627                     50   1
  1   1            85       66             29             0         26.6 0.351                     31   0
  2   8            183      64             0              0         23.3 0.672                     32   1
  3   1            89       66             23             94        28.1 0.167                     21   0
  4   0            137      40             35             168       43.1 2.288                     33   1
  """
  # load dataset
  pima = pd.read_csv("diabetes.csv")
  print(pima.head())
  print(len(pima))

  #split dataset in features and target variable
  feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
  
  # X = pima[feature_cols]  # Features
  # y = pima['Outcome']  # Target variable
  return pima[feature_cols], pima['Outcome'], feature_cols



def generate_entries(cnt, low=1, high=10, label=None):
  """

  """
  return [
    [
        random.randint(low, high),  # f1
        random.randint(low, high),  # f2
        random.randint(low, high),  # f3
        label
    ]
    for _ in range(cnt)
  ]

def generate_dataset():
  """

  """
  feature_cols = COLUMNS[:-1]
  class_col = COLUMNS[-1]
  df1 = pd.DataFrame(generate_entries(5, 1, 10, LABELS[0]), columns=COLUMNS)
  df2 = pd.DataFrame(generate_entries(10, 100, 200, LABELS[1]), columns=COLUMNS)
  df3 = pd.DataFrame(generate_entries(5, 1, 200, LABELS[2]), columns=COLUMNS)
  
  result = pd.concat([df1, df2, df3])
  print(result)
  print(len(result))
  return result[feature_cols], result[class_col], feature_cols


# X, y, feature_cols = load_dataset()
X, y, feature_cols = generate_dataset()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

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

# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


from sklearn.tree import export_graphviz
from six import StringIO  
import pydotplus
from IPython.display import Image  

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
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree_classifier.png')
Image(graph.create_png())