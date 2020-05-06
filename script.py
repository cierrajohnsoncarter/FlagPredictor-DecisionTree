import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the data
flags = pd.read_csv('flags.csv', header=0)

# Take a look at the names of the columns in the dataframe
print(flags.columns)

# Take a look at the first couple of columns
print(flags.head())

# Create labels and data 'x and y values' 'train/test data'
labels = flags[['Landmass']]
data = flags[['Red', 'Green', 'White', 'Black', 'Crosses', 'Crescent', 'Zone', 'Religion', 'Language']]

# Split data into training/testing set
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)


# Create an empty list containing all the scores
scores = []

# Create a for loop that tunes the model
for i in range(1, 21):

  # Create and fit the Decision Tree Model to the training data
  tree = DecisionTreeClassifier(random_state=1, max_depth = i)
  tree.fit(train_data, train_labels)

  # Test the model and append each score to the 'scores' list
  scores.append(tree.score(test_data, test_labels))

# Plot the points
plt.plot(range(1, 21), scores)
plt.show()

# Print the model's predictions against the actual results
predictions = tree.predict(test_data)
print(predictions)
print(test_labels)

# Check precision, recall, f1-score and accuracy
print(classification_report(test_labels, predictions))
print(accuracy_score(test_labels, predictions))
