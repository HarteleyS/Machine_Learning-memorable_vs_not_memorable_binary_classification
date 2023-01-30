from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
print("importing....")
# import the neccessary libraries

print("reading data......")
# Load data and store them in the Pandas objects
training_df = pd.read_csv('training.csv')
# Create annotation confidence for training.csv files
training_df['sample_weight'] = pd.Series(
    [1.0 for i in range(training_df.shape[0])])

# Load addtional Data
additional_df = pd.read_csv('additional_training.csv')
annoted_df = pd.read_csv('annotation_confidence.csv')
additional_df['sample_weight'] = annoted_df['confidence']

# Load Test Data
test_df = pd.read_csv('testing.csv')
test_ids = test_df['ID'].values
X_test = test_df.iloc[:, 1:].values
# Fill missing values with medians of the column
df = pd.concat([training_df, additional_df])

for column in df.columns:
    median = df[column].median()
    df[column] = df[column].fillna(median)


print("PCa analysis...")
# Do PCA Analysis
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.iloc[:, 1:-2].values)
sns.scatterplot(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    hue=df.iloc[:, -2].values
)
plt.show()

# Scale the data
sc = StandardScaler()
X_test = sc.fit_transform(X_test)

ones = df[df['prediction'] == 1]
zeros = df[df['prediction'] == 0]

N = 25
models = []
total_accuracy_sum = 0
predictions = np.array([0]*len(X_test))
for i in range(51):
    print("{} model processing...".format(i))
    new_sample_df = pd.concat([zeros.sample(300), ones.sample(1500)])
    sample_X = new_sample_df.iloc[:, 1:-2].values
    sample_y = new_sample_df.iloc[:, -2:].values
    sample_X = sc.fit_transform(sample_X)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        sample_X, sample_y, test_size=0.2)
    sample_weights = Ytrain[:, -1]
    Ytrain = Ytrain[:, -2].astype('int32')
    classifier = GaussianNB(priors=[0.6, 0.4])
    classifier.fit(Xtrain, Ytrain, sample_weight=sample_weights)
    predictions += classifier.predict(X_test)
    cur_prediction_validation_set = classifier.predict(Xtest)
    cur_accuracy = (accuracy_score(
        Ytest[:, -2], cur_prediction_validation_set))
    print("Model accuracy = {}".format(cur_accuracy))
    total_accuracy_sum += cur_accuracy
    cur_prediction = classifier.predict(X_test)

print("Final Average Accuracy on validation set = {}".format(
    (total_accuracy_sum * 100) / (2 * N)))
sorted_predictions = np.argsort(predictions)
counter = 4569
final_predictions = np.zeros(len(X_test))
for i in range(1, counter):
    final_predictions[sorted_predictions[-i]] = 1

X_test
output = pd.DataFrame({
    'ID': test_ids,
    'prediction': final_predictions.astype('int32')
})

output.to_csv('my_submission.csv', index=False)
