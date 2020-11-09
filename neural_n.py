from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def mlp():
	data = pd.read_csv("EEG_Eye_State.csv")

# Split into train data and training labels
	X = data.loc[:,data.columns != 'Eye_detection']
	y = data['Eye_detection']
# Split into train and test data 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 8)
# Train a multi-layer perceptron
	clf0 = MLPClassifier(random_state=0, max_iter = 800)
	clf0.fit(X_train, y_train)
# Predict accuracy of classifier
	y_pred = clf0.predict(X_test)
	acc = accuracy_score(y_pred, y_test)
	print('Accuracy : ', acc*100)


def main():
	mlp()
main()
