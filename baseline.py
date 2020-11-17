import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from scipy.stats import entropy

def main():
	random_state = 8
	data = pd.read_csv('EEG_Eye_State.csv')
	# print(data.shape)
	# print(data)
	X = data.loc[:,data.columns !='Eye_detection']
	y = data['Eye_detection']
	# print(y)
	normalized_X = preprocessing.normalize(X, axis=0)
	# print(normalized_X)
	# print(normalized_X.shape)
	np.linalg.norm(normalized_X[:,0])
	print("Calculating...")
	# Split into train and test data 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
	# Train a multi-layer perceptron
	clf0 = MLPClassifier(random_state=random_state)
	clf0.fit(X_train, y_train)
	# Predict accuracy of classifier
	y_pred = clf0.predict(X_test)
	acc = accuracy_score(y_pred, y_test)
	print('Accuracy : ', acc*100)


	# # Split into train and test data 
	# X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size = 0.2, random_state = random_state)
	# # Train a multi-layer perceptron
	# clf0 = MLPClassifier(random_state=random_state, verbose=True)
	# clf0.fit(X_train, y_train)
	# # Predict accuracy of classifier
	# y_pred = clf0.predict(X_test)
	# acc = accuracy_score(y_pred, y_test)
	# print('Accuracy normalized : ', acc*100)


	#Feature scaling
	scaled_X = preprocessing.scale(X)
	np.mean(scaled_X[:,0])

	# Split into train and test data 
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.2, random_state = random_state)
	y_train = y_train.to_numpy()
	y_test = y_test.to_numpy()


	# Least confidence split into training and pool data
	X_pool,X_train,y_pool,y_train = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)
	
	clf0 = MLPClassifier(hidden_layer_sizes=(100,100),random_state=random_state, verbose=False, max_iter=1000)
	clf0.fit(X_train, y_train)
	# Predict accuracy of classifier
	y_pred = clf0.predict(X_test)
	acc = accuracy_score(y_pred, y_test)
	print('Accuracy on scaled data: ', acc*100)
	
	#include iterations

	for i in range(25):
		# print("LC ",i+1)
		clf0 = MLPClassifier(hidden_layer_sizes=(100,100),random_state=random_state, verbose=False, max_iter=1000)
		clf0.fit(X_train, y_train)
		temp = clf0.predict_proba(X_pool)

		probab = []
		for item in temp:
			probab.append(max(item))

		prob = 	np.asarray(probab) 
		minimum = min(prob)
		ind = np.where(prob==minimum) #index to add to X_train and drop in X_pool

		X_train = np.append(X_train,X_pool[ind],axis=0)
		y_train = np.append(y_train,y_pool[ind],axis=0)
		
		X_pool = np.delete(X_pool,(ind),axis=0)
		y_pool = np.delete(y_pool,(ind),axis=0)

	y_pred = clf0.predict(X_test)
	acc = accuracy_score(y_pred, y_test)
	print('Least confidence accuracy : ', acc*100)

	#Entropy
	for i in range(25):
		# print("Entropy ",i+1)
		prob = clf0.predict_proba(X_pool)
		temp_ent = entropy(prob.T)
		#Index of max entropy element
		max_ele = np.where(temp_ent == np.amax(temp_ent))

		X_train = np.append(X_train,X_pool[max_ele],axis=0)
		y_train = np.append(y_train,y_pool[max_ele],axis=0)
		
		X_pool = np.delete(X_pool,(max_ele),axis=0)
		y_pool = np.delete(y_pool,(max_ele),axis=0)

		temp_ent = np.delete(temp_ent,(max_ele),axis=0)

	y_pred = clf0.predict(X_test)
	acc = accuracy_score(y_pred, y_test)
	print('Entropy accuracy : ', acc*100)
	


main()

