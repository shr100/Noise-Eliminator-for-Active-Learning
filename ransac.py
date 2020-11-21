import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import random
import pandas as pd
import csv
from sklearn.metrics import f1_score

# from modAL.models import ActiveLearner
# from modAL.uncertainty import uncertainty_sampling,entropy_sampling


def main():
	random_state = 8
	# data = pd.read_csv('EEG_small.csv')
	data = pd.read_csv('EEG_Eye_State.csv')
	
	X = data.loc[:,data.columns !='Eye_detection']
	y = data['Eye_detection']
	
	normalized_X = preprocessing.normalize(X, axis=0)
	
	np.linalg.norm(normalized_X[:,0])
	print("Calculating...")
	# Split into train and test data 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)


	#Feature scaling
	scaled_X = preprocessing.scale(X)
	np.mean(scaled_X[:,0])

	# Split into train and test data 
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.2, random_state = random_state)
	X_train, X_valid, y_train,y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)
	y_train = y_train.to_numpy()
	y_test = y_test.to_numpy()
	y_valid = y_valid.to_numpy()


	# Least confidence split into training and pool data
	X_pool,X_train,y_pool,y_train = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)
	
	clf0 = MLPClassifier(hidden_layer_sizes=(100,100),random_state=random_state, verbose=False, max_iter=1000)
	clf0.fit(X_train, y_train)
	# Predict accuracy of classifier
	y_pred = clf0.predict(X_test)
	acc = accuracy_score(y_pred, y_test)
	print('Accuracy on scaled data: ', acc*100)
	print('F1 score on scaled data : ', f1_score(y_test,y_pred)*100)

	original_X_train = X_train
	original_X_pool = X_pool
	original_y_train = y_train
	original_y_pool = y_pool

	print("Number of training data instances before pooling : ",X_train.shape[0])

	for i in range(20): #Change value of iterations according to AL methods
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

	print('Baseline least confidence accuracy : ', acc*100)
	print('F1 score on least confidence AL : ', f1_score(y_test,y_pred)*100)

	X_train = original_X_train
	X_pool = original_X_pool
	y_train = original_y_train
	y_pool = original_y_pool

	for i in range(20): #Change value of iterations according to AL methods
		clf1 = MLPClassifier(hidden_layer_sizes=(100,100),random_state=random_state, verbose=False, max_iter=1000)
		clf1.fit(X_train, y_train)
		prob = clf1.predict_proba(X_pool)
		temp_ent = entropy(prob.T)

		# Index of max entropy element
		max_ele = np.where(temp_ent == np.amax(temp_ent))

		X_train = np.append(X_train,X_pool[max_ele],axis=0)
		y_train = np.append(y_train,y_pool[max_ele],axis=0)
		
		X_pool = np.delete(X_pool,(max_ele),axis=0)
		y_pool = np.delete(y_pool,(max_ele),axis=0)

		temp_ent = np.delete(temp_ent,(max_ele),axis=0)

	y_pred = clf1.predict(X_test)
	acc = accuracy_score(y_pred, y_test)
	print('Baseline entropy accuracy : ', acc*100)
	print('F1 score on baseline entropy AL : ', f1_score(y_test,y_pred)*100)


	X_train = original_X_train
	X_pool = original_X_pool
	y_train = original_y_train
	y_pool = original_y_pool

	iti = 0
	while iti < 10 :
	# while X_train.shape[0] < (original_X_train.shape[0] + 50) : 	
		print("Iteration ",iti+1)
		print("Number of training data instances for this iti : ",X_train.shape[0])
		iti = iti+1
		print("Start of K")
		for i in range(30): # k->iterations of active learning
	
			clf1 = MLPClassifier(hidden_layer_sizes=(100,100),random_state=random_state, verbose=False, max_iter=1000)
			clf1.fit(X_train, y_train)
			prob = clf1.predict_proba(X_pool)
			temp_ent = entropy(prob.T)

			# Index of max entropy element
			max_ele = np.where(temp_ent == np.amax(temp_ent))

			X_train = np.append(X_train,X_pool[max_ele],axis=0)
			y_train = np.append(y_train,y_pool[max_ele],axis=0)
			
			X_pool = np.delete(X_pool,(max_ele),axis=0)
			y_pool = np.delete(y_pool,(max_ele),axis=0)

			temp_ent = np.delete(temp_ent,(max_ele),axis=0)


		percent = int(0.95*X_train.shape[0])
		acc = 0.00
		max_file = 0

		print("Start of M")
		for j in range(10):
			# print("Size of X_train : ",X_train.shape[0])
			
			# Select 95% of training data randomly
			data_ind = []
			for i in range(percent):
				data_ind.append(random.randrange(X_train.shape[0]))
			

			new_X_train = np.zeros((percent,X_train.shape[1]),dtype=float)
			new_y_train = np.zeros(percent,dtype=int)
			for i,item in enumerate(data_ind):
				new_X_train[i] = X_train[item]
				new_y_train[i] = y_train[item]

			# print("Size of new_X_train : ",new_X_train.shape[0])
			clf1 = MLPClassifier(hidden_layer_sizes=(100,100),random_state=random_state, verbose=False, max_iter=1000)
			clf1.fit(new_X_train, new_y_train)
			y_pred = clf1.predict(X_valid)
			if acc < (accuracy_score(y_pred, y_valid)*100):
				acc = accuracy_score(y_pred, y_valid)*100
				max_file = j
				with open("test.csv","w+") as test:
					csvWriter = csv.writer(test,delimiter=",")
					csvWriter.writerows(new_X_train)

				with open('test.csv', 'r') as read_obj, open('Test_data/test_'+str(j)+'.csv', 'w', newline='') as write_obj:
					csv_reader = csv.reader(read_obj)
					k=0
					csv_writer = csv.writer(write_obj)
					for row in csv_reader:
						row.append(new_y_train[k])
						csv_writer.writerow(row)
						k = k+1
		

		data = pd.read_csv("Test_data/test_"+str(max_file)+".csv")
		data.columns = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4','Eye_detection']
		X = data.loc[:,data.columns !='Eye_detection']
		y = data['Eye_detection']
		X_train = X.to_numpy()
		y_train = y.to_numpy()
	

	clf1 = MLPClassifier(hidden_layer_sizes=(100,100),random_state=random_state, verbose=False, max_iter=1000)
	clf1.fit(X_train, y_train)
	y_pred = clf1.predict(X_test)
	acc = accuracy_score(y_pred, y_test)
	
	print('RANSAC accuracy : ', acc*100)
	print('F1 score on RANSAC AL : ', f1_score(y_test,y_pred)*100)



main()

