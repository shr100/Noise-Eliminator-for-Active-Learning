import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# In[2]:
# ADD F1 SCORE
# MAKE EVERY SCORE * 100


random_state = 8
data = pd.read_csv('EEG_Eye_State.csv')
# print(data.shape)
# print(data)
X = data.loc[:,data.columns !='Eye_detection']
y = np.array(data['Eye_detection'])
# print(y)


# In[5]:


# Split into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
# Train a multi-layer perceptron
clf0 = MLPClassifier(hidden_layer_sizes=(100,100),random_state=random_state, verbose=False, max_iter=1000)
clf0.fit(X_train, y_train)
# Predict accuracy of classifier
y_pred = clf0.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print('Accuracy on raw : ', acc*100)
print('F1 score on raw : ', f1_score(y_test,y_pred)*100)


# In[6]:


#Feature scaling
scaled_X = preprocessing.scale(X)
# Split into train and test data 
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.2, random_state = random_state)
# Train a multi-layer perceptron
clf0 = MLPClassifier(hidden_layer_sizes=(100,100),random_state=random_state, verbose=False, max_iter=1000)
clf0.fit(X_train, y_train)
# Predict accuracy of classifier
y_pred = clf0.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print('Accuracy on raw scaled : ', acc*100)
print('F1 score on raw scaled : ', f1_score(y_test,y_pred)*100)


# In[24]:


################ split data #######################
# train: 0.2 total  
# pool: 0.6 total
# test: 0.2 total
# ################################################# 

def split_data(scaled_X, y, noise_probability = 0.0, add_noise_to_train=True):
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.2, random_state = random_state)
    
    np.random.seed(random_state)
    X_train, X_pool,y_train, y_pool = train_test_split(X_train, y_train, test_size = 0.75 , random_state = random_state)
    if add_noise_to_train:
    #adding noise to train
        y_train = np.abs((np.random.random(y_train.shape)<noise_probability).astype(int)  -y_train)  # -> either |1-y_train|,or |0-y_train| for each data sample
    #adding noise to pool
    y_pool = np.abs((np.random.random(y_pool.shape)<noise_probability).astype(int)  -y_pool)  # -> either |1-y_pool|,or |0-y_pool| for each data sample

    print ("---------")
    print(f"total: {y.size}\ntrain: {y_train.size} -> {y_train.size/y.size:.2f}x \npool: {y_pool.size} -> {y_pool.size/y.size:.2f}x \ntest: {y_test.size} -> {y_test.size/y.size:.2f}x")
    print ("---------")
    return X_train, X_pool, X_test, y_train, y_pool, y_test


# In[13]:




def find_most_ambigious(y_proba_pred, y, ambigious_amount =1, method='least_confidence') -> list:
    """This function finds most ambigous predicted data and returns their indexes. It assumes
        we have only two class.

	Args:
		y_proba_pred ([list]): [predicted probabilities]
		y ([list]): [ground truth labels]
		ambigious_amount (int, optional): [quantity of most ambigious]. Defaults to 1.
		method (str, optional): [method type i.e, least_confidence]. Defaults to 'least_confidence'.

	Returns:
		indexes ([list]): [indexes of the most ambigious]
	"""
    indexes = []
    if method == 'least_confidence':
        difference = np.abs(y_proba_pred[:,0]-y_proba_pred[:,1])
        indexes = np.argsort(difference)[:ambigious_amount]
    else:
        print("method is not defined. Use 'least_confidence'")
        
    return indexes

def train_one_iter_active_learning(X_train, y_train, X_pool, y_pool, X_test, y_test, model, ambigious_amount=1  , method='least_confidence'):

    y_proba_pred = model.predict_proba(X_pool)

    most_ambigious_indexes = find_most_ambigious(y_proba_pred, y_pool, ambigious_amount =ambigious_amount, method='least_confidence')
    
    X_train = np.append(X_train,X_pool[most_ambigious_indexes],axis = 0)
    y_train = np.append(y_train, y_pool[most_ambigious_indexes])
    X_pool = np.delete(X_pool,most_ambigious_indexes,axis=0)
    y_pool = np.delete(y_pool,most_ambigious_indexes)
    model.fit(X_train,y_train)
    acc = model.score(X_test,y_test)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test,y_pred)

    return X_train, y_train, X_pool, y_pool, model, acc, f1


# In[27]:


## pure active learning
noise_probability = 0.2
ambigious_amount = 100
K = 10
add_noise_to_train = True  # pool always has noise
#############################
X_train, X_pool, X_test, y_train, y_pool, y_test = split_data(scaled_X, y, noise_probability=noise_probability, add_noise_to_train=add_noise_to_train)
clf1 = MLPClassifier(verbose=0, hidden_layer_sizes=(100,100),random_state = random_state)
clf1.fit(X_train, y_train)
acc = clf1.score(X_test,y_test)

print (f"iteration -1:   accuracy = {acc:0.5f}")
print ("--")

for k in range(K):
    X_train, y_train, X_pool, y_pool, clf1, acc, f1 = train_one_iter_active_learning(X_train, y_train, X_pool, y_pool, X_test, y_test, model=clf1, ambigious_amount=ambigious_amount , method='least_confidence')
    # print (f"iteration {k}:   accuracy = {acc:0.5f}")
    # print ("--")

pure_al = clf1
#pure_al_disp = plot_roc_curve(pure_al, X_test, y_test)
pred_probs1 = pure_al.predict_proba(X_test)
print(f"Least confidence accuracy = {acc*100:0.5f}")
print(f"Least confidence F1 score = {f1*100:0.5f}")
print(f"train size: {y_train.shape}")


# In[28]:


## Active learning with Ransac

##################### in total N*k times active learning iterations, N*M times Ransac iterations ##############

noise_probability = 0.2
K = 2
N = 5
ambigious_amount = 100    
M = 30   #RANSAC
ransac_percent = 0.95
add_noise_to_train = True  # pool always has noise

################################################################################################################
X_train, X_pool, X_test, y_train, y_pool, y_test = split_data(scaled_X, y,noise_probability = noise_probability, add_noise_to_train=add_noise_to_train)

clf1 = MLPClassifier(verbose=0, hidden_layer_sizes=(100,100),random_state = random_state)
clf1.fit(X_train, y_train)
acc = clf1.score(X_test,y_test)
print (f"iteration -1 accuracy = {acc:0.5f}")
print ("--")

for n in range(N):

    print(f"################ Outer iteration {n} ################ ")

    ############ K iteration active learning -> everytime label ambigious_amount data ############
    for k in range(K):
        X_train, y_train, X_pool, y_pool, clf1, acc, f1 = train_one_iter_active_learning(X_train, y_train, X_pool, y_pool, X_test, y_test, clf1, ambigious_amount=ambigious_amount , method='least_confidence')
        print (f"AL iteration {k}:   accuracy = {acc:0.5f}")
    print(f"train size: {y_train.shape}")
    print ("--------")
    
    ###########################################################################


    ############ M iteration RANSAC ###########################################
    
    stats_history =[]
    for m in range(M):
        ransac_random_state = random_state + m # to make sure repeatable results

        r_X_train,r_X_outlier, r_y_train, r_y_outlier = train_test_split(X_train, y_train, train_size = ransac_percent, random_state = ransac_random_state)
        clf1 = MLPClassifier(verbose=0, hidden_layer_sizes=(100,100),random_state = random_state)
        clf1.fit(r_X_train, r_y_train)
        acc = clf1.score(X_test, y_test)
        y_pred = clf1.predict(X_test)
        f1 = f1_score(y_test,y_pred)
        print (f"Ransac iteration {m}:   accuracy = {acc:0.5f}")
        print (f"Ransac iteration {m}:   f1 score = {f1:0.5f}")
        
        stat = {"model":clf1, "X_train":r_X_train, "y_train":r_y_train, "accuracy":acc, "f1":f1}
        stats_history.append(stat)

    # Take the best model and 95% data that gives best accuracy     
    best = sorted(stats_history, key=lambda x: x["f1"])[-1]
    clf1 = best["model"]
    X_train = best["X_train"]
    y_train = best["y_train"]
    acc = best["accuracy"]
    f1 = best["f1"]
    print ("----------------")

    print(f"train size: {y_train.shape}")
    print (f"Final accuracy = {acc*100:0.5f}")
    print (f"Final f1 score = {f1*100:0.5f}")
    print ("----------------")


#ransac_disp = plot_roc_curve(clf1, X_test, y_test, ax=svc_disp.ax_)
#rfc_disp.figure_.suptitle("ROC curve comparison")


pred_probs2 = clf1.predict_proba(X_test)

from sklearn.metrics import roc_curve

fpr1, tpr1, thresh1 = roc_curve(y_test, pred_probs1[:,1],pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_probs2[:,1],pos_label=1)

random_probs = [0 for i in range(len(y_test))]

p_fpr, p_tpr,_ = roc_curve(y_test,random_probs,pos_label = 1)

from sklearn.metrics import roc_auc_score

auc_score1 = roc_auc_score(y_test,pred_prob1[:,1]) 
auc_score2 = roc_auc_score(y_test,pred_prob2[:,1])

print("AUC Scores for Pure AL",auc_score1)
print("AUC Scores for RANSAC",auc_score2)

import matplotlib.pyplot as plt
plt.style.use('seaborn')

plt.plot(fpr1,tpr1,linestyle='--',color='orange',label = 'Pure Active Learning')
plt.plot(fpr2,tpr2,linestyle='--',color='green',label = 'Active Learning with NEAL')

plt.title('AUC-ROC curve comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()
