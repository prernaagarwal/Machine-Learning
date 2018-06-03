#!/usr/bin/python3
#Prerna Agarwal


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


minimal_std_dev = 0.0001 
total_features = 57

##################### STEP 1. CREATE TRAINING AND TEST SET ########################
input_data = np.loadtxt('spambase.data', delimiter=',', dtype = float)
np.random.shuffle(input_data)
X, target = input_data[:,:-1], input_data[:,-1]
#print(input_data)

# Split the data into a training and test set.
# Each of these should have about 2,300 instances i.e., 50% of given data set
train_input, test_input, train_target, test_target = train_test_split(X, target, test_size=0.50, random_state=42)
################################### END STEP 1 ###################################



############################# STEP 2. CREATE PROBABILISTIC MODEL ###################

# Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in the training data. 
# P(spam) should be about 0.4
train_spam = 0
for i in range(len(train_target)):
	if (train_target[i] == 1):
		train_spam+=1
P_train_spam = (train_spam*1.0)/len(train_target)
P_train_nspam = 1 - P_train_spam
#print(P_train_spam)
#print(P_train_nspam)

# Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in the test data. 
# P(spam) should be about 0.4
test_spam = 0
for i in range(len(test_target)):
	if (test_target[i] == 1):
		test_spam+=1
P_test_spam = (test_spam*1.0)/len(test_target)
P_test_nspam = 1 - P_test_spam
#print(P_test_spam)
#print(P_test_nspam)


# For each of the 57 features, compute the mean and standard deviation in the training set of the values given each class. 
mean_spam, mean_nspam = [], []
stddev_spam, stddev_nspam = [], []

for feature in range(0,total_features):
	spam_data, nspam_data = [], []
	
	for row in range(len(train_input)):
		if (train_target[row] == 1):  #spam class
			spam_data.append(train_input[row][feature])
		else:                         #not spam class
			nspam_data.append(train_input[row][feature])
	
	mean_spam.append(np.mean(spam_data))
	#print(len(mean_spam))
	mean_nspam.append(np.mean(nspam_data))
	#print(len(mean_nspam)) 
	stddev_spam.append(np.std(spam_data))
	stddev_nspam.append(np.std(nspam_data))

#print(stddev_spam, "\n")

# If any of the features has zero standard deviation, assign it a “minimal” standard deviation (e.g., 0.0001) 
# to avoid a divide-by-zero error in Gaussian Naïve Baye
for i in range(len(stddev_spam)):
	if (stddev_spam[i] == 0):
		#print ("0 at index:", i, "\n")
		stddev_spam[i] = minimal_std_dev
	if (stddev_nspam[i] == 0):
		#print ("0 at index:", i, "\n")
		stddev_nspam[i] = minimal_std_dev

#print(stddev_spam)

################################### END STEP 2 ###################################



############################# STEP 3. RUN NAIVE BAYES ON THE TEST DATA ###################

class_x = 0
result = []

#Function to calculate normal distribution
def gaussian(x,mu,stddev):
	step_1 = float(1/(np.sqrt(2*np.pi)*stddev))
	#if (step_1 < 0.001):
	 #   print(1)
	#print(-((x-mu)**2))
	#print(np.exp(-((x-mu)**2))) 
	#print(stddev)
	step_2 = step_1 * float(np.exp(-((x-mu)**2)/(2*float(stddev*stddev))))
	if (step_2 <= 0.000000000000000000001):
		step2 = 0.000000000000000000001
	

	return step_2


for row in range(len(test_input)):
	probx_1 = np.log(P_train_spam)
	probx_0 = np.log(P_train_nspam)

	for feature in range(0,total_features):
		x = test_input[row][feature]
		probx_1 += np.log(gaussian(x, mean_spam[feature], stddev_spam[feature]))
		probx_0 += np.log(gaussian(x, mean_nspam[feature], stddev_nspam[feature]))

	#find class_x for each row of test input and add it to result	
	class_x = np.argmax([probx_0, probx_1])
	result.append(class_x)


#print(len(result)) #2301

#Compute a confusion matrix for the test set
print("\n ")
cfm = confusion_matrix(test_target, result)
print("\nConfusion Matrix: \n\n", cfm)
print("\n")

#Calculating accuracy, precision, and recall on the test set
#TP = True Positive, TN = True Negative, FP = False Positive, FN = Flase Negative
TP,TN,FP,FN = 0,0,0,0

for row in range(len(result)):
	if (result[row] == 1 and test_target[row] == 1):
		TP += 1
	elif (result[row] == 0 and test_target[row] == 0 ):
		TN += 1
	elif (result[row] == 1 and test_target[row] == 0 ):
		FP += 1
	else:     # (result[row] == 0 and test_target[row] == 1 ):
		FN += 1

accuracy = float(TP + TN)/(TP+TN+FP+FN)
precision = float(TP)/(TP+FP)
recall = float(TP)/(TP+FN)
print ("Accuracy : ", accuracy)
print ("Precision: ", precision)
print ("Recall   : ", recall)

################################### END STEP 3 ###################################
