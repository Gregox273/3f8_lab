import numpy as np
from appendix import *

### Constants ###
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')
#plot_data(X,y)
XX = np.insert(X,0,1,axis=1)  # Append column of ones as beta[0] coeffs
N = y.size  # 1000
D = 2

def dL_db(b,y,X):

	a = np.exp(np.dot(X,b))
	b = a / (1 + a)
	c = y - b

	rtn = np.dot(c,X)

	return rtn

def train(iterations, rate, ll_train, ll_test, x_tr, y_tr, x_t, y_t, b):
	for count in range(0,iterations):
		# Calculate ll
		#btxn = np.dot(XX_train, beta)  # N x 1 array
		#ll[count] = np.dot(y_train,btxn) - np.sum(0 - np.log(1+np.exp(btxn)))

		ll_train.append( compute_average_ll(x_tr, y_tr, b) )
		ll_test.append( compute_average_ll(x_t, y_t, b) )
		# Update beta
		b += rate * dL_db(b,y_tr,x_tr)

	return b

### Split into training and test data 70:30 ###
N_test = int(round(N*0.3))
# Generate random array containing n_test 'True's
test_data = np.array([False]*(N-N_test) + [True]*N_test, dtype=bool)  # Boolean mask
np.random.shuffle(test_data)  # Randomise

y_train = y[np.logical_not(test_data)]
X_train = X[np.logical_not(test_data),:]
XX_train = XX[np.logical_not(test_data),:]

y_test = y[test_data]
X_test = X[test_data,:]
XX_test = XX[test_data,:]

### Implement gradient ascent ###
beta = np.zeros(D+1)
l_rate = 0.001  # Learning rate
iterations = 9999
ll_train = []  # Temporary placeholders
ll_test = []

beta = train(iterations, l_rate, ll_train, ll_test, XX_train, y_train, XX_test, y_test, beta)

# Define global for appendix functions
w = np.roll(beta,-1)  # due to way x_tilde is defined in appendix
print "Beta = " + np.array2string(beta)

### Report the final training and test log-likelihoods per datapoint ###
print "Final training ll = %f" %(ll_train[-1])
print "Final test ll = %f" %(ll_test[-1])

### Make predictions based on beta ###
class_2_mask = logistic(np.dot(XX_test, beta)) > 0.5  # p(y_n = 1 | x_tilde_n)
y_predict = np.zeros(N_test)
y_predict[class_2_mask] = 1

### Determine confusion matrix ###
conf_00 = np.logical_not( np.logical_or(y_predict, y_test) )  # p(0|y=0) true negatives
conf_01 = np.logical_and( np.logical_not(y_test), y_predict )  # p(1|y=0) false positives
conf_10 = np.logical_and( y_test, np.logical_not(y_predict) )  # p(0|y=1) false negatives
conf_11 = np.logical_and( y_test, y_predict )  # p(1|y=1) true positives

conf_00 = np.sum(conf_00)/(y_test.size - np.sum(y_test))  # num of 00 / num of zeros in y
conf_01 = np.sum(conf_01)/(y_test.size - np.sum(y_test))
conf_10 = np.sum(conf_10)/np.sum(y_test)
conf_11 = np.sum(conf_11)/np.sum(y_test)

confusion = np.array([[conf_00, conf_01],[conf_10, conf_11]])
print "Confusion matrix:"
print confusion

### Report training curves showing log likelihood on training and test datasets
#     per datapoint (averaged) as the optimisation proceeds ###
#plot_ll(np.array(ll_train))
#plot_ll(np.array(ll_test))

### Visualise the predictions by adding probability contours to the plots made in part (c) ###
plot_predictive_distribution(X, y, w, predict_for_plot)


### Expand X ###
X_expand = expand_inputs(0.01, X, X_train)
X_train_expand = expand_inputs(0.01, X_train, X_train)
X_test_expand = expand_inputs(0.01, X_test, X_train)
XX_expand = np.insert(X_expand,0,1,axis=1)
XX_train_expand = np.insert(X_train_expand,0,1,axis=1)
XX_test_expand = np.insert(X_test_expand,0,1,axis=1)

### Train model on expanded inputs
beta_expand = np.zeros(y_train.size + 1)
l_rate = 0.001
iterations = 9999
ll_train = []
ll_test = []

beta_expand = train(iterations, l_rate, ll_train, ll_test, XX_train_expand, y_train,
                    XX_test_expand, y_test, beta_expand)
w = np.roll(beta_expand,-1)