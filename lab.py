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
	


### Split into training and test data 70:30 ###
N_test = int(round(N*0.3))
# Generate random array containing n_test ones
test_data = np.array([False]*(N-N_test) + [True]*N_test, dtype=bool)
np.random.shuffle(test_data)

y_train = y[np.logical_not(test_data)]
X_train = X[np.logical_not(test_data),:]
XX_train = XX[np.logical_not(test_data),:]

y_test = y[test_data]
X_test = X[test_data,:]
XX_test = XX[test_data,:]


### Implement gradient ascent ###
beta = np.zeros(D+1)
l_rate = 0.001  # Learning rate
iterations = 999
ll_train = []  # Temporary placeholders
ll_test = []
for count in range(0,iterations):
	# Calculate ll
	#btxn = np.dot(XX_train, beta)  # N x 1 array
	#ll[count] = np.dot(y_train,btxn) - np.sum(0 - np.log(1+np.exp(btxn)))
	
	ll_train.append( compute_average_ll(XX_train, y_train, beta) )
	ll_test.append( compute_average_ll(XX_test, y_test, beta) )
	# Update beta
	beta += l_rate * dL_db(beta,y_train,XX_train)
	
print beta

# Report training curves showing log likelihood on training and test datasets
#     per datapoint (averaged) as the optimisation proceeds
#plot_ll(np.array(ll_train))
#plot_ll(np.array(ll_test))

# Visualise the predictions by adding probability contours to the plots made in part (c)
plot_predictive_distribution(X, y, beta, predict_for_plot)


# Report the final training and test log-likelihoods per datapoint