def nnEfficient(F_train,T_train,F_test,T_test,
	nn_iters=100,percent_nn=.25,weights='uniform',compare='score'):
	'''
	Steps from 1 to [percent_nn] percent of total NN in 10 increments.

	F_train: Feature training matrix
	T_train: Target training matrix
	F_test: Feature testing matrix
	T_test: Testing training matrix
	nn_iters: number of increments for the near neighbors
	percent_nn: max percent of nn to test

	Returns correlation coefficient vs nn_range
	Weight/non_weight code adapted/stolen from scikit-learn

	Weights can take on two different forms:
	1. uniform
	2. weighted
	'''

	try:
		predict_out = T_test.shape[1]
	except:
		predict_out=1


	# Step through NN
	min_nn = 1
	max_nn = np.around(F_train.shape[0] * percent_nn)
	step = max_nn/nn_iters
	nn_range = np.arange(min_nn,max_nn,step)

	cc=np.empty((nn_range.shape[0],predict_out))

	#calculate the distances to max_nn using scikit-learn
	knn = neighbors.KNeighborsRegressor(int(max_nn),weights=weights,metric='minkowski') # uniform | distance
	knn_func = knn.fit(F_train,T_train)
	d,ind = knn_func.kneighbors(F_test)

	for ii in range(nn_range.shape[0]):
		nn= int(nn_range[ii])
		
		if weights== 'uniform':

			neigh_ind = ind[:,0:nn]

			y_pred = np.mean(T_train[neigh_ind], axis=1)

		elif weights =='distance':
			dist = d[:,0:nn]
			neigh_ind = ind[:,0:nn]
			W = 1./dist

			y_pred = np.empty((F_test.shape[0], T_test.shape[1]), dtype=np.float)
			denom = np.sum(W, axis=1)

			for j in range(T_test.shape[1]):
				num = np.sum(T_train[ neigh_ind , j] * W, axis=1)
				y_pred[:, j] = num / denom
				#print num.shape

		if compare == 'score':
			cc[ii,:] = score(y_pred,T_test)


	return nn_range,cc

