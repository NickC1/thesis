def nnClassificationEfficient(F_train,T_train,F_test,T_test,nn_iters=100,
	percent_nn=.01,weights='uniform',compare='classCompare'):
	'''
	Steps from 1 to [percent_nn] percent of total NN in nn_iters increments.

	F_train: Feature training matrix
	T_train: Target training matrix
	F_test: Feature testing matrix
	T_test: Testing training matrix
	nn_iters: number of increments for the near neighbors
	percent_nn: max percent of nn to test

	Returns percent correct vs nn_range
	Weight/non_weight code adapted/stolen from scikit-learn
	
	Weights can take on two different forms:
	1. uniform
	2. weighted
	'''

	# Step through NN
	min_nn = 1
	max_nn = np.around(F_train.shape[0] * percent_nn)
	step = max_nn/nn_iters
	nn_range = np.arange(min_nn,max_nn,step)

	cc=np.empty((nn_range.shape[0],predict_out))

	#calculate the distances to max_nn using scikit-learn
	knn = neighbors.KNeighborsRegressor(int(max_nn),weights=weights,metric='hamming')
	knn_func = knn.fit(F_train,T_train)
	d,ind = knn_func.kneighbors(F_test)

	_y = T_train

	classes_ = np.unique(_y)

	for ii in range(nn_range.shape[0]):

		nn= int(nn_range[ii])
		neigh_dist = d[:,0:nn]
		neigh_ind = ind[:,0:nn]

		n_outputs = len(classes_)
		n_samples = T_test.shape[0]

		W = 1./neigh_dist

		y_pred = np.empty((n_samples, predict_out), dtype=classes_[0].dtype)

		#for k, classes_k in enumerate(classes_):
		if weights =='uniform':
			mode, _ = stats.mode(_y[neigh_ind], axis=1)
		else:
			for j in range(T_test.shape[1]):
				mode, _ = weighted_mode(_y[neigh_ind,j], W, axis=1)
				
				mode = np.asarray(mode.ravel(), dtype=np.intp)
			
				y_pred[:, j] = mode

		if compare =='classCompare':
			cc[ii,:] = classCompare(y_pred,T_test)

	return nn_range,cc