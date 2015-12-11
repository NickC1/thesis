def train_predict(X,embed,embed_step,predict,
				train_split=.5,
				percent_train=.1,percent_test=.1,
				nn_iters=100,percent_nn=.01,weights='uniform',compare='score',
				metric='minkowski'):
	'''
	This function does it all. It takes in the raw space,
	splits it into a training and test set, transforms it into
	features and targets, and then near neighbors it.
	'''

	# Split into a training set and test set
	train,test = split(X,train_split=train_split)

	# Reshape the space for the knn algorithm. 

	f_train,t_train = ftReshapePercent(train,embed,embed_step,predict,
		percent=percent_train)
	f_test,t_test = ftReshapePercent(test,embed,embed_step,predict,
		percent=percent_test)
	


	# Calling the main driver. This is an efficient implementation of the knn
	# algorithm

	nn_range,cc = nnEfficient(f_train,t_train,f_test,t_test,nn_iters=nn_iters,
						percent_nn=percent_nn,weights=weights,compare=compare,
						metric=metric)

	return nn_range, cc