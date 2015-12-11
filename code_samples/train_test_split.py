def trainTestSplit(features,target,train_size=.33):
	'''
	Split into test and training sets
	'''

	nsamples,nfeatures = features.shape

	split = int(nsamples*train_size)

	f_train = features[:split]
	f_test = features[split:]

	t_train = target[:split]
	t_test = target[split:]

	return f_train,f_test,t_train,t_test