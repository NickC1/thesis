def lagReshape(X,em,lag,predict):
	'''
	Constructs the space for prediction using the specified lag
	value and embedding dimension.

	Example: 
	
	X = [0,1,2,3,4,5,6,7,8,9,10]
	
	em = 3
	lag = 2
	predict=3

	returns:
	features = [0,2,4], [1,3,5], [2,4,6], [3,5,7]
	targets = [5,6,7], [6,7,8], [7,8,9], [8,9,10]
	'''

	tsize = X.shape[0]
	t_iter = tsize-predict-(lag*(em-1))

	features = np.zeros((t_iter,em))
	targets = np.zeros((t_iter,predict))

	for ii in range(t_iter):

		end_val = ii+lag*(em-1)+1

		part = X[ii : end_val]

		features[ii,:] = part[::(lag)]
		targets[ii,:] = X[end_val:end_val+predict]
	return features, targets