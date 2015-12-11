def split(X,train_split=2./3):
	'''
	Split a space into training and test set.
	'''

	r_split = int( X.shape[0] * train_split )

	try:
		XTrain= X[0:r_split,:]  # select the top
		XTest = X[r_split:,:]  # select the bottom
	except:
		XTrain[0:r_split]
		XTest[r_split::]

	return XTrain, XTest