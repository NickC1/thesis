def classCompare(preds,test):
	'''
	Compares the predicted to the actual classes. It's a binary classification.
	The output is what percent it got correct.
	'''

	try:
		num_preds = preds.shape[1]
	except:
		num_preds=1

	cc = np.empty((num_preds,))

	if num_preds ==1:
		cc = np.mean( preds == test )

	else:

		for ii in range(num_preds):
			cc[ii] =np.mean( preds[:,ii] == test[:,ii] )

	return cc