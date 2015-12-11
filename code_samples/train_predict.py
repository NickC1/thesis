def train_predict(x,em,lag,predict,step_NN=25,percent_NN=.25):
	'''
	This function does it all. It takes in the raw time series,
	transforms it into features and targets, splits it into a 
	training and test set, and then near neighbors it.
	'''

	features, target = lagReshape(x,em,lag,predict)

	f_train,f_test,t_train,t_test = trainTestSplit(features,target)

	range_NN,preds,r2 = NN_check(f_train,t_train,f_test,t_test,
		step_NN=step_NN,percent_NN=percent_NN)

	return range_NN, preds, r2