def deterministic(X,mut_info):
	'''
	Calculates the deterministic metric out to 1/2 of the mutual
	information for $R^2$ values over a range of near neighbors and forecast distances

	Input:
	X : R^2 values over a range of near neighbors and forecast distances
	mut_info : mutual information of the system
	'''

	sz = np.floor(mut_info/2.)
	det_metric = np.zeros((sz,))
	half = X.shape[1]


	for ii in range(sz):

		det_metric[ii] += np.max(X[0:half,ii]) - np.min(X[half::,ii])

	return det_metric