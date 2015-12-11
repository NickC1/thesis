def mutual_information_spatial(M,max_lag,percent_calc=.5):
	'''
	Calculates the mutual information along the rows and columns at a
	certain number of indices (percent_calc) and returns 
	the sum of the mutual informaiton along the columns and along the rows.

	M: Input matrix
	num_bins: Parameter for mutual info calculation 
	max_lag: How far to shift the space
	percent_calc: How many rows and columns to use 

	Returns:
	R_mut: the mutual inforation down the rows (vertical)
	C_mut: the mutual information across the columns (horizontal)

	'''

	rs, cs = np.shape(M)

	rs_iters = int(rs*percent_calc)
	cs_iters = int(cs*percent_calc)

	r_picks = np.random.choice(np.arange(rs),size=rs_iters,replace=False)
	c_picks = np.random.choice(np.arange(cs),size=cs_iters,replace=False)


	# The r_picks are used to calculate the MI in the columns
	# and the c_picks are used to calculate the MI in the rows

	c_mi = np.zeros((max_lag,rs_iters))
	r_mi = np.zeros((max_lag,cs_iters))

	for ii in range(rs_iters):

		m_slice = M[r_picks[ii],:]

		c_mi[:,ii] = mutual_infomation(m_slice,max_lag)

	for ii in range(cs_iters):

		m_slice = M[:,c_picks[ii]]
		r_mi[:,ii] = mutual_infomation(m_slice,max_lag)

	r_mut = np.sum(r_mi,axis=1)
	c_mut = np.sum(c_mi,axis=1)

	return r_mut, c_mut, r_mi, c_mi