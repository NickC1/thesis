def ftReshapePercent(X,lag,embed,predict,percent=None):
	'''
	X:			Matrix to embed
	lag:		tuple of row and column lag values (r,c) 
	embed: 	tuple of embedding shape (r,c)
	predict: 	How far to predict outward 
	percent:	What percent of the space to reshape (pics random locs)

	Can think of the lag and embedding as height,width


	Example:
	lag = (3,4)
	embed = (2,5) 
	predict = 2


	[ ] _ _ _ [ ] _ _ _ [ ] _ _ _ [ ] _ _ _ [ ]
	|         |         |         |         |
	|         |         |         |         |
	[ ] _ _ _ [ ] _ _ _ [ ] _ _ _ [ ] _ _ _ [ ]
	 *
	 *
	'''

	rsize = X.shape[0]
	csize = X.shape[1]

	r_lag,c_lag = lag
	rem,cem = embed


	# determine how many iterations we will have and
	# the empty feature and target matrices

	c_iter = csize - c_lag*(cem-1)
	r_iter = rsize  - predict - r_lag*(rem-1)

	#create tuples of all the possible x,y values for the image
	# creates a bunch of tuples
	xx,yy = np.meshgrid(range(r_iter),range(c_iter))
	z = zip(xx.ravel(),yy.ravel())

	#choose only a percent of them if percent is defined
	if percent:
		tot = r_iter*c_iter
		percent_tot = int(tot*percent)
		rand_pic = np.random.choice(tot,percent_tot,replace=False)

		z = [z[ii] for ii in rand_pic]

		targets = np.zeros((percent_tot,predict))
		features = np.zeros((percent_tot,rem*cem))


	else:
		targets = np.zeros((c_iter*r_iter,predict))
		features = np.zeros((c_iter*r_iter,rem*cem))



	print 'targets before loop:', targets.shape

	for ii in range(features.shape[0]):

		rs,cs = z[ii]


		r_end_val = rs+r_lag*(rem-1)+1
		c_end_val = cs+c_lag*(cem-1)+1

		part = X[rs : r_end_val, cs : c_end_val ]

		features[ii,:] = part[::r_lag,::c_lag].ravel()
		targets[ii,:] = X[r_end_val:r_end_val+predict,cs+ c_lag*(cem-1)/2]


	return features,targets