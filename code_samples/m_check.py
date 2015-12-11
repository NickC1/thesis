def em_check(xtrain,xtest,lag,predict):

    em_mean = np.empty((10,))
    em_max = np.empty((10,))
    for ii in range(10):
        em = ii+1
        f_train, t_train = lagReshape(xtrain,em,lag,predict)
        f_test, t_test = lagReshape(xtest,em,lag,predict)
        nn_range,cc = nnEfficient(f_train,t_train,f_test,t_test,weights='uniform')

        em_mean[ii] = np.mean(cc)
        em_max[ii] = np.max(cc)
    return em_mean, em_max