def nnEfficient(F_train,T_train,F_test,T_test,
    nn_iters=100,percent_nn=.25,weights='uniform',compare='score',metric='minkowski'):
    '''
    Steps from 1 to [percent_nn] percent of total NN in 10 increments.

    F_train: Feature training matrix
    T_train: Target training matrix
    F_test: Feature testing matrix
    T_test: Testing training matrix
    nn_iters: number of increments for the near neighbors
    percent_nn: max percent of nn to test

    Returns correlation coefficient vs nn_range
    Weight/non_weight code adapted/stolen from scikit-learn
    45 seconds to run a 128 x 128 region.
    There are three ways to compare the predicted and actual values:
    1. Forecast Error - fnc: forecastError
    2. Correlation Coefficient - fnc: corrCoef
    3. Class Compare - fnc: classCompare  
    ** Must use metric='hamming' for classCompare. Also toggles 'mode' for predictions **

    Weights can take on two different forms:
    1. uniform
    2. weighted
    '''

    try:
        predict_out = T_test.shape[1]
    except:
        predict_out=1


    # Step through NN
    min_nn = 1
    max_nn = np.around(F_train.shape[0] * percent_nn)
    step = max_nn/nn_iters
    nn_range = np.arange(min_nn,max_nn,step)

    cc=np.empty((nn_range.shape[0],predict_out))

    #calculate the distances to max_nn using scikit-learn
    knn = neighbors.KNeighborsRegressor(int(max_nn),weights=weights,metric=metric) # uniform | distance
    knn_func = knn.fit(F_train,T_train)
    d,ind = knn_func.kneighbors(F_test)


    #print 'weights=', weights
    #print 'metric=', metric
    #print 'compare=', compare

    for ii in range(nn_range.shape[0]):
        nn= int(nn_range[ii])



        if weights== 'uniform':

            neigh_ind = ind[:,0:nn]


            y_pred = np.mean(T_train[neigh_ind], axis=1)


        elif weights =='distance':
            dist = d[:,0:nn]
            neigh_ind = ind[:,0:nn]
            W = 1./dist

            y_pred = np.empty((F_test.shape[0], T_test.shape[1]), dtype=np.float)
            denom = np.sum(W, axis=1)

            for j in range(T_test.shape[1]):
                num = np.sum(T_train[ neigh_ind , j] * W, axis=1)
                y_pred[:, j] = num / denom
                #print num.shape

        # do the testing
        if compare == 'forecastError':
            cc[ii,:] = forecastError(y_pred,T_test)

        elif compare == 'score':
            cc[ii,:] = score(y_pred,T_test)

        elif compare == 'corrCoef':
            cc[ii,:] = corrCoef(y_pred,T_test)

        elif compare =='classCompare':
            cc[ii,:] = classCompare(y_pred,T_test)

        elif compare =='varianceExplained':
            cc[ii,:] = varianceExplained(y_pred,T_test)


    return nn_range,cc

def score(preds,test):
    '''
    The coefficient R^2 is defined as (1 - u/v), where u is the regression
    sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual 
    sum of squares ((y_true - y_true.mean()) ** 2).sum(). Best possible 
    score is 1.0, lower values are worse.
    '''

    num_preds = preds.shape[1]
    r2 = np.empty((num_preds,))

    for ii in range(num_preds):

        u = np.square(test[:,ii]-preds[:,ii]).sum()
        v = np.square(test[:,ii]-test[:,ii].mean()).sum()
        r2[ii] = 1 - u/v

    return r2


def em_check(xtrain,xtest,lag,predict,max_em):
    
    em_mean = np.empty((max_em,))
    em_max = np.empty((max_em,))
    for ii in range(max_em):
        em = ii+1
        f_train, t_train = lagReshape(xtrain,em,lag,predict)
        f_test, t_test = lagReshape(xtest,em,lag,predict)
        nn_range,cc = nnEfficient(f_train,t_train,f_test,t_test,weights='uniform')
        
        em_mean[ii] = np.mean(cc)
        em_max[ii] = np.max(cc)
    return em_mean, em_max


em_mean,em_max = em_check(xtrain,xtest,lag,30,10)

fig,ax = plt.subplots(nrows=2,sharex=True)
m = np.arange(1,11)

ax[0].plot(m,em_mean,linewidth=2)
ax[0].set_ylabel('Mean $R^2$')
sns.despine()

ax[1].plot(m,em_max,linewidth=2)
ax[1].set_ylabel('Max $R^2$')
sns.despine()


fig.subplots_adjust(hspace=.4)