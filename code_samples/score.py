def score_metric(preds,test):
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