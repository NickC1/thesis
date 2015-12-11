em = 2
lag = 3
predict = 20
f_train, t_train = lagReshape(xtrain,em,lag,predict)
f_test, t_test = lagReshape(xtest,em,lag,predict)

plt.plot(f_train[:,0],f_train[:,1],linewidth=.5)
sns.despine()