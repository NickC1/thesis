nn_range,cc = nnEfficient(f_train,t_train,f_test,t_test,weights='uniform')
plt.contourf(cc.T);
plt.colorbar()