import nonLinPy.nonLinPy as nl

embed=(2,2)
embed_step = (1,1)
predict=7
nn_range,cc = nl.train_predict(X,embed,embed_step,predict)