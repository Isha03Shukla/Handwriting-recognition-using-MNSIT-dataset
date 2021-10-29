from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784')
x,y=mnist["data"],mnist["target"]
x.shape
y.shape
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
some_digit=x[37000]
some_digit_image=some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
y[37000]
x_train=x[:60000]
x_test=x[60000:]
y_train=y[:60000]
y_test=y[60000:]
import numpy as np
shuffle_index=np.random.permutation(60000)
x_train, y_train=x_train[shuffle_index],y_train[shuffle_index]
y_train=y_train.astype(np.int8)
y_test=y_test.astype(np.int8)
y_train_4=(y_train==2)
y_test_4=(y_test==2)

y_train
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(tol=0.1,solver='lbfgs')
clf.fit(x_train,y_train_4)
clf.predict([some_digit])
from sklearn.model_selection import cross_val_score
a=cross_val_score(clf,x_train,y_train_4,cv=3,scoring="accuracy")
a.mean()
