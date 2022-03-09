# Question 1

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=80, n_features=600, noise=10, random_state=0) #Generates a random linear combination of random features, with noise.

model = Ridge(alpha=1e-8)
model.fit(X, y)
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)

print(f"\nMean Squared Error: {mse}")
print("MSE is 0 up to machine precision:", np.allclose(mse, 0))

X, y = make_regression(n_samples=160, n_features=600, noise=10, random_state=0)
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# My code:
model = Ridge(alpha=1e-8)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print('The compute prediction error is: ' + str(mse))

# Question 2

from sklearn.datasets import make_regression
from sklearn import model_selection
from sklearn.linear_model import Ridge

X, y = make_regression(noise=10) #Generates a random linear combination of random features, with noise.
model = Ridge()

score = model_selection.cross_validate(model, X, y, scoring="mean_squared_error", cv=model_selection.KFold(5))
print('The validation score is: ' + str(score))

# Question 3
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from matplotlib import pyplot as plt

X, y = make_classification() #Generate random data
# my code:
model = GridSearchCV(LogisticRegression(penalty={'l1', 'l2'}, C={0.01, 0.1, 1.0}))
scores = cross_validate(model, X, y)
print(scores)

#question4
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
mnist = fetch_openml('mnist_784', version=1, as_frame= False) #~130MB, might take a little time to download!
mnist.keys()

X, y = mnist["data"], mnist["target"]
print(X.shape)

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

print(y[0])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train = y_train.astype(np.int8) #Casting labels from strings to integers

#Here we are binarizing our labels. All labels that are 5 are converted to True, and the rest to False.
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42) #42 is arbitrarily chosen. From documentation: "Pass an int for reproducible output across multiple function calls"

sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
predict = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, predict)

# week1_l2
#question 1
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.asarray([[0, 1, -10], [0, -1, 0], [1, 0, 10], [1, 0, 0]])
print(f"X:\n{X}\n")

X_scaled = StandardScaler.fit_transform(X)

print(f"X scaled:\n{X_scaled}\n")
print(f"mean: {X_scaled.mean(axis=0)}\nstd: {X_scaled.std(axis=0)}")
#question2
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt

X, y = make_regression(noise=10, n_features=5000, random_state=0)

X_reduced = SelectKBest(f_regression).fit_transform(X, y)
scores = cross_validate(Ridge(), X_reduced, y)["test_score"]
print("feature selection in 'preprocessing':", scores)

model = make_pipeline(SelectKBest(f_regression), Ridge())
scores_pipe = cross_validate(model, X, y)["test_score"]
print("feature selection on train set:", scores_pipe)

# Plotting our results!
plt.boxplot(
    [scores_pipe, scores],
    vert=False,
    labels=[
        "feature selection on train set",
        "feature selection on whole data",
    ],
)
plt.gca().set_xlabel("R² score")
plt.tight_layout()
plt.show()

# Question 3
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

compo = PCA(n_components=2)
compo.fit(X_scaled)
X_PCA = PCA.transform(X_scaled)
print('Origin shape : ' + str(X_scaled.shape))
print("Transform shape: " + str(X_PCA.shape))
