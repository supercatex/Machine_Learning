import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


def import_data(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()

    data = []
    for line in lines:
        line = line.replace("\n", "")
        tmp = line.split(" ")
        data.append([float(tmp[1]), float(tmp[0])])
    return np.array(data)


def feature_process(data, scale=1.0):
    feature = np.array(data, dtype=np.float64).reshape(-1, 1)
    feature = feature / scale
    feature = np.hstack([feature, feature ** 2])
    return feature


data = import_data("dataset/dataset_03.txt")
print(data.shape)

max_x = np.max(data[:, 0])
X = feature_process(data[:, 0], max_x)
y = data[:, 1]

model = LinearRegression()
model.fit(X, y)

print(model.score(X, y))
print(model.predict(feature_process([1000], max_x)))

X_lin = np.linspace(0, max(data[:, 0]) * 1.1, 1000)
X_lin2 = np.array(X_lin, dtype=np.float64).reshape(-1, 1)
X_test = feature_process(X_lin, max_x)
y_test = model.predict(X_test)

plt.title("Macao Housing Price & Size Relationship")
plt.xlabel("Square feet")
plt.ylabel("Price HKD 10K")
plt.scatter(data[:, 0], y, 5, color="blue", marker="o")
plt.scatter(X_lin2[:, 0], y_test, 1, color="red", marker=".")
plt.show()
