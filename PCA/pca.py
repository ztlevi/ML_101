# https://plot.ly/ipython-notebooks/principal-component-analysis/
#
import numpy as np
import pandas as pd
import plotly
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    filepath_or_buffer="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    header=None,
    sep=",",
)

df.columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
df.dropna(how="all", inplace=True)  # drops the empty line at file-end

df.tail()


# split data table into data X and class labels y

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values


# plotting histograms
data = []

legend = {0: False, 1: False, 2: False, 3: True}

colors = {
    "Iris-setosa": "#0D76BF",
    "Iris-versicolor": "#00cc96",
    "Iris-virginica": "#EF553B",
}

for col in range(4):
    for key in colors:
        trace = dict(
            type="histogram",
            x=list(X[y == key, col]),
            opacity=0.75,
            xaxis="x%s" % (col + 1),
            marker=dict(color=colors[key]),
            name=key,
            showlegend=legend[col],
        )
        data.append(trace)

layout = dict(
    barmode="overlay",
    xaxis=dict(domain=[0, 0.25], title="sepal length (cm)"),
    xaxis2=dict(domain=[0.3, 0.5], title="sepal width (cm)"),
    xaxis3=dict(domain=[0.55, 0.75], title="petal length (cm)"),
    xaxis4=dict(domain=[0.8, 1], title="petal width (cm)"),
    yaxis=dict(title="count"),
    title="Distribution of the different Iris flower features",
)

fig = dict(data=data, layout=layout)
plotly.offline.plot(fig, filename="exploratory-vis-histogram.html")


X_std = StandardScaler().fit_transform(X)


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
print("Covariance matrix \n%s" % cov_mat)

print("NumPy covariance matrix: \n%s" % np.cov(X_std.T))

# TODO
