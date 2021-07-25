# Unsupervised Learning

### [K-means](https://github.com/ztlevi/Machine_Learning_Questions/tree/26cb30cb7a3ec95f737534585c8ae80567d03d7b/docs/General/Clustering.html$Kmeans)

### t-SNE

t-Distributed Stochastic Neighbor Embedding \(t-SNE\) is a technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets. The technique can be implemented via Barnes-Hut approximations, allowing it to be applied on large real-world datasets. We applied it on data sets with up to 30 million examples.‌

​[https://github.com/ztlevi/t-sne-python/blob/master/tsne.py](https://github.com/ztlevi/t-sne-python/blob/master/tsne.py)

### Principal Component Analysis

Statistical procedure that uses an **orthogonal transformation** to convert a set of observations of possibly **correlated variables** \(entities each of which takes on various numerical values\) into a set of values of **linearly uncorrelated variables** called **principal components**.

* Principal Component Analysis \(PCA\) is a dimension reduction technique that projects the data into a lower dimensional space
* PCA uses Singular Value Decomposition \(SVD\), which is a matrix factorization method that decomposes a matrix into three smaller matrices \(more details of SVD [here](https://en.wikipedia.org/wiki/Singular-value_decomposition)\)
* PCA finds top N principal components, which are dimensions along which the data vary \(spread out\) the most. Intuitively, the more spread out the data along a specific dimension, the more information is contained, thus the more important this dimension is for the pattern recognition of the dataset
* PCA can be used as pre-step for data visualization: reducing high dimensional data into 2D or 3D. An alternative dimensionality reduction technique is [t-SNE](https://lvdmaaten.github.io/tsne/)

Here is a visual explanation of PCA

## 

