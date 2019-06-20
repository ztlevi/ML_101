# K means

![clustering](../../assets/clustering.png)

- Clustering is a unsupervised learning algorithm that groups data in such a way that data points in the same group are more similar to each other than to those from other groups
- Similarity is usually defined using a distance measure (e.g, Euclidean, Cosine, Jaccard, etc.)
- The goal is usually to discover the underlying structure within the data (usually high dimensional)
- The most common clustering algorithm is K-means, where we define K (the number of clusters) and the algorithm iteratively finds the cluster each data point belongs to

[scikit-learn](http://scikit-learn.org/stable/modules/clustering.html) implements many clustering algorithms. Below is a comparison adopted from its page.

## Algorithm

- Input:
  - $$K$$ (number of clusters)
  - Training set $${x^1,x^2,...,x^m}$$ ($$x^i \in \mathbb{R}^n$$)
- Algorithm:

  1. Randomly initialized $$K$$ cluster centroids $$\mu_1,\mu_2,...,\mu_K \in \mathbb{R}^n$$
  2. Repeat {

     - for i = 1 to $$m$$
       - $$c^i$$ := index (from 1 to K) of cluster centroid closest to $$x^i$$
     - for k = 1 to $$K$$
       - $$\mu_k$$ := average (mean) of points assigned to Cluster k

     }

**[Python implementation](https://github.com/ztlevi/Machine_Learning_Questions/blob/master/codes/kmeans/kmeans.py)**

## Random initialization

- How we initialize K-means

  - And how avoid local optimum

- Consider clustering algorithm
  - Never spoke about how we initialize the centroids
    - A few ways - one method is most recommended
- Have number of centroids set to less than number of examples (K < m) (if K > m we have a problem)0
  - Randomly pick K training examples
  - Set μ1 up to μK to these example's values
- K means can converge to different solutions depending on the initialization setup
  - Risk of local optimum ![img](../../assets/kmeans_local_optima.png)
  - The local optimum are valid convergence, but local optimum not global ones
- If this is a concern
  - We can do multiple random initializations
    - See if we get the same result - many same results are likely to indicate a global optimum
- Algorithmically we can do this as follows;

    <figure>
    <img src="../../assets/kmeans_random_init.png" alt="" style="width:60%;display:block;margin-left:auto;margin-right:auto;"/>
    <figcaption style="text-align:center"></figcaption>
    </figure>

  - A typical number of times to initialize K-means is 50-1000
  - Randomly initialize K-means

    - For each 100 random initialization run K-means
    - Then compute the distortion on the set of cluster assignments and centroids at convergent
    - End with 100 ways of cluster the data
    - Pick the clustering which gave the lowest distortion

- If you're running K means with 2-10 clusters can help find better global optimum
  - If K is larger than 10, then multiple random initializations are less likely to be necessary
  - First solution is probably good enough (better granularity of clustering)

## Determine optimal k

### Elbow mothod

The technique to determine <u>K, the number of clusters</u>, is called <u>the elbow method</u>. With a bit of fantasy, you can see an elbow in the chart below.

We’ll plot:

- values for K on the horizontal axis
- the distortion on the Y axis (the values calculated with the cost function). This results in:

![img](../../assets/elbow-method.png)

When K increases, the centroids are closer to the clusters centroids. The improvements will decline, at some point rapidly, creating the elbow shape. That point is the optimal value for K. In the image above, K=3.

```python
# clustering dataset
# determine k using elbow method

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

### Another method for choosing K

- Using K-means for market segmentation
- Running K-means for a later/downstream purpose
  - See how well different number of clusters serve you later needs
- e.g.

  - T-shirt size example
    - If you have three sizes (S,M,L)
    - Or five sizes (XS, S, M, L, XL)
    - Run K means where K = 3 and K = 5
  - How does this look

    ![img](../../assets/kmeans_another_chosing_K.png)

  - This gives a way to chose the number of clusters
    - Could consider the cost of making extra sizes vs. how well distributed the products are
    - How important are those sizes though? (e.g. more sizes might make the customers happier)
    - So applied problem may help guide the number of clusters
