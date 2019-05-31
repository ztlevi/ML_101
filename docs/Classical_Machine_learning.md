<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->

**Table of Contents**

- [Classical Machine learning](#classical-machine-learning)
  - [KNN](#knn)
    - [Naive Bayes](#naive-bayes)
      - [TODO explain formula](#todo-explain-formula)
  - [SVM](#svm)
  - [Decision tree](#decision-tree)

<!-- markdown-toc end -->

# Classical Machine learning

## KNN

- Keywords: Non-parametric Method, Time consuming
- Given a data point, we compute the K nearest data points (neighbors) using certain distance metric (e.g., Euclidean metric). For classification, we take the majority label of neighbors; for regression, we take the mean of the label values.
- Note for KNN we don't train a model; we simply compute during inference time. This can be computationally expensive since each of the test example need to be compared with every training example to see how close they are.
- There are approximation methods can have faster inference time by partitioning the training data into regions (e.g., [annoy](https://github.com/spotify/annoy))
- When K equals 1 or other small number the model is prone to overfitting (high variance), while when K equals number of data points or other large number the model is prone to underfitting (high bias)

![KNN](../assets/knn.png)

### Naive Bayes

#### TODO explain formula

- Naive Bayes (NB) is a supervised learning algorithm based on applying [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- It is called naive because it builds the naive assumption that each feature are independent of each other
- NB can make different assumptions (i.e., data distributions, such as Gaussian, Multinomial, Bernoulli)
- Despite the over-simplified assumptions, NB classifier works quite well in real-world applications, especially for text classification (e.g., spam filtering)
- NB can be extremely fast compared to more sophisticated methods

## SVM

Try to find a **optimal hyperplane** to separate two classes of data.

Cost function: $$ min_{\theta}C\sum_{i=1}^m[y^icost_1(\theta^Tx^i)+(1-y^i)cost_0(\theta^Tx^i)] + \frac{1}{2}\sum_{j=1}^n\theta_j^2 $$

- Can perform linear, nonlinear, or outlier detection (unsupervised) depending on the kernel funciton
- Large margin classifier: using SVM we not only have a decision boundary, but want the boundary to be as far from the closest training point as possible

- (Optional): Why Large margin classifier? Let's say a linear svm. If you take a look at the cost function, in order to minimize the cost, the inner product of $$\theta^Tx$$ need to be greater than 1 or less than -1. In this case, if $$\theta$$ is not the perfect decision boundary, it will have larger cost.
- The closest training examples are called support vectors, since they are the points based on which the decision boundary is drawn
- SVMs are sensitive to feature scaling
- If C is very large, SVM is very sensitive to outliers.But if C is reasonably small, or a not too large, then you stick with the decision boundary more robust with outliers.

![svm](../assets/svm.png)

## Decision tree

- Non-parametric, supervised learning algorithms
- Given the training data, a decision tree algorithm divides the feature space into regions. For inference, we first see which region does the test data point fall in, and take the mean label values (regression) or the majority label value (classification).
- **Construction**: top-down, chooses a question to split the data such that the target variables within each region are as homogeneous as possible. Calculate the gini impurity and information gain, then pick the question with the most information gain.
- Advantage: simply to understand & interpret, mirrors human decision making
- Disadvantage:
  - can overfit easily (and generalize poorly) if we don't limit the depth of the tree
  - can be non-robust: A small change in the training data can lead to a totally different tree
  - instability: sensitive to training set rotation due to its orthogonal decision boundaries
