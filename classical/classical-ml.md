# Classical Machine Learning

## **Linear Regression**

It attempts to model the relationship between two variables by fitting a linear equation to observed data. One variable is considered to be an **explanatory variable**, and the other is considered to be a **dependent variable**.

- Hypothesis Funtion: $$h(x) = \theta^{T}x$$
- Cost Function: $$J(\theta) = \frac{1}{2}\sum_{i = 1}^{m}(h(x^{(i)}) - y^{(i)})^2$$
- $$\theta$$ - the weight
- m - total number of samples
- To simplify our notation, we introduce the convention of letting $$x_0 = 1$$

### Close Form of Linear Regression

$$\theta = (X^{T}X)^{-1}X^{T}y$$ , which assume $$(X^{T}X)$$ is invertible. Intuition see \[cs299-notes1\]\([\[http://cs229.stanford.edu/notes/cs229-notes1.pdf\]\(http://cs229.stanford.edu/notes/cs229-notes1.pdf\)](https://github.com/ztlevi/Machine_Learning_Questions/tree/26cb30cb7a3ec95f737534585c8ae80567d03d7b/docs/[http:/cs229.stanford.edu/notes/cs229-notes1.pdf]%28http:/cs229.stanford.edu/notes/cs229-notes1.pdf%29)\)

## Logistic Regression

- Hypothesis Funtion: $$h(x) = sigmoid(\theta^{T}x)$$
- Cost Function: $$J(\theta) = \sum_{i = 1}^{m}y^{(i)}log(h(x^{(i)})) + \sum_{i = 1}^{m} (1- y^{(i)})log(1 -h(x^{(i)}))$$
- $$\theta$$ - the weight
- m - total number of samples
- To simplify our notation, we introduce the convention of letting $$x_0 = 1$$

## KNN

- Keywords: Non-parametric Method, Time consuming
- Given a data point, we compute the K nearest data points \(neighbors\) using certain distance metric \(e.g., Euclidean metric\). For classification, we take the majority label of neighbors; for regression, we take the mean of the label values.
- Note for KNN we don't train a model; we simply compute during inference time. This can be computationally expensive since each of the test example need to be compared with every training example to see how close they are.
- There are approximation methods can have faster inference time by partitioning the training data into regions \(e.g., [annoy](https://github.com/spotify/annoy)\)
- When K equals 1 or other small number the model is prone to overfitting \(high variance\), while when K equals number of data points or other large number the model is prone to underfitting \(high bias\)

![KNN](../.gitbook/assets/knn.png)

### Naive Bayes

#### TODO explain formula

- Naive Bayes \(NB\) is a supervised learning algorithm based on applying [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- It is called naive because it builds the naive assumption that each feature are independent of each other
- NB can make different assumptions \(i.e., data distributions, such as Gaussian, Multinomial, Bernoulli\)
- Despite the over-simplified assumptions, NB classifier works quite well in real-world applications, especially for text classification \(e.g., spam filtering\)
- NB can be extremely fast compared to more sophisticated methods

## SVM

Try to find a **optimal hyperplane** to separate two classes of data.

Cost function:

$$min_{\theta}C\sum_{i=1}^m[y^icost_1(\theta^Tx^i)+(1-y^i)cost_0(\theta^Tx^i)] + \frac{1}{2}\sum_{j=1}^n\theta_j^2$$

- Prime Problem:

  $$minimize_{w, w_{0}} \frac{1}{2}w^{T}w$$

  $$s.t. \forall i: y_{i}(w^{T}x_{i} + w_{0}) >= 1$$

  which is a quadratic program with $$d+1$$ variables to be optimized for and $$i$$ constraints.

- KKT condition:

  Orignal Blog from: [http://mypages.iit.edu/~jwang134/posts/KKT-SVs-SVM.html](http://mypages.iit.edu/~jwang134/posts/KKT-SVs-SVM.html)

  - Applying the standard method of Lagrange multipliers, the Lagrangian function is:

    $$J=\frac{1}{2}w^{T}w+C\sum_{i=1}^{n} \varepsilon_{i} + \sum_{i=1}^{n}\alpha_{i}(y_{i}(w^Tx + w_0)-1 + \varepsilon_{i}) - \sum_{i = 1}^{n}\beta_{i} \varepsilon_{i}$$

  - Thus, the corresponding KKT conditions\(Karush–Kuhn–Tucker_conditions\) are as following:

    - $$\frac{\partial J}{\partial w} = w - \sum_{i = 1}^n \alpha_i y_ix_i = 0 => w = \sum_{i = 1}^n \alpha_i y_ix_i$$

      $$\frac{\partial J}{\partial b} = \sum_{i = 1}^n \alpha_i y_i = 0$$

      if is the dataset is not sepratable:

      $$\frac{\partial J}{\partial \varepsilon_{i}} = C - \alpha_i - \beta_i = 0$$

- Dual Problem:

  $$maximize_{\alpha} \sum_{i = 1}^{n}\alpha_{i}-\frac{1}{2}\sum_{i = 1}^{n}\sum_{j = 1}^{n}y_{i}y_{j}\alpha_{i}\alpha_{j}x_{i}^{T}x_{j}$$

  $$s.t. \forall i:\alpha_{i} >= 0 \wedge \sum_{i = 1}^{n}y_{i}\alpha_{i}=0$$

  which is a quadratic program with $$n+1$$ variables to be optimized for and $$n$$ inequality and $$n$$ equality constraints.

- Why use dual problem?\(answer from [here](https://stats.stackexchange.com/questions/19181/why-bother-with-the-dual-problem-when-fitting-svm)\)

  1. Solving the primal problem, we obtain the optimal $$w$$, but **know nothing about the** $$\alpha_{i}$$. In order to classify a query point $$x$$ we need to explicitly compute the scalar product $$w^Tx$$, which may be **expensive** if $$d$$ is large.
  2. Solving the dual problem, we obtain the $$\alpha_{i}$$ \(where $$\alpha_{i} = 0$$ for all but a few points - the support vectors\). In order to classify a query point $$x$$, we calculate:

     $$w^{T}x + w^{0} = (\sum_{i = 1}^{n}\alpha_{i}y_{i}x_{i})^{T}x + w_{0}= \sum_{i = 1}^{n}\alpha_{i}y_{i}\langle\,x_{i},x\rangle + w_{0}$$

     This term is very **efficiently calculated** if there are only few support vectors. Further, since we now have a scalar product only involving _data_ vectors, we may **apply the kernel trick**.

- Can perform linear, nonlinear, or outlier detection \(unsupervised\) depending on the kernel funciton
- Large margin classifier: using SVM we not only have a decision boundary, but want the boundary to be as far from the closest training point as possible
- \(Optional\): Why Large margin classifier? Let's say a linear svm. If you take a look at the cost function, in order to minimize the cost, the inner product of $$\theta^Tx$$ need to be greater than 1 or less than -1. In this case, if $$\theta$$ is not the perfect decision boundary, it will have larger cost.
- The closest training examples are called support vectors, since they are the points based on which the decision boundary is drawn
- SVMs are sensitive to feature scaling
- If C is very large, SVM is very sensitive to outliers.But if C is reasonably small, or a not too large, then you stick with the decision boundary more robust with outliers.

![svm](../.gitbook/assets/svm.png)

## Decision tree

![Image result for Decision tree image](../.gitbook/assets/B03905_05_01-compressor.png)

- Non-parametric, supervised learning algorithms
- Given the training data, a decision tree algorithm **divides the feature space into regions**. For inference, we first see which region does the test data point fall in, and take the mean label values \(regression\) or the majority label value \(classification\).
- **Construction**: top-down, chooses a question to split the data such that the target variables within each region are as homogeneous as possible. Calculate the gini impurity and information gain, then pick the question with the most information gain.
- Advantage:
  - simply to understand & interpret, mirrors human decision making
- Disadvantage:
  - can overfit easily \(and generalize poorly\) if we don't limit the depth of the tree
  - can be non-robust: A small change in the training data can lead to a totally different tree
  - instability: sensitive to training set rotation due to its orthogonal decision boundaries
- How to choose depth:
  - It depends on the possible questions you can have for certain problems.
  - Also you need to use validation data to see which depth works best

## Random forests

![Image result for Random forest](../.gitbook/assets/Architecture-of-the-random-forest-model.png)

An ensemble learning method for classification, regression and other tasks that operates by constructing a **multitude of decision trees** at training time and outputting the class that is the **mode** of the classes \(classification\) or **mean** prediction \(regression\) of the individual trees.

- The number of tree? In general, the more trees you use the better get the results. However, the improvement decreases as the number of trees increases, at a certain point **the benefit in prediction performance from learning more trees will be lower than the cost in computation time for learning these additional trees**.

### Bagging in Random Forest

**1**. Suppose there are $$N$$ observations and M features in training data set. First, a sample from training data set is taken randomly with replacement. \(**Bagging for Dataset**\)

**2**. A subset of $$M$$ features\($$\sqrt{M}$$\) are selected randomly and whichever feature gives the best split is used to split the node iteratively. \(**Bagging for Feature**\)

**3**. The tree is grown to the largest.

**4**. Above steps are repeated and prediction is given based on the aggregation of predictions from $$n$$ number of trees.
