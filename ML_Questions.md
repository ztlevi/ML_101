<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->

**Table of Contents**

- [General](#general)
  - [Project Workflow](#project-workflow)
  - [Activation Function](#activation-function)
  - [Cost function](#cost-function)
    - [L1, L2](#l1-l2)
    - [cross-entropy](#cross-entropy)
  - [Pooling](#pooling)
  - [How to Prevent Overfitting](#how-to-prevent-overfitting)
    - [Cross-validation](#cross-validation)
    - [Train with more data](#train-with-more-data)
    - [Remove features](#remove-features)
    - [Early stopping](#early-stopping)
    - [Regularization](#regularization)
      - [DropOut (Regularization technique)](#dropout-regularization-technique)
      - [DropConnect](#dropconnect)
      - [Comparison](#comparison)
    - [Ensembling](#ensembling)
  - [Clustering - K-means](#clustering---k-means)
  - [Principal Component Analysis](#principal-component-analysis)
  - [Non maximal supression](#non-maximal-supression)
  - [Blur image](#blur-image)
- [Supervised Learning](#supervised-learning)
  - [KNN](#knn)
  - [SVM](#svm)
  - [Decision tree](#decision-tree)
- [Neural Network](#neural-network)
  - [GEMM](#gemm)
  - [Model compression](#model-compression)
    - [Weight Pruning](#weight-pruning)
  - [Optimization](#optimization)
    - [Gradient Descent](#gradient-descent)
      - [derivative of a matrix-matrix product](#derivative-of-a-matrix-matrix-product)
    - [Backpropagation](#backpropagation)
      - [Chain Rule](#chain-rule)
      - [Applying the chain rule](#applying-the-chain-rule)
    - [Batch gradient descent](#batch-gradient-descent)
    - [Stochastic gradient descent](#stochastic-gradient-descent)
    - [Mini-batch gradient descent](#mini-batch-gradient-descent)
    - [SGD momentum](#sgd-momentum)
    - [ADAM](#adam)
  - [Models](#models)
    - [CNN](#cnn)
    - [RNN and LSTM](#rnn-and-lstm)
    - [Resnet](#resnet)
    - [Mobilenet v1](#mobilenet-v1)
- [Two MTL methods for Deep Learning](#two-mtl-methods-for-deep-learning)
  - [Hard parameter sharing](#hard-parameter-sharing)
  - [Soft parameter sharing](#soft-parameter-sharing)
    - [Yolo](#yolo)
      - [YOLO loss](#yolo-loss)

<!-- markdown-toc end -->

# General

## Project Workflow

Given a data science / machine learning project, what steps should we follow? Here's how I would tackle it:

- **Specify business objective.** Are we trying to win more customers, achieve higher satisfaction, or gain more revenues?
- **Define problem.** What is the specific gap in your ideal world and the real one that requires machine learning to fill? Ask questions that can be addressed using your data and predictive modeling (ML algorithms).
- **Create a common sense baseline.** But before you resort to ML, set up a baseline to solve the problem as if you know zero data science. You may be amazed at how effective this baseline is. It can be as simple as recommending the top N popular items or other rule-based logic. This baseline can also server as a good benchmark for ML algorithms.
- **Review ML literatures.** To avoid reinventing the wheel and get inspired on what techniques / algorithms are good at addressing the questions using our data.
- **Set up a single-number metric.** What it means to be successful - high accuracy, lower error, or bigger AUC - and how do you measure it? The metric has to align with high-level goals, most often the success of your business. Set up a single-number against which all models are measured.
- **Do exploratory data analysis (EDA).** Play with the data to get a general idea of data type, distribution, variable correlation, facets etc. This step would involve a lot of plotting.
- **Partition data.** Validation set should be large enough to detect differences between the models you are training; test set should be large enough to indicate the overall performance of the final model; training set, needless to say, the larger the merrier.
- **Preprocess.** This would include data integration, cleaning, transformation, reduction, discretization and more.
- **Engineer features.** Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering. This step usually involves feature selection and creation, using domain knowledge. Can be minimal for deep learning projects.
- **Develop models.** Choose which algorithm to use, what hyperparameters to tune, which architecture to use etc.
- **Ensemble.** Ensemble can usually boost performance, depending on the correlations of the models/features. So it’s always a good idea to try out. But be open-minded about making tradeoff - some ensemble are too complex/slow to put into production.
- **Deploy model.** Deploy models into production for inference.
- **Monitor model.** Monitor model performance, and collect feedbacks.
- **Iterate.** Iterate the previous steps. Data science tends to be an iterative process, with new and improved models being developed over time.

![](assets/workflow.png)

## Activation Function

the activation function is usually an abstraction representing the rate of action potential firing in the cell. In its simplest form, this function is binary—that is, either the neuron is firing or not.

**Important**: The most important meaning add activation function is by adding the activation funciton, we are adding non-linearity to the model.

For neural networks

- Non-linearity: ReLU is often used. Use Leaky ReLU (a small positive gradient for negative input, say, `y = 0.01x` when x < 0) to address dead ReLU issue
- Multi-class: softmax
- Binary: sigmoid
- Regression: linear

## Cost function

A Loss Functions tells us “how good” our model is at making predictions for a given set of parameters. The cost function has its own curve and its own gradients. The slope of this curve tells us how to update our parameters to make the model more accurate.

### L1, L2

### cross-entropy

If M>2 (i.e. multiclass classification), we calculate a separate loss for each class label per observation and sum the result. Usually an activation function (Sigmoid / Softmax) is applied to the scores before the CE Loss computation.

$
-\sum_{c=1}^M y_{o,c}log(p_{o,c})
$

> Note:
>
> 1.  M - number of classes (dog, cat, fish)
> 2.  log - the natural log
> 3.  y - binary indicator (0 or 1) if class label c is the correct classification for observation o
> 4.  p - predicted probability observation o is of class c

## Pooling

1. Pooling layers control the number of features the CNN model is learning and it avoids over fitting.
2. There are 2 different types of pooling layers - MAX pooling layer and AVG pooling layer. As the names suggest the MAX pooling layer picks maximum values from the convoluted feature maps and AVG pooling layer takes the average value of the features from the feature maps.

## How to Prevent Overfitting

Detecting overfitting is useful, but it doesn’t solve the problem. Fortunately, you have several options to try.

Here are a few of the most popular solutions for overfitting:

### Cross-validation

Cross-validation is a powerful preventative measure against overfitting.

The idea is clever: Use your initial training data to generate multiple mini train-test splits. Use these splits to tune your model.

In standard k-fold cross-validation, we partition the data into k subsets, called folds. Then, we iteratively train the algorithm on k-1 folds while using the remaining fold as the test set (called the “holdout fold”).

![K-Fold Cross-Validation](https://elitedatascience.com/wp-content/uploads/2017/06/Cross-Validation-Diagram.jpg)

K-Fold Cross-Validation

Cross-validation allows you to tune hyperparameters with only your original training set. This allows you to keep your test set as a truly unseen dataset for selecting your final model.

We have another article with a [more detailed breakdown of cross-validation](https://elitedatascience.com/machine-learning-iteration#micro).

### Train with more data

It won’t work every time, but training with more data can help algorithms detect the signal better. In the earlier example of modeling height vs. age in children, it’s clear how sampling more schools will help your model.

Of course, that’s not always the case. If we just add more noisy data, this technique won’t help. That’s why you should always ensure your data is clean and relevant.

### Remove features

Some algorithms have built-in feature selection.

For those that don’t, you can manually improve their generalizability by removing irrelevant input features.

An interesting way to do so is to tell a story about how each feature fits into the model. This is like the data scientist's spin on software engineer’s [rubber duck debugging](https://en.wikipedia.org/wiki/Rubber_duck_debugging) technique, where they debug their code by explaining it, line-by-line, to a rubber duck.

If anything doesn't make sense, or if it’s hard to justify certain features, this is a good way to identify them.

In addition, there are several [feature selection heuristics](https://elitedatascience.com/dimensionality-reduction-algorithms#feature-selection) you can use for a good starting point.

### Early stopping

When you’re [training a learning algorithm iteratively](https://elitedatascience.com/machine-learning-iteration#model), you can measure how well each iteration of the model performs.

Up until a certain number of iterations, new iterations improve the model. After that point, however, the model’s ability to generalize can weaken as it begins to overfit the training data.

Early stopping refers stopping the training process before the learner passes that point.

![Early stopping graphic](https://elitedatascience.com/wp-content/uploads/2017/09/early-stopping-graphic.jpg)

Today, this technique is mostly used in deep learning while other techniques (e.g. regularization) are preferred for classical machine learning.

### Regularization

Regularization refers to a broad range of techniques for artificially forcing your model to be simpler.

The method will depend on the type of learner you’re using. For example, you could prune a decision tree, use dropout on a neural network, or add a penalty parameter to the cost function in regression.

Oftentimes, the regularization method is a hyperparameter as well, which means it can be tuned through cross-validation.

We have a more detailed discussion here on [algorithms and regularization methods](http://elitedatascience.com/machine-learning-algorithms).

#### DropOut (Regularization technique)

To apply DropOut, we randomly select a subset of the units and clamp their output to zero, regardless of the input; this effectively removes those units from the model. A different subset of units is randomly selected every time we present a training example.

Below are two possible network configurations. On the first presentation (left), the 1st and 3rd units are disabled, but the 2nd and 3rd units have been randomly selected on a subsequent presentation. At test time, we use the complete network but rescale the weights to compensate for the fact that all of them can now become active (e.g., if you drop half of the nodes, the weights should also be halved).

[![DropOut examples](https://i.stack.imgur.com/CewjH.png)](https://i.stack.imgur.com/CewjH.png)

#### DropConnect

DropConnect works similarly, except that we disable individual weights (i.e., set them to zero), instead of nodes, so a node can remain partially active. Schematically, it looks like this:

[![DropConnect](https://i.stack.imgur.com/D1QC7.png)](https://i.stack.imgur.com/D1QC7.png)

#### Comparison

These methods both work because they effectively let you train several models at the same time, then average across them for testing. For example, the yellow layer has four nodes, and thus 16 possible DropOut states (all enabled, #1 disabled, #1 and #2 disabled, etc).

DropConnect is a generalization of DropOut because it produces even more possible models, since there are almost always more connections than units. However, you can get similar outcomes on an individual trial. For example, the DropConnect network on the right has effectively dropped Unit #2 since all of the incoming connections have been removed.

### Ensembling

Ensembles are machine learning methods for combining predictions from multiple separate models. There are a few different methods for ensembling, but the two most common are:

_Bagging_ attempts to reduce the chance overfitting complex models.

- It trains a large number of "strong" learners in parallel.
- A strong learner is a model that's relatively unconstrained.
- Bagging then combines all the strong learners together in order to "smooth out" their predictions.

_Boosting_ attempts to improve the predictive flexibility of simple models.

- It trains a large number of "weak" learners in sequence.
- A weak learner is a constrained model (i.e. you could limit the max depth of each decision tree).
- Each one in the sequence focuses on learning from the mistakes of the one before it.
- Boosting then combines all the weak learners into a single strong learner.

While bagging and boosting are both ensemble methods, they approach the problem from opposite directions.

Bagging uses complex base models and tries to "smooth out" their predictions, while boosting uses simple base models and tries to "boost" their aggregate complexity.

## Clustering - K-means

- Clustering is a unsupervised learning algorithm that groups data in such a way that data points in the same group are more similar to each other than to those from other groups
- Similarity is usually defined using a distance measure (e.g, Euclidean, Cosine, Jaccard, etc.)
- The goal is usually to discover the underlying structure within the data (usually high dimensional)
- The most common clustering algorithm is K-means, where we define K (the number of clusters) and the algorithm iteratively finds the cluster each data point belongs to

[scikit-learn](http://scikit-learn.org/stable/modules/clustering.html) implements many clustering algorithms. Below is a comparison adopted from its page.

K-means algorithm:

- Input:
  - $K$ (number of clusters)
  - Training set ${x^1,x^2,...,x^m}$ ($x^i \in \mathbb{R}^n$)
- Algorithm:

  1. Randomly initialized $K$ cluster centroids $\mu_1,\mu_2,...,\mu_K \in \mathbb{R}^n$
  2. Repeat {

     - for i = 1 to $m$
       - $c^i$ := index (from 1 to K) of cluster centroid closest to $x^i$
     - for k = 1 to $K$
       - $\mu_k$ := average (mean) of points assigned to Cluster k

     }

[kmeans python implementation](./kmeans/kmeans.py)

![clustering](assets/clustering.png)

## Principal Component Analysis

- Principal Component Analysis (PCA) is a dimension reduction technique that projects the data into a lower dimensional space
- PCA uses Singular Value Decomposition (SVD), which is a matrix factorization method that decomposes a matrix into three smaller matrices (more details of SVD [here](https://en.wikipedia.org/wiki/Singular-value_decomposition))
- PCA finds top N principal components, which are dimensions along which the data vary (spread out) the most. Intuitively, the more spread out the data along a specific dimension, the more information is contained, thus the more important this dimension is for the pattern recognition of the dataset
- PCA can be used as pre-step for data visualization: reducing high dimensional data into 2D or 3D. An alternative dimensionality reduction technique is [t-SNE](https://lvdmaaten.github.io/tsne/)

Here is a visual explanation of PCA

![pca](assets/pca.gif)

## Non maximal supression

[NMS](./NMS/nms.py)

[NMS_Slow](./NMS/nms_slow.py)

## Blur image

# Supervised Learning

## KNN

- Given a data point, we compute the K nearest data points (neighbors) using certain distance metric (e.g., Euclidean metric). For classification, we take the majority label of neighbors; for regression, we take the mean of the label values.
- Note for KNN we don't train a model; we simply compute during inference time. This can be computationally expensive since each of the test example need to be compared with every training example to see how close they are.
- There are approximation methods can have faster inference time by partitioning the training data into regions (e.g., [annoy](https://github.com/spotify/annoy))
- When K equals 1 or other small number the model is prone to overfitting (high variance), while when K equals number of data points or other large number the model is prone to underfitting (high bias)

![KNN](assets/knn.png)

## SVM

Cost function: $
min_{\theta}C\sum_{i=1}^m[y^icost_1(\theta^Tx^i)+(1-y^i)cost_0(\theta^Tx^i)] + \frac{1}{2}\sum_{j=1}^n\theta_j^2
$

- Can perform linear, nonlinear, or outlier detection (unsupervised)
- Large margin classifier: using SVM we not only have a decision boundary, but want the boundary to be as far from the closest training point as possible

- (Optional): Why Large margin classifier? Let's say a linear svm. If you take a look at the cost function, in order to minimize the cost, the inner product of $\theta^Tx$ need to be greater than 1 or less than -1. In this case, if $\theta$ is not the perfect decision boundary, it will have larger cost.
- The closest training examples are called support vectors, since they are the points based on which the decision boundary is drawn
- SVMs are sensitive to feature scaling
- If C is very large, SVM is very sensitive to outliers.But if C is reasonably small, or a not too large, then you stick with the decision boundary more robust with outliers.

![svm](assets/svm.png)

[back to top](#data-science-question-answer)

## Decision tree

- Non-parametric, supervised learning algorithms
- Given the training data, a decision tree algorithm divides the feature space into regions. For inference, we first see which region does the test data point fall in, and take the mean label values (regression) or the majority label value (classification).
- **Construction**: top-down, chooses a question to split the data such that the target variables within each region are as homogeneous as possible. Calculate the gini impurity and information gain, then pick the question with the most information gain.
- Advantage: simply to understand & interpret, mirrors human decision making
- Disadvantage:
  - can overfit easily (and generalize poorly) if we don't limit the depth of the tree
  - can be non-robust: A small change in the training data can lead to a totally different tree
  - instability: sensitive to training set rotation due to its orthogonal decision boundaries

# Neural Network

## GEMM

## Model compression

### Weight Pruning

**Getting faster/smaller networks is important for running these deep learning networks on mobile devices.**

The ranking can be done according to the L1/L2 mean of neuron weights, their mean activations, the number of times a neuron wasn’t zero on some validation set, and other creative methods . After the pruning, the accuracy will drop (hopefully not too much if the ranking clever), and the network is usually trained more to recover.

If we prune too much at once, the network might be damaged so much it won’t be able to recover.

So in practice this is an iterative process - often called ‘Iterative Pruning’: Prune / Train / Repeat.

![Pruning steps](http://jacobgil.github.io/assets/pruning_steps.png)

## Optimization

### Gradient Descent

Let's say your hypothesis function contains multiple parameters, defined as $\theta_1$, $\theta_2$, $\theta_3$.

Your cost function takes both your model output and your ground truth, e.g.:

$
J(\theta_0, \theta_1,...,\theta_n) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^i) - y^i)^2
$

Then we need to calculate the partial deriative of the cost function with respect to the each $\theta$, from 0 to n.

Then we can simultaneously update the $\theta_j = \theta_j - \alpha\frac{}{\theta_j}J(\theta_0, \theta_1,...,\theta_n)$

#### derivative of a matrix-matrix product

$
D = W \cdot X
dW = dD \cdot X^T
dX = W^T \cdot dD
$

### Backpropagation

https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html

A neural network propagates the signal of the input data forward through its parameters towards the moment of decision, and then backpropagates information about the error through the network so that it can alter the parameters one step at a time.

#### Chain Rule

As seen above, foward propagation can be viewed as a long series of nested equations. If you think of feed forward this way, then backpropagation is merely an application the Chain rule to find the Derivatives of cost with respect to any variable in the nested equation. Given a forward propagation function:

$
f(x) = A(B(C(x)))
$

A, B, and C are activation functions at different layers. Using the chain rule we easily calculate the derivative of $f(x)$ with respect to $x$:

$
f'(x) = f'(A) \cdot A'(B) \cdot B'(C) \cdot C'(x)
$

How about the derivative with respect to B? To find the derivative with respect to B you can pretend $B(C(x))$ is a constant, replace it with a placeholder variable B, and proceed to find the derivative normally with respect to B.

$
f'(B) = f'(A) \cdot A'(B)
$

This simple technique extends to any variable within a function and allows us to precisely pinpoint the exact impact each variable has on the total output.

#### Applying the chain rule

Let’s use the chain rule to calculate the derivative of cost with respect to any weight in the network. The chain rule will help us identify how much each weight contributes to our overall error and the direction to update each weight to reduce our error. Here are the equations we need to make a prediction and calculate total error, or cost:

![_images/backprop_ff_equations.png](https://ml-cheatsheet.readthedocs.io/en/latest/_images/backprop_ff_equations.png)

Given a network consisting of a single neuron, total cost could be calculated as:

$
Cost = C(R(Z(XW)))
$

Using the chain rule we can easily find the derivative of Cost with respect to weight W.

$
C'(W) = C'(R) \cdot R'(Z) \cdot Z'(W)
= (\hat{y} - y) \cdot R'(Z) \cdot X
$

### Batch gradient descent

Vanilla gradient descent, aka batch gradient descent, computes the gradient of the cost function to the parameters $\theta$ for the entire training dataset.

```python
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
```

### Stochastic gradient descent

Stochastic gradient descent (SGD) in contrast performs a parameter update for each training example $x^i$ and label $y^i$. Note that we shuffle the training data at every epoch.

SGD performs frequent updates with a high variance that cause the objective function to fluctuate heavily as in Image 1.

![img](http://ruder.io/content/images/2016/09/sgd_fluctuation.png)

```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```

### Mini-batch gradient descent

Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of n training examples. Common mini-batch sizes range between 50 and 256, but can vary for different applications.

```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

### SGD momentum

Momentum is a mehod that helps accelerate SGD in the relevant direction and dampends oscillations seen in the image below.

<figure style="width:100%;display:block;margin-left:auto;margin-right:auto;">
<img src="https://raw.githubusercontent.com/ztlevi/picee_images/master/common/image.s3kghjrlof.png" alt=""/>
<figcaption></figcaption>
</figure>
$
m_t = \gamma m_{t-1}+\eta\nabla_{\theta}J(\theta)$

$\theta=\theta - m_t
$

Essentially, when using momentum, we push a ball down a hill. The ballaccumulates momentum as it rolls downhill, becoming faster and faster onthe way (until it reaches its terminal velocity if there is air resistance, i.e. $\gamma<1​$). The same thing happens to our parameter updates: The momentumterm increases for dimensions whose gradients point in the same directionsand reduces updates for dimensions whose gradients change directions. Asa result, we gain faster convergence and reduced oscillation.

### ADAM

Adaptive Moment Estimation (ADAM) is a method that computes adaptive learning rate for each parameter. In addition to storing an exponentially decaying averagae of past squared gradients \$v_t like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients $m_t$, similar to momentum. Whereas momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction, which thus prefers flat minima in the error surface.

$m_t$ and $v_t$ are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively.

$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$

$
v_t = \beta_2 v_{t-1} + (1-\beta_2) v_t^2
$

Compute bias-corrected first moment estimate and bias-corrected second raw moment estimate.

$
\hat{m_t} = \frac{m_t}{1-\beta_1^t}
$

$
\hat{v_t} = \frac{v_t}{1-\beta_2^t}
$

$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v_t}} + \epsilon } \hat{m_t}
$

The authors propose default values of 0.9 for $\beta_1$, 0.999 for $\beta_2$, and $10^{-8}$ for $\epsilon$.

## Models

### CNN

The Conv layer is the building block of a Convolutional Network. The Conv layer consists of a set of learnable filters (such as 5 _ 5 _ 3, width _ height _ depth). During the forward pass, we slide (or more precisely, convolve) the filter across the input and compute the dot product. Learning happens when the network back propagate the error layer by layer.

Initial layers capture low-level features such as angle and edges, while later layers learn a combination of the low-level features and in the previous layers and can therefore represent higher level feature, such as shape and object parts.

![CNN](assets/cnn.jpg)

### RNN and LSTM

https://colah.github.io/posts/2015-08-Understanding-LSTMs/

RNN is another paradigm of neural network where we have difference layers of cells, and each cell not only takes the cell from the previous layer as input, but also the previous cell within the same layer. This gives RNN the power to model sequence.

![RNN](assets/rnn.jpeg)

A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens if we unroll the loop:

![An unrolled recurrent neural network.](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)

This seems great, but in practice RNN barely works due to exploding/vanishing gradient, which is cause by a series of multiplication of the same matrix. To solve this, we can use a variation of RNN, called long short-term memory (LSTM), which is capable of learning long-term dependencies.

The math behind LSTM can be pretty complicated, but intuitively LSTM introduce

- input gate
- output gate
- forget gate
- memory cell (internal state)

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

LSTM resembles human memory: it forgets old stuff (old internal state _ forget gate) and learns from new input (input node _ input gate)

![lstm](assets/lstm.png)

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![A LSTM neural network.](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

### Resnet

Increasing network depth does not work by simply stacking layers together. Deep networks are hard to train because of the notorious vanishing gradient problem — as the gradient is back-propagated to earlier layers, repeated multiplication may make the gradient infinitively small. As a result, as the network goes deeper, its performance gets saturated or even starts degrading rapidly.

The core idea of ResNet is introducing a so-called shortcut.

- When the input and output are of the same dimensions, we use “identity shortcut connection” that skips one or more layers, as shown in the following figure:

  ![img](https://cdn-images-1.medium.com/max/1500/1*ByrVJspW-TefwlH7OLxNkg.png)

- When the dimensions increase, we consider two options: (A) THe shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. (B) The projection shortcut is used to match dimensions.

The authors argues that stacking layers shouldn’t degrade the network performance, because we could simply stack identity mappings (layer that doesn’t do anything) upon the current network, and the resulting architecture would perform the same. This indicates that the deeper model should not produce a training error higher than its shallower counterparts.

### Mobilenet v1

Standard convolutions have the computational cost of :

$
D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F
$

where the computational cost depends multiplicatively onthe number of input channels M, the number of output channe is N, the kernel size $D_K \cdot D_K$ and the feature map size $D_F \cdot D_F$.

<figure style="width:50%;  display: block; margin-left: auto; margin-right: auto;">
<img src="https://raw.githubusercontent.com/ztlevi/picee_images/master/common/image.w1oy7a5qkr.png" alt=""/>
<figcaption></figcaption>
</figure>

Depthwise convolution is extremely efficient relative to standard convolution. However it only filters input channels, it does not combine them to create new features. So an additional layer that computes a linear combination ofthe output of depthwise convolution via $1 \times 1$ convolutionis needed in order to generate these new features.

The combination of depthwise convolution and $1 \times 1$ (pointwise) convolution is called depthwise separable con-volution.

Depthwise separable convolutions cost:

$
D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + \cdot M \cdot N \cdot D_F \cdot D_F
$

# Two MTL methods for Deep Learning

So far, we have focused on theoretical motivations for MTL. To make the ideas of MTL more concrete, we will now look at the two most commonly used ways to perform multi-task learning in deep neural networks. In the context of Deep Learning, multi-task learning is typically done with either _hard_ or _soft parameter sharing_ of hidden layers.

## Hard parameter sharing

Hard parameter sharing is the most commonly used approach to MTL in neural networks and goes back to <sup class="footnote-ref">[[6]](http://ruder.io/multi-task/index.html#fn6)</sup>. It is generally applied by sharing the hidden layers between all tasks, while keeping several task-specific output layers.

<figure>![](http://ruder.io/content/images/2017/05/mtl_images-001-2.png "Hard parameter sharing")

<figcaption>Figure 1: Hard parameter sharing for multi-task learning in deep neural networks</figcaption></figure>

Hard parameter sharing greatly reduces the risk of overfitting. In fact, <sup class="footnote-ref">[[7]](http://ruder.io/multi-task/index.html#fn7)</sup> showed that the risk of overfitting the shared parameters is an order N -- where N is the number of tasks -- smaller than overfitting the task-specific parameters, i.e. the output layers. This makes sense intuitively: The more tasks we are learning simultaneously, the more our model has to find a representation that captures all of the tasks and the less is our chance of overfitting on our original task.

## Soft parameter sharing

In soft parameter sharing on the other hand, each task has its own model with its own parameters. The distance between the parameters of the model is then regularized in order to encourage the parameters to be similar. <sup class="footnote-ref">[[8]](http://ruder.io/multi-task/index.html#fn8)</sup> for instance use the ℓ2<math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>ℓ</mi><mn>2</mn></msub></math> norm for regularization, while <sup class="footnote-ref">[[9]](http://ruder.io/multi-task/index.html#fn9)</sup> use the trace norm.

<figure>![](http://ruder.io/content/images/2017/05/mtl_images-002-1.png "Soft parameter sharing")

<figcaption>Figure 2: Soft parameter sharing for multi-task learning in deep neural networks</figcaption></figure>

The constraints used for soft parameter sharing in deep neural networks have been greatly inspired by regularization techniques for MTL that have been developed for other models, which we will soon discuss.

### Yolo

#### YOLO loss

- the classification loss.
- the localization loss (errors between the predicted boundary box and the ground truth).
- the confidence loss (the objectness of the box).
