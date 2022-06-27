# Binary Cross-Entropy Loss

Also called **Sigmoid Cross-Entropy loss**. It is a **Sigmoid activation** plus a **Cross-Entropy loss**. Unlike **Softmax loss** it is independent for each vector component (class), meaning that the loss computed for every CNN output vector component is not affected by other component values. That’s why it is used for **multi-label classification**, were the insight of an element belonging to a certain class should not influence the decision for another class. It’s called **Binary Cross-Entropy Loss** because it sets up a binary classification problem between $$C'=2$$ classes for every class in $$C$$, as explained above. So when using this Loss, the formulation of **Cross Entroypy Loss** for binary problems is often used:

$$
CE = -\sum_{i=1}^{C'=2}t_{i} log (f(s_{i})) = -t_{1} log(f(s_{1})) - (1 - t_{1}) log(1 - f(s_{1}))
$$

![](<../.gitbook/assets/sigmoid\_CE\_pipeline (1).png>)

This would be the pipeline for each one of the $$C$$ clases. We set $$C$$ independent binary classification problems ($$C'=2$$). Then we sum up the loss over the different binary problems: We sum up the gradients of every binary problem to backpropagate, and the losses to monitor the global loss. $$s_1$$ and $$t_1$$ are the score and the gorundtruth label for the class $$C1$$, which is also the class $$C_i$$ in $$C$$. $$s_2=1-s_1$$ and $$t_2=1-t_1$$ are the score and the ground truth label of the class $$C_2$$, which is not a "class" in our original problem with $$C$$ classes, but a class we create to set up the binary problem with $$C_1=C_i$$. We can understand it as a background class.

The loss can be expressed as:

$$
CE = \left\{\begin{matrix} & - log(f(s_{1})) & & if & t_{1} = 1 \\ & - log(1 - f(s_{1})) & & if & t_{1} = 0 \end{matrix}\right.
$$

Where $$t_1=1$$ means that the class $$C_1=C_i$$ is positive for this sample.

In this case, the activation function does not depend in scores of other classes in $$C$$ more than $$C_1=C_i$$. So the gradient respect to the each score $$s_i$$ in $$s$$ will only depend on the loss given by its binary problem.

The gradient respect to the score $$s_i=s_1$$ can be written as:

$$
\frac{\partial}{\partial s_{i}} \left ( CE(f(s_{i})\right) = t_{1} (f(s_{1}) - 1) + (1 - t_{1}) f(s_{1})
$$

Where $$f()$$ is the **sigmoid** function. It can also be written as:

$$
\frac{\partial}{\partial s_{i}} \left ( CE(f(s_{i})\right) = \begin{Bmatrix} f(s_{i}) - 1 && if & t_{i} = 1\\ f(s_{i}) && if & t_{i} = 0 \end{Bmatrix}
$$

```python
import numpy as np
from sklearn.metrics import log_loss

import tensorflow as tf


def binary_cross_entropy(X, y):
    m = y.shape[0]
    y = y.reshape((m))
    # apply sigmod 1/(1+e^-x)
    fX = 1 / (1 + np.exp(-X))

    # -Y * log(fX) - (1-Y) * (1-log(fX))
    a = -Y * np.log(fX) - (1 - Y) * np.log(1 - fX)
    ce = np.sum(-Y * np.log(fX) - (1 - Y) * np.log(1 - fX)) / m
    return ce


X = np.array([[9.7], [0]])  # (N, 1)
Y = np.array([[0],[1]])  # (N, 1)

print(binary_cross_entropy(X, Y))
```

> Refer [here](https://www.ics.uci.edu/\~pjsadows/notes.pdf) for a detailed loss derivation.

* Caffe: [Sigmoid Cross-Entropy Loss Layer](http://caffe.berkeleyvision.org/tutorial/layers/sigmoidcrossentropyloss.html)
* Pytorch: [BCEWithLogitsLoss](https://pytorch.org/docs/master/nn.html#bcewithlogitsloss)
* TensorFlow: [sigmoid\_cross\_entropy](https://www.tensorflow.org/api\_docs/python/tf/losses/sigmoid\_cross\_entropy).
