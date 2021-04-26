[Refreence](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

## Cross-Entropy loss

The **Cross-Entropy Loss** is actually the only loss we are discussing here. The other losses names written in the title are other names or variations of it. The CE Loss is defined as:

$$
CE = -\sum_{i}^{C}t_{i} log (s_{i})
$$

Where $$t_i$$ and $$s_i$$ are the ground truth and the CNN score for each $$class_i$$ in $$C$$. As **usually an activation function \(Sigmoid / Softmax\) is applied to the scores before the CE Loss computation**, we write $$f(si)$$ to refer to the activations.

In a **binary classification problem**, where $$C'=2$$, the Cross Entropy Loss can be defined also as [\[discussion\]](https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks):

$$
CE = -\sum_{i=1}^{C'=2}t_{i} log (s_{i}) = -t_{1} log(s_{1}) - (1 - t_{1}) log(1 - s_{1})
$$

Where it’s assumed that there are two classes: $$C_1$$ and $$C_2$$. $$t_1$$ \[0,1\] and $$s_1$$ are the ground truth and the score for $$C_1$$, and $$t_2=1-t_1$$ and $$s_2=1-s_1$$ are the ground truth and the score for $$C_2$$. That is the case when we split a Multi-Label classification problem in $$C$$ binary classification problems. See next Binary Cross-Entropy Loss section for more details.

**Logistic Loss** and **Multinomial Logistic Loss** are other names for **Cross-Entropy loss**. [\[Discussion\]](https://stats.stackexchange.com/questions/166958/multinomial-logistic-loss-vs-cross-entropy-vs-square-error/172790)

```python
def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)


def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce


predictions = np.array([[0.25, 0.25, 0.25, 0.25], [0.01, 0.01, 0.01, 0.97]]) # (N, num_classes)
targets = np.array([[1, 0, 0, 0], [0, 0, 0, 1]]) # (N, num_classes)

cross_entropy(predictions, targets)
# 0.7083767843022996

log_loss(targets, predictions)
# 0.7083767843022996

log_loss(targets, predictions) == cross_entropy(predictions, targets)
# True
```

The layers of Caffe, Pytorch and Tensorflow than use a Cross-Entropy loss without an embedded activation function are:

- Caffe: [Multinomial Logistic Loss Layer](http://caffe.berkeleyvision.org/tutorial/layers/multinomiallogisticloss.html). Is limited to multi-class classification \(does not support multiple labels\).
- Pytorch: [BCELoss](https://pytorch.org/docs/master/nn.html#bceloss). Is limited to binary classification \(between two classes\).
- TensorFlow: [log_loss](https://www.tensorflow.org/api_docs/python/tf/losses/log_loss).
