# [CORAL Loss](https://arxiv.org/pdf/1901.07884.pdf)

Consistant Rank Ligist for Ordinal Regression

## Network design

After the last fully-connected layer with num of classes as one, a 1D linear bias layer is introduced.

```python
self.fc = nn.Linear(4096, 1, bias=False)
self.linear_1_bias = nn.Parameter(torch.zeros(num_classes-1).float())
```

## Loss function

Let $$W$$ denote the weight parameters of the neural network excluding the bias units of the final layer. The penultimate layer, whose output is denoted as $$g(x_i,W)$$, shares a single weight with all nodes in the final output layer. $$K-1$$ independent bias units are then added to $$g(x_i, W)$$ such that $${g(x_i, W)+b_k}_{k=1}^{K-1}$$ are the inputs to the crresponding binary classifiers in the final layer. Let $$s(z)=1/(1+exp(-z))$$ be the logistic sigmoid function. The predicted empirical probability for task k is defined as:

$$
\hat{P}(y_i^k=1) = s(g(x_i, W) +b_k)
$$

For model training, we minimize the loss function:

$$
L(W,b) = - \sum_{i=1}^N \sum_{k=1}^{K-1} \lambda ^k [log(s(g(x_i,W) +b_k))y_i^k + log(1-s(g(x_i,
W) + b_k))(1-y_i^k)]
$$

which is the weighted cross-entropy of K-1 binary classifiers. For rank prediction, the binary labels are obtained via:

$$
f_k(x_i) = 1{ \hat{P}(y_i^k=1) > 0.5 }
$$

### Example

Let's take a look at the labels, for 7 ranks:

- Cross-Entropy, the one hot encoded label for class 3 is denoted as $$[0,0,1,0,0,0,0]^T$$,
- CROAL-Loss, it's $$[1,1,1,0,0,0 ]^T$$
  ```python
  levels = [[1] * label + [0] * (self.num_classes - 1 - label) for label in batch_y]
  ```

The logits for CORAL-loss looks like this $$[0.9, 0.8, 0.6, 0.4, 0.2, 0.1]^T$$, we find the last num >= 0.5, it's index 3 is our prediction.

During training, the loss for the current sample is calculated as

$$
\begin{aligned}
L = -\sum_{k=1}^{K-1} [1,1,1,0,0,0]^T * log( [0.9,0.8,0.6,0.4,0.2,0.1]^T ) \\
+ (1 - [1,1,1,0,0,0]^T) * log ( [0.9 0.8 0.6 0.4 0.2 0.1]^T ) \\
= - \sum_{k=1}^{K-1} [1,1,1,0,0,0]^T * log( [0.9,0.8,0.6,0.4,0.2,0.1]^T ) \\
+ [0,0,0,1,1,1]^T * log ( [0.9 0.8 0.6 0.4 0.2 0.1]^T ) \\
\end{aligned}
$$

# Ordinal Regression

## Network design

Last fc layer outputs `(num_classes-1)*2` logits.

```python
self.fc = nn.Linear(2048 * block.expansion, (self.num_classes-1)*2)
```

Final output is similar to CORAL-loss:

```python
probas = F.softmax(logits, dim=2)[:, :, 1]
predict_levels = probas > 0.5
predicted_labels = torch.sum(predict_levels, dim=1)
```

## Loss function

```python
def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*levels
                      + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels))*imp, dim=1))
    return torch.mean(val)
```
