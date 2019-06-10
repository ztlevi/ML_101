# Categorical Cross-Entropy loss

Also called **Softmax Loss**. It is a **Softmax activation** plus a **Cross-Entropy loss**. If we use this loss, we will train a CNN to output a probability over the $$C$$ classes for each image. It is used for multi-class classification.

<figure>
<img src="../../../assets/softmax_CE_pipeline.png" alt="" style="width:60%;display:block;margin-left:auto;margin-right:auto;"/>
<figcaption style="text-align:center"></figcaption>
</figure>

In the specific (and usual) case of Multi-Class classification the labels are one-hot, so only the positive class $$C_p$$ keeps its term in the loss. There is only one element of the Target vector $$t$$ which is not zero $$t_i=t_p$$. So discarding the elements of the summation which are zero due to target labels, we can write:

$$
CE = -log\left ( \frac{e^{s_{p}}}{\sum_{j}^{C} e^{s_{j}}} \right )
$$

Where **Sp** is the CNN score for the positive class.

Defined the loss, now we’ll have to compute its **gradient respect to the output neurons** of the CNN in order to backpropagate it through the net and optimize the defined loss function tuning the net parameters. So we need to compute the gradient of CE Loss respect each CNN class score in ss. The loss terms coming from the negative classes are zero. However, the loss gradient respect those negative classes is not cancelled, since the Softmax of the positive class also depends on the negative classes scores.

The gradient expression will be the same for all $$C$$ except for the ground truth class $$C_p$$, because the score of $$C_p (s_p)$$ is in the nominator.

After some calculus, the derivative respect to the positive class is:

$$
\frac{\partial}{\partial s_{p}} \left ( -log\left ( \frac{e^{s_{p}}}{\sum_{j}^{C} e^{s_{j}}} \right ) \right ) = \left ( \frac{e^{s_{p}}}{\sum_{j}^{C}e^{s_{j}}} - 1 \right )
$$

And the derivative respect to the other (negative) classes is:

$$
\frac{\partial}{\partial s_{n}} \left (-log\left ( \frac{e^{s_{p}}}{\sum_{j}^{C} e^{s_{j}}} \right ) \right ) = \left ( \frac{e^{s_{n}}}{\sum_{j}^{C}e^{s_{j}}}\right )
$$

Where snsn is the score of any negative class in $$C$$ different from CpCp.

- Caffe: [SoftmaxWithLoss Layer](http://caffe.berkeleyvision.org/tutorial/layers/softmaxwithloss.html). Is limited to multi-class classification.
- Pytorch: [CrossEntropyLoss](https://pytorch.org/docs/master/nn.html#crossentropyloss). Is limited to multi-class classification.
- TensorFlow: [softmax_cross_entropy](https://www.tensorflow.org/api_docs/python/tf/losses/softmax_cross_entropy). Is limited to multi-class classification.

> In [this Facebook work](https://research.fb.com/publications/exploring-the-limits-of-weakly-supervised-pretraining/) they claim that, despite being counter-intuitive, **Categorical Cross-Entropy loss**, or **Softmax loss** worked better than **Binary Cross-Entropy loss** in their multi-label classification problem.

**→ Skip this part if you are not interested in Facebook or me using Softmax Loss for multi-label classification, which is not standard.**

When Softmax loss is used is a multi-label scenario, the gradients get a bit more complex, since the loss contains an element for each positive class. Consider $$M$$ are the positive classes of a sample. The CE Loss with Softmax activations would be:

$$
CE = \frac{1}{M} \sum_{p}^{M} -log\left ( \frac{e^{s_{p}}}{\sum_{j}^{C} e^{s_{j}}} \right )
$$

Where each $$s_p$$ in $$M$$ is the CNN score for each positive class. As in Facebook paper, I introduce a scaling factor $$1/M$$ to make the loss invariant to the number of positive classes, which may be different per sample.

The gradient has different expressions for positive and negative classes. For positive classes:

$$
\frac{\partial}{\partial s_{pi}} \left ( \frac{1}{M} \sum_{p}^{M} -log\left ( \frac{e^{s_{p}}}{\sum_{j}^{C} e^{s_{j}}} \right ) \right ) = \frac{1}{M} \left ( \left ( \frac{e^{s_{pi}}}{\sum_{j}^{C}e^{s_{j}}} - 1 \right ) + (M - 1) \frac{e^{s_{pi}}}{\sum_{j}^{C}e^{s_{j}}} \right )
$$

Where $$s_pi$$ is the score of any positive class.

For negative classes:

$$
\frac{\partial}{\partial s_{n}} \left ( \frac{1}{M} \sum_{p}^{M} -log\left ( \frac{e^{s_{p}}}{\sum_{j}^{C} e^{s_{j}}} \right ) \right ) = \frac{e^{s_{n}}}{\sum_{j}^{C}e^{s_{j}}}
$$

This expressions are easily inferable from the single-label gradient expressions.

As Caffe Softmax with Loss layer nor Multinomial Logistic Loss Layer accept multi-label targets, I implemented my own PyCaffe Softmax loss layer, following the specifications of the Facebook paper. Caffe python layers let’s us easily customize the operations done in the forward and backward passes of the layer:

## Forward pass: Loss computation

```python
def forward(self, bottom, top):
   labels = bottom[1].data
   scores = bottom[0].data
   # Normalizing to avoid instability
   scores -= np.max(scores, axis=1, keepdims=True)
   # Compute Softmax activations
   exp_scores = np.exp(scores)
   probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
   logprobs = np.zeros([bottom[0].num,1])
   # Compute cross-entropy loss
   for r in range(bottom[0].num): # For each element in the batch
       scale_factor = 1 / float(np.count_nonzero(labels[r, :]))
       for c in range(len(labels[r,:])): # For each class
           if labels[r,c] != 0:  # Positive classes
               logprobs[r] += -np.log(probs[r,c]) * labels[r,c] * scale_factor # We sum the loss per class for each element of the batch

   data_loss = np.sum(logprobs) / bottom[0].num

   self.diff[...] = probs  # Store softmax activations
   top[0].data[...] = data_loss # Store loss
```

We first compute Softmax activations for each class and store them in _probs_. Then we compute the loss for each image in the batch considering there might be more than one positive label. We use an _scale_factor_ ($$M$$) and we also multiply losses by the labels, which can be binary or real numbers, so they can be used for instance to introduce class balancing. The batch loss will be the mean loss of the elements in the batch. We then save the _data_loss_ to display it and the _probs_ to use them in the backward pass.

## Backward pass: Gradients computation

```python
def backward(self, top, propagate_down, bottom):
   delta = self.diff   # If the class label is 0, the gradient is equal to probs
   labels = bottom[1].data
   for r in range(bottom[0].num):  # For each element in the batch
       scale_factor = 1 / float(np.count_nonzero(labels[r, :]))
       for c in range(len(labels[r,:])):  # For each class
           if labels[r, c] != 0:  # If positive class
               delta[r, c] = scale_factor * (delta[r, c] - 1) + (1 - scale_factor) * delta[r, c]
   bottom[0].diff[...] = delta / bottom[0].num
```

In the backward pass we need to compute the gradients of each element of the batch respect to each one of the classes scores $$s$$. As the gradient for all the classes $$C$$ except positive classes $$M$$ is equal to _probs_, we assign _probs_ values to _delta_. For the positive classes in $$M$$ we subtract 1 to the corresponding _probs_ value and use _scale_factor_ to match the gradient expression. We compute the mean gradients of all the batch to run the backpropagation.

> The Caffe Python layer of this Softmax loss supporting a multi-label setup with real numbers labels is available [here](https://gist.github.com/gombru/53f02ae717cb1dd2525be090f2d41055)
