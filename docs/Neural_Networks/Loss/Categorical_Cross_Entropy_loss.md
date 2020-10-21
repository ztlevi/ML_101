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

For full code, take a look at [here](https://github.com/ztlevi/Machine_Learning_Questions/blob/master/codes/softmax_loss/softmax_loss.py).

```python
    def forward(ctx, x, target):
        """
        forward propagation
        """
        assert x.dim() == 2, "dimension of input should be 2"
        exp_x = torch.exp(x)
        y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)

        # parameter "target" is a LongTensor and denotes the labels of classes, here we need to convert it into one hot vectors
        t = torch.zeros(y.size()).type(y.type())
        for n in range(t.size(0)):
            t[n][target[n]] = 1

        output = (-t * torch.log(y)).sum() / y.size(0)

        # output should be a tensor, but the output of sum() is float
        output = torch.Tensor([output]).type(y.type())
        ctx.save_for_backward(y, t)

        return output
```

## Backward pass: Gradients computation

```python
    @staticmethod
    def backward(ctx, grad_output):
        """
        backward propagation
        # softmax with ce loss backprop see https://www.youtube.com/watch?v=5-rVLSc2XdE
        """
        y, t = ctx.saved_tensors

        # grads = []
        # for i in range(y.size(0)):
        #     grads.append(softmax_grad(y[i]))

        grads = softmax_grad_vectorized(y)
        grad_input = grad_output * (y - t) / y.size(0)
        return grad_input, None
```

> The Caffe Python layer of this Softmax loss supporting a multi-label setup with real numbers labels is available [here](https://gist.github.com/gombru/53f02ae717cb1dd2525be090f2d41055)
