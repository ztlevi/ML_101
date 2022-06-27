# Class Visualization

By starting with a random noise image and performing gradient ascent on a target class, we can generate an image that the network will recognize as the target class. This idea was first presented in \[[2](https://arxiv.org/pdf/1312.6034.pdf)] ; \[[3](https://arxiv.org/abs/1506.06579)] extended this idea by suggesting several regularization techniques that can improve the quality of the generated image.

Concretely, let $$I$$ be an image and let $$y$$ be a target class. Let $$s_y(I)$$ be the score that a convolutional network assigns to the image $$I$$ for class $$y$$; note that these are raw unnormalized scores, not class probabilities. We wish to generate an image $$I^*$$ that achieves a high score for the class $$y$$ by solving the problem

$$
I^* = argmax_I(s_y(I) - R(I))
$$

where $$R$$ is a (possibly implicit) regularizer (note the sign of $$R(I)$$ in the argmax: we want to minimize this regularization term). We can solve this optimization problem using gradient ascent, computing gradients with respect to the generated image. We will use (explicit) L2 regularization of the form

$$
R(I) = \lambda \parallel I \parallel _2^2
$$

and implicit regularization as suggested by \[[3](https://arxiv.org/abs/1506.06579)] by periodically blurring the generated image. We can solve this problem using gradient ascent on the generated image.

In the cell below, complete the implementation of the `create_class_visualization` function.

## Implementation

```python
from scipy.ndimage.filters import gaussian_filter1d
def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X

def create_class_visualization(target_y, model, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.

    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to jitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    X = 255 * np.random.rand(224, 224, 3)
    X = preprocess_image(X)[None]

    ########################################################################
    # TODO: Compute the loss and the gradient of the loss with respect to  #
    # the input image, model.image. We compute these outside the loop so   #
    # that we don't have to recompute the gradient graph at each iteration #
    #                                                                      #
    # Note: loss and grad should be TensorFlow Tensors, not numpy arrays!  #
    #                                                                      #
    # The loss is the score for the target label, target_y. You should     #
    # use model.classifier to get the scores, and tf.gradients to compute  #
    # gradients. Don't forget the (subtracted) L2 regularization term!     #
    ########################################################################

    loss = model.classifier[0, target_y] - l2_reg * tf.nn.l2_loss(model.image) # scalar loss
    grad = tf.gradients(loss, model.image) # gradient of loss with respect to model.image, same size as model.image

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
        Xi = X.copy()
        X = np.roll(np.roll(X, ox, 1), oy, 2)

        ########################################################################
        # TODO: Use sess to compute the value of the gradient of the score for #
        # class target_y with respect to the pixels of the image, and make a   #
        # gradient step on the image using the learning rate. You should use   #
        # the grad variable you defined above.                                 #
        #                                                                      #
        # Be very careful about the signs of elements in your code.            #
        ########################################################################

        dx = sess.run(grad, feed_dict={model.image: X})
        X += dx[0] * learning_rate

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Undo the jitter
        X = np.roll(np.roll(X, -ox, 1), -oy, 2)

        # As a regularizer, clip and periodically blur
        X = np.clip(X, -SQUEEZENET_MEAN/SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN)/SQUEEZENET_STD)
        if t % blur_every == 0:
            X = blur_image(X, sigma=0.5)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess_image(X[0]))
            class_name = class_names[target_y]
            plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.show()
    return X
```

## Example

Once you have completed the implementation in the cell above, run the following cell to generate an image of Tarantula:

```python
target_y = 76 # Tarantula
out = create_class_visualization(target_y, model)
```

|                                                                |                                                                 |                                                                |
| -------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------- |
| ![img](<../.gitbook/assets/class\_vis\_tarantula\_1 (1).png>)  | ![img](<../.gitbook/assets/class\_vis\_tarantula\_25 (1).png>)  | ![img](<../.gitbook/assets/class\_vis\_tarantula\_50 (1).png>) |
| ![img](<../.gitbook/assets/class\_vis\_tarantula\_75 (1).png>) | ![img](<../.gitbook/assets/class\_vis\_tarantula\_100 (1).png>) |                                                                |
