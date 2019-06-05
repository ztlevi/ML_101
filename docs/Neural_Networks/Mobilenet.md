<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->

**Table of Contents**

- [Mobilenet v1](#mobilenet-v1)
  - [Depthwise Separable Convolution.](#depthwise-separable-convolution)
  - [Width Multiplier: Thinner Models](#width-multiplier-thinner-models)
- [Mobilenet v2](#mobilenet-v2)
  - [Inverted residuals](#inverted-residuals)

<!-- markdown-toc end -->

# Mobilenet v1

## Depthwise Separable Convolution.

Standard convolutions have the computational cost of :

$$
D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F
$$

where the computational cost depends multiplicatively onthe number of input channels M, the number of output channe is N, the kernel size $$D_K \cdot D_K$$ and the feature map size $$D_F \cdot D_F$$.

<figure>
<img style="width:50%;  display: block; margin-left: auto; margin-right: auto;" src="../../assets/depth-wise-conv.png" alt=""/>
<figcaption></figcaption>
</figure>

Depthwise convolution is extremely efficient relative to standard convolution. However it only filters input channels, it does not combine them to create new features. So an additional layer that computes a linear combination ofthe output of depthwise convolution via $$1 \times 1$$ convolutionis needed in order to generate these new features.

The combination of depthwise convolution and $$1 \times 1$$ (pointwise) convolution is called depthwise separable con-volution.

Depthwise separable convolutions cost:

$$
D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + \cdot M \cdot N \cdot D_F \cdot D_F
$$

- $$D_{F}$$ is the spatial width and height of a square input feature map1
- $$M$$ is the number of input channels (input depth)
- $$D_{G}$$ is the spatial width and height of a square output feature map
- $$N$$ is the number of output channel (output depth).

## Width Multiplier: Thinner Models

For a given layer, and width multiplier $$\alpha$$, the number of input channels $$M$$ becomes $$\alpha M$$ and the number of output channels $$N$$ becomes $$\alpha N$$

# Mobilenet v2

## Inverted residuals

The bottleneck blocks appear similar to residual block where each block contains an input followed by several bottlenecks then followed by expansion. detail code [here](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py#L425).

<figure>
<img style="width:70%;display:block;margin-left:auto;margin-right:auto;" src="../../assets/IR.png" alt="inverted residuals in mobilenet v2"/>
<figcaption></figcaption>
</figure>

- Use shortcuts directly between the bottlenecks.

- The ratio between the size of the input bottleneck and the inner size as the **expansion ratio**.

<figure>
<img style="width:70%;display:block;margin-left:auto;margin-right:auto;" src="../../assets/mobilenetv2.png" alt="mobilenet v2 structure"/>
<figcaption></figcaption>
</figure>
