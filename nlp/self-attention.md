# Self Attention

## Simple RNN + Self Attention



$$
c_0 = 0 \\
h_0 = 0 \\
$$

Simple RNN: $$h_i = tanh(A \cdot [ \begin{matrix} x_i \\ h_{i-1}\end{matrix} ] + b)$$

Simple RNN + Self Attention: $$h_i = tanh(A \cdot [ \begin{matrix} x_i \\ c_{i-1}\end{matrix} ] + b)$$

![calculate h1](../.gitbook/assets/screen-shot-2021-08-14-at-5.13.59-pm.png)

![calculate h2](../.gitbook/assets/screen-shot-2021-08-14-at-5.14.46-pm.png)

Calculate Weights: $$\alpha_i=align(h_i, h_2)$$

![calculate c2](../.gitbook/assets/screen-shot-2021-08-14-at-5.16.29-pm.png)

## Summary

* With self-attention, RNN is less likely to forget.
* Pay attention to the context relevant to the new input.

![self attention focus](../.gitbook/assets/screen-shot-2021-08-14-at-5.20.08-pm.png)

