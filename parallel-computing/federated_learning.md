# Federated Learning

[Slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/14_Parallel_4.pdf)

[Youtube](https://www.youtube.com/watch?v=STxtRucv_zo)

## Motivating Examples

![federated\_learning\_1](../.gitbook/assets/federated_learning_1.png)

![federated\_learning\_2](../.gitbook/assets/federated_learning_2.png)

## What is federated learning

Federated learning , \[1\] is \[2\] a kind of distributed learning.

How does federated learning differ from traditional distributed learning?

1. Users have control over their device and data.
2. Worker nodes are unstable.
3. Communication cost is higher than computation cost.
4. Data stored on worker nodes are not IID.
5. The amount of data is severely imbalanced.

## Let us recall parallel gradient descent

![federated\_learning\_3](../.gitbook/assets/federated_learning_3.png)

![federated\_learning\_4](../.gitbook/assets/federated_learning_4.png)

## Federated Averaging Algorithm

![federated\_learning\_5](../.gitbook/assets/federated_learning_5.png)

![federated\_learning\_6](../.gitbook/assets/federated_learning_6.png)

## Computation vs. Communication

![federated\_learning\_7](../.gitbook/assets/federated_learning_7.png)

![federated\_learning\_8](../.gitbook/assets/federated_learning_8.png)

## References

* \[1\] McMahan and others: Communication-efficient learning of deep networks from decentralized data. In AISTATS, 2017. . 
* \[2\] Konevcny, McMahan, and Ramage: Federated optimization: distributed optimization beyond the datacenter. arXiv:1511.03575, 2015

