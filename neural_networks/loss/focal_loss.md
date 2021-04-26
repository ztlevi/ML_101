## Class Imbalance Problem of One-Stage Detector

- A much larger set of candidate object locations is regularly sampled across an image \(~100k locations\), which densely cover spatial positions, scales and aspect ratios.
- The training procedure is still **dominated by easily classified background examples**. It is typically addressed via bootstrapping or hard example mining. But they are not efficient enough.

## alpha-Balanced CE Loss

$$
CE(p_t) = - \alpha_t t_i log(p_t)
$$

- To address the class imbalance, one method is to add a weighting factor $$\alpha$$ for class 1 and $$1 - \alpha$$ for class -1. $$\alpha$$ may be set by inverse class frequency or treated as a hyperparameter to set by cross validation.

## Focal Loss

$$
FL = -\sum_{i=1}^{C=n}(1 - p_{i})^{\gamma }t_{i} log (p_{i})
$$

- The loss function is reshaped to **down-weight easy examples** and thus focus training on hard negatives. A modulating factor $$(1-p_{t})^{\gamma}$$ is added to the cross entropy loss where $$\gamma$$ is tested from $$[0,5]$$ in the experiment.
- There are two properties of the FL:
- When an example is misclassified and $$p_{t}$$ is small, the modulating factor is near 1 and the loss is unaffected. **As** $$p_{t} \rightarrow 1$$**, the factor goes to 0 and the loss for well-classified examples is down-weighted**.
- The focusing parameter $$\gamma$$ **smoothly adjusts the rate** at which easy examples are down-weighted. When $$\gamma = 0$$, FL is equivalent to CE. When $$\gamma$$ is increased, the effect of the modulating factor is likewise increased. \($$\gamma = 2$$ works best in experiment.\)

## alpha-Balanced Variant of FL

$$
FL = -\sum_{i=1}^{C=n} - \alpha_t (1 - p_{i})^{\gamma }t_{i} log (p_{i})
$$

- The above form is used in experiment in practice where α is added into the equation, which yields slightly improved accuracy over the one without α. And using sigmoid activation function for computing p resulting in greater numerical stability.
- $$\gamma$$: Focus more on hard examples.
- $$\alpha$$: Offset class imbalance of number of examples.
