# Basics

## Project Workflow

Given a data science / machine learning project, what steps should we follow? Here's how I would tackle it:

* **Specify business objective.** Are we trying to win more customers, achieve higher satisfaction, or gain more revenues?
* **Define problem.** What is the specific gap in your ideal world and the real one that requires machine learning to fill? Ask questions that can be addressed using your data and predictive modeling (ML algorithms).
* **Create a common sense baseline.** But before you resort to ML, set up a baseline to solve the problem as if you know zero data science. You may be amazed at how effective this baseline is. It can be as simple as recommending the top N popular items or other rule-based logic. This baseline can also server as a good benchmark for ML algorithms.
* **Review ML literatures.** To avoid reinventing the wheel and get inspired on what techniques / algorithms are good at addressing the questions using our data.
* **Set up a single-number metric.** What it means to be successful - high accuracy, lower error, or bigger AUC - and how do you measure it? The metric has to align with high-level goals, most often the success of your business. Set up a single-number against which all models are measured.
* **Do exploratory data analysis (EDA).** Play with the data to get a general idea of data type, distribution, variable correlation, facets etc. This step would involve a lot of plotting.
* **Partition data.** Validation set should be large enough to detect differences between the models you are training; test set should be large enough to indicate the overall performance of the final model; training set, needless to say, the larger the merrier.
* **Preprocess.** This would include data integration, cleaning, transformation, reduction, discretization and more.
* **Engineer features.** Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering. This step usually involves feature selection and creation, using domain knowledge. Can be minimal for deep learning projects.
* **Develop models.** Choose which algorithm to use, what hyperparameters to tune, which architecture to use etc.
* **Ensemble.** Ensemble can usually boost performance, depending on the correlations of the models/features. So it's always a good idea to try out. But be open-minded about making tradeoff - some ensemble are too complex/slow to put into production.
* **Deploy model.** Deploy models into production for inference.
* **Monitor model.** Monitor model performance, and collect feedbacks.
* **Iterate.** Iterate the previous steps. Data science tends to be an iterative process, with new and improved models being developed over time.

![](<../.gitbook/assets/workflow (1).png>)

## [IID: independent and identically distributed](../codes/IID/IID.ipynb)

This assumption has two parts:

1. Independent
2. Identically distributed

[Youtube](https://www.youtube.com/watch?v=lhzndcgCXeo)

## Confusion matrix

*   Accuracy:

    <img src="../.gitbook/assets/cm_accuracy (1).png" alt="img" data-size="original">
*   Precision:

    <img src="../.gitbook/assets/cm_precision (1).png" alt="img" data-size="original">
*   Recall:

    <img src="../.gitbook/assets/cm_recall (1).png" alt="img" data-size="original">
* F1 Score:
  *   It is also called the F Score or the F Measure. Put another way, the F1 score conveys the balance between the precision and the recall.

      $$
      F_1 score = \frac{2 * precision * recall}{precision + recall}
      $$

## Weight Initialization

`W = 0.01 * np.random.randn(D,H)`, where `randn` generates a random list with size n. It samples from a zero mean, unit standard deviation gaussian. One problem with the above suggestion is that the distribution of the outputs from a randomly initialized neuron has a variance that grows with the number of inputs. It turns out that we can normalize the variance of each neuron's output to 1 by scaling its weight vector by the square root of its fan-in (i.e. its number of inputs). `w = np.random.randn(n) / sqrt(n)`, where n is the number of its inputs.

In practice, the current recommendation is:

* to use ReLU units: use the `w = np.random.randn(n) * sqrt(2.0/n)`
* To use Tanh units: use the `w = np.random.randn(n) * sqrt(1.0/n)`

## Non-maximal suppression

### Codes

* [IOU](https://github.com/ztlevi/Machine\_Learning\_Questions/blob/master/codes/NMS/IOU.py)
* [NMS](https://github.com/ztlevi/Machine\_Learning\_Questions/blob/master/codes/NMS/nms.py)
* [NMS\_Slow](https://github.com/ztlevi/Machine\_Learning\_Questions/blob/master/codes/NMS/nms\_slow.py)
* [NMS\_Fast](https://github.com/ztlevi/Machine\_Learning\_Questions/blob/master/codes/NMS/nms\_fast.py) uses numpy

## Image interpolation

* Nearest Neighbor
* Bilinear
* Bicubic

![img](<../.gitbook/assets/image\_interpolation (1).jpg>)

![img](<../.gitbook/assets/bicubic (1).jpg>)

### Examples

![img](<../.gitbook/assets/image\_interpolation\_2 (1).jpg>)

## Types of Stratified Sampling

### Proportionate Stratified Random Sampling

The sample size of each stratum in this technique is proportionate to the population size of the stratum when viewed against the entire population. This means that each stratum has the same sampling fraction.

For example, you have 3 strata with 100, 200 and 300 population sizes respectively. And the researcher chose a sampling fraction of ½. Then, the researcher must randomly sample 50, 100 and 150 subjects from each stratum respectively.

| Stratum           | A   | B   | C   |
| ----------------- | --- | --- | --- |
| Population Size   | 100 | 200 | 300 |
| Sampling Fraction | 1/2 | 1/2 | 1/2 |
| Final Sample Size | 50  | 100 | 150 |

The important thing to remember in this technique is to use the same sampling fraction for each stratum regardless of the differences in population size of the strata. It is much like assembling a smaller population that is specific to the relative proportions of the subgroups within the population.

### Disproportionate Stratified Random Sampling

The only difference between proportionate and disproportionate stratified random sampling is their sampling fractions. With disproportionate sampling, the different strata have different sampling fractions.

The precision of this design is highly dependent on the sampling fraction allocation of the researcher. If the researcher commits mistakes in allotting sampling fractions, a stratum may either be overrepresented or underrepresented, which will result in skewed results.

## Kalman Filter

* Kalman filter helps us to obtain more reliable estimates from a sequence of observed measurements.
* We make a prediction of a state, based on some previous values and model to get $$X_k'$$ and $$P_k$$
* We obtain the measurement of that state, from the sensor.
* We update our prediction, based on our errors
*   Repeat.

    <img src="../.gitbook/assets/1_s2kA7oclIHoCAQsao2fXhw (1).jpeg" alt="img" data-size="original">
* process noise $$W_k \sim \mathcal{N} (0, Q_k)$$
* process noise covariance matrix try to **keep the state convariance matrix from being too small**
*   We can also write the update formula as:

    $$X_k = (1 - KH)X_k' + KY_k$$

    $$P_k = (1 - KH)P_k'$$

### Kalman Gain

$$K = \frac{P_{k}'H_{k}^{T}}{HP_{k}'H_{k} + R}$$

$$S = HP_{k}'H_{k} + R$$

The Kalman gain tells you **how much** I want to change my estimate by given a measurement.

* $$S$$ is the estimated covariance matrix of the measurements. this tells us the "variability" in our measurements, if $$S$$ is small, variability is low, our confidence in the measurement increases.
* $$P_{k}'$$ is the estimated state covariance matrix. This tells us the "variability" of the state $$X_k'$$. If $$P_k'$$ is large, it means that the state is estimated to change a lot.
