# L1 and L2 Loss

### L1 Loss vs L2 Loss

* **Robustness**: L1 &gt; L2

  Intuitively speaking, since a L2-norm squares the error \(increasing by a lot if error &gt; 1\), the model will see a much larger error than the L1-norm, so the model is much more sensitive to outliers.

* **Stability**: L2 &gt; L1

  In the case of a more “outlier” point, both norms still have big change, but again the L1-norm has more changes in general.

* **Solution uniqueness**: Minimizing the L2 loss corresponds to calculating **the arithmetic mean**, which is unambiguous, while minimizing the L1 loss corresponds to calculating **the median**, which is ambiguous if an even number of elements are included in the median calculation, So L2 has unique solution while L1 has multiple solution
* Smooth l1 loss

![img](../.gitbook/assets/smooth_l1_loss.png)

Smooth L1 loss that is less sensitive to outliers than the L2 loss used in R-CNN and SPPne.

