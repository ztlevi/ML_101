# Metrics

The following 12 metrics are used for characterizing the performance of an object detector on COCO:

### Average Precision (AP):

Compute average precision (AP) from prediction scores

AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight:

$$
\text{AP} = \sum_n (R_n - R_{n-1}) P_n
$$

where $$P_n$$ and $$R_n$$ are the precision and recall at the nth threshold. This implementation is not interpolated and is different from computing the area under the precision-recall curve with the trapezoidal rule, which uses linear interpolation and can be too optimistic.

Note: this implementation is restricted to the binary classification task or multilabel classification task.

AP: AP at IoU=.50:.05:.95 (primary challenge metric)

$$AP^{IoU=.50}$$: AP at IoU=.50 (PASCAL VOC metric)

$$AP^{IoU=.75}$$: AP at IoU=.75 (strict metric)

### AP Across Scales:

$$AP{small}$$: AP for small objects: $$area < 32^2$$

$$AP{medium}$$: AP for medium objects: $$32^2 < area < 96^2$$

$$AP{large}$$: AP for large objects: $$area > 96^2$$

### Average Recall (AR):

$$AR^{max=1}$$: AR given 1 detection per image

$$AR^{max=10}$$: AR given 10 detections per image

$$AR^{max=100}$$: AR given 100 detections per image

### AR Across Scales:

$$AR^{small}$$: AR for small objects: $$area < 32^2$$

$$AR^{medium}$$: AR for medium objects: $$32^2 < area < 96^2$$

$$AR^{large}$$: AR for large objects: $$area > 96^2$$

## Description

1.Unless otherwise specified, AP and AR are averaged over multiple Intersection over Union (IoU) values. Specifically we use 10 IoU thresholds of .50:.05:.95. This is a break from tradition, where AP is computed at a single IoU of .50 (which corresponds to our metric $$AP^{IoU=.50}$$). Averaging over IoUs rewards detectors with better localization.

2.AP is averaged over all categories. Traditionally, this is called "mean average precision" (mAP). We make no distinction between AP and mAP (and likewise AR and mAR) and assume the difference is clear from context.

3.AP (averaged across all 10 IoU thresholds and all 80 categories) will determine the challenge winner. This should be considered the single most important metric when considering performance on COCO.

4.In COCO, there are more small objects than large objects. Specifically: approximately 41% of objects are small ($$area < 32^2$$), 34% are medium ($$32^2 < area < 96^2$$), and 24% are large ($$area > 96^2$$). Area is measured as the number of pixels in the segmentation mask.

5.AR is the maximum recall given a fixed number of detections per image, averaged over categories and IoUs. AR is related to the metric of the same name used in proposal evaluation but is computed on a per-category basis.

6.All metrics are computed allowing for at most 100 top-scoring detections per image (across all categories). The evaluation metrics for detection with bounding boxes and segmentation masks are identical in all respects except for the IoU computation (which is performed over boxes or masks, respectively).
