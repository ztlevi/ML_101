# FPN

![fpn](../../.gitbook/assets/fpn_1.png)

- \(a\) Using an image pyramid to build a feature pyramid. Features are computed on each of the image scales independently, which is slow.
- \(b\) Recent detection systems have opted to use only single scale features for faster detection.
- \(c\) An alternative is to reuse the pyramidal feature hierarchy computed by a ConvNet as if it were a featurized image pyramid.
- \(d\) Our proposed Feature Pyramid Network \(FPN\) is fast like \(b\) and \(c\), but more accurate. In this figure, feature maps are indicate by blue outlines and thicker outlines denote semantically stronger features.

![fpn](../../.gitbook/assets/fpn_2.png)

A building block illustrating the lateral connection and the top-down pathway, merged by addition.
