# Hinge Loss

From our SVM model, we know that hinge $$loss = [0, 1- y*f(x)]$$.

Looking at the graph for SVM in Fig 4, we can see that for $$y*f(x) \geq 1$$, hinge loss is ‘**0**’. However, when $$y*f(x) < 1$$, then hinge loss increases massively. As $$y*f(x)$$ increases with every misclassified point \(very wrong points in Fig 5\), the upper bound of hinge loss $${1- y*f(x)}$$ also increases exponentially.

Hence, the points that are farther away from the decision margins have a greater loss value, thus penalising those points.

_Conclusion_: This is just a basic understanding of what loss functions are and how hinge loss works. I will be posting other articles with greater understanding of ‘Hinge loss’ shortly

