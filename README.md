# ReducedVarianceReparamGradients

Code for ``Reducing Reparameterization Gradient Variance``. 

### Abstract
Optimization with noisy gradients has become ubiquitous in statistics and machine learning.  Reparameterization gradients, or gradient estimates computed via the "reparameterization trick," represent a class of noisy gradients often used in Monte Carlo variational inference (MCVI).  However, when these gradient estimators are too noisy, the optimization procedure can be slow or fail to converge.  One way to reduce noise is to use more samples for the gradient estimate, but this can be computationally expensive.  Instead, we view the noisy gradient as a random variable, and form an inexpensive approximation of the generating procedure for the gradient sample.  This approximation has high correlation with the noisy gradient by construction, making it a useful control variate for variance reduction.  We demonstrate our approach on non-conjugate multi-level hierarchical models and a Bayesian neural net where we observed gradient variance reductions of multiple orders of magnitude (20-2,000x).

Authors: [Andrew Miller](http://andymiller.github.io/), [Nick Foti](http://nfoti.github.io/), [Alex D'Amour](http://www.alexdamour.com/), and [Ryan Adams](http://people.seas.harvard.edu/~rpa/).

### Requires

* [`autograd`](https://github.com/HIPS/autograd) + its requirements (`numpy`, etc).  Our code is compatible with `autograd`'s [latest commit](https://github.com/HIPS/autograd/tree/42a57226442417785efe3bd5ba543b958680b765).
* [`pyprind `](https://github.com/rasbt/pyprind)
