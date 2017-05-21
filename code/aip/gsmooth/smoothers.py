import autograd.numpy as np


class GradFilter(object):
    def __init__(self, step_size):
        """ object that takes new gradients as an input, and 
        outputs a de-noised version (estimate) """

    def smooth_gradient(self, noisy_gradient, output_variance):
        """ Takes in a gradient, and outputs a smoothed version of it """
        raise NotImplementedError

    def update(self, param, noisy_gradient, step_size):
        """ takes a gradient step
        Input:
            param          : current parameter setting (numpy array)
            noisy_gradient : noisy gradient estimate
            step_size      : how big should the gradient step be?

        Output:
            new_param         : gradient updated parameter
            filtered_gradient : filtered gradient used for update
        """
        filtered_gradient = self.smooth_gradient(noisy_gradient)
        return param - step_size*filtered_gradient, filtered_gradient


class IdentityFilter(GradFilter):
    def __init__(self):
        """ simple do-nothing filter --- for testing as a baseline """
        pass

    def smooth_gradient(self, noisy_gradient):
        return noisy_gradient


class AdamFilter(GradFilter):
    def __init__(self, beta1=.9, beta2=.999, eps=1e-8):
        """Adam uses a type of 'de-biased' exponential smoothing, so it can 
        be viewed as a gradient smoother for stochastic gradient methods"""
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = 0.
        self.v     = 0.
        self.t     = 0
        self.ave_step_sizes = []

    def smooth_gradient(self, noisy_gradient, output_variance=False):
        b1, b2 = self.beta1, self.beta2
        self.t = self.t + 1
        self.m = b1 * self.m + (1. - b1) * noisy_gradient
        self.v = b2 * self.v + (1. - b2) * noisy_gradient**2
        # de-bias
        mhat = self.m / (1 - b1**self.t)
        vhat = self.v / (1 - b2**self.t)
        # smoothed gradient is
        if output_variance:
            return mhat, vhat - mhat**2
        return mhat

    def update(self, param, noisy_gradient, step_size):
        mhat, vhat = \
            self.smooth_gradient(noisy_gradient, output_variance=True)
        vhat += mhat**2
        dparam = mhat / (np.sqrt(vhat) + self.eps)
        self.ave_step_sizes.append(np.mean(1./(np.sqrt(vhat) + self.eps)))
        return param - step_size * dparam, (mhat, vhat)


class SGDMomentumFilter(GradFilter):
    def __init__(self, beta=.9, eps=1e-8):
        """Momentum can also be considered a simple estimator of the true
        gradient.  Beta is the "mass" parameter here"""
        self.beta = .9
        self.eps  = eps
        self.v    = 0.
        self.t    = 0

    def smooth_gradient(self, noisy_gradient):
        self.v = self.beta * self.v - (1. - self.beta) * noisy_gradient
        self.t += 1
        return -self.v
