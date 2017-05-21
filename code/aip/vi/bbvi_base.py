from autograd import elementwise_grad, hessian, grad, \
                     hessian_vector_product, make_hvp
from autograd import numpy as np

class BBVI(object):

    def __init__(self, lnpdf, D, glnpdf=None, lnpdf_is_vectorized=False):
        """ Black Box Variational Inference using stochastic gradients"""
        if lnpdf_is_vectorized:
            self.lnpdf = lnpdf
            if glnpdf is None:
                self.glnpdf = elementwise_grad(lnpdf)
        else:
            # create vectorized version
            self.glnpdf_single = grad(lnpdf)
            self.glnpdf = lambda z: np.array([self.glnpdf_single(zi)
                                              for zi in np.atleast_2d(z)])
            self.lnpdf = lambda z: np.array([lnpdf(zi)
                                             for zi in np.atleast_2d(z)])
            #if glnpdf is None:
            #    self.glnpdf = grad(lnpdf)

        # hessian and elementwise_grad of glnpdf
        self.gglnpdf   = elementwise_grad(self.glnpdf)
        self.hlnpdf    = hessian(self.lnpdf)
        self.hvplnpdf  = hessian_vector_product(self.lnpdf)

        # this function creates a generator of Hessian-vector product functions
        #  - make hvp = hvp_maker(lnpdf)(z)
        #  - now  hvp(v) = hessian(lnpdf)(z) v
        self.hvplnpdf_maker = make_hvp(self.lnpdf)

    #################################################
    # BBVI exposes these gradient functions         #
    #################################################

    def elbo_grad_mc(self, lam, t, n_samps=1, eps=None):
        """ monte carlo approximation of the _negative_ ELBO
            eps: seed randomness (could be uniform, could be Gaussian)
        """
        raise NotImplementedError

    def elbo_grad_delta_approx(self, lam, t):
        """ delta method approximation of the _negative_ ELBO """
        raise NotImplementedError

    def elbo_grad_fixed_mixture_approx(self, lam, t, rho = .5):
        """ combine a sample w/ the elbo grad mean """
        raise NotImplementedError

    def elbo_grad_adaptive_mixture_approx(self, lam, t):
        raise NotImplementedError

    def elbo_mc(self, lam, n_samps=100, full_monte_carlo=False):
        """ approximate the ELBO with samples """
        raise NotImplementedError

    def true_elbo(self, lam, t):
        """ approximates the ELBO with 20k samples """
        raise NotImplementedError

    def sample_z(self, lam, n_samps=1, eps=None):
        raise NotImplementedError

    def nat_grad(self, lam, standard_grad):
        """ converts standard gradient into a natural gradient at parameter
        value lam """
        raise NotImplementedError

