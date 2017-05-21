"""
Functions for computing control variate noise reduced gradients
"""
import autograd.numpy as np
from autograd import grad, hessian_vector_product, make_hvp

def construct_cv_grads(vbobj, lam, eps,
                       elbo_gsamps=None,
                       method="hessian"):
    """ main method to construct reduced variance reparameterization gradients
        using a variety of methods.

    Methods:
        - "mc"           : full monte carlo estimator
        - "hessian"      : uses full hessian information
        - "hessian_diag" : uses only hessian diag information
        - "hvp_with_loo_diag_approx" : uses other samples to approximate 
        - "hvp_with_mc_variance"     : 
    """
    # unpack variational parameters
    assert eps.ndim == 2, "epsilon needs to be nsamps x D"
    ns, D = eps.shape
    m_lam, s_lam = lam[:D], np.exp(lam[D:])

    # generate samples if necessary
    if elbo_gsamps is None:
        elbo_gsamps = elbo_grad_samps_mat(vbobj, lam, eps)

    if method == "mc": 
        # full monte carlo --- this is a No-op
        return elbo_gsamps

    elif method == "hessian":
        """ full hessian approximation
        """
        # full hessian, including diagonal
        gmu   = vbobj.glnpdf(m_lam)
        H     = vbobj.hlnpdf(m_lam)
        Hdiag = np.diag(H)

        # construct normal approx samples of data term
        dLdz  = gmu + np.dot(H, (s_lam*eps).T).T
        #dLds  = (dLdz*eps + 1/s_lam[None,:]) * s_lam
        dLds  = dLdz*eps*s_lam + 1.
        elbo_gsamps_tilde = np.column_stack([dLdz, dLds])

        # characterize the mean of the dLds component (and z comp)
        dLds_mu = (Hdiag*s_lam + 1/s_lam) * s_lam
        gsamps_tilde_mean = np.concatenate([gmu, dLds_mu])

        # subtract mean to compute control variate
        elbo_gsamps_cv = elbo_gsamps - \
                         (elbo_gsamps_tilde - gsamps_tilde_mean)
        return elbo_gsamps_cv

    elif method == "hessian_diag":
        """ use only hessian diagonal for RV model """
        gmu   = vbobj.glnpdf(m_lam)
        H     = vbobj.hlnpdf(m_lam)
        Hdiag = np.diag(H)

        # construct normal approx samples of data term
        dLdz = gmu + Hdiag * s_lam * eps
        dLds = (dLdz*eps + 1/s_lam[None,:]) * s_lam
        elbo_gsamps_tilde = np.column_stack([dLdz, dLds])

        # construct mean
        dLds_mu = (Hdiag*s_lam + 1/s_lam) * s_lam
        gsamps_tilde_mean = np.concatenate([gmu, dLds_mu])
        elbo_gsamps_cv    = elbo_gsamps - \
                            (elbo_gsamps_tilde - gsamps_tilde_mean)
        return elbo_gsamps_cv

    elif method == "hvp_with_loo_diag_approx":
        """ use other samples to estimate a per-sample diagonal
        expectation
        """
        assert ns > 1, "loo approximations require more than 1 sample"
        # compute hessian vector products and save them for both parts
        #hvps = np.array([vbobj.hvplnpdf(m_lam, s_lam*e) for e in eps])
        hvp_lam = vbobj.hvplnpdf_maker(m_lam)
        hvps    = np.array([hvp_lam(s_lam*e) for e in eps])
        gmu     = vbobj.glnpdf(m_lam)

        # construct normal approx samples of data term
        dLdz    = gmu + hvps
        #dLds   = (dLdz*eps + 1/s_lam[None,:]) * s_lam
        dLds    = dLdz * (eps*s_lam) + 1

        # compute Leave One Out approximate diagonal (per-sample mean of dLds)
        Hdiag_sum = np.sum(eps*hvps, axis=0)
        Hdiag_s   = (Hdiag_sum[None,:] - eps*hvps) / float(ns-1)
        dLds_mu   = (Hdiag_s + 1/s_lam[None,:]) * s_lam

        # compute gsamps_cv - mean(gsamps_cv), and finally the var reduced
        #elbo_gsamps_tilde_centered = \
        #    np.column_stack([ hvps, dLds - dLds_mu ])
        #elbo_gsamps_cv = elbo_gsamps - elbo_gsamps_tilde_centered
        #return elbo_gsamps_cv
        elbo_gsamps[:,:D] -= hvps
        elbo_gsamps[:,D:] -= (dLds - dLds_mu)
        return elbo_gsamps

    elif method == "hvp_with_loo_direct_approx":
        # compute hessian vector products and save them for both parts
        assert ns > 1, "loo approximations require more than 1 sample"
        gmu  = vbobj.glnpdf(m_lam)
        hvps = np.array([vbobj.hvplnpdf(m_lam, s_lam*e) for e in eps])

        # construct normal approx samples of data term
        dLdz    = gmu + hvps
        dLds    = (dLdz*eps + 1/s_lam[None,:]) * s_lam
        elbo_gsamps_tilde = np.column_stack([dLdz, dLds])

        # compute Leave One Out approximate diagonal (per-sample mean of dLds)
        dLds_sum = np.sum(dLds, axis=0)
        dLds_mu  = (dLds_sum[None,:] - dLds) / float(ns-1)

        # compute gsamps_cv - mean(gsamps_cv), and finally the var reduced
        elbo_gsamps_tilde_centered = \
            np.column_stack([ dLdz - gmu, dLds - dLds_mu ])
        elbo_gsamps_cv = elbo_gsamps - elbo_gsamps_tilde_centered
        return elbo_gsamps_cv

    elif method == "hvp_with_mc_variance":
        hvp_lam = vbobj.hvplnpdf_maker(m_lam)
        hvps    = np.array([hvp_lam(s_lam*e) for e in eps])
        elbo_gsamps[:,:D] -= hvps
        return elbo_gsamps

    # not implemented
    raise NotImplementedError("%s not implemented"%method)


def elbo_grad_samps_mat(vbobj, lam, eps):
    """ function to compute grad g = [g_m, g_lns] 
            g_m   = dELBO / dm
            g_lns = dELBO / d ln-sigma]
    from some base randomness, eps

    Returns:
        - [dELBO/dm, dELBO/dlns] as a Nsamps x D array
    """
    assert eps.ndim == 2, "epsilon must be Nsamps x D"
    D = vbobj.D

    # generate samples
    m_lam, s_lam = lam[:D], np.exp(lam[D:])
    zs = m_lam[None,:] + s_lam[None,:] * eps

    # generate dElbo/dm (which happens to be dElbo / dz)
    dL_dz = vbobj.glnpdf(zs)
    dL_dm = dL_dz

    # generate dElbo/d sigma, convert via d sigma/d ln-sigma
    dL_ds   = dL_dz*eps + 1/s_lam
    dL_dlns = dL_ds * s_lam
    return np.column_stack([dL_dm, dL_dlns])


#def elbo_grad_samps(vbobj, lam, eps):
#    """ Computes elbo grad sample and averages """
#    dL_dlam_samps = elbo_grad_samps_mat(vbobj, lam, eps)
#
#    # dL_dlam and var dL_dlam
#    dL_dlam = -1.*np.mean(dL_dlam_samps, 0)
#    dL_dlam_var = np.var(dL_dlam_samps, 0)
#
#    # also compute mini bootstrap variance
#    L = eps.shape[0]
#    return dL_dlam, dL_dlam_var / L
#
#
#def elbo_grad_approx(vbobj, lam, eps, approx_type="trace"):
#    """ function to compute approx grad mean and variance and samples
#            gtilde = [ gtilde_m, gtilde_s ]
#
#            gtilde_m   ~ N(gm, gs*gs)
#            gtilde_lns ~
#
#        based on some randomness, eps
#
#    Input: 
#        vbobj: essentially access to the dlnp_delta approx handle
#        lam  : variational params = [mu, lns]
#        eps  : num_reps x L x D sample of N(0, I) randomness
#               where L is the number of samples for the estimator
#               and num_reps is the number of estimators called for
#    Returns:
#        - gmean, gvar, gsamps
#    """
#    assert eps.ndim > 1, "epsilon must be num_reps x L x D"
#    if eps.ndim == 2:
#        eps = np.array([eps])
#    D = vbobj.D
#
#    # compute approximate distribution (mean and SIGNED standard dev)
#    # of dL / dz at lambda
#    m_lam, s_lam = lam[:D], np.exp(lam[D:])
#    gz_m, gz_s = vbobj.dlnp_delta_approx(lam,
#                                         approx_type=approx_type,
#                                         eps_mat = eps)
#
#    # generative correlated samples
#    dL_dz_tilde = gz_m[None,None,:] + gz_s[None,None,:]*eps
#
#    # mean and var samples
#    dL_dm_tilde = dL_dz_tilde
#
#    # var samples --- need to multiply by eps + convert by s_lam
#    dL_ds_tilde   = dL_dz_tilde * eps + 1/s_lam[None,None,:]
#    dL_dlns_tilde = dL_ds_tilde * s_lam
#
#    # create grad estimator
#    dL_dlam = np.column_stack([np.mean(dL_dm_tilde, axis=1),
#                               np.mean(dL_dlns_tilde, axis=1)])
#
#    # create mean and variance
#    grad_mean = np.concatenate([gz_m, gz_s*s_lam + 1])
#    grad_var  = np.concatenate([gz_s*gz_s, s_lam**2*(gz_m**2 + 2*gz_s*gz_s)])
#    L = eps.shape[1]
#    grad_var /= float(L)
#    return -1.*dL_dlam.squeeze(), -1.*grad_mean, grad_var
#
#
#def elbo_grad_and_approx_samps(vbobj, lam, eps):
#    """ Attempt to approximate the covariance between g and g_tilde on the 
#    fly --- compute t
#    """
#    D = vbobj.D
#
#    def gen_approx():
#        # compute approximate distribution (mean and SIGNED standard dev)
#        # of dL / dz at lambda
#        m_lam, s_lam = lam[:D], np.exp(lam[D:])
#        gz_m, gz_s = vbobj.dlnp_delta_approx(lam)
#
#        # generative correlated samples
#        dL_dz_tilde = gz_m[None,:] + gz_s[None,:]*eps
#
#        # mean and var samples
#        dL_dm_tilde = dL_dz_tilde
#
#        # var samples --- need to multiply by eps + convert by s_lam
#        dL_ds_tilde   = dL_dz_tilde * eps + 1/s_lam[None,:]
#        dL_dlns_tilde = dL_ds_tilde * s_lam
#
#        # approx samps
#        dL_dlam_tilde_samps = np.column_stack([dL_dm_tilde, dL_dlns_tilde])
#
#        # create mean and variance
#        grad_mean = np.concatenate([gz_m, gz_s*s_lam + 1])
#        grad_var  = np.concatenate([gz_s*gz_s, s_lam**2*(gz_m**2 + 2*gz_s**2)])
#        L = eps.shape[0]
#        grad_var /= float(L)
#        return dL_dlam_tilde_samps, grad_mean, grad_var
#
#    # true gradient samples
#    dL_dlam_samps     = elbo_grad_samps_mat(vbobj, lam, eps)
#    L, num_var_params = dL_dlam_samps.shape
#    dL_var = np.var(dL_dlam_samps, axis=0) / L
#
#    # approx grad samples and mean/var
#    dL_dlam_tilde_samps, grad_mean, grad_var = gen_approx()
#    covs = np.array([ np.cov(dL_dlam_samps[:,l],
#                             dL_dlam_tilde_samps[:,l])[0,1]
#                      for l in xrange(num_var_params)]) / L
#
#    return -1.*dL_dlam_samps.mean(0), -1.*dL_dlam_tilde_samps.mean(0), \
#           -1.*grad_mean, grad_var, covs, dL_var
#
#
##################################
## older control variate code    #
##################################
#
#def compute_control_variate_estimator(X, C, Cvar=None):
#    """
#    Args:
#      - X: Nsamp x D random vectors
#      - C: Nsamp x D control variate values
#      - Cvar: D-dimensional variances of each dimension of C
#    """
#    if Cvar is None:
#        Cvar = np.var(C, axis=0)
#
#    D       = X.shape[1]
#    Xcv     = np.zeros_like(X)
#    c_stars = np.zeros(D)
#    for d in xrange(D):
#        c_stars[d] = -np.cov(X[:,d], C[:,d])[0,1] / Cvar[d]
#        Xcv[:,d]   = X[:,d] + c_stars[d] * C[:,d]
#
#    return Xcv, c_stars
#
