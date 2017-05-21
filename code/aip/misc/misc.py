import autograd.numpy as np
from autograd.scipy.special import gammaln

def sigmoid(a):
    return 1. / (1. + np.exp(-a))


def logit(a):
    return np.log(a) - np.log(1-a)


def mvn_diag_logpdf(x, mean, log_std):
    D = len(mean)
    qterm = -.5 * np.sum((x - mean)**2 / np.exp(2.*log_std), axis=1)
    coef  = -.5*D * np.log(2.*np.pi) - np.sum(log_std)
    return qterm + coef


def mvn_diag_logpdf_grad(x, mean, log_std):
    pass


def mvn_diag_entropy(log_std):
    D = len(log_std)
    return .5 * (D*np.log(2*np.pi*np.e) + np.sum(2*log_std))


def mvn_logpdf(x, mean, icholSigma):
    D     = len(mean)
    coef  = -.5*D*np.log(2.*np.pi)
    dterm = np.sum(np.log(np.diag(icholSigma)))
    white = np.dot(np.atleast_2d(x) - mean, icholSigma.T)
    qterm = -.5*np.sum(white**2, axis=1)
    ll = coef + dterm + qterm
    if len(ll) == 1:
        return ll[0]
    return ll


def mvn_fisher_info(params):
    """ returns the fisher information matrix (diagonal) for a multivariate
    normal distribution with params = [mu, ln sigma] """
    D = len(params) / 2
    mean, log_std = params[:D], params[D:]
    return np.concatenate([np.exp(-2.*log_std),
                           2*np.ones(D)])


def kl_mvn(m0, S0, m1, S1):
    """KL divergence between two normal distributions - can 
        m0: N x
    """
    #    .5 log det (Sig1 Sig0^-1)
    # +  .5 tr( Sig1^-1 * ((mu_0 - mu_1)(mu_0 - mu_1)^T + Sig0 - Sig1) )
    det_term = .5 * np.log(npla.det(npla.solve(S0, S1).T))
    S1inv    = npla.inv(S1)
    diff     = m0 - m1
    outers   = np.einsum("id,ie->ide", diff, diff) + S0 - S1
    tr_term  = .5 * np.einsum("de,ide->i", S1inv, outers)
    return det_term + tr_term


def kl_mvn_diag(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.

    - accepts stacks of means, but only one S0 and S1

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[1]
    iS1 = 1./S1
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.sum(iS1 * S0)
    det_term  = np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = np.sum( (diff*diff) * iS1, axis=1)
    return .5 * (tr_term + det_term + quad_term - N)


def gamma_lnpdf(x, shape, rate):
    """ shape/rate formulation on wikipedia """
    coef  = shape * np.log(rate) - gammaln(shape)
    dterm = (shape-1.) * np.log(x) - rate*x
    return coef + dterm


def make_fixed_cov_mvn_logpdf(Sigma):
    icholSigma = np.linalg.inv(np.linalg.cholesky(Sigma))
    return lambda x, mean: mvn_logpdf(x, mean, icholSigma)


def unpack_params(params):
    mean, log_std = np.split(params, 2)
    return mean, log_std


def unconstrained_to_simplex(rhos):
    rhosf = np.concatenate([rhos, [0.]])
    pis   = np.exp(rhosf) / np.sum(np.exp(rhosf))
    return pis


def simplex_to_unconstrained(pis):
    lnpis = np.log(pis)
    return (lnpis - lnpis[-1])[:-1]

