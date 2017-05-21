"""
Implementation of the hierarchical poisson glm model, with a precinct-specific
term, an ethnicity specific term, and an offset term.

The data are tuples of (ethnicity, precinct, num_stops, total_arrests), where
the count variables num_stops and total_arrests refer to the number of stops
and total arrests of an ethnicity in the specified precinct over a period of
15 months.  The rate we are measuring is the rate of stops-per-arrest
for certain ethnicities in different precincts.  

    Y_ep       = num stops of ethnicity e in precinct p
    N_ep       = num arests of e in p
    log lam_ep = alpha_e + beta_p + mu + log(N_ep * 15/12)  #yearly correction term
    Y_ep       ~ Pois(lam_ep)

"""

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.misc as scpm
import pandas as pd
import os


# credit dataset
def process_dataset():
    data_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(data_dir, 'data/frisk/frisk_with_noise.dat'), skiprows=6, delim_whitespace=True)

    # compute proportion black in precinct, black = 1
    # first aggregate by precinct/ethnicity, and sum over populations
    popdf = df[['pop', 'precinct', 'eth']]. \
                groupby(['precinct', 'eth'])['pop'].apply(sum)
    percent_black = np.array([ popdf[i][1] / float(popdf[i].sum())
                               for i in xrange(1, 76)] )
    precinct_type = pd.cut(percent_black, [0, .1, .4, 1.])  #
    df['precinct_type'] = precinct_type.codes[df.precinct.values-1]
    return df

df = process_dataset()

def make_model_funs(crime=1., precinct_type=0):
    """ crime: 1=violent, 2=weapons, 3=property, 4=drug
        eth  : 1=black, 2 = hispanic, 3=white
        precincts: 1-75
        precinct_type = (0, .1], (.1, .4], (.4, 1.]
    """

    # subselect crime/precinct, set up design matrix
    sdf = df[ (df['crime']==crime) & (df['precinct_type']==precinct_type) ]

    # make dummies for precincts, etc
    one_hot   = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
    precincts = np.sort(np.unique(sdf['precinct']))
    Xprecinct = one_hot(sdf['precinct'], 76)[:, precincts]
    Xeth      = one_hot(sdf['eth'], 4)[:, 1:-1]
    yep       = sdf['stops'].values
    lnep      = np.log(sdf['past.arrests'].values) + np.log(15./12)
    num_eth      = Xeth.shape[1]
    num_precinct = Xprecinct.shape[1]

    # unpack a flat param vector
    aslice     = slice(0, num_eth)
    bslice     = slice(num_eth, num_eth + num_precinct)
    mslice     = slice(bslice.stop, bslice.stop + 1)
    lnsa_slice = slice(mslice.stop, mslice.stop + 1)
    lnsb_slice = slice(lnsa_slice.stop, lnsa_slice.stop+1)
    num_params = lnsb_slice.stop

    pname = lambda s, stub: ['%s_%d'%(stub, i)
                             for i in xrange(s.stop-s.start)]
    param_names = [pname(s, stub)
          for s, stub in zip([aslice, bslice, mslice, lnsa_slice, lnsb_slice],
                             ['alpha', 'beta', 'mu', 'lnsigma_a', 'lnsigma_b'])]
    param_names = [s for pn in param_names for s in pn]

    def unpack(th):
        """ unpack vectorized lndf """
        th = np.atleast_2d(th)
        alpha_eth, beta_prec, mu, lnsigma_alpha, lnsigma_beta = \
            th[:, aslice], th[:, bslice], th[:, mslice], \
            th[:, lnsa_slice], th[:, lnsb_slice]
        return alpha_eth, beta_prec, mu, lnsigma_alpha, lnsigma_beta

    hyper_lnstd = np.array([[np.log(10.)]])

    def lnpdf(th):
        # params
        alpha, beta, mu, lns_alpha, lns_beta = unpack(th)
        # priors
        ll_alpha  = normal_lnpdf(alpha,     0, lns_alpha)
        ll_beta   = normal_lnpdf(beta,      0, lns_beta)
        ll_mu     = normal_lnpdf(mu,        0, hyper_lnstd)
        ll_salpha = normal_lnpdf(np.exp(lns_alpha), 0, hyper_lnstd)
        ll_sbeta  = normal_lnpdf(np.exp(lns_beta),  0, hyper_lnstd)
        logprior  = ll_alpha + ll_beta + ll_mu + ll_salpha + ll_sbeta

        # likelihood
        lnlam = (mu + lnep[None,:]) + \
                np.dot(alpha, Xeth.T) + np.dot(beta, Xprecinct.T)
        loglike = np.sum(lnpoiss(yep, lnlam), 1)
        return (loglike + logprior).squeeze()

    return lnpdf, unpack, num_params, sdf, param_names


from scipy.special import gammaln
def lnpoiss(y, lnlam):
    """ log likelihood of poisson """
    return y*lnlam - np.exp(lnlam) - gammaln(y+1)

def normal_lnpdf(x, mean, ln_std):
    x = np.atleast_2d(x)
    D = x.shape[1]
    dcoef = 1.
    if ln_std.shape[1] != D:
        dcoef = D
    qterm = -.5 * np.sum((x - mean)**2 / np.exp(2.*ln_std), axis=1)
    coef  = -.5*D * np.log(2.*np.pi) - dcoef * np.sum(ln_std, axis=1)
    return qterm + coef

