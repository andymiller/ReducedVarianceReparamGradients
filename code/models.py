import autograd.numpy as np
from aip.models import frisk, uci, nn
from scipy.stats import multivariate_normal as mvn


def set_lnpdf(model="baseball", dset="boston"):
    if model == "baseball":
        return lambda x: np.squeeze(baseball.lnpdf_flat(x, 0)), baseball.D, model
    if model == "frisk":
        lnpdf, unpack, num_params, frisk_df, param_names = \
            frisk.make_model_funs(crime=2., precinct_type=1)
        return lnpdf, num_params, model
    if model == "normal":
        D, r       = 10, 2
        mu0        = np.zeros(D)
        C_true     = np.random.randn(D, r) * 2.
        v_true     = np.random.randn(D)
        Sigma_true = np.dot(C_true, C_true.T) + np.diag(np.exp(v_true))
        print Sigma_true
        lnpdf = lambda x: misc.make_fixed_cov_mvn_logpdf(Sigma_true)(x, mean=mu0)
        return lnpdf, D, model
    if model == "bnn":
        (Xtrain, Ytrain), (Xtest, Ytest) = \
            uci.load_dataset(dset, split_seed=0)
        lnpdf, predict, loglike, parser, (std_X, ustd_X), (std_Y, ustd_Y) = \
            nn.make_nn_regression_funs(Xtrain[:100], Ytrain[:100],
                                       layer_sizes=None, obs_variance=None)
        lnpdf_vec = lambda ths: np.array([lnpdf(th)
                                          for th in np.atleast_2d(ths)])
        return lnpdf_vec, parser.N, "-".join([model, dset])

