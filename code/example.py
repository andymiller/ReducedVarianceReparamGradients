"""
Simple example script fitting a model for synthetic data
"""
from __future__ import print_function

import autograd.numpy as np
from aip.vi.bbvi_mvn_diag import DiagMvnBBVI
from aip.gsmooth.opt import FilteredOptimization
from aip.gsmooth.smoothers import AdamFilter, SGDMomentumFilter
import control_variate_grads as cvg
import pyprind
import models

#########################################
# construct model function + vb object  #
#########################################
#lnpdf, D = make_model()
lnpdf, D, name = models.set_lnpdf("frisk")
th0 = np.random.randn(D)
print(lnpdf(th0))   # example use

# create bbvi object --- this just keeps references to lnpdf,
# grad(lnpdf), hvp(lnpdf), etc
vbobj = DiagMvnBBVI(lnpdf, D, lnpdf_is_vectorized=False)

# initialize params
np.random.seed(1)
lam0 = np.random.randn(vbobj.num_variational_params)*.01 - 1
lam0[D:] = -3.

################################
# set up optimization function #
################################
n_samps   = 2
n_iters   = 800
step_size = .05

def run_timed_opt(gradfun, num_iters):
    """ runs num_iters without computing intermediate values, 
    then computes 2000 sample elbo values (for timing)
    """
    mc_opt = FilteredOptimization(
              grad_fun    = gradfun,
              init_params = lam0.copy(),
              save_params = True,
              save_grads  = False,
              grad_filter = AdamFilter(),
              fun         = lambda lam, t: 0.,
              callback    = lambda th, t, g: 0.)
    print("  ... optimizing ")
    mc_opt.run(num_iters=num_iters, step_size=step_size)
    print("  ... wall time: %2.4f" % mc_opt.wall_clock)
    print("computing ELBO values")

    # compute ~ 50 equally spaced elbo values here
    skip = 16
    fun_vals = np.array([vbobj.elbo_mc(lam, n_samps=500)
                         for lam in pyprind.prog_bar(mc_opt.param_trace[::skip])])
    return fun_vals, mc_opt.wall_clock, mc_opt


#################################################
# define pure MC gradient function and optimize #
#################################################
print("\n ======== running MC, nsamps = %d =======" % n_samps)
def mc_grad_fun(lam, t):
    eps = np.random.randn(n_samps, D)
    return -1.*cvg.construct_cv_grads(vbobj, lam, eps,
                  method="mc").mean(0)

mc_vals, mc_wall_time, mc_opt = \
    run_timed_opt(mc_grad_fun, num_iters=3*n_iters) #about 3 x for non hvp


################################################
# define RV-RGE gradient function and optimize #
################################################
print("\n ======= running CV, nsamps = %d ======" % n_samps)
def cv_gfun(lam, t):
    eps = np.random.randn(n_samps, D)
    return -1.*cvg.construct_cv_grads(vbobj, lam, eps,
                  method="hvp_with_loo_diag_approx").mean(0)

cv_vals, cv_wall_time, cv_opt = \
    run_timed_opt(cv_gfun, num_iters=n_iters)


################
# plot results #
################
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white")

fig, ax = plt.figure(figsize=(8,4)), plt.gca()
ax.plot(mc_vals, label="MC")
ax.plot(cv_vals, label="RV-RGE")
ax.set_ylim(mc_vals[-1] - 20, cv_vals[-1]+10)
ax.legend(loc='best')
ax.set_xlabel("iteration")
ax.set_ylabel("ELBO")
ax.set_title("MC vs RV-RGE Comparison, step size = %2.3f"%step_size)

fig, ax = plt.figure(figsize=(8,4)), plt.gca()
ax.plot(np.linspace(0, mc_wall_time, len(mc_vals)), mc_vals, label="MC")
ax.plot(np.linspace(0, cv_wall_time, len(cv_vals)), cv_vals, label="RV-RGE")
ax.set_ylim(mc_vals[-1] - 20, cv_vals[-1]+10)
ax.legend(loc='best')
ax.set_xlabel("wall clock (seconds)")
ax.set_ylabel("ELBO")
ax.set_title("MC vs RV-RGE Comparison, step size = %2.3f"%step_size)
