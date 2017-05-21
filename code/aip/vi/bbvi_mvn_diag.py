import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad, jacobian, hessian
from scipy.stats import ncx2
from aip.misc import mvn_diag_logpdf, mvn_diag_entropy
from .bbvi_base import BBVI


class DiagMvnBBVI(BBVI):

    def __init__(self, lnpdf, D, glnpdf=None, lnpdf_is_vectorized=False):
        """
        Implements MCVI --- exposes elbo gradient and sampling methods.
        This class breaks the gradient down into parts

        dg/dz = dlnpdf(z)/dz * dz/dlam - dlnq(z)/dz * dz/dlam - dlnq(z)/dlam

        Parameterizes with mean and log-std! (not variance!)
            lam = [mean, log-std]
        """
        # base class sets up the gradient function organization
        super(DiagMvnBBVI, self).__init__(lnpdf, D, glnpdf, lnpdf_is_vectorized)

        # we note that the second two terms, with probability one, 
        # create the vector [0, 0, 0, ..., 0, 1., 1., ..., 1.]
        self.mask = np.concatenate([np.zeros(D), np.ones(D)])
        self.num_variational_params = 2*D
        self.D = D


    #####################################################################
    # Methods for various types of gradients of the ELBO                #
    #    -- that can be plugged into FilteredOptimization routines      #
    #####################################################################

    def elbo_grad_mc(self, lam, t, n_samps=1, eps=None):
        """ monte carlo approximation of the _negative_ ELBO """
        if eps is None:
            eps = np.random.randn(n_samps, self.D)
        return -1.*np.mean(self.dlnp(lam, eps) + self.mask, axis=0)

    def elbo_grad_delta_approx(self, lam, t):
        """ delta method approximation of the _negative_ ELBO """
        gmu, gvar = self.delta_grad(lam)
        return -1 * gmu

    def elbo_grad_fixed_mixture_approx(self, lam, t, n_samps=1, rho = .5):
        """ combine a sample w/ the elbo grad mean """
        gmu, gvar = self.delta_grad(lam)
        gmu   = -1. * gmu
        gsamp = self.elbo_grad_mc(lam, t, n_samps=n_samps)
        return (rho*gsamp + (1-rho)*gmu)

    def elbo_grad_adaptive_mixture_approx(self, lam, t, n_samps=1,
                                                        use_hessian=False):
        # seed randomness
        eps = np.random.randn(n_samps, self.D)

        # data term distribution approximation, E[ dlnp / dz ] and var
        dmu, ds = self.dlnp_delta_approx(lam, use_hessian=use_hessian)

        # generate samples of dlnp/dz
        dlnp_dz_samp = np.mean(self.dlnp(lam, eps), axis=0)

        # component-wise log likelihood
        zscore = np.exp( -(.5/(ds**2)) * (dlnp_dz_samp[:self.D] - dmu)**2 )
        print "comp score: ", zscore.min(), zscore.max()
        rhos = np.concatenate([zscore, zscore])

        # generate gradient sample using same eps
        gsamp     = dlnp_dz_samp + self.mask
        gmu, gvar = self.delta_grad(lam)
        return -1. * (rhos*gmu + (1-rhos)*gsamp)

    def nat_grad(self, lam, standard_gradient):
        finv = 1./self.fisher_info(lam)
        return finv * standard_gradient

    #############################
    # ELBO objective functions  #
    #############################

    def elbo_mc(self, lam, n_samps=100, full_monte_carlo=False):
        """ approximate the ELBO with samples """
        D = len(lam)/2
        zs  = self.sample_z(lam, n_samps=n_samps)
        if full_monte_carlo:
            elbo_vals = self.lnpdf(zs) - mvn_diag_logpdf(zs, lam[:D], lam[D:])
        else:
            elbo_vals = self.lnpdf(zs) + mvn_diag_entropy(lam[D:])
        return np.mean(elbo_vals)

    def true_elbo(self, lam, t):
        """ approximates the ELBO with 20k samples """
        return self.elbo_mc(lam, n_samps=20000)

    def sample_z(self, lam, n_samps=1, eps=None):
        """ sample from the variational distribution """
        D = self.D
        assert len(lam) == 2*D, "bad parameter length"
        if eps is None:
            eps = np.random.randn(n_samps, D)
        z = np.exp(lam[D:]) * eps + lam[None, :D]
        return z

    #####################################################
    # utility functions                                 #
    #####################################################

    def dlnp(self, lam, eps):
        """ the first gradient term (data term), computes
                dlnp/dz * dz/dlambda

            If `lnpdf` is vectorized ---, eps can 

        Args:
            lam = [mean, log-std], 2xD length array
            eps = Nsamps x D matrix, for sampling randomness 
                  z_0 ~ N(0,1)
        """
        # sample from the variational distribution using seed randomness `eps`
        z       = self.sample_z(lam, eps=eps)
        log_s   = lam[self.D:]

        # compute the gradient of the data term
        dlnp_dz = self.glnpdf(z)

        # multiply the gradient of the data term with dz/dlam
        # dz/dlam  = [ I_D ; diag(eps) ]
        # ds/dlogs = exp(s)  (the log std term needs
        #                     an extra multiplication here)
        gt = np.column_stack([dlnp_dz, dlnp_dz * (eps*np.exp(log_s))])
        return gt

    def dlnq(self, lam, eps):
        """ it turns out this isn't totally necessary ... """
        D = len(eps)
        dlnq_dz = -eps * np.exp(-lam[D:])
        return np.concatenate([dlnq_dz, dlnq_dz * (eps * np.exp(lam[D:]))])

    def dlnq_dlam(self, lam, eps):
        """ d lnq / d lam, holding z constant.
        This term can be used as a control variate --- might be useful to track
        the correlation of this term with the true elbo gradient values
        over time.

        TODO: we should keep a running estimate of correlation...
        """
        sinv = np.exp(-lam[eslf.D:])
        return np.concatenate([eps*sinv, ((eps**2) - 1)*sinv*np.exp(lam[D:])])


    #########################################################
    # create variance approximation to the first term, dlnp #
    #########################################################

    def delta_grad(self, lam, use_hessian=False):
        gmu, gvar = self.dlnp_dlam_approx(lam, use_hessian)
        return gmu + self.mask, gvar

    def dlnp_delta_approx(self, lam, approx_type="trace", eps_mat=None):
        """ compute the normal approximation to the data term gradient,
            (dlnp / dz).

                z    ~ N(mu, s^2) implies approximately
                g(z) ~ N(g(mu), (g'(mu) * s)^2)

            Note that the generative process is sensitive to the sign
            of g'(mu) ---

                dlnp_dz_approx ~ g(mu) + g'(mu)*s * eps_0

            is the normal approximation to dlnp_dz, with seed
            randomness eps_0.

        Args:
            - lam: variational parameters
            - approx_type: the type of approximation made.  currently we have
                - "hessian": computes full hessian
                - "hessian_diag": computes true hessian diag (also slow)
                - "trace": computes gradient of gradient sum --- hessian diag if hessian is diagonal
                - "random": computes a random direction
                - "lstsq": computes least square between eps and grad

            - eps_mat: [Nsamps x L x D] normal(0, 1) to generate sample
                       this indicates that it is an L-sample estimator.

        Returns:
            - gmu: simply the gradient of lnpdf at mu_lam
            - gs : the elementwise gradient of gradient * sig_lam.
                   This is the signed sqroot of the variance of the approx.
        """
        D             = len(lam)/2
        mu_lam, s_lam = lam[:D], np.exp(lam[D:])
        gmu           = self.glnpdf(mu_lam)
        if approx_type=="hessian":
            H   = self.hlnpdf(mu_lam)
            hht = np.dot(H, H.T)
            gs  = np.sqrt(np.diag(hht)) * s_lam
        elif hessian_type=="diag":
            H  = self.hlnpdf(mu_lam)
            gs = np.diag(H) * s_lam
        elif hessian_type=="trace":
            gs = self.gglnpdf(mu_lam) * s_lam
        elif hessian_type=="rand":
            gs = np.random.randn(self.D) * s_lam
        # returns mean and (signed) standard deviation
        return gmu, gs

    def dlnp_dlam_approx(self, lam, use_hessian=False):
        # compute the normal approximation to dlnp_dz)
        D = len(lam)/2
        mz, sz = self.dlnp_delta_approx(lam)

        # mean/variance for mean component (m)
        gmu_m  = mz
        gvar_m = sz**2

        # mean/variance for sigma component
        #  --- the sigma component is essentially a location scaled xi
        slam   = np.exp(lam[D:])
        gmu_s  = sz * slam            # times e^lam for the transformation
        gvar_s = (mz*mz + 3*sz*sz)*slam*slam

        # multiply this by the 
        return np.concatenate([gmu_m, gmu_s]), \
               np.concatenate([gvar_m, gvar_s])

    def delta_grad_variance_approx(self, lam, dim):
        """ the variance variational parameters have a special form ---
        a location shifted non-central chi sq

            if we approximate dlnp / dz with the delta method as 

                dlnp/dz ~ N(m_p, s_p^2)

            then the variance component is 

                dL/dv ~ dlnp/dz * z_0 * s  + 1
                      ~ (m_p + s_p * z0) * z_0 * s + 1

            where s = exp(v).  completing the sq, w get
                dL/dv ~ s_p*s * [ (z_0 + m_p/2s_p)^2 - mp^2/(4sp^2) + 1/(sp*s)]

        """
        mp, sp = self.dlnp_delta_approx(lam)
        mp, sp = mp[dim], sp[dim]
        s      = np.exp(lam[self.D:])[dim]
        return loc_shift_nc_chi_sq_grid(mp, sp, s, m=1.)

    def dlnp_dlam_sample_approx(self, lam, n_samps=1000, d=None):
        D = len(lam)/2
        m, s   = self.dlnp_delta_approx(lam)
        eps    = np.random.randn(n_samps, D)

        # samples of mean parameter gradient
        samps  = eps*s + m

        # samples corresponding to variance parameter gradient
        vsamps = (eps*s + m) * eps * np.exp(lam[D:])

        if d is not None:
            # write out location-shifted non-central chi sq params
            mm = (m * np.exp(lam[D:]))[d]
            ss = (s * np.exp(lam[D:]))[d]

            scale_sign = np.sign(ss)
            scale      = ss
            nc         = mm / (2*ss)
            loc        = - (mm*mm) / (4*ss)

            # return sampled grid as well
            xgrid   = np.linspace(vsamps[:,d].min(), vsamps[:,d].max(), 100)
            lnpgrid = ncx2.logpdf(np.sign(scale)*xgrid,
                                  df=1, nc=nc*nc, scale=np.abs(scale),
                                  loc=np.sign(scale)*loc)
            return np.column_stack([samps, vsamps]), (xgrid, np.exp(lnpgrid))

        return np.column_stack([samps, vsamps])

    def dlnp_dz_samps(self, lam, n_samps=10):
        """
        returns dlnp/dz samples
        """
        import pyprind
        D = self.D
        samps = []
        for n in pyprind.prog_bar(xrange(n_samps)):
            eps = np.random.randn(D)
            z   = self.sample_z(lam, eps=eps)
            #z   = lam[:D] + np.exp(lam[D:])*eps
            samps.append(self.glnpdf(z))
        return np.array(samps).squeeze()

    def fisher_info(self, lam):
        """ returns the fisher information matrix (diagonal) for a multivariate
        normal distribution with params = [mu, ln sigma] """
        D = len(lam) / 2
        mean, log_std = lam[:D], lam[D:]
        return np.concatenate([np.exp(-2.*log_std), 2*np.ones(D)])

    ##### plotting functions #####

    def compare_delta_approx_to_samples(self, lam, dims = [0, 1],
                                              grad_samps = None,
                                              axarr      = None,
                                              use_hessian=False):
        import matplotlib.pyplot as plt; plt.ion()
        from aip.vboost import plots as pu
        import pyprind
        if axarr is None:
            fig, axarr = plt.subplots(1, len(dims), figsize=(12,6))

        # true samples of the gradient
        if grad_samps is None:
            grad_samps = np.array([ self.elbo_grad_mc(lam, t=0, n_samps=1)
                                    for _ in pyprind.prog_bar(xrange(2000)) ])

        # delta approximation to the gradient (mean/var)
        gmu, gvar = self.delta_grad(lam, use_hessian=use_hessian)
        print "Delta grad gmu, gvar", gmu, gvar

        #for each dimension --- plot
        for dim, ax in zip(dims, axarr.flatten()):

            # plot true grad samples
            b, n, _ = ax.hist(grad_samps[:, dim], bins=30, normed=True)
            xlim, ylim = (n.min(), n.max()), ax.get_ylim()

            # compute plot bounds (Based on our approximation)
            xmin, xmax = gmu[dim] - 3*np.sqrt(gvar[dim]), \
                         gmu[dim] + 3*np.sqrt(gvar[dim])

            # plot approximate distribution PDF
            if dim < self.D:
                print "  plotting mean param %d"%dim
                glam = np.concatenate([gmu, .5*np.log(gvar)])
                xgrid = np.linspace(xmin, xmax, 100)
                pu.plot_normal_marginal(dim, glam, xgrid=xgrid, ax=ax)
                ax.set_title("variational mean %d"%dim)
            else:
                print "  plotting log var param %d"%dim
                xgrid, pgrid = self.delta_grad_variance_approx(lam, self.D - dim)
                #gsamps, (xgrid, pgrid) = \
                #    self.dlnp_dlam_sample_approx(lam, d=dim-self.D)
                ax.plot(xgrid,pgrid)
                xmin = np.max([n.min(), xmin])
                xmax = np.min([n.max(), xmax])
                ax.set_title("variational log std %d"%dim)

            ax.plot([gmu[dim], gmu[dim]], [0., b.max()],
                    c='red', label="delta approx mean = %2.2f"%gmu[dim])

            dmu = grad_samps[:,dim].mean()
            ax.plot([dmu, dmu], [0., b.max()], c='blue', label="sample mean")
            #ax.set_xlim((xmin, xmax))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend()

        return axarr

    def visualize_delta_approx_dlnq_dlam(self, lam, dims        = [0,1],
                                                    axarr       = None,
                                                    use_hessian = False,
                                                    n_samps     = 1000):
        import matplotlib.pyplot as plt; plt.ion()
        from aip.vboost import plots as pu
        import pyprind
        import scipy.stats as ss
        if axarr is None:
            fig, axarr = plt.subplots(1, len(dims), figsize=(12,6))

        # delta method approx
        gmu, gs = self.dlnp_delta_approx(lam, use_hessian=use_hessian)
        print "gmu ... ", gmu
        print "gs ...  ", gs

        # real gradient samples
        dlnp_dz_samps = self.dlnp_dz_samps(lam, n_samps=n_samps)
        print "samps shape:", dlnp_dz_samps.shape

        # compare samps and approx
        for dim, ax in zip(dims, axarr.flatten()):
            samps = dlnp_dz_samps[:,dim]
            print "skew, kurt: ", ss.skew(samps), ss.kurtosis(samps)
            n, bins, p = ax.hist(samps, bins=25, normed=True,
                                                 label="dlnp/dz samples")

            glam  = np.concatenate([gmu, np.log(np.abs(gs))])
            xgrid = np.linspace(samps.min(), samps.max(), 100)
            pu.plot_normal_marginal(dim, glam, xgrid=xgrid, ax=ax)
            ax.set_title("dlnp/dz vs delta approx dim %d"%dim)

            # plot means
            ax.plot([gmu[dim], gmu[dim]], [0, np.max(n)],
                    label="gauss mean")
            ax.plot([samps.mean(), samps.mean()], [0, np.max(n)],
                    label="samp mean")
            ax.legend()
        return axarr

    def callback(self, th, t, g, tskip=20, n_samps=10):
        """ custom callback --- prints statistics of all gradient comps"""
        if t % tskip == 0:
            fval = self.elbo_mc(th, n_samps=n_samps)
            gm, gv = np.abs(g[:self.D]), np.abs(g[self.D:])
            print \
"""
iter {t}; val = {val}, abs gm = {m} [{mlo}, {mhi}]
                           gv = {v} [{vlo}, {vhi}]
""".format(t=t, val="%2.4f"%fval,
                m  ="%2.4f"%np.mean(gm),
                mlo="%2.4f"%np.percentile(gm, 1.),
                mhi="%2.4f"%np.percentile(gm, 99.),
                v  ="%2.4f"%np.mean(gv),
                vlo="%2.4f"%np.percentile(gv, 1.),
                vhi="%2.4f"%np.percentile(gv, 99.))


def loc_shift_nc_chi_sq_sample(mp, sp, s, m, nsamps=10000):
    """ samples 
        (mp + sp*eps)*eps*s + m
    """
    eps = np.random.randn(nsamps)
    return (mp + sp*eps)*eps*s + m

def loc_shift_nc_chi_sq_grid(mp, sp, s, m):
    """ samples 
        (mp + sp*eps)*eps*s + m
    """
    from scipy.stats import ncx2
    scale      = sp * s
    scale_sign = np.sign(sp*s)
    nc         = mp / (2*sp)
    loc        = - ((mp**2) *s) / (4*sp) + m

    # create lim/grid
    xlim = np.sort(ncx2.ppf(q=[.01, .99], nc=nc*nc, df=1)*scale + loc)
    print "scale, nc, loc:", scale, nc, loc
    print xlim
    xgrid = np.linspace(xlim[0], xlim[1], 100)
    lgrid = ncx2.logpdf((xgrid-loc)/scale, nc=nc*nc, df=1) - np.log(np.abs(scale))#, scale=scale, loc=loc)
    return xgrid, np.exp(lgrid)
