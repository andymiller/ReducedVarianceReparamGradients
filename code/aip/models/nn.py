""" Generic MLP function for regression """
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import flatten
from autograd.optimizers import adam
from aip.misc import gamma_lnpdf


def make_nn_regression_funs(Xtrain, Ytrain,
                            layer_sizes = None, obs_variance=1.,
                            batch_size  = None):
    """ """
    assert Xtrain.ndim == 2
    assert Ytrain.ndim == 2

    # make scaling functions
    (std_X, ustd_X), (std_Y, ustd_Y), \
    (mean_Xtrain, std_Xtrain), (mean_Ytrain, std_Ytrain) \
        = make_standardize_funs(Xtrain, Ytrain)
    Xtrain = std_X(Xtrain)
    Ytrain = std_Y(Ytrain)

    # make batching functions
    batch_size   = Xtrain.shape[0] if batch_size is None else batch_size
    batch_slices = make_batches(Xtrain.shape[0], batch_size)
    n_batches    = len(batch_slices)
    print "batch slices: ", batch_slices
    print "Obs variance is : ", obs_variance

    # set up network layer architecture
    if layer_sizes is None:
        ydim = 1 if Ytrain.ndim == 1 else Ytrain.shape[1]
        layer_sizes = [Xtrain.shape[1], 50, ydim]

    # create parameter vector
    parser = WeightsParser()
    for l, lsize in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        parser.add_weights("W_%d"%l, lsize)
        parser.add_weights("b_%d"%l, lsize[1])

    parser.add_weights("lnalpha", (1,))  # inverse variance over weights
    parser.add_weights("lngamma", (1,))  # inverse observation variance
    num_layers = len(layer_sizes) - 1

    # prediction function
    def predict(th, inputs, unstandardized_data=False):
        if unstandardized_data:
            return ustd_Y(neural_net_predict(th, std_X(inputs), parser, num_layers))
        return neural_net_predict(th, inputs, parser, num_layers)

    def lnprior(params):
        # unpack params
        lnalpha = parser.get(params, "lnalpha")[0]
        lngamma = parser.get(params, "lngamma")[0]

        # weight log probability func
        ln_pw = -0.5 * (parser.N-2) * (np.log(2*np.pi)-lnalpha)
        for l in range(num_layers):
            W, b = parser.get(params, 'W_%d'%l), parser.get(params, 'b_%d'%l)
            ln_pw = ln_pw + -.5 * np.exp(lnalpha) * np.sum(W**2) \
                            -.5 * np.exp(lnalpha) * np.sum(b**2)
        a0 = 1
        b0 = 0.1
        lnprob_alpha = (a0-1) * lnalpha - b0 * np.exp(lnalpha) + lnalpha
        lnprob_gamma = (a0-1) * lngamma - b0 * np.exp(lngamma) + lngamma
        return ln_pw + lnprob_alpha + lnprob_gamma

    def lnpdf(th, batch_i=None):
        if batch_i is None:
            Xt, Yt = Xtrain, Ytrain
        else:
            bi = batch_slices[batch_i % n_batches]
            Xt, Yt = Xtrain[bi, :], Ytrain[bi, :]
        batch_ratio = float(Xtrain.shape[0]) / float(Xt.shape[0])
        # compute data loglike and log prior
        log_data_like = np.sum(loglike(th, Xt, Yt)) * batch_ratio
        log_prior     = lnprior(th)
        return log_prior + log_data_like

    def loglike(th, Xt, Yt, unstandardized_data=False):
        """ log likelihood of data with NN params th. 
        supports fixed obs_variance --- returns a value for each row
        """
        # first --- set the observation variance
        if obs_variance is None:
            v = np.exp(-1.*parser.get(th, "lngamma")[0])
        else:
            v = obs_variance

        # next --- predict using Xt.  If Xt is unstandardized
        # (e.g. unstandardized_data=True), then we will get unstandardized_Y's
        # back --- to compute ll we need to unstandardize the variance
        # parameter
        Ypred = predict(th, Xt, unstandardized_data)
        if unstandardized_data:
            v = v * (std_Ytrain[0]**2)

        # conditionally gaussian likelihood
        ll = -(.5/v) * np.sum((Ypred - Yt)**2, axis=1) \
             -.5*(np.log(2*np.pi) + np.log(v))

        return ll

    return lnpdf, predict, loglike, parser, (std_X, ustd_X), (std_Y, ustd_Y)


def neural_net_predict(params, inputs, parser, num_layers):
    for l in range(num_layers):
        W, b = parser.get(params, 'W_%d'%l), parser.get(params, 'b_%d'%l)
        outputs = np.dot(inputs, W) + b
        inputs  = np.maximum(outputs, 0.)
    return outputs


def neural_net_predict_discrete(params, inputs, parser, num_layers):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    outputs = neural_net_predict(params, inputs, parser, num_layers)
    return outputs - logsumeexp(outputs, axis=1, keepdims=True)


def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


def make_standardize_funs(Xtrain, Ytrain):
    """ functions to scale/unscale data """
    # first create scale functions
    std_Xtrain = np.std(Xtrain, 0)
    std_Xtrain[ std_Xtrain == 0 ] = 1.
    mean_Xtrain = np.mean(Xtrain, 0)

    std_Ytrain  = np.std(Ytrain, 0)
    mean_Ytrain = np.mean(Ytrain, 0)

    std_X   = lambda X: (X - mean_Xtrain) / std_Xtrain
    ustd_X  = lambda X: X*std_Xtrain + mean_Xtrain

    std_Y   = lambda Y: (Y - mean_Ytrain) / std_Ytrain
    ustd_Y  = lambda Y: Y*std_Ytrain + mean_Ytrain
    return (std_X, ustd_X), (std_Y, ustd_Y), \
           (mean_Xtrain, std_Xtrain), (mean_Ytrain, std_Ytrain)


def make_batches(N_total, N_batch):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + N_batch))
        start += N_batch
    return batches


class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

