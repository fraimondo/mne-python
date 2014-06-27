# Authors: Federico Raimondo <fraimondo@dc.uba.ar>
#
# License: BSD (3-clause)

import math
from ..utils import verbose
from .cudaica import (initDefaultConfig, setIntParameter,
                      setRealParameter,
                      setStringParameter, selectDevice,
                      checkDefaultConfig,
                      printConfig, transfer2DDataTo,
                      transferWeightsFrom,
                      preprocess, process, postprocess)


@verbose
def infomax(data, l_rate=None, block=None, w_change=1e-12,
            anneal_deg=60., anneal_step=0.9, extended=False, max_iter=200,
            cuda=True, verbose=None):
    """Run the (extended) Infomax ICA decomposition on raw data

    based on the publications of Bell & Sejnowski 1995 (Infomax)
    and Lee, Girolami & Sejnowski, 1999 (extended Infomax)

    Uses CUDAICA implementation based on publication
    CUDAICA: GPU Optimization of Infomax-ICA EEG Analysis by
    Federico Raimondo, Juan E. Kamienkowski,  Mariano Sigman
    and Diego Fernandez Slezak, 2012

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        The data to unmix.
    l_rate : float
        This quantity indicates the relative size of the change in weights.
        Note. Smaller learining rates will slow down the procedure.
        Defaults to 0.010d / alog(n_features ^ 2.0)
    block : int
        The block size of randomly chosen data segment.
        Defaults to floor(sqrt(n_times / 3d))
    w_change : float
        The change at which to stop iteration. Defaults to 1e-12.
    anneal_deg : float
        The angle at which (in degree) the learning rate will be reduced.
        Defaults to 60.0
    anneal_step : float
        The factor by which the learning rate will be reduced once
        ``anneal_deg`` is exceeded:
            l_rate *= anneal_step
        Defaults to 0.9
    extended : bool
        Wheather to use the extended infomax algorithm or not. Defaults to
        True.
    max_iter : int
        The maximum number of iterations. Defaults to 200.
    verbose : bool, str, int, or None
        if not None, override default verbose level (see mne.verbose).

    Returns
    -------
    unmixing_matrix : np.ndarray of float, shape (n_features, n_features)
        The linear unmixing operator.
    """

    # check data shape
    n_samples, n_features = data.shape

    # check input parameter
    # heuristic default - may need adjustment for
    # large or tiny data sets
    if l_rate is None:
        l_rate = 0.01 / math.log(n_features ** 2.0)

    if block is None:
        block = int(math.floor(math.sqrt(n_samples / 3.0)))

    selectDevice(0, 0)
    cfg = initDefaultConfig()
    #Compulsory: set nchannels, nsamples
    setIntParameter(cfg, 'nchannels', n_features)
    setIntParameter(cfg, 'nsamples', n_samples)

    #Optional: other parameters
    setRealParameter(cfg, 'lrate', l_rate)
    setIntParameter(cfg, 'blocksize', block)
    setRealParameter(cfg, 'nochange', w_change)
    setRealParameter(cfg, 'annealdeg', anneal_deg)
    setRealParameter(cfg, 'annealstep', anneal_step)
    setIntParameter(cfg, 'maxsteps', max_iter)
    setIntParameter(cfg, 'biasing', 1)
    setIntParameter(cfg, 'extended', 1 if extended else 0)
    setStringParameter(cfg, 'sphering', 'off')
    if verbose is None or verbose is False:
        setIntParameter(cfg, 'verbose', 0)

    #Compulsory: check configuration
    checkDefaultConfig(cfg)

    #Optional: print configuration to stdout
    printConfig(cfg)

    #transfer data
    transfer2DDataTo(data, cfg)

    #preprocess
    preprocess(cfg)

    # Main function: ICA
    process(cfg)

    postprocess(cfg)

    # sph = transferSphereFrom(cfg)
    wts = transferWeightsFrom(cfg)

    return wts.T
