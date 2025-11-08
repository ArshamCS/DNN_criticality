import numpy as np

############ DATA ############################################################

def distribution(x,
                 *,
                 discrete=False,
                 xmin=None,
                 xmax=None,
                 min_count=100):

    # ---------- helpers ----------
    def _renorm_by_area(heights, widths):
        Z = float(np.sum(heights * widths))
        if Z > 0 and np.isfinite(Z):
            return heights / Z
        return heights

    def _geomean_pairwise(a, b):
        # geometric mean of two arrays (assumed >0)
        return np.sqrt(a * b)

    # --- clean and crop ----------------------------------------------------
    x = np.asarray(x, float)
    mask = np.isfinite(x) & (x > 0)
    x = x[mask]
    if x.size == 0:
        return np.array([]), np.array([])

    if xmin is None: xmin = x.min()
    if xmax is None: xmax = x.max()
    xmin = float(xmin); xmax = float(xmax)
    if xmin <= 0:
        xmin = np.nextafter(0, 1)
    x = x[(x >= xmin) & (x <= xmax)]
    if x.size == 0:
        return np.array([]), np.array([])

    if discrete:
        # Treat x as integers (round if needed)
        k = np.rint(x).astype(int)
        k = k[k >= 1]
        if k.size == 0:
            return np.array([]), np.array([])

        vals, cnts = np.unique(k, return_counts=True)
        N = cnts.sum()

        # Merge consecutive integers until >= min_count
        edges_idx = [0]
        acc = 0
        for i, c in enumerate(cnts):
            acc += c
            if acc >= min_count:
                edges_idx.append(i + 1)
                acc = 0
        if edges_idx[-1] != len(vals):
            edges_idx.append(len(vals))

        centers, heights, widths_int = [], [], []
        for a, b in zip(edges_idx[:-1], edges_idx[1:]):
            v_seg = vals[a:b]
            c_seg = cnts[a:b].sum()
            if c_seg == 0:
                continue
            # geometric mean center (positive ints)
            center = np.exp(np.mean(np.log(v_seg)))
            width_int = v_seg[-1] - v_seg[0] + 1  # number of integers covered
            mass = c_seg / N                      # PMF mass per merged bin
            height = mass / width_int         # density-like height

            centers.append(center)
            heights.append(height)
            widths_int.append(width_int)

        centers = np.asarray(centers, float)
        heights = np.asarray(heights, float)
        widths_int = np.asarray(widths_int, float)
        heights = _renorm_by_area(heights, widths_int)
        m = (centers > 0) & (heights > 0) & np.isfinite(heights)
        return centers[m], heights[m]
    else:
        # Continuous case

        x_sorted = np.sort(x)
        edges = [xmin]
        counts = []
        n = x_sorted.size
        i = 0
        while i < n:
            j = min(n, i + min_count)
            bin_max = x_sorted[j - 1]
            if bin_max <= edges[-1]:
                # ensure strictly increasing edges
                j += 1
                if j > n:
                    break
                bin_max = x_sorted[j - 1]
            edges.append(bin_max)
            counts.append(j - i)
            i = j

        edges = np.asarray(edges, float)
        counts = np.asarray(counts, float)
        widths  = np.diff(edges)
        centers = _geomean_pairwise(edges[:-1], edges[1:])

        m = (widths > 0) & (counts > 0) & (centers > 0)
        widths, centers, counts = widths[m], centers[m], counts[m]

        heights = counts / widths  # histogram height
        heights = _renorm_by_area(heights, widths)

        mm = (heights > 0) & np.isfinite(heights)
        return centers[mm], heights[mm]


def avgS_D(s,d, min_count = 0):
    d_sorted_set = sorted(list(set(d)))
    d_avgS = np.array([(T, np.mean(s[np.where(d == T)])) for T in d_sorted_set if len(s[np.where(d == T)])> min_count ])
    d_avg = np.array([x[0] for x in d_avgS])
    s_avg = np.array([x[1] for x in d_avgS])
    return s_avg, d_avg

########### Filter ##############################################################################

def coupled_filter(s, d, Dmin=1, Dmax=None, Smin=0, Smax=None):
    """
    Filters a (2, N) array x where x[0] = s, x[1] = t.
    Removes entries where:
        - s < Smin or (Smax is not None and s > Smax)
        - t < Tmin or (Tmax is not None and t > Tmax)
    Returns the filtered (2, N_filtered) array.
    Prints lengths before and after filtering.
    """
    bad_s = (s < Smin)
    if Smax is not None:
        bad_s |= (s > Smax)
    bad_d = (d < Dmin)
    if Dmax is not None:
        bad_d |= (d > Tmax)
    bad_indices = bad_s | bad_d

    # Indices to keep
    keep = ~bad_indices

    s_filt = s[keep]
    d_filt = d[keep]
    return s_filt, d_filt

def single_filter(x, xmin=1, xmax=None):
    N_before = len(x)
    # Find indices to remove
    bad_x = (x < xmin)
    if xmax is not None:
        bad_x |= (x > xmax)
    # Indices to keep
    keep = ~bad_x
    N_after = np.sum(keep)

    #print(f"Single filtering: {N_before} -> {N_after}")

    x_filt = x[keep]
    return x_filt

########## FITTING #############################################################################

def MLE_exponent_tau(x,xmin=1,discrete=True):
    import powerlaw
    fit = powerlaw.Fit(x, xmin=xmin, discrete=discrete, verbose=False)

    tau = fit.power_law.alpha
    tau_err = fit.power_law.sigma
    return tau, tau_err


import numpy as np

def estimate_gamma_SD(
    S, D, *,
    min_count=10, Dmin=None, Dmax=None,
    weighted=True, eps_var=1e-8
):
    """
    Estimate gamma from S ~ D^gamma via conditional ⟨S|D⟩ and log-log regression.

    Steps:
      1) Keep events with Dmin <= D <= Dmax (if given) and finite, positive S.
      2) For each integer D value with at least `min_count` events, compute:
           - M1(D) = mean(S | D)
           - counts(D)
           - var_logS(D) = var(log10 S | D)  [for weights]
      3) Regress log10 M1 vs log10 D using:
           - Weighted (York-style): w = counts / var_logS
           - Unweighted OLS otherwise
      4) Return gamma, gamma_err, intercept, and per-bin series.

    Returns
    -------
    gamma_hat, gamma_err, intercept, Dvals, M1, counts
    """
    S = np.asarray(S)
    D = np.asarray(D)

    # basic cleaning
    m = np.isfinite(S) & np.isfinite(D) & (S > 0)
    if Dmin is not None: m &= (D >= Dmin)
    if Dmax is not None: m &= (D <= Dmax)
    S, D = S[m], D[m].astype(int)

    # aggregate per integer D
    Dvals, M1, counts, var_logS = [], [], [], []
    for dval in np.unique(D):
        idx = (D == dval)
        n = int(np.sum(idx))
        if n < min_count:
            continue
        S_bin = S[idx]
        if np.any(S_bin <= 0):
            continue
        Dvals.append(dval)
        M1.append(float(np.mean(S_bin)))
        counts.append(n)
        v = np.var(np.log10(S_bin), ddof=1) if n > 1 else np.nan
        var_logS.append(v if np.isfinite(v) and v > 0 else eps_var)

    Dvals = np.asarray(Dvals, dtype=int)
    M1 = np.asarray(M1, dtype=float)
    counts = np.asarray(counts, dtype=int)
    var_logS = np.asarray(var_logS, dtype=float)

    if len(Dvals) < 2:
        raise ValueError("Not enough D bins after filtering to perform regression.")

    x = np.log10(Dvals)
    y = np.log10(M1)

    if weighted:
        w = counts / (var_logS + eps_var)
        W = np.sum(w)
        x_bar = np.sum(w * x) / W
        y_bar = np.sum(w * y) / W
        Sxx = np.sum(w * (x - x_bar) ** 2)
        Sxy = np.sum(w * (x - x_bar) * (y - y_bar))
        slope = Sxy / Sxx
        resid = y - (slope * x + (y_bar - slope * x_bar))
        s2 = np.sum(w * resid**2) / max(W - 2, 1)
        std_slope = np.sqrt(s2 / Sxx)
        intercept = y_bar - slope * x_bar
    else:
        X = np.vstack([x, np.ones_like(x)]).T
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        slope, intercept = beta[0], beta[1]
        resid = y - (slope * x + intercept)
        dof = max(len(x) - 2, 1)
        s2 = np.sum(resid**2) / dof
        Sxx = np.sum((x - np.mean(x))**2)
        std_slope = np.sqrt(s2 / Sxx)

    gamma_hat = slope
    gamma_err = std_slope

    return gamma_hat, gamma_err, intercept