import numpy as np
import matplotlib.pyplot as plt

def distribution_plot(Dis, color = 'k', label = '_Hidden Label',
               marker = None,linewidth = 2, alpha =1):
    x = Dis[0]; y = Dis[1]
    markersize = None
    if marker is not None:
        markersize = 3
    plt.plot(x, y, color=color, linewidth=linewidth,
             marker = marker, markersize = markersize ,alpha=alpha, label=label)
    plt.xscale('log')
    plt.yscale('log')

def distribution_plot_fit(Dis, tau, xmin = 1, color = 'lightgreen', CCDF = True,
                   label = '_Hidden Label', linewidth = 2, alpha =1, shift =3):
    x = Dis[0]; y = Dis[1]
    x_fit = x[ x >= xmin ]
    y_fit = x_fit**(-tau)
    y_rescaling = 10 ** np.interp(np.log10(xmin), np.log10(x), np.log10(y))
    if xmin > 0:
        y_fit = shift *(y_fit * y_rescaling)/y_fit[0]

    label = f'{label} = {tau : .2f} '
    plt.plot(x_fit, y_fit, color=color, linewidth=linewidth, linestyle='--', alpha=alpha, label=label)
    plt.xscale('log')
    plt.yscale('log')

def ST_plot(s_avg, d_avg, dot_size =10,color = 'k', label = '_Hidden Label', alpha =.2):
    plt.scatter(d_avg,s_avg, color=color, s=dot_size, alpha=alpha, label=label)
    plt.xscale('log')
    plt.yscale('log')

def ST_plot_fit(s_avg, d_avg, gamma= 2, shift =-2.5,
                color = 'lightgreen', label = r'$\gamma$', alpha =1, linewidth =2):
    intercept = np.log10(s_avg[0])+ shift

    x_line = np.linspace(min(d_avg), max(d_avg), 100)
    y_line = 10 ** (intercept + gamma * np.log10(x_line))
    plt.plot(x_line, y_line, color=color, linestyle='--',
            linewidth=linewidth, alpha=alpha, label=label)
    plt.xscale('log');plt.yscale('log')


from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def gamma_shape_collapse(
        avals,
        *,
        min_T=40,
        max_T=400,
        min_count=20,
        n_grid=50,
        g_min=1.2,
        g_max=2.4,
        g_step=0.01):

    # ---------- 1.  build mean profile for every qualifying T ----------
    dur = np.fromiter((len(a) for a in avals), dtype=int)
    uniqT, counts = np.unique(dur, return_counts=True)

    keepT = [T for T, c in zip(uniqT, counts)
             if min_T <= T <= max_T and c >= min_count]
    if not keepT:
        raise ValueError("No durations satisfy (min_T,max_T,min_count).")

    u_grid = np.linspace(0, 1, n_grid, endpoint=False) + 0.5 / n_grid
    resampled, durations = [], []

    for T in keepT:
        curves_T = [a for a in avals if len(a) == T]
        mean_T = np.mean(curves_T, axis=0)  # mean shape for this T
        t_axis = (np.arange(T) + 0.5) / T
        resampled.append(np.interp(u_grid, t_axis, mean_T))
        durations.append(T)

    resampled = np.vstack(resampled)  # (n_T , n_grid)
    durations = np.asarray(durations)  # (n_T ,)

    # ---------- 2.  NMSE collapse metric ------------------------------
    def nmse(gamma):
        scaled = resampled / durations[:, None] ** (gamma - 1.0)
        mean = scaled.mean(axis=0)
        return np.sum((scaled - mean) ** 2) / np.sum(mean ** 2)

    gamma_grid = np.arange(g_min, g_max + 1e-9, g_step)
    nmse_grid = np.array([nmse(g) for g in gamma_grid])

    idx_best = nmse_grid.argmin()
    gamma_best = gamma_grid[idx_best]
    nmse_best = nmse_grid[idx_best]

    # ---------- 3.  error bar via local parabola ----------------------
    lo = max(idx_best - 2, 0)
    hi = min(idx_best + 3, len(gamma_grid))
    p, _ = curve_fit(lambda x, a, b, c: a * x ** 2 + b * x + c,
                     gamma_grid[lo:hi], nmse_grid[lo:hi],
                     p0=(1, -2 * gamma_best, nmse_best))
    a = float(p[0])
    sigma_gamma = np.sqrt(1 / (2 * a)) if a > 0 else np.nan

    return gamma_best, sigma_gamma, nmse_best, {
        'gamma_grid': gamma_grid,
        'nmse_grid': nmse_grid,
        'durations': durations,
        'profiles': resampled
    }


def shape_collapse_plot(s_shape_prime, Tmin=30, Tmax=300, gamma=None,
                        cmap=plt.cm.rainbow_r, plot_every=3, alpha=.5, linewidth=3,
                        f_axis=15, f_ticks=13, f_title=15):
    from matplotlib.ticker import ScalarFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np
    import matplotlib.pyplot as plt


    if gamma == None:
        gamma_star, sigma, nmse_min, diagnostics = gamma_shape_collapse(
            s_shape_prime,  # your list/array of avalanches
            min_T=Tmin,
            max_T=Tmax,
            min_count=20,
            n_grid=50,
            g_min=1.3,
            g_max=2.0,
            g_step=0.01)
        gamma = gamma_star
        print(f"\nBest γ  = {gamma_star:.4f} ± {sigma:.4f}   (NMSE = {nmse_min:.4g}) and your γ  = {gamma}")

    t = np.array([len(x) for x in s_shape_prime])
    T_list = np.array(sorted(list(set(t))))
    T_list = T_list[np.where(T_list < Tmax)]
    T_list = T_list[np.where(T_list > Tmin)]

    norm = mcolors.Normalize(vmin=np.min(T_list), vmax=np.max(T_list))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])

    for t_ in range(len(T_list)):
        T = int(T_list[t_])
        if t_ % plot_every == 0:
            s_shape_avg = [x for x in s_shape_prime if len(x) == T]
            if len(s_shape_avg) > 0:
                y = np.mean(np.array(s_shape_avg), axis=0)
                x = np.linspace(0, len(y), len(y))
                plt.plot((x) / (T), y / (T) ** (gamma - 1),
                         color=scalar_map.to_rgba(T),
                         linewidth=linewidth, alpha=alpha)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.tick_params(axis='both', labelsize=f_ticks)
    plt.xlabel(r"$\frac{\ell}{D}$", fontsize=f_axis)
    plt.ylabel(r"$\frac{\bar V}{D^{\gamma^{*} - 1}}$", fontsize=f_axis)
    plt.title('(f)', fontsize=f_title)

    # Place colorbar as inset *after* labels are set
    inset_ax = inset_axes(plt.gca(), width="3%", height="30%", loc='lower center', borderpad=1)
    cbar = plt.colorbar(scalar_map, cax=inset_ax)
    cbar.set_label(r'$D$', fontsize=f_ticks)
