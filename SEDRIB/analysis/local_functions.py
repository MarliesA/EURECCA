import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def psi(uorb, d50, delta=1.65, g=9.8):
    """
    mobility number Psi

    """
    return uorb ** 2 / delta / g / d50

def nu(Temp=10):
    return 4e-05 / (20 + Temp)

def fw_swart(Tmm10, u_orb, d50=300e-6):
    A = Tmm10*u_orb/2/np.pi # representative semi-excursion  # after Soulsby 1997 p. 77
    return np.exp(5.213*(2.5*d50/A)**0.194-5.977) # Swart 1974, as in Masselink 2007

def uz(uf, z, f=0.02, d50=0.0002, z0=0.0025, kappa=0.4, Tmm10=[], u_orb=[]):
    """ 
    as according to Puleo et al. 2012 (paragraph 5.3), with roughness height estimate from Hegermiller et al. 20122
    uf: velocity measured at specific height above the bed
    z: height above the bed to estimate the velocity
    returns: velocity at height z
    """
    if not (len(Tmm10) == 0 or len(u_orb) == 0):
        f = fw_swart(Tmm10, u_orb, d50=d50)
    return np.sign(uf) * np.sqrt(f * uf ** 2 / 2) / kappa * np.log((z - 0.7 * d50) / z0)

def critical_shields(d50, delta=1.65, nu=1e-06, g=9.8, method='vanRijn93'):
    """
    d50 median grain size
    delta relative density of sand (rho_s-rho_w)/rho_w
    nu kinematic viscosity
    g gravitational acceleration
    """
    if len(np.atleast_1d(d50)) > 1:
        theta_crit_list = []
        for id in d50:
            theta_crit_list.append(critical_shields(id, delta=delta, nu=nu, g=g, method=method))
        return np.array(theta_crit_list)
    dstar = d50 * (delta * g / nu ** 2) ** (1 / 3)
    if method == 'vanRijn93':
        if dstar < 4:
            theta_crit = 0.24 * dstar ** (-1)
        elif dstar < 10:
            theta_crit = 0.14 * dstar ** (-0.64)
        elif dstar < 20:
            theta_crit = 0.04 * dstar ** (-0.1)
        elif dstar < 150:
            theta_crit = 0.013 * dstar ** 0.29
        else:
            theta_crit = 0.055
        return theta_crit
    elif method == 'soulsby_whitehouse97':
        return 0.3 / (1 + 1.2 * dstar) + 0.055 * (1 - np.exp(-0.02 * dstar))

def plot_migrating_moments(ax, alpha=0.2):
    if len(np.atleast_1d(ax)) == 1:
        ax = [ax]
    for axi in ax:
        axi.axvspan(pd.to_datetime('20231102 00:00'), pd.to_datetime('20231102 02:30'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231102 02:30'), pd.to_datetime('20231102 06:30'), alpha=alpha, ec=None, fc='grey')
        axi.axvspan(pd.to_datetime('20231102 07:30'), pd.to_datetime('20231102 14:00'), alpha=alpha, ec=None, fc='red')
        axi.axvspan(pd.to_datetime('20231102 14:00'), pd.to_datetime('20231102 20:00'), alpha=alpha, ec=None, fc='grey')
        axi.axvspan(pd.to_datetime('20231102 20:00'), pd.to_datetime('20231103 00:30'), alpha=alpha, ec=None, fc='red')
        axi.axvspan(pd.to_datetime('20231103 14:30'), pd.to_datetime('20231103 17:45'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231104 00:30'), pd.to_datetime('20231104 02:15'), alpha=alpha, ec=None, fc='red')
        axi.axvspan(pd.to_datetime('20231104 03:30'), pd.to_datetime('20231104 08:00'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231104 09:45'), pd.to_datetime('20231104 13:15'), alpha=alpha, ec=None, fc='red')
        axi.axvspan(pd.to_datetime('20231104 14:00'), pd.to_datetime('20231104 15:30'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231104 15:30'), pd.to_datetime('20231104 18:30'), alpha=alpha, ec=None, fc='grey')
        # axi.axvspan(pd.to_datetime('20231104 18:30'), pd.to_datetime('20231104 20:00'), alpha=alpha, ec=None, fc='red')
        axi.axvspan(pd.to_datetime('20231105 01:00'), pd.to_datetime('20231105 07:30'), alpha=alpha, ec=None, fc='red')
        # axi.axvspan(pd.to_datetime('20231105 19:30'), pd.to_datetime('20231105 20:30'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231106 17:00'), pd.to_datetime('20231106 20:00'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231107 07:00'), pd.to_datetime('20231107 08:00'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231107 18:15'), pd.to_datetime('20231107 21:00'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231108 07:30'), pd.to_datetime('20231108 10:30'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231108 15:45'), pd.to_datetime('20231108 18:00'), alpha=alpha, ec=None, fc='red')

def shields_parameter_ribberink98(ds, d50, g=9.8, rho_s=2650, rho_w=1000, option=1):
    """ 
    Various flaviours of velocity components that are used to compute bedload transport with Ribberink (1998)
    option=7: Use the total velocity signal that is given as input argument (preprocessing of this signal possible)
    option=49: Explicitly state the sea-swell component (uc2_ss) and the mean flow component (ucm3) that are included in the cross-shore, no IG variance included. In the alongshore include all components
    """
    A = ds['Tmm10'] * ds['ud_ssm'] / 2 / np.pi
    fw_dash = np.exp(5.213 * (2.5 * d50 / A) ** 0.194 - 5.977)
   if option == 7:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc'] ** 2) * ds['uc'] / ((rho_s - rho_w) * g * d50)
    elif option == 8:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul'] ** 2) * ds['ul'] / ((rho_s - rho_w) * g * d50)
    elif option == 49: # only sea swell (IG variance removed) with along ripple component included in cross-shore, cros-ripple mean flow removed
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['uc2_ss']+ds['ucm3']) ** 2) * (ds['uc2_ss']+ds['ucm3']) / ((rho_s - rho_w) * g * d50)
    elif option == 50:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul'] ** 2) * ds['ul'] / ((rho_s - rho_w) * g * d50)
    return theta_prime

def bedload_ribberink98(ds, d50, dir='cross', g=9.8, rho_s=2650, rho_w=1025, nu=1e-06, option=1, m=11, n=1.5, total_excess_shields=False):
    if total_excess_shields:
        theta_prime_c = shields_parameter_ribberink98(ds, d50, g=g, rho_s=rho_s, rho_w=rho_w, option=41)
        theta_prime_l = shields_parameter_ribberink98(ds, d50, g=g, rho_s=rho_s, rho_w=rho_w, option=42)
        theta_prime_mag = np.sqrt(theta_prime_c ** 2 + theta_prime_l ** 2)
        theta_crit = critical_shields(d50, delta=(rho_s - rho_w) / rho_w, nu=nu, g=g, method='soulsby_whitehouse97')
        theta_eff = theta_prime_mag - theta_crit
        theta_eff = np.where(theta_eff > 0, theta_eff, 0)
        theta_prime_c_component = shields_parameter_ribberink98(ds, d50, g=g, rho_s=rho_s, rho_w=rho_w, option=option)
        T = m * (theta_eff ** n * theta_prime_c_component / theta_prime_mag).mean(dim='N')
    else:
        theta_prime_c = shields_parameter_ribberink98(ds, d50, g=g, rho_s=rho_s, rho_w=rho_w, option=option)
        theta_prime_l = shields_parameter_ribberink98(ds, d50, g=g, rho_s=rho_s, rho_w=rho_w, option=option + 1)
        theta_prime_mag = np.sqrt(theta_prime_c ** 2 + theta_prime_l ** 2)
        theta_crit = critical_shields(d50, delta=(rho_s - rho_w) / rho_w, nu=nu, g=g, method='soulsby_whitehouse97')
        theta_eff = theta_prime_mag - theta_crit
        theta_eff = np.where(theta_eff > 0, theta_eff, 0)
        if dir=='cross':
            T = m * (theta_eff ** n * theta_prime_c / theta_prime_mag).mean(dim='N')
        elif dir=='along':
            T = m * (theta_eff ** n * theta_prime_l / theta_prime_mag).mean(dim='N')
    return T * np.sqrt((rho_s - rho_w) / rho_w * g * d50 ** 3)

def spearman_stats(a, b):
    dat = np.vstack([a, b]).T
    dat = dat[~np.isnan(dat).any(axis=-1)]
    r = stats.spearmanr(dat)
    return r

def pearson_stats(a, b):
    dat = np.vstack([a, b]).T
    dat = dat[~np.isnan(dat).any(axis=-1)]
    r = stats.pearsonr(dat[:, 0], dat[:,1])
    return r
    
def density_scatter(x, y, range, ax=None, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    from scipy.interpolate import interpn
    from matplotlib.colors import Normalize
    import matplotlib as mpl
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, range=range, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method='splinef2d', bounds_error=False)
    z[np.where(np.isnan(z))] = 0.0
    if sort:
        idx = z.argsort()
        x, y, z = (x[idx], y[idx], z[idx])
    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    ax.scatter(x, y, c=z, norm=mpl.colors.LogNorm(), **kwargs)
    return ax

def my_density_scatter(q, qs, ax, loc='bottom'):
    ax.plot([-3, 3], [-3, 3], linewidth=0.5, color='k')

    innan = ~np.logical_or(np.isnan(q), np.isnan(qs))
    q = q[innan]
    qs = qs[innan]
    px = np.sign(qs)*np.log(np.abs(qs)+1)
    py = np.sign(q)*np.log(np.abs(q)+1)
    ran0 = [np.min([np.min(px), np.min(py)]), np.max([np.max(px), np.max(py)])]
    ran = [ran0, ran0]
    # ran = [[-2, 2], [-2, 2]]
    density_scatter(px, py, ax=ax, range=ran, bins=40, **{'cmap':'copper', 's':3})
    if loc=='bottom':
        t = ax.text(0.05, 0.05, 'r={:.2f}'.format(spearman_stats(q, qs).correlation), transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5)
        t.set_bbox(dict(facecolor='white', alpha=0.3, edgecolor='None'))
    else:
        t = ax.text(0.05, 0.95, 'r={:.2f}'.format(spearman_stats(q, qs).correlation), transform=ax.transAxes, ha='left', va='top', fontsize=7.5)   
        t.set_bbox(dict(facecolor='white', alpha=0.3, edgecolor='None'))     
    print(spearman_stats(q, qs))
    return