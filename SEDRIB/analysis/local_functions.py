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
        axi.axvspan(pd.to_datetime('20231104 18:30'), pd.to_datetime('20231104 20:00'), alpha=alpha, ec=None, fc='red')
        axi.axvspan(pd.to_datetime('20231105 01:00'), pd.to_datetime('20231105 07:30'), alpha=alpha, ec=None, fc='red')
        axi.axvspan(pd.to_datetime('20231105 19:30'), pd.to_datetime('20231105 20:30'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231106 17:00'), pd.to_datetime('20231106 20:00'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231107 07:00'), pd.to_datetime('20231107 08:00'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231107 18:15'), pd.to_datetime('20231107 21:00'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231108 07:30'), pd.to_datetime('20231108 10:30'), alpha=alpha, ec=None, fc='green')
        axi.axvspan(pd.to_datetime('20231108 15:45'), pd.to_datetime('20231108 18:00'), alpha=alpha, ec=None, fc='red')

def shields_parameter_ribberink98(ds, d50, g=9.8, rho_s=2650, rho_w=1000, option=1):
    """ 
    option=1: waves only crossshore direction
    option=2: waves only alongshore direction
    option=3: vector addition of the stirring component, but transport only in cross direction
    option=4: vector addition of the stirring component, but transport only in along direction
    option=5: only cross-shore stirring component, and transport only in cross direction
    option=6: only alongshore stirring component, and transport only in along direction
    option=7: only cross-shore based on total velocity signal, and transport only in cross direction
    option=8: only alongshore sbased on, and transport only in along direction      
    option=9: only cross-shore based on total velocity signal**3, and transport only in cross direction
    option=10: only alongshore based on total velocity signal**3, and transport only in along direction    
    """
    A = ds['Tmm10'] * ds['ud_ssm'] / 2 / np.pi
    fw_dash = np.exp(5.213 * (2.5 * d50 / A) ** 0.194 - 5.977)
    if option == 1:
        theta_prime = 0.5 * rho_w * fw_dash * ds['uc_ss'] ** 2 * np.sign(ds['uc_ss']) / ((rho_s - rho_w) * g * d50)
    elif option == 2:
        theta_prime = 0.5 * rho_w * fw_dash * ds['ul_ss'] ** 2 * np.sign(ds['ul_ss']) / ((rho_s - rho_w) * g * d50)
    elif option == 3:
        fc = 0.0025
        fw = fw_dash
        alfa = ds['ud_ssm'] / (ds['ud_ssm'] + np.sqrt(ds['ucm'] ** 2 + ds['ulm'] ** 2))
        fcw = alfa * fc + (1 - alfa) * fw
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc'] ** 2 + ds['ul'] ** 2) * (ds['ucm'] + ds['uc_ss']) / ((rho_s - rho_w) * g * d50)
    elif option == 4:
        fc = 0.0025
        fw = fw_dash
        alfa = ds['ud_ssm'] / (ds['ud_ssm'] + np.sqrt(ds['ucm'] ** 2 + ds['ulm'] ** 2))
        fcw = alfa * fc + (1 - alfa) * fw
        theta_prime = theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc'] ** 2 + ds['ul'] ** 2) * (ds['ulm'] + ds['ul_ss']) / ((rho_s - rho_w) * g * d50)
    elif option == 5:
        fc = 0.0025
        fw = fw_dash
        alfa = ds['ud_ssm'] / (ds['ud_ssm'] + np.sqrt(ds['ucm'] ** 2))
        fcw = alfa * fc + (1 - alfa) * fw
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['ucm'] + ds['uc_ss']) ** 2) * (ds['ucm'] + ds['uc_ss']) / ((rho_s - rho_w) * g * d50)
    elif option == 6:
        fc = 0.0025
        fw = fw_dash
        alfa = ds['ud_ssm'] / (ds['ud_ssm'] + np.sqrt(ds['ulm'] ** 2))
        fcw = alfa * fc + (1 - alfa) * fw
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['ulm'] + ds['ul_ss']) ** 2) * (ds['ulm'] + ds['ul_ss']) / ((rho_s - rho_w) * g * d50)
    elif option == 7:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc'] ** 2) * ds['uc'] / ((rho_s - rho_w) * g * d50)
    elif option == 8:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul'] ** 2) * ds['ul'] / ((rho_s - rho_w) * g * d50)
    elif option == 9:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * ds['uc'] ** 3 / ((rho_s - rho_w) * g * d50)
    elif option == 10:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * ds['ul'] ** 3 / ((rho_s - rho_w) * g * d50)
    elif option == 11:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ucm'] ** 2 + np.sqrt(ds['uc_ss'] ** 2).mean(dim='N')) * ds['uc'] / ((rho_s - rho_w) * g * d50)
    elif option == 12:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ulm'] ** 2 + np.sqrt(ds['ul_ss'] ** 2).mean(dim='N')) * ds['ul'] / ((rho_s - rho_w) * g * d50)
    elif option == 13:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc2'] ** 2) * ds['uc2'] / ((rho_s - rho_w) * g * d50)
    elif option == 14:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul2'] ** 2) * ds['ul2'] / ((rho_s - rho_w) * g * d50)
    elif option == 15:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc_ss'] ** 2) * ds['uc_ss'] / ((rho_s - rho_w) * g * d50)
    elif option == 16:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul_ss'] ** 2) * ds['ul_ss'] / ((rho_s - rho_w) * g * d50)
    elif option == 17:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc3'] ** 2) * ds['uc3'] / ((rho_s - rho_w) * g * d50)
    elif option == 18:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul3'] ** 2) * ds['ul3'] / ((rho_s - rho_w) * g * d50)
    elif option == 19:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc4'] ** 2) * ds['uc4'] / ((rho_s - rho_w) * g * d50)
    elif option == 20:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul4'] ** 2) * ds['ul4'] / ((rho_s - rho_w) * g * d50)
    elif option == 21:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc5'] ** 2) * ds['uc5'] / ((rho_s - rho_w) * g * d50)
    elif option == 22:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul5'] ** 2) * ds['ul5'] / ((rho_s - rho_w) * g * d50)
    elif option == 23:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc_lf'] ** 2) * ds['uc_lf'] / ((rho_s - rho_w) * g * d50)
    elif option == 24:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul_lf'] ** 2) * ds['ul_lf'] / ((rho_s - rho_w) * g * d50)
    elif option == 25:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc_hf'] ** 2) * ds['uc_hf'] / ((rho_s - rho_w) * g * d50)
    elif option == 26:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul_hf'] ** 2) * ds['ul_hf'] / ((rho_s - rho_w) * g * d50)
    elif option == 27:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['uc_lf'] + ds['uc_ss']) ** 2) * (ds['uc_lf'] + ds['uc_ss']) / ((rho_s - rho_w) * g * d50)
    elif option == 28:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['ul_lf'] + ds['ul_ss']) ** 2) * (ds['ul_lf'] + ds['ul_ss']) / ((rho_s - rho_w) * g * d50)
    elif option == 29:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc2_ss'] ** 2) * ds['uc2_ss'] / ((rho_s - rho_w) * g * d50)
    elif option == 30:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul2_ss'] ** 2) * ds['ul2_ss'] / ((rho_s - rho_w) * g * d50)
    elif option == 31:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc2_ig'] ** 2) * ds['uc2_ig'] / ((rho_s - rho_w) * g * d50)
    elif option == 32:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul2_ig'] ** 2) * ds['ul2_ig'] / ((rho_s - rho_w) * g * d50)
    elif option == 33:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['uc2'] + ds['ucm']) ** 2) * (ds['uc2'] + ds['ucm']) / ((rho_s - rho_w) * g * d50)
    elif option == 34:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['ul2'] + ds['ulm']) ** 2) * (ds['ul2'] + ds['ulm']) / ((rho_s - rho_w) * g * d50)
    elif option == 35:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['uc'] - ds['ucm']) ** 2) * (ds['uc'] - ds['ucm']) / ((rho_s - rho_w) * g * d50)
    elif option == 36:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['ul'] - ds['ulm']) ** 2) * (ds['ul'] - ds['ulm']) / ((rho_s - rho_w) * g * d50)
    elif option == 37:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc'] ** 2) * ds['uc'] / ((rho_s - rho_w) * g * d50)
    elif option == 38:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['ul'] - ds['ulm']) ** 2) * (ds['ul'] - ds['ulm']) / ((rho_s - rho_w) * g * d50)
    elif option == 39:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['uc2_ss'] + ds['uc2_ig']) ** 2) * (ds['uc2_ss'] + ds['uc2_ig']) / ((rho_s - rho_w) * g * d50)
    elif option == 40:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['ul2_ss'] + ds['ul2_ig']) ** 2) * (ds['ul2_ss'] + ds['ul2_ig']) / ((rho_s - rho_w) * g * d50)
    elif option == 41:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['uc'] - ds['ucm']) ** 2) * (ds['uc'] - ds['ucm']) / ((rho_s - rho_w) * g * d50)
    elif option == 42:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul'] ** 2) * ds['ul'] / ((rho_s - rho_w) * g * d50)
    elif option == 43:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['ucm'] + 0 * ds['uc']) ** 2) * (ds['ucm'] + 0 * ds['uc']) / ((rho_s - rho_w) * g * d50)
    elif option == 44:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt((ds['ulm'] + 0 * ds['uc']) ** 2) * (ds['ulm'] + 0 * ds['uc']) / ((rho_s - rho_w) * g * d50)
    elif option == 45:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['uc2_ss'] ** 2) * ds['uc2_ss'] / ((rho_s - rho_w) * g * d50)
    elif option == 46:
        fcw = fw_dash
        theta_prime = 0.5 * rho_w * fcw * np.sqrt(ds['ul'] ** 2) * ds['ul'] / ((rho_s - rho_w) * g * d50)
    return theta_prime

def bedload_ribberink98(ds, d50, g=9.8, rho_s=2650, rho_w=1025, nu=1e-06, option=1, m=11, n=1.65, total_excess_shields=False):
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
        T = m * (theta_eff ** n * theta_prime_c / theta_prime_mag).mean(dim='N')
    return T * np.sqrt((rho_s - rho_w) / rho_w * g * d50 ** 3)

def spearman_stats(a, b):
    dat = np.vstack([a, b]).T
    dat = dat[~np.isnan(dat).any(axis=-1)]
    r = stats.spearmanr(dat)
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