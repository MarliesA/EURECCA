import numpy as np 
import matplotlib.pyplot as plt
import scipy as sc
from scipy import signal
import os
from matplotlib import colors

def spectrum_simple_2D(x,y,z, jacorrectplane=True, jacorrectvar=True, jafig=False, jawindow=False, figpath=None, lf=1):

    dx = np.round(np.median(np.diff(x)), 3)
    dy = np.round(np.median(np.diff(y.T)), 3)

    # crop to area between -0.5 and 0.5
    ix = np.logical_and(np.logical_and(np.logical_and(x>-0.5, x<=0.5), y>-0.5), y<=0.5)
    ns = int(np.sqrt(x[ix].shape)[0])
    xc = x[ix].reshape([ns, ns])
    yc = y[ix].reshape([ns, ns])
    zc = z[ix].reshape([ns, ns])
    ny, nx = zc.shape

    zc0 = zc.copy()

    # fill with mean
    zc[np.isnan(zc)] = np.nanmean(zc)

    # remove mean
    zc -= np.nanmean(zc)

    # planar detrend
    if jacorrectplane:
        if np.sum(np.isnan(zc.flatten()))>0:
            a=1
        A = np.c_[xc.flatten(), yc.flatten(), np.ones(ny*nx)]
        C,_,_,_ = sc.linalg.lstsq(A, zc.flatten())    # coefficients

        # evaluate it on grid
        plane = C[0]*xc + C[1]*yc + C[2]
        zc = zc-plane
        zc0 = np.where(~np.isnan(zc0), zc, zc0)
        
        if jafig:
            fig = plt.figure(figsize=[3, 2])
            plt.pcolor(xc, yc, plane)
            plt.title('plane fitting')
            plt.colorbar(label='[m]')
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            # fig.tight_layout()
            if not (figpath==None):
                plt.savefig(os.path.join(figpath, 'fitted_plane.png'), dpi=200, bbox_inches='tight')

    varpre0 = np.nanvar(zc0)
    varpre = np.var(zc)

    if jafig:
        plt.figure()
        plt.pcolor(xc, yc, zc)
        plt.title('zb')
        plt.colorbar(label='[m]')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        if not (figpath==None):
            plt.savefig(os.path.join(figpath, 'zb_minus_plane.png'), dpi=200, bbox_inches='tight')

    # apply window
    if jawindow:
        wx= sc.signal.windows.hann(nx)
        wy= sc.signal.windows.hann(ny)
        WX, WY = np.meshgrid(wx, wy)
        zc = WX*WY*zc
        
        if True: #jafig:
            plt.figure(figsize=[3, 2.5])
            limit = max(abs(np.min(100*zc)), abs(np.max(100*zc)) )
            plt.pcolor(xc, yc, 100*zc, cmap='RdBu', vmin=-limit, vmax=limit)
            # plt.title('zb * window')
            cb = plt.colorbar(label='$\Delta z_b$ [cm]')
            cb.ax.set_yticks([-np.floor(limit), 0, np.floor(limit)])
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.xticks([-0.5, 0, 0.5])
            plt.yticks([-0.5, 0. ,0.5])
            
            if not (figpath==None):
                plt.savefig(os.path.join(figpath, 'zb_times_window.png'), dpi=200, bbox_inches='tight')

    if lf>=2:
        zc2 = np.zeros([lf*zc.shape[0], lf*zc.shape[1]])
        zc2[:zc.shape[0], :zc.shape[1]] = zc
        nx = lf*nx
        ny = lf*ny
        zc = zc2

    Fboost = varpre0/np.var(zc)
    # Fboost = varpre/np.var(zc)

    # in wave number space, using freqshift
    dkx = 1/(dy*ny)
    dky = 1/(dx*nx)
    kx = sc.fft.fftfreq(nx, d=dx)
    kkx = sc.fft.fftshift(kx)
    ky = sc.fft.fftfreq(ny, d=dy)
    kky = sc.fft.fftshift(ky)
    KKX, KKY = np.meshgrid(kkx, kky)

    vz = sc.fft.fft2(zc)
    vz2 = sc.fft.fftshift(vz)

    # power density
    V = dx*dy/nx/ny*np.abs(vz2)**2
    
    # check that fft2 is variance preserving
    if (np.var(zc)-np.trapz(np.trapz(V, KKX), KKY[:,1]))/np.var(zc)>0.01:
        print('var(zc)={:.3e}, integral(V)={:.3e}'.format(np.var(zc), np.trapz(np.trapz(V, KKX), KKY[:,1])))

    if jafig:
        fig, ax = plt.subplots()
        plt.pcolor(KKX, KKY, V, norm=colors.LogNorm())
        plt.title('freqspace')
        plt.colorbar(label='[m2/m2]')
        plt.xlabel('kx [m-1]')
        plt.ylabel('ky [m-1]')
        if not (figpath==None):
            plt.savefig(os.path.join(figpath, '2dfft_jawindow{}_jacorrectplane{}_lf{}.png'.format(int(jawindow), int(jacorrectplane), lf)), dpi=200, bbox_inches='tight')

    if jacorrectvar:
        V = Fboost*V

    return KKX, KKY, V


def spectrum_simple_1D(x,z, jacorrectplane=True, jacorrectvar=True, jafig=False, jawindow=False, figpath=None, lf=1):

    dx = np.round(np.median(np.diff(x)), 3)

    # crop to area between -0.5 and 0.5
    ix = np.logical_and(x>-0.5, x<=0.5)
    xc = x[ix]
    zc = z[ix]
    nx = len(zc)

    # fill with mean
    zc[np.isnan(zc)] = np.nanmean(zc)

    # remove mean
    zc -= np.nanmean(zc)

    # planar detrend
    if jacorrectplane:
        zc = signal.detrend(zc)

    varpre = np.var(zc)

    if jafig:
        plt.figure()
        plt.plot(xc, zc)
        plt.title('zb')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        if not (figpath==None):
            plt.savefig(os.path.join(figpath, 'zb_minus_plane.png'), dpi=200, bbox_inches='tight')

    # apply window
    if jawindow:
        wx= sc.signal.windows.hann(nx)
        zc = wx*zc
        
        if jafig:
            plt.figure()
            plt.plot(xc, zc)
            plt.title('zb * window')
            plt.xlabel('x [m]')
            plt.ylabel('z [m]')
            if not (figpath==None):
                plt.savefig(os.path.join(figpath, 'zb_times_window.png'), dpi=200, bbox_inches='tight')

    if lf>=2:
        zc2 = np.array(lf*list(np.zeros(len(zc))))
        zc2[:len(zc)] = zc
        nx = lf*nx
        zc = zc2

    Fboost = varpre/np.var(zc)

    # in wave number space, using freqshift
    dkx = 1/(dx*nx)
    kx = sc.fft.fftfreq(nx, d=dx)
    kx = sc.fft.fftshift(kx)

    vz = sc.fft.fft(zc)
    vz2 = sc.fft.fftshift(vz)

    # power density
    V = dx/nx*np.abs(vz2)**2
    
    # check that fft2 is variance preserving
    if (np.var(zc)-np.trapz(V, kx))/np.var(zc)>0.01:
        print('var(zc)={:.3e}, integral(V)={:.3e}'.format(np.var(zc), np.trapz(np.trapz(V, kx))))

    if jacorrectvar:
        V = Fboost*V

    return kx, V
