import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from scipy import signal
from scipy.optimize import minimize
from bisect import bisect

#         #units:
#         # 1dbar = 1m waterdiepte
#         # 1hpa = 1cm waterdiepte
#         # 1e4 pa = 1m waterdiepte

def disper(w, h, g=9.8):
    '''
    DISPER  Linear dispersion relation.
    
    absolute error in k*h < 5.0e-16 for all k*h
    
    Syntax:
    k = disper(w, h, [g])
    
    Input:
    w = 2*pi/T, were T is wave period
    h = water depth
    g = gravity constant
    
    Output:
    k = wave number
    
    Example
    k = disper(2*pi/5,5,g = 9.81);
    
    Copyright notice
    --------------------------------------------------------------------
    Copyright (C) 
    G. Klopman, Delft Hydraulics, 6 Dec 1994
    M. van der Lugt conversion to python, 11 Jan 2021
    
    '''    
    #make sure numpy array
    listType = type([1,2])
    Type = type(w)

    w = np.atleast_1d(w)
    
    #check to see if warning disappears
    wNul = w==0
    w[w==0] = np.nan

    
    w2 = w**2*h/g
    q = w2 / (1-np.exp(-w2**(5/4)))**(2/5)
    
    for j in np.arange(0,2):
        thq = np.tanh(q)
        thq2 = 1-thq**2
        aa = (1 - q*thq) * thq2
        
        #prevent warnings, we don't apply aa<0 anyway
        aa[aa<0] = 0
        
        bb = thq + q*thq2
        cc = q*thq - w2
        
        
        D = bb**2-4*aa*cc
        
        # initialize argument with the exception
        arg = -cc/bb
        
        # only execute operation on those entries where no division by 0 
        ix = np.abs(aa*cc)>=1e-8*bb**2 
        arg[ix] = (-bb[ix]+np.sqrt(D[ix]))/(2*aa[ix]) 

                
        q = q + arg

              
    k = np.sign(w)*q/h
    
    #set 0 back to 0
    k = np.where(wNul,0,k)

    #if input was a list, return also as list
    if Type==listType:
        k = list(k)
    elif len(k)==1:
        k = k[0]
        
    return k

def disperGuo(w,h,g=9.8):
     return w**2/g * (1 - np.exp(-(w*np.sqrt(h/g))**2.5))**(-0.4)  
     
def disper_cur(w,h,u,g=9.8):  
    def f(g,k,h,u,w):
        return np.sqrt(g*k*np.tanh(k*h))+k*u-w
  
    def fk(g,k,h,u):
        return np.sqrt(g/k*np.tanh(k*h)) * (0.5+ k * h / np.sinh(2 * k * h)) + u
    
    k = disperGuo(w,h)
    
    count = 0
    while (abs(f(g,k,h,u,w))>1e-5 ):
        count += 1
        
        k = k - f(g,k,h,u,w) / fk(g,k,h,u)    
        
        if count>10:
            print('not converged: too strong opposing current')
            return g/(4*u**2)
            
    print('k = {} m-1 in {} iterations'.format(k,count))       
    return k   

def Ursell(hm0, k, d):
    return 3/4 * 0.5 * hm0*k/(k*d)**3


def fourier_window(Nr,windowFilter = 'Hann'):
    
    F = np.arange(0,Nr)/(Nr-1)
    if windowFilter == 'Hann':
        F = 0.5 * (1-np.cos(2*np.pi*F))
    elif windowFilter == 'Hamming':
        F = 0.54 - 0.46*np.cos(2*np.pi*F)  
        
    return F
        




def spectrum_simple(x,y,
                    fresolution=0.01,
                    Nr=None,
                    detrend=True,
                    overlap=0.5,
                    windowFilter='Hamming',
                    tolerance = 1e-3,
                    strict = False,
                    correctvar = True):   
    """
    spectrum_simple(sf,zs) 
    wrapper function for fast fourier transform that takes care of windowing of the inputsignal to obtain a power spectrum at the desired resolution in frequency space.
    
    Parameters
    ----------
    x : FLOAT
        DATETIME64 TIME ARRAY OR SAMPLING FREQUENCY (WHEN ONLY ONE VALUE).
    y : FLOAT
        SURFACE ELEVATION [M].
    fresolution : FLOAT, optional
        DESIRED RESOLUTION IN THE FREQUENCY DOMAIN. The default is 0.01.
    detrend : BOOL, optional
        DETREND THE SIGNAL YES/NO. The default is True.
    overlap : FLOAT, optional
        OVERLAPPING PERCENTAGE OF THE WINDOWS. The default is 0.5.
    windowFilter : STRING, optional
        WINDOW TYPE. The default is 'Hamming'.
    tolerance : FLOAT, optional
        WARNING TOLLERANCE. The default is 1e-3.
    correctvar : BOOL, optional
        RESTORE TOTAL VARIANCE IN FREQ SPACE YES/NO. The default is True.

    Returns
    -------
    fx, vy
    fx = frequency axis [Hz]
    vy = power density [m2/Hz]
    
    Matlab to Python: Marlies van der Lugt 14-01-2020

    """
    import numpy as np
    from scipy.fft import fft, ifft
    #input checks differentiating for datetime64 and seconds
    if len(np.atleast_1d(x))==1:
        dt = 1/x # 1/frequency
    else:
        try:
            if ((np.min(np.diff(x))/np.timedelta64(1,'s')<0) or 
                ((np.max(np.diff(x))-np.min(np.diff(x)))/np.timedelta64(1,'s')>1e-8 )):   
                print('Input xx must be monotonic increasing')
                exit
            else:
                x = (x - x[0])/np.timedelta64(1,'s')
        except:
            if ((np.min(np.diff(x))<0) or 
            ((np.max(np.diff(x))-np.min(np.diff(x)))>1e-8 )):   
                print('Input xx must be monotonic increasing')
            else:
                x = x-x[0]
        
        # Time step in time/space axis
        dt= np.mean(np.diff(x))

    # Number of data points in the total signal
    N = len(y)   
    if Nr==None:
        # Number of data points required for desired fresolution      
        Nr = int(np.ceil(1/fresolution/dt))
        if strict:
            #TODO: make a nextpow2 function
            Nr = Nr
    
        # Check if there are sufficient data points to acquire the set frequency
        # resolution
        if Nr>N:
            # reset Nr
            Nr = N
            print('Warning: The required frequency resolution could not be achieved.')

    # Number of Welch repetitions
    Nw = int(np.ceil((N-Nr)/(overlap*Nr))+1)

    # Index input arrays for start and end of each Welch repetition
    indend = np.linspace(Nr,N,Nw).astype(int)
    indstart = (indend-Nr).astype(int)

    # Time and frequency array 
    T  = dt*Nr
    df = 1/T
    ffa = np.arange(0,np.round(Nr/2))
    ffb = -np.arange(np.round(Nr/2),0,-1)
    ff = df*np.append(ffa,ffb)
    fx = ff[0:int(Nr/2)]

    # Storage arrays
    vy = np.zeros([int(Nr/2)])

    # % Detrend input signal
    if  detrend:
        # pdb.set_trace()
        y = signal.detrend(y)
        
    varpre = np.var(y)
    
    # do Welch repetition
    for i in np.arange(0,Nw):
        d = y[indstart[i]:indend[i]]
        if (windowFilter == 'Hann'):
            d = d * 0.5*(1-np.cos(2*np.pi*np.arange(0,Nr)/(Nr-1)))
            varpost = np.var(d)            
        elif (windowFilter == 'Hamming'):
            d = d * (0.54-0.46*np.cos(2*np.pi*np.arange(0,Nr)/(Nr-1)))
            varpost = np.var(d)            
        elif (windowFilter == None):
            varpost = varpre

        # FFT
        Q = fft(d)
        # Power density
        V = 2*T/Nr**2*np.abs(Q)**2
        # Store in array
        vy = vy + 1/Nw*V[0:int(Nr/2)]


    # Restore the total variance
    if correctvar:
       vy = (varpre/np.trapz(vy,dx = df))*vy


    # input/output check
    hrmsin = 4*np.std(y)/np.sqrt(2)
    hrmsout =  np.sqrt(8*np.trapz(vy, dx = df))
    dif = np.abs(hrmsout-hrmsin)
    if (dif > tolerance):
        print('Warning: Difference in input and output wave height ({}) is greater than set tolerance ({})'.format(dif,tolerance))

    return fx, vy

def get_peak_frequency(fx,vy,fpmin=0.05):
    '''
    

    Parameters
    ----------
    fx : NUMPY ARRAY
        FREQUENCY AXIS.
    vy : NUMPY ARRAY
        POWER DENSITY.
    fpmin : OPTIONAL FLOAT
        mininum believed peak frequency

    Returns
    -------
    fp : FLOAT
        PEAK FREQUENCY.

    '''
    fp    = fx[np.argmax(vy)] # peak frequency	
    
    count = 0
    fxtest = fx
    vytest = vy
    while fp<=fpmin:
        count+=1
        if count>= len(vytest)-1:
            return None
        fxtest = fxtest[count:]
        vytest = vy[count:]
        try:
            fp = fxtest[np.argmax(vytest)]
        except:
            return None
    return fp


def attenuation_factor(Type, elev, h, fx,
                       maxSwfactor = 5,
                       fcorrmaxBelieve = 1,
                       fcorrmax = 1.5):
    """
    computes attenuation correction factor based on linear theory

    Parameters
    ----------
    Type : STRING
        TYPE OF SIGNAL, EITHER HORIZONTAL, VERTICAL OR PRESSURE.
    elev : FLOAT
        HEIGHT ABOVE BED.
    h : FLOAT
        WATER DEPTH.
    fx : NUMPY ARRAY
        FREQUENCY AXIS.
    maxSwfactor : FLOAT 
        OPTIONAL MAX CUTOFF.    
    fcorrmaxBelieve : FLOAT 
        STARTFREQ LINEAR TAPERING OF CORRECTION TO 0.          
    fcorrMax : FLOAT 
        STOPFREQ LINEAR TAPERING OF CORRECTION TO 0.
        
    Returns
    -------
    Sw : NUMPY ARRAY
        ATTENUATION CORRECTION FACTOR.

    """
    #compute attenuation factor per wave number
    w = 2*np.pi*fx
    k = disper(w,h)
    if Type == 'pressure':       
        Sw = np.cosh(k*h)/np.cosh(k*elev)    
    elif Type == 'horizontal':
        w[w==0] = np.nan
        Sw = 1/w*np.sinh(k*h)/np.cosh(k*elev)
        Sw[w==0] = 0
    elif Type == 'vertical':
        w[w==0] = np.nan
        Sw = np.divide(1,w)*np.sinh(k*h)/np.sinh(k*elev)  
        Sw[w==0] = 0              
    else:
        print('Type must be set to either of: p,u,w')
    
    
    
    #create mask to taper the attenuation from 1 at fmaxBelieve to 0 at fmax
    mask = np.ones(fx.shape) 
    taperRange = fcorrmax - fcorrmaxBelieve
    mask[fx>fcorrmax] = 0
    ix = np.logical_and(fx>fcorrmaxBelieve,fx<= fcorrmax)
    mask[ix] = 1-(np.abs(fx[ix])-fcorrmaxBelieve)/taperRange
    
    # last check that swfactor remains between limits
    Sw[fx==0] = 0    
    Sw[Sw>maxSwfactor] = maxSwfactor  
    Sw = Sw * mask
    return Sw**2

def attenuation_corrected_wave_spectrum(Type,sf,x,h,zi,zb,
                                        fresolution=0.02,
                                        detrend = True,
                                        removeNoise = False,
                                        **kwargs):
    '''
    wraps attenuation_factor to imeediately return spectra

    Parameters
    ----------
    sf : FLOAT
        SAMPLING FREQUENCY.
    x : NUMPY ARRAY
        SIGNAL.
    h : FLOAT
        WATER DEPTH [M].
    zi : FLOAT
        INSTRUMENT HEIGHT WRT REFERENCE LEVEL.
    zb : FLOAT
        BOTTOM HEIGHT WRT REFERENCE LEVEL.
    Type : STRING, optional
        SWITCH FOR VELOCITIES OR PRESSURE. The default is 'pressure'.    
    fresolution : FLOAT, optional
        DESIRED SPECTRAL RESOLUTION. The default is 0.02.

    Returns
    -------
    fx : NUMPY ARRAY
        FREQUENCY AXIS.
    vy : NUMPY ARRAY
        POWER DENSITY.

    '''
    if Type=='pressure':
        x = x/1e4 #Pa to dBar
    if detrend:
        x = signal.detrend(x)
    
    fx,vy = spectrum_simple(sf,x,fresolution=fresolution)
    
    elev = zi - zb #height of instrument above the bed
    
    swfactor = attenuation_factor(Type, elev, h, fx, **kwargs)    
    
    # optional: remove noise floor based on curvature spectrum:
    if removeNoise:
        fxc,vyc = spectrum_simple(sf,x,fresolution=0.1)
        dvy3 = np.diff(vyc,n=3) 
        dvy3[0] = 0
        ffloor = fxc[dvy3.argmax()+4]
        vy[fx>ffloor] = 0
        
    return fx,swfactor * vy

def attenuate_signal(Type,f,x,hmean,zi,zb,
                     detrend = True,  
                     rho = 1000,
                     g = 9.8,
                     removeNoise = False,
                     **kwargs
                     ):
    '''
    attenuate_signal: wraps attenuation_factor to return reconstructed surface
    variation signals
    
    
    Syntax:
    t,zs = attenuate_signal(Type,t,x,hmean,zinst,[detrend=True],[fmax=5],[fmaxBelieve=3],
                  [windowing=False],[rho=1000],
                   [windowFilter='Hann'],[g=9.8])
    
    Input:
    Type        = the ADV signal type, one of ['pressure','horizontal','vertical']
    f           = sampling frequency
    x           = signal
    hmean       = mean water depth [m]
    zi          = instrument level w.r.t. reference level (e.g. NAP)
    zb          = bed level w.r.t. reference level (e.g. NAP)    
    detrend     = logical detrend yes/no
    fmax        = maximum frequency included in attenuation [Hz]
    fmaxBelieve = frequency to start tapering input to zero at fmax [Hz]
    windowing   = logical to use 3 windows to improve the zs signal at start and
                  end of signal
    g           = constant of gravitational acceleration

    Output:
    t           = time array
    zs          = surface elevation
    
    Example
    t,zs = attenuate_signal('pressure',f,x,zb,h,zi,windowing=False)
    
    M. van der Lugt 14 Jan 2021    
    
    '''
    
    # #get frequency from time array
    # if len(np.atleast_1d(t))==1:
    #     f = t
    # else:
    #     try:
    #         f = 1/ ( np.mean(np.diff(t))/ np.timedelta64(1, 's'))        
    #     except:
    #         f = 1/ ( np.mean(np.diff(t)))
    
    if Type == 'pressure':
        x = x/rho/g
        
    if detrend: 
        pex = x-signal.detrend(x)
        x = x-pex
   
    elev = zi-zb #height of instrument above the bed
    
     # construct time array
    Nr = len(x)
    deltaf = f/Nr
    ff = deltaf*np.append(np.arange(0,np.round(Nr/2)),
                          - np.arange(np.round(Nr/2),0,-1))

     
        
    # into frequency domain
    xf = fft(x)
    # optional: remove noise floor based on curvature spectrum:
    if removeNoise:
        fxc,vyc = spectrum_simple(f,x,fresolution=0.1)
        dvy3 = np.diff(vyc,n=3) 
        dvy3[0] = 0
        ffloor = fxc[dvy3.argmax()+4]
        xf[ff>ffloor] = 0
    
    elev = zi - zb
    swfactor = attenuation_factor(Type, elev, hmean, ff, **kwargs)           
    #this was computed for vardensity spectrum but fft is not a power spectrum!
    swfactor = np.sqrt(swfactor)
    
    vardens = swfactor * xf            
    zs = ifft(vardens).real
    
    if (detrend & (Type=='pressure')):
        zs = zs + pex
     
    return zs


def jspect(X,Y,N,DT=1,DW='hann',OVERLAP=0,DETREND=1,jaPrint = False):
    '''
    function F,P=jspect(X,Y,N,DT,DW,OVERLAP,DETREND)
    
    JSPECT (Similiar to CROSGK), but uses scipy.signal.csd  
    
    Usage: F,P=jspect(X,Y,N,DT,DW,OVERLAP,DETREND)
    
    Input:
    X  contains the data of series 1
    Y  contains the data of series 2
    N  is the number of samples per data segment (power of 2)
    DT is the time step (optional), default DT=1 s
    DW is the data window type (optional): DW = 1 for hann window (default)
                                            DW = 2 for rectangular window
                                            DW = 3 for bartlett window
                                            DW = 4 for blackman window
                                            DW = 5 for hamming window
    specification of scipy window types is recommended
    
    OVERLAP(optional) : 0=0% (default) 1=50% overlap, etc
    DETREND(optional) : 1 = linear(default), 2 = quadratic;
    jaPrint(optional) : prints  output spectrum characteristics to screen yes/no
    Output:
    P contains the (cross-)spectral estimates: list element 1 = Pxx, 2 = Pyy, 3 = Pxy
    F contains the frequencies at which P is given
    
    JAMIE MACMAHAN 19/6/2002
    Matlab to Python: Vincent Vuik, 29/5/2019
    '''
    # Window selection
    if DW=='hann' or DW==1:
        # from windowFunctions import hann_local
        # win = hann_local(N) # specify win vector by function hann_local
        DOF = np.floor(8/3*np.size(X)/(N/2))
        win = 'hann'
    elif DW=='rectwin' or DW==2:
        # from windowFunctions import rectwin
        # win = rectwin(N)
        DOF = np.floor(np.size(X)/(N/2))
        win = 'rectwin' # not an option in python! if needed: implement in windowFunctions.py
    elif DW=='bartlett' or DW==3:
        # from windowFunctions import bartlett
        # win = bartlett(N)
        DOF = np.floor(3*np.size(X)/(N/2))
        win = 'bartlett'
    elif DW=='blackman' or DW==4:
        # from windowFunctions import blackman
        # win = blackman(N)
        DOF = np.floor(np.size(X)/(N/2))
        win = 'blackman'
    elif DW=='hamming' or DW==5:
        # from windowFunctions import hamming
        # win = hamming(N)
        DOF = np.floor(2.5164*np.size(X)/(N/2))
        win = 'hamming'
    else:
        DOF = np.nan
        win = DW # choose a valid name of python window type
    
    
    # Original detrend code 19/06/2002
    if DETREND==1: # linear detrend
        X = signal.detrend(X)
        Y = signal.detrend(Y)
    elif DETREND==2: # quadratically detrend
        eps = 1e-9
        t  = np.arange(0,(DT*np.size(X))-DT+eps,DT)
        P1 = np.polyfit(t,X,2)
        P2 = np.polyfit(t,Y,2)
        X  = X-np.polyval(P1,t)
        Y  = Y-np.polyval(P2,t) # Matlab code: P1, check when DETREND=2 is to be used
    # Alternative: detrend per block (see Matlab code)
    
    # Estimate the cross power spectral density, Pxy, using Welch’s method
    Fs = 1/DT # sampling frequency
    # pdb.set_trace()
    fxx, Pxx = signal.csd(X,X,window=win,noverlap=OVERLAP,nperseg=N,fs=Fs)
    fyy, Pyy = signal.csd(Y,Y,window=win,noverlap=OVERLAP,nperseg=N,fs=Fs) 
    fxy, Pxy = signal.csd(X,Y,window=win,noverlap=OVERLAP,nperseg=N,fs=Fs)
    F   = fxx[0:-1]
    Pxx = Pxx[0:-1] # P(:,1)
    Pyy = Pyy[0:-1] # P(:,2)
    Pxy = Pxy[0:-1] # P(:,3)
            
    # Variance Perserving   
    fc1 = np.trapz(Pxx,F) # trapz y,x !
    fc2 = np.trapz(Pyy,F) # trapz y,x !
    fc3 = np.sqrt(fc1*fc2)
    Pxx = np.var(X)/fc1*Pxx   
    Pyy = np.var(Y)/fc2*Pyy
    Pxy = np.sqrt(np.var(Y)*np.var(X))/fc3*Pxy
    
    df = F[1]-F[0]
    
    # output spectrum characteristics
    if jaPrint:
        print('number of samples used : ' + str(np.size(X)))
        print('degrees of freedom     : ' + str(DOF))
        print('resolution             : ' + str(df))
    
    P = [Pxx,Pyy,Pxy]
    
    return F,P



def calcmoments(fx,vy,fmin=-9999,fmax=9999):
    """
    Created on Tue Mar 26 13:13:17 2019
    
    @author: vuik
    """    
    if np.min(np.diff(fx))<=0:
        raise ValueError('fx should be increasing')
    
    imin = bisect(fx,fmin)
    imax = bisect(fx,fmax)
    
    #    df = F(1);                                                      % bandwidth (Hz)
    #    for i=-2:4                                                      % calculation of moments of spectrum
    #    	moment(i+3)=sum(F(integmin:integmax).^i.*Snf(integmin:integmax))*df;
    #    %	fprintf('m%g  =  %g\n',i,moment(i+3));
    #    end
    
    fsel = fx[imin:imax]
    vsel = vy[imin:imax]
    
    #to prevent runtimewarnings on fx[0]=0 we exclude this value if present
    if fsel[0]==0:
        fsel = fsel[1:]
        vsel = vsel[1:]
        
    m0  = np.trapz(vsel*(fsel**0),fsel)
    m1  = np.trapz(vsel*(fsel**1),fsel)
    m2  = np.trapz(vsel*(fsel**2),fsel)
    m3  = np.trapz(vsel*(fsel**3),fsel)
    m4  = np.trapz(vsel*(fsel**4),fsel)
    mm1 = np.trapz(vsel*(fsel**(-1)),fsel)
    mm2 = np.trapz(vsel*(fsel**(-2)),fsel)
    
    moments = pd.DataFrame({'mm2':mm2,'mm1':mm1,'m0':m0,'m1':m1,'m2':m2,'m3':m3,'m4':m4},index=[0])
    
    return moments


def fungrad(mu,P_Ni,ntheta,dtheta,qk1):
    """
    Created on Mon Jun  3 21:23:52 2019
    
    Computes cost function and partial derivatives for 2D MEM estimator
    
    aims to minimize the function q(mu) with gradient function g(mu)
    additional arguments (just passed to function):
    P_Ni,ntheta,dtheta,qk1
    
    cost function for 2D MEM estimator    
    compute function value q(mu)
    
    @author: vuik
    """    

    n = 5
    b = 70
    
    a = np.zeros(ntheta)
    a = -np.inner(qk1,mu)
    # prevent a from going to infinity
    if np.all(a > 0): # original Matlab code: only "if a>0"
        q3 = (a**(-n) + b**(-n))**(-1/n)
    else:
        q3 = a
    q2  = np.sum(np.exp(q3))*dtheta
    q01 = np.sum(mu*P_Ni)
    q   = (q2 + q01)
    
    # compute gradient g(mu)
    b = 70
    n = 5
    a = -np.inner(qk1,mu)
    q3 = a
    if np.all(a > 0): # original Matlab code: only "if a>0"
        fb = (1+a**n/b**n)**(-1-1/n)
    else:
        fb = 1
    
    g = []
    for j in np.arange(5):
        gval = P_Ni[j] - np.sum(qk1[:,j]*fb*(np.exp(q3))*dtheta)
        g.append(gval)
    g = np.array(g)
    
    return q, g



def wave_MEMpuv(_p, _u, _v, depth, sensorlevel, bedlevel,freq,
                fresolution=0.05,
                ntheta = 64,      
                fcorrmin = 0.05,
                fcorrmax = 1.00, # default in Neumeier toolbox: 0.33 Hz. Changed into 1.00 Hz for low frequency waves in lakes (VV)
                maxiter = 100,    # search options
                ):
    """
    Created on Thu Aug  1 14:23:20 2019
    
    @author: vuik
    
    Modifications:
    22-08-2019:	fmaxS changed from 1.0 into 2.0 Hz
    30-08-2019:	correction for pressure attenuation modified into Neumeier method
    
    calculate directional spectra from p,u,v measurements
    using maximum entropy method (MEM)
    
    entropy is a measure for number of ways to realize the considered state of a system. 
    the system with maximum entropy is the most probable system.
    
    p: surface elevation signal (in m hydrostatic pressure)
    u: cross-shore velocity signal (in m/s)
    v: alongshore velocity signal (in m/s)
    
    output
    Hm0: significant wave height
    vy: frequency distribution
    theta: directional distribution
    S: spectral densities
    
    Based on: Ad Reniers 5-8-1999, MEM_dir.m (Matlab)
    Method by Lygre & Krogstad 1986, adapted by Cees de Valk
    Converted to python and adapted by Vincent Vuik, 29-5-2019
    Adapted my Marlies van der lugt, 15-1-2021
    
    """    
    p = _p.tolist()
    u = _u.tolist()
    v = _v.tolist()
    
    zIns = sensorlevel # height of instrument (m+NAP)
    zBot = bedlevel   # bottom height near instrument (m+NAP)
    
    elev  = zIns-zBot # distance from instrument to bottom
    hmean = depth+bedlevel # GEM_WATHTE in Aquo: average water level (m+NAP) = mean(p)+zIns

    dt    = 1/freq  # sampling interval (at 4 Hz: 1/4 sec)
    
    # determine cross-spectral densities (co-spectra only)
    # cross spectra, for p,u: 1st column is p,p, 2nd u,u, 3rd p,u
    
    #ML adapted to desired fresolution to determine instead of fixed width
    # nsample = 2**12 # default
    # nsample = 2**7 # Robert: opletten! 2^7
    
    # Number of data points required for desired fresolution        
    #make sure it is a power of 2:    
    xx = -np.log(dt*fresolution)/np.log(2)
    nsample = 2**(np.ceil(xx))
    
    # pdb.set_trace()
    
    # Check if there are sufficient data points to acquire the set frequency
    # resolution
    if nsample>len(p):
        # reset Nr
        nsample = len(p)
        print('Warning: The required frequency resolution could not be achieved.')
    
    
    F,P = jspect(p,p,nsample,dt,DW='hann',OVERLAP=1,DETREND=1)
    Ppp = np.real(P[2]) # PHI(:,1): p,p
    df  = F[1]-F[0]
    F,P = jspect(p,u,nsample,dt,DW='hann',OVERLAP=1,DETREND=1)
    Ppu = np.real(P[2]) # PHI(:,2): p,u
    F,P = jspect(p,v,nsample,dt,DW='hann',OVERLAP=1,DETREND=1)
    Ppv = np.real(P[2]) # PHI(:,3): p,v
    F,P = jspect(u,u,nsample,dt,DW='hann',OVERLAP=1,DETREND=1)
    Puu = np.real(P[2]) # PHI(:,5): u,u
    F,P = jspect(v,v,nsample,dt,DW='hann',OVERLAP=1,DETREND=1)
    Pvv = np.real(P[2]) # PHI(:,4): v,v
    F,P = jspect(u,v,nsample,dt,DW='hann',OVERLAP=1,DETREND=1)
    Puv = np.real(P[2]) # PHI(:,6): u,v
    
    # pdb.set_trace()
    # mean direction: cross spectra p,v divided by p,u
    theta_m = np.arctan(Ppv/Ppu)
    
    # construct normalised spectral densities
    P_N1 = np.ones(np.size(F))
    P_N2 = Ppu/np.sqrt(Ppp)/np.sqrt(Puu+Pvv)
    P_N3 = Ppv/np.sqrt(Ppp)/np.sqrt(Puu+Pvv)
    P_N4 = (Puu-Pvv)/(Puu+Pvv) 
    P_N5 = 2*Puv/(Puu+Pvv)
    
    Ppp_MAX = np.max(Ppp)
    
    # compute spectrum
    # construct theta-array
    ntheta  = ntheta
    thetmax = np.pi
    dtheta  = 2*thetmax/ntheta
    theta   = np.arange(start=-thetmax,stop=thetmax,step=dtheta)
    
    # construct fourier components
    qk1_1 = np.ones(np.size(theta))
    qk1_2 = np.cos(theta)
    qk1_3 = np.sin(theta)
    qk1_4 = np.cos(2*theta)
    qk1_5 = np.sin(2*theta)
    qk1   = np.column_stack((qk1_1,qk1_2,qk1_3,qk1_4,qk1_5))
    

    
    # obtain f-ky spectral estimate for nl frequencies
    # df    = F[1]-F[0]
    # fmaxS = 4.0 #ML: increased up to 4 Hz # Vincent: maximum frequency for 2D spectrum: changed 
                 #from 1.0 to 2.0 Hz for short waves in Markermeer/IJsselmeer

    
    # first row with zeros for zero frequency
    S = np.zeros(ntheta)
    
    trans =  np.zeros(F.shape)
    D     =  np.zeros([len(F),ntheta])
    for iF in np.arange(1,len(F)): # per non-zero frequency, skip the first
        P_N  = [P_N1[iF],P_N2[iF],P_N3[iF],P_N4[iF],P_N5[iF]]
        P_Ni = np.conj(P_N) # complex conjugate of normalized spectral densities
        
        if Ppp[iF]/Ppp_MAX > 0.0001: # only calculate if significant energy is present
            
            mu0 = np.zeros(5) # starting point
            
            # find minumum of cost function for 2D MEM estimator, as a function of mu
            res = minimize(fungrad, jac=True, method='BFGS', x0=mu0, 
                           args=(P_Ni,ntheta,dtheta,qk1), options={'maxiter':maxiter})
            mu1 = res.x
            
            # construct directional spreading function
            D[iF,:] = np.exp(-(np.inner(qk1,mu1))) # original code: matrix with size of S
        else:
            D[iF,:] = np.zeros(ntheta) 
        
        # apply pressure attenuation correction, similar to Neumeier
        f        = F[iF]
        k        = disper(2*np.pi*f,depth)
        with np.errstate(over='ignore'):
            KptFull  = np.cosh(k*depth)/(np.cosh(k*elev))  
        maxcorr  = 5 # 5 in terms of pressure is 5^2=25 in terms of variance density (trans = Kpt**2)
        fvals    = [0,0.75*fcorrmin,fcorrmin,fcorrmax,1.25*fcorrmax,10] # linear tapering of maximum value (VV)
        maxvals  = [1,1,maxcorr,maxcorr,1,1]
        KptMax   = np.interp(f,fvals,maxvals)
        
        Kpt      = np.min([KptFull,KptMax])
        trans[iF]    = Kpt**2
        
        # original code in MEMpuv:
        # trans = (np.cosh(k*depth)/(np.cosh(k*elev)))**2
        # trans = np.min([trans,9]) # 9 = maxcorr**2
    
        # determine 2D variance density spectrum
        S     = np.vstack((S,Ppp[iF]*D[iF,:]*trans[iF])) 
        
    # analyse 1D spectrum
    vy    = np.trapz(S,theta)
    
    return F,vy,theta,S
    # return F, theta, S

def puv_wavedir(freq,_p,_u,_v,fmin=0.01,fmax=1,fresolution=0.02):
    
    p = _p.tolist()
    u = _u.tolist()
    v = _v.tolist()
    
    dt    = 1/freq  # sampling interval (at 4 Hz: 1/4 sec)
        
        
    # Number of data points required for desired fresolution        
    #make sure it is a power of 2:    
    xx = -np.log(dt*fresolution)/np.log(2)
    nsample = 2**(np.ceil(xx))    

    
    
    # Check if there are sufficient data points to acquire the set frequency
    # resolution
    if nsample>len(p):
        # reset Nr
        nsample = len(p)
        print('Warning: The required frequency resolution could not be achieved.')
    
    F,P = jspect(p,u,nsample,dt,DW='hann',OVERLAP=1,DETREND=1)
    Ppu = np.real(P[2]) # PHI(:,2): p,u
    F,P = jspect(p,v,nsample,dt,DW='hann',OVERLAP=1,DETREND=1)
    Ppv = np.real(P[2]) # PHI(:,3): p,v
          
    # mean direction: cross spectra p,v divided by p,u
    ef = np.where(np.logical_and(F>fmin,F<fmax))
    
    return np.arctan2(np.sum(Ppv[ef]),np.sum(Ppu[ef]))*180/np.pi

def _calcParabolaVertex(x1, y1, x2, y2, x3, y3):
    '''
    computes the coordinates of the vertex of a parabola through the three points (x1,y1), (x2,y2) and (x3,y3)
    '''
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

    xv = -B / (2 * A)
    yv = C - B * B / (4 * A)
    return xv, yv

def compute_wave_params(fx,vy,S=None,theta=None,fmin=0.01,fmax = 1.5,
                        returntype = 'list'):
    '''

    Converted to python and adapted by Vincent Vuik, 29-5-2019
    Adapted my Marlies van der lugt, 15-1-2021    
    
    '''
    ef    = np.where(np.logical_and(fx>=fmin,fx<=fmax))
    fsel  = fx[ef]
    vsel  = vy[ef]
    em    = np.argmax(vsel) 
    fp    = fsel[em] # peak frequency								
    Tp    = 1/fp # peak period

    # smoothes peak wave period
    if em > 0:
        fps, _ = _calcParabolaVertex(fsel[em-1], vsel[em-1], fsel[em], vsel[em], fsel[em+1], vsel[em+1])
        Tps = 1 / fps  # smoothed peak period
    else:
        Tps = Tp

    mom   = calcmoments(fx,vy,fmin=fmin,fmax=fmax)
    m0    = mom['m0']
    Hm0   = 4*np.sqrt(mom['m0']) # Hm0 = 4*m0^0.5
    Tm01  = mom['m0']/mom['m1'] # Tm01 = m0/m1
    Tm02  = np.sqrt(mom['m0']/mom['m2']) # Tm02 = (m0/m2)^0.5
    Tmm10 = mom['mm1']/mom['m0'] # Tm-1,0 = (m-1/m0)

    if ((S is None) or (theta is None)):
  
        if returntype == 'list':
            return Hm0, Tp, Tm01, Tm02, Tmm10, Tps
        else:
            return {'Hm0':Hm0, 'Tp':Tp, 'Tm01':Tm01, 'Tm02':Tm02,
               'Tmm10':Tmm10, 'Tps': Tps}
    else:            
        # determine mean wave direction and directional spreading, Kuik et al. 1988
        # see SWAN user manual for definitions
        # theta0 and dspr only for range [fmin,fmax]
        dtheta = theta[1]-theta[0]
        sT     = np.sin(theta)
        cT     = np.cos(theta)
        idsel  = np.where(np.logical_and(fx>=fmin, fx<=fmax))
        m0Sin  = np.trapz((np.dot(S[idsel,:], sT))*dtheta,fx[idsel])
        m0Cos  = np.trapz((np.dot(S[idsel,:], cT))*dtheta,fx[idsel])
        theta0 = np.rad2deg(np.arctan2(m0Sin, m0Cos)) % 360 # cartesian, degrees
        dspr2  = 2*(1-np.sqrt((m0Sin/m0)**2+(m0Cos/m0)**2))
        dspr   = np.rad2deg(np.sqrt(dspr2)) # directional spreading in degrees 
         
        if returntype == 'list':
            return Hm0, Tp, Tm01, Tm02, Tmm10, Tps, theta0, dspr
        else:
            return {'Hm0': Hm0, 'Tp': Tp, 'Tm01': Tm01, 'Tm02': Tm02,
               'Tmm10': Tmm10, 'Tps': Tps, 'theta0': theta0[0], 'dirspread': dspr[0]}
   
    
def compute_dirspread(theta,S,theta0):
    '''
    Directional width like in Ruessink et al. 2012
    '''
    m0 = np.trapz(S,theta)
    m1 = np.trapz(S*np.cos(theta-theta0/180*np.pi),theta)
    # um = S*np.cos(theta-theta0/180*np.pi)
    # m1=np.zeros(S.shape[0])
    # for ff in np.arange(S.shape[0]):
    #     m1[ff] = np.trapz(um[ff,:],theta)
    with np.errstate(divide='ignore', invalid='ignore'):    
        dspr2  = np.sqrt(2*(1-m1/m0))
    dspr   = np.rad2deg(np.sqrt(dspr2)) # directional spreading in degrees 
    return dspr


def compute_spectral_width(fx,vy,fmin=None,fmax=None):
    '''
    computes kappa (Battjes and van Vledder 1984, see also Holthuysen eq 4.2.6 and p. 67)
    Matlab implementation by Matthieu de Schipper
        '''
        
    if not (fmin is None or fmax is None):
        ef  = np.where(np.logical_and(fx>=fmin,fx<=fmax))
        fx  = fx[ef]
        vy  = vy[ef]
    
    mom   = calcmoments(fx,vy)
    
    m0 = mom['m0']
    Tm02  = np.sqrt(mom['m0']/mom['m2']) # Tm02 = (m0/m2)^0.5
    fbar0 = 1/Tm02
    
    part1 = np.trapz(np.cos(2*np.pi*fx/fbar0)*vy,fx)
    part2 = np.trapz(np.sin(2*np.pi*fx/fbar0)*vy,fx)
    
    # pdb.set_trace()
    kappa = np.sqrt(1/m0**2*(part1**2+part2**2))
    
    return kappa


def quality_check_signal(x,
                         pflags = 0.05,
                         vari = 4,
                         tresuval = 0.01,
                         replace=False,
                         interpMethod = 'cubic'):
    '''
    quality_check_signal

    Parameters
    ----------
    x : PANDAS TIMESERIES 
        SIGNAL.
    pflags : FLOAT, optional
        PERCENTAGE OF SAMPLES THAT MAY BE FLAGGED. The default is 0.05.
    vari : INTEGER, optional
        ALLOWED VARIATION FROM ONE SAMPLE TO NEXT SAMPLE OF VARI*STD. The default is 4.
    tresuval : FLOAT, optional
        AT LEAST PERCENTAGE TRESUVAL OF UNIQUE VALUES IN SIGNAL. The default is 0.001.
    replace : LOGICAL, optional
        REPLACE FLAGGED SAMPLES BY INTERPOLATION. The default is False.
    replace : STR, optional
        INTERPOLATION METHOD. The default is cubic.
    Returns
    -------
    x2 : PANDAS TIMESERIES
        CORRECTED OR QUALITY CHECKED SIGNAL.

    Marlies van der lugt, 15-1-2021 
    '''
    
    if np.sum(np.isnan(x))>0:
        print('x contains nans')
        return x
    
    pex = x.copy()
    #only perform on non-nan'values of x
    pex[np.logical_not(pd.isna(pex))] = signal.detrend(pex[np.logical_not(pd.isna(pex))])
    # pex = signal.detrend(x)
    trend = x - pex
    x = pd.Series(pex)
    
    #normal coordinates, no. of realizations that are more than 4 std from mean
    z = (x-x.mean())/x.std()
    
    o1 = z>vari
    
    #jumps from one sample to then next, no of jumps larger than 4 std
    dz = np.diff(x)/x.std() 
    o2 = dz>vari
    
    outlier = np.logical_or(o1[1:],o2)
    
    if ( np.sum(outlier)>len(z)*pflags ) :
        print('bad quality signal, percentage outliers > {}'.format(pflags))
        x2   = np.nan*x
    elif len(np.unique(np.round(z,3)))/len(z) < tresuval:
        print('precision of instrument too low or only measuring noise')
        x2 = np.nan*x
    elif replace is True and np.sum(outlier)>0:
        print('replacing {}/{} outlier values'.format(np.sum(outlier),len(x)))
        x[(x-x.mean())/x.std()>vari] = np.nan
        x[np.abs(x-x.shift(-1))/x.std()>vari]=np.nan
        x2 = x.interpolate(interpMethod)        
    else:
        x2 = x
        
    return x2+trend


def fft_with_frequency(sf,p):
    '''

    Parameters
    ----------
    sf : FLOAT
        SAMPLING FREQUENCY.
    p : NP.ARRAY 
        SIGNAK.

    Returns
    -------
    dict
        FFT SPECTRA IN SEVERAL FLAVOURS, SEE SCRIPT FOR INTERPRETATION.
        {'V':V,'A':A,'PHI':PHI,'Qn':Qn,'f':f,'Q':Q,'ft':ft} 

    '''
    

    #frequency range
    nf = int(np.floor(len(p)/2))
    df = sf/len(p)
    # ft = df*(np.arange(0,nf) - np.arange(nf,0,-1))
    ffa = np.arange(0,nf)
    ffb = -np.arange(nf,0,-1)
    ft = df*np.append(ffa,ffb)
    
    #positive side only
    f = sf/2*np.linspace(0,1,int(len(p)/2+1))
    f = f[0:nf]
    
    #fourier transform
    Q = fft(p)
    
    #positive side and normalized
    Qn = Q/len(p)
    Qn = 2*Qn[0:nf]
    
    # variance density
    V = np.abs(Qn)**2/df
    
    #amplitude spectrum
    A = abs(Qn)
    
    #phase spectrum
    PHI = np.arctan2(Qn.imag,Qn.real)

    return {'V':V,'A':A,'PHI':PHI,'Qn':Qn,'f':f,'Q':Q,'ft':ft}    
 

def pressure2velocity(sf,p,z,hmean,zi,zb,
                      rho=1000,
                      g = 9.8,
                      fminfac = 0.5,
                      fmaxfac = 3.5,
                      freelow = False,
                      freehigh = False,
                      windowing = False):
    '''
    pressure2velocity reconstructs the horizontal orbital velocity and the surface
    elevation from pressure observations using linear wave theory. Only variance 
    between fminfac*fp and fmaxfac*fp is included, unless freelow or freehigh are 
    set to TRUE.

    Parameters
    ----------
    sf : FLOAT
        SAMPLING FREQUENCY.
    p : NUMPY ARRAY
        PRESSURE SIGNAL IN [PA].
    z : FLOAT
        ELEVATION ABOVE THE BED THE ORBITAL VELOCITY WILL BE COMPUTED.
    hmean : FLOAT
        WATER DEPTH.
    zi : FLOAT
        INSTRUMENT LEVEL WRT. REFERENCE [M+NAP].
    zb : FLOAT
        BED LEVEL WRT REFERENCE [M+NAP].
    rho : FLOAT, optional
        WATER DENSITY [KG/M3]. The default is 1000.
    g : FLOAT, optional
        GRAVITATION ACCELERATION [M/S2]. The default is 9.8.
    fminfac : FLOAT, optional
        LOWER LIMIT OF INCLUDED FREQUENCIES IN RECONSTRUCTING SIGNAL. The default is 0.5.
    fmaxfac : FLOAT, optional
        UPPER LIMIT OF INCLUDED FREQUENCIES IN RECONSTRUCTING SIGNAL. The default is 3.5.
    freelow : BOOLEAN, optional
        APPLY NOW LOWER LIMIT TO INCLUDED FREQS IN RECONSTRUCTING SIGNAL. The default is False.
    freehigh : BOOLEAN, optional
        APPLY NO UPPER LIMIT TO INCLUDED FREQS IN RECONSTRUCTING SIGNAL. The default is False.
    windowing : BOOLEAN, optional
        APPLY WINDOW [PROBABLY UNNECESSARY, RECOMMENDED TO LEAVE TO FALSE]. The default is False.

    Returns
    -------
    eta : NUMPY ARRAY
        RECONSTRUCTED SURFACE ELEVATION, UNREFERENCED [M].
    uf_z : NUMPY ARRAY
        RECONSTRUCTED VELOCITY AT INQUIRED ELEVATION ABOVE BED.
    pt : NUMPY ARRAY
        RECONSTRUCTED PRESSURE SIGNAL FOR CHECKING PURPOSES.

    '''
    
    p = signal.detrend(p)/rho/g
       
    #find spectral peak
    fx,vy = spectrum_simple(sf,p)
    fp = fx[vy==vy.max()]    
    if fp==0:
        vy[0]=0
        fp = fx[vy==vy.max()]
        
    
    # use 3 partially overlapping windows of half the length
    # of the original signal to prevent blowing up the tail and toes of the signal
    windowing = False
    if not windowing:            
        #time array
        t = 1/sf*np.arange(0,len(p))
       
        #compute phase and amplitude spectrum
        dic = fft_with_frequency(sf,p)
        f = dic['f']
        A = dic['A']
        PHI = dic['PHI']
        
        
        w = 2*np.pi*f
        k = disper(w,hmean,g=g)
        
        z = 0.3 # meter above the bed
        elev = zi-zb
        
        
        pt = 0*t
        eta = 0*t
        uf_z = 0*t
        uf_zi = 0*t
        
        for i in np.arange(0,len(w)):
            
            if (f[i]>=fminfac*fp or freelow) and (f[i]<=fmaxfac*fp or freehigh):
                
                eta   = eta   + A[i]      * np.cosh(k[i]*hmean)/np.cosh(k[i]*elev)                                       * np.cos(w[i]*t+PHI[i])               
                uf_z  = uf_z  + w[i]*A[i] * np.cosh(k[i]*z)/np.cosh(k[i]*elev) * np.cosh(k[i]*hmean)/np.sinh(k[i]*hmean) * np.cos(w[i]*t+PHI[i])                
                uf_zi = uf_zi + w[i]*A[i] * np.cosh(k[i]*hmean) / np.sinh(k[i]*hmean)                                    * np.cos(w[i]*t+PHI[i])
                
                #check reconstruction of pressure 
                pt = pt + A[i]*np.cos(w[i]*t+PHI[i])  
                
                
    else:
        # we apply a window, but at the cost of half the signal.
        
        Nr = int(2*np.floor(len(p)/4))  
        F = fourier_window(Nr)
        etaList = []
        for it in np.arange(0,3):
            
            indexRange =np.arange(it*Nr/2,(it+1)*Nr/2+Nr/2).astype(int)
            pp = F*p[indexRange]
            
            #time array
            t = 1/sf*np.arange(0,len(pp))
           
            #compute phase and amplitude spectrum
            dic = fft_with_frequency(sf,pp)
            f = dic['f']
            A = dic['A']
            PHI = dic['PHI']
            
            
            w = 2*np.pi*f
            k = disper(w,hmean,g=g)
            
            z = 0.3 # meter above the bed
            elev = zi-zb
            
            
            pt = 0*t
            eta = 0*t
            uf_z = 0*t
            uf_zi = 0*t
            
            for i in np.arange(0,len(w)):
                
                if (f[i]>=fminfac*fp or freelow) and (f[i]<=fmaxfac*fp or freehigh):
                    
                    eta   = eta   + A[i]      * np.cosh(k[i]*hmean)/np.cosh(k[i]*elev)                                       * np.cos(w[i]*t+PHI[i])               
                    uf_z  = uf_z  + w[i]*A[i] * np.cosh(k[i]*z)/np.cosh(k[i]*elev) * np.cosh(k[i]*hmean)/np.sinh(k[i]*hmean) * np.cos(w[i]*t+PHI[i])                
                    uf_zi = uf_zi + w[i]*A[i] * np.cosh(k[i]*hmean) / np.sinh(k[i]*hmean)                                    * np.cos(w[i]*t+PHI[i])
                    
                    #check reconstruction of pressure 
                    pt = pt + A[i]*np.cos(w[i]*t+PHI[i])
    
            etaList.append([eta,uf_z,uf_zi,pt])
            
       
        # construct the center half of the surface elevation timeseries from 
        # the overlapping windows using the windowFilter as weights    
       
        # glueing the overlapping windows in an ugly way
        ix = int(Nr/2)       
        w1 = F[ix:]
        w2 = F[0:ix]
        
        etaa = etaList[0][0][ix:]*w1 + etaList[1][0][0:ix]*w2 / (w1+w2)
        etab = etaList[1][0][ix:]*w1 + etaList[2][0][0:ix]*w2 / (w1+w2)
        eta = np.append(etaa,etab)    
        
        uf_za = etaList[0][1][ix:]*w1 + etaList[1][1][0:ix]*w2 / (w1+w2)
        uf_zb = etaList[1][1][ix:]*w1 + etaList[2][1][0:ix]*w2 / (w1+w2)
        uf_z = np.append(uf_za,uf_zb)     
    
        uf_zia = etaList[0][2][ix:]*w1 + etaList[1][2][0:ix]*w2 / (w1+w2)
        uf_zib = etaList[1][2][ix:]*w1 + etaList[2][2][0:ix]*w2 / (w1+w2)
        uf_zi = np.append(uf_zia,uf_zib)      
        
        pta = etaList[0][3][ix:]*w1 + etaList[1][3][0:ix]*w2 / (w1+w2)
        ptb = etaList[1][3][ix:]*w1 + etaList[2][3][0:ix]*w2 / (w1+w2)
        pt = np.append(pta,ptb)  
    
    
    return eta,uf_z,pt
    
def band_pass_filter(sf,x,fmin=0.05,fmax=3):
    '''
    band pass filters a signal to the range fmin and fmax. Gives identical results to band_pass_filter2 BELOW
        
    Parameters
    ----------
    sf : FLOAT
        SAMPLING FREQUENCY.
    x : NUMPY ARRAY
        SIGNAL.
    fmin : FLOAT, optional
        LOWER BOUND OF BAND PASS FILTER. The default is 0.05.
    fmax : FLOAT, optional
        UPPER BOUND OF BAND PASS FILTER. The default is 3.

    Returns
    -------
    NUMPY ARRAY
        BAND-PASS FILTERED SIGNAL.
    '''
        
    pex = x-signal.detrend(x)
    x = x-pex
           
    #compute phase and amplitude spectrum
    dic = fft_with_frequency(sf,x)
    f = dic['f']
    A = dic['A']
    PHI = dic['PHI']
       
    w = 2*np.pi*f   
    t = 1/sf*np.arange(0,len(x))
    
    xf = 0*t        
    for i in np.arange(0,len(w)):       
        if (f[i]>=fmin) and (f[i]<=fmax):           
            #check reconstruction of pressure 
            xf = xf + A[i]*np.cos(w[i]*t+PHI[i]) 
            
    return xf

def band_pass_filter2(sf,x,fmin=0.05,fmax=3,retrend=True):
    '''
    band pass filters a signal to the range fmin and fmax. Gives identical results to band_pass_filter above
        

    Parameters
    ----------
    sf : FLOAT
        SAMPLING FREQUENCY.
    x : NUMPY ARRAY
        SIGNAL.
    fmin : FLOAT, optional
        LOWER BOUND OF BAND PASS FILTER. The default is 0.05.
    fmax : FLOAT, optional
        UPPER BOUND OF BAND PASS FILTER. The default is 3.

    Returns
    -------
    NUMPY ARRAY
        BAND-PASS FILTERED SIGNAL.
    '''  

    pex = x-signal.detrend(x)
    x = x-pex
           
    #frequency range
    nf = int(np.floor(len(x)/2))
    df = sf/len(x)
    ffa = np.arange(0,nf)
    ffb = -np.arange(nf,0,-1)
    f = df*np.append(ffa,ffb)
    
    Q = fft(x)
        
    Q2 = Q[:]
    Q2[np.logical_or(abs(f)<fmin,abs(f)>=fmax)]=0
    
    if retrend is True:
        return ifft(Q2).real + pex 
    else: 
        return ifft(Q2).real


def compute_SVD_angle(sf,u,v,fmin,fmax):
    u = band_pass_filter2(sf,u,fmin=fmin,fmax=fmax)
    v = band_pass_filter2(sf,v,fmin=fmin,fmax=fmax)
    
    coords = np.vstack((u,v))
    U, sigma, VH = np.linalg.svd(coords,full_matrices=False)
    T = U*sigma
    #rotation angle
    thet = (np.arctan2(U[1,0],U[0,0])) #%np.pi #cartesian radians  because tan-1(dy,dx)
    
    return thet

def rotate_velocities(u,v,thet):
    '''
    rotates vector (or array) [u,v] clockwise over angle thet (degrees)
    '''
    theta = thet/180*np.pi
    coords = np.vstack((u,v))
    rotMatrix = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    uv,vd = rotMatrix @ coords
    return uv, vd

def SVD_dirspread(ud,vd):
    '''
    computed as in Ruessink (2012) on the parameterization of .....
    max is 45 deg: both ud and vd are equally important
    returns: degrees
    '''
    return np.arctan2(np.sqrt(np.sum(vd**2)/np.sum(ud**2)))/np.pi*180

def compute_SkAs(sf,p,fpfac =None, fbounds = None):
    '''
    

    Parameters
    ----------
    sf : float
        SAMPLING FREQUENCY.
    p : NUMPY ARRAY
        SIGNAL (E.G. PRESSURE OR ORBITAL VELOCITY).
    fpfac : LIST, optional
        BANDPASS LIMITS [FPFACMIN,FPFACMAX]. The default is None.
    fbounds : LIST, optional
        BANDPASS LIMITS [FMIN, FMAX]. The default is None.

    Returns
    -------
    Sk : FLOAT
        SKEWNESS [m3].
    As : FLOAT
        Asymmetry [m3].
    sig : FLOAT
        STD [m].
    '''
    
    p = signal.detrend(p)
    
    if (fpfac == None) and (fbounds == None):
        pf = p    
    elif fpfac == None:
        pf = band_pass_filter2(sf,p,fmin=fbounds[0],fmax=fbounds[1]) 
    elif fbounds == None:
        fx,vy = spectrum_simple(sf,p)
        fp = fx[vy==np.max(vy)]
        if fp==0:
            vy[0]=0
            fp = fx[vy==vy.max()]  
        pf = band_pass_filter2(sf,p,fmin=fpfac[0]*fp,fmax=fpfac[1]*fp) 
        
    Sk  = np.mean(pf**3)
    As  = np.mean((signal.hilbert(pf).imag)**3)  
    sig = np.std(pf)
    return Sk, As, sig



def guza_split_waves(t,zsi,umi,zb,boundopt):
    '''
    Guza_split_waves(t,zsi,umi,zb,boundopt)
    [zsin zsout uin uout] = Guza_split_waves(t,zs,um,zb,boundopt,quietopt)
    t is timevector (s)
    zsi = total surface elevation vector (m+SWL)
    um = depth-averaged current vector (m/s)
    zb = bed level at location of zs and um (m+SWL)
    boundopt = 'boundin', 'boundout', 'boundall', 'free', 'boundupper',
    'sqrt(gh)'
    
    original matlab implementation: Robert McCall
    converted to Python: Marlies van der Lugt
    06/04/2022
    '''
    
    g=9.8
    zsm = np.mean(zsi)
    umm = np.mean(umi)
    
    #average water depth
    h = zsm - zb
    
    #adjust to zero-centered water level and velocity
    zs = zsi - zsm
    um = umi- umm
    
    #detrend signals
    zsd = signal.detrend(zs)
    umd = signal.detrend(um)
    zsm = zsm + (zs-zsd)
    zs = zsd
    umm = umm + (um-umd)
    um = umd
    
    if boundopt=='sqrt(gh)':
        #in timespace
        hh = zs + h
        c = np.sqrt(g*hh)
        q = umd*hh
        ein = (zs*c+q)/(2*c)
        eout = (zs*c-q)/(2*c)
        zsin = ein + zsm
        zsout = eout + zsm
        uin = (np.sqrt(1/hh**2)*c*ein)+umm
        uout = -(np.sqrt(1/hh**2)*c*eout)+umm
        
    else:
        # Time and frequency array 
        n = zs.size
        Z = fft(zs,n)
        U = fft(um,n)

        if len(np.atleast_1d(t))==1:
            dt = t
        else:
            dt = t[1]-t[0]
            
        T  = dt*n
        df = 1/T
        ffa = np.arange(0,np.round(n/2))
        ffb = -np.arange(np.round(n/2),0,-1)
        ff = df*np.append(ffa,ffb)+0.5*df
        w = 2*np.pi*ff
        k = disper(w,np.mean(h),g)
        c=w/k
        
        #hf cutoff
        # pdb.set_trace()
        filterfreq = 0.01 #filter length over 0.01Hz
        minfreq = 0.03 #33s wave is longst considered in primary spectrum
        incl = ff>=minfreq
        ftemp = ff[incl]
        vartemp = (Z.real)**2
        vartemp = vartemp[incl]
        
        # window = np.max(np.round(filterfreq/df),1)
        # vartemp = filterwindow()
        
        fp = ftemp[vartemp==np.max(vartemp)][0]
        Tp = 1/fp
        
        
        
        
        #cutoff high frequency c
        hfc = min(10*fp,np.max(ff))
        hwc = 2*np.pi*hfc
        kfc = disper(hwc,h,g)
        c = np.maximum(hwc/kfc,c)
        
    

        #find the properties of the peak waves (used for bound components)
        wp = 2*np.pi*fp
        kp = disper(wp,h,g)
        cp = wp/kp
        nnp = 0.5*(1+2*kp*h/np.sinh(2*kp*h)) 
        cgp = nnp*cp
        
        #select frequencies that are thought to be bound
        if boundopt=='boundin':
            freein = np.abs(ff)>0.5*fp          #free above half the peak frequency
            boundin = np.abs(ff)<=0.5*fp        #bound below half the peak frequency
            freeout = np.full(ff.shape, True)
            boundout = np.full(ff.shape,False)
        elif boundopt=='boundout':
            freein = np.full(ff.shape,True)
            boundin = np.full(ff.shape,False)
            freeout = np.abs(ff)>0.5*fp          #free above half the peak frequency
            boundout = np.abs(ff)<=0.5*fp        #bound below half the peak frequency
        elif boundopt=='boundall':
            freein = np.abs(ff)>0.5*fp          #free above half the peak frequency
            boundin = np.abs(ff)<=0.5*fp        #bound below half the peak frequency
            freeout = np.abs(ff)>0.5*fp          #free above half the peak frequency
            boundout = np.abs(ff)<=0.5*fp        #bound below half the peak frequency   
        elif boundopt=='free':
            freein = np.full(ff.shape,True)
            boundin = np.full(ff.shape,False)            
            freeout = np.full(ff.shape, True)
            boundout = np.full(ff.shape,False)   
        elif boundopt =='boundupper':
            freein = np.abs(ff)>2*fp          #free above double the peak frequency
            boundin = np.abs(ff)<=2*fp        #bound below double the peak frequency            
            freeout = np.abs(ff)>2*fp          #free above double the peak frequency
            boundout = np.abs(ff)<=2*fp        #bound below double the peak frequency
            
        #find the velocity for all fourier components
        cin = np.nan*np.ones(ff.shape)
        cout = np.nan*np.ones(ff.shape)
        
        cin[freein] = c[freein]
        
        cout[freeout] = c[freeout]
        cout[boundout] = cgp
        
        if boundopt=='boundupper':
            cin[boundin]=cp
            cout[boundout]==cp
        else:
            cin[boundin]=cgp
            cout[boundout]=cgp
            
        #maximize to the long wave celerity
        cin = np.minimum(cin,np.sqrt(g*h))
        cout = np.minimum(cout,np.sqrt(g*h))
        
        #cutoff hf noise
        Set = 0*np.ones(Z.shape)
        Set[np.abs(ff)<=hfc] = 1
        Set[0] = 0
        Set[-1] = 0
        
        ein = ifft(Set*(Z*cout+U*h)/(cin+cout)).real
        eout = ifft(Set*(Z*cout-U*h)/(cin+cout)).real
        zsin = ein + zsm
        zsout = eout + zsm
        uin = ifft(Set*(np.sqrt(1/h**2)*cin*fft(ein))).real+umm
        uout = -ifft(Set*(np.sqrt(1/h**2)*cout*fft(ein))).real+umm
        
    reflc = np.var(zsout)/np.var(zsin)
        
    return zsin, zsout, uin, uout, reflc


def ruessink_predict_shape(Ur):
    '''
    ruessink_predict_shape(Ur)
    
    returns the predicted Sk, As for a given Ur. 
    
    input: numpy array or a float
    output: numpy array or a float
    '''
    p1 = 0
    p2 = 0.857
    p3 = -0.471
    p4 = 0.297
    p5 = 0.815
    p6 = 0.672
           
    B    = p1 + (p2 - p1)/( 1+np.exp((p3-np.log(Ur)/np.log(10))/p4))
    Psi  = -90 + 90*np.tanh(p5/(Ur**p6))
    Sk   = B*np.cos(np.pi*Psi/180)
    As   = B*np.sin(np.pi*Psi/180) 
    return Sk, As

# def wave_zero_crossing(eta,fs,detrend = True,jaFig = False):
#     '''
#     wave_zero_crossing(eta,fs,detrend=True,jaFig=False)

#     Parameters
#     ----------
#     eta : NUMPY ARRAY
#         SURFACE ELEVATION [M].
#     fs : FLOAT
#         DSAMPLING FREQUENCY [Hz].
#     detrend : BOOLEAN, optional
#         DETREND ETA YES/NO. The default is True.
#     jaFig : BOOLEAN, optional
#         PLOT THE RESULT YES/NO. The default is False.

#     Returns
#     -------
#     var : DICT
#         ZERO CROSSING MEAN AND SIGNIFICANT WAVE HEIGHT AND PERIOD.

#     Converted to python and adapted by Marlies van der lugt, 02-02-2021 
#     Matlab version by Arash Karimpour   , 2020-08-01 
    
#     '''
#     if np.sum(np.isnan(eta))>0:
#         print('Signal contains infs or NaNs')
#         return {'EtaRMS':np.nan, 'Hz':np.nan, 'Tz':np.nan, 'Hs':np.nan, 'Ts':np.nan}
    
#     dt = 1/fs
#     t = dt * np.arange(0,len(eta)) + dt
    
#     if detrend: 
#         etaex = eta-signal.detrend(eta)
#         eta = eta-etaex
        
#     # determine zero-crossings
    
    
#     #--------------------------------------------------------------------------
#     # detecting the start point of first wave (fisr complete crest-trough)
    
#     if eta[0]==0 and eta[1]>0 :
#         len3=0
    
#     if (eta[0]==0 and eta[1]<0) or (eta[0]!=0):
#         for i in np.arange(0,len(eta)-2):
#             if eta[i]<0 and eta[i+1]>0 : 
#                 len3 = i
#                 break
    
#     # detecting the end point of last wave (fisr complete crest-trough)
#     if eta[-1]==0 and eta[-2]<0 : 
#         len4 = len(eta)
    
    
#     if (eta[-1]==0 and eta[-2]>0) or (eta[-1]!=0 ):
#         for i in np.arange(len(eta)-2,0,-1):
#             if eta[i]<0 and eta[i+1]>0 :
#                 len4 = i
#                 break
    
#     #--------------------------------------------------------------------------
#     # detecting zero crossing points from original data
    
    
#     xUpCross = []; yUpCross = []; positionXUpCross=[];
#     xDownCross = []; yDownCross = []; positionXDownCross=[];
#     for i in np.arange(len3,len4): 
        
#         #detecting up-crossing zero-crossing points
#         if i==len3 or i==len4:
#             xUpCross.append( t[i])
#             yUpCross.append( 0)
#             positionXUpCross.append( i)
            
#         else:
#             if eta[i]<0 and eta[i+1]>0:
#                 xUpCross.append( 
#                     t[i]-(t[i+1]-t[i])/(eta[i+1]-eta[i])*eta[i] )
#                 yUpCross.append( 0)
#                 positionXUpCross.append( i)
#             elif eta[i]>0 and eta[i+1]<0 :
#                 xDownCross.append(
#                     t[i]-(t[i+1]-t[i])/(eta[i+1]-eta[i])*eta[i] )
#                 yDownCross.append( 0)
#                 positionXDownCross.append( i)
        
    
#     #--------------------------------------------------------------------------
#     # detecting crest and trough from original data
    
#     m = 0
#     n = 0
#     tmax = []; ymax = []; positionXMax = []
#     tmin = []; ymin = []; positionXMin = []    
#     for i in np.arange(positionXUpCross[0],positionXUpCross[-1]):         
       
#         if i>len3 and i<len4 : 
             
#             if eta[i]>eta[i-1] and eta[i]>eta[i+1] and eta[i]>0:
#             # detecting crest                       
#                 if m==n: # check if after last crest, the program detect the trough or not (m==n mean it detected and the new y(i,1) is the crest in next wave)
#                     tmax.append( t[i])
#                     ymax.append( eta[i])
#                     positionXMax.append( i)
#                     m+=1
                    
#                 elif m!=0 and eta[i]>ymax[-1] :
#                 #replacingthe old crest location with new one if the new one is larger                        
#                     tmax[-1] = t[i]
#                     ymax[-1] = eta[i]
#                     positionXMax[-1] = i
    
#             if eta[i]<eta[i-1] and eta[i]<eta[i+1] and eta[i]<0:                                
#             #detecting trough
#                 if n==m-1: # check if after last trough, the program detect the crest or not (n==m-1 mean it detected and the new y(i,1) is the trough in next wave)               
#                     tmin.append( t[i])
#                     ymin.append( eta[i])
#                     positionXMin.append( i)
#                     n+=1
                    
#                 elif n!=0 and eta[i]<ymin[-1]: #replacingthe old crest location with new one if the new one is smaller                   
#                     tmin[-1] = t[i]
#                     ymin[-1] = eta[i]
#                     positionXMin[-1] = i
                        
    
#     #--------------------------------------------------------------------------
#     #calculating Wave height from original data
#     ymin = np.array(ymin)
#     ymax = np.array(ymax)
#     tmin = np.array(tmin)
#     tmax = np.array(tmax)
#     xUpCross = np.array(xUpCross)
#     xDownCross = np.array(xDownCross)
    
    
#     len1 = len(tmax)
#     len2 = len(tmin)
#     if len1!=len2:
#         print('no complete wave cycle, check something went wrong')
        
#     H = np.zeros(len1) #Pre-assigning array
#     H = ymax - ymin
#     # xmean = (tmax+tmin)/2
            
#     #--------------------------------------------------------------------------
#     #calculating Wave period
#     TUp = np.diff(xUpCross)
#     TDown = np.diff(xDownCross)
    
#     xUpCross - xDownCross
    
#     #--------------------------------------------------------------------------
#     #calculating wave parameters
    
#     Etarms = np.sqrt(np.sum(eta**2)/len(eta))
#     HsortIndex = np.argsort(H)
#     Hsort = H[HsortIndex]
#     HTop3rd = np.round(2/3*len(Hsort))
#     Hs= np.mean(Hsort[int(HTop3rd):])
#     Tsort = TUp[HsortIndex]
#     Ts = np.mean(Tsort[int(HTop3rd):])
#     Hz=np.mean(H) # Zero-crossing mean wave height
#     Tz=np.mean(TUp) # Zero-crossing mean wave period

#     var = {'EtaRMS':Etarms, 'Hz':Hz, 'Tz':Tz, 'Hs':Hs, 'Ts':Ts}

#     return var    
    
    
    

      