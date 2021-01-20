import sys
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy.interpolate import griddata
from scipy import optimize
from astropy.cosmology import Planck13 as cosmo
import vegas

import gwaxion

#useful constants
lisaLT=2.5*1e9 # LISA arm lenght in meters
year=31556926 #one year in seconds seconds
t0=cosmo.age(0).value*1e9*year #age of the Universe in seconds
rhocrit=cosmo.critical_density(0).value*1e3 #critical density in SI units
H0=cosmo.H(0).value*1e3/(1e6*gwaxion.PC_SI) #local Hubble constant in SI units

# ###########################################################################
# FUNCTIONS for PSDs and SNR

def PSD_Analytic(f):
    ''' Analytical approximation for the LISA PSD from https://arxiv.org/pdf/1803.01944.pdf, see eqs. 9-13
    
    Arguments
    ---------
    f: float
        frequency in Hz.

    Returns
    -------
    Sn: float
        PSD at frequency f.
    '''
      
    Pacc = (3e-15)**2 *(1.0+(0.4e-3/f)**2)* (1.0+(f/8e-3)**4)
    Poms = (15e-12)**2 * (1.0 + (2e-3/f)**4)
    x = 2.*np.pi*lisaLT*f/C_SI
    R=3/10*(1/(1+0.6*(x)**2))
   
    Sn= 1/(lisaLT**2) * (Poms+2.0*(1.0 + np.cos(x)**2)*Pacc/((2*np.pi*f)**4))/R
    
    return Sn

def PSD_gal(f,Tobs=4):
    ''' Fit to the galactic WD confusion noise from https://arxiv.org/pdf/1803.01944.pdf, see eq. 14 and Table I.
        
    Arguments
    ---------
    f: float
        frequency in Hz.
        
    Tobs: float
        LISA observation time in years. Only available for Tobs=0.5, 1, 2 or 4  (def. 4).

    Returns
    -------
    Sgal(f): float
        PSD at frequency f.
    '''
     
    if Tobs == 0.5 or Tobs == 1 or Tobs == 2 or Tobs == 4:
    
        A=9e-45
        alpha={0.5: 0.133, 1: 0.171, 2: 0.165, 4: 0.138}
        beta={0.5: 243, 1: 292, 2: 299, 4: -221}
        k={0.5: 482, 1: 1020, 2: 611, 4: 521}
        gamma={0.5: 917, 1: 1680, 2: 1340, 4: 1680}
        fk={0.5: 0.00258, 1: 0.00215, 2: 0.00173, 4: 0.00113}

        Sgal = A*f**(-7./3.)*np.exp(f**alpha[Tobs]+beta[Tobs]*f*np.sin(k[Tobs]*f))*(1.+np.tanh(gamma[Tobs]*(fk[Tobs]-f)))
   
    else:
        raise ValueError("galactic background fit only available for Tobs=0.5,1,2,4")

    return Sgal


def SNRav(hav, f, PSD, tgw, Tobs = 4, gal=True, fmin=None, fmax=None): 
    '''
    Signal is of the form h(t)=h0/(1+t/tgw)*cos(\omega_gw t+\phi). Assuming a monochromatic source we approximate the SNR as SNR~2/S_n(f)\int_0^Tobs h(t)^2 (eq. 1 in https://arxiv.org/pdf/1808.07055.pdf),therefore SNR~h0*sqrt(Teff/S_n(f)) where Teff=tgw*Tobs/(tgw+Tobs). 
    For Tobs>>tgw this gives SNR~h0*sqrt(Tobs/S_n(f)) whereas for Tobs<<tgw this gives SNR~h0*sqrt(tgw/S_n(f)).
    For scalar fields the condition Tobs<<tgw is always true in the LISA band, but not necessarily for vector and tensor fields 
    
    The formulas assumes two independent channels. This should be only valid for f<19.09 mHz https://arxiv.org/pdf/1803.01944.pdf, corresponding to boson masses m_b< 5*10^-17 eV. I'll ignore this issue for the moment.
    
    Arguments
    ---------
    hav: float
        inclination averaged GW amplitude (corresponding to the factor A*2/sqrt(5) in eq. 16 of https://arxiv.org/pdf/1803.01944.pdf).
        
    f: float
        frequency in hz.
        
    PSD: float, array
        PSD (without galactic WD background noise) as a function of frequency. Format: 0: frequency; 1: PSD.
        
    tgw: float
        half-life time of the signal in the detector frame
        
    Tobs: float
        LISA observation time in years. If used with gal=True (default) only available for Tobs=0.5, 1, 2 or 4  (def. 4).

    gal: bool
        whether to include galactic WD background noise or not (def. True).
    
    fmin: float
        set min frequency (def. None, use min in PSD).
    
    fmax: float
        set max frequency (def. None, use max in PSD).
    
    Returns
    -------
    SNR: float
        SNR of the signal.
    '''
        
    Tobs_seconds=Tobs*year
    Teff=tgw*Tobs_seconds/(tgw+Tobs_seconds)
    
    Sn = interpolate.interp1d(PSD[:,0],PSD[:,1])
    
    if fmin==None:
        fmin=min(PSD[:,0])
    elif fmin<min(PSD[:,0]):
        raise ValueError("minimun frequency smaller than available in the PSD data. Increase fmin")

    if fmax==None:
        fmax=max(PSD[:,0])
    elif fmax>max(PSD[:,0]):
        raise ValueError("maximum frequency larger than available in the PSD data. Decrease fmax")

        
    if f < fmin or f > fmax:
        SNR=0
        #print('input frequency outside the range: [fmin=%.2e, fmax=%.2e]. Setting SNR=0.'%(fmin,fmax))
    else:
        if gal==True:
            ASD=np.sqrt(Sn(f)+PSD_gal(f,Tobs))
        else:
            ASD=np.sqrt(Sn(f))
        SNR=hav*np.sqrt(Teff)/ASD
 
    return SNR

def SNRback(OmegaGW,PSD,Tobs=4,gal=False, fmin=None, fmax=None,**kwargs):
    '''
    Estimating the SNR of a given background in LISA using eq. 36 of https://arxiv.org/pdf/1310.5300.pdf 
    
    Arguments
    ---------
    OmegaGW: array
         Spectrum \Omega_GW(f). Format: 0: frequency; 1: Omega_GW. 
        
    PSD: float, array
        LISA PSD (without galactic WD background noise) as a function of frequency. Format: 0: frequency; 1: PSD.
        
    Tobs: float
        LISA observation time in years. If used with gal=True (default) only available for Tobs=0.5, 1, 2 or 4  (def. 4).

    gal: bool
        whether to include galactic WD background noise or not (def. True).
    
    fmin: float
        set min frequency (def. None, use min in PSD).
    
    fmax: float
        set max frequency (def. None, use max in PSD).
    
    Accepts all options of integrate.quad()
    
    Returns
    -------
    SNRback: float
        SNR of the background.
    '''
    
    
    Sn = interpolate.interp1d(PSD[:,0],PSD[:,1])
    OmegaGWint = interpolate.interp1d(OmegaGW[:,0],OmegaGW[:,1])
    
    if fmin==None:
        fmin=max(min(OmegaGW[:,0]),min(PSD[:,0]))
    elif fmin<max(min(OmegaGW[:,0]),min(PSD[:,0])):
        raise ValueError("minimun frequency outside available range. Increase fmin")

    if fmax==None:
        fmax=min(max(OmegaGW[:,0]),max(PSD[:,0]))
    elif fmax>min(max(OmegaGW[:,0]),max(PSD[:,0])):
        raise ValueError("maximum frequency outside available range. Decrease fmax")

    
    Tobs_seconds=Tobs*year

    Sh = lambda f: 3*H0**2*OmegaGWint(f)/(2*np.pi**2*f**3) #eq. 3 in https://arxiv.org/pdf/1310.5300.pdf

    if gal==True:
        integrand= lambda f: Sh(f)**2/(Sn(f)+PSD_gal(f,Tobs))**2
    else:
        integrand= lambda f: Sh(f)**2/Sn(f)**2
        
    SNRsq_1s=integrate.quad(integrand, fmin, fmax,**kwargs)
    #error=np.sqrt(Tobs_seconds*SNRsq_1s[1])
    SNRback=np.sqrt(Tobs_seconds*SNRsq_1s[0])
    
    return SNRback

def SfromOmegaGW(OmegaGW):
    '''
    Compute PSD from OmegaGW the SNR using eq. 3 in https://arxiv.org/pdf/1310.5300.pdf
    
    Arguments
    ---------
    OmegaGW: array
         Spectrum \Omega_GW(f). Format: 0: frequency; 1: Omega_GW. 
    
    Returns
    -------
    Sh: array
        PSD of signal for a corresponding Omega_GW.
    '''
    
    Sh=[]
   
    for i in range(0,len(OmegaGW)):
        Sh.append([OmegaGW[:,0][i],3*H0**2*OmegaGW[:,1][i]/(2*np.pi**2*OmegaGW[:,0][i]**3)])
    
    return np.array(Sh)

# ###########################################################################
# FUNCTIONS to compute number of expected CW detections

#TODO: generalize for vectors or tensor fields. This should only require adding those cases to the class BosonCloud in gwaxion
def dN(dn,z,log10mbh, chi_bh, m_b, PSD, Tobs=4, SNRcut=10.,lgw=2, **kwargs):
    '''
    Integrand of eq.62 in  https://arxiv.org/abs/1706.06311. This only works for scalar fields and assuming dominant l=m=1 mode for the moment. 
    
    Arguments
    ---------
    dn: array
       array containing mass function. should be in format: 0: log10mbh; 1: BH spin; 2: redshift; 3: dnoverdlogMdchi.
        
    z: float
        redshift.
        
    log10mbh: float
        logarithm base 10 of black hole mass (initial) in solar masses.
        
    chi_bh: float
        BH spin (initial)
        
    m_b: float
        boson mass in electronvolts.
        
    PSD: float, array
        PSD (without galactic WD background noise) as a function of frequency. Format: 0: frequency; 1: PSD.
    
    Tobs: float
        LISA observation time in years. If used with gal=True (default) only available for Tobs=0.5, 1, 2 or 4  (def. 4).

    SNRcut: float
        cuttoff SNR above which GW signals are observable (def. 10).
        
    lgw: float
        angular multipole number of the GW amplitude (def. 2). At the moment only lgw=2 and lgw=3 for scalar fields are available.
    
    Accepts optional parameters of SNRav().
    
    Returns
    -------
    integrand: float
        integrand of eq.62 in  https://arxiv.org/abs/1706.06311
    '''
    
    Tobs_seconds=Tobs*year

    cloud = gwaxion.BosonCloud.from_parameters(1, 1, 0, m_b=m_b, m_bh=10**log10mbh, chi_bh=chi_bh, 
                                               evolve_params={'y_0': 1E-10}) 
   
    hgwr, fgw = cloud.gw(lgw).h0r, cloud.gw(lgw).f
    tinst= cloud.number_growth_time
    
    distance=cosmo.comoving_distance(z).value*1e6*gwaxion.PC_SI
    h0=np.sqrt(5/(4*np.pi))*hgwr/distance #see notes
    hav=h0*np.sqrt(4/5) #see eq. 16 in https://arxiv.org/pdf/1803.01944.pdf
    fdetector=fgw/(1+z)
        
    dnoverdlogMdchi = griddata(dn[:, [0, 1, 2]], dn[:,3], (log10mbh, chi_bh,z), method='nearest')
    tform=cosmo.lookback_time(z).value*10**9*year
        
    if tinst>0. and tinst<tform:
        tgw=cloud.get_life_time([lgw,lgw])
        tgwredshift=tgw*(1+z) #take into account cosmological redshift in duration of the signal
        SNR=SNRav(hav=hav, f=fdetector, PSD=PSD, tgw=tgwredshift, **kwargs)

        if SNR>SNRcut:
            deltat=min(tgw,t0)
            integrand=4*np.pi*(dnoverdlogMdchi/t0)*(deltat+Tobs_seconds/(1+z))*cosmo.differential_comoving_volume(z).value
        else:
            integrand=0.0
    else:
        integrand=0.0
    
    
    return integrand


def Nevents(dn,m_b,PSD,intlims,method='vegas',nitn=10,neval=1e3,nsumint=30j,**kwargs):
    '''
    Number of expected CW events, Eq.62 in  https://arxiv.org/abs/1706.06311. This only works for scalar fields and assuming dominant l=m=1 mode for the moment. 
    
    Arguments
    ---------
    dn: array
       array containing mass function. should be in format: 0: log10mbh; 1: BH spin; 2: redshift; 3: dnoverdlogMdchi.
       
    m_b: float
        boson mass in electronvolts.
        
    PSD: float, array
        PSD (without galactic WD background noise) as a function of frequency. Format: 0: frequency; 1: PSD.

    intlims: float
        limits of integration. Should be in format: [[log10Mmin, log10Mmax], [spinmin, spinmax], [zmin, zmax]]
        
    method: 'vegas' or 'Riemann sum'
        method to use to compute integral (def. 'vegas').
        
    nitn: int
        number of iterations of the vegas algorithm (def. 10)
        
    neval: int
        number of evaluations of the integrand at each iteration of the vegas algorithm (def. 1e3)
    
    nsumint: int or complex number
        controls step lenght in the Riemann sum (approximate integral as a sum). If number is a float, then it represents the    interval the step lenght. If number is complex, integer part of the complex number is the number of points in a given dimension of the grid (see numpy.mgrid). For the moment all 3 dimensions use the same 'nsumint' (def. 30j).
    
    Accepts optional parameters of SNRav() and dN().
    
    Returns
    -------
    Ntotal: float
        number of expected resolvable CW sources for given boson mass
    '''
    
    dNvec=np.vectorize(dN,excluded=['dn','PSD'])
    
    if method=='vegas':
        @vegas.batchintegrand
        def func(x):
            log10mbh=x[:,0]
            chi_bh=x[:,1]
            z=x[:,2]
            return dNvec(dn=dn,z=z,log10mbh=log10mbh,chi_bh=chi_bh,m_b=m_b,PSD=PSD, **kwargs)

        integ = vegas.Integrator(intlims)

        # step 1 -- adapt to dN; discard results
        integ(func, nitn=5, neval=neval)

        # step 2 -- compute integ but keep results now
        result = integ(func, nitn=nitn, neval=neval)
        print('mb=%.2e, Nevents=%.2f, std=%.2f, chisq/dof=%.2f, Q=%.2f'%(m_b,result.mean,result.sdev,result.chi2/result.dof,result.Q))
        Ntotal=result.mean
        
    elif method=='Riemann sum':
        
        logMmin=min(intlims[0])
        logMmax=max(intlims[0])
        spinmin=min(intlims[1])
        spinmax=max(intlims[1])
        zmin=min(intlims[2])
        zmax=max(intlims[2])
        
        log10mbh,chi_bh,z = np.mgrid[logMmin:logMmax:nsumint,spinmin:spinmax:nsumint,zmin:zmax:nsumint]
        dlog10mbh=(log10mbh[1,0,0]-log10mbh[0,0,0])
        dspin=(chi_bh[0,1,0]-chi_bh[0,0,0])
        dz=(z[0,0,1]-z[0,0,0])
        
        Ntotal=np.sum(dNvec(dn=dn,z=z,log10mbh=log10mbh,chi_bh=chi_bh,m_b=m_b,PSD=PSD, **kwargs))*dlog10mbh*dspin*dz
        print('mb=%.2e, Nevents=%.2f'%(m_b,Ntotal))
        
    return Ntotal


# ###########################################################################
# FUNCTIONS to compute stochastic background

def dOmega(dn,f,log10mbh, chi_bh, m_b, PSD, SNRcut=10., lgw=2, **kwargs):
    '''
    Integrand of eq.64 in  https://arxiv.org/abs/1706.06311. This only works for scalar fields and assuming dominant l=m=1 mode for the moment. Note that since we approximate the flux as a Dirac delta, the redshift integral is done analytically 
    
    Arguments
    ---------
    dn: array
       array containing mass function. should be in format: 0: log10mbh; 1: BH spin; 2: redshift; 3: dnoverdlogMdchi.
        
    f: float
        detector frame frequency in Hz.
        
    log10mbh: float
        logarithm base 10 of black hole mass (initial) in solar masses.
        
    chi_bh: float
        BH spin (initial)
        
    m_b: float
        boson mass in electronvolts.
        
    PSD: float, array
        PSD (without galactic WD background noise) as a function of frequency. Format: 0: frequency; 1: PSD.

    SNRcut: float
        cuttoff SNR above which GW signals are observable (def. 10).
        
    lgw: float
        angular multipole number of the GW amplitude (def. 2). At the moment only lgw=2 and lgw=3 for scalar fields are available.
    
    Accepts optional parameters of SNRav().
    
    Returns
    -------
    integrand: float
        integrand of eq.64 in  https://arxiv.org/abs/1706.06311
    '''
    
    
    cloud = gwaxion.BosonCloud.from_parameters(1, 1, 0, m_b=m_b, m_bh=10**log10mbh, chi_bh=chi_bh, evolve_params={'y_0': 1E-10}) 

    hgwr, fgw = cloud.gw(lgw).h0r, cloud.gw(lgw).f
    tinst = cloud.number_growth_time
    
    z = fgw/f-1
    
    distance=cosmo.comoving_distance(z).value*1e6*gwaxion.PC_SI
    h0=np.sqrt(5/(4*np.pi))*hgwr/distance #see notes
    hav=h0*np.sqrt(4/5) #see eq. 16 in https://arxiv.org/pdf/1803.01944.pdf
    
    dnoverdlogMdchi = griddata(dn[:, [0, 1, 2]], dn[:,3], (log10mbh, chi_bh,z), method='nearest')
    tform=cosmo.lookback_time(z).value*10**9*year
    
    if tinst>0. and tinst<tform:
        tgw=cloud.get_life_time([lgw,lgw])
        tgwredshift=tgw*(1+z) #take into account cosmological redshift in duration of the signal
        
        SNR=SNRav(hav=hav, f=f, PSD=PSD, tgw=tgwredshift, **kwargs) #TO DO: to be fully self-consistent, SNR should also include effect of background itself, so I should put this in a loop adding the background at each step until it converges. The effect should be small though, and actually negligible in terms of deciding detectable range of boson masses, so I'll ignore this issue for the moment
    
        if SNR<SNRcut and f<fgw:
            #if mergers==False: #decide whether it's worth adding this latter. impact of mergers should be very small anyways
            #    Nm=0
            #else:
            #    Nm=Nmergers(dNm=mergers,z=z,tgw=tform,log10mbh=log10mbh)    
            deltat=tform #min(tgw/(1+Nm),t0)#min(tgw/(1+Nm),t0)
            Mcsat = cloud.mass
            integrand=(1/(1e6*gwaxion.PC_SI)**3)*(dnoverdlogMdchi/t0)*Mcsat*deltat/(deltat+tgw)*cosmo.lookback_time_integrand(z)/H0
        else:
            integrand=0.0
    else:
        integrand=0.0
    
    return integrand/rhocrit

def OmegaGW(dn,m_b,PSD,intlims,method='vegas',
            log10freqmin=-5,log10freqmax=-1,num_f=50,nitn=10,neval=1e3,nsumint=50j,printresults=False,**kwargs):
    '''
    Stochastic background spectrum Omega_GW, Eq.64 in  https://arxiv.org/abs/1706.06311. This only works for scalar fields and assuming dominant l=m=1 mode for the moment. 
    
    Arguments
    ---------
    dn: array
       array containing mass function. should be in format: 0: log10mbh; 1: BH spin; 2: redshift; 3: dnoverdlogMdchi.
        
    m_b: float
        boson mass in electronvolts.
        
    PSD: float, array
        PSD (without galactic WD background noise) as a function of frequency. Format: 0: frequency; 1: PSD.

    intlims: float
        limits of integration. Should be in format: [[log10Mmin, log10Mmax], [spinmin, spinmax], [zmin, zmax]]
        
    method: 'vegas' or 'Riemann sum'
        method to use to compute integral (def. 'vegas').
    
    log10freqmin: float
        minimum log10(frequency) (def. -5)
        
    log10freqmax: float
        maximum log10(frequency) (def. -5) (def. -1)
    
    nitn: int
        number of iterations of the vegas algorithm (def. 10)
        
    neval: int
        number of evaluations of the integrand at each iteration of the vegas algorithm (def. 1e3)
    
    nsumint: int or complex number
        controls step lenght in the Riemann sum (approximate integral as a sum). If number is a float, then it represents the    interval the step lenght. If number is complex, integer part of the complex number is the number of points in a given dimension of the grid (see numpy.mgrid). For the moment all 3 dimensions use the same 'nsumint' (def. 30j).
    
    Accepts optional parameters of SNRav() and dOmega().
    
    Returns
    -------
    OmegaGWall: array
         Stochastic background spectrum Omega_GW vs frequency. Formart: 0:frequency, 1: OmegaGW
    '''
    
    
    OmegaGW=[]
    dOmegavec=np.vectorize(dOmega,excluded=['dn','PSD'])
 
    if method == 'vegas':
        for freq in np.logspace(log10freqmin, log10freqmax, num=num_f):
            
            @vegas.batchintegrand
            def func(x):
    
                log10mbh=x[:,0]
                chi_bh=x[:,1]
                return dOmegavec(dn=dn,f=freq,log10mbh=log10mbh, chi_bh=chi_bh, m_b=m_b,PSD=PSD,**kwargs)
    
            integ = vegas.Integrator(intlims)

            result = integ(func, nitn=nitn, neval=neval)
            OmegaGW0=result.mean
            OmegaGW.append([freq,OmegaGW0])
            
            if printresults == True:
                print('freq=%.2e, OmegaGW=%.2e'%(freq,OmegaGW0))
            
    elif method == 'Riemann sum':

        logMmin=min(intlims[0])
        logMmax=max(intlims[0])
        spinmin=min(intlims[1])
        spinmax=max(intlims[1])
            
        log10mbh,chi_bh = np.mgrid[logMmin:logMmax:nsumint,spinmin:spinmax:nsumint]
        dlog10mbh=(log10mbh[1,0]-log10mbh[0,0])
        dspin=(chi_bh[0,1]-chi_bh[0,0])
        
        for freq in np.logspace(log10freqmin, log10freqmax, num=num_f):
            OmegaGW0=np.sum(dOmegavec(dn=dn,f=freq,log10mbh=log10mbh, 
                                           chi_bh=chi_bh, m_b=m_b, PSD=PSD,**kwargs))*dlog10mbh*dspin
            OmegaGW.append([freq,OmegaGW0])
            if printresults == True:
                print('freq=%.2e, OmegaGW=%.2e'%(freq,OmegaGW0))
                
    return np.array(OmegaGW)

