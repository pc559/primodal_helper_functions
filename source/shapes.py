import numpy as np
from numba import jit

@jit(nopython=True)
def sum_1(xs,ys,zs,p):
    return xs**p+ys**p+zs**p

@jit(nopython=True)
def sum_2(xs,ys,zs,p,q):
    return xs**p*ys**q + ys**p*xs**q + ys**p*zs**q + zs**p*ys**q + zs**p*xs**q + xs**p*zs**q
    #res = xs**p*ys**q
    #res += ys**p*xs**q
    #res += ys**p*zs**q
    #res += zs**p*ys**q
    #res += zs**p*xs**q
    #res += xs**p*zs**q
    #return res

@jit(nopython=True)
def sum_3(xs,ys,zs,p,q,r):
    res = xs**p * ys**q * zs**r
    res += xs**p * zs**q * ys**r
    res += ys**p * zs**q * xs**r
    res += ys**p * xs**q * zs**r
    res += zs**p * xs**q * ys**r
    res += zs**p * ys**q * xs**r
    return res

def quadratic_sr(xs,ys,zs,eps=1.,eta=2.,n_s=1):
    nxs,nys,nzs = xs**(1+(1-n_s)/3.),ys**(1+(1-n_s)/3.),zs**(1+(1-n_s)/3.)
    P3 = nxs*nys*nzs
    K2 = (nxs+nys+nzs)**2
    K = nxs+nys+nzs
    K3 = sum_1(nxs,nys,nzs,3)
    S3 = sum_2(nxs,nys,nzs,1,2)
    Q4 = sum_2(nxs,nys,nzs,2,2)/2.
    #eps = 1.
    #eta = 2*eps
    return ((eta-eps)*K3+eps*(S3)+8*eps*Q4/K)/P3

def single_field_sr(xs,ys,zs,H=0.00408,eps=2e-6,eta=4e-6,n_s=1-2*2e-6-4e-6,k_piv=1):
        return quadratic_sr(xs,ys,zs,eps,eta,n_s)*pow(H,4)*pow(eps,-2)/(32.)*(xs*ys*zs/k_piv**3)**((n_s-1)*2./3)

def single_field_sr_func(xs,ys,zs,H,eps,eta):
        k_eval = np.max([xs,ys,zs],axis=0)
        return quadratic_sr(xs,ys,zs,eps(k_eval),eta(k_eval))*pow(H(k_eval),4)*pow(eps(k_eval),-2)/(32.)

def const(xs,ys,zs):
    return np.ones_like(xs)

def local(xs,ys,zs):
    return sum_3(xs,ys,zs,2,-1,-1)/3.

def equil(xs,ys,zs):
    return -(ys+zs-xs)*(zs+xs-ys)*(xs+ys-zs)/(xs*ys*zs)

def flat(xs,ys,zs):
    return equil(xs,ys,zs)+1

def flat_p_gen(p):
    def flat_p(xs,ys,zs):
        return flat(xs,ys,zs)**p
    return flat_p

def orthog(xs,ys,zs):
    return equil(xs,ys,zs)+2./3

def osc(xs,ys,zs,freq,phase=0):
    return np.cos(freq*(xs+ys+zs)+phase)

def log_osc(xs,ys,zs,freq,phase=0):
    return np.cos(freq*np.log(xs+ys+zs)+phase)

def eql_osc(xs,ys,zs,freq,phase):
    return np.cos(freq*(xs+ys+zs)+phase)*equil(xs,ys,zs)

def eql_log_osc(xs,ys,zs,freq,phase):
    return np.cos(freq*np.log(xs+ys+zs)+phase)*equil(xs,ys,zs)

def flt_log_osc(xs,ys,zs,freq,phase):
    '''res = np.exp((1-ys/xs-zs/xs)*20)
    res += np.exp((1-zs/ys-xs/ys)*20)
    res += np.exp((1-xs/zs-ys/zs)*20)
    res *= np.cos(freq*np.log(xs+ys+zs)+phase)
    return res
    '''
    return np.cos(freq*np.log(xs+ys+zs)+phase)*flat(xs,ys,zs)

def flt_osc(xs,ys,zs,freq,phase):
    return np.cos(freq*(xs+ys+zs)+phase)*flat(xs,ys,zs)

#def DBI(xs, ys, zs, H=1.30685e-6, eps=1.45e-3, eta=2.65e-2, c_s=8.98e-2, eps_s=8.6e-3, k_piv=1., scaling='sum', n_NG=None):
#@jit(nopython=True)
def DBI(xs, ys, zs, H=1.3080460512043527e-06, eps=0.00011474192842338179, eta=0.026628810103461423, c_s=0.09062842059127779, eps_s=0.008681165364940952, k_piv=0.05, scaling='no', n_NG=None, correct_ampl=False):
    if scaling=='prod' or scaling=='sum':
        if n_NG is None:
            n_NG = 2*(-2*eps-eps_s-eta)-2*eps_s
    else:
        n_NG = 0.
    if correct_ampl:
        ampl_factor = 1-(-2*eps-eps_s-eta)
    else:
        ampl_factor = 1.
    ## # Eqn 7 of arxiv:1303.5084
    A = sum_1(xs,ys,zs,5)
    B = 2*sum_2(xs,ys,zs,4,1)
    C = -3*sum_2(xs,ys,zs,3,2)
    D = sum_3(xs,ys,zs,3,1,1)
    E = -4*sum_3(xs,ys,zs,2,2,1)
    P = xs*ys*zs
    K2 = (xs+ys+zs)**2
    ## # From (55) in arxiv:1905.05697, c_s>0.08
    ## # Get running from ampl and fnl, ie 2(-2eps-eps_s-eta)-2eps_s
    #fnl = -(35./108.)*(1./c_s**2-1.)*((xs/k_piv)*(ys/k_piv)*(zs/k_piv))**(n_NG/3.)
    ## # arxiv.org/abs/hep-th/0605045
    fnl = (-35./108.)*(1./c_s**2-1.)#*np.ones_like(xs, dtype=np.float64)
    if scaling=="sum":
        fnl = fnl*((xs/(3*k_piv) + ys/(3*k_piv) + zs/(3*k_piv)))**(n_NG)
    elif scaling=='prod':
        fnl = fnl*((xs/k_piv)*(ys/k_piv)*(zs/k_piv))**(n_NG/3.)
    ampl = (H**2)/(4*c_s*eps)#*np.ones_like(xs, dtype=np.float64)
    temp = (3./5)*(6*ampl**2)*fnl*(-3./7)*(A+B+C+D+E)/(P*K2)
    return temp*ampl_factor

def planck_def_1(k1,k2,k3):
    return k1**2*(k2*k3)**-1+k2**2*(k3*k1)**-1+k3**2*(k1*k2)**-1

def planck_def_2(k1,k2,k3):
    return 1.

def planck_def_3(k1,k2,k3):
    res = k1**1*k2**0*k3**-1
    res += k1**1*k3**0*k2**-1
    res += k2**1*k3**0*k1**-1
    res += k2**1*k1**0*k3**-1
    res += k3**1*k1**0*k2**-1
    res += k3**1*k2**0*k1**-1
    return res

def planck_equil(k1,k2,k3):
    return -planck_def_1(k1,k2,k3)-2*planck_def_2(k1,k2,k3)+planck_def_3(k1,k2,k3)

def planck_ortho(k1,k2,k3):
    return -3*planck_def_1(k1,k2,k3)-8*planck_def_2(k1,k2,k3)+3*planck_def_3(k1,k2,k3)

def planck_local(k1,k2,k3):
    return (1./3)*planck_def_1(k1,k2,k3)

def planck_eft1(xs, ys, zs, Ampl=1, fnl=1):
    A = sum_1(xs,ys,zs,6)
    B = 3*sum_2(xs,ys,zs,5,1)
    C = -1*sum_2(xs,ys,zs,4,2)
    D = -3*sum_2(xs,ys,zs,3,3)
    E = 3*sum_3(xs,ys,zs,4,1,1)
    F = -9*sum_3(xs,ys,zs,3,2,1)
    G = -2*sum_3(xs,ys,zs,2,2,2)
    P = xs*ys*zs
    K3 = (xs+ys+zs)**3
    prefac = (-9./17)*6*Ampl**2*(fnl)
    return prefac*(A+B+C+D+E+F+G)/(P*K3)

def planck_eft2(xs, ys, zs, Ampl=1, fnl=1):
    P = xs*ys*zs
    K3 = (xs+ys+zs)**3
    prefac = 27*6.*Ampl**2*(fnl)
    return prefac*P/K3

def plot_shape(filename,shape,k_min,k_max,Nk):
    ks = np.linspace(k_min,k_max,Nk)
    m = np.meshgrid(ks,ks,ks)
    xs = m[0].reshape(Nk**3)
    ys = m[1].reshape(Nk**3)
    zs = m[2].reshape(Nk**3)
    tetra = True
    if tetra:
        triangle_ineq = (xs<=ys+zs)*(ys<=zs+xs)*(zs<=xs+ys)#*(xs+ys+zs<2*xs[-1])
        points_in_tetra = (triangle_ineq).reshape((Nk,Nk,Nk))
    res = shape(xs,ys,zs)
    if tetra:
        xs = xs[triangle_ineq]
        ys = ys[triangle_ineq]
        zs = zs[triangle_ineq]
        res = res[triangle_ineq]
    to_print = np.array([xs,ys,zs,res]).T
    np.savetxt(filename,to_print,delimiter=',',header='"X","Y","Z","f(X,Y,Z)"')

def scale_template(template, n_scalar, k_pivot=0.05):
    # Corrects for non-unit n_scalar values in the given template
    def new_template(k1,k2,k3):
        k1p = k_pivot * pow(k1/k_pivot, (4-n_scalar)/3)
        k2p = k_pivot * pow(k2/k_pivot, (4-n_scalar)/3)
        k3p = k_pivot * pow(k3/k_pivot, (4-n_scalar)/3)
        pref = (k1*k2*k3)**2 / (k1p*k2p*k3p)**2
        return pref * template(k1p, k2p, k3p)
    return new_template
