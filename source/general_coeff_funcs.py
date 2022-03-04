"""
The functions below are mainly for use when you have a set of Primodal coefficients already, and you want
to do things with them---e.g. plot the bispectrum, convert to a different basis set, multiply the expansion
by some k-dependent prefactor. Also below is code for setting up basis sets.

To reduce the need to evaluate the basis functions at the same sample
points repeatedly, we use pseudo-Vandermonde matrices a lot, see for example
`this Numpy function <https://www.numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legvander.html>`_.

Example use: to check for convergence of a Primodal expansion, use the appropriate "set_up_X_basis"
to set up a smaller basis, use convert_between_bases to reduce the coefficients to that smaller basis,
then use err_between_coeffs to get the fractional difference between the two sets of coefficients.
"""

## # TODO: add optimize=True to all the einsums (though this is only in the most recent numpy version).
## # Add docstring to all functions
## # Remove unnecessary functions
## # Include example coefficients for tests (one equil, one local)

import numpy as np
import math
import functools
import time
import matplotlib.pyplot as plt

try:
    from gl_c_wrapper import leggauss_c as leggauss
    ## # Python doesn't catch the failed import until the function is called.
    leggauss(3)
except:
    print('Using numpy leggauss, less accurate than QUADPTS')
    from numpy.polynomial.legendre import leggauss
from numpy.polynomial.legendre import legvander
from scipy.special import eval_legendre
#from scipy.interpolate import splrep, splev
#import copy
#try:
#    import tensorly as tl
#    from tensorly.decomposition import parafac
#    tl.__version__
#    tensorly_imported = True
#except:
#    tensorly_imported = False

LG_INV_FUNC_RES = 600
LG_LOW_RES = 75
LG_DECOMP = 50
Nk_aug = 2500
N_SCALAR = 0.9649
N_S_MOD = N_SCALAR - 1

def load_coeffs(filename):
    '''Loads in coefficients file, returns kmin, kmax and the coefficients.'''
    with open(filename) as f:
        first_line = f.readline().strip().split('#')[1]
    k_min, k_max = np.array(first_line.split(',')[0:2], dtype=np.float64)
    coeffs = np.loadtxt(filename, delimiter=',', skiprows=1)
    L3 = np.shape(coeffs)[0]
    L = int(round(L3**(1./3)))
    assert L**3==L3
    coeffs = coeffs.reshape((L, L, L, 4))
    coeffs = coeffs[:, :, :, 3]
    return k_min, k_max, coeffs

def coeff_print(k_min, k_max, coeffs, basis_funcs, f_name, basis_label='unlabelled'):
    '''Print coefficients to file.'''
    data_file = open(f_name,'w')
    l = len(coeffs[0,0,:])
    data_file.write( '"i","j","k","a_ijk"#'+str(k_min)+','+str(k_max)+','+basis_label+'\n' )
    for i in range(l):
        for j in range(l):
            for k in range(l):
                data_file.write( str(i)+','+str(j)+','+str(k)+','+str(coeffs[i,j,k])+'\n' )
            data_file.write( '\n' )
        data_file.write( '\n' )
    data_file.close()

def get_coeffs(func_to_fit,basis_funcs,k_min,k_max, Nk=LG_DECOMP, cube=False):
    '''Decomposes given function in terms of orthonormal basis functions.
    Same logic as basis_vander, but in 3d.'''
    decomp_xs,decomp_ws = leggauss(Nk)
    vander = np.zeros((len(basis_funcs), len(decomp_xs)))
    for j, b_func in enumerate(basis_funcs):
        vander[j, :] = b_func(decomp_xs)*decomp_ws[:]
    Nk = len(decomp_xs)
    f_evals = np.zeros((Nk,Nk,Nk))
    u_decomp_xs = unbar(decomp_xs ,k_min, k_max)
    for i, ux in enumerate(u_decomp_xs):
        for j, uy in enumerate(u_decomp_xs):
            uzs = np.copy(u_decomp_xs)
            f_evals[i,j,:] = func_to_fit(ux,uy,uzs)
    coeffs = np.einsum('abc,ia->bci', f_evals, vander, optimize=True)
    coeffs = np.einsum('bci,jb->cij', coeffs, vander, optimize=True)
    coeffs = np.einsum('cij,kc->ijk', coeffs, vander, optimize=True)
    return coeffs

def basis_vander(basis_funcs,k_min,k_max,decomp_xs,decomp_ws):
    '''Pseudo-Vandermonde matrix, used to decompose functions
    in the given basis without having to repeatedly evaluate them.
    The output is used to map function evaluations to coefficients.
    Similar to numpy.polynomial.legendre.legvander,
    but including integration weights.'''
    norms = np.zeros(len(basis_funcs))
    for i,b in enumerate(basis_funcs):
        bs = b(decomp_xs)
        norms[i] = np.dot(bs**2,decomp_ws)
    vander = np.zeros((len(basis_funcs),len(decomp_xs)))
    for j,b_func in enumerate(basis_funcs):
        vander[j,:] = b_func(decomp_xs)*decomp_ws[:]/norms[j]
    return vander

def eval_vander(basis_funcs,k_min,k_max,decomp_xs):
    '''Pseudo-Vandermonde matrix, used to evaluate functions
    that have been decomposed in the given basis.
    The output is used to map coefficients to function evaluations.
    Similar to numpy.polynomial.legendre.legvander.'''
    vander = np.zeros((len(basis_funcs),len(decomp_xs)))
    for j,b_func in enumerate(basis_funcs):
        vander[j,:] = b_func(decomp_xs)
    return vander

def fractional_difference(k_min,k_max,template,coeffs,basis_funcs,Nk=LG_LOW_RES,ampl=False, cube=False):
    '''Return the fractional difference between the expansion and the template.
    By default, ignores errors in the overall scale, for ease of checking against
    scale-invariant templates.'''
    decomp_xs,decomp_ws = leggauss(Nk)
    vander = np.zeros((len(basis_funcs),len(decomp_xs)))
    for i,x in enumerate(decomp_xs):
        for j,b_func in enumerate(basis_funcs):
            vander[j,i] = b_func(x)
    Nk = len(decomp_xs)
    f_evals = np.zeros((Nk,Nk,Nk))
    weights = np.zeros((Nk,Nk,Nk))
    for i,x in enumerate(decomp_xs):
        for j,y in enumerate(decomp_xs):
            zs = np.copy(decomp_xs)
            ux,uy = unbar(np.array([x,y]),k_min,k_max)
            uzs   = unbar(zs,k_min,k_max)
            in_tetra = 1.
            if cube:
                in_tetra = (ux<=uy+uzs)*(uy<=uzs+ux)*(uzs<=ux+uy)
            f_evals[i,j,:] = template(ux,uy,uzs)*in_tetra
            weights[i,j,:] = decomp_ws[i]*decomp_ws[j]*decomp_ws*in_tetra

    basis_evals = np.einsum('ji,jqr->iqr',vander,coeffs, optimize=True)
    basis_evals = np.einsum('ji,qjr->qir',vander,basis_evals, optimize=True)
    basis_evals = np.einsum('ji,qrj->qri',vander,basis_evals, optimize=True)

    fg = np.einsum('ijk,ijk->',basis_evals,f_evals*weights)
    ff = np.einsum('ijk,ijk->',f_evals,f_evals*weights)
    gg = np.einsum('ijk,ijk->',basis_evals,basis_evals*weights)

    if ampl:
        return np.sqrt(np.abs(1+(gg-2*fg)/ff))
    else:
        ## # Ignores error in overall scale.
        return np.sqrt(2-2*fg/np.sqrt(ff*gg))
   
def err_between_coeffs(k_min, k_max, cs1, basis1, cs2, basis2, Ncorr=200, cube=False):
    '''Fractional error between two sets of coefficients.
    The parameter "cube" determines whether the difference is
    evaluated on the tetrapyd only, or on the whole cube.'''
    xs, ws = leggauss(Ncorr)
    ks = unbar(xs, k_min, k_max)

    _,_,_,S1_evals = eval_on_grid(k_min, k_max, cs1, basis1, ks, True, cube)
    S1_evals = S1_evals.reshape((Ncorr,Ncorr,Ncorr))
    sqrt_mean_S1_2 = np.sqrt(np.einsum('ijk,i,j,k->',(S1_evals)**2,ws,ws,ws, optimize=True))

    _,_,_,S2_evals = eval_on_grid(k_min, k_max, cs2, basis2, ks, True, cube)
    S2_evals = S2_evals.reshape((Ncorr,Ncorr,Ncorr))
    sqrt_mean_S2_2 = np.sqrt(np.einsum('ijk,i,j,k->',(S2_evals)**2,ws,ws,ws, optimize=True))

    sqrt_mean_diff2 = np.sqrt(np.einsum('ijk,i,j,k->',(S1_evals-S2_evals)**2,ws,ws,ws, optimize=True))
    return sqrt_mean_diff2/sqrt_mean_S1_2

def various_lines_plot(k_min, k_max, coeffs, basis_funcs, Nk=10000, title='Lines through the tetrapyd', ax=None, style='-'):
    '''Plots the folded limit, equilateral limit and two values of the squeezed limit.
    Easier to spot errors here than on a full tetrapyd plot, but on this plot errors
    can seem more important that they actually are in the full 3d cube.'''
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ks = np.linspace(k_min, k_max, Nk, endpoint=True)
    sqznesses = [2., 1., 0.01, 2*k_min/k_max]
    for sqz in sqznesses:
        sqz_points = np.array([[k, k, sqz*k] for k in ks[(ks>k_min/sqz)*(ks<k_max/sqz)]]).T
        sqz_decomp = tidied_eval(k_min, k_max, coeffs, basis_funcs, sqz_points)
        xs, ys, zs = sqz_points
        kS = np.sqrt(0.5*(xs**2+ys**2)-0.25*zs**2)
        ax.plot(xs+ys+zs, sqz_decomp, style, label=r'$k_3/k_1='+np.format_float_scientific(sqz, precision=1)+'$')
    ax.set_title(title)
    ax.set_xlabel(r'$k_1+k_2+k_3$')
    ax.set_ylabel(r'$S(k_1,k_2,k_3)$')
    ax.legend()

def print_and_plot_tetra(k_min, k_max, cs, basis, fname, Nk=100, title='', override_colour=None):
    '''Plot the 3d tetrapyd'''
    print_tetra(fname, k_min, k_max, cs, basis, Nk)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize
    import pandas as pd
    import matplotlib.pyplot as plt
    tetra_data = pd.read_csv(fname)
    fig = plt.figure(dpi=200)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    cmap = plt.cm.rainbow
    ## Cutting the colormap to exclude outliers, from numerical innaccuracies at the edges.
    print('min, max =', tetra_data['f(X,Y,Z)'].min(), tetra_data['f(X,Y,Z)'].max(), sep=',')
    vmin = tetra_data['f(X,Y,Z)'].quantile(q=0.001)
    vmax = tetra_data['f(X,Y,Z)'].quantile(q=0.999)
    if override_colour is not None:
        vmin, vmax = override_colour
    print('Color map limits:', vmin, vmax, sep=',')
    print('(0.001 and 0.999 quantile limits.)')
    norm = Normalize(vmin=vmin, vmax=vmax)
    colours = cmap(norm(tetra_data['f(X,Y,Z)']))
    ## # Set transparency
    colours[:, 3] = np.minimum(np.abs(tetra_data['f(X,Y,Z)'])/np.abs(vmax), 1.)**4
    #print(colours[:, 3].min(), colours[:, 3].max())

    ax1.scatter(tetra_data['X'], tetra_data['Y'], tetra_data['Z'], c=colours, s=0.2)
    ax1.set_xlabel(r'$k_1$')
    ax1.set_xlabel(r'$k_2$')
    ax1.set_xlabel(r'$k_3$')
    #sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #cbar1 = fig.colorbar(sm1)
    #cbar1.set_label(r'$S(k_1,k_2,k_3)$')

    ax2.view_init(elev=90.-54.74, azim=45.)
    ax2.scatter(tetra_data['X'], tetra_data['Y'], tetra_data['Z'], c=colours, s=0.2)
    ax2.set_xlabel(r'$k_1$')
    ax2.set_xlabel(r'$k_2$')
    ax2.set_xlabel(r'$k_3$')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm)
    cbar.set_label(r'$S(k_1,k_2,k_3)$')

    ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(fname[:-4]+'_fig.png', dpi=200)
    plt.show()

def print_tetra(filename,k_min,k_max,coeffs,basis_funcs,Nk,ret=False):
    '''Evaluates the function, saves values to a file.'''
    points = np.linspace(k_min,k_max,Nk)
    xs,ys,zs,res = eval_on_grid(k_min,k_max,coeffs,basis_funcs,points)
    to_print = np.array([xs,ys,zs,res]).T
    np.savetxt(filename,to_print,delimiter=',',header='"X","Y","Z","f(X,Y,Z)"',comments='')
    if ret:
        return xs,ys,zs,res

def print_slice(filename,k_min,k_max,coeffs,basis_funcs,Nk):
    '''Extracts slice plot.'''
    points = np.linspace(k_min,k_max,Nk)
    xs,ys,zs,res = eval_on_grid(k_min,k_max,coeffs,basis_funcs,points)
    slice_indices = np.where(xs==ys)
    to_print = np.array([xs[slice_indices],ys[slice_indices],zs[slice_indices],res[slice_indices]]).T
    np.savetxt(filename,to_print,delimiter=',',header='"X","Y","Z","f(X,Y,Z)"',comments='')

def check_orthog(basis_funcs,k_min,k_max,Nk=Nk_aug-100, normalise=False):
    '''Check to see if the basis has been successfully orthogonalised.'''
    test_xs,test_ws = leggauss(Nk)
    vander = np.array([b(test_xs) for b in basis_funcs])
    p_max = len(basis_funcs)
    res = np.zeros((p_max,p_max))
    for i in range(p_max):
        for j in range(p_max):
            prod = np.sum(test_ws*vander[i,:]*vander[j,:])
            res[i,j] = prod
    if normalise:
        norm = [np.sqrt(res[i,i]) for i in range(p_max)]
        for i in range(p_max):
            for j in range(p_max):
                res[i,j] = res[i,j] / (norm[i] * norm[j])
    return res

def set_up_fourier_basis(k_min, k_max, Nb, Nk=Nk_aug):
    '''Set up a Fourier basis'''
    Nb_set = Nb+Nb%2-1
    basis_funcs = [zero_func]*Nb
    normed_basis_funcs = [zero_func]*Nb
    norms = np.zeros(Nb)
    xs, ws = leggauss(Nk)
    basis_funcs[0] = lambda x: np.ones_like(x)
    for r in range(1, (Nb_set-1)//2+1):
        basis_funcs[2*r-1] = lambda x, r=r: np.sin(np.pi*r*x)
        basis_funcs[2*r]   = lambda x, r=r: np.cos(np.pi*r*x)
    if Nb%2==0:
        r = (Nb_set-1)//2+1
        basis_funcs[2*r-1] = lambda x, r=r: np.sin(np.pi*r*x)
    for i in range(Nb):
        fs = basis_funcs[i](xs)
        norms[i] = np.sqrt(np.dot(fs**2, ws))
    for i in range(len(basis_funcs)):
        normed_basis_funcs[i] = lambda x, i=i: basis_funcs[i](x)/norms[i]
    return normed_basis_funcs

def set_up_flat_basis(k_min,k_max,Nb,Nk=Nk_aug,normalise=True):
    '''Set up a Legendre basis'''
    basis_funcs = [zero_func]*Nb
    norms = np.zeros(Nb)

    xs,ws = leggauss(Nk)
    for i in range(Nb):
        fs = eval_legendre(i,xs)
        norms[i] = np.sqrt(np.dot(fs**2,ws))

    if normalise:
        for i in range(len(basis_funcs)):
            basis_funcs[i] = lambda x,i=i:eval_legendre(i,x)/norms[i]
    else:
        for i in range(len(basis_funcs)):
            basis_funcs[i] = lambda x,i=i:eval_legendre(i,x)

    return basis_funcs

def set_up_mixed_basis(k_min, k_max, Nb, n_s_mod=N_S_MOD, verbose=True, quad=False):
    '''Set up a Legendre basis, augmented with negative power term(s)'''
    def inv_func(x):
        return x**-(1-n_s_mod)
    def inv_func2(x):
        return x**-(2-n_s_mod)
    aug_funcs = [inv_func]
    if quad:
        aug_funcs.append(inv_func2)
    base_basis = set_up_flat_basis(k_min,k_max,Nb-len(aug_funcs))
    basis_funcs = np.copy(base_basis)
    for i,af in enumerate(aug_funcs):
        basis_funcs = augment_basis_arb_func(k_min,k_max,basis_funcs,af,Nk=Nk_aug)
        if verbose:
            print('# Aug func',i,':',af.__name__)
            print('# Aug func ln(f(2))/ln(2)', i, ':', np.log(af(2.))/np.log(2.))
    if verbose:
        print('# n_s_mod:',n_s_mod)
        print('# Ortho check:',Nb,'=', np.sum(check_orthog(basis_funcs,k_min,k_max)), flush=True)
    return basis_funcs

def set_up_log_basis(k_min, k_max, Nb, inv=True, verbose=True):
    '''Set up a Legendre basis, augmented with log term(s)'''
    if not inv:
        aug_funcs = [lambda x: np.log(x)]
    else:
        aug_funcs = [lambda x: 1./x, lambda x: np.log(x)/x]
    base_basis = set_up_flat_basis(k_min,k_max,Nb-len(aug_funcs))
    basis_funcs = np.copy(base_basis)
    for i,af in enumerate(aug_funcs):
        basis_funcs = augment_basis_arb_func(k_min,k_max,basis_funcs,af,Nk=Nk_aug)
    if verbose:
        print('# inv:', inv)
        print('# Ortho check:',Nb,'=', np.sum(check_orthog(basis_funcs,k_min,k_max)), flush=True)
    return basis_funcs

def set_up_f1_basis(k_min, k_max, Nb, verbose=True, no_lin=False, no_inv=False):
    '''Set up a Fourier basis, augmented with monomial term(s)'''
    aug_funcs = [lambda x: x**2]
    if not no_lin:
        aug_funcs = [lambda x: x]+aug_funcs
    if not no_inv:
        aug_funcs = aug_funcs+[lambda x: x**-1]
    base_basis = set_up_fourier_basis(k_min,k_max,Nb-len(aug_funcs))
    basis_funcs = np.copy(base_basis)
    for i,af in enumerate(aug_funcs):
        basis_funcs = augment_basis_arb_func(k_min,k_max,basis_funcs,af,Nk=Nk_aug)
    if verbose:
        print('# Ortho check:',Nb,'=', np.sum(check_orthog(basis_funcs,k_min,k_max)), flush=True)
    return basis_funcs

def set_up_log_inv_basis(k_min, k_max, Nb, verbose=True):
    '''Set up a Legendre basis, with log-mapped parameter,
    divided by sqrt() to maintain orthogonality.'''
    base_basis = set_up_flat_basis(k_min, k_max, Nb)
    def x_normed(x):
        ubx = unbar(x, k_min, k_max)
        return np.log(ubx**2/(k_min*k_max))/np.log(k_max/k_min)

    xs, ws = leggauss(Nk_aug)
    norms = np.ones(len(base_basis))
    for i in range(Nb):
        fs = base_basis[i](x_normed(xs))/np.sqrt(unbar(xs, k_min, k_max))
        norms[i] = np.sqrt(np.dot(fs**2, ws))

    basis_funcs = [lambda x, i=i: base_basis[i](x_normed(x))/np.sqrt(unbar(x, k_min, k_max))/norms[i] for i in range(len(base_basis))]
    if verbose:
        print('# Ortho check:', Nb, '=', np.sum(check_orthog(basis_funcs, k_min, k_max)), flush=True)
    return basis_funcs

def set_up_reso_scaling_basis(k_min, k_max, Nb, verbose=True):
    '''Set up reso basis, with a log term to include scaling.'''
    aug_funcs = [lambda x: np.log(x)]
    base_basis = set_up_log_inv_basis(k_min,k_max,Nb-len(aug_funcs), verbose=False)
    basis_funcs = np.copy(base_basis)
    for i,af in enumerate(aug_funcs):
        basis_funcs = augment_basis_arb_func(k_min,k_max,basis_funcs,af,Nk=Nk_aug)
    if verbose:
        print('# Ortho check:',Nb,'=', np.sum(check_orthog(basis_funcs,k_min,k_max)), flush=True)
    return basis_funcs

def convert_between_bases(old_basis, new_basis, k_min, k_max, coeffs, Nk=LG_INV_FUNC_RES):
    '''Convert a set of coeffs from one basis to another,
    for example to reduce the size of a given basis.'''
    Q = get_conversion_matrix(old_basis, new_basis, k_min, k_max, Nk=Nk)
    coeffs = np.copy(coeffs)
    coeffs = np.einsum('ij,jqr->iqr', Q, coeffs, optimize=True)
    coeffs = np.einsum('ij,qjr->qir', Q, coeffs, optimize=True)
    coeffs = np.einsum('ij,qrj->qri', Q, coeffs, optimize=True)
    return coeffs

def get_coeffs_1d(func_to_fit, basis_funcs, k_min, k_max, Nk=LG_INV_FUNC_RES):
    '''Decompose a 1d function in the given basis.
    A good example of how to use the output of basis_vander.'''
    decomp_xs, decomp_ws = leggauss(Nk)
    vander = np.zeros((len(basis_funcs), len(decomp_xs)))
    for j, b_func in enumerate(basis_funcs):
        vander[j, :] = b_func(decomp_xs)*decomp_ws
    f_evals = func_to_fit(unbar(decomp_xs, k_min, k_max))
    coeffs = np.einsum('ij,j->i', vander, f_evals, optimize=True)
    return coeffs

################################################################
## # The functions below are more for internal use.
################################################################

def legmulx_no_trim(c):
    '''Internal use. Taken from numpy legendre module, but without trimming zeros.'''
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]*0
    prd[1] = c[0]
    for i in range(1, len(c)):
        j = i + 1
        k = i - 1
        s = i + j
        prd[j] = (c[i]*j)/s
        prd[k] += (c[i]*i)/s
    return prd

def bar(x, k_min, k_max):
    '''Internal use. Map to [-1,1]'''
    return (2*x-(k_min+k_max))/(k_max-k_min)

def unbar(x, k_min, k_max):
    '''Internal use. Map to [kmin,kmax]'''
    return 0.5*((k_max-k_min)*x+k_min+k_max)

def mixed_basis_eval_3d(k_min, k_max, xs, ys, zs, coeffs, basis_funcs):
    '''Internal use. Very innefficient. If you are doing anything other
    than evaluating the bispectrum at a single point,
    use eval_on_grid instead.'''
    xs,ys,zs = bar(np.array([xs,ys,zs]),k_min,k_max)
    res = np.zeros_like(xs)
    f1s = np.array([ f(xs) for f in basis_funcs ])
    f2s = np.array([ f(ys) for f in basis_funcs ])
    f3s = np.array([ f(zs) for f in basis_funcs ])
    res = np.einsum( 'ijk,iq,jq,kq->q', coeffs,f1s,f2s,f3s)
    return res

def zero_func(x):
    '''Internal use. A "basis function" that is just all zeros.
    Only ever used as a (bad) placeholder.'''
    return np.zeros_like(x)

def in_tetra(p, lower_lim=0):
    '''Internal use. Check if a point is in the tetrapyd.'''
    x,y,z = p
    return ( (x+y>=z) and (y+z>=x) and (z+x>=y) and (x+y+z>lower_lim) )

def load_coeffs_scan_format(filename):
    '''Internal use. Load coefficients from the scan format used with Wuhyun Sohn.'''
    with open(filename) as f:
        first_line = f.readline().strip().split('#')[1]

    k_min, k_max = np.array(first_line.split('= ')[1].split(','), dtype=np.float64)

    coeffs = np.loadtxt(filename, delimiter=',', skiprows=2)
    L3 = np.shape(coeffs)[0]
    L = int(round(L3**(1./3)))
    assert L**3==L3
    coeffs = coeffs.reshape((L, L, L, 4))
    coeffs = coeffs[:, :, :, 3]
    return k_min, k_max, coeffs

def get_coeffs_1d_from_samples(fs,vander):
    '''Internal use. Example of how to use the output of basis_vander
    to get expansion coefficients.'''
    coeffs = np.dot(vander,fs)
    return coeffs

def eval_on_grid(k_min, k_max, coeffs, basis_funcs, points, zero_non_tetra=False, cube=False):
    '''Internal use. Efficiently evaluates expansion on grid.'''
    bpoints = bar(points,k_min,k_max)
    vander = np.zeros((len(basis_funcs),len(points)))
    for j,b_func in enumerate(basis_funcs):
        vander[j,:] = b_func(bpoints)
    
    basis_evals = np.einsum('ji,jqr->iqr',vander,coeffs)
    basis_evals = np.einsum('ji,qjr->qir',vander,basis_evals)
    basis_evals = np.einsum('ji,qrj->qri',vander,basis_evals)
    
    Np = len(points)
    m = np.meshgrid(points,points,points)
    xs = m[0].reshape(Np**3)
    ys = m[1].reshape(Np**3)
    zs = m[2].reshape(Np**3)
    triangle_ineq = (xs<=ys+zs)*(ys<=zs+xs)*(zs<=xs+ys)#*(xs+ys+zs<2*xs[-1])
    points_in_tetra = (triangle_ineq).reshape((Np,Np,Np))
    if not cube:
        basis_evals *= points_in_tetra
    if not zero_non_tetra:
        xs = xs[triangle_ineq]
        ys = ys[triangle_ineq]
        zs = zs[triangle_ineq]
        basis_evals = basis_evals[points_in_tetra]
    return xs,ys,zs,basis_evals

def get_conversion_matrix(old_basis, new_basis, k_min, k_max, Nk=LG_INV_FUNC_RES):
    '''Internal use. Get conversion matrix between two basis sets.'''
    Q = np.zeros((len(new_basis), len(old_basis)))
    for i, ob in enumerate(old_basis):
        Q[:, i] = get_coeffs_1d(ob, new_basis, -1, 1, Nk=Nk)
    return Q
    
def coeffs_mulx_mixed(coeffs,k_min,k_max,vs,pad=False):
    '''Internal use, and deprecated, use gen_mulx_matrix instead.'''
    xcoeffs = np.zeros(len(coeffs)+1*pad)
    xcoeffs[0] = 0.
    a,b = (k_min+k_max)*0.5,(k_max-k_min)*0.5
    xcoeffs[1:len(vs)+1]  = a*(coeffs[1:]-coeffs[0]*vs)
    xcoeffs[1:]         += b*(legmulx_no_trim(coeffs[1:]-coeffs[0]*vs))[:len(vs)+1*pad]
    xcoeffs[1] += coeffs[0]
    return xcoeffs

def coeffs_mulx_flat(coeffs,k_min,k_max,vs,pad=False):
    '''Internal use, and deprecated, use gen_mulx_matrix instead.'''
    xcoeffs = np.zeros(len(coeffs))
    a,b = (k_min+k_max)*0.5,(k_max-k_min)*0.5
    xcoeffs = a*coeffs
    xcoeffs += b*(legmulx_no_trim(coeffs))[:len(xcoeffs)]
    return xcoeffs

def gen_mulx_matrix(basis_funcs,k_min,k_max,res_pad=False):
    '''Internal use. Generates a matrix to multiply a series by k.'''
    Nb = len(basis_funcs)
    if Nb>70:
        print("Warning: low sampling rate")
        return
    mul_matrix = np.zeros((Nb,Nb))
    for i in range(Nb):
        mul_matrix[:,i] = get_coeffs_1d(lambda k:basis_funcs[i](bar(k,k_min,k_max))*k,basis_funcs,k_min,k_max)
    return mul_matrix[:Nb+1*res_pad,:Nb+1*res_pad]

def gen_divx_matrix(basis_funcs,k_min,k_max,res_pad=False):
    '''Internal use. Generates a matrix to divide a series by k.'''
    Nb = len(basis_funcs)
    if Nb>70:
        print("Warning: low sampling rate")
        return
    div_matrix = np.zeros((Nb,Nb))
    for i in range(Nb):
        div_matrix[:,i] = get_coeffs_1d(lambda k:basis_funcs[i](bar(k,k_min,k_max))/k,basis_funcs,k_min,k_max)
    return div_matrix[:Nb+1*res_pad,:Nb+1*res_pad]

def gen_series_product_matrix(in_basisA, in_basisB, out_basis, k_min, k_max):
    '''Internal use. Expand products of functions in two basis sets in out_basis.'''
    decomp_xs,decomp_ws = leggauss(LG_INV_FUNC_RES)
    vander = np.zeros((len(out_basis), len(decomp_xs)))
    for j,b_func in enumerate(out_basis):
        vander[j,:] = b_func(decomp_xs)*decomp_ws[:]
    integrand = np.zeros((len(in_basisA), len(in_basisB), len(decomp_xs)))
    for j, b_func1 in enumerate(in_basisA):
        for k, b_func2 in enumerate(in_basisB):
            #for i,x in enumerate(decomp_xs):
            integrand[j, k, :] = b_func1(decomp_xs)*b_func2(decomp_xs)
    product_matrix = np.einsum('pq,ijq->ijp', vander, integrand)
    return product_matrix

def coeffs_algebra_3d(p,q,r,coeffs,basis_funcs,k_min,k_max,pad_len=0,mul_m=None,div_m=None):
    '''Internal use. Efficiently multiplies/divides series by prefactors.'''
    l = len(coeffs[0,0,:])
    temp_coeffs = np.zeros((l+pad_len,l+pad_len,l+pad_len))
    temp_coeffs[:l,:l,:l] = np.copy(coeffs)
    Nb = len(temp_coeffs[0,0,:])
    if p>0 or q>0 or r>0:
        if mul_m is None:
            mul_matrix = gen_mulx_matrix(basis_funcs,k_min,k_max)
        else:
            mul_matrix = mul_m
    if p<0 or q<0 or r<0:
        if div_m is None:
            div_matrix = gen_divx_matrix(basis_funcs,k_min,k_max)
        else:
            div_matrix = div_m
    if p>=0:
        for m in range(p):
            temp_coeffs = np.einsum('pi,ijk->pjk',mul_matrix,temp_coeffs)
    else:
        for m in range(-p):
            temp_coeffs = np.einsum('pi,ijk->pjk',div_matrix,temp_coeffs)
    if q>=0:
        for m in range(q):
            temp_coeffs = np.einsum('pj,ijk->ipk',mul_matrix,temp_coeffs)
    else:
        for m in range(-q):
            temp_coeffs = np.einsum('pj,ijk->ipk',div_matrix,temp_coeffs)
    if r>=0:
        for m in range(r):
            temp_coeffs = np.einsum('pk,ijk->ijp',mul_matrix,temp_coeffs)
    else:
        for m in range(-r):
            temp_coeffs = np.einsum('pk,ijk->ijp',div_matrix,temp_coeffs)
    return temp_coeffs

def tidied_eval(k_min,k_max,coeffs,basis_funcs,points):
    '''Internal use. This is usually going to be very innefficient,
    see eval_on_grid instead.''' 
    l = len(points[0])
    res = np.zeros(l)
    if l==0:
        return res
    xs = points[0,:]
    ys = points[1,:]
    zs = points[2,:]
    if l<=10**4:
        res = mixed_basis_eval_3d(k_min,k_max,xs,ys,zs,coeffs,basis_funcs)
    else:
        temp_a = mixed_basis_eval_3d(k_min,k_max,xs[:10**4],ys[:10**4],zs[:10**4],coeffs,basis_funcs)
        temp_b = mixed_basis_eval_3d(k_min,k_max,xs[10**4:],ys[10**4:],zs[10**4:],coeffs,basis_funcs)
        res = np.concatenate([temp_a,temp_b])
    return res

def augment_basis_arb_func(k_min,k_max,basis,func,Nk=Nk_aug):
    '''Internal use. Orthogonalises a function and adds it to
    an existing basis.'''
    # Assumes that the basis is normalised
    klim = [k_min,k_max]
    xs,ws = leggauss(Nk)
    #norms = np.zeros(len(basis))
    #for i,b in enumerate(basis):
    #    fs = b(xs)
    #    norms[i] = np.sqrt(np.dot(fs**2,ws))
    uxs = unbar(xs, k_min, k_max)
    f_vals = func(uxs)
    vander = np.array([b(xs) for b in basis])
    cs = np.zeros_like(basis, dtype=np.float64)
    cs[0] = np.dot(ws*vander[0],f_vals)
    f_hat = np.copy(f_vals)
    ## # Modified GS
    for i in range(1,len(basis)):
        f_hat = f_hat - cs[i-1]*vander[i-1]
        cs[i] = np.dot(ws*vander[i],f_hat)
    new_basis = [None]*(len(basis)+1)
    new_basis[1:] = basis
    f_orth_vals = f_vals-np.sum(cs*vander.T,axis=1)
    norm = np.sqrt(np.sum(ws*f_orth_vals**2))
    if norm/np.sqrt(np.sum(ws*f_vals**2))<1e-6:
        print(func.__name__, 'norm is suspiciously small:',norm)
    def f_orth(x):
        ux = 0.5*(x*(klim[1]-klim[0])+(klim[1]+klim[0]))
        vander = np.array([b(x) for b in basis])
        return (func(ux)-np.sum((cs*vander.T).T,axis=0))/norm
    new_basis[0] = f_orth
    return new_basis

