import numpy as np
from scipy.interpolate import interp1d
import trimesh

def predict_m_at_n(n_known, nm, n_new, *, kind='linear', extrapolate=True):
    """
    Interpolation 

    Parameters 
    -------------
    n_known: array_like shape (N,)
    nm:      shape (N, M)  -> rows indexed by n, columns are M features
    n_new:   scalar or array of new n values

    returns: shape (M,) if n_new is scalar, else (len(n_new), M)
    """
    if nm.shape[0] != n_known.shape[0]:
        raise ValueError("nm.shape[0] must equal n_known.shape[0]")

    # sort by n to satisfy interp1d
    order = np.argsort(n_known)
    n_sorted = n_known[order]
    nm_sorted = nm[order]

    fill_value = 'extrapolate' if extrapolate else (nm_sorted[0], nm_sorted[-1])
    f = interp1d(
        n_sorted, nm_sorted,
        kind=kind, axis=0,
        bounds_error=False, fill_value=fill_value
    )
    return f(n_new)

def get_lun_eps(r, R=1.0, eps_min=1.0):
    """
    Luneburg lens dielectric distribution. 
    2 at the lens center -> 1 at the surface.
    
    Parameters
    -------------
    r: (float)
        Radius from center of spherical lens (in mm)
    R: (float)
        Radius of lens (in mm)

    Returns
    ------------- 
    eps: (float)
        Dielectric constant (relative permitivity)
    """
    return 2.0 - (2.0-eps_min)*(r/R)**2.0

def get_vf(eps_emt, eps_mat):
    """
    Compute volume fraction (density) to obtain given dielectric constant.
    Used Maxwell-Garnett theory approximation for composite materials.

    Parameters
    -------------
    eps_mat: (float)
        Dielectric constant of the printed material.
    eps_emt: (float)
        Desired dielectric constant.
    """

    vf = ((eps_mat+2.0)*(eps_emt-1.0))/((eps_mat-1.0)*(eps_emt+2.0))
    vf[vf<0] = 0
    vf[vf>1] = 1.0
    return vf

def get_eps_from_vf(vf, eps_mat):
    """
    Compute volume fraction (density) to obtain given dielectric constant.
    Used Maxwell-Garnett theory approximation for composite materials.

    Parameters
    -------------
    eps_mat: (float)
        Dielectric constant of the printed material.
    eps_emt: (float)
        Desired dielectric constant.
    """

    eps_emt = (eps_mat * (-2*vf-1) +2*vf-2) / (eps_mat * (vf - 1)-vf-2)
  
    return eps_emt


def sphere_mesh(radius=10.0):
    """Make a simple mesh of a sphere 

    Args:
        radius (float, optional): Radius of the sphere. Defaults to 10.0.

    Returns:
        sphere: trimesh mesh object
    """
    subdivisions = 4   # 0–6. Higher = smoother
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

def cylinder_mesh(radius=10.0, height=10.0):
    """Make a cylindrical mesh

    Args:
        radius (float, optional): radius of cylinder. Defaults to 10.0.
        height (float, optional): height of cylinder. Defaults to 10.0.

    Returns:
        cylinder: trimesh object
    """
    
    return trimesh.creation.cylinder(radius=radius, height=height, sections=128)