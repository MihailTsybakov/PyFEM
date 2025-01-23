"""
    Various FEM routines
"""


import numpy as np
from scipy.sparse.linalg import spsolve


def solve_linear(K, f, fixed_dofs):
    """
    Performs a linear sparse system solution for given matrix and RHS

    Parameters:
        K (scipy.sparse.csr_matrix) - sparse stiffness matrix
        f (np.array)                - load vector
        fixed_dofs (np.array)       - List of fixed Degrees of Freedom to exclude from solution

    Returns:
        u (np.array) - displacement vector
    """

    n_dofs    = f.shape[0]
    all_dofs  = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    U = np.zeros(n_dofs)

    # Solving
    U[free_dofs] = spsolve(K[free_dofs, :][:, free_dofs], f[free_dofs])
    
    return U


def solve_dynamic(K, M_lump, fixed_dofs, load_dofs, load_schedule, dt, n_iter):
    """
    Performs a FEM linear dynamic simulation (1st order-accurate central difference)

    Parameters:
        K      (scipy.sparse.csr_matrix) - global stiffness matrix
        M_lump (scipy.sparse.cst_matrix) - global mass matrix (LUMPED)
        fixed_dofs (list) - fixed degrees of freedom
        load_dofs  (list) - list of DOFs under load
        load_schedule (list) - load values in time
        dt    (float) - time step 
        n_iter (float) - number of time steps

        Warning: CFL should be checked separately

    Returns:
        u_hist (list) - displacement history
    """
    n_dofs = K.shape[0]
    all_dofs  = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    u_hist = [np.zeros(n_dofs)] * 2

    for _ in range(n_iter):
        f = np.zeros(n_dofs)
        f[load_dofs] = load_schedule[_]
        
        U = np.zeros(n_dofs)
        U[free_dofs] = (2 * u_hist[-1] - u_hist[-2] + dt**2 * (f - K @ u_hist[-1]) / M_lump)[free_dofs]

        u_hist.append(U.copy())

    return u_hist



def hookean_matrix_3d(E, nu):
    """
    Constructs 6 by 6 linear elastic matrix for given material properties

    Parameters:
        E (float) - Young Modulus
        nu (float) - Poisson ratio

    Returns:
        D (2D np.array) - 6 x 6 elastic matrix to connect stress and strains 
    """

    D = E/((1 + nu)*(1 - 2*nu)) * np.array([
        [1-nu, nu,   nu,   0,          0,                   0],
        [nu,   1-nu, nu,   0,          0,                   0],
        [nu,   nu,   1-nu, 0,          0,                   0],
        [0,    0,    0,    (1-2*nu)/2, 0,                   0],
        [0,    0,    0,    0,          (1-2*nu)/2,          0],
        [0,    0,    0,    0,          0,          (1-2*nu)/2]
    ])

    return D 


def get_DOFs(element):
    """
    Computes all DOFs which belong to given element

    Parameters:
        element (list) - elem's node IDs list (without elem ID)

    Returns:
        dofs (np.array) - elem's dofs list 
    """ 

    n_nodes = len(element)
    dofs    = np.kron(np.ones(n_nodes), np.arange(3)) + np.kron(np.array(element)*3, np.ones(3))

    return dofs