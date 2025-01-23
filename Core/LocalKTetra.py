"""
    1st order tetrahedron stiffness matrix
"""


from Core.Common import *


def local_K_tetra(nodes, E, nu, rho):
    """
    Local stiffness matrix assembly for 1st order tetrahedron element

    Parameters:
        nodes (2D np.array 4x3) - elem's nodal position vector
        E (float) - Young modulus
        nu (float) - Poisson ratio

    Returns:
        local_K (2D np.array 12 x 12), B (2D np.array 6 x 12) - local derivatives matrix
    """
    
    jacobian = np.array([
        nodes[1] - nodes[0],  nodes[2] - nodes[0],  nodes[3] - nodes[0]
    ])
    
    volume_matrix = np.column_stack([np.ones(4), nodes])
    
    inv_jacobian = np.linalg.inv(jacobian)
    
    a, b, c = inv_jacobian[0, 0], inv_jacobian[0, 1], inv_jacobian[0, 2]
    d, e, f = inv_jacobian[1, 0], inv_jacobian[1, 1], inv_jacobian[1, 2]
    g, h, k = inv_jacobian[2, 0], inv_jacobian[2, 1], inv_jacobian[2, 2]
    
    N1_block_top = np.array([
        [-a-b-c, 0, 0],
        [0, -d-e-f, 0],
        [0, 0, -g-h-k]
    ])
    
    N1_block_bottom = np.array([
        [-d-e-f, -a-b-c, 0],
        [0, -g-h-k, -d-e-f],
        [-g-h-k, 0, -a-b-c],
    ])
    
    N2_block_top = np.array([
        [a, 0, 0],
        [0, d, 0],
        [0, 0, g]
    ])
    
    N2_block_bottom = np.array([
        [d, a, 0],
        [0, g, d],
        [g, 0, a]
    ])
    
    N3_block_top = np.array([
        [b, 0, 0],
        [0, e, 0],
        [0, 0, h]
    ])
    
    N3_block_bottom = np.array([
        [e, b, 0],
        [0, h, e],
        [h, 0, b]
    ])
    
    N4_block_top = np.array([
        [c, 0, 0],
        [0, f, 0],
        [0, 0, k]
    ])
    
    N4_block_bottom = np.array([
        [f, c, 0],
        [0, k, f],
        [k, 0, c]
    ])
    
    B_top    = np.concatenate([N1_block_top, N2_block_top, N3_block_top, N4_block_top], axis = 1)
    B_bottom = np.concatenate([N1_block_bottom, N2_block_bottom, N3_block_bottom, N4_block_bottom], axis = 1)
    
    B = np.concatenate([B_top, B_bottom], axis = 0)

    D = hookean_matrix_3d(E, nu)
    
    tetra_volume = np.abs(np.linalg.det(volume_matrix)) / 6
    
    local_K = B.T @ D @ B * tetra_volume
    
    return local_K


def local_M_tetra(nodes, E, nu, rho):
    """
    Local mass matrix for 1st order tetrahedron element

    Parameters:
        nodes (2D np.array) - nodal position vector
        rho (float)         - density

    Returns:
        M (2D np.array) - local mass matrix
    """

    volume_matrix = np.column_stack([np.ones(4), nodes])

    volume = np.abs(np.linalg.det(volume_matrix)) / 6

    mass = volume*rho

    M = mass/24 * np.array([
        [2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2]
    ])

    return M


def local_M_tetra_lumped(nodes, E, nu, rho):
    """
    Local LUMPED mass matrix for 1st order tetrahedron element

    Parameters:
        nodes (2D np.array) - nodal position vector
        rho (float)         - density

    Returns:
        M (2D np.array) - local mass matrix
    """

    M_no_lump = local_M_tetra(nodes, E, nu, rho)

    return np.diag(np.sum(M_no_lump, axis = 1))