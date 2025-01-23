"""
    Global FEM Stiffness matrix assembly
"""


from Core.LocalKTetra import *

from scipy import sparse


def assemble_K(nodes, elements, mat_props, callback = local_K_tetra):
    """
    Global stiffness matrix assembly

    Parameters:
        nodes (2D np.array N x 4) - nodal position vector
        elements (list[list]) - element data
        mat_props (list[tuple]) - material properties for each element
        callback (optional) - which assembly function to call

    Returns:
        K (scipy.sparse.csr_matrix) - global stiffness matrix
    """

    n_dofs = nodes.shape[0] * 3
    K_loc_sizes = np.array([3*(len(elem) - 1) for elem in elements])
    K_loc_entries = K_loc_sizes**2

    ik = np.zeros(np.sum(K_loc_entries))
    jk = np.zeros(np.sum(K_loc_entries))
    ak = np.zeros(np.sum(K_loc_entries))

    for ei, elem in enumerate(elements):
        elem_dofs = get_DOFs(elem[1:])
        elem_nodes = nodes[elem[1:]]

        ik_loc = np.kron(elem_dofs, np.ones(K_loc_sizes[ei]))
        jk_loc = np.reshape(np.kron(elem_dofs, np.reshape(np.ones(12), (12, 1))), -1)

        K_loc = callback(elem_nodes[:, 1:], mat_props[ei][0], mat_props[ei][1], mat_props[ei][2])

        ak_loc = np.reshape(K_loc, -1)

        start_id = np.sum(K_loc_entries[:ei])
        end_id   = np.sum(K_loc_entries[:ei+1])
        
        ik[start_id : end_id] = ik_loc 
        jk[start_id : end_id] = jk_loc 
        ak[start_id : end_id] = ak_loc 

    K = sparse.coo_matrix((ak, (ik, jk)), shape = (n_dofs, n_dofs)).tocsr()

    return K