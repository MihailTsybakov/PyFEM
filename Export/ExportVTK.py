"""
    VTK field export
"""


import numpy as np 
import pyvista as pv


cell_types = {
    4 : pv.CellType.TETRA,
    5 : pv.CellType.PYRAMID,
    6 : pv.CellType.WEDGE,
    8 : pv.CellType.HEXAHEDRON
}


def export_VTK(nodes, elems, vtk_path, node_scalars = None, elem_scalars = None):
    """
    Exports given mesh with given grid scalars to VTK

    Parameters:
        nodes   (2D np.array) - nodal position vector (no IDs)
        elems   (list[list])  - elems data list 
        vtk_path (str)
        node_scalars (dict)   - nodal scalars dict {"field_name" : nodal_values}
        elem_scalars (dict)   - cell scalars dict  {"field_name" : cell_values }

    Returns:
        - 
    """

    # Element data with node number prefix
    cells = np.array([[len(e)-1] + e[1:] for e in elems])

    # Cell Types list
    ctypes = np.array([cell_types[len(e)-1] for e in cells])

    grid = pv.UnstructuredGrid(cells, ctypes, nodes)

    if (node_scalars is not None):
        for field_name in node_scalars:
            grid.point_data[field_name] = node_scalars[field_name]

    if (elem_scalars is not None):
        for field_name in elem_scalars:
            grid.cell_data[field_name] = elem_scalars[field_name]

    grid.save(vtk_path)