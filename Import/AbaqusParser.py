"""
    Abaqus nodes/elems/sets/... parser
"""


import numpy as np


def parse_nodes_abaqus(node_data):
    """
    Parses nodal IDs and positions from TXT node data block

    Parameters:
        node_data (list[str]) - raw TXT node data

    Returns:
        nodes (2D np.array)
    """
    
    nodes = [ ]

    for line in node_data:
        node = list(map(lambda x: float(x), line.split(",")))
        node[0] -= 1
        nodes.append(node)

    return np.array(nodes)


def parse_elems_abaqus(elem_data):
    """
    Parses element ID and contained nodes from TXT elem data block

    Parameters:
        elem_data (list[str]) - raw TXT elem data

    Returns:
        elems (list[list])
    """
    
    elems = [ ]

    for line in elem_data:
        line = line[:-1] if not line[-1].isdigit() else line 
        elems.append(list(map(lambda x: int(x) - 1, line.split(','))))

    return elems


def parse_set_abaqus(nset_data):
    """
    Parses NSET's contained nodes from TXT set data block

    Parameters:
        nset_data (list[str]) - raw TXT nset data

    Returns:
        nset (np.array)
    """

    nset = [ ]

    for line in nset_data:
        line = line[:-1] if not line[-1].isdigit() else line
        nset += list(map(lambda x: int(x)-1, line.split(',')))

    return nset