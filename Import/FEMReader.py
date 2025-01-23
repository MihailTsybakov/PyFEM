"""
    Reading Abaqus/... setup file
"""


from Import.AbaqusParser import *


keyword = "*"
comment = "**"


# Mapping between keyword and corresponding parser func
parsers_dict = {
    "*NODE"    : parse_nodes_abaqus,
    "*ELEMENT" : parse_elems_abaqus,
    "*NSET"    : parse_set_abaqus
}


def read_setup(file_path):
    """
    Reads FEM setup (nodes, elems, sets, ...) from given setup file

    Parameters:
        file_path (str)

    Returns:
        setup (dict) - FEA setup with nodes, elements, sets
    """
    
    data_blocks = read_preliminary(file_path)
    setup       = parse_data(data_blocks)

    return setup


def read_preliminary(file_path):
    """
    Performs a preliminary setup file scan, splitting data by blocks

    Parameters:
        file_path (str)

    Returns:
        data_blocks (dict) - raw data blocks list in txt form 
    """

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.count(comment) == 0]

    keylines = [i for i in range(len(lines)) if lines[i].count(keyword) == 1] + [len(lines)]

    data_blocks = { lines[keylines[i]] : lines[keylines[i]+1 : keylines[i+1]] for i in range(len(keylines)-1) }

    return data_blocks


def parse_data(data_blocks):
    """
    Applies the corresponding parser to each data block given

    Parameters:
        data_blocks (dict) - raw txt data blocks for parsing

    Returns:
        data_blocks_parsed (dict) - parsed FEM setup data
    """

    data_blocks_parsed = { }

    for db_name in data_blocks:
        for parser_tag in parsers_dict:
            if (db_name.count(parser_tag) == 1):
                data_blocks_parsed[db_name] = parsers_dict[parser_tag](data_blocks[db_name])

    return data_blocks_parsed