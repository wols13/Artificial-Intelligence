# Look for #IMPLEMENT tags in this file. These tags indicate what has
# to be implemented to complete the warehouse domain.

"""
Construct and return Tenner Grid CSP models.
"""

from cspbase import *
import itertools


def add_not_equal_constraint(tenner_csp, variables, cord1, cord2):
    if cord1[0] < 0 or cord2[0] < 0 or cord1[0] > 9 or cord2[0] > 9:
        return
    if cord1[1] < 0 or cord2[1] < 0 or cord1[1] >= len(variables) or cord2[1] >= len(variables):
        return

    name = "RowConstraint({},{})({}, {})".format(cord1[0], cord1[1], cord2[0], cord2[1])
    scope = [variables[cord1[1]][cord1[0]], variables[cord2[1]][cord2[0]]]
    row_constraint = Constraint(name, scope)

    pair_domain = list(range(10)), list(range(10))
    satisfying_tuples = [(x, y) for x, y in itertools.product(*pair_domain) if x != y]
    row_constraint.add_satisfying_tuples(satisfying_tuples)
    tenner_csp.add_constraint(row_constraint)
    return


def add_col_sum_constraints(tenner_csp, variables, last_row):
    col_domain = (list(range(10)),) * len(variables)
    for col in range(10):
        name = "ColSumConstraint({})".format(col)
        scope = []
        for row in range(len(variables)):
            scope.append(variables[row][col])
        col_sum_constraint = Constraint(name, scope)

        satisfying_tuples = [i for i in itertools.product(*col_domain) if sum(i) == last_row[col]]
        col_sum_constraint.add_satisfying_tuples(satisfying_tuples)
        tenner_csp.add_constraint(col_sum_constraint)
    return


def generate_cell_variables(tenner_csp, variables, n_grid):
    for y in range(len(n_grid)):
        variable_row = []
        for x in range(10):
            domain = list(range(10)) if n_grid[y][x] == -1 else [n_grid[y][x]]
            new_variable = Variable("Cell({}, {})".format(x, y), domain)
            tenner_csp.add_var(new_variable)
            variable_row.append(new_variable)
        variables.append(variable_row)
    return


def generate_supporting_tuples(satisfying_tuples, row_domain, current_tuple):
    if len(current_tuple) == 9:
        for val in row_domain[9]:
            if val not in current_tuple:
                new_current_tuple = current_tuple
                new_current_tuple += (val,)
                satisfying_tuples.append(new_current_tuple)
        return

    for val in row_domain[len(current_tuple)]:
        if val not in current_tuple:
            new_current_tuple = current_tuple
            new_current_tuple += (val,)
            generate_supporting_tuples(satisfying_tuples, row_domain, new_current_tuple)
    return


def generate_row_product(variables, row):
    row_product = []
    for var in variables[row]:
        row_product.append(var.domain())

    for i in range(10):
        if len(row_product[i]) == 1:
            for j in range(10):
                if j != i and len(row_product[j]) != 1:
                    index = row_product[j].index(row_product[i][0])
                    row_product[j].pop(index)
    return row_product


def add_model2_row_constraints(tenner_csp, variables, row):
    satisfying_tuples = []
    generate_supporting_tuples(satisfying_tuples, generate_row_product(variables, row), ())

    name = "RowConstraint({})".format(row)
    scope = []
    for cell in range(10):
        scope.append(variables[row][cell])
    row_constraint = Constraint(name, scope)
    row_constraint.add_satisfying_tuples(satisfying_tuples)
    tenner_csp.add_constraint(row_constraint)
    return


def add_row_adjacency_constraints(tenner_csp, variables, mode):
    for y in range(len(variables)):
        if mode == "model_2":
            # Add n-ary all different constraints for each row
            add_model2_row_constraints(tenner_csp, variables, y)
        for x in range(10):
            if mode == "model_1":
                # Add not equal constraints to row neighbours to the right
                for row_neighbour in range(x + 1, 10):
                    add_not_equal_constraint(tenner_csp, variables, (x, y), (row_neighbour, y))
            # Add not equal constraints to adjacent cells in the row below
            add_not_equal_constraint(tenner_csp, variables, (x, y), (x, y + 1))
            add_not_equal_constraint(tenner_csp, variables, (x, y), (x + 1, y + 1))
            add_not_equal_constraint(tenner_csp, variables, (x, y), (x - 1, y + 1))
    return


def tenner_csp_model_1(initial_tenner_board):
    """Return a CSP object representing a Tenner Grid CSP problem along
       with an array of variables for the problem. That is return

       tenner_csp, variable_array

       where tenner_csp is a csp representing tenner grid using model_1
       and variable_array is a list of lists

       [ [  ]
         [  ]
         .
         .
         .
         [  ] ]

       such that variable_array[i][j] is the Variable (object) that
       you built to represent the value to be placed in cell i,j of
       the Tenner Grid (only including the first n rows, indexed from 
       (0,0) to (n,9)) where n can be 3 to 8.
       
       
       The input board is specified as a pair (n_grid, last_row). 
       The first element in the pair is a list of n length-10 lists.
       Each of the n lists represents a row of the grid. 
       If a -1 is in the list it represents an empty cell. 
       Otherwise if a number between 0--9 is in the list then this represents a 
       pre-set board position. E.g., the board
    
       ---------------------  
       |6| |1|5|7| | | |3| |
       | |9|7| | |2|1| | | |
       | | | | | |0| | | |1|
       | |9| |0|7| |3|5|4| |
       |6| | |5| |0| | | | |
       ---------------------
       would be represented by the list of lists
       
       [[6, -1, 1, 5, 7, -1, -1, -1, 3, -1],
        [-1, 9, 7, -1, -1, 2, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, -1, -1, -1, 1],
        [-1, 9, -1, 0, 7, -1, 3, 5, 4, -1],
        [6, -1, -1, 5, -1, 0, -1, -1, -1,-1]]
       
       
       This routine returns model_1 which consists of a variable for
       each cell of the board, with domain equal to {0-9} if the board
       has a -1 at that position, and domain equal {i} if the board has
       a fixed number i at that cell.
       
       model_1 contains BINARY CONSTRAINTS OF NOT-EQUAL between
       all relevant variables (e.g., all pairs of variables in the
       same row, etc.).
       model_1 also contains n-nary constraints of sum constraints for each
       column.
    """

    n_grid = initial_tenner_board[0]
    last_row = initial_tenner_board[1]
    tenner_csp = CSP("TennerModel1")
    variables = []

    # Generate cell variables with appropriate domains & update tenner_csp
    generate_cell_variables(tenner_csp, variables, n_grid)

    # Add row and adjacency constraints to tenner_csp
    # Use binary row constraints for "model_1", use n-ary row constraints for "model_2"
    add_row_adjacency_constraints(tenner_csp, variables, "model_1")

    # Add n-ary column sum constraints to tenner_csp
    add_col_sum_constraints(tenner_csp, variables, last_row)

    return tenner_csp, variables


def tenner_csp_model_2(initial_tenner_board):
    """Return a CSP object representing a Tenner Grid CSP problem along
       with an array of variables for the problem. That is return

       tenner_csp, variable_array

       where tenner_csp is a csp representing tenner using model_1
       and variable_array is a list of lists

       [ [  ]
         [  ]
         .
         .
         .
         [  ] ]

       such that variable_array[i][j] is the Variable (object) that
       you built to represent the value to be placed in cell i,j of
       the Tenner Grid (only including the first n rows, indexed from 
       (0,0) to (n,9)) where n can be 3 to 8.

       The input board takes the same input format (a list of n length-10 lists
       specifying the board as tenner_csp_model_1.
    
       The variables of model_2 are the same as for model_1: a variable
       for each cell of the board, with domain equal to {0-9} if the
       board has a -1 at that position, and domain equal {i} if the board
       has a fixed number i at that cell.

       However, model_2 has different constraints. In particular, instead
       of binary non-equals contains model_2 has a combination of n-nary
       all-different constraints: all-different constraints for the variables in
       each row, contiguous cells (including diagonally contiguous cells), and 
       sum constraints for each column. Each of these constraints is over more 
       than two variables (some of these variables will have
       a single value in their domain). model_2 should create these
       all-different constraints between the relevant variables.
    """

    n_grid = initial_tenner_board[0]
    last_row = initial_tenner_board[1]
    tenner_csp = CSP("TennerModel2")
    variables = []

    # Generate cell variables with appropriate domains & update tenner_csp
    generate_cell_variables(tenner_csp, variables, n_grid)

    # Add row and adjacency constraints to tenner_csp
    # Binary constraints for "model_1", n-ary constraints for "model_2"
    add_row_adjacency_constraints(tenner_csp, variables, "model_2")

    # Add n-ary column sum constraints to tenner_csp
    add_col_sum_constraints(tenner_csp, variables, last_row)

    return tenner_csp, variables
