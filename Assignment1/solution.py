#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete the Sokoban warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

# import os for time functions
import os
from search import * #for search engines
from sokoban import SokobanState, Direction, PROBLEMS, sokoban_goal_state #for Sokoban specific classes and problems

#  Additional imports
import math
from scipy.sparse.csgraph import shortest_path
from scipy.optimize import linear_sum_assignment
import numpy as np

#Global Directions
UP = Direction("up", (0, -1))
RIGHT = Direction("right", (1, 0))
DOWN = Direction("down", (0, 1))
LEFT = Direction("left", (-1, 0))

#Globals for alternate heuristic
distance_matrix = None
previous_hvals = {}
last_index = 0


#SOKOBAN HEURISTICS
def heur_displaced(state):
  '''trivial admissible sokoban heuristic'''
  '''INPUT: a sokoban state'''
  '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''       
  count = 0
  for box in state.boxes:
    if box not in state.storage:
      count += 1
  return count

def heur_manhattan_distance(state):
#IMPLEMENT
    '''admissible sokoban heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''      
    #We want an admissible heuristic, which is an optimistic heuristic. 
    #It must always underestimate the cost to get from the current state to the goal.
    #The sum Manhattan distance of the boxes to their closest storage spaces is such a heuristic.  
    #When calculating distances, assume there are no obstacles on the grid and that several boxes can fit in one storage bin.
    #You should implement this heuristic function exactly, even if it is tempting to improve it.
    #Your function should return a numeric value; this is the estimate of the distance to the goal.

    sum_manhattan_distance = 0
    valid_storage = state.storage

    for box in state.boxes:
        min_manhattan_distance = math.inf
        if state.restrictions is not None:
            valid_storage = state.restrictions[state.boxes[box]]
        for storage in valid_storage:
            manhattan_distance = abs(box[0] - storage[0]) + abs(box[1] - storage[1])
            min_manhattan_distance = min(manhattan_distance, min_manhattan_distance)
        sum_manhattan_distance += min_manhattan_distance

    return sum_manhattan_distance

def heur_alternate(state):
#IMPLEMENT
    '''a better sokoban heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''        
    #heur_manhattan_distance has flaws.   
    #Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    #Your function should return a numeric value for the estimate of the distance to the goal.

    global distance_matrix
    global previous_hvals
    global last_index
    all_box_storage_distances = []

    #  If box positions are the same as parent, h_val remains unchanged
    if state.parent and (state.boxes == state.parent.boxes):
        previous_hvals[state.index] = previous_hvals[state.parent.index]
        last_index = state.index
        return previous_hvals[state.index]

    #  Generate all pairs distance matrix only once per new puzzle & store as global
    if last_index == 0 or (last_index - state.index) > 100:
        distance_matrix = all_pairs_distance(state)

    #  last_index global is used to detect new puzzles
    last_index = state.index

    #  Generate matrix that represents shortest distance between each box and each storage
    #  For box x, if storage y is not within x's restriction, distance[x][y] = 100000000
    for box in state.boxes:
        row = (box[0] * state.height) + box[1]
        distance_to_all_storages = []
        for store in state.storage:
            if state.restrictions and (store not in state.restrictions[state.boxes[box]]):
                distance_to_all_storages.append(100000000)
            else:
                col = (store[0] * state.height) + store[1]
                distance_to_all_storages.append(distance_matrix[row][col])
        all_box_storage_distances.append(distance_to_all_storages)

    #  Use SciPy's hungarian algorithm implementation to determine optimal box-storage assignment
    all_box_storage_distances = np.array(all_box_storage_distances)
    row_ind, col_ind = linear_sum_assignment(all_box_storage_distances)
    previous_hvals[state.index] = all_box_storage_distances[row_ind, col_ind].sum()

    return previous_hvals[state.index]


def all_pairs_distance(state):
    '''Helper that computes shortest path from each puzzle cell to every other cell.
    It also observes obstacles and deadlocks'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: all pairs distance matric'''

    no_of_cells = state.width * state.height
    input_matrix = []

    #  initialize input matrix: no_of_cells by no_of_cells matrix of zeroes
    for _ in range(no_of_cells):
        zero_row = [0] * no_of_cells
        input_matrix.append(zero_row)

    #  for each cell, distance to each of its immediate neighbours is 1
    #  except the neighbour is an obstacle, distance = 10000000
    for x in range(state.width):
        for y in range(state.height):
            deadends = 0
            for direction in (UP, RIGHT, DOWN, LEFT):
                neighbour = direction.move((x, y))
                if out_of_bounds(state, neighbour):
                    deadends += 2
                    continue
                row = (x * state.height) + y
                column = (neighbour[0] * state.height) + neighbour[1]
                if neighbour in state.obstacles:
                    deadends += 1
                    input_matrix[row][column] = 10000000
                else:
                    input_matrix[row][column] = 1
            #  a cell is a deadend if it has: 1 or more neighbouring walls
            #                                 2 or more neighbouring obstacles
            if deadends > 1:
                row = (x * state.height) + y
                input_matrix[row] = [10000000] * no_of_cells

    #  Use SciPy's shortest path function to compute all pairs shortest distance
    input_matrix = np.array(input_matrix)
    return shortest_path(input_matrix, method='auto', directed=True,
                         return_predecessors=False, unweighted=False, overwrite=False)


def out_of_bounds(state, cell):
    '''Checks if 'cell' is out of bounds in 'state' '''
    '''INPUT: a sokoban state and a tuple of two integers representing a grid location'''
    '''OUTPUT: True iff input cell is NOT within the boundaries of the current state, False otherwise.'''

    if cell[0] < 0 or cell[0] >= state.width:
        return True
    if cell[1] < 0 or cell[1] >= state.height:
        return  True
    return False


def fval_function(sN, weight):
#IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """
  
    #Many searches will explore nodes (or states) that are ordered by their f-value.
    #For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue is the hval of the state.
    #You can use this function to create an alternate f-value for states; this must be a function of the state and the weight.
    #The function must return a numeric f-value.
    #The value will determine your state's position on the Frontier list during a 'custom' search.
    #You must initialize your search engine object as a 'custom' search engine if you supply a custom fval function.
    return sN.gval + (weight * sN.hval)

def anytime_gbfs(initial_state, heur_fn, timebound = 10):
#IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''

    start_time = os.times()[0]
    new_se = SearchEngine('best_first', 'full')
    new_se.init_search(initial_state, sokoban_goal_state, heur_fn)
    result = new_se.search(timebound)

    #After initial iteration, search for more optimal solution
    temp = result
    while temp and (timebound > 0):
        temp = new_se.search(timebound, (temp.gval - 1, math.inf, math.inf))
        if temp:
            result = temp
        timebound = timebound - (os.times()[0] - start_time)

    return result

def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound = 10):
#IMPLEMENT
    '''Provides an implementation of anytime weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''

    start_time = os.times()[0]
    new_se = SearchEngine('custom')
    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    new_se.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)
    result = new_se.search(timebound)

    # After initial iteration, search for more optimal solution
    temp = result
    while temp and (timebound > 0):
        temp = new_se.search(timebound, (math.inf, math.inf, temp.gval + heur_fn(temp) - 1))
        if temp:
            result = temp
        timebound = timebound - (os.times()[0] - start_time)

    return result

if __name__ == "__main__":
  #TEST CODE
  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 2; #2 second time limit for each problem
  print("*************************************")  
  print("Running A-star")     

  for i in range(0, 10): #note that there are 40 problems in the set that has been provided.  We just run through 10 here for illustration.

    print("*************************************")  
    print("PROBLEM {}".format(i))
    
    s0 = PROBLEMS[i] #Problems will get harder as i gets bigger

    se = SearchEngine('astar', 'full')
    se.init_search(s0, goal_fn=sokoban_goal_state, heur_fn=heur_displaced)
    final = se.search(timebound)

    if final:
      final.print_path()
      solved += 1
    else:
      unsolved.append(i)    
    counter += 1

  if counter > 0:  
    percent = (solved/counter)*100

  print("*************************************")  
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))  
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))      
  print("*************************************") 

  solved = 0; unsolved = []; counter = 0; percent = 0; timebound = 8; #8 second time limit 
  print("Running Anytime Weighted A-star")   

  for i in range(0, 10):
    print("*************************************")  
    print("PROBLEM {}".format(i))

    s0 = PROBLEMS[i] #Problems get harder as i gets bigger
    weight = 10
    final = anytime_weighted_astar(s0, heur_fn=heur_displaced, weight=weight, timebound=timebound)

    if final:
      final.print_path()   
      solved += 1 
    else:
      unsolved.append(i)
    counter += 1      

  if counter > 0:  
    percent = (solved/counter)*100   
      
  print("*************************************")  
  print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))  
  print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))      
  print("*************************************") 



