# https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html -> np.argwhere()
# https://stackoverflow.com/questions/14681609/create-a-2d-list-out-of-1d-list -> np.array().reshape()
# https://www.geeksforgeeks.org/python-ways-to-flatten-a-2d-list/ -> flatten_list
# https://www.poftut.com/python-how-to-print-without-newline-or-space/#:~:text=print%20function%20accepts%20more%20parameters,end%20of%20line%20or%20space. -> sep
# https://numpy.org/doc/stable/reference/generated/numpy.copy.html -> deepcopy()

'''
// Main File:        funny_puzzle.py
// Semester:         CS 540 Fall 2020
// Authors:          Tae Yong Namkoong
// CS Login:         namkoong
// NetID:            kiatvithayak
// References:       https://www.geeksforgeeks.org/a-search-algorithm/
                     https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
                     https://rosettacode.org/wiki/A*_search_algorithm
                     https://stackabuse.com/basic-ai-concepts-a-search-algorithm/
                     TA's Office Hours
'''
import heapq
import numpy as np
import copy
import math

#This function converts an 1D array to a 2D matrix
def get_2d(arr):
    matrix = np.array(arr).reshape(3, 3)
    return matrix

# This function when given a state of the puzzle, represented as a single list of integers with a 0 in the empty space,
# print to the console all of the possible successor states
def print_succ(state):
    # get successor states of initial state, sort, then print to console all possible states
    succ = sorted(get_succ(state))
    for state in succ: # iterate through each state
        arr_state = get_2d(state)
        md = get_man_d(arr_state)
        print(state, " h=", md, sep ='')

# #This function gets all successor states for the initial state
def get_succ(state):
    arr_succ_states = []
    # get start state as 2D matrix
    arr_start_state = get_2d(state)
    # get number of successor states given a current state
    arr_curr_state = np.transpose(np.nonzero(arr_start_state == 0))
    x = arr_curr_state[0][0]
    y = arr_curr_state[0][1]
    # define 4 possible moves (top, bottom, left, right)
    arr_moves = [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]
    length = len(arr_moves)

    # iterate through each possible moves and get valid successors
    for i in range(length):
        row = arr_moves[i][0]
        col = arr_moves[i][1]
        arr_copy = copy.copy(arr_start_state)

        if 0 <= row < 3 and 0 <= col < 3:
            swap = arr_copy[row][col]
            arr_copy[x][y] = swap
            arr_copy[row][col] = 0
            # get successor as 1D array
            one_d = [elem for arr in arr_copy for elem in arr]
            arr_succ_states.append(one_d)
    return arr_succ_states

# This function returns the manhattan distance for each square according to the goal state
def get_man_d(arr):
    goal_state = [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 0]]
    md = 0
    for row in range(3):
        for col in range(3):
            # filter out squares that are already in correct position and empty square
            if arr[row][col] != 0 and arr[row][col] != goal_state[row][col]:
                goal_row = math.floor((arr[row][col] - 1) / 3)
                goal_col = (arr[row][col] - 1) % 3
                md += abs(row - goal_row) + abs(col - goal_col)
    return md

#This function traces solution path using a dictionary
def trace(goal_state, path):
    output = []
    moves = 0
    output.append(goal_state)
    parent = path[str(goal_state)]
    # trace dictionary until start_state and add current state's parent
    while parent != 0: #trace until start state
        output.append(parent)
        parent = path[str(parent)]
    output.reverse()
    # print the solution
    for state in output:
        arr_state = get_2d(state)
        md = get_man_d(arr_state)
        print(state, " h=", md, " moves: ", moves, sep='')
        moves += 1

#This function, when given a state of the puzzle, perform the A* search algorithm and
# print the path from the current state to the goal state
def solve(state):
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0] # goal_state in 1D array
    max_len = 0
    #define dictionary for tracing a given state and its parent states
    dictionary = dict()
    state_arr = get_2d(state)
    md = get_man_d(state_arr)
    h = get_man_d(state_arr)
    state = (md, state, (0, h, 0))
    #declare heap pq
    pq = [state]
    while len(pq) > 0:
        max_len = max(len(pq), max_len)
        # find state with smallest cost
        current = heapq.heappop(pq)
        current_state = current[1]

        # if current state is not the best solution
        if str(current_state) not in dictionary:

            # add key, value pair to dictionary which correspond to current state and parent
            dictionary[str(current[1])] = current[2][2]

            # if there is a goal state, print max queue length
            if current[1] == goal_state:
                trace(current[1], dictionary)
                print("Max queue length:", max_len)
                break

            successor_list = get_succ(current_state)
            for succ in successor_list: #iterate through successors of current state
                if str(succ) not in dictionary: # if this is not the best solution

                    # compute (g+h, state, (g, h, parent_index)) representing both the cost, state and the
                    # parent index in A* search
                    g = 1 + int(current[2][0])
                    succ_arr = get_2d(succ)
                    h = get_man_d(succ_arr)
                    cost = g + h
                    # add to heapq cur state's successors
                    heapq.heappush(pq, (cost, succ, (g, h, current_state)))



