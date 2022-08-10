import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

def setFigGrid(observables, grid_div):
    total_graphs = sum([len(v) for v in observables.values()])
    rows, cols = math.ceil(total_graphs/grid_div), total_graphs*math.ceil(total_graphs/grid_div)
    fig, axs = plt.subplots(rows, cols, figsize = (6*cols,4*rows))
    if rows == 1 and cols == 1:
        axs = [axs]
    return fig, axs

def LineData(init_indx, observables, axs, graph_type):
    for indx, observable in enumerate(sorted(observables[graph_type].keys())):
        if isinstance(observables['1D'][observable]['x'], float) == True:
            line = axs[indx + init_indx].scatter(observables['1D'][observable]['x'],observables['1D'][observable]['y'], marker = '*', s = 30, c='black')
            axs[indx + init_indx].set_ylim([-1.2, 1.2])
            axs[indx + init_indx].set_xlim([-3,3])
        else:
            axs[indx].cla()
            line = axs[indx + init_indx].plot(observables['1D'][observable]['x'], observables['1D'][observable]['y'])
            axs[indx + init_indx].set_ylim([0, 8])
            axs[indx + init_indx].set_xlim([-4,5])

    init_indx = indx + init_indx
    return line, axs, init_indx

def GridData(init_indx, observables, axs, graph_type):
    for indx, observable in enumerate(sorted(observables[graph_type].keys())):
        plt.imshow(observables[graph_type][observable])
    
    return axs, init_indx


def makeSubPlots(init_indx, observables, axs, graph_type):
    line = 0
    if graph_type == '1D':
        line, axs, init_indx  = LineData(init_indx, observables, axs, graph_type)

    if graph_type == '2D':
        axs, init_indx = GridData(init_indx, observables, axs, graph_type)
    return line, axs, init_indx
