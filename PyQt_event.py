import sys
import numpy as np
import math
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import protocol
import setAx
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer


##these are all the window functions
class mainScreen(FigureCanvas):
    def __init__(self):
        # instantiate the window and create first frame
        self.frame = 0
        #set main system configuration
        self.mainData = mainData
        self.import_parameters(observables)

        FigureCanvas.__init__(self, self.fig)
        #figure buttons and stuff#
        self.start_button = QPushButton('Start', self)
        self.end_button = QPushButton('End', self)
        self.label = QLabel(self)
        self.start_button.clicked.connect(self.Start)
        self.end_button.clicked.connect(self.End)

        #*set up the button and label placements on the canvas* #
        self.start_button.move(50,0)
        self.end_button.move(500,0)
        self.label.setGeometry(300, 0, 75,75)
        ### in setGeometry (x,y, w, h) ####
        self.fig.canvas.draw()


    def Start(self):
        self.timer = self.startTimer(1000)

    def End(self):
        self.timer = self.killTimer(self.timer)

    def import_parameters(self, observables):
        # *first determine the grid of the figure, how many plots do you want in one row?*
        grid_div = 3
        fig, axs = setAx.setFigGrid(observables, grid_div)

        self.fig = fig
        self.axs = axs
        self.observables = observables
        init_indx = 0
        for graph_type in observables.keys():
            line, axs, init_indx = setAx.makeSubPlots(init_indx, observables, axs, graph_type)

    def timerEvent(self, event):
        self.frame += 1
        movement_data ={'neg_velocity':[], 'pos_velocity':[]}

        # *this is where the update function goes* 
        self.mainData, self.observables, movement_data = protocol.dynamics(self.frame, self.observables, self.mainData,stepsize,'HO', movement_data)
        self.label.setText('Time: ' + str(self.frame))
        init_indx = 0
        for graph_type in self.observables.keys():
            line, axs, init_indx = setAx.makeSubPlots(init_indx, self.observables, self.axs, graph_type)
        self.fig.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    N = 27

    # *these are the changing variables, please edit based on what you want to visualize*
    observables = {'2D':{'correlations': []}, '1D':{'fitness_distribution':{'x':[], 'y':[]}, 'magnetization':{'x':[], 'y':[]}}}

    # *this is the Edwards-Anderson spin glass, change to whatever you are analyzing. This sets up the structure of the system*
    mainData, h = protocol.collect_data('EA',N)

    # *these are all the initial measurements*
    fitnessCounts, bin_edges = np.histogram(mainData.FitnessVec, np.arange(-2,5,1/np.sqrt(N)))
    average_m,threshold, stepsize = protocol.init_param(mainData.config, N, 1)

    # *initialize your hash table with the above measurements*
    observables['1D']['fitness_distribution']['x'] = bin_edges[:-1]
    observables['1D']['fitness_distribution']['y'] = fitnessCounts
    observables['1D']['magnetization']['x'] = h
    observables['1D']['magnetization']['y'] = average_m
    observables['2D']['correlations'] = mainData.correlations

    w = mainScreen()

    ## name
    w.setWindowTitle("fitness dynamics")
    w.show()
    sys.exit(app.exec_())
