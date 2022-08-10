import numpy as np
import makeSystem as EA
#""python code for EO""

class spinActivity:

    def __init__(self, time, SpinIndex, SpinData, movement_data):
        self.time = time
        self.SpinIndex = SpinIndex
        self.config  = SpinData.config
        self.correlations = SpinData.correlations
        self.oldFitnessVec =  movement_data['oldFitnessVec']
        self.newFitnessVec =  movement_data['newFitnessVec']
        self.neg_velocity = movement_data['neg_velocity']
        self.pos_velocity = movement_data['pos_velocity']

    def fill_in_correlations(self):
        deltaFitness = np.round(self.newFitnessVec - self.oldFitnessVec, 10)
        frustratedSpins = deltaFitness[deltaFitness < 0]
        frustratedSpinsIndices = np.where(deltaFitness < 0)[0]
        satisfiedSpins = deltaFitness[deltaFitness > 0]
        satisfiedSpinsIndices = np.where(deltaFitness > 0)[0]

        assert len(frustratedSpinsIndices) <= 7;
        assert len(satisfiedSpinsIndices) <= 7;

        self.correlations[self.SpinIndex][frustratedSpinsIndices] = 1.0
        self.correlations.T[self.SpinIndex][frustratedSpinsIndices] = 1.0
        self.correlations[self.SpinIndex][satisfiedSpinsIndices] = -1.0
        self.correlations.T[self.SpinIndex][satisfiedSpinsIndices] = -1.0
        return self.correlations

    def flow_data(self):
        deltaFitness = np.round(self.newFitnessVec - self.oldFitnessVec, 10)

        negmovement = deltaFitness[self.config == -1]
        posmovement = deltaFitness[self.config == 1]

        neg_velocity += np.mean(negmovement)
        pos_velocity += np.mean(posmovement)
        return self.neg_velocity, self.pos_velocity

def get_fitness(interaction_matrix, config, h):
    sum_Jij = np.sum(interaction_matrix, axis=0)
    newFitnessVec = sum_Jij + config*h
    return newFitnessVec

def update_PairInteractions(SpinData):
    pairInteractions, energy = EA.energy(SpinData.bonds, SpinData.config)
    SpinData.pairinter = pairInteractions
    return SpinData


def gradientDescent(time, FitnessVec,SpinData,h,N,threshold, mode, movement_data):
    spinIndices = np.arange(0, N)
    mask = (FitnessVec <= threshold)
    unstablePos = spinIndices[mask]
    position_index = np.random.randint(0,len(unstablePos))
    SpinIndex = unstablePos[position_index]
    movement_data['oldFitnessVec'] = SpinData.FitnessVec
    SpinData = update_PairInteractions(SpinData)
    if FitnessVec[SpinIndex] <= threshold:
        SpinData.config[SpinIndex] = SpinData.config[SpinIndex]*(-1)

        SpinData = update_PairInteractions(SpinData)
        newFitnessVec = get_fitness(SpinData.pairinter, SpinData.config, h)

        movement_data['newFitnessVec']= newFitnessVec
        SpinActivity = spinActivity(time, SpinIndex, SpinData, movement_data)
        SpinActivity.fill_in_correlations()
    
        SpinData.FitnessVec = newFitnessVec
        SpinData.correlations = SpinActivity.correlations

    return SpinData, movement_data
