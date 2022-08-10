import numpy as np

class Tracker:
    #tracks the state of the Ising sping after avalanche
    def __init__(self, SpinData):
        self.FitnessVec = SpinData.FitnessVec
        self.SpinConfig = SpinData.config
        magnetization = np.mean(self.SpinConfig)
        self.magnetization = self.track_magnetization(SpinData.config)
        self.negFitness, self.posFitness = self.separate_fitness(SpinData.FitnessVec, SpinData.config)
        self.FitnessCounts, self.negFitnessCounts, self.posFitnessCounts, self.localFields = self.get_distributions(SpinData.FitnessVec, SpinData.config)

    def track_magnetization(self, SpinConfig):
        #magnetization.append(np.mean(SpinConfig))
        self.magnetization = np.mean(SpinConfig)
        return self.magnetization

    def separate_fitness(self, FitnessVec, SpinConfig):
        self.negFitness = self.FitnessVec[SpinConfig == -1]
        self.posFitness = self.FitnessVec[SpinConfig == 1]
        return self.negFitness, self.posFitness

    def get_distributions(self, FitnessVec, SpinConfig):
        self.negFitness, self.posFitness = self.separate_fitness(FitnessVec, SpinConfig)
        N = len(self.FitnessVec)
        bin_spacing = np.arange(-2,5,1/np.sqrt(N))
        self.FitnessCounts, localFields = np.histogram(FitnessVec, bin_spacing)
        self.negFitnessCounts, localFields = np.histogram(self.negFitness, bin_spacing)
        self.posFitnessCounts, localFields = np.histogram(self.posFitness, bin_spacing)
        self.localFields = localFields[:-1]
        return self.localFields, self.FitnessCounts, self.negFitnessCounts, self.posFitnessCounts
