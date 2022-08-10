import numpy as np
import update
import makeSystem as EA
import track

def collect_data(model,N):
    L = np.int(N**(1/3) + .01)
    SpinData, h = EA.main(L,3)
    return SpinData, h

def init_param(config, N, c):
    average_m = np.mean(config)
    threshold = 0
    stepsize = -1*(c/np.sqrt(N))
    return average_m, threshold, stepsize

def dynamics(time, observables,SpinData,stepsize,mode, movement_data):
    threshold  = 0
    N = len(SpinData.config)

    if len(SpinData.FitnessVec[(SpinData.FitnessVec < 0)]) == 0: ### while fitness vector is still stable, pick an external field that will destabilize##
        observables['1D']['magnetization']['x'] = observables['1D']['magnetization']['x'] + stepsize
        SpinData.FitnessVec = SpinData.FitnessVec + stepsize*(SpinData.config)


    if len(SpinData.FitnessVec[(SpinData.FitnessVec <= 0)]) != 0:
        print('Prior GD Fitness', SpinData.FitnessVec)

        SpinData, movement_data = update.gradientDescent(time,SpinData.FitnessVec,SpinData,observables['1D']['magnetization']['x'], N, threshold,'HO', movement_data)
        observables['2D']['correlations'] = SpinData.correlations

    assert SpinData.FitnessVec.shape == (N,)

    Track = track.Tracker(SpinData)
    average_m = Track.track_magnetization(SpinData.config)
    localFields, FitnessCounts, negFitnessCounts, posFitnessCounts = Track.get_distributions(SpinData.FitnessVec, SpinData.config)

    observables['1D']['fitness_distribution']['x'] = localFields
    observables['1D']['fitness_distribution']['y'] = FitnessCounts
    observables['1D']['magnetization']['y'] = average_m


    return SpinData, observables, movement_data

def MonteCarlo(SpinData,observables,mode):
    N = len(SpinData.config)
    downstep = -1.0
    m_final_neg = -1.0
    stepsize = (-1)/np.sqrt(N)
    for time in range(0,1000):
        dynamics(observables, SpinData, stepsize,mode)
