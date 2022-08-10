import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

class EA:
    def __init__(self, bonds,config, interactions, frustrations, FitnessVec):
        self.bonds = bonds
        self.config = config
        self.pairinter = interactions
        self.correlations = frustrations
        self.FitnessVec = FitnessVec

def assign_spins(d,L):
    N = L**d
    lattice = np.full((N,), 1)
    return lattice

def index_conversion_3D_1D(L,i,j,k):
    arrIndex = i*(L**2) + j*(L) + k
    i_p, j_p, k_p = index_conversion_1D_3D(arrIndex, L)
    assert (i_p, j_p, k_p) == (i, j, k)
    return arrIndex

def index_conversion_1D_3D(arrIndex, L):
    i = arrIndex//(L**2)
    j = (arrIndex - i*(L**2))//L
    k = arrIndex - (i*(L**2) + j*L)
    return i,j,k

def fill_BondMatrix(BondMatrix, indX, indY, J):
    ### make a symmetric Bond Matrix ###
    BondMatrix[indX, indY] = J
    BondMatrix[indY, indX] = J
    return BondMatrix

def energy(bonds,config):
    X,Y = np.meshgrid(config, config)
    N = len(bonds)
    JMatrix = bonds*(X*Y)
    energy = -sum(sum(JMatrix))/2.0
    energy = energy/N
    return JMatrix, energy

def init_fitness(interaction_matrix, config):
    sum_J = np.sum(interaction_matrix, axis=0)
    FitnessVec = sum_J 
    h = (-1)*np.min(FitnessVec)
    FitnessVec = h + FitnessVec
    return FitnessVec, h


def assign_couplings(L,d):
    N = L**3
    BondMatrix = np.zeros((N,N))
    std = 1.0/(np.sqrt(2*d))
    J = np.random.normal(0, std, size = N*(N-1))
    # Going through the whole coordinate space to assign Gaussian bonds to nearest neighbors
    for i in np.arange(0,L):
        iN = np.arange(0,L)[(i-1)%L]

        for j in np.arange(0,L):
            jN = np.arange(0,L)[(j -1)%L]

            for k in np.arange(0,L):
                kN = np.arange(0,L)[(k-1)%L]
                indX = index_conversion_3D_1D(L, i, j, k)
                N1 = index_conversion_3D_1D(L, iN,j,k)
                N3 = index_conversion_3D_1D(L, i, jN, k)
                N5 = index_conversion_3D_1D(L, i, j, kN)
                neighbors = [N1, N3, N5]
                for count, neighbor in enumerate(neighbors):
                    BondMatrix = fill_BondMatrix(BondMatrix, indX, neighbor, random.choice(J))
    return BondMatrix

def main(L,d):
    bonds = assign_couplings(L,d)
    config = assign_spins(d,L)
    JMatrix, initialenergy = energy(bonds, config)
    frustrations = np.zeros((JMatrix.shape))
    FitnessVec, h = init_fitness(JMatrix, config)
    EAdata = EA(bonds, config, JMatrix, frustrations, FitnessVec)
    return EAdata, h
