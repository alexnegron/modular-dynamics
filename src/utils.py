import numpy as np
import torch

def generate_w_matrix(device, nNeurons, nBumps,length=40, wWeight=8.0, wShift=2, wScaling=True):
    """
    Generating synaptic connectivity matrix

    Inputs
    ------
    wWeight : positive value; sets the strength of the most inhibitory connection
    wShift : synaptic output shift for L and R populations (xi)
    wScaling : scale the raw wWeight by nNeurons and nBumps
    """
    # Calculating synaptic connectivity values #TODO change for diff connect, too small will result in multiple bumps
    # length = nNeurons / (
    #     2.28 * nBumps
    # )  # inhibition length l that produces nBumps (Eq. 47)
    length2 = int(2 * np.ceil(length))
    positions = np.arange(-length2, length2 + 1)  # Only short-range synapses between -2l and 2l
    if wScaling:  # Scale wWeight so bump shape remains the same
        strength = wWeight * nBumps / nNeurons
    else:
        strength = wWeight

    # Cosine-based connectivity function (Eq. 38)
    values = strength * (np.cos((np.pi * positions / length)) - 1) / 2
    values *= np.abs(positions) < 2 * length

    # Adding values to form unshifted row of w matrix. We form the row this way so that
    # synaptic weights are wrapped around the network in case 4 * length > nNeurons
    # (Eq. 127)
    wUnshifted = torch.zeros(nNeurons, device=device)
    for position, w in zip(positions, values):
        wUnshifted[position % nNeurons] += w

    # Form unshifted matrix of dim (nNeurons, nNeurons), then shift and form final matrix
    # of dim (2 * nNeurons, 2 * nNeurons)
    wQuadrant = torch.vstack([wUnshifted.roll(i) for i in range(nNeurons)])
    wMatrix = torch.hstack((wQuadrant.roll(-wShift, 0), wQuadrant.roll(wShift, 0)))
    wMatrix = torch.vstack((wMatrix, wMatrix))

    return wMatrix

def generate_alpha_matrix(neurons,length,alpha,wWeight=8,
                wShift=2,wScaling=True,device='cpu'):
    cutoff = int((1-alpha)*length)
    wMatrix = generate_w_matrix(device,neurons+cutoff,1,length,wWeight,wShift,wScaling)
    real_w = torch.zeros((neurons*2,neurons*2),device=device)
    end = 2*neurons + cutoff
    real_w[:neurons,:neurons] = wMatrix[:neurons,:neurons] #top left
    real_w[neurons:,neurons:] = wMatrix[neurons+cutoff:end,neurons+cutoff:end] #bottom right
    real_w[:neurons,neurons:] = wMatrix[:neurons,neurons+cutoff:end] #off diagonals
    real_w[neurons:,:neurons] = wMatrix[neurons+cutoff:end,:neurons]
    return real_w




