import numpy as np
from ase import Atoms


def features(atoms, ksis, lambs, etas, angularEtas, rss, rc):
    """ Builds feature vectors for every atom in the structure object atoms. Returns array of featurevectors in order of the appearence in the structure objects.

            Parameters
            ----------
            atoms: A structure object containing a list of atoms objects.
            ksis: HyperParameters, should be chosen as 1, 2, 4, 16, 64 ..
            lambs: HyperParameters, should be chosen as -1, 1.
            etas: HyperParameters. Is chosen between 0.05 and 80 in the cited paper.
            angularEtas: HyperParameters. Is chosen between 0.001 and 2 in the cited paper.
            rss: HyperParameters. Unknown for now
            rc: Cut-off radius. Interactions between atoms outside of this radius from the atom in question, will be neglected

            """
    # A matrix of all distances
    distances = atoms.get_all_distances()
    # For storing the featurevectors
    featureVectors = []
    for i, atom in enumerate(atoms, 0):
        featureVectors.append(featureVector(i, atoms, ksis, lambs, etas, angularEtas, rss, rc, distances))
    return featureVectors


def featureVector(i, atoms, ksis, lambs, etas, angularEtas, rss, rc, distances):
    """ Building a feature vector for one atom, with index i the structure object atoms.

        Parameters
        ----------
        i: The index of the respective atom
        atoms: A structure object containing a list of atoms objects.
        ksis: HyperParameters, should be chosen as 1, 2, 4, 16, 64 ..
        lambs: HyperParameters, should be chosen as -1, 1.
        etas: HyperParameters. Is chosen between 0.05 and 80 in the cited paper.
        etas: HyperParameters. Is chosen between 0.001 and 2 in the cited paper.
        rss: HyperParameters. Unknown for now
        rc: Cut-off radius. Interactions between atoms outside of this radius from the atom in question, will be neglected
        distances: Matrix of distances between atoms.

        """
    # For storing the resulting feature vector
    result = []
    for ksi in ksis:
        for lamb in lambs:
            for eta in angularEtas:
                result.append(angular(i, atoms, ksi, lamb, eta, rc, distances))
    for eta in etas:
        for rs in rss:
            result.append(radial(i, atoms, eta, rs, rc, distances))
    return result


def fc(r, rc):
    """ Support function for building the feature vector. Used for constraining the range of interaction.

            Parameters
            ----------
            r: Distance between two atoms.
            rc: The cut-off distance, will return 0 for r>rc.
            """
    if rc > r:
        return 0.5*(1+np.cos(np.pi*r/rc))
    else:
        return 0
def angular(i, atoms, ksi, lamb, eta, rc, distances):
    """ An angular feature function as per Behler and Parrinello. Returns a single angular feature, in the form of a number.

    Parameters
    ----------
    atoms: A structure object containing a list of atoms objects.
    ksi: HyperParameter, should be chosen as 1, 2, 4, 16, 64 ..
    lamb: HyperParameter, should be chosen as -1, 1.
    eta: HyperParameter. Is chosen between 0.001 and 2 in the cited paper.
    rc: Cut-off radius. Interactions between atoms outside of this radius from the atom in question, will be neglected
    distances: Matrix of distances between atoms.

    """
    # For storing the resulting feature
    result = 0
    # Distances between atoms i,j and k
    rij = 0
    rik = 0
    rjk = 0
    # Angle between j and k, centered on i
    phiijk = 0
    # The following should have if-checks (instead of fc returning 0) implemented for the sake of efficiancy, but this is form in the paper and it will do for now.
    # (2**(1-ksi))*(1-lamb*np.cos(phijk))**ksi*np.exp(-eta*(rij**2+rik**2+rjk**2)/rc**2)*fc(rij, rc)*fc(rik, rc)*fc(rjk, rc)
    number_of_atoms = atoms.get_global_number_of_atoms()
    for j in range(0, number_of_atoms):
        if(i!=j):
            for k in range(0, number_of_atoms):
                if(k!=i and k!=j):
                    rij = distances[i, j]
                    rik = distances[i, k]
                    rjk = distances[j, k]
                    if(rij<rc and rik<rc and rjk<rc):
                        phiijk = atoms.get_angle(j, i, k) #Denne metode er fucked og SKAL erstattes inden grendel.
                        result += (2**(1-ksi))*(1-lamb*np.cos(phiijk))**ksi*np.exp(-eta*(rij**2+rik**2+rjk**2)/rc**2)*fc(rij, rc)*fc(rik, rc)*fc(rjk, rc)
                    else:
                        result+=0
    return result





def radial(i, atoms, eta, rs , rc, distances):
    """ An radial feature function as per Behler and Parrinello. Returns a single radial feature, in form of a number.

    Parameters
    ----------
    atoms: A structure object containing a list of atoms objects.
    eta: Is chosen between 0.001 and 2 in the cited paper.
    rs: HyperParameter. Unknown for now
    rc: Cut-off radius. Interactions between atoms outside of this radius from the atom in question, will be neglected
    distances: Matrix of distances between atoms.

    """
    # For storing the resulting feature
    result = 0
    # Distances between atoms i,j and k
    rij = 0
    number_of_atoms = atoms.get_global_number_of_atoms()
    for j in range(0, number_of_atoms):
        if(i != j):
            rij = distances[i,j]
            if(rij<rc):
                # The following should have if-checks implemented for the sake of efficiancy,
                # but this is form in the paper and it will do for now.
                result += np.exp(-eta * (rij ** 2 - rs ** 2) / rc ** 2) * fc(rij, rc)
            else:
                result += 0
    return result
