import numpy as np
from ase import Atoms
from sklearn.cluster import KMeans
import torch

class EnergyClassifier:
    classifier = None
    ksis = None
    lambs = None
    etas = None
    angularEtas = None
    rss = None
    rc = None
    atomicLambda = None
    clusters = None
    lambda_for_labels = None
    energy_labels = None
    def __init__(self, classifier=None):
        self.classifier = classifier

    def features(self, atoms, ksis=None, lambs=None, etas=None, angularEtas=None, rss=None, atomicLambda=None, rc=None, atomic=True):
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
        if ksis is None:
            ksis = self.ksis
        else:
            self.ksis = ksis
        if lambs is None:
            lambs = self.lambs
        else:
            self.lambs = lambs
        if etas is None:
            etas = self.etas
        else:
            self.etas = etas
        if angularEtas is None:
            angularEtas = self.angularEtas
        else:
            self.angularEtas = angularEtas
        if rss is None:
            rss = self.rss
        else:
            self.rss = rss
        if atomicLambda is None:
            atomicLambda = self.atomicLambda
        if rc is None:
            rc = self.rc
        else:
            self.rc = rc
        # A matrix of all distances
        distances = atoms.get_all_distances()
        number_of_atoms = len(atoms)
        number_of_features = len(ksis)*len(lambs)*len(angularEtas)+len(etas)*len(rss)
        aNumbers = atoms.get_atomic_numbers()
        atomic_numbers = dict(zip(np.arange(number_of_atoms), aNumbers))
        if atomic:
            number_of_features += len(list(np.unique(aNumbers)))+1
        # For storing the featurevectors
        featureVectors = np.empty((number_of_atoms, number_of_features))
        for i, atom in enumerate(atoms, 0):
            featureVector = self.featureVector(i, atoms, ksis, lambs, etas, angularEtas, rss, atomicLambda, rc, distances, atomic_numbers)
            featureVectors[i] = featureVector
        return featureVectors, number_of_features

    def setHyperParameters(self, ksis=None, lambs=None, etas=None, angularEtas=None, rss=None, atomicLambda=None, rc=None, clusters=None):
        if ksis is not None:
            self.ksis = ksis
        if lambs is not None:
            self.lambs = lambs
        if etas is not None:
            self.etas = etas
        if angularEtas is not None:
            self.angularEtas = angularEtas
        if rss is not None:
            self.rss = rss
        if rc is not None:
            self.rc = rc
        if atomicLambda is not None:
            self.atomicLambda = atomicLambda
        if clusters is not None:
            self.clusters = clusters

    def featureVector(self, i, atoms, ksis, lambs, etas, angularEtas, rss, atomiclambda, rc, distances, atomic_numbers):
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
                    result.append(self.angular(i, atoms, ksi, lamb, eta, rc, distances))
        for eta in etas:
            for rs in rss:
                result.append(self.radial(i, atoms, eta, rs, rc, distances))
        result.append(atomic_numbers[i])
        numbers_to_iterate = np.unique(list(atomic_numbers.values()))
        for atomic_number in numbers_to_iterate:
            result.append(self.atomic(i, atomic_number, atoms, atomiclambda, rc, distances, atomic_numbers))
        return np.array(result).reshape(1, -1)

    def fc(self, r, rc):
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

    def angular(self, i, atoms, ksi, lamb, eta, rc, distances):
        """ An angular feature function as per Behler and Parrinello. Returns a single angular feature, in the form of a number.

        Parameters
        ----------
        i: Index of the current atom
        atoms: A structure object containing a list of atoms objects.
        ksi: HyperParameter, should be chosen as 1, 2, 4, 16, 64 ..
        lamb: HyperParameter, should be chosen as -1, 1.
        eta: HyperParameter. Is chosen between 0.001 and 2 in the cited paper.
        rc: Cut-off radius. Interactions between atoms outside of this radius from the atom in question, will be neglected
        distances: Matrix of distances between atoms.

        """

        def getAngle(ij, ik, jk):
            """ Returns the angle centered on atom i"""
            input = (ij**2+ik**2-jk**2)/(2*ij*ik)
            if input > 1:
                input = 0.999
            if input < -1:
                input = -0.999
            return np.arccos(input)

        def Behler(phiijk, rij, rik, rjk):
            return (2**(1-ksi))*(1-lamb*np.cos(phiijk))**ksi*np.exp(-eta*(rij**2+rik**2+rjk**2)/rc**2)*self.fc(rij, rc)*self.fc(rik, rc)*self.fc(rjk, rc)

        # For storing the resulting feature
        result = 0
        # Distances between atoms i,j and k
        rij = 0
        rik = 0
        rjk = 0
        # Angle between j and k, centered on i
        phiijk = 0
        # The following should have if-checks (instead of fc returning 0) implemented for the sake of efficiancy, but this is form in the paper and it will do for now.
        # (2**(1-ksi))*(1-lamb*np.cos(phijk))**ksi*np.exp(-eta*(rij**2+rik**2+rjk**2)/rc**2)*fc(rij, rc)*fc(rik, rc)*fc
        def get_distances(i, j, k):
            return (distances[i, j], distances[i, k], distances[j, k])

        def getAtoms():
            return len(atoms)
        number_of_atoms = getAtoms()
        for j in range(0, number_of_atoms):
            if(i != j):
                for k in range(0, number_of_atoms):
                    if(k != i and k != j):
                        rij, rik, rjk = get_distances(i, j, k)
                        if(rij<rc and rik<rc and rjk<rc):
                            phiijk = getAngle(rij, rik, rjk)
                            # phiijk = atoms.get
                            result += Behler(phiijk, rij, rik, rjk)
                        else:
                            result += 0
        return result

    def radial(self, i, atoms, eta, rs , rc, distances):
        """ A radial feature function as per Behler and Parrinello. Returns a single radial feature, in form of a number.

        Parameters
        ----------
        i: Index of the current atom
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
        number_of_atoms = len(atoms)
        for j in range(0, number_of_atoms):
            if(i != j):
                rij = distances[i,j]
                if(rij < rc):
                    # The following should have if-checks implemented for the sake of efficiency,
                    # but this is form in the paper and it will do for now.
                    result += np.exp(-eta * (rij ** 2 - rs ** 2) / rc ** 2) * self.fc(rij, rc)
                else:
                    result += 0
        return result

    def atomic(self, i, atomic_number, atoms, atomicLambda, rc, distances, atomic_numbers):
        """ An atomic feature function as per Behler and Parrinello. Returns an atomic feature vector.

                Parameters
                ----------
                i: Index of the current atom
                atomic_number: The atomic number of the current atom.
                atoms: A structure object containing a list of atoms objects.
                atomicLambda: Hyper Parameter
                rc: Cut-off radius. Interactions between atoms outside of this radius from the atom in question, will be neglected
                distances: Matrix of distances between atoms.

                """
        result = 0
        number_of_atoms = len(atoms)
        for j in range(number_of_atoms):
            if(atomic_number == atomic_numbers[j]):
                if(i!=j):
                    rij = distances[i, j]
                    if(rij < rc):
                        result += np.exp(-rij/atomicLambda)*self.fc(rij, rc)
                    else:
                        result += 0
        return result

    def trainModel(self, data, clusters):
        """ Trains a clustering algorithm, using K-means. Returns the kmeans object, and sets the classifier object.
            TO-DO: Train entire classifier
        Parameters
        ----------
        data: A list of feature vectors for clustering
        clusters: The number of clusters for the K-means to create

        """

        kmeans = KMeans(n_clusters=clusters).fit(data)
        self.clusters = clusters
        kmeans.fit(data)
        self.classifier = kmeans
        return kmeans

    def structure_to_clusterCountVector(self, structure):
        """ Returns a vector, counting the number of occurences for each cluster.

        Parameters
        ----------
        structure: ASE atoms object

        """
        if (self.ksis is None or self.lambs is None or self.etas is None or self.angularEtas is None or self.rss is None or self.rc is None):
            raise Exception("One or several hyperparameters are None, consider using setHyperParamaters for setting them.")
        features, _ = self.features(structure)
        predictions = self.classifier.predict(features)
        result = []
        for label in range(0, self.clusters):
            result.append(np.sum(np.where(predictions == label, 1, 0)))
        return result

    def featureVectors_to_clusterCountVector(self, featurevectors):
        """ Returns a vector, counting the number of occurences for each cluster.

        Parameters
        ----------
        featurevector: ASE atoms object

        """
        if (self.ksis is None or self.lambs is None or self.etas is None or self.angularEtas is None or self.rss is None or self.rc is None):
            raise Exception("One or several hyperparameters are None, consider using setHyperParamaters for setting them.")

        predictions = self.classifier.predict(featurevectors)
        result = []
        for label in range(0, self.clusters):
            result.append(np.sum(np.where(predictions == label, 1, 0)))
        return result

    def get_energy_labels(self, clusterCountVectors, energies, lamb_for_labels = None):
        """ Returns a vector of the energy associated with each cluster.

        Parameters
        ----------
        clusterCountVectors: a list of vectors, counting the number of atoms in each cluster for each structure with a known energy.
        energies: List of energies from structures
        lamb_for_labels: Hyperparameter.

        """
        X = torch.tensor(clusterCountVectors).double()
        energies = torch.tensor(energies).reshape(-1, 1).double()
        if lamb_for_labels is None:
            lamb_for_labels = self.lambda_for_labels
        XtX = torch.mm(X.t(), X)
        XtX = (XtX+lamb_for_labels*torch.tensor(np.eye(XtX.size()[0])).double())
        result = torch.mm(XtX.inverse(), X.t())
        # print(result.size())
        # print(energies.size())
        result = torch.mm(result, energies)
        result = sorted(enumerate(result), key=lambda x: x[1])
        result = [(i, e.item()) for (i, e) in result]
        self.energy_labels = result
        return result
    def set_clustering_model(self, model):
        self.classifier = model
        self.clusters = len(model.cluster_centers_)