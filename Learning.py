import numpy as np
from ase import Atoms
from sklearn.cluster import KMeans


class EnergyClassifier:
    classifier = None
    ksis = None
    lambs = None
    etas = None
    angularEtas = None
    rss = None
    rc = None
    def __init__(self, classifier=None):
        self.classifier = classifier

    def features(self, atoms, ksis=None, lambs=None, etas=None, angularEtas=None, rss=None, rc=None):
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
        if rc is None:
            rc = self.rc
        else:
            self.rc = rc
        # A matrix of all distances
        distances = atoms.get_all_distances()
        # For storing the featurevectors
        featureVectors = []
        for i, atom in enumerate(atoms, 0):
            featureVectors.append(self.featureVector(i, atoms, ksis, lambs, etas, angularEtas, rss, rc, distances))
        return featureVectors

    def setHyperParameters(self, ksis=None, lambs=None, etas=None, angularEtas=None, rss=None, rc=None):
        self.ksis = ksis
        self.lambs = lambs
        self.etas = etas
        self.angularEtas = angularEtas
        self.rss = rss
        self.rc = rc

    def featureVector(self, i, atoms, ksis, lambs, etas, angularEtas, rss, rc, distances):
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
        return result

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
            if(i != j):
                for k in range(0, number_of_atoms):
                    if(k != i and k != j):
                        rij = distances[i, j]
                        rik = distances[i, k]
                        rjk = distances[j, k]
                        if(rij<rc and rik<rc and rjk<rc):
                            phiijk = getAngle(rij, rik, rjk)
                            # phiijk = atoms.get
                            result += (2**(1-ksi))*(1-lamb*np.cos(phiijk))**ksi*np.exp(-eta*(rij**2+rik**2+rjk**2)/rc**2)*self.fc(rij, rc)*self.fc(rik, rc)*self.fc(rjk, rc)
                        else:
                            result += 0
        return result

    def radial(self, i, atoms, eta, rs , rc, distances):
        """ A radial feature function as per Behler and Parrinello. Returns a single radial feature, in form of a number.

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
                    # The following should have if-checks implemented for the sake of efficiency,
                    # but this is form in the paper and it will do for now.
                    result += np.exp(-eta * (rij ** 2 - rs ** 2) / rc ** 2) * self.fc(rij, rc)
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
        features = self.features(structure)
        predictions = self.classifier.predict(features)
        result = []
        for label in self.classifier.labels_:
            result.append(np.sum(np.where(predictions == label, 1, 0)))
        return result
