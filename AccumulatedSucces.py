from ase import atoms
from ase.io import read, write, iread
import matplotlib.pyplot as plt
import numpy as np
import Learning
import pickle
from sklearn.cluster import KMeans
from ase.visualize.plot import plot_atoms
from ase.data.colors import jmol_colors

atomsite = iread("runs0/run0/structures.traj")
energies = []
dataSet = []
clusterCounts = []

energyClassifier = Learning.EnergyClassifier()


ksis = [1, 2, 4]
lambs = [1, -1]
etas = [0.05, 2, 4, 8, 20, 40, 80]
angularEtas = [0.005]
rss = [0]
rc = 11.9
atomicLambda = 1
energyClassifier.setHyperParameters(ksis, lambs, etas, angularEtas, rss, atomicLambda, rc)
# for name in range(10):
#     for i, atom in enumerate(iread("BestProb/"+"run"+str(name)+"/structures.traj"), 0):
#         if i%10 ==0:
#             energies.append(atom.get_total_energy())
#             point, _ = energyClassifier.features(atom)
#             dataSet.append(point) # Denne linje tager MEGET lang
#             print(i)
#         if i>400:
#             break
# dataSet = np.array(dataSet)
# np.save("DataSet", dataSet)
data = np.load("DataSet.npy")
trainingData = data.reshape(-1, 16)
print(data)
kmeans = energyClassifier.trainModel(trainingData, 20)
pickle.dump(kmeans, open("save.pkl", "wb"))
kmeans = pickle.load(open("save.pkl", "rb"))
energyClassifier.set_clustering_model(kmeans)
labels = kmeans.labels_
print(labels)

atomS = 0
for i, atom in enumerate(atomsite, 0):
    # if i <= 401:
    #     # energies.append(atom.get_total_energy())
    #     dav = energyClassifier.featureVectors_to_clusterCountVector(data[i])
    #     clusterCounts.append(dav)
    #     # print(i)
    if i == 418:
        atomS = atom
for i in range(len(data)):
    clusterCounts.append(energyClassifier.featureVectors_to_clusterCountVector(data[i]))
# np.save("Energies", energies)
energies = np.load("Energies.npy")
print(energyClassifier.structure_to_clusterCountVector(atomS))
atomFeatures, _ = energyClassifier.features(atomS, ksis, lambs, etas, angularEtas, rss, rc)
print("labels = "+str(kmeans.predict(atomFeatures)))
energyLabels = energyClassifier.get_energy_labels(clusterCountVectors=clusterCounts, energies=energies, lamb_for_labels=1.0)
space = np.linspace(0, len(energyLabels), len(energyLabels))
plt.plot(space, [e[1] for e in energyLabels])
plt.show()
print(energyLabels)
np.save("Energy_Labels", energyLabels)
colors = [jmol_colors[kmeans.predict(np.array(e).reshape(1, -1))].flatten() for e in atomFeatures]
fig, ax = plt.subplots()
plot_atoms(atomS, ax, radii=0.3, rotation=('45x, 45y, 45z'), colors=colors)
plt.show()
plot_atoms(atomS, ax, radii=0.3, rotation=('90x, 90y, 0z'), colors=colors)
plt.show()
# space = np.linspace(0, len(energies), len(energies))
# emin = np.amin(energies)
# plt.plot(space, [e - emin for e in energies])
# print(emin)
# plt.show()

# ksis = [1, 2, 4]
# lambs = [1, -1]
# etas = [0.05, 2, 4, 8, 20, 40, 80]
# rss = [0]
# rc = 1
# print(features())

def accSuccess(runs=30, directory="runs0", tolerance=5): #Max spaghetti
    convergenceIndecies = []
    lengths = []
    for i in range(0, runs):
        try:
            structures = iread(directory+"/run"+str(i)+"/structures.traj")
            energies = [structure.get_total_energy() for structure in structures]
            lengths.append(len(energies))
        except:
            continue
        emin = np.amin(energies)
        relativeEnergies = [e-emin for e in energies]
        # convergenceEnergy = np.amin(relativeEnergies) #Den er naturligvis 0 for den valgte reference
        convergence = np.where(np.array(relativeEnergies) < tolerance, 1, 0) #Skal have fundet en bedre reference, men denne virker i tilfælde, hvor problemet er garanteret konvergens. Eventuelt brug Emin fra baseCase som reference i CrapGrænsen.
        convergenceIndex=np.nonzero(convergence)
        convergenceIndecies.append(convergenceIndex[0][0])

    space = np.linspace(0, np.amin(lengths)-1, np.amin(lengths))

    accSucces = []
    a = 0
    for index in space:
        for convindex in convergenceIndecies:
            if convindex == int(index):
                a += 1
        accSucces.append(a/len(convergenceIndecies))

    plt.plot(space, accSucces, color="black")
    plt.xlabel("Iterations")
    plt.ylabel("Accumulated Succes (%)")
    plt.show()


# accSuccess()