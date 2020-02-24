import os
from ase import atoms
from ase.io import read, write, iread
import Learning

energyClassifier = Learning.EnergyClassifier()
energies = []
for root, dirs, files in os.walk("./runs0", topdown=True):
   # for name in files:
   #    print(os.path.join(root, name))
   for name in dirs:
      print(os.path.join(root, name))
for root, dirs, files in os.walk("./runs0", topdown=True):
    for dir in dirs:
        for i, atom in enumerate(iread(str(root)+str(dir)), 0):
            # if i%4 ==0:
            energies.append(atom.get_total_energy())
            point, _ = energyClassifier.features(atom)
            dataSet.append(point) # Denne linje tager MEGET lang
            print(i)
            if i>400:
                break
dataSet = np.array(dataSet)
np.save("DataSet", dataSet)