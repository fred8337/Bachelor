from ase import atoms
from ase.io import read, write, iread
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase.data.colors import jmol_colors
import matplotlib.colors as color
import matplotlib.cm as cmx
from ase.calculators.abinit import Abinit


moleculesIterator = iread("runs0/run0/structures.traj")
index = 419

for i, atom in enumerate(moleculesIterator, 0):
    if i == index:
        # atom.set_calculator(Abinit(...))
        energies = atom.get_forces()
        energies = [np.linalg.norm(e) for e in energies]
        cmap = plt.get_cmap("inferno")
        cNorm = color.Normalize(vmin=np.amin(energies), vmax=np.amax(energies))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        colors = [scalarMap.to_rgba(e) for e in energies]
        fig, ax = plt.subplots()
        # test = atom.get_tags()
        print(energies)
        # print(test)
        # colors = jmol_colors[0:len(test)]
        # print(colors)
        plot_atoms(atom, ax, radii=0.3, rotation=('90x, 90y, 0z'), colors= colors)
        plt.title(str(index))
        plt.show()

