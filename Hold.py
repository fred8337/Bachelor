model = pickle.load(open("model.pkl", "rb"))
energy_labels = np.load("Energy_Labels.npy")
sorted_labels = [e[0] for e in energy_labels]
rattle_ranges = np.linspace(0, rattle_range, len(sorted_labels))
np.dict(zip(sorted_labels, rattle_ranges))
ksis = [1, 2, 4]
lambs = [1, -1]
etas = [0.05, 2, 4, 8, 20, 40, 80]
angularEtas = [0.005]
rss = [0]
rc = 3
Energyclassifer = Energyclassifer()
Energyclassifer.set_clustering_model(model)
Energyclassifer.setHyperParameters(ksis, lambs, etas, angularEtas, rss, rc)
features = Energyclassifer.features(a)
predictions = Energyclassifer.classifier.predict(features)
rattle_ranges_to_use = np.dict(zip(np.arange(0, Natoms)), [rattle_ranges[e] for e in predictions])