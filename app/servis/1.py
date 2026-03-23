import h5py

path = "flowchart_model.weights.h5"
with h5py.File(path, "r") as f:
    print("H5 keys:", list(f.keys()))
    f.visit(lambda name: print(name) if "kernel" in name or "bias" in name else None)