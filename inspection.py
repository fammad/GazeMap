from scipy.io import loadmat

data = loadmat("./dataset/Data/Normalized/p01/day02.mat", squeeze_me=True, struct_as_record=False)


def walk(name, obj, indent=0):
    pad = "  " * indent
    if hasattr(obj, "_fieldnames"):
        print(f"{pad}{name}/")
        for field in obj._fieldnames:
            walk(field, getattr(obj, field), indent + 1)
    else:
        print(f"{pad}{name}: shape={getattr(obj, 'shape', '-')} dtype={getattr(obj, 'dtype',type(obj).__name__)}")

for key, value in data.items():
    if key.startswith("__"):
        continue
    walk(key, value)