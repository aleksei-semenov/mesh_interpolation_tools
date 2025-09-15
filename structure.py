import os

def print_tree(root, prefix=""):
    for item in sorted(os.listdir(root)):
        path = os.path.join(root, item)
        if os.path.isdir(path):
            print(f"{prefix}{item}/")
            print_tree(path, prefix + "    ")
        else:
            print(f"{prefix}{item}")

print_tree(".")