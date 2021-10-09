import umap
from sklearn.datasets import load_digits

if __name__ == "__main__":
    digits = load_digits()
    embedding = umap.UMAP().fit_transform(digits.data)
