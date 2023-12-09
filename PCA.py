import numpy as np
import torch
from Debug import debug


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # Centering the data
        self.mean = torch.mean(X, 0)
        X = X - self.mean

        # Computing the covariance matrix
        cov_matrix = torch.matmul(X.T, X) / (X.size(0) - 1)

        # Computing eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # Sorting eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort(descending=True)
        eigenvectors = eigenvectors[:, idx]

        # Selecting the top n_components
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X = X - self.mean
        return torch.matmul(X, self.components)

    def inverse_transform(self, X):
        return torch.matmul(X, self.components.T) + self.mean

    def error(self, input):
        coeffs = self.transform(input)
        output = self.inverse_transform(coeffs)
        error = torch.norm(output - input)
        print(f"input has norm {torch.norm(input):.6f} -> {len(coeffs)} coefficients -> error={error:.6f}")
        return error


if False:
    def randWave(length):
        a = np.random.uniform()
        b = np.random.uniform()
        c = np.random.uniform()
        
        t = t = np.linspace(0, 2*np.pi, length)
        signal = np.sin(a*t + b*np.sin(c * t))
        return torch.tensor(signal)
        
    length = 256
    w = randWave(length)
    debug("wave", w)

    signals = torch.stack([randWave(length) for _ in range(100)])
    debug("signals", signals)


    pca = PCA(15)
    pca.fit(signals)

    for signal in signals:
        error = pca.error(signal)


from MakeSTFTs import *

stfts, names = load_STFTs()



stfts = torch.stack([adjust_stft_length(torch.tensor(stft), sequence_length) for stft in stfts])
debug("stfts", stfts)

flattened = stfts.reshape(stfts.size(0), -1)
debug("flattened", flattened)
pca = PCA(20)
pca.fit(flattened)


