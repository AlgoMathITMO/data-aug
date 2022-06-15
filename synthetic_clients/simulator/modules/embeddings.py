from abc import ABC

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import SpectralEmbedding

import numpy as np
import pandas as pd
import torch


class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        input_dim : int,
        hidden_dim : int,
        emb_dim : int,
        device='cpu'
    ):
        super().__init__()

        self.dense1 = torch.nn.Linear(input_dim, hidden_dim)
        self.dense2 = torch.nn.Linear(hidden_dim, emb_dim)
        self.dense3 = torch.nn.Linear(emb_dim, hidden_dim)
        self.dense4 = torch.nn.Linear(hidden_dim, input_dim)

        self.device = torch.device(device)

    def forward(
        self,
        X
    ):
        X = torch.nn.functional.relu(self.dense1(X))
        X = torch.nn.functional.relu(self.dense2(X))
        X = torch.nn.functional.relu(self.dense3(X))
        X = self.dense4(X)

        return X

    def fit(
        self,
        X,
        num_epochs : int = 100,
    ):
        if isinstance(X, pd.DataFrame):
            X = X.values

        train_loader = torch.utils.data.DataLoader(
            X, batch_size=128, shuffle=True, num_workers=4
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        loss = torch.nn.MSELoss()

        for _ in range(num_epochs):
            total_loss = 0
            for X_batch in train_loader:
                X_batch = X_batch.view(-1, X.shape[1]).to(self.device).float()

                optimizer.zero_grad()
                out = self.forward(X_batch)
                train_loss = loss(out, X_batch)
                train_loss.backward()
                optimizer.step()

                total_loss += train_loss.item()


    def transform(
        self,
        X
    ):
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = torch.tensor(X).to(self.device).float()

        with torch.no_grad():
            X = torch.nn.functional.relu(self.dense1(X))
            X = self.dense2(X)

        return X.to(torch.device('cpu')).numpy()


class EmbeddingCreator(ABC):
    def __init__(
        self,
        pca_params={'n_components' : 0.8},
        spectral_params={'n_components' : 80},
        autoencoder_params={}
    ):
        """
        Embeddings transform class with a bunch of methods to create
        vector embeddings from original data

        :param pca_params: parameters of sklearn.decomposition.PCA
            class to define the PCA model, defaults to
            {'n_components' : 0.8}
        :type pca_params: dict, optional
        :param spectral_params: parameters of sklearn.manifold.SpectralEmbedding
            class to define the spectral embeddings model, defaults
            to {'n_components' : 80}
        :type spectral_params: dict, optional
        :param autoencoder_params: parameters of Autoencoder class
            to define the neural net model. Necessary parameters include:
            input_dim, hidden_dim, emb_dim, defaults to {}
        :type autoencoder_params: dict, optional
        """

        self.pca = PCA(**pca_params)
        self.se = SpectralEmbedding(**spectral_params)
        self.ae = Autoencoder(**autoencoder_params)

        self.scaler = StandardScaler()

    def fit(
        self,
        X
    ):
        """
        Fits the necessary models for transforming data

        :param X: Data with shape (n_samples, n_features)
        :type X: numpy.ndarray or pandas.DataFrame
        """

        self.scaler.fit(X)

        X = self.scaler.transform(X)

        self.pca.fit(X)
        self.ae.fit(X)

    def transform(
        self,
        X,
        method='pca'
    ) -> np.ndarray:
        """
        Creates embeddings from input data

        :param X: Input data with shape (n_samples, n_features)
        :type X: numpy.ndarray or pandas.DataFrame
        :param method: Embedding method to use. Possible values are:
            'pca', 'spectral', 'autoencoder', defaults to 'pca'
        :type method: str, optional
        :return: Transformed data
        :rtype: numpy.ndarray
        """

        method_d = {
            'pca' : self.pca,
            'autoencoder' : self.ae
        }

        if method == 'spectral':
            return self.se.fit_transform(X)
        else:
            return method_d[method].transform(X)
