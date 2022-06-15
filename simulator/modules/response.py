from abc import ABC
from collections.abc import Iterable

import pandas as pd
import numpy as np

from deeptables.models import deeptable


def sample_response(
    user_matrix : pd.DataFrame,
    item_matrix : pd.DataFrame,
    theta: float
):
    # """
    # Selecting responses according to the Bernoulli scheme

    # :param response_frame: responces data frame in form of
    #     (user_id, item_id)
    # :type response_frame: pd.DataFrame
    # :param theta: probability parameter
    # :type theta: float
    # :return: subset of the original dataframe, selected according
    #     to the Bernoulli scheme
    # :rtype: pd.DataFrame
    # """

    if theta > 1.0 or theta < 0.0:
        raise ValueError('theta must be in range of [0;1]')

    if len(user_matrix) != len(item_matrix):
        raise ValueError('user and item matrices length mismatch')
    
    probability = np.random.rand(len(user_matrix))
    
    return (probability >= theta).astype('int')


class ResponseHeuristic(ABC):
    def __init__(
        self
    ):
        """
        Base class for heuristic response

        """
        raise NotImplementedError('')

    def predict(
        self,
        user_matrix : pd.DataFrame,
        item_matrix : pd.DataFrame
    ) -> np.ndarray:
        """
        Returns array of predicted responses for each user-item pair
        based on some heuristic. user_matrix and item_matrix are matrices
        of splitted user-item matrix, where (user_matrix[i], item_matrix[i])
        represents i-th user-item pair

        :param user_matrix: Left part (users part) of splitted user-item
            matrix
        :type user_matrix: pd.DataFrame
        :param item_matrix: Right part (items part) of splitted user-item
            matrix
        :type item_matrix: pd.DataFrame
        :return: 1-d array of predicted responses
        :rtype: np.ndarray
        """

        raise NotImplementedError()


class ResponseFunction(ABC):
    def __init__(
        self,
        models : Iterable
    ):
        """
        Base class for custom response function

        :param models: Array of models that predicts response value
        :type models: Iterable
        """

        self.models = models

    def __call__(
        self,
        user_matrix : pd.DataFrame,
        item_matrix : pd.DataFrame,
        weights : Iterable,
    ) -> np.ndarray:
        """
        Returns array of predicted responses for each user-item pair
        based on models ensemble (weighted sum). user_matrix and
        item_matrix are matrices of splitted user-item matrix, where
        (user_matrix[i], item_matrix[i]) represents i-th user-item pair

        :param user_matrix: Left part (users part) of splitted user-item
            matrix
        :type user_matrix: pd.DataFrame
        :param item_matrix: Right part (items part) of splitted user-item
            matrix
        :type item_matrix: pd.DataFrame
        :param weights: weight for each of the model used in response
            calculation
        :type weights: Iterable
        :return: 1-d array of predicted responses
        :rtype: np.ndarray
        """

        raise NotImplementedError()


class NoiseResponse(ABC):
    def __init__(
        self,
        mu : float = 1.0,
        sigma : float = 1.0,
        clip_negative : bool = True
    ):
        """
        Model that predicts noise response from normal distribution
        (mu, sigma)
        """

        self._mu = mu
        self._sigma = sigma
        self._clip_negative = clip_negative

    def predict(
        self,
        num_responses : int
    ):
        resps = np.random.normal(1, 1, size=num_responses)

        if self._clip_negative:
            resps = np.clip(resps, 0.0, None)

        return resps


class ConstantResponseHeuristic(ResponseHeuristic):
    def __init__(
        self,
        value : float = 0.0
    ):
        """
        Model for constant response prediction
        """
        self._value = value

    def predict(
        self,
        user_matrix : pd.DataFrame,
        item_matrix : pd.DataFrame
    ):
        if len(user_matrix) != len(item_matrix):
            raise ValueError('Lenght of user and item matrices does not match')

        return np.zeros(len(user_matrix)) + self._value


class ResponseFunctionSim(ResponseFunction):
    def __init__(
        self,
        models : Iterable
    ):
        """
        Model with three predction components: noise, heuristic, deepfm.
        The aforementioned order should be preserved
        """

        if not isinstance(models[0], NoiseResponse):
            raise ValueError('First model must be instance of NoiseResponse')
            
        if not isinstance(models[1], ResponseHeuristic):
            raise ValueError('Second model must be instance of ResponseHeuristic')

        if not isinstance(models[2], deeptable.DeepTable):
            raise ValueError('Third model must be instance of DeepTable')

        super().__init__(models)

    def __call__(
        self,
        user_matrix : pd.DataFrame,
        item_matrix : pd.DataFrame,
        weights : Iterable,
    ):
        if len(self.models) != len(weights):
            raise ValueError('models and weights length mismatch')

        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)

        weights_sum = weights.sum()
        weights = weights / weights_sum

        user_item_matrix = pd.concat(
            (user_matrix.reset_index(), item_matrix.reset_index()),
            axis=1
        ).drop(columns=['index'])

        resps = np.zeros((len(user_item_matrix), len(self.models)))

        resps[:, 0] = self.models[0].predict(len(user_item_matrix))
        resps[:, 1] = self.models[1].predict(user_matrix, item_matrix)
        resps[:, 2] = self.models[2].predict(user_item_matrix)

        return (resps * weights).sum(axis=1)
