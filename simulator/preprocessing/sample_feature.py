import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.preprocessing

from typing import Tuple, List

class ClusterModel():
    
    """
    Parent class of mesolevel modeling methods
    
    Attributes:
    model -- clustering model
    default_parameters -- model default parameters
    input_parameters -- model call parameters
    
    """
    
    def __init__(self, **kwargs):
        self.model = None
        self.default_parameters = None
        self.input_parameters = None
        
        self.initialize_parameters()
        self.set_parameters(kwargs)
        self.model = self.model(**self.input_parameters)
    
    def initialize_parameters(self):
        return 0
    
    def set_parameters(self, parameters):
        initial_parameters = self.default_parameters
        call_parameters = {}
        
        assert set(parameters.keys()).issubset(set(initial_parameters.keys()))
        for param in initial_parameters.keys():
            if param in parameters:
                call_parameters[param] = parameters[param]
                
        self.input_parameters = call_parameters
    
    
    def fit(self, X):
        self.model.fit(X)
        
    def predict(self, X):
        return self.model.predict(X)
    
    
class AffinityModel(ClusterModel):
    
    def initialize_parameters(self):
        self.model = sklearn.cluster.AffinityPropagation
        self.default_parameters = {'damping':0.5, 'max_iter':200, 'convergence_iter':15, 'copy':True, 'preference':None, 
                                   'affinity':'euclidean', 'verbose':False, 'random_state':None}

        
class AgglomerativeModel(ClusterModel):
    
    def initialize_parameters(self):
        self.model = sklearn.cluster.AgglomerativeClustering
        self.default_parameters = {'n_clusters':2, 'affinity':'euclidean', 'memory':None, 'connectivity':None, 'compute_full_tree':'auto', 
                                   'linkage':'ward', 'distance_threshold':None, 'compute_distances':False}
    
    def fit(self, X):
        pass
    
    def predict(self, X):
        return self.model.fit_predict(X)
    
class KMeansModel(ClusterModel):
    
    def initialize_parameters(self):
        self.model = sklearn.cluster.KMeans
        self.default_parameters = {'n_clusters':8, 'init':'k-means++', 'n_init':10, 'max_iter':300, 'tol':0.0001, 'verbose':0, 'random_state':None, 'copy_x':True, 'algorithm':'lloyd'}
    
    
class QuantileModel(ClusterModel):
    
    def quantiles_clusters(self, X: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        
        quantiles_np = np.asarray(quantiles)
        assert len(quantiles_np.shape) == 1 or quantiles_np.shape[1] == 1
        quantiles_np = quantiles_np.reshape(-1)
        
        assert quantiles_np[np.where( (quantiles_np <= 1.) & (quantiles_np >= 0.))].shape == quantiles_np.shape
        assert len(set(quantiles_np)) == quantiles_np.shape[0]
        quantiles_np = np.asarray(sorted( set([0., *quantiles_np, 1.]) ))
        
        X_np = np.asarray(X)
        assert len(X_np.shape) == 1 or X_np.shape[1] == 1
        X_np = X_np.reshape(-1)
        
        quantiles_values = [*map(lambda x: np.quantile(X_np, x), quantiles_np)]
        clusters = np.empty_like(X_np)
        
        for cluster_num, i in enumerate(range(len(quantiles_values) - 1)):
            q_idx = np.where( (X_np >= quantiles_values[i]) & (X_np <= quantiles_values[i+1]) )
            clusters[q_idx] = cluster_num
            
        return clusters
    
    def __init__(self, **kwargs):
        self.model = None
        self.default_parameters = None
        self.input_parameters = None
        
        self.initialize_parameters()
        self.set_parameters(kwargs)
        
    def initialize_parameters(self):
        self.default_parameters = {'quantiles': [0., 0.25, 0.5, 0.75, 1.]}
    
    def fit(self, X):
        pass
    
    def predict(self, X):
        return self.quantiles_clusters(X, **self.input_parameters)
    
    
    
    
def sample_feature(input_frame: pd.DataFrame, 
                   features: List[str], 
                   model: ClusterModel, 
                   scaler: bool = True
) -> pd.DataFrame:
    
    """
    Selecting a subset of a dataframe for any feature

    Arguments:
    input_frame -- initial dataframe
    features -- single or several features name
    model -- clustering model
    scaler -- data scaler parameter
    
    Returns:
    y -- clasters labels for dataframe rows
    """
    
    features = np.reshape(np.asarray([features]), -1)
    assert set(features).issubset(input_frame.columns)
    
    X = np.asarray(input_frame.loc[:, features]) if features.shape[0] > 1 else np.asarray(input_frame.loc[:, features]).reshape(-1, 1)
    
    
    if scaler:
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        X = np.copy(X_scaled)

    model.fit(X)
    y = model.predict(X)
    
    return y



def netflix_sample_feature(input_frame: pd.DataFrame,
                           rating_frame: pd.DataFrame,
                           features: List[str] = ["rating_cnt", "rating_avg"],
                           model_params: List[float] = [0.25, 0.5, 0.75],
                           clusters_num: List[int] = [2, 1]
) -> pd.DataFrame:
    
    """
    Sampling a subset of users from frame "input_frame", according to the meso-level of feature distribution.
    
    Arguments:
    input_frame -- initial netflix dataframe.
    rating_frame -- rating dataframe.
    features -- a list of features for clustering.
    model_params -- parameters (quantiles) for QuantileModel.
    quantiles -- quantiles of features rating_cnt, rating_avg.
    
    Returns:
    df_frame -- a subset of users netflix.
    rating_out_frame -- a subset of users rating from netflix.
    """
    assert len(features) == len(clusters_num), "Set for EACH feature OWN cluster number"
    
    model = QuantileModel(quantiles=model_params)
    
    cluster_columns = []
    df_frame = input_frame.copy()
    
    for i in range(len(features)):
        df_frame["cluster_column_" + str(i)] = sample_feature(input_frame, features[i], model)
        cluster_columns.append("cluster_column_" + str(i))
        
    for i in range(len(features)):
        df_frame = df_frame[df_frame[cluster_columns[i]] == clusters_num[i]]
        
    df_frame = df_frame.drop(cluster_columns, axis = 1).reset_index(drop=True)
    rating_out_frame = rating_frame[rating_frame.user_Id.isin(df_frame.user_Id.unique())].reset_index(drop=True)
    
    cat_dict = pd.Series(df_frame.user_Id.astype("category").cat.codes.values, index=df_frame.user_Id).to_dict()
    df_frame.user_Id = df_frame.user_Id.apply(lambda x: cat_dict[x])
    rating_out_frame.user_Id = rating_out_frame.user_Id.apply(lambda x: cat_dict[x])
    
    return df_frame, rating_out_frame


def movielens_sample_feature(input_frame: pd.DataFrame,
                             rating_frame: pd.DataFrame,
                             features: List[str] = ["genre" + str(i) for i in range(19)],
                             n_clusters: int = 10,
                             clusters_num: List[int] = [2, 7, 5, 3]
) -> pd.DataFrame:
    
    """
    Sampling a subset of users from frame "input_frame", according to the meso-level of clusters.
    
    Arguments:
    input_frame -- initial movielens20 dataframe.
    rating_frame -- rating dataframe.
    features -- a list of features for clustering.
    n_clusters -- clusters count.
    clusters_num -- cluster numbers for sampling.
    
    Returns:
    df_frame -- a subset of users movielens20.
    rating_out_frame -- a subset of users rating from movielens20.
    """
    
    model = KMeans(n_clusters = n_clusters)
    
    df_frame = input_frame.copy()
    
    df_frame["cluster_column"] = sample_feature(input_frame, features, model)
    df_frame = df_frame[df_frame["cluster_column"].isin(clusters_num)]
    df_frame = df_frame.drop("cluster_column", axis = 1).reset_index(drop=True)
    rating_out_frame = rating_frame[rating_frame.userId.isin(df_frame.userId.unique())].reset_index(drop=True)
    
    cat_dict = pd.Series(df_frame.userId.astype("category").cat.codes.values, index=df_frame.userId).to_dict()
    df_frame.userId = df_frame.userId.apply(lambda x: cat_dict[x])
    rating_out_frame.userId = rating_out_frame.userId.apply(lambda x: cat_dict[x])
    
    return df_frame, rating_out_frame
