import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def pca_loadings(df, cutoff_score=0.5):
    '''
    Selects "high variqance features" by looking at their correlations with the
    prinicpal components

    Parameters:
    -----------
    df: pandas DataFrame, shape=(n_features, n_samples)
        Input dataframe, features names as rows (also row indices), samples as columns
    cutoff_score: float default=0.5
        cut-off for the correlation between the feature and principal components
    Returns:
    --------
    features: list, type: str
        features with high correlation witht the principal components
    '''
    df = df.dropna()
    X = df.T.values
    n_components = min(df.shape[0], df.shape[1])
    pca = PCA(n_components)
    Xpca = pca.fit_transform(X)
    factor_loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
    dfload = pd.DataFrame(factor_loadings)
    dfload.index = df.index
    dfload.columns = ['pc%s'%(i+1) for i in range(dfload.shape[1])]
    dfload = dfload.applymap(lambda x: abs(x))
    features = []
    for col in dfload.columns:
        features.append(dfload[dfload[col]>cutoff_score].index.tolist())
    features = list(set([y for x in features for y in x]))
    return features
