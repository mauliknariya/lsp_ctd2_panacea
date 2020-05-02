import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import synapseclient


def pca_loadings(synid, cutoff_score=0.5):
    '''
    Parameters:
    -----------
    '''
    syn = synapseclient.Synapse()
    username, password = open('/Users/mauliknariya/synapse_login.txt')\
                        .read().splitlines()
    syn.login(username, password)
    df = pd.read_csv(open(syn.get(synid).path), index_col=0)
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
