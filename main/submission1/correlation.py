import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
from synapseclient import Synapse

def gdsc_ccle(drug):
    '''
    Calculates Spearman correlation between IC50 values from GDSC datasets and 
    baseline RNAseq profiles from CCLE for a given drug and returns a dataframe 
    of rho and pval
    Parameters:
    -----------
    drug: str
        input drug
    Returns:
    -------
    dfcorr: pandas dataframe
        columns: rho, pval
        rows: gene names

    '''
    warnings.filterwarnings('ignore')
    syn = Synapse()
    username, password = open('/Users/mauliknariya/synapse_login.txt').\
                        read().splitlines()
    syn.login(username, password, silent=True)
    dfgdsc = pd.read_csv(syn.get('syn22051024').path)
    dfgdsc['IC50'] = dfgdsc['LN_IC50'].apply(lambda x: np.exp(x))
    dfccle = pd.read_csv(syn.get('syn21822697').path, index_col=0)
    dfccle = dfccle.groupby(dfccle.index).mean()
    dfic50 = dfgdsc[dfgdsc.DRUG_NAME==drug][['CELL_LINE_NAME', 'IC50']]
    dfic50.index = dfic50.CELL_LINE_NAME
    dfic50 = dfic50.groupby(by=dfic50.index).mean()
    dfc = pd.concat([dfic50, dfccle.T], axis=1, sort=True).dropna() 
    dfcorr = pd.DataFrame(index=dfccle.index)
    rho_pval = [np.asarray(spearmanr(dfc['IC50'], dfc.loc[:, gene]))
                for gene in dfccle.index]
    dfcorr = pd.DataFrame(rho_pval, index=dfccle.index, columns=['rho', 'pval']) 
    return dfcorr


    
