import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch, random, numpy as np, pandas as pd
import itertools
from math import floor, ceil
# ———————————————————————————————————————
def getIVP(cov, **kargs):
    """
    Compute the inverse-variance portfolio
    :param cov: Covariance matrix
    :param kargs:
    :return:
    """
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getICOVP(cov):
    """
    Compute the inverse-covariance portfolio
    :param cov:
    :return:
    """
    icovp = np.linalg.inv(cov)
    icovp /= icovp.sum()

    return np.dot(icovp, np.ones(icovp.shape[0]))

# ———————————————————————————————————————
def getClusterVar(cov, cItems):


    # Compute variance per cluster
    cov_ = cov.loc[cItems, cItems]  # matrix slice
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar


# ———————————————————————————————————————
# ———————————————————————————————————————
def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index;
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()
# ———————————————————————————————————————
def getRecBipart(cov, sortIx):


    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, floor(len(i) / 2)), \
                                                      (floor(len(i) / 2), len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w


# ———————————————————————————————————————
def correlDist(corr):


    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.) ** .5  # distance matrix
    return dist

def euclideanDist(dist_mat):
    dist = dist_mat.values
    result = np.zeros_like(dist)
    for t in itertools.product(range(dist.shape[0]), range(dist.shape[1])):
        i,j = t
        result[i,j] = np.sum((dist[:,i] - dist[:,j])**2)**0.5
    return pd.DataFrame(result, columns=dist_mat.columns, index=dist_mat.index)

def portfolio_variance(portfolio, returns):
    portfolio_returns = (portfolio * returns).sum(axis=1)

    return portfolio_returns.std(axis=0)



# ———————————————————————————————————————
def plotCorrMatrix(path, corr, labels=None):


    # Heatmap of the correlation matrix
    if labels is None: labels = []
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5, corr.shape[0] + .5), labels)
    mpl.xticks(np.arange(.5, corr.shape[0] + .5), labels)
    mpl.savefig(path)
    mpl.clf();
    mpl.close()  # reset pylab
    return


# ———————————————————————————————————————
def generateData(nObs, size0, size1, sigma1):

    # Time series of correlated variables
    # 1) generating some uncorrelated data
    np.random.seed(seed=12345)
    random.seed(12345)
    x = np.random.normal(0, 1, size=(nObs, size0))  # each row is a variable
    # 2) creating correlation between the variables
    cols = [random.randint(0, size0-1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1] + 1))

    return x, cols

def rolling_cov(returns, lookback=20, cov=True):
    """

    :param returns: N obs by P assets
    :param lookback: lookback period
    :return: an N-L by P by P matrix of historical covariances
    """
    (N, P) = returns.shape
    out = np.zeros((N-lookback,P,P))
    for i in range(lookback, N):
        ret_slice = returns.iloc[i-lookback:i, :]
        slice = ret_slice.cov().values if cov else ret_slice.corr().values
        out[i-lookback, :, :] = slice

    return out

# ———————————————————————————————————————
def main():

    from get_data import df0 as x
    lookback=500
    x.fillna(0, inplace=True)
    ivar_df = pd.DataFrame().reindex_like(x)
    hrp_df = pd.DataFrame().reindex_like(x)
    icov_df = pd.DataFrame().reindex_like(x)
    T = x.shape[0] - 500
    for i in range(lookback, x.shape[0]):
        ret = x.iloc[i-lookback:i, :]
        cov, corr = ret.cov(), ret.corr()
        dist = correlDist(corr)
        eu_dist = euclideanDist(dist)
        link = sch.linkage(eu_dist, 'single')
        sortIx = getQuasiDiag(link)
        labelIx = corr.index[sortIx].tolist()  # recover labels
        hrp = pd.Series(getRecBipart(cov, labelIx))
        invVar = pd.Series(getIVP(cov), index=cov.index)
        invCov = pd.Series(getICOVP(cov), index=cov.index)
        hrp_df.iloc[i, :] = hrp
        ivar_df.iloc[i, :] = invVar
        icov_df.iloc[i, :] = invCov
        print(i / T)

    hrp_df.to_csv('hrp.csv')
    ivar_df.to_csv('ivar.csv')
    icov_df.to_csv('icov.csv')

    return hrp_df, ivar_df, icov_df
# ———————————————————————————————————————
if __name__ == '__main__':
    main()
    # from get_data import df0 as x
    # x.fillna(0, inplace=True)
    # wt = pd.read_csv('ivar.csv', index_col='datetime')
    # print(x.shape)
    # print(wt.shape)
    # print(portfolio_variance(wt, x))