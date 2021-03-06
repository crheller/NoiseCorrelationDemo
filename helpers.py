import numpy as np
from sklearn.decomposition import PCA

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def generate_data(u, cov):
    x = np.random.multivariate_normal(u, cov, (1000))
    return x

def get_null_axis(x, y):
    '''
    Return unit vector from centroid of x to centroid of y
    x and y must be of dimensions: O x N, where O are observations and N are
    number of dimensions. For example, this could be trials x neurons
    '''
    x, y = normalize_variance(x, y)
    ux = x.mean(axis=0)
    uy = y.mean(axis=0)

    d = ux - uy

    return unit_vector(d)

def get_rsc(x, y):
    """
    get noise correlation strength (stimulus independent correlations)
    """
    ux = x.mean(axis=0)
    uy = y.mean(axis=0)

    x = x - ux
    y = y - uy

    X = np.concatenate((x, y), axis=0)

    return round(np.corrcoef(X.T)[0, 1], 3)

def get_noise_PC(x, y):
    ux = x.mean(axis=0)
    uy = y.mean(axis=0)

    x = x - ux
    y = y - uy

    X = np.concatenate((x, y), axis=0)

    pca = PCA()
    pca.fit(X)
    noise_axis = pca.components_[0]

    return noise_axis

def get_LDA_axis(x, y):
    '''
    x and y must be of dimensions: O x N, where O are observations and N are
    number of dimensions. For example, this could be trials x neurons
    '''
    x, y = normalize_variance(x, y)
    n_classes = 2
    n_dims = x.shape[-1]
    if x.shape[0] != y.shape[0]:
        if x.shape[0] < y.shape[0]:
            n = x.shape[0]
            idx = np.random.choice(np.arange(0, y.shape[0]), n, replace=False)
            y = y[idx, :]
        else:
            n = y.shape[0]
            idx = np.random.choice(np.arange(0, x.shape[0]), n, replace=False)
            x = x[idx, :]

    X = np.concatenate((x[np.newaxis, :, :], y[np.newaxis, :, :]), axis=0)

    # find best axis using LDA
    # STEP 1: compute mean vectors for each category
    mean_vectors = []
    for cl in range(0, n_classes):
        mean_vectors.append(np.mean(X[cl], axis=0))

    # STEP 2.1: Compute within class scatter matrix
    n_units = X.shape[-1]
    S_W = np.zeros((n_units, n_units))
    n_observations = X.shape[1]
    for cl, mv in zip(range(0, n_classes), mean_vectors):
        class_sc_mat = np.zeros((n_units, n_units))
        for r in range(0, n_observations):
            row, mv = X[cl, r, :].reshape(n_units, 1), mv.reshape(n_units, 1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat

    # STEP 2.2: Compute between class scatter matrix
    overall_mean = np.mean(X, axis=0).mean(axis=0)[:, np.newaxis]
    S_B = np.zeros((n_units, n_units))
    X_fl = X.reshape(-1, n_units)
    for i in range(X_fl.shape[0]):
        S_B += (X_fl[i, :].reshape(n_units, 1) - overall_mean).dot((X_fl[i, :].reshape(n_units, 1) - overall_mean).T)

    # STEP 3: Solve the generalized eigenvalue problem for the matrix S_W(-1) S_B
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    if np.iscomplexobj(eig_vecs):
        eig_vals, eig_vecs = np.linalg.eigh(np.linalg.pinv(S_W).dot(S_B))
    #if np.any(eig_vals<0):
    #    import pdb; pdb.set_trace()
    # STEP 4: Sort eigenvectors and find the best axis (number of nonzero eigenvalues
    # will be at most number of categories - 1)
    sorted_idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, sorted_idx]
    eig_vals = eig_vals[sorted_idx]

    # STEP 5: Project data onto the top axis
    discrimination_axis = eig_vecs[:, 0]

    return discrimination_axis


def normalize_variance(x, y):
    """
    Normalize such that denominator of dprime calculation goes
    to 1
    """
    s1 = x.copy()
    s2 = y.copy()

    norm_factor = np.sqrt(0.5 * (np.var(s1) + np.var(s2)))

    if (np.std(s1) != 0) | (np.std(s2) != 0):
        if len(x.shape) == 1:
            norm_factor = np.sqrt(0.5 * (np.var(s1) + np.var(s2)))
            s1 = s1 / norm_factor
            s2 = s2 / norm_factor
        else:
            for n in range(0, x.shape[-1]):
                if (np.std(s1[:, n]) != 0) | (np.std(s2[:, n]) != 0):
                    norm_factor = np.sqrt(0.5 * (np.var(s1[:, n]) + np.var(s2[:, n])))
                    s1[:, n] = s1[:, n] / norm_factor
                    s2[:, n] = s2[:, n] / norm_factor
    
    return s1, s2
'''
def normalize_variance(x, y):
    """
    Normalize by the variance of each stimulus, for each neurons,
    so that var(neuron1) = var(neuron2)
    """
    s1 = x.copy()
    s2 = y.copy()
    if len(x.shape) == 1:
        s1 = s1 / np.std(s1)
        s2 = s2 / np.std(s2)
    else:
        for n in range(0, x.shape[-1]):
            s1[:, n] = s1[:, n] / np.std(s1[:, n])
            s2[:, n] = s2[:, n] / np.std(s2[:, n])
    
    return s1, s2
'''


def get_dprime(x, y):
    if len(x.shape) == 1:
        dprime = abs(x.mean() - y.mean()) / (np.sqrt((np.var(x) + np.var(y)) / 2))
        return dprime
    else:
        ax1 = get_LDA_axis(x, y)
        ax2 = get_null_axis(x, y)
        s1 = np.matmul(x, ax1)
        s2 = np.matmul(y, ax1)
        dprime_LDA = abs(s1.mean() - s2.mean()) / (np.sqrt((np.var(s1) + np.var(s2)) / 2))

        s1 = np.matmul(x, ax2)
        s2 = np.matmul(y, ax2)
        dprime_NULL = abs(s1.mean() - s2.mean()) / (np.sqrt((np.var(s1) + np.var(s2)) / 2))

        return dprime_LDA, dprime_NULL

def get_table_values(_x, _y):
    """
    Return a list of table values for the given dataset
    """
    x, y = normalize_variance(_x, _y)
    dprime_pop_LDA, dprime_pop_NULL = get_dprime(x, y)
    dprime_ind = np.sqrt(get_dprime(x[:, 0], y[:, 0])**2 + get_dprime(x[:, 1], y[:, 1])**2)
    ratio_LDA = np.log(dprime_pop_LDA / dprime_ind)
    ratio_NULL = np.log(dprime_pop_NULL / dprime_ind)
    noise = get_noise_PC(x, y)
    NULL = get_null_axis(x, y)
    LDA = get_LDA_axis(x, y)
    null_vs_noise = abs(np.dot(NULL, noise))
    LDA_vs_noise = abs(np.dot(LDA, noise))
    rsc = get_rsc(x, y)

    return [dprime_pop_LDA, dprime_pop_NULL, dprime_ind, ratio_LDA, ratio_NULL, null_vs_noise, LDA_vs_noise, rsc]
