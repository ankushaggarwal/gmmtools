import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler
import pickle
from scipy.stats import multivariate_normal
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from scipy.stats import norm
import numpy as np
from scipy.special import logsumexp

class GMM_Custom:
    def __init__(self, data, gmm=None):
        self.data = data
        self.cols = data.columns
        self.gmm = gmm
        try:
            self.ndim = gmm.n_features_in_
        except:
            self.ndim = data.shape[-1]
        n, m = data.shape
        self.ndata = n
        assert(self.ndim==m)
        self.new_gmm = None #used to store the conditioned+marginalised GMM

    def __str__(self):
        lines = []

        lines.append("=" * 36)
        lines.append("Full GMM in the following variables")
        lines.append(str(self.cols))
        lines.append(f"Components: {self.gmm.n_components}")
        lines.append(f"Dimensions: {self.gmm.means_.shape[1]}")

        for k in range(self.gmm.n_components):
            lines.append(f"Component {k}")
            lines.append(f" weight: {self.gmm.weights_[k]}")
            lines.append(f" mean  : {self.gmm.means_[k]}")

        lines.append("=" * 36)

        if self.new_gmm is not None:
            lines.append("Reduced GMM in the following variables")
            # handle naming safely
            reduce_cols = getattr(self, "reduce_cols", getattr(self, "reduced_cols", None))
            lines.append(str(reduce_cols))

            if len(self.cond_i) > 0:
                lines.append("Conditioned at the following variables")
                lines.append(
                    f"{self.cols[self.cond_i]} = {self.i_transform(self.x_cond)}"
                )

            if len(self.marg_i) > 0:
                lines.append("Marginalised along the following variables")
                lines.append(str(self.cols[self.marg_i]))

            lines.append(f"Components: {self.new_gmm.n_components}")
            lines.append(f"Dimensions: {self.new_gmm.means_.shape[1]}")

            for k in range(self.new_gmm.n_components):
                lines.append(f"Component {k}")
                lines.append(f" weight: {self.new_gmm.weights_[k]}")
                lines.append(f" mean  : {self.new_gmm.means_[k]}")

            lines.append("=" * 36)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def fit(self):
        if self.gmm is not None:
            print("The full GMM should not be overwritten")
            return
        #else fit a GMM
        X = self.transform(self.data)
        [train_X, test_X] = train_test_split(X, random_state=42)
        #scaler = StandardScaler()
        #scaler.fit(train_X)
        #train_X, test_X = scaler.transform(train_X), scaler.transform(test_X) #scaling was not doing much, so to make it easy, I removed it
        min_n, min_bic = -1, np.inf
        for n in range(2,7): #choose the #components
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(train_X)
            bic_train, bic_test = gmm.bic(train_X)/len(train_X), gmm.bic(test_X)/len(test_X)
            print(n, bic_train, bic_test)
            if bic_test < min_bic:
                min_n = n
                min_bic = bic_test
        self.gmm = GaussianMixture(n_components=min_n, random_state=42)
        self.gmm.fit(train_X)
        print("Fitted GMM", self.gmm)

    def transform(self,x):
        #forward transform
        return np.log(x)

    def i_transform(self,x):
        #inverse transform
        return np.exp(x)

    def reduce(self, condition = [], x_cond = [], marginalise = []):
        #creates a new GMM after conditioning and marginalising on given axes
        if type(x_cond) is not list and type(x_cond) is not np.ndarray:
            x_cond = [x_cond]

        if type(condition) is not list:
            condition = [condition]

        #if len(condition)>0 and len(x_cond)==0:
        #assert(len(condition) == len(x_cond))

        if type(marginalise) is not list:
            marginalise = [marginalise]

        #find the index of x_cond
        try:
            cond_i = [self.cols.get_loc(c) for c in condition]
        except:
            print("Error: some condition variable names are not present in the original data", self.cols, condition)

        if len(condition)>0 and len(x_cond)==0:
            #x_cond = self.data[condition].mean().to_numpy() #wouldn't work if data has been removed
            means = self.i_transform(np.sum(np.array([a*b for a,b in zip(self.gmm.means_, self.gmm.weights_)]),axis=0))
            x_cond = means[cond_i]
        assert(len(condition) == len(x_cond))

        #find the index of marginalise
        try:
            marg_i = [self.cols.get_loc(c) for c in marginalise]
        except:
            print("Error: some marginalise variable names are not present in the original data", self.cols, marginalise)

        self.cond_cols = condition
        self.marg_cols = marginalise

        #find remaining indices
        x_i = list(np.arange(len(self.cols)))
        #print(x_i, cond_i, marg_i)
        remove_i = cond_i + marg_i
        for ii in remove_i:
            x_i.remove(ii)
        self.reduce_cols = self.cols[x_i]
        print('Remaining indices are', self.reduce_cols)

        if len(cond_i)>0:
            print('setting the value of ', self.cols[cond_i], 'to', x_cond)
            self.x_cond = self.transform(x_cond)

        self.x_i = x_i
        self.cond_i = cond_i
        self.marg_i = marg_i
        if len(cond_i)==0:
            self._marginalize_gmm()
        else:
            self._create_conditioned_gmm()

        self.reduce_ndim = self.new_gmm.means_.shape[-1]
        if self.reduce_ndim == 2:
            self._create_arrays()

    def reduce_to_cols(self, left_cols):
        """
        Reduce the GMM to the specified columns by marginalising
        all other variables. No conditioning is performed.

        Parameters
        ----------
        cols : list[str]
            Column names to retain in the reduced GMM (others are marginalised).
        """
        if not isinstance(left_cols, (list, tuple)):
            left_cols = [left_cols]

        #find the remaining indeces
        try:
            x_i = [self.cols.get_loc(c) for c in left_cols]
        except:
            print("Error: somevariable names are not present in the original data", self.cols, left_cols)

        #find marginal indices
        marg_i = list(np.arange(len(self.cols)))
        for ii in x_i:
            marg_i.remove(ii)
        marg_cols = list(self.cols[marg_i])
        print("In reduce_to_cols",marg_cols)
        self.reduce(marginalise=marg_cols)

    def _create_conditioned_gmm(self):
        self.n_components = self.gmm.weights_.shape[0]
        self.conditioned_weights = []
        self.conditioned_means = []
        self.conditioned_covariances = []
        self.means_a = []
        self.means_b = []
        self.Sigmas_aa = []
        self.Sigmas_ab = []
        self.Sigmas_ba = []
        self.Sigmas_bb = []
        self.inv_Sigmas_aa = []

        for k in range(self.n_components):
            mean = self.gmm.means_[k]
            cov = self.gmm.covariances_[k]

            # Split mean and cov into known and unknown
            self.means_a.append(mean[self.cond_i])
            self.means_b.append(mean[self.x_i])

            self.Sigmas_aa.append(cov[np.ix_(self.cond_i, self.cond_i)])
            self.Sigmas_ab.append(cov[np.ix_(self.cond_i, self.x_i)])
            self.Sigmas_ba.append(cov[np.ix_(self.x_i, self.cond_i)])
            self.Sigmas_bb.append(cov[np.ix_(self.x_i, self.x_i)])

            self.inv_Sigmas_aa.append(np.linalg.inv(self.Sigmas_aa[k]))

            # Compute conditional mean and cov
            mu_cond = self.means_b[k] + self.Sigmas_ba[k] @ self.inv_Sigmas_aa[k] @ (self.x_cond - self.means_a[k])
            cov_cond = self.Sigmas_bb[k] - self.Sigmas_ba[k] @ self.inv_Sigmas_aa[k] @ self.Sigmas_ab[k]

            # Compute new weight using Bayes' rule
            weight = self.gmm.weights_[k] * multivariate_normal.pdf(self.x_cond, mean=self.means_a[k], cov=self.Sigmas_aa[k])
            self.conditioned_weights.append(weight)
            self.conditioned_means.append(mu_cond)
            self.conditioned_covariances.append(cov_cond)

        # Normalize weights
        self.conditioned_weights = np.array(self.conditioned_weights)
        self.conditioned_weights /= self.conditioned_weights.sum()

        # Create a new GMM with conditioned parameters
        self.new_gmm = GaussianMixture(n_components=len(self.conditioned_weights), covariance_type='full')
        self.new_gmm.weights_ = np.array(self.conditioned_weights)
        self.new_gmm.means_ = np.array(self.conditioned_means)
        self.new_gmm.covariances_ = np.array(self.conditioned_covariances)

        # Required to make GMM usable for sampling/scoring
        self.new_gmm.precisions_cholesky_ = _compute_precision_cholesky(self.new_gmm.covariances_, 'full')
        print("Created a reduced gmm with",self.new_gmm.means_.shape[-1],"dimensions")
        
    def _marginalize_gmm(self):
        weights = self.gmm.weights_
        means = self.gmm.means_[:, self.x_i]
        covs = self.gmm.covariances_
        covs_new = np.array([cov[np.ix_(self.x_i, self.x_i)] for cov in covs])

        new_gmm = GaussianMixture(n_components=len(weights), covariance_type='full')
        new_gmm.weights_ = weights
        new_gmm.means_ = means
        new_gmm.covariances_ = covs_new
        new_gmm.precisions_cholesky_ = _compute_precision_cholesky(covs_new, 'full')
        self.new_gmm = new_gmm
        print("Created a reduced gmm with",self.new_gmm.means_.shape[-1],"dimensions")

    def update_reduced_gmm(self, x_cond):
        if type(x_cond) is not list and type(x_cond) is not np.ndarray:
            x_cond = [x_cond]

        assert(len(self.cond_i) == len(x_cond))

        if len(self.cond_i)>0:
            print('setting the value of ', self.cols[self.cond_i], 'to', x_cond)
            self.x_cond = self.transform(x_cond)

        #updating the means
        for k in range(self.n_components):
            self.conditioned_means[k] = self.means_b[k] + self.Sigmas_ba[k] @ self.inv_Sigmas_aa[k] @ (self.x_cond - self.means_a[k])
        self.new_gmm.means_ = np.array(self.conditioned_means)

        #updating the weights
        #self.conditioned_weights = []
        vals = []
        for k in range(self.n_components):
            # Compute new weight using Bayes' rule
            #weight = self.gmm.weights_[k] * multivariate_normal.pdf(known_values, mean=self.means_a[k], cov=self.Sigmas_aa[k])
            #self.conditioned_weights.append(weight)
            vals.append(multivariate_normal.logpdf(self.x_cond, mean=self.means_a[k], cov=self.Sigmas_aa[k]))

        log_weights = np.log(self.gmm.weights_) + np.array(vals)
        weights = np.exp(log_weights)
        weights /= weights.sum()
        self.new_gmm.weights_ = weights
        # Normalize weights
        #self.conditioned_weights = np.array(self.conditioned_weights)
        #self.conditioned_weights /= self.conditioned_weights.sum()
        #self.new_gmm.weights_ = np.array(self.conditioned_weights)

    def get_gmm_contour_data(self, log=False):
        assert self.reduce_ndim == 2, "Reduced dimension is not 2, cannot create arrays. First reduce appropriately"
        vals = self.new_gmm.score_samples(self.pos.reshape(-1,2))
        if not log:
            vals = np.exp(vals)
        self.Z = vals.reshape(self.X.shape)
        #self.Z[:,:] = 0.
        #for k in range(self.new_gmm.n_components):
        #    rv = multivariate_normal(mean=self.new_gmm.means_[k], cov=self.new_gmm.covariances_[k])
        #    self.Z += self.new_gmm.weights_[k] * rv.pdf(self.pos)

        return self.X, self.Y, self.Z

    def contour_levels_from_percentiles(self, percentiles = np.linspace(0.9,0.1,9), log=False):
        """
        Compute density contour levels corresponding to given probability percentiles.

        percentiles: list of floats in (0,1), e.g. [0.5, 0.9, 0.95]

        Important: Assumes uniform X-Y grid (in the density space)
        """
        X, Y, Z = self.get_gmm_contour_data(log=False)

        dx, dy = np.log(X[0,1])-np.log(X[0,0]), np.log(Y[1,0])-np.log(Y[0,0])
        Z_sorted = np.sort(Z.flatten())[::-1]
        cum_prob = np.cumsum(Z_sorted*dx*dy)
        if cum_prob[-1]<0.99 or cum_prob[-1]>1:
            raise ValueError("Cumulative probability is outside the desired range of 0.99 and 1.\n
                              The contour levels might be spurious")

        levels = []
        for p in percentiles:
            if not (0 < p < 1):
                raise ValueError("Percentiles must be in (0,1)")
            i = np.searchsorted(cum_prob, p)
            levels.append(Z_sorted[i])

        if log:
            Z = np.log(Z)
        return X, Y, Z, levels

    def _create_arrays(self, resolution=120):
        assert self.reduce_ndim == 2, "Reduced dimension is not 2, cannot create arrays. First reduce appropriately"
        # Create X, Y, and Z arrays
        means = self.new_gmm.means_
        covs = self.new_gmm.covariances_

        x_vals = means[:, 0]
        x_std = np.sqrt([cov[0, 0] for cov in covs])
        x_min = np.min(x_vals - 3 * x_std)
        x_max = np.max(x_vals + 4 * x_std)
        xlim = (x_min, x_max)

        y_vals = means[:, 1]
        y_std = np.sqrt([cov[1, 1] for cov in covs])
        y_min = np.min(y_vals - 3 * y_std)
        y_max = np.max(y_vals + 3 * y_std)
        ylim = (y_min, y_max)

        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.pos = np.dstack((self.X, self.Y))
        self.X = self.i_transform(self.X)
        self.Y = self.i_transform(self.Y)

        self.Z = np.zeros(self.X.shape)

    def prob(self, x, condition = [], x_cond = [], marginalise = [], log = False):
        x = np.atleast_2d(x)
        if condition is [] and marginalise is []:
            print('setting the value of ', self.cols, 'to', x)
            x = self.transform(x)
            return np.exp(self.gmm.score_samples(x))
        
        #else, reduce the dimensions and find the new gmm
        assert(x.shape[1] + len(condition) + len(marginalise) == self.ndim)
        self.reduce(condition, x_cond, marginalise)

        print('setting the value of ', self.cols[self.x_i], 'to', x)
        x = self.transform(x)
        if log:
            return self.new_gmm.score_samples(x)
        return np.exp(self.new_gmm.score_samples(x))

    def prob_reduced(self, x, x_cond=None, log=False):
        x = np.atleast_2d(x)
        x = self.transform(x)

        if x_cond is not None:
            print('setting the value of ', self.cols[self.cond_i], 'to', x_cond)
            self.update_reduced_gmm(x_cond)
        elif len(self.cond_i)>0:
            print('Using the previous conditioned values for ', self.cols[self.cond_i], 'as', self.i_transform(self.x_cond))
            
        if log:
            return self.new_gmm.score_samples(x)
        return np.exp(self.new_gmm.score_samples(x))

    def z_score_reduced(self, x, x_cond=None):
        assert self.reduce_ndim==1, "z_score_reduced only works when the reduced dimension is 1"

        x = np.atleast_2d(x)
        x = self.transform(x)

        if x_cond is not None:
            print('setting the value of ', self.cols[self.cond_i], 'to', x_cond)
            self.update_reduced_gmm(x_cond)
        elif len(self.cond_i)>0:
            print('Using the previous conditioned values for ', self.cols[self.cond_i], 'as', self.i_transform(self.x_cond))
        
        gmm = self.new_gmm

        weights = gmm.weights_
        means = gmm.means_.ravel()
        stds = np.sqrt(gmm.covariances_[:, 0, 0])

        # Mixture CDF
        mix_cdf = np.zeros_like(x, dtype=float)
        for w, mu, sigma in zip(weights, means, stds):
            mix_cdf += w * norm.cdf((x - mu) / sigma)

        # Numerical safety
        eps = np.finfo(float).eps
        mix_cdf = np.clip(mix_cdf, eps, 1 - eps)

        # Convert to equivalent standard normal z
        z = norm.ppf(mix_cdf)
        return z.flatten()
            
    def get_sf(self, var_name, threhold, condition=True):
        if condition:
            gmm = self.new_gmm
            dim = self.reduce_cols.get_loc(var_name)
        else:
            gmm = self.gmm
            dim = self.cols.get_loc(var_name)
        prob = 0.0

        threshold = self.transform(threshold)

        for k in range(gmm.n_components):
            mean_k = gmm.means_[k][dim]
            var_k = gmm.covariances_[k][dim, dim]  # works for full covariance
            std_k = np.sqrt(var_k)
            weight_k = gmm.weights_[k]

            tail_prob = norm.sf(threshold, loc=mean_k, scale=std_k)  # sf = survival function = 1 - cdf
            prob += weight_k * tail_prob
        #print(prob)
        return prob

def CMI_gmms(gmms,x_cols,z_cols):
    """
    Compute I(X; C | Z) using posterior entropies, where
    C is a discrete class represented by multiple GMM instances.

    Parameters
    ----------
    gmms : list[GMM]
        One GMM per class. All must share identical columns.
    x_cols : list[str]
        Columns corresponding to X
    z_cols : list[str]
        Columns corresponding to Z

    Returns
    -------
    float
        Conditional mutual information I(X; C | Z)
    """

    # --------------------------------------------------
    # sanity checks
    # --------------------------------------------------
    ref_cols = gmms[0].cols
    for i, gmm in enumerate(gmms[1:], start=1):
        if not ref_cols.equals(gmm.cols):
            raise ValueError(
                f"GMM at index {i} has different columns.\n"
                f"Expected: {list(ref_cols)}\n"
                f"Found:    {list(gmm.cols)}"
            )

    # --------------------------------------------------
    # class priors from data
    # --------------------------------------------------
    counts = np.array([len(gmm.data) for gmm in gmms], dtype=float)
    priors = counts / counts.sum()
    log_priors = np.log(priors)

    # --------------------------------------------------
    # combined data (empirical expectation)
    # --------------------------------------------------
    combined_df = pd.concat([gmm.data for gmm in gmms],axis=0,ignore_index=True)

    # extract Z and (X, Z) using column names directly
    Z = combined_df[z_cols].to_numpy()
    XZ = combined_df[z_cols + x_cols].to_numpy()

    # --------------------------------------------------
    # entropy helper
    # --------------------------------------------------
    def entropy_from_posteriors(p):
        eps = 1e-12
        return -np.mean(np.sum(p * np.log(p + eps), axis=1))

    # ==================================================
    # H(C | Z)
    # ==================================================
    logp_c_z = []
    for gmm, log_prior in zip(gmms, log_priors):
        gmm.reduce_to_cols(z_cols)
        logp = gmm.prob_reduced(Z, log=True) + log_prior
        logp_c_z.append(logp)

    logp_c_z = np.vstack(logp_c_z)
    log_norm_z = logsumexp(logp_c_z, axis=0)
    p_c_given_z = np.exp(logp_c_z - log_norm_z).T

    H_C_given_Z = entropy_from_posteriors(p_c_given_z)

    # ==================================================
    # H(C | X, Z)
    # ==================================================
    logp_c_xz = []
    for gmm, log_prior in zip(gmms, log_priors):
        gmm.reduce_to_cols(z_cols + x_cols)
        logp = gmm.prob_reduced(XZ, log=True) + log_prior
        logp_c_xz.append(logp)

    logp_c_xz = np.vstack(logp_c_xz)
    log_norm_xz = logsumexp(logp_c_xz, axis=0)
    p_c_given_xz = np.exp(logp_c_xz - log_norm_xz).T

    H_C_given_XZ = entropy_from_posteriors(p_c_given_xz)

    # --------------------------------------------------
    # conditional mutual information
    # --------------------------------------------------
    return H_C_given_Z - H_C_given_XZ

def CMI_gmms_MC(gmms,x_cols,z_cols,nsamples=100_000,random_state=None):
    """
    Monte Carlo estimate of I(X; C | Z), where C is represented
    by multiple GMM instances.

    Parameters
    ----------
    gmms : list[GMM]
        One GMM per class. All must share identical columns.
    x_cols : list[str]
        Columns corresponding to X
    z_cols : list[str]
        Columns corresponding to Z
    nsamples : int
        Number of Monte Carlo samples
    random_state : int or None
        RNG seed

    Returns
    -------
    float
        Monte Carlo estimate of I(X; C | Z)
    """

    rng = np.random.default_rng(random_state)

    # --------------------------------------------------
    # sanity check: same columns
    # --------------------------------------------------
    ref_cols = gmms[0].cols
    for i, gmm in enumerate(gmms[1:], start=1):
        if not ref_cols.equals(gmm.cols):
            raise ValueError(
                f"GMM at index {i} has different columns.\n"
                f"Expected: {list(ref_cols)}\n"
                f"Found:    {list(gmm.cols)}"
            )

    # --------------------------------------------------
    # class priors from data
    # --------------------------------------------------
    counts = np.array([len(gmm.data) for gmm in gmms], dtype=float)
    priors = counts / counts.sum()
    log_priors = np.log(priors)

    n_classes = len(gmms)
    classes = np.arange(n_classes)

    # --------------------------------------------------
    # reduce GMMs once
    # --------------------------------------------------
    for gmm in gmms:
        gmm.reduce_to_cols(z_cols + x_cols)

    # --------------------------------------------------
    # sample classes
    # --------------------------------------------------
    class_idx = rng.choice(n_classes, size=nsamples, p=priors)

    # --------------------------------------------------
    # sample (X,Z) conditionally
    # --------------------------------------------------
    XZ = np.zeros((nsamples, len(z_cols) + len(x_cols)))

    for c in range(n_classes):
        idx = np.where(class_idx == c)[0]
        if len(idx) == 0:
            continue
        samples, _ = gmms[c].new_gmm.sample(len(idx))
        XZ[idx] = samples

    # --------------------------------------------------
    # log p(c | x, z)
    # --------------------------------------------------
    logp_xz = np.column_stack([
        gmm.new_gmm.score_samples(XZ) + lp
        for gmm, lp in zip(gmms, log_priors)
    ])

    log_norm_xz = logsumexp(logp_xz, axis=1)
    log_p_c_given_xz = (
        logp_xz[np.arange(nsamples), class_idx] - log_norm_xz
    )

    # --------------------------------------------------
    # now reduce to Z ONCE
    # --------------------------------------------------
    for gmm in gmms:
        gmm.reduce_to_cols(z_cols)

    Z = XZ[:, :len(z_cols)]

    logp_z = np.column_stack([
        gmm.new_gmm.score_samples(Z) + lp
        for gmm, lp in zip(gmms, log_priors)
    ])

    log_norm_z = logsumexp(logp_z, axis=1)
    log_p_c_given_z = (
        logp_z[np.arange(nsamples), class_idx] - log_norm_z
    )

    # --------------------------------------------------
    # Monte Carlo estimate
    # --------------------------------------------------
    return float(np.mean(log_p_c_given_xz - log_p_c_given_z))


# Compute MI matrix based on data alone
def MI_data_matrix(df, discrete_cols = []):
    from npeet import entropy_estimators as ee
    cols = df.columns
    n = len(cols)
    mi_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i,n):
            x = df.iloc[:,i].to_numpy().reshape(-1, 1)
            y = df.iloc[:,j].to_numpy().reshape(-1, 1)
            print(i,j)
            if i==j:
                #print('equal')
                mi_matrix[i,j] = 0.
            elif cols[i] in discrete_cols and cols[j] in discrete_cols:
                mi_matrix[i,j] = ee.midd(y,x)
            elif cols[i] in discrete_cols: 
                mi_matrix[i,j] = ee.micd(y,x,k=5)
            elif cols[j] in discrete_cols:
                mi_matrix[i,j] = ee.micd(x,y,k=5)
            else:
                #print('none')
                mi_matrix[i,j] = ee.mi(x, y,k=5)
    for i in range(n):
        for j in range(i):
            mi_matrix[i,j] = mi_matrix[j,i]
    return pd.DataFrame(mi_matrix, index=cols, columns=cols)

# Compute CMI based on data alone (does not work with discrete data)
def CMI_data(df, Xcols, Ycols, Zcols):
    #Mutual information of x and y, conditioned on z
    from npeet import entropy_estimators as ee
    x = np.atleast_2d(df[Xcols].to_numpy())
    y = np.atleast_2d(df[Ycols].to_numpy())
    z = np.atleast_2d(df[Zcols].to_numpy())
    return ee.cmi(x, y, z, k=5)
