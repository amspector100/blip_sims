from re import S
import sys
import numpy as np
import scipy
import scipy.linalg
from scipy import stats
import sklearn.linear_model
import warnings

from . import tree_methods
try:
	import pyblip
except ModuleNotFoundError:
	import os
	import sys
	main_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
	main_dir = os.path.split(main_dir)[0]
	sys.path.insert(0, os.path.abspath(main_dir + '/pyblip'))
	import pyblip

MIN_PVAL = 1e-16

def l2pval(inner_product):
	k = inner_product.shape[0]
	T = np.power(inner_product, 2).sum() 
	p = 1 - stats.chi2.cdf(T, df=k)
	if p < MIN_PVAL:
		p = MIN_PVAL
	return p

def maxpval(inner_product):
	k = inner_product.shape[0]
	T = np.power(inner_product, 2).max()
	p = 1 - np.power(stats.chi2.cdf(T, df=1), k)
	if p < MIN_PVAL:
		p = MIN_PVAL
	return p

class MultipleDCRT():
	"""
	Lasso-based distilled CRT for Gaussian data. 
	Uses analytic p-values.

	Parameters
	----------
	y : np.ndarray
		``n``-shaped array of response data
	X : np.ndarray
		``(n,p)``-shaped array of covariate data.
	Sigma : np.ndarray
		``(p,p)``-shaped covariance matrix of X.
	mu : np.ndarray
		``(n,)``-shaped mean of X. Defaults to ``X.mean(axis=0)``.
	screen : bool
		If true, run a cross-validated lasso and screen out 
		coefficients with zeros.
	suppress_warnings : bool
		If True, suppresses convergence warnings from sklearn.
	"""
	def __init__(self, y, X, Sigma, mu=None, screen=True, suppress_warnings=True):

		self.n = y.shape[0]
		self.p = X.shape[1]
		self.y = y
		self.X = X
		self.Sigma = Sigma
		# For ~20x speedup:
		# https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
		self.invSigma = np.linalg.inv(Sigma) 
		self.mu = mu if mu is not None else X.mean(axis=0)
		self.screen = screen
		self.suppress_warnings = suppress_warnings
		self.pTree = None

		# Check if y is binary or Gaussian
		unique_y = np.sort(np.unique(y))
		self.logistic = False
		if unique_y.shape[0] == 2:
			if np.all(unique_y == np.array([0, 1])):
				self.logistic = True

		if self.screen:
			with warnings.catch_warnings():
				if self.suppress_warnings:
					warnings.filterwarnings("ignore")
				if self.logistic:
					lasso = sklearn.linear_model.LogisticRegressionCV(
						penalty='l1', solver='saga'
					)
				else:
					lasso = sklearn.linear_model.LassoCV()
				lasso.fit(self.X, self.y)
				self.active_set = set(np.where(lasso.coef_ != 0)[0].tolist())
		else:
			self.active_set = set(list(range(self.p)))

		self.pTree = None

	def create_tree(self, levels=0, max_size=100):
		"""
		Creates a tree for p-values.
		"""
		self.pTree, self.levels = tree_methods.corr_matrix_to_pval_tree(
			self.Sigma, levels=levels, max_size=max_size, return_levels=True
		)

	def multiple_pvals(self, levels=0, max_size=100, **kwargs):
		"""
		Computes many (group) p-values.
		"""
		if self.pTree is None:
			self.create_tree(levels=levels, max_size=max_size)
		
		# Todo: could speed up computation using nested structure
		for node in self.pTree.nodes:
			node.p = self.p_value(
				inds=list(node.group),
				node=node,
				**kwargs
			)

		return self.pTree

	def compute_X_given_Z(
		self,
		inds,
		Z_inds,
		Z,
	):
		# Block matrices
		mu_Z = self.mu[Z_inds]
		Sigma_XZ = self.Sigma[inds][:, Z_inds]
		Sigma_Z = self.Sigma[Z_inds][:, Z_inds]
		inv_Sigma_Z = np.linalg.inv(Sigma_Z) # Todo: could use rank-k updates to do this more efficiently
		# Dimension: p - k x 1
		cond_mean_transform = np.dot(Sigma_XZ, inv_Sigma_Z).T
		cond_mean = np.dot(Z - mu_Z, cond_mean_transform) + self.mu[inds]
		# Conditional variance
		if len(inds) == 1:
			ind = inds[0]
			cond_var = 1 / self.invSigma[ind, ind]
		else:
			Sigma_X = self.Sigma[inds][:, inds]
			cond_var = Sigma_X - np.dot(
				Sigma_XZ, np.dot(inv_Sigma_Z, Sigma_XZ.T)
			)
		# Return
		return cond_mean, cond_var

		

	def distill(
		self, 
		inds,
		cond_mean_transform=None,
		cond_var=None,
		model_type='lasso',
		node=None,
		**kwargs
	):
		"""
		Computes a p-value to test whether X[:, inds] and y are conditionally
		independent given the other X data.

		Parameters
		----------
		inds : int, list or np.ndarray
			Integer or list of integers identifying the variables to be tested.
		cond_mean_transform : np.ndarray
			The linear function such that the conditional mean of X[:, inds]
			equals ``np.dot(cond_mean_transform, X[:, ~inds]).``
		cond_var : np.ndarray
			The conditional covariance matrix of X[:, inds] given the other X data. 
		node : PNode
			Corresponds to node in p-value tree. Optional but allows caching.
		agg : string
			Either 'max' or 'l2'.
		Returns
		-------
		X_distilled : np.ndarray
			``(n, k)``-shaped array where ``k=len(inds)``.
			The components are marginally i.i.d. standard
			Gaussian conditional on Z.
		y_distilled : np.ndarray
			``(n,)``-shaped array of distilled y-data.
		"""

		# Construct Z_inds (this is inefficient but not a bottleneck)
		Z_inds = [j for j in np.arange(self.p) if j not in inds]

		# Special case where inds is all of the indices
		if len(Z_inds) == 0:
			whitening_transform = np.linalg.cholesky(np.linalg.inv(self.Sigma))
			X_distilled = np.dot(
				whitening_transform, self.X.T - self.mu.reshape(-1, 1),
			).T
			y_distilled = np.zeros(self.n)
			return X_distilled, y_distilled

		# The notation now matches https://arxiv.org/pdf/2006.03980.pdf
		X = self.X[:, inds]
		mu_X = self.mu[inds]
		Z = self.X[:, Z_inds]
		mu_Z = self.mu[Z_inds]

		# suppress warnings fromsklearn for distillation
		with warnings.catch_warnings():
			if self.suppress_warnings:
				warnings.filterwarnings("ignore")

			# 1. Distill information about y, set some kwarg defaults
			kwargs['cv'] = kwargs.get('cv', 5)
			kwargs['max_iter'] = kwargs.get('max_iter', 500)
			kwargs['tol'] = kwargs.get('tol', 5e-3)
			if self.logistic:
				kwargs['penalty'] = kwargs.get('penalty', 'l1')
				kwargs['solver'] = kwargs.get('solver', 'liblinear')
				lasso = sklearn.linear_model.LogisticRegressionCV(**kwargs)
				lasso.fit(Z, self.y)
				#y_distilled = np.dot(Z, lasso.coef_.T)
				y_distilled = lasso.predict_proba(Z)[:, 1]
			else:
				kwargs['selection'] = 'random'
				if model_type in ['elasticnet', 'lasso']:
					if model_type == 'elasticnet':
						lasso = sklearn.linear_model.ElasticNetCV(**kwargs)
					elif model_type == 'lasso':
						lasso = sklearn.linear_model.LassoCV(**kwargs)
					lasso.fit(Z, self.y)
					y_distilled = lasso.predict(Z)
				elif model_type == 'bayes':
					for key in ['max_iter', 'tol', 'selection', 'cv']:
						kwargs.pop(key, '')
					lm = pyblip.linear.LinearSpikeSlab(
						X=np.ascontiguousarray(Z), y=self.y, **kwargs
					)
					lm.sample(N=5000, chains=1, bsize=3)
					beta = lm.betas.mean(axis=0)
					y_distilled = np.dot(Z, beta)
				else:
					raise ValueError(f"Unrecognized model_type={model_type}")
				

		# 2. Distill information about X using Z
		if cond_mean_transform is None or (cond_var is None and len(inds) != 1):
			Sigma_XZ = self.Sigma[inds][:, Z_inds]
			Sigma_Z = self.Sigma[Z_inds][:, Z_inds]
			inv_Sigma_Z = np.linalg.inv(Sigma_Z) # Todo: could use rank-k updates to do this more efficiently
		if cond_mean_transform is None:
			# Dimension: p - k x 1
			cond_mean_transform = np.dot(Sigma_XZ, inv_Sigma_Z).T
		if cond_var is None:
			if len(inds) == 1:
				ind = inds[0]
				cond_var = 1 / self.invSigma[ind, ind]
			else:
				Sigma_X = self.Sigma[inds][:, inds]
				cond_var = Sigma_X - np.dot(
					Sigma_XZ, np.dot(inv_Sigma_Z, Sigma_XZ.T)
				)
		if len(inds) == 1:
			whitening_transform = np.sqrt(1/cond_var).reshape(1, 1)
		else: 
			whitening_transform = scipy.linalg.sqrtm(np.linalg.inv(cond_var))

		# Whitening transformation
		cond_mean_X = np.dot(Z - mu_Z, cond_mean_transform) + mu_X
		X_distilled = np.dot(
			whitening_transform, X.T - cond_mean_X.T
		).T # dimension: n x k

		return X_distilled, y_distilled

	def p_value(
		self,
		inds,
		cond_mean_transform=None,
		cond_var=None,
		node=None,
		agg='l2',
		X_distilled=None,
		y_distilled=None,
		**kwargs
	):
		"""
		Computes a p-value to test whether X[:, inds] and y are conditionally
		independent given the other X data.

		Parameters
		----------
		inds : int, list or np.ndarray
			Integer or list of integers identifying the variables to be tested.
		cond_mean_transform : np.ndarray
			The linear function such that the conditional mean of X[:, inds]
			equals ``np.dot(cond_mean_transform, X[:, ~inds]).``
		cond_var : np.ndarray
			The conditional covariance matrix of X[:, inds] given the other X data. 
		node : PNode
			Corresponds to node in p-value tree. Optional but allows caching.
		agg : string
			Either 'max' or 'l2'.
		rec_prop : float
			proportion to recycle, between 0 and 1
		X_distilled : np.ndarray
			``(n, k)``-shaped array of distiled X-data, where ``k=len(inds)``.
			Defaults to None.
		y_distilled : np.ndarray
			``(n,)``-shaped array of distilled y-data. Defaults to None.

		Returns
		-------
		A lasso-based analytic p-value.
		"""

		if isinstance(inds, int):
			inds = [inds]
		if isinstance(inds, np.ndarray):
			inds = inds.tolist()
		if set(inds).intersection(self.active_set) == set():
			return 1

		# 1. Distill, first check if distillation is cached
		if X_distilled is None or y_distilled is None:
			if node is not None:
				X_distilled = node.data.get('X_distilled', None)
				y_distilled = node.data.get('y_distilled', None)
		# Actually distill
		if X_distilled is None or y_distilled is None:
			X_distilled, y_distilled = self.distill(
				inds=inds, 
				cond_mean_transform=cond_mean_transform,
				cond_var=cond_var,
				node=node,
				**kwargs
			) 
		# Possibly cache
		if node is not None:
			## Cache in case we want to compute another p-value later
			node.data['X_distilled'] = X_distilled
			node.data['y_distilled'] = y_distilled

		# 2. Test statistic, with caching
		y_resid = (self.y - y_distilled).reshape(-1, 1)
		y_scale = np.power(y_resid, 2).sum()
		inner_product = np.dot(X_distilled.T, y_resid) / np.sqrt(y_scale)
		if node is not None:
			node.data['inner_product'] = inner_product

		# 3. Convert to p-value
		agg = str(agg).lower()
		if agg == 'max':
			return maxpval(inner_product)
		elif agg == 'l2':
			return l2pval(inner_product)
		else:
			raise ValueError(
				f"agg ({agg}) must be one of ['l2', 'max']"
			)

	def full_p_value(
		self, 
		inds,
		test_stat='bayes',
		M=200,
		**test_stat_kwargs,
	):
		# Initialize
		Z_inds = [j for j in np.arange(self.p) if j not in inds]
		Z = self.X[:, Z_inds]
		if test_stat == 'bayes':
			test_stat = bayes_test_stat
		
		# Sample distribution for X | Z
		cond_mean, cond_var = self.compute_X_given_Z(
			inds=inds, Z_inds=Z_inds, Z=Z
		)
		# Sample X | Z
		noise = np.random.randn(M, self.n, len(inds))
		# Univariate case
		if len(cond_var.shape) < 2:
			X_sample = np.sqrt(cond_var) * noise
		else:
			var_transform = np.linalg.cholesky(cond_var).T
			X_sample = np.dot(noise, var_transform)
		X_sample += cond_mean.reshape(-1, self.n, len(inds))

		# Loop through and compute test statistics
		true_test_stat = test_stat(
			Xstar=self.X[:, inds], 
			Z=Z,
			y=self.y, 
			**test_stat_kwargs
		)
		rand_test_stats = np.zeros(M)
		for j in range(M):
			rand_test_stats[j] = test_stat(
				Xstar=X_sample[j, :, :],
				Z=Z,
				y=self.y,
				**test_stat_kwargs
			)
		# Return p-value
		pval = (np.sum(true_test_stat <= rand_test_stats) + 1) / (M+1)
		return pval

def bayes_test_stat(
	Xstar, Z, y, params=None, sample_kwargs=None,
):
	if params is None:
		params = dict()
	if sample_kwargs is None:
		sample_kwargs = dict()

	# Concatenate
	k = Xstar.shape[1]
	X = np.ascontiguousarray(np.concatenate([Xstar, Z], axis=1)) 
	# Run MCMC
	lm = pyblip.linear.LinearSpikeSlab(X=X, y=y, **params)
	lm.sample(**sample_kwargs)
	# Test stat
	pip = np.any(lm.betas[:, 0:k] != 0, axis=1).mean()
	return pip