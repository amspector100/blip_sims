# for generating synthetic data
import numpy as np
from scipy import stats

def create_sparse_coefficients(
	p,
	sparsity,
	coeff_dist,
	coeff_size,
	min_coeff=0.1
):

	# Create sparse coefficients,
	beta = np.zeros(p)
	kp = np.around(sparsity * p).astype(int) # num non-nulls
	if coeff_dist == 'normal':
		nonnull_coefs = np.sqrt(coeff_size) * np.random.randn(kp)
		small = np.abs(nonnull_coefs) < min_coeff
		nonnull_coefs[small] = 0.1 * np.sign(nonnull_coefs[small])
	elif coeff_dist == 'uniform':
		nonnull_coefs = coeff_size * np.random.uniform(1/2, 1, size=kp)
		nonnull_coefs *= (1 - 2*np.random.binomial(1, 0.5, size=kp))
	else:
		raise ValueError(f"Unrecognized coeff_dist={coeff_dist}")
	beta[np.random.choice(np.arange(p), kp, replace=False)] = nonnull_coefs
	return beta

def generate_regression_data(
	n=100,
	p=500,
	covmethod='ark',
	y_dist='gaussian', # one of gaussian, probit, binomial
	coeff_size=1,
	coeff_dist='normal',
	sparsity=0.05,
	k=1, # specifies the k in ark
	alpha0=0.2, # for ark model
	alphasum=1, # for ark model
	temp=3, # for hmm
	rank=10, # for factor model
	min_coeff=0.1,
	max_corr=0.99,
	permute_X=False,
	return_cov=False, # for CRT return cov matrix
	dgp_seed=None,
):
	# if dgp_seed is not None, ensure data-generating 
	# process (dgp) is constant
	rstate = np.random.get_state() # save old state
	np.random.seed(dgp_seed) # only for creating dgp

	# Beta
	beta = create_sparse_coefficients(
		p=p, sparsity=sparsity, coeff_dist=coeff_dist, 
		coeff_size=coeff_size, min_coeff=min_coeff
	)

	# Generate X-data
	covmethod = str(covmethod).lower()
	if covmethod == 'ark' or covmethod == 'hmm':
		alpha = np.zeros(k+1)
		alpha[0] = alpha0
		alpha[1:] = (alphasum - alpha0) / k
		rhos = stats.dirichlet.rvs(alpha, size=p-1) # p-1 x k+1
		# Enforce maximum correlation
		rhos[:, 0] = np.maximum(rhos[:, 0], 1 - max_corr)
		rhos = rhos / rhos.sum(axis=1).reshape(-1, 1) # ensure sums to 1
		rhos = np.sqrt(rhos)

		# Compute eta so that the autoregressive process is equivalent to
		# Z = etas @ W for W i.i.d. standard normals
		etas = np.zeros((p, p))
		etas[0, 0] = 1
		for j in range(1, p):
			# Account for correlations between Xj and X_{1:j-1} 
			rhoend = min(j+1, k+1)
			for i, r in enumerate(np.flip(rhos[j-1,1:rhoend])):
				etas[j] += etas[j-i-1] * r
			# Rescale so Var(Xj) = 1
			scale = np.sqrt((1 - rhos[j-1, 0]**2) / np.power(etas[j], 2).sum())
			etas[j] = etas[j] * scale
			# Add extra noise
			etas[j, j] = rhos[j-1, 0]

		# Ensure data is not constant
		np.random.set_state(rstate)
		W = np.random.randn(n, p)
		Z = np.dot(W, etas.T)
		V = np.dot(etas, etas.T)

		if covmethod == 'ark':
			X = Z
		else:
			expZ = np.exp(temp * Z.astype(np.float32))
			probs = expZ / (1.0 + expZ)
			X = np.random.binomial(
				2, probs
			) - 1
			X = X.astype(np.float64)
	elif covmethod == 'factor':
		diag_entries = np.random.uniform(low=0.01, high=1, size=p)
		noise = np.random.randn(p, rank) / np.sqrt(rank)
		V = np.diag(diag_entries) + np.dot(noise, noise.T)
		L = np.linalg.cholesky(V)
		# Ensure data is not constant
		np.random.set_state(rstate)
		X = np.dot(np.random.randn(n, p), L.T)
	# This is a cool idea, but it is not PSD.
	# elif covmethod == 'block_decay':
	# 	block_size = min(p, block_size)
	# 	nblocks = int(np.ceil(p / block_size))
	# 	# cumulative product of betas
	# 	rhos = np.ones(nblocks + 1)
	# 	rhos[1:] = np.exp(np.cumsum(np.log(
	# 		stats.beta.rvs(size=nblocks, a=a, b=b)
	# 	)))
	# 	# create V
	# 	inds = np.arange(p)
	# 	dists = np.abs(inds.reshape(1, -1) - inds.reshape(-1, 1))
	# 	V = rhos[np.ceil(dists / block_size).flatten().astype(int)].reshape(p, p)
	else:
		raise ValueError(f"Unrecognized covmethod={covmethod}")
		
	# Possibly permute to make slightly more realistic
	if permute_X:
		perminds = np.arange(p)
		np.random.shuffle(perminds)
		X = np.ascontiguousarray(X[:, perminds])

	# Create Y
	mu = np.dot(X, beta)
	if y_dist == 'gaussian' or y_dist=='linear':
		y = mu + np.random.randn(n)
	elif y_dist == 'probit':
		y = ((mu + np.random.randn(n)) < 0).astype(float)
	elif y_dist == 'binomial':
		probs = np.exp(mu)
		probs = probs / (1.0 + probs)
		y = np.random.binomial(1, probs)
	else:
		raise ValueError(f"unrecognized y_dist=={y_dist}")

	# Return covariance, useful for CRT
	if return_cov:
		return X, y, beta, V
	return X, y, beta

def gen_changepoint_data(
	T=100,
	coeff_size=1,
	coeff_dist='normal',
	min_coeff=0.1,
	sigma=1,
	sparsity=0.01
):
	beta = create_sparse_coefficients(
		p=T, sparsity=sparsity, coeff_dist=coeff_dist, 
		coeff_size=coeff_size, min_coeff=min_coeff,
	)
	beta[0] = 0
	Y = np.random.randn(T) + np.cumsum(beta)
	# In case we want to use a dummy X variable for regression
	X = np.ones((T, T))
	for j in range(T):
		X[0:j, j] = 0
	return X, Y, beta