# for generating synthetic data
import numpy as np
from scipy import stats

def create_sparse_coefficients(
	p,
	sparsity,
	coeff_dist,
	coeff_size,
	min_coeff=0.1,
	spacing='random'
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
	elif coeff_dist == 'none':
		nonnull_coeffs = coeff_size * (1 - 2*np.random.binomial(1, 0.5, size=kp))
	else:
		raise ValueError(f"Unrecognized coeff_dist={coeff_dist}")
	# Decide location of non-nulls
	if spacing == 'random':
		nnulls = np.random.choice(np.arange(p), kp, replace=False)
	elif spacing == 1:
		nnulls = (np.arange(kp) + np.random.randint(0, p-kp+1)) % p
	else:
		max_spacing = np.floor(p / kp)
		spacing = max(1, min(spacing, max_spacing - 1))
		diff = min(spacing - 1, max_spacing - spacing)
		spacings = np.random.randint(spacing - diff, spacing + diff, size=kp)
		nnulls = np.unique(np.cumsum(spacings) % p)
		# If there are duplicates, just randomly sample a few more
		if nnulls.shape[0] != kp:
			extra_nnulls = np.random.choice(
				list(set(np.arange(p).tolist()) - set(nnulls.tolist())), 
				kp - nnulls.shape[0], 
				replace=False
			)
			nnulls = np.concatenate((nnulls, extra_nnulls))
	
	beta[nnulls] = nonnull_coefs
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
	delta=1,
	permute_X=False,
	return_cov=False, # for CRT return cov matrix
	dgp_seed=None,
	spacing='random',
):
	# if dgp_seed is not None, ensure data-generating 
	# process (dgp) is constant
	rstate = np.random.get_state() # save old state
	np.random.seed(dgp_seed) # only for creating dgp

	# Beta
	beta = create_sparse_coefficients(
		p=p, 
		sparsity=sparsity, 
		coeff_dist=coeff_dist, 
		coeff_size=coeff_size, 
		min_coeff=min_coeff,
		spacing=spacing
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

		# Scale down V (only for specific CRT experiments)
		if delta != 1:
			for i in range(p):
				for j in range(i):
					V[i, j] = delta * V[i,j]
					V[j, i] = delta * V[j,i]
			Z = np.dot(W, np.linalg.cholesky(V).T)
		
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
	reversion_prob=0,
	coeff_size=1,
	coeff_dist='normal',
	min_coeff=0.1,
	spacing='random',
	sparsity=0.01
):
	beta = create_sparse_coefficients(
		p=T, 
		sparsity=sparsity, 
		coeff_dist=coeff_dist, 
		coeff_size=coeff_size,
		min_coeff=min_coeff,
		spacing=spacing,
	)
	beta[0] = 0

	# Add some probability of reverting to original mean
	for j in range(T):
		if beta[j] != 0:
			u = np.random.uniform()
			prevsum = np.sum(beta[0:j-1])
			if prevsum != 0 and u <= reversion_prob:
				beta[j] = -1 * prevsum

	Y = np.random.randn(T) + np.cumsum(beta)
	# In case we want to use a dummy X variable for regression
	X = np.ones((T, T))
	for j in range(T):
		X[0:j, j] = 0
	return X, Y, beta