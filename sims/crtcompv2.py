"""
Template for running simulations.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from context import pyblip, blip_sims
from blip_sims import parser
from blip_sims import utilities
from blip_sims.gen_data import generate_regression_data
from statsmodels.stats.multitest import multipletests

from sklearn import linear_model

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# columns of dataframe
COLUMNS = [
	'method',
	'model_time',
	'pep',
	'well_specified',
	'y_dist',
	'covmethod',
	'kappa',
	'p',
	'sparsity',
	'k',
	'coeff_size',
	'seed', 
]

def single_seed_sim(
	seed,
	y_dist,
	covmethod,
	kappa,
	p,
	sparsity,
	k,
	coeff_size,
	args,
):
	np.random.seed(seed)
	output = []
	dgp_args = [
		y_dist, covmethod, kappa, p, sparsity, k, coeff_size, seed
	]

	# Create data
	n = int(kappa * p)
	sample_kwargs = dict(
		y_dist=y_dist,
		covmethod=covmethod,
		n=n,
		p=p,
		sparsity=sparsity,
		k=k,
		coeff_dist=args.get('coeff_dist', ['uniform'])[0],
		coeff_size=coeff_size,
		min_coeff=args.get('min_coeff', [0.1 * coeff_size])[0],
		dgp_seed=args.get("dgp_seed", [seed])[0],
		max_corr=args.get("max_corr", [0.99])[0],
		return_cov=True
	)
	X, y, beta, V = generate_regression_data(**sample_kwargs)

	# Pick non-null to analyze
	ind = np.where(np.abs(beta) > 0)[0][0]
	inds = [ind]

	# Parse sampling args
	bsize = args.get('bsize', [1])[0]
	chains = args.get('chains', [1])[0]
	nsample = args.get('nsample', [1000])[0]
	sample_kwargs = dict(bsize=bsize, chains=chains, N=nsample)

	# Gibbs + BLiP
	for well_specified in args.get('well_specified', [True]):
		if well_specified:
			p0 = 1 - sparsity
			min_p0 = 0
			tau2 = coeff_size
			sigma2 = 1
			update = False
		else:
			p0 = 0.99
			min_p0 = 0.9
			tau2 = 1
			sigma2 = 1
			update = True

		# Method type 1: BLiP + SpikeSlab
		# Run Gaussian sampler if (1) not well-specified or (2) gaussian response
		if y_dist == 'gaussian' or not well_specified:
			lss_model = pyblip.linear.LinearSpikeSlab(
				X=X,
				y=y.astype(np.float64),
				p0=p0,
				min_p0=min_p0,
				update_p0=update,
				tau2=tau2,
				update_tau2=update,
				sigma2=sigma2,
				update_sigma2=update,
			)
			models = [lss_model]
			method_names = ['LSS']
		else:
			models = []
			method_names = []
		if y_dist != 'gaussian':
			models.append(pyblip.probit.ProbitSpikeSlab(
				X=X, y=y.astype(int), p0=p0, update_p0=update, #min_p0=min_p0 TODO
			))
			method_names.append('PSS')

		for model, mname in zip(models, method_names):
			t0 = time.time()
			model.sample(**sample_kwargs)
			inclusions = model.betas != 0
			mtime = time.time() - t0
			# Compute PEP
			pep = 1 - np.any(model.betas[:, inds] != 0, axis=1).mean()
			output.append(
				[mname, mtime, pep, well_specified] + dgp_args
			)
	
	# Run full (non-distilled) CRT for this p-value
	t0 = time.time()
	M = args.get("m", [200])[0]
	crt_model = blip_sims.crt.MultipleDCRT(
		y=y, X=X, Sigma=V, screen=False, suppress_warnings=args.get("suppress_warnings", [True])[0]
	)
	params = dict(
		p0=1-sparsity,
		update_p0=False,
		tau2=coeff_size,
		update_tau2=False,
		sigma2=1,
		update_sigma2=False,
	)
	pval = crt_model.full_p_value(
		inds=inds, 
		M=M, 
		params=params, 
		sample_kwargs=sample_kwargs
	)	
	crt_mtime = time.time() - t0
	output.append(
		['CRT', crt_mtime, pval, True] + dgp_args
	)

	# Distilled CRT
	t0 = time.time()
	dpval = crt_model.p_value(
		inds=inds,
	)
	dcrt_mtime = time.time() - t0
	output.append(
		['dCRT', dcrt_mtime, dpval, True] + dgp_args
	)

	print(f"Finished with seed={seed}.")
	sys.stdout.flush()
	return output

def main(args):
	# Parse arguments
	args = parser.parse_args(args)
	print(args['description'])
	sys.stdout.flush()
	reps = args.get('reps', [1])[0]
	num_processes = args.get('num_processes', [1])[0]

	# Save args, create output dir
	output_dir, today, hour = utilities.create_output_directory(
		args, dir_type=DIR_TYPE, return_date=True
	)
	result_path = output_dir + "/results.csv"

	# Run outputs
	time0 = time.time()
	all_outputs = []
	seed_start = max(args.get('seed_start', [1])[0], 1)
	for covmethod in args.get('covmethod', ['ark']):
		for p in args.get('p', [500]):
			for kappa in args.get('kappa',[0.2]):
				for sparsity in args.get('sparsity', [0.05]):
					for k in args.get('k', [1]):
						for y_dist in args.get('y_dist', ['gaussian']):
							for coeff_size in args.get('coeff_size', [1]):
								constant_inputs=dict(
									y_dist=y_dist,
									covmethod=covmethod,
									kappa=kappa,
									p=p,
									sparsity=sparsity,
									k=k,
									coeff_size=coeff_size,
								)
								msg = f"Finished with {constant_inputs}"
								constant_inputs['args'] = args
								outputs = utilities.apply_pool(
									func=single_seed_sim,
									seed=list(range(seed_start, reps+seed_start)), 
									constant_inputs=constant_inputs,
									num_processes=num_processes, 
								)
								msg += f" at {np.around(time.time() - time0, 2)}"
								print(msg)
								for out in outputs:
									all_outputs.extend(out)

								# Save
								q = args.get('q', [0.1])[0]
								out_df = pd.DataFrame(all_outputs, columns=COLUMNS)
								out_df['power'] = out_df['pep'] < q
								out_df.to_csv(result_path, index=False)
								groupers = [
									'method', 'y_dist', 'covmethod', 'kappa', 
									'p', 'sparsity', 'k', 'well_specified',
								]
								meas = ['model_time', 'pep', 'power']
								summary_df = out_df.groupby(groupers)[meas].agg(
									['mean', 'sem']
								)
								print(summary_df.reset_index())



if __name__ == '__main__':
	main(sys.argv)