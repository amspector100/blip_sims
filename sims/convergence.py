"""
Template for running simulations.
"""

import os
import sys
import time
import copy
import json

import numpy as np
import pandas as pd
from context import pyblip, blip_sims
from blip_sims import parser
from blip_sims import utilities
from blip_sims.gen_data import generate_regression_data

from sklearn import linear_model
import scipy.cluster.hierarchy as hierarchy

# Global seed for dgp
DGP_SEED = 111

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# columns of dataframe
COLUMNS = [
	'method',
	'bsize',
	'model_time',
	'blip_time',
	'power', 
	'nfd',
	'fdr',
	'well_specified',
	'nsample',
	'chains',
	'y_dist',
	'covmethod',
	'kappa',
	'p',
	'sparsity',
	'k',
	'rank',
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
	rank,
	coeff_size,
	args,
	tol=1e-3
):
	np.random.seed(seed)
	output = []
	dgp_args = [
		y_dist, covmethod, kappa, p, sparsity, k, rank, coeff_size, seed
	]

	# Create data
	n = int(kappa * p)
	X, y, beta = generate_regression_data(
		y_dist=y_dist,
		covmethod=covmethod,
		n=n,
		p=p,
		sparsity=sparsity,
		k=k,
		rank=rank,
		coeff_dist=args.get('coeff_dist', ['normal'])[0],
		coeff_size=coeff_size,
		min_coeff=args.get('min_coeff', [0.1 * coeff_size])[0],
		dgp_seed=DGP_SEED
	)

	q = args.get('q', [0.1])[0]
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
			method_names = ['LSS + BLiP']
		else:
			models = []
			method_names = []
		if y_dist != 'gaussian':
			models.append(pyblip.probit.ProbitSpikeSlab(
				X=X,
				y=y,
				p0=p0,
				min_p0=min_p0,
				update_p0=update,
				tau2=tau2,
				update_tau2=update,
				sigma2=sigma2,
				update_sigma2=update,
			))
			method_names.append('PSS + BLiP')
		for nsample in args.get("nsample", [1000]):
			for model, mname in zip(models, method_names):
				for bsize in args.get("bsize", [1, 3, 5]):
					chains_vals = args.get("chains", [10])
					max_chains = max(chains_vals)
					t0 = time.time()
					model.sample(
						N=nsample, 
						chains=max_chains,
						burn=int(0.1*nsample),
						bsize=bsize
					)
					mtime = time.time() - t0
					for chains in chains_vals:
						t0 = time.time()
						inclusions = model.betas[0:int(chains*nsample)] != 0
						max_pep = args.get('max_pep', [2*q])[0]
						max_size = args.get('max_size', [25])[0]
						prenarrow = args.get('prenarrow', [0])[0]
						cand_groups = pyblip.create_groups.all_cand_groups(	
								samples=inclusions,
								X=X,
								max_pep=max_pep,
								max_size=max_size,
						)

						# Run BLiP
						detections = pyblip.blip.BLiP(
							cand_groups=cand_groups,
							q=q,
							error='fdr',
							max_pep=max_pep,
							perturb=True,
							deterministic=True
						)
						blip_time = time.time() - t0
						nfd, fdr, power = utilities.nodrej2power(detections, beta)
						output.append(
							[mname, bsize, chains * mtime / max_chains, blip_time, power, nfd, fdr, well_specified, nsample, chains] + dgp_args
						)

	# Method Type 2: susie-based methods
	if args.get('run_susie', [False])[0]:
		t0 = time.time()
		susie_alphas, susie_sets = blip_sims.susie.run_susie(
			X, y, L=np.ceil(p*sparsity), q=q
		)
		susie_time = time.time() - t0
		nfd, fdr, power = blip_sims.utilities.rejset_power(
			susie_sets, beta
		)
		output.append(
			['susie', susie_time, 0, power, nfd, fdr, True, 0] + dgp_args
		)
		# Now apply BLiP on top of susie
		t0 = time.time()
		cand_groups = pyblip.create_groups.susie_groups(
			alphas=susie_alphas, 
			X=X, 
			q=q,
			prenarrow=prenarrow,
			max_size=max_size
		)
		detections = pyblip.blip.BLiP(
			cand_groups=cand_groups,
			q=q,
			error='fdr',
			max_pep=max_pep,
			perturb=True,
			deterministic=True
		)
		blip_time = time.time() - t0
		nfd, fdr, power = utilities.nodrej2power(detections, beta)
		output.append(
			['susie + BLiP', 'all', susie_time, blip_time, power, nfd, fdr, True, 0, 0] + dgp_args
		)

	# # Frequentist methods
	# if kappa > 1:
	# 	t0 = time.time()
	# 	regtree = blip_sims.tree_methods.RegressionTree(
	# 		X=X, y=y, levels=args.get('levels', [10])[0], max_size=max_size
	# 	)
	# 	regtree.fit(family='gaussian') # this works better than using family = binomial 
	# 	# even when y follows a binomial distribution
	# 	mtime = time.time() - t0
	# 	_, rej_yek = regtree.ptree.outer_nodes_yekutieli(q=q)
	# 	rej_fbh, _ = regtree.ptree.tree_fbh(q=q)
	# 	for mname, rej in zip(['FBH', 'Yekutieli'], [rej_fbh, rej_yek]):
	# 		nfd, fdr, power = utilities.nodrej2power(rej, beta)
	# 		output.append(
	# 			[mname, mtime, 0, power, nfd, fdr, True, 0] + dgp_args
	# 		)
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
								for rank in args.get('rank', [1]):
									sample_kwargs=dict(
										y_dist=y_dist,
										covmethod=covmethod,
										kappa=kappa,
										p=p,
										sparsity=sparsity,
										k=k,
										rank=rank,
										coeff_size=coeff_size,
									)
									# Inputs to multiprocessing
									constant_inputs = sample_kwargs.copy()
									constant_inputs['args'] = args
									seeds = list(range(seed_start, reps+seed_start))
									outputs = utilities.apply_pool(
										func=single_seed_sim,
										seed=seeds, 
										constant_inputs=constant_inputs,
										num_processes=num_processes, 
									)
									for out in outputs:
										all_outputs.extend(out)
									msg = f"Finished with {sample_kwargs}"
									msg += f" at {np.around(time.time() - time0, 2)}"
									print(msg)

									# Save
									out_df = pd.DataFrame(all_outputs, columns=COLUMNS)
									out_df.to_csv(result_path, index=False)
									groupers = [
										'method', 'bsize', 'chains', 'y_dist', 'covmethod', 'kappa', 'rank',
										'p', 'sparsity', 'k', 'well_specified', 'nsample',
									]
									meas = ['model_time', 'power', 'fdr', 'blip_time']
									summary_df = out_df.groupby(groupers)[meas].mean()
									print(summary_df.reset_index())



if __name__ == '__main__':
	main(sys.argv)