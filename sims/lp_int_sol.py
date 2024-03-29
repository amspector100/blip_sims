"""
Template for running simulations.
"""

import os
import sys
import time
import copy

import numpy as np
import pandas as pd
from context import pyblip, blip_sims
from blip_sims.gen_data import generate_regression_data
import blip_sims.utilities as utilities
from blip_sims.utilities import elapsed
import blip_sims.parser as parser

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# columns of dataframe
COLUMNS = [
	'seed', 'blip_time', 'num_cand_groups', 'num_zeros', 'num_ones', 'n_singleton', 'n_rand_pairs', 
	'epower_lp', 'epower_ilp', 
	#'epower_sample',
	 'nfd', 'fdp', 'power', 'btrack_iter',
	'kappa', 'p', 'sparsity', 'covmethod', 'error'
]

def single_seed_sim(
	t0,
	seed,
	covmethod,
	kappa,
	p,
	sparsity,
	args,
	tol=1e-3,
):
	"""
	Note to self: this could also save/load data
	so that R can run the same simulation, etc.
	"""
	np.random.seed(seed)
	print(f"At seed={seed}, covmethod={covmethod}, kappa={kappa}, p={p} at {elapsed(t0)}.")
	output = []

	# Create data
	n = int(kappa * p)
	X, y, beta = generate_regression_data(
		covmethod=covmethod,
		n=n,
		p=p,
		sparsity=sparsity,
		coeff_size=args.get('coeff_size', [0.2])[0],
		coeff_dist=args.get('coeff_dist', ['normal'])[0],
		max_corr=args.get('max_corr', [0.99])[0],
		k=args.get('k', [2])[0],
		alpha0=args.get("alpha0", [0.2])[0],
	)

	# Run linear spike slab
	model = pyblip.linear.LinearSpikeSlab(
		X=X,
		y=y,
	)
	nsample = args.get('nsample', [1000])[0]
	chains = args.get('chains', [10])[0]
	bsize = args.get('bsize', [5])[0]
	model.sample(N=nsample, chains=chains, bsize=bsize)
	inclusions = model.betas != 0

	# Calculate PIPs
	q = args.get('q', [0.1])[0]
	max_pep = args.get('max_pep', [2*q])[0]
	max_size = args.get('max_size', [25])[0]
	cand_groups = pyblip.create_groups.all_cand_groups(
		inclusions,
		X=X,
		q=q,
		max_pep=max_pep,
		max_size=max_size,
		prenarrow=args.get('prenarrow', [1])[0],
	)
	# dist_matrix = np.abs(1 - np.dot(X.T, X))
	# cand_groups.extend(pyblip.create_groups.hierarchical_groups(	
	# 		inclusions,
	# 		dist_matrix=dist_matrix,
	# 		max_pep=max_pep,
	# 		max_size=max_size,
	# 		filter_sequential=True,
	# ))

	# Run BLiP
	for error in args.get('error', ['fdr', 'fwer', 'local_fdr', 'pfer']):
		t0 = time.time()
		cgs = [copy.deepcopy(cg) for cg in cand_groups]
		detections, status = pyblip.blip.BLiP(
			samples=inclusions, # for fwer search
			cand_groups=cgs,
			q=q,
			error=error,
			max_pep=max_pep,
			perturb=True,
			deterministic=True,
			return_problem_status=True,
			search_method='binary',
		)
		nfd, fdr, power = utilities.nodrej2power(detections, beta)
		# Count number of non-integer cand_groups
		num_zeros, num_ones, n_single, n_pairs = utilities.count_randomized_pairs(cgs)
		# Calculate randomized solution, which is not recommended
		# detections_sample = pyblip.blip.BLiP(
		# 	samples=inclusions,
		# 	cand_groups=[copy.deepcopy(cg) for cg in cand_groups],
		# 	q=q,
		# 	error=error,
		# 	max_pep=max_pep,
		# 	perturb=True,
		# 	deterministic=False,
		# 	return_problem_status=False,
		# )
		# Final expected power bound calculations
		epower_lp = status['lp_bound']
		epower_ilp = sum([cg.data['weight'] * (1-cg.pep) for cg in detections])
		#epower_sample = sum([cg.data['weight'] * (1-cg.pep) for cg in detections_sample])

		output.append([
			seed, 
			np.around(time.time() - t0, 2),
			len(cand_groups),
			num_zeros,
			num_ones,
			n_single,
			n_pairs,
			epower_lp,
			epower_ilp,
			#epower_sample,
			nfd,
			fdr,
			power,
			status['backtracking_iter'],
			kappa,
			p,
			sparsity,
			covmethod,
			error
		])
	return output

def main(args):
	# Parse arguments
	t0 = time.time()
	args = parser.parse_args(args)
	reps = args.get('reps', [1])[0]
	num_processes = args.get('num_processes', [1])[0]
	seed_start = max(args.get('seed_start', [1])[0], 1)
	job_id = int(args.pop("job_id", [0])[0])

	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	result_path = output_dir + f"/results_id{job_id}_seedstart{seed_start}.csv"

	# Run outputs
	all_outputs = []
	for covmethod in args.get('covmethod', ['ark']):
		for p in args.get('p', [500]):
			for kappa in args.get('kappa',[0.2]):
				for sparsity in args.get('sparsity', [0.05]):
					outputs = utilities.apply_pool(
						func=single_seed_sim,
						seed=list(range(seed_start, reps+seed_start)), 
						constant_inputs=dict(
							covmethod=covmethod,
							kappa=kappa,
							p=p,
							sparsity=sparsity,
							args=args,
							t0=t0,
						),
						num_processes=num_processes, 
					)
					for out in outputs:
						all_outputs.extend(out)

					# Save
					out_df = pd.DataFrame(all_outputs, columns=COLUMNS)
					out_df.to_csv(result_path, index=False)
					summary_df = out_df.groupby(
						['covmethod', 'kappa', 'p', 'sparsity', 'error']
					)[[
						'power', 'nfd', 'btrack_iter', 'n_singleton', 'n_rand_pairs',
						'epower_lp', 'epower_ilp', 'blip_time']].mean()
					print(summary_df.reset_index())



if __name__ == '__main__':
	main(sys.argv)