"""
Template for running simulations.
"""

import os
import sys
import parser
import time

import numpy as np
import pandas as pd
import utilities
from context import pyblip, knockpy, hpt

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# columns of dataframe
COLUMNS = [
	'seed', 'blip_time', 'num_cand_groups', 'num_zeros', 'num_ones', 'n_singleton', 'n_rand_pairs', 
	'epower_lp', 'epower_ilp', 'epower_sample', 'nfd', 'fdp', 'power', 'kappa', 'p', 'sparsity', 'covmethod'
]

def single_seed_sim(
	seed,
	method,
	kappa,
	p,
	sparsity,
	args,
	tol=1e-3
):
	"""
	Note to self: this could also save/load data
	so that R can run the same simulation, etc.
	"""
	np.random.seed(seed)

	# Create data
	n = int(kappa * p)
	dgp = knockpy.dgp.DGP()
	dgp.sample_data(
		method=method,
		n=n,
		p=p,
		sparsity=sparsity,
		coeff_dist=args.get('coeff_dist', ['normal'])[0]
	)

	# Run linear spike slab
	model = pyblip.linear.LinearSpikeSlab(
		X=dgp.X,
		y=dgp.y,
		p0=0.99,
		min_p0=0.9,
		tau2_a0=20,
		tau2_b0=10,
	)
	model.sample(N=300, chains=10)
	inclusions = model.betas != 0

	# Calculate PIPs
	t0 = time.time()
	q = args.get('q', [0.1])[0]
	max_pep = args.get('max_pep', [q])[0]
	max_size = args.get('max_size', [25])[0]
	cand_groups = pyblip.create_groups.sequential_groups(
		inclusions,
		q=q,
		max_pep=max_pep,
		max_size=max_size,
		prenarrow=args.get('prenarrow', [1])[0],
	)
	dist_matrix = np.abs(1 - np.dot(dgp.X.T, dgp.X))
	cand_groups.extend(pyblip.create_groups.hierarchical_groups(	
			inclusions,
			dist_matrix=dist_matrix,
			max_pep=max_pep,
			max_size=max_size,
			filter_sequential=True,
	))

	# Run BLiP
	detections = pyblip.blip.BLiP(
		cand_groups=cand_groups,
		q=q,
		error='pfer',
		max_pep=max_pep,
		perturb=True,
		how_binarize='intlp'
	)
	nfd, fdr, power = utilities.nodrej2power(detections, dgp.beta)
	# Quickly calculate randomized solution
	detections_sample = pyblip.blip.binarize_selections(
		cand_groups=cand_groups,
		p=p,
		v_opt=q,
		error='pfer',
		how_binarize='sample'
	)

	epower_lp = sum([cg.data['weight'] * cg.data['sprob'] for cg in cand_groups])
	epower_ilp = sum([cg.data['weight'] for cg in detections])
	epower_sample = sum([cg.data['weight'] for cg in detections_sample])

	# Count number of non-integer cand_groups
	num_zeros, num_ones, n_single, n_pairs = utilities.count_randomized_pairs(cand_groups)

	return [
		seed, 
		np.around(time.time() - t0, 2),
		len(cand_groups),
		num_zeros,
		num_ones,
		n_single,
		n_pairs,
		epower_lp,
		epower_ilp,
		epower_sample,
		nfd,
		fdr,
		power,
		kappa,
		p,
		sparsity,
		method
	]

def main(args):
	# Parse arguments
	args = parser.parse_args(args)
	reps = args.get('reps', [1])[0]
	num_processes = args.get('num_processes', [1])[0]

	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	result_path = output_dir + "/results.csv"

	# Run outputs
	all_outputs = []
	for method in args.get('method', ['ar1']):
		for p in args.get('p', [500]):
			for kappa in args.get('kappa',[0.2]):
				for sparsity in args.get('sparsity', [0.05]):
					outputs = knockpy.utilities.apply_pool(
						func=single_seed_sim,
						seed=list(range(1, reps+1)), 
						constant_inputs=dict(
							method=method,
							kappa=kappa,
							p=p,
							sparsity=sparsity,
							args=args
						),
						num_processes=num_processes, 
					)
					all_outputs.extend(outputs)

					# Save
					out_df = pd.DataFrame(all_outputs, columns=COLUMNS)
					out_df.to_csv(result_path, index=False)
					summary_df = out_df.groupby(
						['covmethod', 'kappa', 'p', 'sparsity']
					)[[
						'power', 'nfd', 'n_singleton', 'n_rand_pairs',
						'epower_lp', 'epower_ilp', 'epower_sample', 'blip_time']].mean()
					print(summary_df.reset_index())



if __name__ == '__main__':
	main(sys.argv)