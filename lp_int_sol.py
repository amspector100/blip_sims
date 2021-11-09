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
from context import hpt, knockpy

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# columns of dataframe
COLUMNS = [
	'seed', 'blip_time', 'num_nodes', 'num_zeros', 'num_ones', 'n_singleton', 'n_rand_pairs', 
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
	model = hpt.linear.LinearSpikeSlab(
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
	max_pep = args.get('max_pep', [0.25])[0]
	max_size = args.get('max_size', [25])[0]
	nodes = hpt.calc_peps.fast_sequential_peps_posterior(
		inclusions,
		q=q,
		max_pep=max_pep,
		max_size=max_size,
		prenarrow=args.get('prenarrow', [1])[0],
	)
	dist_matrix = np.abs(1 - np.dot(dgp.X.T, dgp.X))
	nodes.extend(hpt.calc_peps.tree_peps_posterior(	
			inclusions,
			dist_matrix=dist_matrix,
			max_pep=max_pep,
			max_size=max_size,
			filter_sequential=True,
	))

	# Run BLiP
	rej_nodes = hpt.blr.BLiP(
		nodes=nodes,
		q=q,
		error='fdr',
		max_pep=max_pep,
		perturb=True,
		how_binarize='intlp'
	)
	nfd, fdr, power = utilities.nodrej2power(rej_nodes, dgp.beta)
	# Quickly calculate randomized solution
	rej_nodes_sample = hpt.blr.binarize_selections(
		nodes=nodes,
		p=p,
		nfd_val=sum([n.data['pep'] * n.data['sprob'] for n in nodes]),
		error='fdr',
		how_binarize='sample'
	)

	epower_lp = sum([node.data['util'] * node.data['sprob'] for node in nodes])
	epower_ilp = sum([node.data['util'] for node in rej_nodes])
	epower_sample = sum([node.data['util'] for node in rej_nodes_sample])

	# Count number of non-integer nodes
	num_zeros, num_ones, n_single, n_pairs = utilities.count_randomized_pairs(nodes)

	return [
		seed, 
		np.around(time.time() - t0, 2),
		len(nodes),
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
						'power', 'fdp', 'n_singleton', 'n_rand_pairs',
						'epower_lp', 'epower_ilp', 'epower_sample', 'blip_time']].mean()
					print(summary_df.reset_index())



if __name__ == '__main__':
	main(sys.argv)