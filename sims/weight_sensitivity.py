"""
Template for running simulations.
"""

import os
import sys
import time
import copy

import numpy as np
import pandas as pd
from context import pyblip
from context import blip_sims
from blip_sims import utilities, gen_data
from blip_sims import parser
from blip_sims.utilities import elapsed

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

COLUMNS = [
	'seed',
	'kappa', 
	'p',
	'sparsity',
	'avg_jaccard',
	'power1',
	'fdr1',
	'power2',
	'fdr2',
]

def compute_avg_jaccard(detect1, detect2, _recursive=True):
	jaccards = []
	if len(detect1) == 0 and len(detect2) == 0:
		return 1.0
	elif len(detect1) == 0:
		raise ValueError(f"Weird that detect1={detect1}, detect2={detect2}")

	# find max_{cg2} (jaccard(cg1, cg2)) for each cg1
	for cg1 in detect1:
		n1 = len(cg1.group)
		max_jaccard = 0.0
		for cg2 in detect2:
			noverlap = len(cg1.group.intersection(cg2.group))
			nunion = len(cg1.group.union(cg2.group))
			max_jaccard = max(max_jaccard, noverlap / nunion)
		jaccards.append(max_jaccard)

	# Repeat for cgs in detect2
	if _recursive:
		jaccards.extend(compute_avg_jaccard(
			detect2, detect1, _recursive=False
		))
		return np.mean(jaccards)
	else:
		return jaccards


def single_seed_sim(**args):
	"""
	Note to self: this could also save/load data
	so that R can run the same simulation, etc.
	"""
	# Get parameters
	kappa = args.get("kappa")
	p = args.get("p")
	sparsity = args.get("sparsity")
	seed = args.get("seed")
	n = int(kappa * p)
	dgp_args = [seed, kappa, p, sparsity]
	t0 = args.get("t0")
	print(f"At n={n}, kappa={kappa}, seed={seed} at {elapsed(t0)}.")

	# Create data
	np.random.seed(seed)
	sample_kwargs = dict(
		y_dist=args.get("y_dist", 'gaussian'),
		covmethod='ark',
		n=n,
		p=p,
		sparsity=sparsity,
		k=args.get('k', 3),
		coeff_dist=args.get('coeff_dist', 'normal'),
		coeff_size=args.get('coef_size', 1),
	)
	X, y, beta = gen_data.generate_regression_data(**sample_kwargs)
	nsignal = np.sum(beta != 0)

	# Fit LSS
	model = pyblip.linear.LinearSpikeSlab(
		X=X, y=y.astype(np.float64), min_p0=args.get("min_p0", 0.9),
	)
	model.sample(
		N=args.get("nsample", 1000),
		chains=args.get("chains", 5),
		bsize=args.get("bsize", 1),
	)

	# Generate candidate groups
	q = args.get("q", 0.05)
	cgs = pyblip.create_groups.all_cand_groups(
		samples=model.betas,
		q=q,
	)
	cgs_cpy = [copy.deepcopy(x) for x in cgs]

	# Run BLiP for two weight functions
	max_pep = args.get("max_pep", 0.2)
	detect1 = pyblip.blip.BLiP(
		cand_groups=cgs, q=q, weight_fn='inverse_size', verbose=False
	)
	_, fdr1, power1 = utilities.nodrej2power(detect1, beta)
	power1 /= nsignal
	detect2 = pyblip.blip.BLiP(
		cand_groups=cgs_cpy, q=q, weight_fn='log_inverse_size', verbose=False
	)
	_, fdr2, power2 = utilities.nodrej2power(detect2, beta)
	power2 /= nsignal
	avg_jaccard = compute_avg_jaccard(detect1, detect2)
	return [
		dgp_args + [avg_jaccard, power1, fdr1, power2, fdr2]
	]


def main(args):
	# Parse arguments
	args = parser.parse_args(args)

	# Key defaults go here
	args['p'] = args.get('p', [100])
	args['kappa'] = args.get('kappa', [2])
	args['sparsity'] = args.get('sparsity', [0.1])
	args['t0'] = [time.time()]

	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	args.pop("description")
	reps = args.pop('reps', [1])[0]
	num_processes = args.pop('num_processes', [1])[0]


	# Run outputs
	outputs = utilities.apply_pool_factorial(
		func=single_seed_sim,
		seed=list(range(1, reps+1)), 
		num_processes=num_processes, 
		**args,
	)
	all_out = []
	for x in outputs:
		all_out.extend(x)
	out_df = pd.DataFrame(all_out, columns=COLUMNS)
	out_df.to_csv(output_dir + "results.csv", index=False)
	print(
	out_df.groupby(
		['p', 'kappa', 'sparsity']
	)['avg_jaccard', 'power1', 'fdr1'].mean()
	)

if __name__ == '__main__':
	main(sys.argv)