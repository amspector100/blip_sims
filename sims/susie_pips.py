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
from blip_sims import parser
from blip_sims import utilities
from blip_sims.gen_data import generate_regression_data

from sklearn import linear_model

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# columns of dataframe
COLUMNS = [
	'method',
	'model_time',
	'blip_time',
	'power', 
	'fdr',
	'L',
	'sparsity',
	'covmethod',
	'kappa',
	'p',
	'k',
	'coeff_size',
	'seed', 
]

def single_seed_sim(
	seed,
	L,
	sparsity,
	covmethod,
	kappa,
	p,
	k,
	coeff_size,
	args,
	tol=1e-3
):
	np.random.seed(seed)
	output = []
	common_args = [
		L, sparsity, covmethod, kappa, p, k, coeff_size, seed
	]

	# Create data
	n = int(kappa * p)
	sample_kwargs = dict(
		covmethod=covmethod,
		n=n,
		p=p,
		sparsity=sparsity,
		k=k,
		coeff_dist=args.get('coeff_dist', ['normal'])[0],
		coeff_size=coeff_size,
		min_coeff=args.get('min_coeff', [0.1 * coeff_size])[0],
		dgp_seed=seed,
		return_cov=True
	)
	X, y, beta, V = generate_regression_data(**sample_kwargs)

	# Parse args for cand groups
	q = args.get("q", [0.1])[0]
	max_pep = args.get('max_pep', [0.5])[0]
	max_size = args.get('max_size', [25])[0]
	prenarrow = args.get('prenarrow', [False])[0]

	# Fit SuSiE
	t0 = time.time()
	susie_alphas, susie_sets = blip_sims.susie.run_susie(
		X, y, L=L, q=q
	)
	susie_time = time.time() - t0
	nfd, fdr, power = blip_sims.utilities.rejset_power(
		susie_sets, beta
	)
	output.append(
		['susie', susie_time, 0, power, fdr] + common_args
	)

	# Fit BLiP on top of SuSiE
	t0 = time.time()
	cand_groups = pyblip.create_groups.susie_groups(
		alphas=susie_alphas, 
		X=X, 
		q=q,
		prenarrow=prenarrow,
		max_size=max_size,
		max_pep=max_pep
	)
	detections = pyblip.blip.BLiP(
		cand_groups=cand_groups,
		q=q,
		error=args.get('error', ['fdr'])[0],
		max_pep=max_pep,
		perturb=True,
		deterministic=True
	)
	blip_time = time.time() - t0
	nfd, fdr, power = utilities.nodrej2power(detections, beta)
	output.append(
		['susie + BLiP', susie_time, blip_time, power, fdr] + common_args
	)

	# PEP-based outputs
	for cg in detections:
		cg.data['detected'] = True
	pep_df = utilities.calc_pep_df(
		beta=beta, cand_groups=cand_groups, alphas=susie_alphas
	)
	pep_df['seed'] = seed
	print(f"Done with seed={seed}.")
	return output, pep_df

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
	pep_path = output_dir + "/peps.csv"

	# Run outputs
	time0 = time.time()
	all_outputs = []
	all_pep_dfs = []
	seed_start = max(args.get('seed_start', [1])[0], 1)
	for covmethod in args.get('covmethod', ['ark']):
		for p in args.get('p', [500]):
			for kappa in args.get('kappa',[0.2]):
				for sparsity in args.get('sparsity', [0.05]):
					for k in args.get('k', [1]):
						for L in args.get('l', [np.ceil(p*sparsity)]):
							for coeff_size in args.get('coeff_size', [1]):
								constant_inputs=dict(
									L=L,
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
									all_outputs.extend(out[0])
									pep_df = out[1]
									for key in constant_inputs:
										if key != 'args':
											pep_df[key] = constant_inputs[key]
									all_pep_dfs.append(pep_df)

								# Save power df
								out_df = pd.DataFrame(all_outputs, columns=COLUMNS)
								out_df.to_csv(result_path, index=False)
								groupers = [
									'method', 'L', 'covmethod', 'kappa', 
									'p', 'sparsity', 'k'
								]
								meas = ['model_time', 'blip_time', 'power', 'fdr']
								summary_df = out_df.groupby(groupers)[meas].mean()
								print(summary_df.reset_index())

								# save pep df
								final_pep_df = pd.concat(all_pep_dfs, axis='index')
								final_pep_df.to_csv(pep_path, index=False)



if __name__ == '__main__':
	main(sys.argv)