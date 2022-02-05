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
from blip_sims.gen_data import gen_changepoint_data

from sklearn import linear_model

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# columns of dataframe
COLUMNS = [
	'method',
	'model_time',
	'blip_time',
	'power', 
	'nfd',
	'fdr',
	'well_specified',
	'nsample',
	'p',
	'sparsity',
	'coeff_size',
	'spacing',
	'reversion_prob',
	'seed', 
]

def single_seed_sim(
	seed,
	p,
	sparsity,
	coeff_size,
	spacing,
	reversion_prob,
	args,
):
	np.random.seed(seed)
	output = []
	dgp_args = [
		p, sparsity, coeff_size, spacing, reversion_prob, seed
	]

	# Create data
	X, Y, beta = gen_changepoint_data(
		T=p,
		sparsity=sparsity,
		spacing=spacing,
		coeff_dist=args.get('coeff_dist', ['normal'])[0],
		coeff_size=coeff_size,
		min_coeff=args.get('min_coeff', [0.1 * coeff_size])[0],
		reversion_prob=reversion_prob,
	)
	q = args.get('q', [0.1])[0]
	max_pep = args.get('max_pep', [2*q])[0]
	max_size = args.get('max_size', [25])[0]
	prenarrow = args.get('prenarrow', [0])[0]
	chains = args.get('chains', [10])[0]
	bsize = args.get('bsize', [5])[0]

	# Method type 1: BLiP + SpikeSlab
	for well_specified in args.get('well_specified', [False, True]):
		if well_specified:
			p0 = 1 - sparsity
			min_p0 = 0.0
			tau2 = coeff_size
			sigma2 = 1
			update = False
		else:
			p0 = 0.99
			min_p0 = 0.9
			tau2 = 1
			sigma2 = 1
			update = True
		model = pyblip.linear.LinearSpikeSlab(
			X=X,
			y=Y,
			p0=p0,
			min_p0=min_p0,
			update_p0=update,
			tau2=tau2,
			update_tau2=update,
			sigma2=sigma2,
			update_sigma2=update,
		)
		for nsample in args.get("nsample", [5000]):
			t0 = time.time()
			model.sample(N=nsample, chains=chains, bsize=bsize)
			# Add inclusions
			mtime = time.time() - t0

			# Calculate PIPs
			t0 = time.time()
			inclusions = model.betas != 0
			inclusions[:, 0] = 0
			cand_groups = pyblip.create_groups.all_cand_groups(
				inclusions=inclusions,
				X=X,
				q=q,
				max_pep=max_pep,
				max_size=max_size,
				prenarrow=prenarrow,
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
				['LSS + BLiP', mtime, blip_time, power, nfd, fdr, well_specified, nsample] + dgp_args
			)

			# Method Type 2: BCP + BLiP
			if args.get('run_bcp', [True])[0]:
				t0 = time.time()
				inclusions = blip_sims.bcp.run_bcp(
					y=Y, nsample=nsample, chains=chains, 
					p0=1-sparsity if well_specified else 0.1, 
					w0=1/(coeff_size + 1) if well_specified else 0.2 
				)
				mtime = time.time() - t0
				t0 = time.time()
				cand_groups = pyblip.create_groups.all_cand_groups(
					inclusions=inclusions, q=q, max_pep=max_pep, max_size=max_size, prenarrow=prenarrow
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
					['BCP + BLiP', mtime, blip_time, power, nfd, fdr, well_specified, nsample] + dgp_args
				)


	# Method Type 3: susie-based methods
	t0 = time.time()
	susie_alphas, susie_sets = blip_sims.susie.run_susie_trendfilter(
		Y, 0, L=np.ceil(p*sparsity), q=q
	)
	susie_time = time.time() - t0
	susie_sets = [x for x in susie_sets if len(x) < max_size]
	# Adjust by one for changepoint detection and remove meaningless
	# detections (corresponds to index p)
	susie_sets = [
		x + 1 for x in susie_sets
	]
	for j in range(len(susie_sets)):
		susie_sets[j] = susie_sets[j][susie_sets[j] != p]
	
	

	nfd, fdr, power = blip_sims.utilities.rejset_power(
		susie_sets, beta
	)
	output.append(
		['susie', susie_time, 0, power, nfd, fdr, True, 0] + dgp_args
	)
	# Now apply BLiP on top of susie
	t0 = time.time()
	if susie_alphas is not None:
		susie_alphas = susie_alphas[:, 0:-1]
		cand_groups = pyblip.create_groups.susie_groups(
			alphas=susie_alphas,
			X=X[:, 1:],
			q=q,
			prenarrow=prenarrow,
			max_size=max_size,
			max_pep=max_pep
		)
		detections = pyblip.blip.BLiP(
			cand_groups=cand_groups,
			q=q,
			error='fdr',
			max_pep=max_pep,
			perturb=True,
			deterministic=True
		)
		# Adjust index for detections for changepoint detection
		for x in detections:
			x.group = set((np.array(list(x.group)) + 1).tolist())

		blip_time = time.time() - t0
		nfd, fdr, power = utilities.nodrej2power(detections, beta)
		output.append(
			['susie + BLiP', susie_time, blip_time, power, nfd, fdr, True, 0] + dgp_args
		)
	else:
		output.append(
			['susie + BLiP', susie_time, 0, 0, 0, 0, True, 0] + dgp_args
		)

	return output

def main(args):
	# Parse arguments
	args = parser.parse_args(args)
	print(args['description'])
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
	for p in args.get('p', [500]):
		for sparsity in args.get('sparsity', [0.05]):
			for coeff_size in args.get('coeff_size', [1]):
				for spacing in args.get('spacing', ['random']):
					for reversion_prob in args.get('reversion_prob', [0]):
						constant_inputs=dict(
							p=p,
							sparsity=sparsity,
							coeff_size=coeff_size,
							spacing=spacing,
							reversion_prob=reversion_prob,
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
						out_df = pd.DataFrame(all_outputs, columns=COLUMNS)
						out_df.to_csv(result_path, index=False)
						groupers = [
							'method', 'spacing', 'reversion_prob', 'p', 'sparsity', 'coeff_size', 'well_specified', 'nsample'
						]
						meas = ['model_time', 'power', 'fdr', 'blip_time']
						summary_df = out_df.groupby(groupers)[meas].mean()
						print(summary_df.reset_index())



if __name__ == '__main__':
	main(sys.argv)