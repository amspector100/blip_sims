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

# Global seed for dgp
DGP_SEED = 111

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
	dap_prefix,
	tol=1e-3
):
	np.random.seed(seed)
	output = []
	dgp_args = [
		y_dist, covmethod, kappa, p, sparsity, k, coeff_size, seed
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
		coeff_dist=args.get('coeff_dist', ['normal'])[0],
		coeff_size=coeff_size,
		min_coeff=args.get('min_coeff', [0.1 * coeff_size])[0],
		dgp_seed=DGP_SEED
	)

	# Method 0: DAP
	q = args.get('q', [0.1])[0]
	if p <= 1000 and args.get("run_dap", [False])[0]:
		t0 = time.time()
		rej_dap, _, _ = blip_sims.dap.run_dap(
			X=X, 
			y=y, 
			q=q, 
			file_prefix=dap_prefix + str(seed), 
			pi1=str(sparsity),
			msize=str(1.1 * sparsity * p),
		)
		nfd, fdr, power = utilities.rejset_power(rej_dap, beta=beta)
		dap_time = time.time() - t0
		output.append(
			["dap-g", dap_time, 0, power, nfd, fdr, True, 0] + dgp_args
		)

	for well_specified in args.get('well_specified', [False, True]):
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
				X=X, y=y.astype(int), p0=p0, update_p0=update, #min_p0=min_p0 TODO
			))
			method_names.append('PSS + BLiP')
		chains = args.get("chains", [10])[0]
		bsize = args.get("bsize", [1])[0]
		for nsample in args.get("nsample", [1000]):
			for model, mname in zip(models, method_names):
				t0 = time.time()
				# Assemble arguments for sampling
				skwargs = {'N':nsample, 'chains':chains, 'burn':int(0.1*nsample)}
				if y_dist == 'gaussian':
					skwargs['bsize']= bsize
				# Sample
				model.sample(**skwargs)
				inclusions = model.betas != 0
				mtime = time.time() - t0
				#print(f"min_p0={min_p0}")
				#print(model.p0s.mean())
				#print(model.p0s)
				# Calculate PIPs
				t0 = time.time()
				max_pep = args.get('max_pep', [2*q])[0]
				max_size = args.get('max_size', [25])[0]
				prenarrow = args.get('prenarrow', [0])[0]
				cand_groups = pyblip.create_groups.sequential_groups(
					inclusions,
					q=q,
					max_pep=max_pep,
					max_size=max_size,
					prenarrow=prenarrow,
				)
				dist_matrix = np.abs(1 - np.corrcoef(X.T))
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
					error='fdr',
					max_pep=max_pep,
					perturb=True,
					deterministic=True
				)
				blip_time = time.time() - t0
				nfd, fdr, power = utilities.nodrej2power(detections, beta)
				output.append(
					[mname, mtime, blip_time, power, nfd, fdr, well_specified, nsample] + dgp_args
				)

				# Save posterior samples for well-specified case
				# for use as test-statistics for frequentist methods
				if well_specified:
					post_prefix = dap_prefix.replace('dap_data', 'oracle_pval')
					np.save(
						post_prefix + str(seed) + ".npy",
						model.betas
					)

	# Method Type 2: susie-based methods
	if args.get('run_susie', [True])[0]:
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
			['susie + BLiP', susie_time, blip_time, power, nfd, fdr, True, 0] + dgp_args
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
								sample_kwargs=dict(
									y_dist=y_dist,
									covmethod=covmethod,
									kappa=kappa,
									p=p,
									sparsity=sparsity,
									k=k,
									coeff_size=coeff_size,
								)
								# Create directory for DAP
								dap_prefix = blip_sims.utilities.create_dap_prefix(
									today=today,
									hour=hour,
									**sample_kwargs,
								)
								# Create directory for caching posterior samples
								post_prefix = dap_prefix.replace('dap_data', 'oracle_pval')
								print(f'post_prefix={post_prefix}')
								if not os.path.exists(post_prefix):
									os.makedirs(post_prefix)
								# Inputs to multiprocessing
								constant_inputs = sample_kwargs.copy()
								constant_inputs['args'] = args
								constant_inputs['dap_prefix'] = dap_prefix
								seeds = list(range(seed_start, reps+seed_start))
								outputs = utilities.apply_pool(
									func=single_seed_sim,
									seed=seeds, 
									constant_inputs=constant_inputs,
									num_processes=num_processes, 
								)
								msg = f"Finished with {sample_kwargs}"
								msg += f" at {np.around(time.time() - time0, 2)}"
								print(msg)
								os.rmdir(os.path.dirname(dap_prefix))
								for out in outputs:
									all_outputs.extend(out)

								# Run Yekutieli/FBH]
								# Step 1: load posterior samples
								t0 = time.time()
								beta_fnames = [
									post_prefix + str(seed) + ".npy" for seed in seeds 								
								]
								print(f"Finished loading beta samples, took {blip_sims.utilities.elapsed(t0)}.")

								# Step 2: calculate p-values
								p_out = blip_sims.tree_methods.compute_bayesian_pvals(
									beta_fnames=beta_fnames,
									dgp_seed=DGP_SEED,
									sample_kwargs=sample_kwargs,
									qbins=np.arange(11) / 10,
									levels=args.get('levels', [10])[0],
									max_size=args.get('freq_max_size', [25])[0],
									how_compute=args.get('how_compute', ['ref_dist'])[0],
								)
								pvals, group_attr, group_dict, peps, beta, regtree = p_out # unpack

								# Print/check calibration
								calib = utilities.check_pep_calibration(
									beta=beta, peps=peps, group_dict=group_dict
								)
								print(calib)

								# Step 3: cache p-values and peps
								group_attr.to_csv(post_prefix + "group_attr.csv")
								np.savetxt(post_prefix + "beta.npy", (beta != 0).astype(bool))
								for obj, file in zip(
									[pvals, group_dict, peps], ['pvals', 'group_dict', 'peps']
								):
									with open(post_prefix + file + '.json', 'w') as thefile:
										json.dump(obj, thefile)

								# Delete cache of posterior samples
								os.rmdir(post_prefix)
								for bf in beta_fnames:
									os.remove(bf)

								# Step 4: run yekutieli/FBH and add to output
								dgp_args = [
									y_dist, covmethod, kappa, p, sparsity, k, coeff_size
								]
								q = args.get('q', [0.1])[0]
								for j, seed in enumerate(seeds):
									# Reset the p-values
									for i, node in enumerate(regtree.ptree.nodes):
										assert set(group_dict[i]) == node.group
										node.p = pvals[i][j]
										
									# Run FBH
									t0 = time.time()
									_, rej_yek = regtree.ptree.outer_nodes_yekutieli(q=q)
									ytime = time.time() - t0
									t0 = time.time()
									rej_fbh, _ = regtree.ptree.tree_fbh(q=q)
									fbhtime = time.time() - t0
									for mname, rej, itime in zip(
										['FBH', 'Yekutieli'], 
										[rej_fbh, rej_yek],
										[fbhtime, ytime]
									):
										nfd, fdr, power = blip_sims.utilities.nodrej2power(rej, beta)
										all_outputs.append(
												[mname, 0, itime, power, nfd, fdr, True, 0] + dgp_args + [seed]
										)

								# Save
								out_df = pd.DataFrame(all_outputs, columns=COLUMNS)
								out_df.to_csv(result_path, index=False)
								groupers = [
									'method', 'y_dist', 'covmethod', 'kappa', 
									'p', 'sparsity', 'k', 'well_specified', 'nsample'
								]
								meas = ['model_time', 'power', 'fdr', 'blip_time']
								summary_df = out_df.groupby(groupers)[meas].mean()
								print(summary_df.reset_index())



if __name__ == '__main__':
	main(sys.argv)