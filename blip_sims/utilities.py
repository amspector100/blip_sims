"""
Utility functions for simulations.
"""

import os
import sys
import time
import datetime
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

def elapsed(t0):
	return np.around(time.time() - t0, 2)

### Multiprocessing helper
def _one_arg_function(list_of_inputs, args, func, kwargs):
	"""
	Globally-defined helper function for pickling in multiprocessing.
	:param list of inputs: List of inputs to a function
	:param args: Names/args for those inputs
	:param func: A function
	:param kwargs: Other kwargs to pass to the function. 
	"""
	new_kwargs = {}
	for i, inp in enumerate(list_of_inputs):
		new_kwargs[args[i]] = inp
	return func(**new_kwargs, **kwargs)


def apply_pool(func, constant_inputs={}, num_processes=1, **kwargs):
	"""
	Spawns num_processes processes to apply func to many different arguments.
	This wraps the multiprocessing.pool object plus the functools partial function. 
	
	Parameters
	----------
	func : function
		An arbitrary function
	constant_inputs : dictionary
		A dictionary of arguments to func which do not change in each
		of the processes spawned, defaults to {}.
	num_processes : int
		The maximum number of processes spawned, defaults to 1.
	kwargs : dict
		Each key should correspond to an argument to func and should
		map to a list of different arguments.
	Returns
	-------
	outputs : list
		List of outputs for each input, in the order of the inputs.
	Examples
	--------
	If we are varying inputs 'a' and 'b', we might have
	``apply_pool(
		func=my_func, a=[1,3,5], b=[2,4,6]
	)``
	which would return ``[my_func(a=1, b=2), my_func(a=3,b=4), my_func(a=5,b=6)]``.
	"""

	# Construct input sequence
	args = sorted(kwargs.keys())
	num_inputs = len(kwargs[args[0]])
	for arg in args:
		if len(kwargs[arg]) != num_inputs:
			raise ValueError(f"Number of inputs differs for {args[0]} and {arg}")
	inputs = [[] for _ in range(num_inputs)]
	for arg in args:
		for j in range(num_inputs):
			inputs[j].append(kwargs[arg][j])

	# Construct partial function
	partial_func = partial(
		_one_arg_function, args=args, func=func, kwargs=constant_inputs,
	)

	# Don't use the pool object if num_processes=1
	num_processes = min(num_processes, len(inputs))
	if num_processes == 1:
		all_outputs = []
		for inp in inputs:
			all_outputs.append(partial_func(inp))
	else:
		with Pool(num_processes) as thepool:
			all_outputs = thepool.map(partial_func, inputs)

	return all_outputs

def create_output_directory(args, dir_type='misc', return_date=False):
	# Date
	today = str(datetime.date.today())
	hour = str(datetime.datetime.today().time())
	hour = hour.replace(':','-').split('.')[0]
	# Output directory
	file_dir = os.path.dirname(os.path.abspath(__file__))
	parent_dir = os.path.split(file_dir)[0]
	output_dir = f'{parent_dir}/data/{dir_type}/{today}/{hour}/'
	# Ensure directory exists
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# Save description
	args_path = output_dir + "args.json"
	with open(args_path, 'w') as thefile:
		thefile.write(json.dumps(args))
	# Return 
	if return_date:
		return output_dir, today, hour
	return output_dir

def calc_pep_df(beta, cand_groups, alphas):
	rows = []
	for cg in cand_groups:
		g = list(cg.group)
		susie_pep = 1 - np.max(np.sum(alphas[:, g], axis=1))
		rows.append([susie_pep, cg.pep, np.all(beta[g] == 0), len(g)])
	pep_df = pd.DataFrame(
		rows, columns=['pep', 'susie_pep', 'null', 'size']
	)
	#pep_df['bin'] = pd.cut(pep_df['pep'], bins=np.arange(21) / 20)
	#calib = pep_df.groupby('bin')['null'].agg(['count', 'sum', 'mean', 'std'])
	#calib['se'] = calib['std'] / np.sqrt(calib['sum'])
	return pep_df

def create_dap_prefix(today, hour, **kwargs):
	# Output directory
	file_dir = os.path.dirname(os.path.abspath(__file__))
	parent_dir = os.path.split(file_dir)[0]
	file_prefix = f'{parent_dir}/data/dap_data/{today}/{hour}/'
	# Add keys
	for key in kwargs:
		file_prefix += f"{key}{kwargs[key]}"
	file_prefix += "/"
	# Ensure directory exists
	if not os.path.exists(file_prefix):
		os.makedirs(file_prefix)
	return file_prefix + "seed"

def rejset_power(rej_sets, beta):
	power = 0
	nfd = 0
	for s in rej_sets:
		if np.any(beta[s] != 0):
			power += 1 / len(s)
		else:
			nfd += 1
	fdp = nfd / max(1, len(rej_sets))
	return nfd, fdp, power
	
def nodrej2power(
	detections,
	beta
):
	# Calculate FDP
	false_disc = np.array([
		np.all(beta[list(n.group)]==0) for n in detections
	])
	n_false_disc = np.sum(false_disc)
	fdp = n_false_disc / max(1, len(detections))

	# Calculate power
	try:
		weights = np.array([
			n.data['weight'] for n in detections
		])
	except KeyError: # for pval nodes
		weights = np.array([1 / len(n.group) for n in detections])
	power = np.dot(weights, 1 - false_disc)

	return n_false_disc, fdp, power

def count_randomized_pairs(nodes, tol=1e-5):
	num_zeros = 0
	num_ones = 0
	nonints = []
	nonint_groups = []
	# First count number of zeros and ones
	for node in nodes:
		if 'sprob' not in node.data:
			num_zeros += 1
		elif node.data['sprob'] > 1 - tol:
			num_ones += 1
		elif node.data['sprob'] < tol:
			num_zeros += 1
		else:
			nonints.append(node.data['sprob'])
			nonint_groups.append(set(node.group))

	# Find randomized pairs / singletons lurking about
	rand_pairs, rand_singletons = _count_rand_pairs_inner(
		nonints, nonint_groups, tol=tol
	)
	return num_zeros, num_ones, len(rand_singletons), len(rand_pairs)


def _count_rand_pairs_inner(nonints, groups, tol=1e-5):
	"""
	Given a set of non integer values, find the randomized pairs
	and the singletons.
	"""
	m = len(nonints)
	rand_singletons = set(list(range(m)))
	rand_pairs = []
	for j in range(m):
		if j not in rand_singletons:
			continue
		for k in rand_singletons:
			if np.abs(nonints[j] + nonints[k] - 1) < 1e-5:
				if len(groups[j].intersection(groups[k])) > 0:
					rand_pairs.append((j, k))
					rand_singletons -= set([j, k])
					break


	return rand_pairs, rand_singletons