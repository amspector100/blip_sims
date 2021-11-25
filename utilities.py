"""
Utility functions for simulations.
"""

import os
import sys
import datetime
import json
import numpy as np

def create_output_directory(args, dir_type='misc'):
	# Date
	today = str(datetime.date.today())
	hour = str(datetime.datetime.today().time())
	hour = hour.replace(':','-').split('.')[0]
	output_dir = f'data/{dir_type}/{today}/{hour}/'
	# Ensure directory exists
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# Save description
	args_path = output_dir + "args.json"
	with open(args_path, 'w') as thefile:
		thefile.write(json.dumps(args))
	# return
	return output_dir
	
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
    weights = np.array([
    	n.data['weight'] for n in detections
    ])
    power = np.dot(weights, 1 - false_disc)

    return n_false_disc, fdp, power

def count_randomized_pairs(nodes, tol=1e-5):
	num_zeros = 0
	num_ones = 0
	nonints = []
	nonint_groups = []
	# First count number of zeros and ones
	for node in nodes:
		if node.data['sprob'] > 1 - tol:
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