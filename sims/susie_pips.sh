#!/bin/sh
NREPS=1
NPROCESSES=1 

ARGS="
	--reps $NREPS
	--num_processes $NPROCESSES
	--p 200
	--sparsity [0.01,0.02,0.03,0.04,0.05]
	--kappa 0.2
	--L 10
	--max_pep 0.2
	--error fdr
	--q 0.1
	--k 1
"

python3.9 susie_pips.py $ARGS
