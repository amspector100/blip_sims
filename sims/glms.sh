#!/bin/sh
#SBATCH -p partition_name #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 32 #Number of cores
#SBATCH -t 7-00:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../output/glm_pk5hd_%j.out #File to which standard out will be written
#SBATCH -e ../output/glm_pk5hd_%j.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

NREPS=1
NPROCESSES=1

## Figures which vary the dimensionality p
VP_ARGS="
        --reps $NREPS
        --num_processes $NPROCESSES
        --p [500,1000,1500,2000,3000,5000,7500,10000]
        --coeff_size 1
        --kappa [0.5]
        --method [ark]
        --y_dist [gaussian]
	--nsample [5000]
	--chains 10
	--bsize 5
	--well_specified [True]
        --max_corr 0.9999
        --k [2]
        --sparsity [0.05]
	--run_crt [0]
	--run_dap [0]
"

## Main simulation setting
HIGH_ARGS_COMMON="
        --reps $NREPS
        --num_processes $NPROCESSES
        --p 1000
	--coeff_size 0.2
        --kappa [0.2,0.6,1.1,1.5,2.0]
        --covmethod [ark]
        --y_dist [gaussian]
	--max_corr 0.9999
	--k [3]
	--alpha0 0.125
	--nsample 5000
	--chains 10
	--bsize 5
	--screen 0
"

HIGH_P1="${HIGH_ARGS_COMMON}
	--sparsity [0.01]
	--run_dap [1]	
	--run_crt [1]
	--levels 8
"
HIGH_P2="${HIGH_ARGS_COMMON}
	--sparsity [0.1,0.2]
	--run_dap [0]
	--run_crt [1]
	--levels 8
"

## Probit setting
BIN_ARGS="
	--reps $NREPS
	--num_processes $NPROCESSES
	--p 400
	--coeff_size [1]
	--kappa [0.5,1.25,2,3.5,5,6.5]
	--method ark
	--y_dist [probit]
	--well_specified [True,False]
	--k [5]
	--chains 10
	--max_corr 0.99
	--nsample 5000
	--sparsity [0.01,0.03,0.05]
	--screen [0]
	--run_crt [1]
	--run_dap [1]
	--levels 8
"

## Load whatever modules are needed on a cluster

## Main simulation setting
python3.9 glms.py $HIGH_P1
python3.9 glms.py $HIGH_P2
## probit regresion
python3.9 glms.py $BIN_ARGS
## simulation varying the dimensionality p
python3.9 glms.py $VP_ARGS

