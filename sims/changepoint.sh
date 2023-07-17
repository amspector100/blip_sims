#!/bin/sh
#SBATCH -p partition_name #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 32 #Number of cores
#SBATCH -t 7-00:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../output/cp300_%j.out #File to which standard out will be written
#SBATCH -e ../output/cp300_%j.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

NREPS=1
NPROCESSES=1

## main changepoint simulation setting
CP_ARGS="
        --reps $NREPS
        --num_processes $NPROCESSES
        --p [300]
        --coeff_size [5,9,13]
	--nsample [10000]
	--chains 10
	--bsize 5
	--well_specified [True,False]
        --sparsity [0.01,0.05]
	--spacing [2,5,8,11,14,17,20]
	--reversion_prob [0]
	--run_lss 1
	--run_bcp 1
"

## Additional plot which varies the reversion probability
REVERSION_COMMON_ARGS="
        --reps $NREPS
        --num_processes $NPROCESSES
        --p [300]
        --coeff_size [9]
	--nsample [10000]
	--chains 10
	--bsize 5
	--well_specified [True,False]
        --sparsity [0.05]
	--reversion_prob [0.001,0.25,0.5,0.75,0.999]
	--run_lss 1
	--run_bcp 1
"

# Install python/anaconda, activate base environment
# module purge
# module load Gurobi/9.1.2
# module load Anaconda3/5.0.1-fasrc02
# module load GCC/8.2.0-2.31.1 GSL/2.5 # for dap
# source activate adaptexper1

# # Load R 4.0.2
# module load R/4.0.2-fasrc01

# # Load packages
# export R_LIBS_USER=$HOME/apps/R_4.0.2:$R_LIBS_USERi

## Main simulations
python3.9 changepoint.py $CP_ARGS
## Args with different reversion probabilities
# (a) random spacing
python3.9 changepoint.py $REVERSION_COMMON_ARGS
# (b) fixed spacing
python3.9 changepoint.py "
	${REVERSION_CMMON_ARGS}
	--spacing [2,5,8] 
"