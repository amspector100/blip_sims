#!/bin/sh
#SBATCH -p janson_cascade #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 47 #Number of cores
#SBATCH -t 7-00:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../output/lp_int_sol_%j.out #File to which standard out will be written
#SBATCH -e ../output/lp_int_sol_%j.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

NREPS=128
NPROCESSES=47

ARGS="
        --reps $NREPS
        --num_processes $NPROCESSES
	--p 1000
	--kappa start0.025end1numvals20
	--method [ar1,ver,qer]
"

# Install python/anaconda, activate base environment
module purge
module load Gurobi/9.1.2
module load Anaconda3/5.0.1-fasrc02
source activate adaptexper1


# Load R 4.0.2
module load R/4.0.2-fasrc01

# Load packages
export R_LIBS_USER=$HOME/apps/R_4.0.2:$R_LIBS_USERi


python3.9 lp_int_sol.py $ARGS

module purge
