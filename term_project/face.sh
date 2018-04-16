#!/bin/bash -l
# NOTE the -l flag!TE the -l flag!
#

# This is an example job file for a single core CPU bound program
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler.
# Please copy this file to your home directory and modify it
# to suit your needs.
# 
# If you need any help, please email rc-help@rit.edu
#

# Name of the job - You'll probably want to customize this.
#SBATCH -J eigendecomp

# Standard out and Standard Error output files
#SBATCH -o eig_face.output
#SBATCH -e error_eigface.output

#To send emails, set the and remove one of the "#" signs.
##SBATCH --mail-user kpn3569@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# Request 5 hours run time MAX, anything over will be KILLED
#SBATCH -t 5:0:0

# Put the job in the "debug" partition and request one core
# "debug" is a limited partition.  You'll likely want to change
# it to "work" once you understand how this all works.
#SBATCH -p work -c 1

# Job memory requirements in MB
#SBATCH --mem=50000

# Explicitly state you are a free user
#SBATCH --qos=free

#SBATCH --gres=gpu:k20
# Your job script goes below this line.  
#

# module load torch7
module load cuda
module load cv2
module load cuda/6.5
module load python/2.7.12 #For skcuda
python face.py




