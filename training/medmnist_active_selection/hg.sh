#!/bin/bash                                                                     
#SBATCH -N 1                                                                    
#SBATCH -n 2                                                                    
##SBATCH --mem-per-cpu 50000 
##SBATCH -p gpu
##SBATCH -p physicsgpu1                                                         
##SBATCH -p sulcgpu2                                                            
##SBATCH -p rcgpu1
##SBATCH -p mrlinegpu1                                                            
#SBATCH -p asinghargpu1                                                                                                                     
##SBATCH -p cidsegpu1

##SBATCH -p cidsegpu2
##SBATCH -p physicsgpu2
##SBATCH -p sulcgpu1

#SBATCH -q wildfire
##SBATCH -p jlianggpu1                                                          
##SBATCH -q jliang12                                                             
#SBATCH --gres=gpu:1                                                            
#SBATCH -t 0-10:00                                                               
##SBATCH -o slurm.%j.${1}.out                                                   
##SBATCH -e slurm.%j.${1}.err                                                    
#SBATCH --mail-type=END,FAIL                                                    
#SBATCH --mail-user=zzhou82@asu.edu                                             
                                                                                                      
module load tensorflow/1.8-agave-gpu                                            
module unload python/.2.7.14-tf18-gpu

/packages/7x/python/3.6.5-tf18-gpu/bin/python3 -m pip install --upgrade pip --user
#/packages/7x/python/3.6.5-tf18-gpu/bin/python3 -m pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git --user

#python3.6 -m medmnist download

python3.6 -W ignore main.py --run $1 --task $2 --partial $3 --init $4 --act $5
