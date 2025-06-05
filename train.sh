#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=40:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/UNet_SR
#$ -N Train_UNet

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

cd ../CriDiff
source CriDiff_env/bin/activate
export PATH="CriDiff_env/bin:$PATH"
cd ../UNet_SR

python3 train.py --percpt_delay 10

date