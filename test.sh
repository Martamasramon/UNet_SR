#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=40:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/UNet_SR
#$ -N Test_UNet

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

cd ../CriDiff
source CriDiff_env/bin/activate
export PATH="CriDiff_env/bin:$PATH"
cd ../UNet_SR


python3 test.py --checkpoint 'pretrain_PICAI' --test_bs 15

python3 test.py --checkpoint 'masked_stage_1_best' --test_bs 15 --masked
python3 test.py --checkpoint 'masked_stage_2_best' --test_bs 15 --masked

python3 test.py --checkpoint 'masked_down4_stage_1_best' --test_bs 15 --masked --down_factor 4
python3 test.py --checkpoint 'masked_down4_stage_2_best' --test_bs 15 --masked --down_factor 4

python3 test.py --checkpoint 'down4_stage_1_best' --test_bs 15 --down_factor 4
python3 test.py --checkpoint 'down4_stage_2_best' --test_bs 15 --down_factor 4

python3 test.py --checkpoint 'pretrain_PICAI_cont_best' --use_T2W --test_bs 15
python3 test.py --checkpoint 'pretrain_PICAI_cont' --use_T2W --test_bs 15

date