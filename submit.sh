## FROM DM IN AI PROJECT FOLDER ##

#!/bin/sh
#BSUB -J reducedcustom-singleval-synth100
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q gpua100
#BSUB -n 4
###-- Select the resources: 1 gpu in exclusive process mode --
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
###-- send notification at start --
#BSUB -B
###-- send notification at completion--
#BSUB -N
#end of BSUB options

export nnUNet_raw="/zhome/4b/5/187216/dm-i-ai-2025/tumor_segmentation/data_nnUNet"
export nnUNet_preprocessed="/zhome/4b/5/187216/dm-i-ai-2025/tumor_segmentation/data_nnUNet/preprocessed"
export nnUNet_results="/zhome/4b/5/187216/dm-i-ai-2025/tumor_segmentation/data_nnUNet/results"


module load python3
module load python3/3.13.5
source venv/bin/activate
python3 -m pip install -r requirements.txt

# nnUNetv2_train 1 Standard-CV-synth100 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 1 Standard-CV-synth035 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 1 Standard-CV-synth010 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 1 Standard-CV-ctrl30 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 1 reducedcustom-singleval-ctrl30 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 1 reducedcustom-singleval-synth100 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_500epochs --npz --c
# nnUNetv2_train 1 standard-crosval-synth100 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 1 conservative-custom-singleval-synth050 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 1 reduced-custom-singleval-synth050 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz

