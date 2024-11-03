export data_path=./IRef-VLA # PATH TO THE DATASET

export proj_name=vista_eval # PROJECT NAME FOR LOGGING
export run_name=exp-01 # NAME OF CURRENT RUN
export model_ckpt= # PATH TO THE MODEL CHECKPOINT

export test_split='test' # CHANGE THIS TO THE DESIRED TEST SPLIT

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

module load anaconda3/2022
conda activate 3dvista
python -m vista.train_3dvista \
    --data_path $data_path \
    --run_name $run_name \
    --proj_name $proj_name \
    --test_split $test_split \
    --resume_path $model_ckpt \
    --batch_size 12 \
    --val_freq 1 \
    --eval \
    # --use_sr3d 

