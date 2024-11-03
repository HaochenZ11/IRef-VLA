export data_path=./IRef-VLA # PATH TO THE DATASET

export proj_name=mvt_eval # PROJECT NAME FOR LOGGING
export model_ckpt= # PATH TO THE MODEL CHECKPOINT

export train_split='train' # CHANGE THIS TO THE DESIRED TRAINING SPLIT
export test_split='referit3d_test' # CHANGE THIS TO THE DESIRED TEST SPLIT

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export PYTHONPATH=..

cd MVT
module load anaconda3/2022
conda activate 3dvista
python train_MVT.py  \
    --batch_size 24 \
    --data_path $data_path \
    --train_split $train_split \
    --test_split $test_split \
    --proj_name $proj_name \
    --log_dir logs/MVT_sr3d \
    --eval \
    --dataset 'vla' \
    # --resume_path $model_ckpt \