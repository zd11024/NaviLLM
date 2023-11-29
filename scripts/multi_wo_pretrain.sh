# set mp3d path
# export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

# set java path
# export JAVA_HOME=$java_path
# export PATH=$JAVA_HOME/bin:$PATH
# export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# activate environment
# conda activate navillm

# training for 30 epochs
torchrun --nnodes=1 --nproc_per_node=8 --master_port 41000 train.py \
    --stage multi --cfg_file configs/multi.yaml \
    --data_dir data --pretrained_model_name_or_path data/models/Vicuna-7B --precision amp_bf16 \
    --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 30 \
    --enable_og --enable_summarize --enable_fgr2r \
    --test_datasets CVDN SOON R2R REVERIE ScanQA \
    --max_saved_checkpoints 1 --output_dir output/multi_wo_pretrain \