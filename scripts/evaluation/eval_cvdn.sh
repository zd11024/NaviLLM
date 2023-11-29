# set mp3d path
# export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

# set java path
# export JAVA_HOME=$java_path
# export PATH=$JAVA_HOME/bin:$PATH
# export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# activate environment
# conda activate navillm

torchrun --nnodes=1 --nproc_per_node=8 --master_port 41000 train.py \
    --mode test --data_dir data --cfg_file configs/mp3d/multi_v3.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B --precision amp_bf16 \
    --resume_from_checkpoint $model_path \
    --test_datasets CVDN \
    --batch_size 4 --output_dir build/eval --validation_split test --save_pred_results
