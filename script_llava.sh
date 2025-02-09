#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8 # Request 8 core
#$ -l h_rt=100:0:0  # Request 1 hour runtime
#$ -l h_vmem=11G   # Request 1GB RAM
#$ -l gpu=1  # request 1 GPU
#$ -l node_type=rdg

module load gcc/10.2.0
module load anaconda3
module load cuda/11.8.0
module load openssl/1.1.1s
# module load python/3.10.7
# virtualenv videetree_env
# source videoxl/bin/activate
conda activate videoxl 
# Note, if you don't want to reinstall BNBs dependencies, append the `--no-deps` flag!
# pip install bitsandbytes
# pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
# # pip install -e "videoxl/.[train]"
# pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir
# pip install -r requirements.txt
# cd LongVA/
# python -m pip install -e "longva/.[train]"
# pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
# cd ..
# pip install transformers==4.46.3
huggingface-cli login --token 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
export OPENAI_API_KEY="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"
export DECORD_EOF_RETRY_MAX=20480 
export HF_HOME="/data/home/acw652/.cache/huggingface"
# git clone https://github.com/IDEA-Research/GroundingDINO.git
# cd GroundingDINO/
# pip install -e .
# mkdir weights
# cd weights
# wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# cd ../..
accelerate launch --num_processes 1 --main_process_port 16000 -m lmms_eval \
    --model longva_cos \
    --model_args pretrained=lmms-lab/LongVA-7B,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=128,video_decode_backend=decord\
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix longvideobench_val_v_longva \
    --output_path ./logs/

# accelerate launch --num_processes 2 --main_process_port 13000 -m lmms_eval \
#     --model videoxl \
#     --model_args pretrained=/data/DERI-Gong/jh015/Video-XL/assets/VideoXL_weight_8,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=384,video_decode_backend=decord,device_map=""\
#     --tasks videomme \  
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix videoxl_videomme \
#     --output_path ./logs/

# accelerate launch --num_processes 1 --main_process_port 12344 -m lmms_eval \
#     --model videoxl \
#     --model_args pretrained=/data/DERI-Gong/jh015/Video-XL/assets/VideoXL_weight_8,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=128,video_decode_backend=decord\
#     --tasks longvideobench_val_v \ #videomme_w_subtitle
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix videoxl_long \
#     --output_path ./logs/


##longva
# accelerate launch --num_processes 2 --main_process_port 58000 -m lmms_eval \
#     --model longva \
#     --model_args pretrained=lmms-lab/LongVA-7B,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=128,video_decode_backend=decord\
#     --tasks videochatgpt \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix videochatgpt_longva \
#     --output_path ./logs/

####llava-next-video
# accelerate launch --num_processes 1 --main_process_port 37000 -m lmms_eval \
#     --model llava_vid \
#     --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-7B-Qwen,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32，mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
#     --tasks videomme \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix videomme_llava_vid_7B \
#     --output_path ./logs/

# --tasks activitynetqa,videochatgpt,nextqa_mc_test,egoschema,video_dc499,videmme,videomme_w_subtitle,perceptiontest_val_mc \


# git clone https://github.com/EvolvingLMMs-Lab/LongVA.git
# cd LongVA/
# pip cache purge
# python -m pip install -e "longva/.[train]"
# cd ..
# cd flash-attention
# python setup.py install
# cd ..

# python eval/eval_vnbench.py
# cd VNBench
# python eval.py --path /data/DERI-Gong/jh015/Video-XL/submit/vnbench.jsonl