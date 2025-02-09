huggingface-cli login --token 
export OPENAI_API_KEY=""
export DECORD_EOF_RETRY_MAX=20480 
export HF_HOME=""

accelerate launch --num_processes 1 --main_process_port 16000 -m lmms_eval \
    --model longva_cos \
    --model_args pretrained=lmms-lab/LongVA-7B,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=128,video_decode_backend=decord\
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix longvideobench_val_v_longva \
    --output_path ./logs/
