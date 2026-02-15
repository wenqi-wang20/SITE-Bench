TASK="site_bench_image"

# install transformers==4.42.0 for llava and internvl-2.5
pip install transformers==4.42.0

######## llava_onevision_0.5b ########
MODEL_NAME="llava_onevision"
LOG_SUFFIX="${MODEL_NAME}_${TASK}_DEBUG"
PRETRAINED="lmms-lab/llava-onevision-qwen2-0.5b-ov"

python3 -m accelerate.commands.launch \
    --num_processes 4 \
    -m lmms_eval \
    --model $MODEL_NAME \
    --model_args pretrained=$PRETRAINED,device_map="auto",attn_implementation=flash_attention_2 \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $LOG_SUFFIX \
    --output_path "logs/"

# ######## llava_onevision_7b ########
MODEL_NAME="llava_onevision"
LOG_SUFFIX="${MODEL_NAME}_${TASK}"
PRETRAINED="lmms-lab/llava-onevision-qwen2-7b-ov"
python3 -m accelerate.commands.launch \
    --num_processes 4 \
    -m lmms_eval \
    --model $MODEL_NAME \
    --model_args pretrained=$PRETRAINED,device_map="auto",attn_implementation=flash_attention_2 \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $LOG_SUFFIX \
    --output_path "logs/"


# ######## InternVL2_5_4B ########
MODEL_NAME="internvl2"
LOG_SUFFIX="${MODEL_NAME}_${TASK}"
PRETRAINED="OpenGVLab/InternVL2_5-4B"
python3 -m accelerate.commands.launch \
    --num_processes 4 \
    -m lmms_eval \
    --model $MODEL_NAME \
    --model_args pretrained=$PRETRAINED,device_map="auto",modality="image" \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $LOG_SUFFIX \
    --output_path ./logs/ 

# ######## InternVL2_5_8B ########
MODEL_NAME="internvl2"
LOG_SUFFIX="${MODEL_NAME}_${TASK}"
PRETRAINED="OpenGVLab/InternVL2_5-8B"
python3 -m accelerate.commands.launch \
    --num_processes 4 \
    -m lmms_eval \
    --model $MODEL_NAME \
    --model_args pretrained=$PRETRAINED,device_map="auto",modality="image" \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $LOG_SUFFIX \
    --output_path ./logs/ 

# install transformers==4.49.0 for qwen2.5-vl
pip install transformers==4.49.0

# # ######## QWEN2_5-VL-3B ########
MODEL_NAME="qwen2_5_vl"
LOG_SUFFIX="${MODEL_NAME}_${TASK}"
PRETRAINED="Qwen/Qwen2.5-VL-3B-Instruct"
python3 -m accelerate.commands.launch \
    --num_processes 4 \
    -m lmms_eval \
    --model $MODEL_NAME \
    --model_args pretrained=$PRETRAINED,device_map="auto",use_flash_attention_2=True,use_custom_video_loader=True \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $LOG_SUFFIX \
    --output_path ./logs/

# ######## QWEN2_5-VL-7B ########
MODEL_NAME="qwen2_5_vl"
LOG_SUFFIX="${MODEL_NAME}_${TASK}"
PRETRAINED="Qwen/Qwen2.5-VL-7B-Instruct"
python3 -m accelerate.commands.launch \
    --num_processes 4 \
    -m lmms_eval \
    --model $MODEL_NAME \
    --model_args pretrained=$PRETRAINED,device_map="auto",use_flash_attention_2=True,use_custom_video_loader=True \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $LOG_SUFFIX \
    --output_path ./logs/

