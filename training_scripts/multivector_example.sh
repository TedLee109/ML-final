export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="data_example"
export OUTPUT_DIR="./exps/output_dsn"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --scale_lr \
  --learning_rate_unet=2e-6\
  --learning_rate_text=2e-7 \
  --learning_rate_ti=1e-5 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --lr_scheduler_lora="linear" \
  --lr_warmup_steps_lora=100 \
  --placeholder_tokens="<s1>|<s2>" \
  --use_template="style"\
  --save_steps=10 \
  --max_train_steps_ti=100 \
  --max_train_steps_tuning=100 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.0001 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda:0" \
  --lora_rank=1 \
  --lora_clip_target_modules="{'CLIPSdpaAttention'}"
#  --use_face_segmentation_condition\