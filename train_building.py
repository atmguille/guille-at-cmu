import os

MY_NAME = "cmu_cs"
SUBJECT_TYPE = "building"
TRAINING_STEPS = 400
LR = 2e-6
BATCH_SIZE = 1

MODEL_NAME = "atmguille_lr1e-06_steps2000_bs1_sn500"
INSTANCE_DIR = f"{MY_NAME}_images"
OUTPUT_DIR = f"{MY_NAME}_lr{LR}_steps{TRAINING_STEPS}_from-{MODEL_NAME}"
INSTANCE_PROMPT = f"photo of {MY_NAME} {SUBJECT_TYPE}"
PRECISION="no" # no, fp16, bf16

# Run command
os.system(f'accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \
    --train_text_encoder \
    --pretrained_model_name_or_path="{MODEL_NAME}" \
    --instance_data_dir="{INSTANCE_DIR}" \
    --output_dir="{OUTPUT_DIR}" \
    --instance_prompt="{INSTANCE_PROMPT}" \
    --seed=42 \
    --resolution=512 \
    --mixed_precision="{PRECISION}" \
    --train_batch_size="{BATCH_SIZE}" \
    --gradient_accumulation_steps=1 \
    --learning_rate="{LR}" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps="{TRAINING_STEPS}"')
