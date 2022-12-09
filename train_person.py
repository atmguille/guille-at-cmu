import os

MY_NAME = "atmguille"
SUBJECT_TYPE = "person"  # person, man
TRAINING_STEPS = 2000
LR = 1e-6
BATCH_SIZE = 1
SUBJECT_N_IMAGES = 500

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
INSTANCE_DIR = f"{MY_NAME}_images"
CLASS_DIR = "regularization_images/person_ddim"
OUTPUT_DIR = f"{MY_NAME}_lr{LR}_steps{TRAINING_STEPS}_bs{BATCH_SIZE}_sn{SUBJECT_N_IMAGES}"
INSTANCE_PROMPT = f"photo of {MY_NAME} {SUBJECT_TYPE}"
CLASS_PROMPT = f"a photo of a {SUBJECT_TYPE}, ultra detailed"
PRECISION = "no" # no, fp16, bf16

# Run command
os.system(f'accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \
    --train_text_encoder \
    --pretrained_model_name_or_path="{MODEL_NAME}" \
    --instance_data_dir="{INSTANCE_DIR}" \
    --class_data_dir="{CLASS_DIR}" \
    --output_dir="{OUTPUT_DIR}" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="{INSTANCE_PROMPT}" \
    --class_prompt="{CLASS_PROMPT}" \
    --seed=42 \
    --resolution=512 \
    --mixed_precision="{PRECISION}" \
    --train_batch_size="{BATCH_SIZE}" \
    --gradient_accumulation_steps=1 \
    --learning_rate="{LR}" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps="{TRAINING_STEPS}" \
    --num_class_images="{SUBJECT_N_IMAGES}"')
