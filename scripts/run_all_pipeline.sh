# setup envs
echo -e "\033[42;37m Set Up Enviroments\033[0m"
python3 convert_config.py $1
cat config/bash_config.sh
source config/bash_config.sh

# Selective Masking - finetune BERT
echo -e "\033[42;37m Selective Masking - Finetune BERT\033[0m"
bash scripts/finetune_origin.sh

# Selective Masking - downstream mask
echo -e "\033[42;37m Selective Masking - Downstream Mask\033[0m"
bash data/create_data_rule/run.sh

# Selective Masking - train nn
echo -e "\033[42;37m Selective Masking - Train NN\033[0m"
bash scripts/run_mask_model.sh

# Selective Masking - in-domain mask
echo -e "\033[42;37m Selective Masking - In-domain Mask\033[0m"
bash data/create_data_model/run.sh

# TaskPT
echo -e "\033[42;37m TaskPT\033[0m"
bash scripts/run_pretraining.sh

# Fine-tuning
echo -e "\033[42;37m Fine-tuning\033[0m"
bash scripts/finetune_ckpt_all_seed.sh

# Gather results of different seed
echo -e "\033[42;37m Gather results of different seed\033[0m"
python3 gather_results.py ${E_FINE_TUNING_OUTPUT_DIR}
