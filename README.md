# Selective Masking

Source code for "Train No Evil: Selective Masking for Task-Guided Pre-Training"

## Download Data

The datasets can be downloaded from this [link](https://drive.google.com/file/d/1dnDQO6kCNOe2iCpDl-xJ4XXRKLXq-5yw/view?usp=sharing). The datasets need to be put in `data/datasets`.

## Run the Whole Pipeline

1. Modify `config/test.json` for input path, output path, BERT model path, GPU usage etc.

2. run `bash scripts/run_all_pipeline.sh` .

## Run each step

The meaning of each step can be found in the appendix of our paper. The input/output paths are also set in `config/test.json`. Run `python3 convert_config.py config/test.json` to convert the .json file to a .sh file.

### 1 GenePT

We use the training scripts from <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT> for general pre-training.

### 2 Selective Masking

#### 2.1 Finetune BERT

```[bash]
bash scripts/finetune_origin.sh
```

#### 2.2 Downstream Mask

```[bash]
bash data/create_data_rule/run.sh.
```

#### 2.3 Train NN

```[bash]
bash scripts/run_mask_model.sh
```

#### 2.4 In-domain Mask

```[bash]
bash data/create_data_model/run.sh
```

### 3 TaskPT

```[bash]
bash scripts/run_pretraining.sh
```

### 4 Fine-tune

```[bash]
bash scripts/finetune_ckpt_all_seed.sh
python3 gather_results.py $PATH_TO_THE_FINETUNE_OUTPUT
```

## Cite

If you use the code, please cite this paper:

```[]
@inproceedings{gu2020train,
    title={Train No Evil: Selective Masking for Task-Guided Pre-Training},
    author={Yuxian Gu and Zhengyan Zhang and Xiaozhi Wang and Zhiyuan Liu and Maosong Sun},
    year={2020},
    booktitle={Proceedings of EMNLP 2020},
}
```
