#!/usr/bin/env bash

# use specified pretrained bert and vocab file 
CUDA_VISIBLE_DEVICES=5 python3 run_ner.py --data_dir data/CoNll/ --bert_model pretrain_bert_model/bert-base-cased/bert-base-cased.tar.gz --vocab_file vocab/cased_L-12_H-768_A-12/vocab.txt\
 --task_name=ner --output_dir=baseline_cased_bert_ner/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 16 --eval_batch_size 8

# uncased use specified pretrained bert and vocab file 
CUDA_VISIBLE_DEVICES=0 python3 run_ner.py --data_dir data/CoNll/ --bert_model pretrain_bert_model/bert-base-uncased/bert-base-uncased.tar.gz --vocab_file vocab/uncased_L-12_H-768_A-12/vocab.txt \
--task_name=ner --output_dir=output_test_uncased/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 32 --eval_batch_size 8 --do_lower_case


# use pretrained bert and vocab file from huggingface
CUDA_VISIBLE_DEVICES=1 python3 run_ner.py --data_dir data/CoNll/ --bert_model bert-base-cased --task_name=ner --output_dir=output_test_cased_4/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --cache_dir ~/pretrain_bert_model/ --learning_rate 5e-5 --do_train --train_batch_size 16 --eval_batch_size 8

# uncased mask
CUDA_VISIBLE_DEVICES=1 python3 run_ner.py --data_dir data/CoNll/ --bert_model pretrain_bert_model/uncased_mask_16000/uncased_mask_16000.tar.gz --vocab_file vocab/uncased_L-12_H-768_A-12/vocab.txt \
--task_name=ner --output_dir=16000_mask_uncased_out/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 32 --eval_batch_size 8 --do_lower_case

#uncased
CUDA_VISIBLE_DEVICES=1 python3 run_ner.py --data_dir data/CoNll/ --bert_model bert-base-uncased --cache_dir ~/pretrain_bert_model/ \
--task_name=ner --output_dir=output_uncased/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 32 --eval_batch_size 8 --do_lower_case

#uncased origin
CUDA_VISIBLE_DEVICES=1 python3 run_ner.py --data_dir data/CoNll/ --bert_model pretrain_bert_model/origin_model/uncased_test.tar.gz --vocab_file vocab/uncased_L-12_H-768_A-12/vocab.txt \
--task_name=ner --output_dir=output_uncased_origin/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 32 --eval_batch_size 8 --do_lower_case

# cased mask 40000
CUDA_VISIBLE_DEVICES=0 python3 run_ner.py --data_dir data/CoNll/ --bert_model pretrain_bert_model/cased_mask/cased_mask.tar.gz --vocab_file vocab/cased_L-12_H-768_A-12/vocab.txt \
--task_name=ner --output_dir=cased_mask_bert_ner/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 16 --eval_batch_size 8

CUDA_VISIBLE_DEVICES=0 python3 run_ner.py --data_dir data/CoNll-2/ --bert_model pretrain_bert_model/cased_mask/cased_mask.tar.gz --vocab_file vocab/cased_L-12_H-768_A-12/vocab.txt \
--task_name=ner --output_dir=cased_mask_bert_ner-2/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 16 --eval_batch_size 8

CUDA_VISIBLE_DEVICES=1 python3 run_ner.py --data_dir data/CoNll/ --bert_model pretrain_bert_model/cased_mask/cased_mask.tar.gz --vocab_file vocab/cased_L-12_H-768_A-12/vocab.txt \
--task_name=ner --output_dir=cased_mask_bert_ner/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 16 --eval_batch_size 8

# cased mask 16000
CUDA_VISIBLE_DEVICES=1 python3 run_ner.py --data_dir data/CoNll/ --bert_model pretrain_bert_model/2_cased_mask_16000/2_cased_mask_16000.tar.gz --vocab_file vocab/cased_L-12_H-768_A-12/vocab.txt \
--task_name=ner --output_dir=2_16000_cased_mask_bert_ner/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 16 --eval_batch_size 8

# cased mask 32000
CUDA_VISIBLE_DEVICES=2 python3 run_ner.py --data_dir data/CoNll/ --bert_model pretrain_bert_model/cased_mask_32000/cased_mask_32000.tar.gz --vocab_file vocab/cased_L-12_H-768_A-12/vocab.txt \
--task_name=ner --output_dir=32000_cased_mask_bert_ner/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 16 --eval_batch_size 8

# cased mask 24000
CUDA_VISIBLE_DEVICES=3 python3 run_ner.py --data_dir data/CoNll/ --bert_model pretrain_bert_model/cased_mask_24000/cased_mask_24000.tar.gz --vocab_file vocab/cased_L-12_H-768_A-12/vocab.txt \
--task_name=ner --output_dir=24000_cased_mask_bert_ner/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 16 --eval_batch_size 8

# cased mask 8000
CUDA_VISIBLE_DEVICES=0 python3 run_ner.py --data_dir data/CoNll/ --bert_model pretrain_bert_model/cased_mask_8000/cased_mask_8000.tar.gz --vocab_file vocab/cased_L-12_H-768_A-12/vocab.txt \
--task_name=ner --output_dir=8000_cased_mask_bert_ner/ --max_seq_length=128 --num_train_epochs 5 --do_eval --warmup_proportion=0.4 --learning_rate 5e-5 --do_train --train_batch_size 16 --eval_batch_size 8
