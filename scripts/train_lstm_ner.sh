CUDA_VISIBLE_DEVICES=1 python3 train_lstm_ner.py --use_gpu --model_name bert_1g_100dim_nochar.model --crf \
--word_lstm_dim 100 --dir downstream_models/ner/test_100_nochar/ --bert_data_dir data/wiki_1g_cased/final_text_files_sharded/

CUDA_VISIBLE_DEVICES=1 python3 train_lstm_ner.py --use_gpu --model_name bert_5m_100dim_nochar.model --crf \
--word_lstm_dim 100 --dir downstream_models/ner/5m_test_100_nochar/ --bert_data_dir data/small_wiki_5m_base/final_text_files_sharded/

CUDA_VISIBLE_DEVICES=1 python3 train_lstm_ner.py --use_gpu --model_name bert_5m_200dim_nochar.model --crf \
--word_lstm_dim 200 --dir downstream_models/ner/5m_test_200_nochar/ --bert_data_dir data/small_wiki_5m_base/final_text_files_sharded/