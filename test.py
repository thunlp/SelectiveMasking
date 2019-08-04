from mask_utils.mask_generators import Ner

config = {
    "mask_samp": 100,
    "mask_num": 1,
    "mask_rate": 0.0,
    "model_path": "mask_utils/ner/models/test_bert_dim.model",
    "mapping_file": "mask_utils/ner/models/mapping.pkl",
    "gpu": True

}

ner = Ner(config)
f = open("data/small_wiki_5m_base/final_text_files_sharded/corpus.segmented.part.0.txt", "r")
lines = [line.strip().split(" ") for line in f.readlines()]
for i, line in enumerate(lines):
    print(i, ner(line))
