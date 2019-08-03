DATA_DIR=${1:-/home/gyx/nvidia-bert/data}
WIKI_DIR="small_wiki_5m_base"
# WIKI_DIR=${2:-/home/gyx/nvidia-bert/data/small_wiki/}

N_PROCS_PREPROCESS=$(nproc)    # Adjust this based on memory requirements and available number of cores

# Wikiextractor.py - Creates lots of folders/files in "doc format"
echo "Running Wikiextractor"
mkdir -p ./extracted_articles
$HOME/wikiextractor/WikiExtractor.py ./raw_data/wikidump.xml -b 1000M --processes ${N_PROCS_PREPROCESS} -o ./extracted_articles

# Remove XML Tags and extraneous titles (since they are not sentences)
# Also clean to remove lines between paragraphs within article and use space-separated articles
echo "Cleaning and formatting files (one article per line)"
python3 ./remove_tags_and_clean.py ./extracted_articles ./wikipedia_corpus.txt

cd $DATA_DIR
# Create HDF5 files for SMALL_WIKI
bash create_datasets_from_start.sh ${WIKI_DIR} ./${WIKI_DIR}/wikipedia_corpus.txt