# Word2Vec Analysis

Word2Vec-based slur discovery for 4chan /pol/, used to construct the hate-speech lexicons in the paper.

## Scripts

| Script | Purpose |
|--------|---------|
| `word2vec_train.py` | Train Word2Vec models on tokenized posts |
| `slur_discovery.py` | Find nearest neighbours to seed terms (Paper Table 1) |

## Usage

**1. Train models**

Requires tokenized text files preprocessed from the database.

```bash
python word2vec/word2vec_train.py \
  --input <tokenized_text_file> \
  --epochs 3
```

**2. Slur discovery (Table 1)**

```bash
python word2vec/slur_discovery.py
```

Outputs nearest-neighbour candidates for anti-Indian and anti-Semitic seed terms.

## Dependencies

```bash
pip install gensim nltk psycopg2-binary python-dotenv
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```
