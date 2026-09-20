# AIGT Detection Module (AI-Generated Text)

Detect AI-generated text in anti-Indian and anti-Semitic posts on 4chan /pol/ using the Pangram V3 API.

## Components

```
aigt_detector/
├── pangram/               # Pangram V3 API: sample and score posts
├── scripts/
│   └── data_prep/         # Text cleaning utilities
├── db_config.example.json # Database config template
└── requirements.txt
```

## Configuration

Copy `db_config.example.json` to `db_config.json` and fill in your database credentials, or set equivalent environment variables (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`). Pangram API key is read from a `.env` file (`PANGRAM_API_KEY=...`).

## Usage

**Score posts with Pangram**

```bash
python aigt_detector/pangram/run_pangram_sampled.py
```
