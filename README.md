# GenAI-Hate
[![CCS: 2026](https://img.shields.io/badge/ACM_CCS-2026-red.svg)](https://doi.org/10.1145/3830454.3846752)
[![Website](https://img.shields.io/badge/Project-Website-2ea44f.svg)](https://bosh-x.github.io/GenAI-Hate)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97_Dataset-GenAI--Hate--Bench-yellow.svg)](https://huggingface.co/datasets/TrustAIRLab/GenAI-Hate-Bench)
[![license](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)

This is the official repository of the ACM CCS 2026 paper [**Does AI Amplify Online Hate? A Multi-Modal Measurement Study on 4chan's /pol/**](https://doi.org/10.1145/3830454.3846752) by [Bo Shao](https://bosh-x.github.io/), [Xinyue Shen](https://xinyueshen.me/), and [Yang Zhang](https://yangzhangalmo.github.io/).

We ask whether generative AI is being used to produce hateful content in online communities. A summary of the study is on the [project website](https://bosh-x.github.io/GenAI-Hate), and the detector benchmark is released as [GenAI-Hate-Bench](https://huggingface.co/datasets/TrustAIRLab/GenAI-Hate-Bench) on Hugging Face.

> **Disclaimer.** This repository and the paper contain examples of hateful content. Reader discretion is recommended.

## Repository Structure

```
aigi_detector/      AI-generated image detection (AIDE scoring)
aigt_detector/      AI-generated text detection (Pangram V3 API)
clip_search/        CLIP ViT-B/32 image retrieval pipeline
toxicity_detection/ Perspective API batch scoring
word2vec/           Word2Vec slur discovery
```

## Setup

```bash
conda env create -f environment.yml
conda activate genai-hate
```

Each module has its own README with configuration details. All modules read database credentials and API keys from environment variables — no secrets are stored in code.

## Data

### Raw 4chan Text Data

This repository does not redistribute raw 4chan data. Please obtain data from public archives and then import locally.
- Primary entry (official page): https://archive.4plebs.org/_/articles/credits/#archives
- Direct collection on Internet Archive: https://archive.org/details/@fourplebs

### Import Text Data from Public Archives (4plebs)
- Direct CSV dump (Jan 2026): https://archive.org/details/4plebs-org-data-dump-2026-01

### Image Data (CLIP Search & AI-Image Detection)
- We do not redistribute images. Please obtain image files via the 4plebs credits page: https://archive.4plebs.org/_/articles/credits/#archives
- For faithful reproduction (CLIP retrieval and AIDE scoring), prefer original images.
- Place images under a local, git-ignored folder. Recommended: `pol_dumps/` (this path is ignored by `.gitignore`).
- Storage note: the media set is large. Ensure sufficient disk space and avoid committing any media into the repository.

Archive links:
- Up to January 2024:
  - https://archive.org/details/4plebs-org-image-dump-2024-01-part25
  - https://archive.org/details/4plebs-org-image-dump-2024-01-part26
  - https://archive.org/details/4plebs-org-image-dump-2024-01-part27
  - https://archive.org/details/4plebs-org-image-dump-2024-01-part28
- January 2026 additions:
  - https://archive.org/details/4plebs-org-image-dump-2026-01-part4
  - https://archive.org/details/4plebs-org-image-dump-2026-01-part5
  - https://archive.org/details/4plebs-org-image-dump-2026-01-part6

## 🛡️ Ethics & Disclosure

Our study analyzes publicly posted content on 4chan, an anonymous imageboard where all posts are anonymous by design. No personally identifiable information is collected or processed, and this repository redistributes no raw 4chan text or images — everything is obtained from the public 4plebs archives (see [Data](#data)).

The detection pipelines here are built for aggregate trend analysis. They should not be used to label or act against individual users; detector precision on in-the-wild hateful text is far below what any decision about a person would require.

We acknowledge the dual-use risk that our findings could inform adversaries about detection limitations. We believe the benefits of understanding AI-assisted hateful content outweigh this risk, as awareness is a prerequisite for effective countermeasures.

**This repository is intended for research purposes only. Any misuse is strictly prohibited.**

## 📚 Citation

If you find this useful in your research, please consider citing:

```bibtex
@inproceedings{SSZ26,
  author    = {Bo Shao and Xinyue Shen and Yang Zhang},
  title     = {{Does AI Amplify Online Hate? A Multi-Modal Measurement Study on 4chan's /pol/}},
  booktitle = {{ACM SIGSAC Conference on Computer and Communications Security (CCS)}},
  publisher = {ACM},
  year      = {2026},
  doi       = {10.1145/3830454.3846752}
}
```
