# Models Directory

This directory is intended for the pre-trained Word2Vec models and any other large embedding models.

Due to GitHub's file size constraints, these models are not tracked in the repository.

## Required Word2Vec Models

To run the Word2Vec feature extraction pipeline, please structure your models as follows:

```
models/
├── word2vec/
│   ├── wiki_sg_300/
│   │   ├── wikipedia_sg_300 (Model file)
│   │   └── ... (Other associated files)
│   └── wiki_cbow_300/
│       ├── wikipedia_cbow_300 (Model file)
│       └── ... (Other associated files)
└── README.md
```

If your models are named differently or placed in different paths, you can modify the paths in `src/feature_extraction.py`.
