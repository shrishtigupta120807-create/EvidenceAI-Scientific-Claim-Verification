# EvidenceAI: Scientific Claim Verification

An NLP and deep learning project that retrieves scientific research papers and evaluates whether their evidence supports, contradicts, or is neutral toward a given scientific claim.

## Project Overview

EvidenceAI aims to combine semantic search and Natural Language Inference (NLI) to make scientific claim verification more accessible.

### Planned Architecture

1. **Document Retrieval:** Retrieve relevant scientific abstracts using TF-IDF and Sentence-BERT.
2. **Semantic Search:** Generate dense embeddings and perform similarity search using FAISS.
3. **Reranking:** Improve document ranking using a cross-encoder.
4. **Claim Verification:** Classify evidence as SUPPORTS, CONTRADICTS, or NEUTRAL.
5. **Interactive Interface:** Build a demonstration interface using Gradio.

## Dataset

- BEIR SciFact: 5,183 scientific abstracts.
- Original SciFact annotations: supporting and refuting evidence.

Dataset sources:
- https://huggingface.co/datasets/BeIR/scifact
- https://github.com/allenai/scifact

## Current Progress

- [x] SciFact dataset loading and exploration
- [x] TF-IDF retrieval baseline
- [x] Sentence-BERT semantic embeddings
- [x] FAISS vector search
- [ ] Transformer architecture exploration
- [ ] Cross-encoder reranking
- [ ] NLI-based claim verification
- [ ] Model evaluation
- [ ] Gradio application

## Tech Stack

Python, PyTorch, Hugging Face Transformers, Sentence Transformers, FAISS, Scikit-learn, Gradio.

## Evaluation

Retrieval performance will be evaluated using Recall@K, MRR and nDCG. Verification performance will be evaluated using classification metrics such as macro F1.

## Disclaimer

This project is an educational research prototype. Its predictions should not be treated as definitive scientific conclusions.
