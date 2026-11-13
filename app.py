
import streamlit as st
import numpy as np
import faiss
import torch
import torch.nn.functional as F

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="EvidenceAI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 EvidenceAI")
st.subheader("Scientific Evidence Retrieval & Claim Verification")

st.write(
    "Enter a scientific claim to retrieve relevant research "
    "abstracts, rerank the evidence and examine NLI predictions."
)


# ---------------------------------------
# LOAD MODELS AND DATASET
# ---------------------------------------

@st.cache_resource(show_spinner="Loading EvidenceAI models and dataset...")
def load_resources():

    device = "cpu"

    # Load SciFact dataset
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/"
        "thakur/BEIR/datasets/scifact.zip"
    )

    data_path = util.download_and_unzip(url, "data")

    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path
    ).load(split="train")

    doc_ids = list(corpus.keys())

    documents = [
        corpus[doc_id]["title"] + " " + corpus[doc_id]["text"]
        for doc_id in doc_ids
    ]

    # Load Sentence-BERT
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )

    document_embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Build FAISS index
    index = faiss.IndexFlatIP(
        document_embeddings.shape[1]
    )

    index.add(
        document_embeddings.astype("float32")
    )

    # Load Cross-Encoder
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=device
    )

    # Load NLI model
    nli_name = "cross-encoder/nli-deberta-v3-base"

    nli_tokenizer = AutoTokenizer.from_pretrained(
        nli_name
    )

    nli_model = AutoModelForSequenceClassification.from_pretrained(
        nli_name
    )

    nli_model.eval()

    return {
        "corpus": corpus,
        "doc_ids": doc_ids,
        "model": model,
        "index": index,
        "reranker": reranker,
        "nli_tokenizer": nli_tokenizer,
        "nli_model": nli_model,
        "device": device
    }


# ---------------------------------------
# RETRIEVE CANDIDATE DOCUMENTS
# ---------------------------------------

def retrieve_candidates(claim, resources, top_k=20):

    model = resources["model"]
    index = resources["index"]
    doc_ids = resources["doc_ids"]
    corpus = resources["corpus"]

    top_k = min(top_k, index.ntotal)

    query_embedding = model.encode(
        claim,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding.reshape(1, -1),
        top_k
    )

    candidates = []

    for score, idx in zip(scores[0], indices[0]):

        doc_id = doc_ids[idx]

        candidates.append({
            "doc_id": doc_id,
            "title": corpus[doc_id]["title"],
            "text": corpus[doc_id]["text"],
            "retrieval_score": float(score)
        })

    return candidates


# ---------------------------------------
# RERANK DOCUMENTS
# ---------------------------------------

def rerank_documents(
    claim,
    candidates,
    resources,
    top_k=5
):

    reranker = resources["reranker"]

    pairs = [
        (
            claim,
            candidate["title"] + " " + candidate["text"]
        )
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):

        candidate["rerank_score"] = float(score)

    ranked_documents = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked_documents[:top_k]


# ---------------------------------------
# VERIFY CLAIM USING NLI
# ---------------------------------------


def verify_claim(claim, evidence, resources):

    tokenizer = resources["nli_tokenizer"]
    nli_model = resources["nli_model"]

    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    

    full_inputs = tokenizer(
        evidence,
        claim,
        truncation=False
    )

    print(
        "Total tokens before truncation:",
        len(full_inputs["input_ids"])
    )

    print(
        "Tokens after truncation:",
        inputs["input_ids"].shape[1]
    )

    print(
        "Label mapping:",
        nli_model.config.id2label
    )

    with torch.no_grad():

        outputs = nli_model(**inputs)

    probabilities = F.softmax(
        outputs.logits,
        dim=1
    )[0]

    labels = nli_model.config.id2label

    return {
        labels[i].lower(): round(
            probabilities[i].item(), 4
        )
        for i in range(len(labels))
    }


# ---------------------------------------
# COMPLETE EVIDENCEAI PIPELINE
# ---------------------------------------

def evidence_verification_pipeline(
    claim,
    resources,
    retrieval_k=20,
    final_k=5
):

    candidates = retrieve_candidates(
        claim,
        resources,
        top_k=retrieval_k
    )

    ranked_documents = rerank_documents(
        claim,
        candidates,
        resources,
        top_k=final_k
    )

    for document in ranked_documents:

        evidence = document["text"]

        document["verification"] = verify_claim(
            claim,
            evidence,
            resources
        )

    return ranked_documents


# ---------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------

claim = st.text_area(
    "Enter a scientific claim",
    placeholder=(
        "Vitamin D supplementation reduces "
        "respiratory infections."
    ),
    height=120
)

if st.button("Verify Claim", type="primary"):

    if not claim.strip():

        st.warning("Please enter a scientific claim.")

    else:

        try:

            resources = load_resources()

            with st.spinner(
                "Retrieving and analyzing scientific evidence..."
            ):

                results = evidence_verification_pipeline(
                    claim,
                    resources,
                    retrieval_k=10,
                    final_k=3
                )

            st.success(
                f"Analysis completed. Retrieved {len(results)} documents."
            )

            for i, result in enumerate(results, 1):

                with st.expander(
                    f"Evidence {i}: {result['title']}",
                    expanded=True
                ):

                    st.write("Document ID:", result["doc_id"])

                    st.write(
                        "Retrieval score:",
                        round(result["retrieval_score"], 4)
                    )

                    st.write(
                        "Reranking score:",
                        round(result["rerank_score"], 4)
                    )

                    st.write("NLI predictions:")

                    st.json(result["verification"])

                    st.write("Abstract:")

                    st.write(result["text"])

        except Exception as e:

            st.error("Something went wrong while processing your claim.")

            st.exception(e)


