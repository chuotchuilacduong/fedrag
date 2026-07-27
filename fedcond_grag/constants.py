"""Single source of truth for pipeline-wide constants.

Every stage (A tri-graph, B condensation + refine, QA cache, PPR maps,
passage anchors, Stage D/E training) must embed text with the SAME frozen
encoder — cosine similarities across artifacts are only meaningful in one
shared space. Import from here instead of re-declaring the model name.
"""

ENCODER_MODEL = "all-MiniLM-L6-v2"
ENCODER_DIM = 384

# Prefixed id for libraries that need the full HF path (e.g. SentenceTransformer
# accepts both, HippoRAG-style runners need the org prefix).
ENCODER_MODEL_HF = f"sentence-transformers/{ENCODER_MODEL}"
