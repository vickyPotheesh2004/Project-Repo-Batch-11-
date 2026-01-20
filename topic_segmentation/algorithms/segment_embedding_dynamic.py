from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import numpy as np

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def segment(
    segments: List[Dict],
    min_topic_size: int = 3,
    std_factor: float = 0.8
) -> List[Dict]:

    if not segments:
        return []

    texts = [s["text"] for s in segments]
    embeddings = _MODEL.encode(texts)

    # 1️⃣ Compute similarity scores
    sims = []
    for i in range(1, len(embeddings)):
        sim = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]
        sims.append(sim)

    sims_np = np.array(sims)

    # 2️⃣ Dynamic threshold
    mean_sim = sims_np.mean()
    std_sim = sims_np.std()
    threshold = mean_sim - std_factor * std_sim

    # 3️⃣ Topic construction
    topics = []
    current = {
        "topic_id": 0,
        "segments": [segments[0]]
    }

    topic_id = 0

    for i in range(1, len(segments)):
        sim = sims[i - 1]

        boundary = (
            sim < threshold
            and len(current["segments"]) >= min_topic_size
        )

        if boundary:
            topics.append(current)
            topic_id += 1
            current = {
                "topic_id": topic_id,
                "segments": [segments[i]]
            }
        else:
            current["segments"].append(segments[i])

    topics.append(current)
    return topics
