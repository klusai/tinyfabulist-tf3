import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

# ============================================
# CONFIG
# ============================================
DATASET_NAME = "artifacts/ds-tf2-en-ro-3m-enriched"
TEXT_FIELD = "translated_fable"
EMBED_DIM = 384              # for MiniLM
BATCH_SIZE = 300
N_CLUSTERS = 30 
MAX_SAMPLES = 150000           
SAMPLES_PER_CLUSTER = int(MAX_SAMPLES/N_CLUSTERS)    
EMB_FILE = "embeddings.npy"
LABEL_FILE = "cluster_labels.npy"
OUTPUT_DIR = f"final_sft_fables_{MAX_SAMPLES}"

# ============================================
# LOAD DATASET
# ============================================
print("Loading dataset...")
# Handle both local paths (saved with save_to_disk) and HF datasets
if os.path.exists(DATASET_NAME):
    print(f"  Loading from local path: {DATASET_NAME}")
    ds = load_from_disk(DATASET_NAME)
else:
    print(f"  Loading from Hugging Face: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train", verification_mode="no_checks")

n = len(ds)
print(f"Dataset size: {n:,} samples")

# ============================================
# LOAD EMBEDDING MODEL
# ============================================
print("Loading embedder...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ============================================
# MEMORY-MAPPED EMBEDDING ARRAY
# ============================================
if not os.path.exists(EMB_FILE):
    print("Creating empty memmap embedding file...")
    emb = np.memmap(EMB_FILE, dtype="float32", mode="w+", shape=(n, EMBED_DIM))
else:
    print("Reusing existing embedding file...")
    emb = np.memmap(EMB_FILE, dtype="float32", mode="r+", shape=(n, EMBED_DIM))

# ============================================
# STEP 1: EMBED ALL FABLES (BATCHED)
# ============================================
if emb[0].sum() == 0:  # crude check to avoid re-embedding
    print("Embedding dataset...")
    for i in tqdm(range(0, n, BATCH_SIZE)):
        batch = ds[i:i+BATCH_SIZE][TEXT_FIELD]
        batch_emb = embedder.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        emb[i:i+len(batch_emb)] = batch_emb

    print("Flushing embeddings to disk...")
    emb.flush()
else:
    print("Embeddings already computed. Skipping.")

# ============================================
# STEP 2: K-MEANS CLUSTERING (MiniBatch)
# ============================================
print("Clustering with MiniBatchKMeans...")

kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    batch_size=4096,
    random_state=42,
    verbose=1
)

kmeans.fit(emb)

# ============================================
# STEP 3: SAVE CLUSTER LABELS
# ============================================
labels = kmeans.labels_.astype(np.int32)

np.save(LABEL_FILE, labels)
print(f"Saved cluster labels → {LABEL_FILE}")

# ============================================
# STEP 4: SELECT TOP SAMPLES PER CLUSTER
# ============================================
print("\nSelecting top samples from each cluster using centroid similarity...")

selected_indices = []

for c in tqdm(range(N_CLUSTERS), desc="Processing clusters"):
    # Get all indices belonging to this cluster
    cluster_idx = np.where(labels == c)[0]
    
    if len(cluster_idx) == 0:
        print(f"Warning: Cluster {c} is empty!")
        continue
    
    # Get embeddings for this cluster
    cluster_emb = emb[cluster_idx]
    
    # Compute centroid
    centroid = cluster_emb.mean(axis=0, keepdims=True)
    
    # Compute similarity to centroid
    sims = cosine_similarity(cluster_emb, centroid).flatten()
    
    # Select top N most similar to centroid
    n_select = min(SAMPLES_PER_CLUSTER, len(cluster_idx))
    top_local_idx = np.argsort(sims)[::-1][:n_select]
    top_global_idx = cluster_idx[top_local_idx]
    
    selected_indices.extend(top_global_idx.tolist())

print(f"\nTotal selected samples: {len(selected_indices):,}")

# ============================================
# STEP 5: CREATE FINAL DATASET
# ============================================
print("Creating final SFT dataset...")

final_ds = ds.select(selected_indices)

print(f"Saving to {OUTPUT_DIR}...")
final_ds.save_to_disk(OUTPUT_DIR)

print(f"\n✓ DONE! Generated {len(final_ds):,} fables for SFT")
print(f"✓ Dataset saved to: {OUTPUT_DIR}")
print(f"✓ {N_CLUSTERS} clusters × ~{SAMPLES_PER_CLUSTER} samples each")
