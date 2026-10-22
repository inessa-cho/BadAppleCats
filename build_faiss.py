import faiss
import numpy as np

embeddings = np.load("embeddings.npy").astype("float32")

index = faiss.IndexFlatIP(512)
index.add(embeddings)

faiss.write_index(index, "cats.index")

print("vectors:", index.ntotal)
print("saved: cats.index")
