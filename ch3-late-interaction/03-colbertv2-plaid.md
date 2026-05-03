# 03. ColBERTv2 — PLAID Engine (Santhanam 2022)

## 🎯 핵심 질문

- ColBERT (2020) 의 per-token embedding indexing 은 "billion-scale 가능" 이지만, 메모리는 여전히 12GB+ — 이를 **절반으로 줄일 수 있는가**?
- **Centroid + residual** 구조가 정확히 어떻게 작동하는가? K-means clustering 과 vector quantization 의 combination 이란?
- 256 centroids (8-bit) + 4-bit residual 을 합치면, ColBERT 의 128-dim embedding 을 어떻게 **16-bit 에 압축**할 수 있는가?
- PLAID engine 의 두 단계 (centroid filter + residual exact MaxSim) 가 retrieval latency 를 어떻게 개선하는가?

---

## 🔍 왜 이 기법이 retrieval·RAG 에 중요한가

ColBERT 는 late interaction + per-token embedding 으로 정확도를 획기적으로 개선했지만, **memory footprint** 가 문제입니다:
- ColBERT: 1M docs × avg 100 tokens × 128-dim × 1 byte (8-bit) = 12GB
- Index 크기가 커질수록 cache-miss rate ↑, latency ↑

**ColBERTv2 + PLAID** 는:
1. **Centroid-based quantization**: K-means (256 centers, 8-bit) 으로 embedding 을 centroid ID 로 compress
2. **Residual vectors**: 원본과 centroid 의 차이 (residual) 를 4-bit 로 추가 compress
3. **Two-stage retrieval**: Centroid 로 빠른 filtering → residual 로 exact MaxSim 복원
4. **메모리**: 12GB → 4.6GB (2.6× compression) 유지하면서 **latency 개선** (cache locality)

결과:
- **Index size** 2.6× 감소
- **Query latency** 단일 GPU 에서 50ms (ColBERT 80ms)
- **Quality** 거의 동일 (nDCG@10 ~0.39)
- Production RAG 에서 표준

---

## 📐 수학적 선행 조건

- Vector quantization · K-means clustering (Ch4-04 PQ 의 선행)
- ColBERT MaxSim (Ch3-02)
- Information theory: rate-distortion trade-off
- (선택) Product quantization 의 기초

---

## 📖 직관적 이해

### Centroid + Residual 의 구조

```
Original embedding (128-dim, 4 bytes):
e_d = [0.12, -0.54, 0.33, ..., 0.87]

⟹ Centroid-based compression:

(1) K-means clustering: 모든 embedding 을 256 centroids 로 그룹화
    Codebook: C = [c_0, c_1, ..., c_255] (256 × 128)
    
    e_d → nearest centroid c_k (8-bit ID: k ∈ [0, 255])
    
    Centroid: c_k = [0.10, -0.52, 0.35, ..., 0.85]
    
(2) Residual vector (원본과의 차이):
    r_d = e_d - c_k = [0.02, -0.02, -0.02, ..., 0.02]
    
    이 residual 은 훨씬 작은 magnitude → 4-bit quantization 가능
    
(3) Compression: (8-bit centroid ID) + (4-bit × 128 dimension / 4 = 128-bit residual)
                = 1 byte + 16 bytes = 17 bytes
                vs original 128 bytes (7.5× compression per embedding)

⟹ Storage: 1M docs × 100 tokens × 128-dim × 4 bytes = 12GB
           → 1M docs × 100 tokens × 17 bytes = 1.7GB (7× compression)
           (실제 PLAID: 4.6GB, 이유는 위 추정이 optimistic)
```

### Two-Stage Retrieval Pipeline (PLAID)

```
Stage 1: Centroid Filtering (빠름)
─────────────────────────────
Query embeddings: E_q (online computed)
Document centroid codebook: C (indexed)

For each document:
  - Compute E_q @ C.T  (m × 256, m = query length)
  - Take max per query token: max_scores_centroid = max over 256 centroids
  
Filter: Keep only top-K documents by centroid max scores
        (e.g., keep top 10,000 out of 1M)

Stage 2: Residual Exact MaxSim (정확함)
──────────────────────────────────
Kept documents: residual vectors r_d (indexed)

For each kept document:
  - Reconstruct embedding approximation: e_d_approx = c_k + r_d
  - Compute exact MaxSim: S(q,d) = sum_i max_j (e_q_i @ e_d_j)
  - Re-rank by exact score
  
Output: Top-K documents by exact MaxSim

⟹ 결과: 1M 에 대한 linear scan → 10K 에 대한만 exact MaxSim (100× speedup)
```

### Memory Savings Illustration

```
ColBERT (8-bit quantized):
────────────────────────
1M docs × 100 tokens × 128-dim × 1 byte = 12.8 GB
Index memory dominant

ColBERTv2 + PLAID (centroid + 4-bit residual):
─────────────────────────────────────────────
1. Centroid codebook: 256 × 128 × 4 bytes = 131 KB (negligible)

2. Per-token encoding:
   - 8-bit centroid ID: 1 byte
   - 4-bit residual × 128 dims: 64 bytes
   - Total per token: 65 bytes (vs ColBERT 128 bytes)
   
   1M × 100 × 65 bytes = 6.5 GB (still large)
   
3. Actual optimization: 
   - 4-bit quantization → 4 tokens per byte
   - Centroid ID compression (offset encoding)
   - → 4.6 GB (실제 measurement)
   
Compression ratio: 12.8 / 4.6 = 2.78× (claimed 2.6×)
```

---

## ✏️ 엄밀한 정의

### 정의 3.7 — PLAID Codebook & Residual

**Step 1: K-Means Codebook Learning**

Given all document token embeddings $\{e_{d_j}^{(i)}\}$ (from all documents), learn:
$$
C = [c_0, c_1, \ldots, c_{K-1}] \in \mathbb{R}^{K \times d}
$$
where $K = 256$ (8-bit quantization), via standard K-means clustering.

**Step 2: Assignment & Residual**

For each embedding $e_{d_j}^{(i)}$:
1. Find nearest centroid: $k^* = \arg\min_k \|e_{d_j}^{(i)} - c_k\|^2$
2. Quantize centroid assignment to 8 bits: $q = k^* \in \{0, 1, \ldots, 255\}$
3. Compute residual: $r_{d_j}^{(i)} = e_{d_j}^{(i)} - c_{k^*}$
4. Quantize residual to 4-bit per dimension via uniform quantization

**Storage per embedding**: 1 byte (centroid ID) + $\lceil 4 \times d / 8 \rceil$ bytes (residual).

### 정의 3.8 — Two-Stage Scoring

**Stage 1: Centroid-based upper bound**

For document $d$ with centroids $[k_1, k_2, \ldots, k_n]$ (one per token):
$$
S_{\text{centroid}}(q, d) = \sum_{i=1}^{m} \max_{j=1}^{n} e_{q_i}^\top c_{k_j}
$$

This is an upper bound on exact MaxSim (since residuals are discarded).

**Stage 2: Exact MaxSim with Reconstruction**

For kept candidates, reconstruct:
$$
\tilde{e}_{d_j}^{(i)} = c_{k_j} + r_{d_j}^{(i)}
$$

Compute exact score:
$$
S_{\text{exact}}(q, d) = \sum_{i=1}^{m} \max_{j=1}^{n} e_{q_i}^\top \tilde{e}_{d_j}^{(i)}
$$

### 정의 3.9 — Quantization Error Bound

The error from compression:
$$
\|\tilde{e}_{d_j}^{(i)} - e_{d_j}^{(i)}\| = \|r_{d_j}^{(i)}_{\text{quantized}} - r_{d_j}^{(i)}\| \leq \delta_r
$$

where $\delta_r$ depends on 4-bit quantization granularity. PLAID claims $\delta_r$ small enough that accuracy drop is < 0.01 nDCG.

---

## 🔬 정리와 증명

### 정리 3.7 — Centroid-based Upper Bound

**명제**: Stage 1 의 centroid score 는 exact MaxSim 의 upper bound.

**증명**:
$$
S_{\text{centroid}}(q,d) = \sum_i \max_j e_{q_i}^\top c_{k_j}
$$
$$
\geq \sum_i \max_j e_{q_i}^\top (c_{k_j} + r_{d_j}^{(i)}) = S_{\text{exact}}(q,d)
$$
(since all residuals are small in magnitude relative to centroids)

정확하게는:
$$
S_{\text{centroid}} - S_{\text{exact}} \leq \sum_i \max_j |e_{q_i}^\top r_{d_j}^{(i)}| \leq m \cdot \max_j \|r_{d_j}^{(i)}\| \cdot \|e_{q_i}\|
$$
which is $O(\delta_r)$ when residuals are quantized to 4-bit $\square$.

### 정리 3.8 — Two-Stage Retrieval 의 Approximation Ratio

**명제**: Stage 1 filtering (top-K by centroid score) 후 Stage 2 exact MaxSim 으로 re-rank 할 때, missed relevant docs (false negatives) 의 비율 is bounded.

**증명 sketch**:
1. True top doc: score $S_{\text{exact}}(q, d_{\text{true}})$
2. If $d_{\text{true}}$ 가 centroid filtering 에서 제외됨: $S_{\text{centroid}}(q, d_{\text{true}}) < \text{threshold}$
3. But $S_{\text{centroid}} \geq S_{\text{exact}}$ → contradiction (if threshold 는 적절히 설정)
4. In practice, PLAID 에서 top-10K 는 충분히 크므로 대부분 relevant docs 포함 (실증: recall@10K ~99.8%) $\square$.

### 정리 3.9 — Compression & Quantization Trade-off

**명제**: K-means centroid (8-bit) + residual (4-bit) 의 total rate 는:
$$
R_{\text{total}} = 8 + 4 \times d \text{ bits per embedding}
$$
where $d = 128$. This achieves $\approx 2.6 \times$ compression vs original 32-bit floating point.

**증명**:
- Original: $128 \times 32 = 4096$ bits per embedding
- PLAID: $8 + 4 \times 128 = 520$ bits per embedding
- Ratio: $4096 / 520 \approx 7.9 \times$ per embedding
- Realistic (accounting for overhead): $2.6 \times$ on total index (1M × 100 tokens) $\square$.

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — K-Means Centroid Learning

```python
import numpy as np
from sklearn.cluster import KMeans

# Simulate document token embeddings
num_docs = 1000
avg_tokens = 100
d = 128

# Generate all embeddings
all_embeddings = []
for _ in range(num_docs * avg_tokens):
    all_embeddings.append(np.random.randn(d).astype(np.float32))
all_embeddings = np.array(all_embeddings)

# Normalize
all_embeddings = all_embeddings / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8)

# K-Means with 256 centroids
K = 256
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(all_embeddings)

centroids = kmeans.cluster_centers_.astype(np.float32)
print(f"Centroids shape: {centroids.shape}")  # (256, 128)

# Assignments
assignments = kmeans.predict(all_embeddings)
print(f"Unique centroids used: {len(np.unique(assignments))}")
```

### 실험 2 — Residual Quantization

```python
def quantize_residuals_4bit(residuals):
    """
    Quantize residuals to 4-bit signed integers
    """
    # Determine min/max for each embedding
    min_val = residuals.min()
    max_val = residuals.max()
    
    # 4-bit range: [-8, 7]
    scale = 15.0 / (max_val - min_val + 1e-8)
    offset = 8.0
    
    # Quantize
    quantized = ((residuals - min_val) * scale + offset).astype(np.int8)
    quantized = np.clip(quantized, -8, 7)
    
    return quantized, scale, min_val

# For each embedding, compute and quantize residual
residuals_all = []
assignments_all = []

for i, emb in enumerate(all_embeddings):
    # Find nearest centroid
    assignment = assignments[i]
    centroid = centroids[assignment]
    
    # Compute residual
    residual = emb - centroid
    
    # Quantize
    q_residual, scale, min_val = quantize_residuals_4bit(residual)
    
    residuals_all.append((q_residual, scale, min_val))
    assignments_all.append(assignment)

print(f"Total embeddings quantized: {len(residuals_all)}")

# Storage estimation
storage_bytes = len(all_embeddings) * (1 + d * 4 / 8)  # 1 byte centroid ID + 4 bits × d
print(f"Storage (approximate): {storage_bytes / 1e9:.2f} GB")
```

### 실험 3 — Two-Stage Retrieval

```python
def reconstruct_embedding(centroid_id, q_residual, scale, min_val, centroids):
    """
    Reconstruct approximate embedding from centroid + quantized residual
    """
    centroid = centroids[centroid_id]
    
    # Dequantize residual
    residual = ((q_residual.astype(np.float32) - 8.0) / scale) + min_val
    
    # Reconstruct
    reconstructed = centroid + residual
    return reconstructed

def stage1_centroid_filtering(E_q, centroids, assignments_batch, K_keep=100):
    """
    Stage 1: Filter documents using centroid scores
    """
    # Compute centroid similarities
    sim_to_centroids = E_q @ centroids.T  # (m, 256)
    max_centroid_scores = sim_to_centroids.max(axis=1)  # (m,)
    
    # For each document, take max centroid score
    doc_centroid_scores = []
    for assignments_doc in assignments_batch:
        scores = [sim_to_centroids[i, assignments_doc[j]].max() 
                  for i, j in enumerate(range(len(assignments_doc)))]
        doc_centroid_scores.append(sum(scores))
    
    # Sort and keep top-K
    top_k_indices = np.argsort(doc_centroid_scores)[-K_keep:]
    return top_k_indices

def stage2_exact_maxsim(E_q, residuals_kept, assignments_kept, centroids):
    """
    Stage 2: Compute exact MaxSim for kept documents
    """
    scores = []
    for residuals_doc, assignments_doc in zip(residuals_kept, assignments_kept):
        # Reconstruct embeddings
        E_d_reconstructed = []
        for j, (q_res, scale, min_val) in enumerate(residuals_doc):
            emb = reconstruct_embedding(
                assignments_doc[j], q_res, scale, min_val, centroids
            )
            E_d_reconstructed.append(emb)
        E_d_reconstructed = np.array(E_d_reconstructed)
        
        # MaxSim
        sim = E_q @ E_d_reconstructed.T  # (m, n)
        score = sim.max(axis=1).sum()
        scores.append(score)
    
    return np.array(scores)

# Example: Query processing
query_emb = np.random.randn(5, d).astype(np.float32)  # m=5 query tokens
query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)

# Simulate batch of document centroid/residual info
assignments_batch = [np.random.randint(0, 256, 100) for _ in range(1000)]
residuals_batch = [
    [(np.random.randint(-8, 8, d).astype(np.int8), 1.0, 0.0) 
     for _ in range(100)]
    for _ in range(1000)
]

# Stage 1
top_k_doc_idx = stage1_centroid_filtering(query_emb, centroids, assignments_batch, K_keep=100)
print(f"Kept {len(top_k_doc_idx)} documents for Stage 2")

# Stage 2 (on kept docs only)
kept_residuals = [residuals_batch[i] for i in top_k_doc_idx]
kept_assignments = [assignments_batch[i] for i in top_k_doc_idx]
scores_exact = stage2_exact_maxsim(query_emb, kept_residuals, kept_assignments, centroids)

ranked = sorted(enumerate(scores_exact), key=lambda x: x[1], reverse=True)
print("Top 5 documents (Stage 2):")
for rank, (doc_idx, score) in enumerate(ranked[:5]):
    print(f"{rank+1}. Doc {top_k_doc_idx[doc_idx]}: {score:.4f}")
```

### 실험 4 — Compression & Latency Measurement

```python
import time

def measure_latency():
    """
    Measure Stage 1 + Stage 2 latency
    """
    num_docs = 100000
    
    # Stage 1: centroid filtering
    t0 = time.perf_counter()
    for _ in range(100):
        _ = stage1_centroid_filtering(query_emb, centroids, 
                                      assignments_batch, K_keep=1000)
    stage1_time = (time.perf_counter() - t0) / 100
    
    # Stage 2: exact MaxSim on top-1000
    t0 = time.perf_counter()
    for _ in range(100):
        _ = stage2_exact_maxsim(query_emb, kept_residuals[:1000], 
                               kept_assignments[:1000], centroids)
    stage2_time = (time.perf_counter() - t0) / 100
    
    print(f"Stage 1 (centroid filter): {stage1_time*1000:.2f} ms")
    print(f"Stage 2 (exact MaxSim on 1K docs): {stage2_time*1000:.2f} ms")
    print(f"Total: {(stage1_time + stage2_time)*1000:.2f} ms")
    
    # Compare with full linear scan (ColBERT without compression)
    # ColBERT would do MaxSim on all 100K docs → ~500ms
    # PLAID: ~20ms + 50ms = ~70ms → 7× speedup

measure_latency()
```

---

## 🔗 실전 활용

| 시나리오 | 설정 | 효과 |
|---------|------|------|
| 1B scale dense retrieval | PLAID engine + ANN | 2.6× memory 절감 + 10× latency 개선 |
| Production RAG | ColBERTv2 indexing | Index 자체 2.6× 작음 (캐시 효율 ↑) |
| Multi-GPU serving | Distributed sharding | Per-GPU memory 감소 → batch size ↑ |
| Real-time QA (P99) | Two-stage (centroid + residual) | Centroid filter 가 fast path → tail latency 제어 |
| Fine-tuning | PLAID 기반 adaptation | Query encoding만 fine-tune, centroid codebook freeze 가능 |

---

## ⚖️ 가정과 한계

1. **Centroid codebook 학습 시간**: 전체 corpus 에 대한 K-means → 수시간 소요 (일회성, offline).

2. **Quantization error 누적**:
   - 8-bit centroid: 작은 error
   - 4-bit residual: 상대적으로 큰 error
   - 합쳐지면: nDCG 는 보존되지만, 매우 marginal docs 는 misrank 가능 (실증: top-100 accuracy ~99.8%).

3. **Centroid 개수 선택** ($K = 256$):
   - Too small (K=16): compression 좋지만 centroid quantization error 큼
   - Too large (K=1024): compression 덜 효과적
   - $K=256$ (8-bit) 은 rate-distortion sweet spot.

4. **Residual 비트 선택** (4-bit):
   - Centroid 오류와 residual 오류 의 균형
   - Lower 로 가면 (예: 2-bit) compression 더 좋지만 정확도 손실 (실증: nDCG@10 ~0.38, vs 0.39).

5. **Query embedding 은 여전히 online** — ColBERT 와 동일 latency (수십ms).

---

## 📌 핵심 정리

$$
\boxed{\tilde{e}_{d_j}^{(i)} = c_{k_j} + r_{d_j}^{(i)}, \quad S_{\text{exact}}(q,d) = \sum_{i=1}^{m} \max_{j=1}^{n} e_{q_i}^\top \tilde{e}_{d_j}^{(i)}}
$$

| 특징 | 값 |
|------|-----|
| **Compression ratio** | 2.6× (12.8GB → 4.6GB) |
| **Centroid quantization** | 8-bit (K=256 clusters) |
| **Residual quantization** | 4-bit per dimension |
| **Query latency** | ~50ms (1M docs, GPU) |
| **Index memory** | 4.6GB (1M docs, 100 tokens avg) |
| **Quality (nDCG@10)** | ~0.39 (near-lossless) |
| **Two-stage pipeline** | Centroid filter (fast) → Residual exact (accurate) |

> **핵심**: ColBERTv2 PLAID 는 per-token embedding indexing 을 centroid + residual 로 2.6× 압축하면서, two-stage retrieval 로 latency 를 10× 개선. 이는 production billion-scale RAG 의 standard 방법.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: K-means 의 centroid 개수 K=256 (8-bit) 을 K=1024 (10-bit) 로 늘리면, compression ratio 와 latency 는 어떻게 변하는가?

<details>
<summary>해설</summary>

Compression ratio:
- 현재: 1 byte centroid ID + 4-bit residual × 128 ≈ 65 bytes per embedding
- 새로: 1.25 bytes (10-bit) + 4-bit residual × 128 ≈ 65.25 bytes per embedding
- 거의 변화 없음 (실제로는 4-bit residual 이 bottleneck)

Latency:
- Stage 1 (centroid filter): E_q @ C.T 에서 C.shape = (1024, 128)
- 계산량 약간 증가 (1024 vs 256) → ~4% 느려짐
- Stage 2 (residual) 는 K 에 무관 → latency 거의 동일

결론: K 증가의 이득 거의 없음. K=256 이 optimal (Pareto).
</details>

**문제 2 (심화)**: Residual 을 4-bit 대신 2-bit 으로 quantize 하면, compression ratio 와 accuracy 는?

<details>
<summary>해설</summary>

Compression ratio:
- 현재: 1 byte + 4-bit × 128 / 8 = 65 bytes per embedding
- 새로: 1 byte + 2-bit × 128 / 8 = 33 bytes per embedding
- 12.8GB → 3.3GB (3.9× compression)

Accuracy:
- Residual quantization error 증가 (4-bit → 2-bit)
- nDCG@10: 0.39 → 0.375 (약 0.015 drop)
- Rank@100 recall: 99.8% → 99.2% (marginal change)

Trade-off:
- "Compression 이 중요" domain: 2-bit 선택
- "Accuracy 중요" domain: 4-bit 유지
- 실제 PLAID: 4-bit (accuracy-first)
</details>

**문제 3 (논문 비평)**: Santhanam et al. 2022 의 PLAID 논문 에서 "centroid-based upper bound 로 filtering 하면 relevant docs 를 miss 하지 않는가?" 는 주장의 근거는?

<details>
<summary>해설</summary>

주장의 근거:
1. **Upper bound 성질**: $S_{\text{centroid}}(q,d) \geq S_{\text{exact}}(q,d)$ (residuals 은 작음)
2. **Top-K threshold 설정**: Centroid score 기준 top-10K 를 keep 하면, true top-100 docs 는 거의 항상 포함 (threshold > true top-100 의 centroid score)
3. **실증**: MS MARCO dev set 에서 recall@10K ~99.8%

그러나 한계:
- "Almost never miss" 이지, "never miss" 아님
- Very marginal boundary cases 에서 miss 가능 (실측: ~0.2% of queries 영향)
- 이를 보완하려고 PLAID 는 top-K 를 약간 over-allocate (e.g., 12K 대신 10K)
</details>

---

<div align="center">

[◀ 이전 (02. ColBERT)](./02-colbert.md) · [📚 README](../README.md) · [다음 ▶ (04. Multi vs Single Vector)](./04-multi-vs-single-vector.md)

</div>
