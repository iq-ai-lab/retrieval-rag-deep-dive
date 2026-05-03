# 03. IVF — Inverted File Index

## 🎯 핵심 질문

- k-means clustering 으로 $K$ 개의 centroid 를 만들 때, query 는 어떻게 탐색하는가?
- nprobe 파라미터는 몇 개를 탐색할 지 결정하는데, 이 때의 시간 복잡도는?
- $K = \sqrt{N}$ 일 때 왜 $O(\sqrt{N})$ 복잡도가 가능한가?
- IVF-PQ 결합이 왜 FAISS 의 standard 인가?

---

## 🔍 왜 IVF 는 LSH 보다 현실적인가

IVF (Inverted File) 는 정보검색 (IR) 의 고전적 기법을 벡터 검색에 적용한 것이다. 핵심: **k-means 로 rough clustering** → nprobe 개의 cluster 만 탐색. LSH 는 확률론적 이지만 IVF 는 결정론적 (deterministic), 그리고 nprobe 를 조정하면 recall-latency trade-off 를 명시적으로 제어 가능.

더군다나, IVF 는 PQ (Product Quantization) 와 결합되어 IVF-PQ → 메모리-정확도 최적화의 실제 표준. FAISS (Facebook AI Similarity Search) 의 기본 index.

---

## 📐 수학적 선행 조건

- k-means clustering 알고리즘 및 converged centroid 개념
- Voronoi diagram (centroid 주변 cell)
- Distance 계산 (cosine, Euclidean)
- 선택(Selection algorithm) — top-K 구하기

---

## 📖 직관적 이해

### k-means Clustering 의 역할

```
Database 벡터들 (N개, 768차원)
   │
   ├─ k-means with K clusters
   │  → centroid: c₁, c₂, ..., c_K
   │
   └─ Assign each vector to nearest centroid
      v₁ → cluster 3
      v₂ → cluster 1
      v₃ → cluster 3
      ...
      
결과: K개의 "bucket", 각 bucket 에는 N/K 개의 벡터 (평균)

직관: "대략 어느 지역인지 먼저 찾고, 그 지역 내에서만 정밀 탐색"
```

### Inverted Index 구조

```
Centroid 별 저장:
  Centroid 1: [vec_idx_2, vec_idx_5, vec_idx_18, ...]
  Centroid 2: [vec_idx_1, vec_idx_7, vec_idx_9, ...]
  ...
  Centroid K: [vec_idx_3, vec_idx_4, ...]

Query 시:
  1. Compute distance(q, c_i) for all i=1..K (K번의 거리, 빠름)
  2. Sort centroids by distance
  3. nprobe 개의 가장 가까운 centroid 선택
  4. 그 centroid 들의 bucket 에서만 거리 계산 (정밀탐색)
```

### 시간 복잡도 개선

```
Linear scan: O(N·d)
     ↓
IVF: 
  · Coarse: O(K·d) — 모든 centroid 와 거리
  · Fine: O(nprobe·N/K·d) — 선택된 bucket들 내에서만
  · Total: O((K + nprobe·N/K)·d)
  
K = √N 로 설정하면:
  O((√N + nprobe·√N)·d) = O(nprobe·√N·d)
  
충분히 작은 nprobe 면 O(√N·d) ≈ 무시할 수 있을 정도로 빠름
```

### Recall-Latency Trade-off

```
Recall ▲
    100%│●
        │  ╲ exact
        │    ●
     95%│      ● ← nprobe=20
        │        ●
     90%│          ● ← nprobe=10
        │            ●
     80%│              ● ← nprobe=5
        │                ●
        │                  ● ← nprobe=1
     50%└──────────────────────────► Latency
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Inverted File Index

**구성**:
- $K$ centroids $\{c_1, \ldots, c_K\} \subset \mathbb{R}^d$ (from k-means)
- Inverted lists: $L_i = \{j : \arg\min_k d(x_j, c_k) = i\}$ (i번째 centroid 에 속한 벡터 indices)

**Query with nprobe**:
$$
\text{Retrieve}(q, \text{nprobe}) = \bigcup_{i \in \text{TopK}(q, \text{nprobe})} L_i
$$
where $\text{TopK}(q, \text{nprobe})$ = {$i$ : 상위 nprobe 가장 가까운 centroids 의 indices}

### 정의 3.2 — k-means Optimal Clustering

Given $X = \{x_1, \ldots, x_N\}$, find $K$ centroids $C = \{c_1, \ldots, c_K\}$ minimizing:
$$
\mathcal{L}(C) = \sum_{i=1}^N \min_k \|x_i - c_k\|^2
$$

Converged k-means 로 locally optimal $C$ 구함 (global optimal 는 NP-hard).

### 정의 3.3 — Asymptotic Complexity

Assuming **balanced clustering** (각 centroid 마다 $\approx N/K$ 벡터):

$$
T_{\text{IVF}} = \underbrace{O(K \cdot d)}_{\text{coarse search}} + \underbrace{O(\text{nprobe} \cdot \frac{N}{K} \cdot d)}_{\text{fine search}}
$$

**최적화**: $\frac{\partial T}{\partial K} = 0$ 풀면 $K^* \approx \sqrt{N}$ → $T_{\min} = O(\sqrt{N} \cdot d)$

---

## 🔬 정리와 증명

### 정리 3.1 — Complexity Reduction with Optimal K

If $K = c \sqrt{N}$ for some constant $c$, then:
$$
T_{\text{IVF}} = O(c\sqrt{N} \cdot d) + O(\text{nprobe} \cdot \frac{N}{c\sqrt{N}} \cdot d) = O((c + \text{nprobe} \cdot \frac{\sqrt{N}}{c}) \cdot d)
$$

For fixed nprobe (constant), total complexity becomes:
$$
T_{\text{IVF}} = O(\sqrt{N} \cdot d)
$$

**증명**: 
1. Coarse search: $K \cdot d = c\sqrt{N} \cdot d$.
2. Fine search: $\text{nprobe} \cdot \frac{N}{K} \cdot d = \text{nprobe} \cdot \frac{N}{c\sqrt{N}} \cdot d = \text{nprobe} \cdot \frac{\sqrt{N}}{c} \cdot d$.
3. $c$ 와 nprobe 가 상수 → 두 항 모두 $O(\sqrt{N} \cdot d)$ $\square$

### 정리 3.2 — Recall 과 nprobe 의 관계

Centroid 와 true nearest neighbor 의 거리 차이를 고려:

Let $\theta$ = angle between $q$ and its closest centroid. Then:
$$
P(\text{true NN in top-nprobe}) \approx 1 - \exp\left(-\text{nprobe} \cdot \frac{d\theta}{2}\right)
$$

(정밀한 analysis 는 embedding 분포에 의존 — empirical 하게 구함)

**일반적 경험**: nprobe $\approx 5-10$ 이면 95-98% recall, nprobe $\approx 50-100$ 이면 99%+ recall.

### 정리 3.3 — IVF-PQ Storage Reduction

IVF (centroids 만) + PQ (벡터들 양자화) 결합:

**IVF 단독**: $(K \cdot d) + (N \cdot d)$ floats 저장
**IVF-PQ**: $(K \cdot d) + (N \cdot m \log_2 k)$ bits 저장

where $m$ = subvector 개수, $k$ = codebook size per subvector.

예: $K=10K, d=768, N=100M$:
- 원본: $(10K \cdot 768 + 100M \cdot 768) \cdot 4$ bytes ≈ 307 GB
- IVF-PQ (m=96, k=256): $(10K \cdot 768 + 100M \cdot 96 \cdot 8)$ bytes ≈ 96 MB + 1.2 GB = 1.3 GB

**압축률 ≈ 235×** $\square$

---

## 💻 Python / FAISS 구현 검증

### 실험 1 — k-means Clustering 및 IVF Index 구축

```python
import numpy as np
from sklearn.cluster import KMeans
import time

# Generate data
N, d = 100_000, 768
X = np.random.randn(N, d).astype(np.float32)
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# k-means
K = int(np.sqrt(N))  # = ~316
print(f"K = {K}")

t0 = time.time()
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)
elapsed = time.time() - t0
print(f"k-means time: {elapsed:.1f}s")

# Build inverted lists
centroids = kmeans.cluster_centers_
inverted_lists = [[] for _ in range(K)]
for i, label in enumerate(labels):
    inverted_lists[label].append(i)

print(f"Cluster sizes: min={min(len(l) for l in inverted_lists)}, "
      f"max={max(len(l) for l in inverted_lists)}, "
      f"avg={np.mean([len(l) for l in inverted_lists]):.0f}")
```

### 실험 2 — Query with nprobe

```python
def ivf_search(q, centroids, inverted_lists, X, nprobe=10, K_ret=10):
    """IVF search: 1) coarse, 2) fine"""
    # Step 1: Coarse - compute distance to all centroids
    centroid_dists = 1 - (q @ centroids.T)  # shape (K,)
    
    # Step 2: Select top-nprobe centroids
    top_centroid_idx = np.argsort(centroid_dists)[:nprobe]
    
    # Step 3: Fine - retrieve candidates from selected buckets
    candidates = []
    for cid in top_centroid_idx:
        candidates.extend(inverted_lists[cid])
    
    # Step 4: Compute exact distance for candidates
    X_candidates = X[candidates]
    dists = 1 - (q @ X_candidates.T)
    
    # Return top-K
    top_K_idx = np.argsort(dists)[:K_ret]
    return [candidates[i] for i in top_K_idx]

# Test
q = X[0]
nprobes = [1, 5, 10, 50]
true_nn_idx = set(np.argsort(1 - (q @ X.T))[:10])

for nprobe in nprobes:
    results = ivf_search(q, centroids, inverted_lists, X, nprobe=nprobe)
    recall = len(set(results) & true_nn_idx) / 10
    n_probed = sum(len(inverted_lists[i]) for i in np.argsort(1 - (q @ centroids.T))[:nprobe])
    print(f"nprobe={nprobe:2d}: recall={recall:.2%}, "
          f"n_probed={n_probed:6d} ({n_probed/N*100:.1f}%)")
```

### 실험 3 — Latency Benchmark

```python
import time

def benchmark_ivf(X, q, centroids, inverted_lists, nprobe_range):
    results = []
    for nprobe in nprobe_range:
        times = []
        for _ in range(10):  # 10 queries
            q_test = X[np.random.randint(0, len(X))]
            t0 = time.perf_counter()
            _ = ivf_search(q_test, centroids, inverted_lists, X, nprobe=nprobe)
            times.append((time.perf_counter() - t0) * 1000)
        
        results.append({
            'nprobe': nprobe,
            'latency_ms': np.mean(times),
            'p99_ms': np.percentile(times, 99)
        })
    
    return results

results = benchmark_ivf(X, X[0], centroids, inverted_lists, [1, 5, 10, 20, 50])
for r in results:
    print(f"nprobe={r['nprobe']:2d}: latency={r['latency_ms']:.2f}ms, "
          f"P99={r['p99_ms']:.2f}ms")
```

### 실험 4 — FAISS 구현 비교

```python
try:
    import faiss
except ImportError:
    print("Install: pip install faiss-cpu")

# FAISS IVF Index
nlist = int(np.sqrt(N))
quantizer = faiss.IndexFlatL2(d)  # L2 distance
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Training (k-means)
index.train(X)
index.add(X)

# Query
nprobe_range = [1, 5, 10, 50]
q = X[0:1]

for nprobe in nprobe_range:
    index.nprobe = nprobe
    D, I = index.search(q, k=10)
    print(f"nprobe={nprobe:2d}: top-10 indices={I[0][:3]}...")
```

---

## 🔗 실전 활용

| 상황 | IVF 추천 | 이유 |
|------|--------|------|
| 100K-100M scale | Yes | 명시적 latency-recall trade-off |
| < 100K vectors | Sometimes | PQ 압축의 이점 부족 → linear scan 도 가능 |
| > 100M, Commodity HW | IVF-PQ | Memory 제약 극복 |
| Real-time filtering (metadata) | Yes | Centroid 기반 pruning 과 호환 |
| High-dim (d > 2000) | Yes | Coarse search 비용이 상대적으로 low |
| Streaming insertion | Somewhat | Re-clustering 필요 — 주기적 재학습 |

---

## ⚖️ 가정과 한계

- **Balanced cluster 가정**: k-means 가 완벽히 균등한 cluster 를 만든다고 가정 — 실제로는 imbalanced (특히 작은 K 에서).
- **Centroid 최적성**: converged k-means → local optimal. Initialization 에 따라 결과 달라짐.
- **Homogeneous distance**: 모든 쿼리가 비슷한 거리 분포라 가정 — 실제로는 쿼리마다 variability 높음.
- **No memory access pattern 고려**: centroid 간 거리 계산 후 bucket 접근 → cache miss 가능.
- **Exact distance in fine phase**: fine 단계에서도 모든 candidate 의 정확한 거리 계산 — 매우 큰 nprobe 에서는 linear scan 처럼 느려짐.

---

## 📌 핵심 정리

$$\boxed{T_{\text{IVF}} = O(K \cdot d) + O(\text{nprobe} \cdot \frac{N}{K} \cdot d), \quad \min_{K} T = O(\sqrt{N} \cdot d)}$$

| Aspect | Formula / Value |
|--------|----------------|
| Coarse search | K×d floating ops |
| Fine search | nprobe × N/K × d |
| Optimal K | $\sqrt{N}$ |
| Memory (centroids) | K × d floats |
| Memory (inverted lists) | $\approx$ N indices |

**비교**:
| Method | Complexity | Memory | Recall Control |
|--------|-----------|--------|-----------------|
| Linear scan | O(N·d) | O(N·d) | 100% (exact) |
| LSH | O(N^ρ) | O(L×d) | Probabilistic |
| IVF | O(√N·d) | O(√N·d) + O(N) | nprobe (explicit) |
| HNSW | O(log N·d) | O(N·d) | ef (implicit) |

> **핵심**: IVF 는 k-means clustering + nprobe 기반 trade-off. FAISS 의 표준. PQ 결합으로 대규모 시스템 구축 가능.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: N=100M, K=√N≈10K 일 때, nprobe=10 으로 평균 몇 개 벡터를 탐색하는가?

<details>
<summary>해설</summary>

각 centroid 당 평균 N/K = 100M/10K = 10K 개. nprobe=10 → 10×10K = 100K 개 벡터. Linear scan 의 100M 대비 0.1% 만 탐색. 시간복잡도: 100K×768 = 76.8M ops (single precision inner product).

</details>

**문제 2 (심화)**: k-means 학습 비용이 매우 크다 (100M 데이터, 수 시간). 이를 회피하는 방법?

<details>
<summary>해설</summary>

여러 대안:
1. **Sampling**: 전체 데이터 중 일부만으로 k-means (예: 10M samples 사용) → 빠르지만 centroid 품질 하락.
2. **Mini-batch k-means**: minibatch 단위로 점진적 학습 → 메모리 효율, iteration 많이 필요.
3. **Hierarchical k-means**: 2-level clustering (대략 coarse, 세밀 fine) → FAISS IndexIVF 의 변형.
4. **Pre-computed centroids**: 같은 embedding 모델에서 온 데이터라면 이전 학습 중심 재사용.

FAISS 는 GPU 기반 k-means (`faiss.Kmeans` with `gpu_resources`) 로 가속.

</details>

**문제 3 (논문 비평)**: "IVF 는 nprobe=1 일 때 recall 이 나쁘므로 비현실적" 이라는 주장 평가?

<details>
<summary>해설</summary>

이론적으로 부분 맞음: nprobe=1 이면 단 1개 centroid bucket 만 탐색 → true NN 이 다른 bucket 에 있을 확률 높음. 하지만 실제로는:
1. Embedding 이 manifold 에 분포 → nearest centroid 에 true NN 이 많은 확률 (random distribution 이 아님).
2. **nprobe=5-10 정도면 대부분 application 에서 충분** — 이미 linear scan 대비 100-1000배 빠름.
3. **Recall 과 latency 의 explicit trade-off 가능** — nprobe 조정하면 됨. LSH 나 HNSW 처럼 "학습 후 고정" 이 아님.

결론: 실용적이고 flexible.

</details>

---

<div align="center">

[◀ 이전 (02. LSH)](./02-lsh.md) · [📚 README](../README.md) · [다음 ▶ (04. PQ)](./04-pq.md)

</div>
