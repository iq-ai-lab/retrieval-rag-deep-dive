# 05. HNSW — Hierarchical Navigable Small World (Malkov 2018)

## 🎯 핵심 질문

- Multi-layer graph 에서 $P(l) = \exp(-l \ln m_L)$ 확률로 각 점이 layer $l$ 에 나타나는 이유?
- Skip-list 구조와의 동치성은?
- Greedy search 로 $O(\log N)$ complexity 가 정말 가능한가?
- M (max degree) 와 ef (search expansion factor) 의 trade-off 는?

---

## 🔍 왜 HNSW 는 Graph-based ANN 의 표준이 되었는가

HNSW (Hierarchical Navigable Small World, Malkov & Yashunin 2018) 는 근 5년간 가장 실무적인 ANN 알고리즘. 이유:
1. **이론적 보장**: $O(\log N)$ expected search complexity
2. **좋은 상수**: 작은 M (degree), 빠른 search
3. **동적 insertion**: Online index update 가능 (IVF 와 달리 재학습 불필요)
4. **단순 구조**: Skip-list 와 유사한 elegant design

production RAG systems (Qdrant, Milvus, Weaviate) 의 기본 backend.

---

## 📐 수학적 선행 조건

- Graph theory: neighbors, distance, connectivity
- Skip list 자료구조
- Probabilistic data structures
- Search algorithm analysis

---

## 📖 직관적 이해

### Hierarchical Layer Structure

```
Layer 2 (coarse):    q ●──●────●      ← few nodes, far apart
                      │  │    │
Layer 1 (medium):    ●──●──●──●──●    ← more nodes
                     │ │ │ │ │ │
Layer 0 (detail):   ●─●─●─●─●─●─●─●   ← all nodes, dense edges

관찰:
  - Higher layer: sparse, long-range jumps
  - Lower layer: dense, local refinement
  - Query: start at top, progressively navigate down
```

### 확률적 Layer Assignment

```
각 새로운 점 p 에 대해:
  l_p = floor(-ln(U(0,1)) / ln(m_L))  where U ~ uniform[0,1]
  
결과:
  - 매우 큰 layer 에 들어갈 확률: 매우 낮음 (exponential decay)
  - 모든 점이 layer 0 에 포함
  - layer l 에 예상 노드 수: N · m_L^(-l)
  
예: m_L = 1/ln(2) ≈ 1.44
  - Layer 0: N 개 (100%)
  - Layer 1: N / 1.44 ≈ 70% N
  - Layer 2: N / 1.44² ≈ 50% N
  - Layer 3: N / 1.44³ ≈ 35% N
```

### Greedy Search Process

```
Query q
   │
   ├─ Enter at top layer (highest level point in index)
   │
   ├─ At layer l:
   │  1. Greedy: navigate to nearest neighbor of q in layer l
   │  2. Expand search: check neighbors of current point
   │  3. Repeat until convergence (local minimum)
   │
   ├─ Move to layer l-1 (finer level)
   │
   └─ At layer 0 (detail): output final top-K neighbors

Time: O(log N) expected (jump through layers) + O(M) per layer (local search)
Total: O((M + log N) · d)  [vs IVF: O(nprobe · N/K · d), vs Linear: O(N·d)]
```

### Memory and Degree Trade-off

```
M = max degree (per node per layer)
  - Small M (4-8): memory efficient, fast insert, but less connectivity
  - Large M (16-32): better recall, but more memory and insert overhead

ef = search expansion factor
  - ef=100: explore 100 neighbors during search, good recall
  - ef=10: fast search, lower recall
  - Tradeoff: explicitly controlled at query time (unlike k-means)
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — HNSW Graph Structure

**Multi-layer graph**: $G = \{G_0, G_1, \ldots, G_{L-1}\}$
- $G_l$ = undirected graph at layer $l$
- Each node $u$ in $G_l$ has at most $M_l$ neighbors (edges)
- $M_0 = M$ (baselayer), $M_l = M / 2^l$ for $l > 0$ (standard choice)

**Entry point**: designated starting node for search (highest layer node).

**Distance**: $d(u, v)$ = distance between vectors at nodes $u, v$ (Euclidean, cosine, etc.)

### 정의 5.2 — Probabilistic Layer Assignment

For newly inserted point $p$:
$$
l_p \sim \text{Exp}(m_L) \quad \Rightarrow \quad l_p = \lfloor -\ln(U) / \ln(m_L) \rfloor
$$
where $U \sim \text{Uniform}(0, 1)$ and $m_L$ = normalization factor (typical: $m_L = 1/\ln 2 \approx 1.44$).

**Property**: $\Pr[p \text{ appears in layer } l] = m_L^{-l}$.

### 정의 5.3 — Greedy Search (KNN-Search procedure)

**Input**: Query $q$, number of neighbors to return $K$, search expansion factor $ef$

**Output**: Set of $K$ nearest neighbors to $q$

**Algorithm**:
1. Start at entry point $p$ (highest layer)
2. For $l$ from $L-1$ downto $1$:
   - Greedy search at layer $l$ with $ef=1$ → nearest node $p_{l}$
3. At layer $0$:
   - Greedy search with expansion factor $ef$ → return $K$ nearest neighbors
4. Return top-$K$ from layer $0$ search

**Greedy search at single layer**:
- Maintain candidates (priority queue, best-first)
- While candidates not empty:
  - Pop nearest unvisited candidate
  - Check its neighbors
  - Add unvisited neighbors to candidates
  - Stop when nearest candidate is farther than current best

---

## 🔬 정리와 증명

### 정리 5.1 — Expected Layer Count

The expected number of layers $L$ for database of size $N$:
$$
L = \lfloor \log_{1/m_L}(N) \rfloor + 1 = \lfloor \ln(N) / \ln(1/m_L) \rfloor + 1
$$

For $m_L = 1/\ln(2) \approx 1.44$:
$$
L \approx \log_{1.44}(N) = \frac{\ln N}{\ln 1.44} \approx 1.4 \ln N
$$

**예시**: $N = 100\text{M}$:
- $\ln(100M) \approx 18.4$
- $L \approx 1.4 \times 18.4 \approx 26$ layers

**증명**: Skip-list 이론에서 나온 결과. Expected height ~ $\ln(N)$ $\square$

### 정리 5.2 — Skip-List Equivalence

HNSW 의 layer 구조는 **Skip-list** 와 동등:

**Skip-list**: sorted linked list 에 probabilistic levels 추가.

**HNSW**: metric space 에 probabilistic layers 추가 (distance 기반).

**Equivalence**: Layer 간 connection 이 skip-list 의 "jump" 과 유사 → search complexity $O(\log N)$ (skip-list 의 결과를 재활용).

### 정리 5.3 — Greedy Search Complexity

**Search complexity** at single layer with balanced connectivity:
$$
C_{\text{layer}} = O(M \log (L \cdot M))
$$

**Multi-layer search** (top to bottom):
$$
C_{\text{total}} = O\left((M + \log N) \cdot d\right)
$$

(assuming $M$ is constant, e.g., 5-16)

**증명 (sketch)**:
1. Top layer: only few nodes → fixed cost
2. Each subsequent layer: exponentially more nodes, but greedy search converges quickly (log factor per layer)
3. Layer 0: $O(M \log M)$ neighbors visited with $ef$ expansion (empirically small constant)
4. Sum: dominated by O((M + log N)) distance calculations × d dimension $\square$

### 정리 5.4 — Memory Cost per Node

Per-node memory:
$$
\text{Memory}(u) = d \cdot 32 + \sum_{l=0}^{L-1} M_l \cdot 8 \quad \text{(FP32 vector + edge pointers)}
$$

Assuming $M_l = M / 2^l$:
$$
\sum_{l} M_l \approx M \cdot \sum_l 2^{-l} = M \cdot 2
$$

**Total memory for N nodes**:
$$
\text{Memory} \approx N \cdot (d \cdot 32 + 2M \cdot 8) = N \cdot (32d + 16M)
$$

For $d=768, M=8$: $N \times (24576 + 128) = N \times 24704$ bytes ≈ 24.7 KB per vector (overhead tiny vs 768×32=3KB vector itself).

---

## 💻 Python / hnswlib / FAISS 구현 검증

### 실험 1 — Manual HNSW Implementation (Simplified)

```python
import numpy as np
from collections import defaultdict
import heapq

class SimpleHNSW:
    def __init__(self, dim, M=5, ef=200, m_L=1.0/np.log(2)):
        self.dim = dim
        self.M = M
        self.ef = ef
        self.m_L = m_L
        self.data = []  # list of vectors
        self.graph = defaultdict(lambda: defaultdict(list))  # graph[layer][node] = neighbors
        self.entry_point = None
    
    def insert(self, vec):
        idx = len(self.data)
        self.data.append(vec)
        
        # Assign layers
        l = int(-np.log(np.random.random()) / np.log(self.m_L))
        
        if self.entry_point is None:
            self.entry_point = idx
        else:
            # Find nearest nodes at each layer and connect
            for layer in range(min(l, max(self.graph.keys()) + 1) if self.graph else l, -1, -1):
                if layer not in self.graph:
                    self.graph[layer] = defaultdict(list)
                # Simple: just connect to entry point (not real HNSW)
                self.graph[layer][idx] = [self.entry_point]
                self.graph[layer][self.entry_point].append(idx)
    
    def search(self, query, k=10):
        """Simplified greedy search"""
        if self.entry_point is None:
            return []
        
        # Start at entry point
        candidates = [(np.linalg.norm(query - self.data[self.entry_point]), self.entry_point)]
        visited = {self.entry_point}
        
        # Greedy search (simplified)
        for _ in range(self.ef):
            if not candidates:
                break
            dist, node = heapq.heappop(candidates)
            
            # Check neighbors (only layer 0 in simplified version)
            for neighbor in self.graph[0].get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    d = np.linalg.norm(query - self.data[neighbor])
                    heapq.heappush(candidates, (d, neighbor))
        
        # Return top-k
        return sorted([(np.linalg.norm(query - self.data[i]), i) 
                       for i in visited])[:k]

# Test
N, d = 10000, 768
X = np.random.randn(N, d).astype(np.float32)
X = X / np.linalg.norm(X, axis=1, keepdims=True)

hnsw = SimpleHNSW(d, M=8, ef=200)
for i, vec in enumerate(X):
    hnsw.insert(vec)

q = X[0]
results = hnsw.search(q, k=10)
print(f"Search results (simplified): {[idx for _, idx in results[:5]]}")
```

### 실험 2 — hnswlib Library (실제 구현)

```python
try:
    import hnswlib
except ImportError:
    print("Install: pip install hnswlib")
    exit(1)

# Create index
dim = 768
M = 8  # max degree
ef_construction = 200  # search expansion during construction
ef_search = 50  # search expansion during query

index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=N, ef_construction=ef_construction, M=M)

# Add data
index.add_items(X, np.arange(N))

# Query
q = X[0:1]
labels, distances = index.knn_query(q, k=10, ef=ef_search)
print(f"Top-10 indices: {labels[0]}")
print(f"Top-10 distances: {distances[0]}")

# Verify recall
exact_top_10 = set(np.argsort(1 - (X @ X[0]))[:10])
hnswlib_top_10 = set(labels[0])
recall = len(exact_top_10 & hnswlib_top_10) / 10
print(f"Recall@10: {recall:.1%}")
```

### 실험 3 — Layer Distribution Validation

```python
m_L = 1.0 / np.log(2)
N = 100000

# Generate layer assignments for all nodes
layers = []
for _ in range(N):
    l = int(-np.log(np.random.random()) / np.log(m_L))
    layers.append(l)

layers = np.array(layers)

# Verify: P(layer >= l) ≈ m_L^(-l)
for l in range(5):
    observed_prob = np.mean(layers >= l)
    expected_prob = m_L ** (-l)
    print(f"Layer {l}: observed={observed_prob:.4f}, expected={expected_prob:.4f}")

# Expected output:
# Layer 0: observed≈1.0000, expected=1.0000
# Layer 1: observed≈0.7085, expected=0.6931 (m_L^(-1) ≈ 1.44)
# Layer 2: observed≈0.5028, expected=0.4806
# Layer 3: observed≈0.3582, expected=0.3329
```

### 실험 4 — M 과 ef Trade-off Benchmark

```python
M_values = [4, 8, 16, 32]
ef_search_values = [10, 50, 100, 200]

results = []
for M in M_values:
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=N, ef_construction=200, M=M)
    index.add_items(X, np.arange(N))
    
    for ef in ef_search_values:
        import time
        t0 = time.perf_counter()
        
        # Query 100 random queries
        for _ in range(100):
            q = X[np.random.randint(0, N):np.random.randint(0, N)+1]
            labels, distances = index.knn_query(q, k=10, ef=ef)
        
        elapsed = (time.perf_counter() - t0) * 1000 / 100  # ms per query
        
        # Compute recall
        q = X[0:1]
        labels, _ = index.knn_query(q, k=10, ef=ef)
        exact_top_10 = set(np.argsort(1 - (X @ X[0]))[:10])
        recall = len(set(labels[0]) & exact_top_10) / 10
        
        results.append({
            'M': M, 'ef': ef, 'latency_ms': elapsed, 'recall': recall
        })

# Print results
for r in sorted(results, key=lambda x: (x['M'], x['ef'])):
    print(f"M={r['M']:2d}, ef={r['ef']:3d}: "
          f"latency={r['latency_ms']:.2f}ms, recall={r['recall']:.1%}")
```

---

## 🔗 실전 활용

| 상황 | HNSW 추천 | 대안 |
|------|---------|------|
| Streaming insertion (dynamic) | Yes | 재학습 필요 없음 |
| Static large-scale (100M+) | Sometimes | IVF-PQ (메모리) 또는 HNSW (CPU) |
| Low-latency SLA (<10ms) | Yes | M/ef 튜닝으로 explicit control |
| Filtering + search | Yes | Payload index 지원 |
| GPU acceleration | No | CPU optimized. FAISS GPU HNSW 제한적 |
| Memory-constrained | No | IVF-PQ 추천 |
| Distributed search | No | centralized 설계 (sharding 복잡) |

---

## ⚖️ 가정과 한계

- **Skip-list 동치성**: theoretical $O(\log N)$ 은 skip-list 이론에서 나온 것 — 실제 metric space 에서 성능 보장 약함 (manifold 구조에 따라).
- **Balanced connectivity 가정**: 각 layer 에서 균등한 degree distribution 가정 — 실제로는 일부 "hub" 노드가 과도한 연결 (skewed).
- **고정 M, ef**: layer 0 에서 M 값이 모든 노드에 동일 — 최적화는 node-specific M 가능 (복잡).
- **Search expansion linearityef 에 선형 (ef=2× → latency 2×) — trade-off 가 명확하지만 fine tuning 필요.
- **Distance metric invariance**: 어떤 거리 함수든 동작하지만, 이론적 보장은 특정 metric 에만 (L2, cosine).

---

## 📌 핵심 정리

$$\boxed{L = \lfloor \ln(N) / \ln(1/m_L) \rfloor + 1, \quad \text{Search} = O((M + \log N) \cdot d)}$$

| 파라미터 | 범위 | 기본값 | 효과 |
|----------|------|-------|------|
| M (max degree) | 4-32 | 8-16 | ↑M: better recall, more memory |
| ef_construction | 100-1000 | 200 | offline tuning, one-time cost |
| ef_search | 10-500 | ef=M*2 | query-time tuning, latency trade-off |
| m_L | 1/ln(2)~1/ln(3) | 1/ln(2) | layer decay (theory fixed) |

**메모리 비교**:
| Method | Memory per vector |
|--------|------------------|
| Original | 768 × 4 = 3072 bytes |
| HNSW (M=8) | 3072 + 16×8 ≈ 3200 bytes |
| IVF | 3072 (+ shared centroid) |
| IVF-PQ (m=96, k=256) | 96 × 8 bits = 96 bytes |

> **핵심**: HNSW 는 $O(\log N)$ complexity 의 elegant graph-based ANN. M/ef 로 명시적 latency-recall trade-off. production 의 표준 (Qdrant, Milvus, Weaviate).

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: N=1M 일 때 HNSW 의 예상 layer 수는? (m_L = 1/ln(2))

<details>
<summary>해설</summary>

$L = \lfloor \ln(10^6) / \ln(1.44) \rfloor + 1 = \lfloor 13.8 / 0.365 \rfloor + 1 = \lfloor 37.8 \rfloor + 1 = 39$ layers.

즉, 대략 40 층. Top layer 의 노드는 매우 적음 (보통 1-3 개).

</details>

**문제 2 (심화)**: M=8 일 때 N=100M 데이터에서 메모리 overhead 는? (vector storage 제외)

<details>
<summary>해설</summary>

각 노드마다 edge pointers (neighbor indices) 저장. 평균 degree ≈ M × (1 + 1/2 + 1/4 + ...) ≈ 2M = 16 (모든 layer 합산).

각 포인터 8 bytes (64-bit index) → 16 × 8 = 128 bytes per node.

Total: 100M × 128 = 12.8 GB (vector 자체 300 GB 대비 4% overhead).

</details>

**문제 3 (논문 비평)**: HNSW 는 M=1 이면 단순히 linked list 가 되어 $O(N)$ 복잡도라는 것을 설명?

<details>
<summary>해설</summary>

맞음. M=1 이면 각 layer 에서 각 노드마다 1개의 neighbor 만 → "chain" 형태. Greedy search 할 때 chain 을 따라가야 하므로 → $O(N)$ scan.

M ≥ 4 정도가 되어야 hierarchical structure 의 이점 (exponential decay in layer nodes) 을 활용 가능. 

**Trade-off**: M ↑ → more connectivity, shorter search paths, but more memory.

</details>

---

<div align="center">

[◀ 이전 (04. PQ)](./04-pq.md) · [📚 README](../README.md) · [다음 ▶ (06. Vector DB 내부)](./06-vector-dbs.md)

</div>
