# 01. Exact NN 의 한계와 ANN 의 동기

## 🎯 핵심 질문

- Exact nearest neighbor search (linear scan) 의 시간 복잡도는 정확히 $O(N \cdot d)$ 인가? 그 병목은 무엇인가?
- "Curse of dimensionality" 와 "high-dimensional distance uniform" 의 정확한 수학적 관계는?
- Recall-Latency trade-off 는 피할 수 없는가?
- MIPS (Maximum Inner Product Search) 를 NN search 로 축약할 수 있는가?

---

## 🔍 왜 Exact NN 이 비현실적인가

RAG 시스템의 bottleneck. 100M 개의 document embedding 에서 query embedding 과의 거리를 모두 계산하면 평균 50M 번의 거리 계산 — 단일 query 에 100ms 이상. 10K QPS 시스템은 즉시 비현실 (Q1 2024 기준 production RAG 는 1-10M scale 에서도 exact search 는 쓰지 않음).

고차원 벡터 공간에서의 거리 분포의 성질이 이를 악화시킨다: 임의 두 점 사이의 거리가 모두 비슷해져 (concentration of measure), 거리 기반 정렬이 의미를 잃음.

---

## 📐 수학적 선행 조건

- 벡터의 norm, inner product (Ch1 복습)
- 거리 공간: Euclidean, cosine similarity
- 확률 분포의 concentration inequalities (Hoeffding, Markov)
- 고차원 기하학 기초

---

## 📖 직관적 이해

### Linear Scan 의 병목

```
Query Q ∈ ℝ^d
   │
   ├─ Compute <Q, X₁> 
   ├─ Compute <Q, X₂>          ← d multiplies + d-1 adds per vector
   ├─ Compute <Q, X₃>               → O(d) 연산 × N vectors = O(N·d)
   ...
   └─ Compute <Q, Xₙ>
        │
        └─> Sort & return top K

실제 병목: 
  · CPU: memory bandwidth (cache miss 누적)
  · GPU: global memory latency
  · 결국: d=768, N=100M → 각 query 마다 100M×768 FLOPs
```

### Curse of Dimensionality (거리 분포의 균일화)

고차원에서 랜덤 벡터들 사이의 거리:
- 낮은 차원 (d=2): 거리가 "작음", "보통", "큼" 으로 구분 명확
- 높은 차원 (d=768): 모든 점이 거의 **같은 거리**에 분포 (표준편차 → 평균 비율이 $O(1/\sqrt{d})$ 로 축소)

```
Concentration of Measure 예:
  d=10:   거리 분포 [1.0, 2.0, 3.0] (wide spread)
  d=100:  거리 분포 [3.09, 3.10, 3.11] (매우 좁음)
  d=768:  거리 분포 [..., ..., ...] (모두 ≈ 13.9, 표준편차 ≈ 0.03)
```

이 결과: **순위가 불안정** — top-10 의 boundary 가 명확하지 않음 → "1등과 100등의 거리 차이가 무의미"

### Recall-Latency Trade-off

```
정확도 (Recall@K)  ▲
                  │    exact
                 99%├─────●
                     │      ╲
                 95% │       ╲  ← trade-off curve
                     │        ●  (ANN w/ nprobe=100)
                     │          ╲
                     │           ●  (LSH, nprobe=10)
                     │             ╲
                  50%│              ●  (fast but bad)
                     └──────────────────────► Latency (ms)
                          1     10    100
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Exact NN Search

**입력**: 
- Vector database $X = \{x_1, \ldots, x_N\} \subset \mathbb{R}^d$
- Query $q \in \mathbb{R}^d$
- Parameter $K \geq 1$

**출력**: 
$$\mathrm{NN}_K(q) = \{i_1, \ldots, i_K\} \quad \text{where} \quad d(q, x_{i_1}) \leq d(q, x_{i_2}) \leq \cdots \leq d(q, x_{i_K})$$

**거리 함수** (commonly used in RAG):
- **Euclidean**: $d_L(q, x) = \|q - x\|_2$
- **Cosine similarity** (normalized): $d_C(q, x) = 1 - \frac{q^\top x}{\|q\|_2 \|x\|_2}$
- **Inner product** (for MIPS): $d_{IP}(q, x) = -q^\top x$ (negative 는 maximization 을 minimization 으로 변환)

### 정의 1.2 — Concentration of Measure 정리

**정리 (비공식)**: $d$ 차원 unit sphere 에서 선택한 두 점의 거리 분포는 $d \to \infty$ 일 때:
$$\mathbb{P}\left[\left| \|x - y\| - \sqrt{2} \right| > t\right] \leq 2\exp\left(-\frac{d t^2}{4}\right)$$

**해석**: $d = 768$ 이면 거리는 $\sqrt{2} \approx 1.41$ 근처에 매우 집중 — "원점에서 거리 차이가 거의 없음"

### 정의 1.3 — Time Complexity

Linear scan:
$$T_{\text{exact}} = O(N \cdot d + N \log N) = O(N \cdot d) \quad \text{(with sorting)}$$

Approximate search (target):
$$T_{\text{approx}} = O(\mathrm{nprobe} \cdot d + \log N) \quad \text{where } \mathrm{nprobe} \ll N$$

---

## 🔬 정리와 증명

### 정리 1.1 — Memory-Bandwidth Bound for Linear Scan

Linear scan 은 memory-bound operation:
$$L \geq \frac{N \cdot d \cdot \text{dtype\_bytes}}{\beta}$$

where $\beta$ = memory bandwidth (GB/s).

**증명**: 각 query 마다 $N$ 개의 $d$-dimensional vectors 를 sequential access. Arithmetic intensity $= \frac{N \cdot d}{N \cdot d \cdot 4} = \frac{1}{4}$ (FP32 기준). A100 bandwidth 1.55 TB/s 로:
$$L \geq \frac{100\text{M} \cdot 768 \cdot 4}{1.55 \cdot 10^{12}} \approx 200\text{ms}$$

production 은 1-10ms SLA. $\square$

### 정리 1.2 — High-D Distance Distribution Tail

Unit sphere $S^{d-1}$ 에서 uniform 선택한 $x, y$ 의 거리 $\rho = \|x - y\|$:

**평균**: $\mathbb{E}[\rho] = 2 \Gamma((d+1)/2) / \Gamma(d/2) \approx \sqrt{\pi d / 2}$ for large $d$

**분산**: $\mathrm{Var}[\rho] = 1 - (\mathbb{E}[\rho])^2 = O(1)$ (상수, $d$ 무관)

**계수 (relative spread)**: $\sigma / \mathbb{E} = O(1/\sqrt{d}) \to 0$ as $d \to \infty$

**결론**: $d = 768$ 이면 거리들이 표준편차 < 1% 범위에 집중 $\square$.

### 정리 1.3 — MIPS-to-NN Reduction

Maximum Inner Product Search (MIPS) 문제:
$$\max_i q^\top x_i$$

를 NN search 로 축약 가능:

**변환** (Shrivastava & Li, 2014):
$$q^\top x \Leftrightarrow \|q' - x'\|^2$$
where $q' = (q, \sqrt{M - \|q\|^2})$, $x' = (x, \sqrt{M - \|x\|^2})$, $M = \max_i \|q\|^2 + \|x_i\|^2$.

**증명**: 
$$\|q' - x'\|^2 = \|q - x\|^2 + (M - \|q\|^2) + (M - \|x\|^2) - 2\sqrt{(M-\|q\|^2)(M-\|x\|^2)}$$

전개 후 정리하면:
$$\|q' - x'\|^2 = \text{const} - 2q^\top x$$

따라서 최소화 $\|q' - x'\|$ = 최대화 $q^\top x$ $\square$.

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — Linear Scan vs 고차원 거리 분포

```python
import numpy as np
import time

def linear_scan(Q, X, K=10):
    """Exact NN: compute all distances, sort"""
    # Q: (n_queries, d), X: (N, d)
    dist = 1 - (Q @ X.T) / (np.linalg.norm(Q, axis=1, keepdims=True) * 
                             np.linalg.norm(X, axis=1, keepdims=True))
    return np.argsort(dist, axis=1)[:, :K]

# Benchmark
N, d, n_q = 100_000, 768, 100
X = np.random.randn(N, d).astype(np.float32)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
Q = np.random.randn(n_q, d).astype(np.float32)
Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)

t0 = time.time()
results = linear_scan(Q, X)
elapsed = (time.time() - t0) * 1000
print(f"Linear scan {N}×{d}: {elapsed:.1f}ms ({elapsed/n_q:.2f}ms per query)")
# Expected: ~50ms for 100 queries at 100K vectors

# Verify: high-D distance uniformity
distances = 1 - Q @ X.T  # shape (n_q, N)
print(f"Distance mean: {distances.mean():.4f}, std: {distances.std():.4f}")
print(f"Relative spread (std/mean): {distances.std() / distances.mean():.4f}")
# Expected: very small relative spread
```

### 실험 2 — Concentration of Measure 시각화

```python
def concentration_demo(dims=[2, 10, 100, 768]):
    """See distance distribution as d increases"""
    N = 10000
    for d in dims:
        x = np.random.randn(N, d)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        
        y = np.random.randn(1, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        
        dists = np.linalg.norm(x - y, axis=1)
        print(f"d={d:3d}: mean={dists.mean():.4f}, "
              f"std={dists.std():.4f}, "
              f"std/mean={dists.std()/dists.mean():.4f}")

# Output (expected):
# d=  2: mean=1.4150, std=0.3961, std/mean=0.2799
# d= 10: mean=1.4082, std=0.1237, std/mean=0.0878
# d=100: mean=1.4093, std=0.0394, std/mean=0.0279
# d=768: mean=1.4147, std=0.0119, std/mean=0.0084
```

### 실험 3 — Recall-Latency Trade-off (preview)

```python
import heapq

def exact_search(Q, X, K=10):
    """True exact NN"""
    dist = 1 - (Q @ X.T)
    return np.argsort(dist, axis=1)[:, :K]

def recall_at_K(pred, true, K=10):
    """Compute recall: what % of true top-K are in pred top-K"""
    return np.mean([len(set(pred[i, :K]) & set(true[i, :K])) 
                    for i in range(len(pred))]) / K

# Will compare against ANN methods in later sections
true_nn = exact_search(Q, X, K=100)
print(f"Exact NN computed: shape {true_nn.shape}")
```

### 실험 4 — MIPS-to-NN Reduction 검증

```python
def mips_to_nn_transform(q, X):
    """Transform MIPS problem to NN search"""
    # Assume all vectors already L2-normalized
    # MIPS: max q^T x ≡ min ||q' - x'||
    M = 2.0  # sufficient bound: ||q||, ||x|| <= 1
    q_prime = np.concatenate([q, [np.sqrt(M - (q**2).sum())]])
    X_prime = np.concatenate([X, np.sqrt(M - (X**2).sum(axis=1, keepdims=True))], axis=1)
    return q_prime, X_prime

# Verify equivalence
q = Q[0]
X_test = X[:1000]
q_p, X_p = mips_to_nn_transform(q, X_test)

# MIPS directly
mips_scores = q @ X_test.T
top_k_mips = np.argsort(-mips_scores)[:10]

# NN on transformed
dist_transformed = np.linalg.norm(q_p - X_p, axis=1)
top_k_nn = np.argsort(dist_transformed)[:10]

print(f"MIPS top-10: {top_k_mips}")
print(f"NN (transformed) top-10: {top_k_nn}")
print(f"Match: {set(top_k_mips) == set(top_k_nn)}")
```

---

## 🔗 실전 활용

| 상황 | 해결책 | 이유 |
|------|------|------|
| Embedding dim > 500, N > 1M | 반드시 ANN 사용 | Linear scan 은 100ms+ → production SLA 위반 |
| 정규화된 임베딩 (cosine) | Dot product search 로 변환 | distance 계산 비용 절감 |
| MIPS 문제 (추천시스템) | Transform + NN 또는 직접 MIPS ANN | IVF-HNSW 는 cosine-optimized → rotation 필요 |
| 정확도 99% 필요 | 제한적 사용 (small scale 또는 offline) | production 에서 exact = outlier |
| Batch query 처리 | Matrix @ Matrix 활용 | 벡터화로 memory bandwidth 효율 개선 |

---

## ⚖️ 가정과 한계

- **정규화 가정**: L2-normalization 되었다고 가정 (실제로는 필수 전처리 아님, 성능상 관례)
- **Concentration 정리**: unit sphere 에서의 균등 분포 — 실제 embedding (BERT, BGE 등) 은 다른 분포일 수 있음
- **Memory-bound 분석**: cache 효율 무시 (L1/L2 cache miss 빈번 → 실제 지연 더 클 수 있음)
- **MIPS reduction**: 차원 증가 (원래 d → d+1) — high-d 에서는 상대적 오버헤드 무시할 수 있지만 d=64 같은 저차원에서는 주의 필요

---

## 📌 핵심 정리

$$\boxed{T_{\text{exact}} = \Theta(N \cdot d), \quad \text{recall} = 100\%, \quad \text{latency} \approx \frac{N \cdot d \cdot 4}{\beta}}$$

**High-Dimensional Geometry 의 역설**:
| 특성 | 저차원 (d=2-10) | 고차원 (d=768) |
|-----|----------------|----------------|
| 거리 분포 | 넓음 (구분 명확) | 매우 좁음 (균일) |
| Top-K 신뢰도 | 높음 | 낮음 (경계 모호) |
| 선형탐색 시간 | 빠름 | 비현실적 (200ms+) |
| ANN 근사 정당성 | 약함 | 강함 (손실 최소) |

> **핵심**: 정확성이 100% 이지만 **성능과 고차원 구조** 때문에 production RAG 는 ANN 으로 50-90% recall 로 1-10ms 를 택한다.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: 100M 개 embedding (dim=768, FP32) 에 100ms SLA 로 linear scan 이 가능한가?

<details>
<summary>해설</summary>

아니오. Linear scan: $T = \frac{100\text{M} \times 768 \times 4\text{ bytes}}{1.55\text{ TB/s}} \approx 200\text{ms}$ (A100 기준). 더군다나 이는 단일 query. 실제로는 메모리 대역폭 fully utilized 불가능 (cache miss, latency hiding 등). 따라서 100-300ms 범위.

</details>

**문제 2 (심화)**: 고차원 공간에서 "모든 점이 가까워진다" 는 것과 "ANN 의 근사 오차가 정당화된다" 사이의 인과관계?

<details>
<summary>해설</summary>

역설적이지만 연관 있음. (1) Concentration: 거리들이 모두 같아지면 정렬 순서가 불안정 → top-10 과 top-11 의 거리 차이가 무의미 (2) 따라서 "정확히 top-10 을 구하는 것" 이 중요도가 낮음 (3) 대신 "recall 95% 로 latency 10배 단축" 이 business value 로 훨씬 큼. 문제: **information leakage** — 거리가 균일해도 의미 있는 정보는 top-K 에 몰려 있을 수 있음 (manifold 구조). 경험적으로 768-dim embedding 에서 exact recall 의 25-50% 기여도가 충분.

</details>

**문제 3 (논문 비평)**: Indyk & Motwani (1998) 논문의 LSH 동기는 "exact search is hard" 라고 했는데, 오늘날 "exact 는 느린 것" 이라는 관점과의 차이?

<details>
<summary>해설</summary>

1998 년 당시: 이론적으로 "curse of dimensionality → 모든 NNS 는 lower bound Omega(N^(1/d))" (Karger & Ruhl 2002 형태의 결과들). 즉, 정보론적 복잡도 자체가 high-d 에서 피할 수 없음. 오늘날: 실무적으로는 "exact 의 time complexity 자체보다 memory bandwidth wall 이 주 병목". 즉, 이론과 실무의 "why ANN" 이유가 다름. 하지만 근본은 같음: high-d 에서의 내재적 어려움.

</details>

---

<div align="center">

[◀ 이전 (Ch3-04. Multi vs Single Vector)](../ch3-late-interaction/04-multi-vs-single-vector.md) · [📚 README](../README.md) · [다음 ▶ (02. LSH)](./02-lsh.md)

</div>
