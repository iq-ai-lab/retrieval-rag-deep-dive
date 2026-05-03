# 02. LSH — Locality Sensitive Hashing (Indyk & Motwani 1998)

## 🎯 핵심 질문

- Random projection $h(x) = \mathrm{sign}(w^\top x)$ 의 collision probability 가 각도 $\theta$ 에 대해 $\Pr[h(x)=h(y)] = 1 - \theta/\pi$ 인 이유?
- Multi-hash table (K tables, L probes per) 의 parameter tuning 은 어떻게 하는가?
- LSH 의 asymptotic complexity 는 $O(N^\rho)$ 인데, $\rho$ 가 뭔가?
- 실무에서 LSH 를 쓰지 않는 이유?

---

## 🔍 왜 LSH 는 첫번째 ANN 알고리즘인가

Indyk & Motwani (1998)은 처음으로 "서로 다른 점과는 collision probability 가 높고, 멀리 있는 점과는 낮은 hash function" 을 설계했다. 이것이 modern ANN 의 원점.

아이디어는 단순: **random projection 으로 차원을 축소** 하고, **hash table 을 여러 개** 운영해 high-probability 로 nearest neighbor 를 찾는다. 실제로는 k-means (IVF) 와 graph search (HNSW) 에 자리를 넘겼지만, 이론적 가치와 특정 domain (시뮬레이션, 스트리밍) 에서는 여전히 중요.

---

## 📐 수학적 선행 조건

- Random variables 와 probability (collision probability 계산)
- 각도와 벡터 기하학 (θ = arccos(x·y / ||x||||y||))
- Hash table 자료구조
- 조합론 (K, L parameter 와 recall trade-off)

---

## 📖 직관적 이해

### Random Projection 의 직관

```
원본 벡터 공간 (d=768 차원)
  │
  │ h = sign(w · x)  ← random direction w
  │                     1 bit hash 생성
  └─────────────────────────►
  
결과: 각 벡터마다 0 또는 1 (1비트)

비슷한 벡터들은?
  - 같은 hyperplane 의 같은 쪽 → h(x) = h(y) = 1 (collision)
  - 다른 쪽 → h(x) ≠ h(y) (no collision)
  
만약 x, y 의 각도가 작으면 (가까우면)
  → w 가 그들 사이를 분리할 확률이 낮음
  → collision probability 높음
```

### Multi-Hash Table 전략

```
Query q ∈ ℝ^d
   │
   ├─ Hash with table 1: h₁(q) = [bit₁, bit₂, ..., bitₖ]
   ├─ Hash with table 2: h₂(q) = [...]
   ├─ ...
   └─ Hash with table L: hₗ(q) = [...]
       │
       └─> Retrieve candidates from all L tables
           → Candidates might overlap, but high-probability
              contains true NN
```

**Intuition**: 1개 hash table 로는 miss 가능성 높음 → L 개로 redundancy 추가.

### Recall-Latency Trade-off 시각화

```
Recall ▲
    90%│     K=20, L=10 (expensive)
       │    ●
    80%│   ●
       │  ●
    70%│ ●
       │ ●
    50%│●
       └─────────────────────► Latency
         Nprobe (small) → large
```

---

## ✏️ 엄밀한 정의

### 정의 2.1 — LSH Family

Vector family $\{h: \mathbb{R}^d \to \{0, 1\}^K\}$ 는 **LSH family** for metric $(X, d)$ 라 함:
$$\exists p_1 > p_2 > 0, \, 0 < r_1 < r_2 : \quad
\begin{cases}
\Pr[h(x) = h(y)] \geq p_1 & \text{if } d(x, y) \leq r_1 \\
\Pr[h(x) = h(y)] \leq p_2 & \text{if } d(x, y) \geq r_2
\end{cases}$$

**직관**: "가까운 점들은 높은 확률로 collision, 먼 점들은 낮은 확률로 collision"

### 정의 2.2 — Random Projection Hash (Cosine Distance)

L2-normalized vectors $x, y \in S^{d-1}$ (unit sphere) 에 대해:
$$h(x) = \mathrm{sign}(w^\top x)$$
where $w \sim \mathcal{N}(0, I_d)$ (random Gaussian direction).

**Collision probability**:
$$\Pr[h(x) = h(y)] = 1 - \frac{\theta(x, y)}{\pi}$$

where $\theta(x, y) = \arccos(x^\top y) \in [0, \pi]$ 는 two vectors 사이의 각도.

### 정의 2.3 — K-bit Hashing and Multi-Table Lookup

**K-bit hash**: K 개의 independent random projections 결합:
$$H(x) = (h_1(x), h_2(x), \ldots, h_K(x)) \in \{0, 1\}^K$$

**Collision probability for K-bit hash**:
$$\Pr[H(x) = H(y)] = \left(1 - \frac{\theta}{\pi}\right)^K$$

**L-table scheme**: L 개의 independent hash tables, 각각 자신의 K-bit hash 함수 사용.

Query 시:
1. Compute $H_i(q)$ for $i = 1, \ldots, L$
2. Retrieve all vectors in bucket $H_i(q)$ from table $i$
3. Union all candidates, compute exact distance 로 verify

---

## 🔬 정리와 증명

### 정리 2.1 — Collision Probability for Random Projection

L2-normalized $x, y$ 에서 random Gaussian $w$:
$$\Pr[\mathrm{sign}(w^\top x) = \mathrm{sign}(w^\top y)] = 1 - \frac{\theta(x, y)}{\pi}$$

**증명**: 
- $w$ 가 uniform 이므로, $\mathrm{sign}(w^\top x)$ 와 $\mathrm{sign}(w^\top y)$ 는 $w$ 가 hyperplane (normal: unit vector between $x$ 와 $y$) 를 기준으로 같은 쪽인지에만 의존.
- 두 벡터 사이의 각도가 $\theta$ 이면, random direction $w$ 가 그들을 분리할 확률은 $\theta / \pi$.
- 따라서 분리하지 않을 확률 = collision probability = $1 - \theta/\pi$ $\square$

### 정리 2.2 — Recall with L-table Scheme

**Target**: query $q$ 에 대해 true nearest neighbor $x^*$ 를 찾을 확률을 높일 것.

$x^*$ 와 $q$ 사이의 각도가 $\theta_*$ 이면:
- Single table 에서 collision: $p = (1 - \theta_* / \pi)^K$
- L tables 에서 **최소 하나라도 collision**: $1 - (1-p)^L$

원하는 recall $(1-\delta)$ 얻으려면:
$$1 - (1-p)^L \geq 1 - \delta$$
$$L \geq \frac{\ln \delta}{\ln(1 - p)}$$

**Asymptotic complexity**: 
$$T_{\text{LSH}} = O(L \cdot (d + \text{nprobe})) = O(N^\rho)$$
where $\rho = \log_2 p / \log_2(1/c)$ and $c = \theta_{\max} / \pi$ (cutoff threshold ratio) $\square$

### 정리 2.3 — Optimal K, L Trade-off

Given target recall $R$, embedding dimension $d$, database size $N$:

To minimize wall-clock time:
$$\min_{K, L} \quad L \cdot (d \cdot K + \text{nprobe})$$
subject to $\quad 1 - (1 - (1 - \theta_*/\pi)^K)^L \geq R$

**Solution sketch** (not closed-form): Binary search on $K$, compute required $L$ from recall constraint. Typical: $K = 10-20, L = 5-100$.

---

## 💻 Python 구현 검증

### 실험 1 — Collision Probability Validation

```python
import numpy as np

def random_projection_hash(x, w):
    """Single bit hash: sign(w · x)"""
    return np.sign(w @ x)

def collision_probability_empirical(x, y, n_samples=10000):
    """Empirical collision probability"""
    d = len(x)
    collisions = 0
    for _ in range(n_samples):
        w = np.random.randn(d)
        w = w / np.linalg.norm(w)
        if random_projection_hash(x, w) == random_projection_hash(y, w):
            collisions += 1
    return collisions / n_samples

# Test with vectors at different angles
x = np.array([1.0, 0.0])
x = x / np.linalg.norm(x)

for angle_deg in [10, 30, 60, 90, 150]:
    angle_rad = np.deg2rad(angle_deg)
    y = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    empirical = collision_probability_empirical(x, y)
    theoretical = 1 - angle_rad / np.pi
    
    print(f"θ={angle_deg}°: empirical={empirical:.4f}, "
          f"theoretical={theoretical:.4f}, "
          f"error={abs(empirical - theoretical):.4f}")

# Expected: theoretical matches empirical within ~0.02
```

### 실험 2 — K-bit Hash Collision Probability

```python
def k_bit_hash(x, W):
    """K-bit hash: stack K random projections"""
    # W: (d, K) - each column is random direction
    return np.sign(W.T @ x)

def kbit_collision_empirical(x, y, K, n_samples=10000):
    """Empirical K-bit collision probability"""
    d = len(x)
    collisions = 0
    for _ in range(n_samples):
        W = np.random.randn(d, K)
        if np.all(k_bit_hash(x, W) == k_bit_hash(y, W)):
            collisions += 1
    return collisions / n_samples

# Verify: collision_prob(K-bit) ≈ (collision_prob(1-bit))^K
x = np.random.randn(768)
x = x / np.linalg.norm(x)

# y at angle 45 degrees (cosine similarity ≈ 0.707)
angle_rad = np.deg2rad(45)
y = np.random.randn(768)
y = y / np.linalg.norm(y)
# Adjust to exact angle (projection)
y = x * np.cos(angle_rad) + (y - (y @ x) * x) / np.linalg.norm(y - (y @ x) * x) * np.sin(angle_rad)

p1 = collision_probability_empirical(x, y, n_samples=1000)

for K in [1, 2, 4, 8, 16]:
    pK_empirical = kbit_collision_empirical(x, y, K, n_samples=1000)
    pK_predicted = p1 ** K
    print(f"K={K:2d}: empirical={pK_empirical:.4f}, "
          f"predicted={pK_predicted:.4f}")
```

### 실험 3 — L-table Retrieval Recall

```python
def build_lsh_index(X, K, L):
    """Build L LSH tables on database X"""
    # X: (N, d)
    N, d = X.shape
    tables = []
    hashes = []
    
    for l in range(L):
        W = np.random.randn(d, K)  # Random projections for table l
        hash_val = np.sign(X @ W)  # (N, K)
        bucket_dict = {}
        for i in range(N):
            key = tuple(hash_val[i])
            if key not in bucket_dict:
                bucket_dict[key] = []
            bucket_dict[key].append(i)
        tables.append(bucket_dict)
        hashes.append(W)
    
    return tables, hashes

def lsh_query(q, tables, hashes, X):
    """Query: retrieve candidates from all L tables"""
    candidates = set()
    for l, (table, W) in enumerate(zip(tables, hashes)):
        h_q = tuple(np.sign(W.T @ q))
        if h_q in table:
            candidates.update(table[h_q])
    return list(candidates)

# Benchmark
N, d, n_q = 100_000, 768, 100
K, L = 10, 20

X = np.random.randn(N, d).astype(np.float32)
X = X / np.linalg.norm(X, axis=1, keepdims=True)

tables, hashes = build_lsh_index(X, K, L)

# Exact NN baseline
def exact_nn(q, X, K_ret=100):
    dist = 1 - (q @ X.T)
    return np.argsort(dist)[:K_ret]

# Query one
q = X[0]
exact_nn_idx = set(exact_nn(q, X, K_ret=10))
lsh_candidates = set(lsh_query(q, tables, hashes, X))

recall = len(exact_nn_idx & lsh_candidates) / len(exact_nn_idx)
print(f"Recall: {recall:.2%}, Candidates retrieved: {len(lsh_candidates)}")
```

### 실험 4 — K, L Parameter Tuning

```python
def parameter_sweep(X, K_range, L_range, q, true_nn_idx, K_ret=10):
    """Find best K, L for target recall"""
    results = []
    
    for K in K_range:
        for L in L_range:
            tables, hashes = build_lsh_index(X, K, L)
            candidates = lsh_query(q, tables, hashes, X)
            recall = len(set(candidates) & true_nn_idx) / len(true_nn_idx)
            n_probes = len(candidates)
            results.append({
                'K': K, 'L': L, 'recall': recall, 
                'nprobe': n_probes, 'eff': recall / max(n_probes, 1)
            })
    
    return sorted(results, key=lambda x: -x['eff'])

# Test on small subset
X_test = X[:10000]
true_nn = exact_nn(X[0], X_test, K_ret=10)

results = parameter_sweep(X_test, K_range=[5, 10, 15], 
                          L_range=[10, 20, 30],
                          q=X[0], true_nn_idx=set(true_nn))

for r in results[:5]:
    print(f"K={r['K']}, L={r['L']}: "
          f"recall={r['recall']:.2%}, nprobe={r['nprobe']}, eff={r['eff']:.4f}")
```

---

## 🔗 실전 활용

| 시나리오 | LSH 적합 | 이유 |
|---------|--------|------|
| Static database, batch query | No | 1회성 index 구축 비용 높음 |
| Streaming data (online) | Yes | hash table update 빠름 (insertion O(1)) |
| Similarity 아닌 nearest neighbor | No | LSH 는 cosine/inner product 특화 |
| 매우 high-d (d > 50K) | Somewhat | K 가 커져야 해서 메모리 증가 |
| Massive scale (billions) | No | HNSW/IVF 가 더 효율적 |
| 이론적 보장 필요 | Yes | LSH 는 확률적 보장 가능 |

---

## ⚖️ 가정과 한계

- **정규화 벡터 가정**: L2-normalized 벡터만. Dense embedding 은 종종 정규화 필요 (비용 추가).
- **Gaussian random projection**: 다른 분포 (uniform on sphere) 도 가능하지만 이론은 같음.
- **Independence 가정**: K, L 개의 hash functions 가 independent — 실제 구현은 seed-based pseudo-random 으로 만족.
- **Uniform distribution over sphere**: 실제 embedding 은 특정 manifold 에 분포 → theoretical recall 이 optimistic일 수 있음.
- **Asymptotic analysis**: $N \to \infty$ 에 대한 $O(N^\rho)$ 는 작은 $N$ (< 1M) 에서는 상수항이 dominant.

---

## 📌 핵심 정리

$$\boxed{\Pr[h(x) = h(y)] = 1 - \frac{\arccos(x^\top y)}{\pi}, \quad L \geq \frac{\ln \delta}{\ln(1-p^K)}}$$

| 방면 | 수식 / 설정 |
|-----|-----------|
| 1-bit collision | $p = 1 - \theta/\pi$ |
| K-bit collision | $p^K = (1 - \theta/\pi)^K$ |
| L-table 회피확률 | $(1-p^K)^L$ (NN 놓칠 확률) |
| 목표 recall | $1 - (1-p^K)^L \geq 1-\delta$ |
| Asymptotic cost | $O(N^\rho)$, $\rho = \frac{\log_2 p}{\log_2(1/c)}$ |

> **핵심**: Random projection 의 collision probability 가 거리의 함수로 주어지면, L 개의 hash table 로 high-probability recall 보장. 하지만 실무에서는 k-means (IVF) 가 더 효율적.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: 두 벡터가 정확히 수직 (각도 90°) 이면 collision probability 는?

<details>
<summary>해설</summary>

$p = 1 - 90°/180° = 1 - 1/2 = 0.5$. 즉, random projection 으로 정확히 반반의 확률로 같은 쪽. 직관: 수직이면 hyperplane 의 위치가 두 벡터를 분리할 가능성이 정확히 반반.

</details>

**문제 2 (심화)**: K=10 bits, $\theta=45°$ 일 때 single table 의 collision probability 는? L=20 tables 로 "최소 하나라도 collision" 확률은?

<details>
<summary>해설</summary>

$\theta = 45° = \pi/4$ rad. $p_1 = 1 - (1/4) = 3/4 = 0.75$. 
$p_K = (0.75)^{10} \approx 0.0563$ (약 5.6%).
$P(\text{최소 1 collision in L tables}) = 1 - (1 - 0.0563)^{20} = 1 - (0.9437)^{20} \approx 1 - 0.326 \approx 0.674$ (약 67.4%). 

즉, 67% 확률로 nearest neighbor 를 찾음. Recall 을 95% 까지 높이려면 더 큰 L 필요 (L ≈ 50).

</details>

**문제 3 (논문 비평)**: "LSH 의 시간 복잡도가 $O(N^\rho)$ 인데, 실무에서 안 쓰는 이유는 상수항이 크기 때문" 이라는 주장 평가?

<details>
<summary>해설</summary>

부분적으로 맞음. 더 정확하게는: (1) 상수항 크다 — L tables 각 O(d) 메모리, K hashing 각 O(d) 시간. (2) 메모리 비효율 — hash function 을 L×K 개 저장해야 함. (3) 이론적 보장이 약하다 — embedding 이 균등 분포 아니면 worst-case 에 빠짐. (4) **가장 중요**: IVF (k-means) 가 비슷한 복잡도 에서 훨씬 better constant / 더 나은 실제 recall. FAISS 에서도 LSH 는 niche, IVF-PQ 가 mainstream.

</details>

---

<div align="center">

[◀ 이전 (01. Exact NN 한계)](./01-exact-nn-limits.md) · [📚 README](../README.md) · [다음 ▶ (03. IVF)](./03-ivf.md)

</div>
