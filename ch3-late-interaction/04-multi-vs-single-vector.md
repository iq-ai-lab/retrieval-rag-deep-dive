# 04. Multi-Vector vs Single-Vector — Trade-off Analysis

## 🎯 핵심 질문

- **DPR (single-vector dense retrieval)** vs **ColBERT (multi-vector)** vs **Cross-Encoder (full attention)** 는 정확히 3차원 (storage · quality · latency) 의 어느 곳에 위치하는가?
- 이 3가지 방법을 **Pareto frontier** 관점에서 보면, 어느 것이 "optimal" 이고 어느 것이 "dominated" 인가?
- 현실의 **billion-scale retrieval** 에서, corpus 크기와 latency budget 에 따라 어떤 방법을 선택해야 하는가 (1B vs 100M vs 1K rerank)?
- 새로운 retrieval 방법이 나올 때, **이 3차원 trade-off** 에서 기존 방법들을 dominate 해야만 의미가 있는 이유는?

---

## 🔍 왜 이 기법이 retrieval·RAG 에 중요한가

지금까지 Ch2 (dense retrieval) 와 Ch3 (late interaction) 에서 여러 방법들을 봤습니다:

- **Ch2-02 DPR**: single-vector embedding, $O(N)$ scoring (ANN 으로 sub-linear), nDCG ~0.35
- **Ch2-04 Contriever**: unsupervised dense, nDCG ~0.38
- **Ch2-05 E5**: supervised dense, nDCG ~0.40
- **Ch3-01 Cross-Encoder**: full BERT, $O(N)$ inference (rerank only), nDCG ~0.42
- **Ch3-02 ColBERT**: multi-vector (per-token), ANN + MaxSim, nDCG ~0.39
- **Ch3-03 ColBERTv2 PLAID**: quantized multi-vector, 2.6× compression, nDCG ~0.39

**어떤 방법을 써야 할까?** 단순히 "nDCG 가 높은 것" 으로 고르면 틀립니다. 왜냐하면:
1. **Storage**: PLAID 4.6GB vs DPR 3GB vs cross-encoder 0 (rerank only, corpus 별도)
2. **Latency**: DPR 10ms (ANN) vs ColBERT 50ms vs cross-encoder 10초/1000docs
3. **Quality**: DPR 35% vs ColBERT 39% vs cross-encoder 42%

**Pareto optimality** 관점에서, 이들을 3차원 공간에 그려야 합니다. 그러면 "최고 성능" 이 아니라 **상황에 맞는 선택** 이 가능합니다.

---

## 📐 수학적 선행 조건

- Pareto frontier / multi-objective optimization 의 개념
- Information retrieval metrics (nDCG, recall, MRR — Ch1-04)
- Dense retrieval · cross-encoder · late interaction 의 architecture (Ch2, Ch3-01～03)
- Big-O latency analysis (Ch1-01, Ch4)

---

## 📖 직관적 이해

### 3차원 Trade-off: Storage × Quality × Latency

```
3D Pareto Frontier (상상도):
                    Quality (nDCG@10)
                          ▲
                       0.42│     × Cross-Encoder
                          │    /
                       0.40│  × E5
                          │ / \
                       0.39│× ColBERT (PLAID)
                          │ \
                       0.35│  × DPR (with ANN)
                          │
                          └─────────────────────→ Storage (GB)
                         0   3  5   12   20
                         
                        Latency dimension (not shown):
                        DPR: 10ms
                        ColBERT: 50ms
                        Cross-Encoder: 10s (1K docs)

⟹ 각 방법은 다른 trade-off 지점
   - DPR: 저장소 작음, 빠름, 정확도 낮음
   - ColBERT: 중간, 중간, 중간
   - Cross-Encoder: 저장소 없음 (외부), 느림, 정확도 높음
```

### Quality vs Latency (2D Projection)

```
                  Quality (nDCG@10)
                        ▲
                     0.42│  ✗ Cross-Encoder
                        │  (but only on 10-100 docs)
                     0.40│  ● E5
                        │  ◆ Contriever
                     0.39│  ○ ColBERT
                        │
                     0.35│  ▪ DPR
                        │
                        └───────────────────────→ Latency (ms)
                        0  10  50  100 1000+

       파란색 = first-stage 가능 (ANN + sub-linear)
       검은색 = second-stage (reranking) 전용

⟹ Pareto frontier 는 우상향 (quality ↑, latency ↓ 불가)
```

### Storage vs Quality (2D Projection)

```
                  Quality (nDCG@10)
                        ▲
                     0.42│  ✗ Cross-Encoder (없음, corpus 별도)
                        │  (claim: reranker only)
                     0.40│  ● E5 (768-dim, 3GB)
                        │  ◆ Contriever
                     0.39│  ○ ColBERT (128-dim, 4.6GB)
                        │
                     0.35│  ▪ DPR (768-dim, 3GB)
                        │
                        └───────────────────────→ Storage (GB)
                        0  3  5  10  15

⟹ Multi-vector (ColBERT) 는 높은 quality 를 위해
   더 큰 storage 를 "trade" (token별 embedding)
```

### 현실의 시나리오별 선택

```
Corpus size × Latency budget → 최적 방법

1. Billion-scale (1B docs), strict latency (10ms P99):
   ┌─────────────────────────────────────┐
   │ Stage 1: DPR (ANN) or ColBERT (ANN) │
   │         → top 1000 candidates       │
   │ Stage 2: Cross-Encoder (optional)   │
   │         → top 10 reranked           │
   │ Cost: storage 5-10GB + low latency  │
   └─────────────────────────────────────┘

2. 100M docs, moderate latency (100ms):
   ┌─────────────────────────────────────┐
   │ Single-stage: ColBERT (PLAID)       │
   │ → MaxSim directly on top-K ANN docs │
   │ Quality nDCG~0.39 with latency ok   │
   └─────────────────────────────────────┘

3. Small corpus (1M docs), offline batch:
   ┌─────────────────────────────────────┐
   │ Linear scan: Cross-Encoder          │
   │ → Score all docs directly           │
   │ Highest quality, cost irrelevant     │
   └─────────────────────────────────────┘

4. Hybrid quality-critical (finance, legal):
   ┌─────────────────────────────────────┐
   │ Ensemble: DPR (recall) +            │
   │           ColBERT (interaction) +   │
   │           Cross-Encoder (final) via │
   │           RRF (Ch6-06)              │
   │ Cost: 2-3× compute, best quality    │
   └─────────────────────────────────────┘
```

---

## ✏️ 엄밀한 정의

### 정의 3.10 — 3D Performance Point

각 retrieval method $M$ 를 3-tuple $(S_M, Q_M, L_M)$ 로 표현:
$$
M = (S_M, Q_M, L_M)
$$

- $S_M$: index storage (GB)
- $Q_M$: quality = nDCG@10 (between 0 and 1)
- $L_M$: latency = P50 query time (milliseconds)

Example:
- DPR: $(3, 0.35, 10)$
- ColBERT: $(4.6, 0.39, 50)$
- Cross-Encoder: $(0, 0.42, 10000)$ (for 1K reranking; corpus separate)

### 정의 3.11 — Pareto Dominance

Method $M_1$가 method $M_2$ 를 **Pareto dominate** 한다는 것:
$$
M_1 \text{ dominate } M_2 \iff \begin{cases}
S_{M_1} \leq S_{M_2} & \text{(storage smaller or equal)} \\
Q_{M_1} \geq Q_{M_2} & \text{(quality better or equal)} \\
L_{M_1} \leq L_{M_2} & \text{(latency smaller or equal)} \\
\text{and at least one strict inequality}
\end{cases}
$$

**Pareto frontier**: dominated 되지 않는 모든 methods 의 집합.

### 정의 3.12 — Lattency under ANN

Single-vector method $M$ (e.g., DPR) 를 $N$ 개 documents 에서 run 할 때, ANN (e.g., HNSW, IVF) 와 함께 사용할 시:
$$
L_M^{\text{ANN}} = L_{\text{ANN}} + L_{\text{scoring}}
$$
where:
- $L_{\text{ANN}}(\alpha, N)$: ANN 으로 top $\alpha N$ candidates 찾기 (typical $\alpha = 0.001$ for 1K out of 1M)
- $L_{\text{scoring}}(\alpha N, d)$: $\alpha N$ candidates 에 대해 scoring

For DPR: $L_{\text{scoring}} = O(\alpha N \cdot d) = O(\alpha N)$ (dot product).
For ColBERT: $L_{\text{scoring}} = O(\alpha N \cdot m \cdot n) \approx O(\alpha N)$ (MaxSim, fixed doc len $n$).

### 정의 3.13 — Quality-Storage Trade-off Coefficient

단위 storage 당 얻을 수 있는 quality gain:
$$
\rho = \frac{\Delta Q}{\Delta S}
$$

Example:
- DPR → ColBERT: $\Delta Q = 0.04$, $\Delta S = 1.6$ GB → $\rho = 0.025$ (nDCG per GB)
- ColBERT → Ensemble: $\Delta Q = 0.03$, $\Delta S = 0$ (additional storage negligible) → $\rho = \infty$ (free quality!)

---

## 🔬 정리와 증명

### 정리 3.10 — Single-Vector 와 Multi-Vector 의 근본적 Trade-off

**명제**: Single-vector dense method (DPR) 는 항상 multi-vector method (ColBERT) 보다:
1. Storage 더 작음
2. Latency 더 짧음
3. Quality 더 낮음

(즉, Pareto frontier 에서 분리된 점들)

**증명**:
1. **Storage**: DPR 1 embedding/doc (768-dim × 4B) vs ColBERT m embeddings/doc (m tokens × 128-dim × 1B) → DPR <<.
2. **Latency (ANN + scoring)**: DPR dot product $O(d) = O(768)$ vs ColBERT MaxSim $O(m \times n)$ (though both $O(N)$ under ANN) → DPR faster in constant factors.
3. **Quality**: Information theory (Ch2 InfoNCE) 에 따르면, single vector 는 $\log_2(d) \approx 10$ bits of information (max) 포착. Multi-vector 는 $\sum_i \log_2(d) = m \times \log_2(d)$ bits (token별 정보 보존) → ColBERT more expressive $\square$.

### 정리 3.11 — Cross-Encoder 와 Dense 의 근본적 Trade-off

**명제**: Cross-Encoder 는 항상 dense method 보다:
1. Storage 더 작음 (index 불필요)
2. Latency 훨씬 김
3. Quality 더 높음

(즉, Pareto frontier 에서 다른 축)

**증명**:
1. **Storage**: Cross-encoder 는 inference 하면서 query 와 doc 의 조합을 real-time compute → index 불필요. Dense 는 embedding index 필수.
2. **Latency**: Cross-encoder full BERT forward pass ($O(d^2 \log d)$ attention, $d=768$) vs dense dot product ($O(d)$) → ~100배 느림.
3. **Quality**: Token-level full attention (Ch3-01 정리 3.1) 의 universal approximation capability → highest quality achievable (training data limit 내에) $\square$.

### 정리 3.12 — Billion-Scale 에서의 Optimal Strategy

**명제**: 1B documents + 10ms latency budget 일 때, DPR (or ColBERT) + Cross-Encoder 2-stage 가 single-stage 어떤 방법보다 Pareto-optimal.

**증명**:
1. **DPR single-stage**: ANN (sub-linear) + dot product → $L \approx 10$ms, $Q \approx 0.35$
2. **ColBERT single-stage**: ANN + MaxSim → $L \approx 50$ms (latency budget 초과), $Q \approx 0.39$
3. **Cross-encoder single-stage**: impossible (linear scan $L \gg 10$s, unreachable)
4. **2-stage (DPR + cross-encoder)**: 
   - Stage 1: DPR ANN top-1000 in 10ms
   - Stage 2: Cross-encoder on 1K docs ≈ 100ms (acceptable for offline batch)
   - Quality: $Q \approx 0.42$ (cross-encoder rerank 효과)
   - Overall latency: 110ms ≈ within budget for relaxed online service

Pareto frontier: 이 2-stage 가 다른 모든 단일 방법을 dominate $\square$.

### 정리 3.13 — Scalability 의 정량화

**명제**: Corpus 크기 $N$ 에 따라 optimal method 가 달라짐.

| Regime | Optimal Method | Reason |
|--------|----------------|--------|
| $N < 10K$ | Cross-Encoder linear | Full accuracy, latency irrelevant |
| $10K < N < 1M$ | ColBERT (no ANN) | Per-token info + manageable latency |
| $1M < N < 1B$ | DPR or ColBERT + ANN | Sublinear required, quality trade-off |
| $N > 1B$ | DPR + ANN (first) + optional rerank | Extreme scalability + tiered reranking |

Proof by cost analysis: For $N$ documents, cost $C(N) = C_{\text{storage}} + C_{\text{latency}} + C_{\text{quality loss}}$. Minimize $C$ over choice of method $\square$.

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — Three-Method Comparison Framework

```python
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RetrievalResult:
    method: str
    storage_gb: float
    quality_ndcg10: float
    latency_ms: float
    
    def dominates(self, other: 'RetrievalResult') -> bool:
        """Check if self Pareto-dominates other"""
        return (self.storage_gb <= other.storage_gb and
                self.quality_ndcg10 >= other.quality_ndcg10 and
                self.latency_ms <= other.latency_ms and
                (self.storage_gb < other.storage_gb or
                 self.quality_ndcg10 > other.quality_ndcg10 or
                 self.latency_ms < other.latency_ms))

# 실제 또는 literature 값들
methods = [
    RetrievalResult("DPR", storage_gb=3.0, quality_ndcg10=0.35, latency_ms=10),
    RetrievalResult("Contriever", storage_gb=3.0, quality_ndcg10=0.38, latency_ms=10),
    RetrievalResult("E5", storage_gb=3.0, quality_ndcg10=0.40, latency_ms=10),
    RetrievalResult("ColBERT", storage_gb=4.6, quality_ndcg10=0.39, latency_ms=50),
    RetrievalResult("ColBERTv2-PLAID", storage_gb=4.6, quality_ndcg10=0.39, latency_ms=50),
    RetrievalResult("Cross-Encoder (rerank 1K)", storage_gb=0, quality_ndcg10=0.42, latency_ms=1000),
]

# Pareto frontier 찾기
def find_pareto_frontier(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Find all non-dominated methods"""
    frontier = []
    for candidate in results:
        dominated = False
        for other in results:
            if other.dominates(candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier

frontier = find_pareto_frontier(methods)
print("Pareto Frontier Methods:")
for m in frontier:
    print(f"  {m.method}: storage={m.storage_gb}GB, quality={m.quality_ndcg10:.3f}, latency={m.latency_ms}ms")
```

### 실험 2 — Scenario-Based Selection

```python
def select_method_for_scenario(
    corpus_size: int,
    latency_budget_ms: float,
    quality_importance: float,  # 0.0 ~ 1.0
    frontier: List[RetrievalResult]
) -> RetrievalResult:
    """
    Select method based on scenario constraints
    """
    # Filter by latency constraint
    candidates = [m for m in frontier if m.latency_ms <= latency_budget_ms]
    
    if not candidates:
        raise ValueError(f"No method satisfies latency budget {latency_budget_ms}ms")
    
    # Score each candidate: weighted combination
    # weight: quality_importance * nDCG - (1 - quality_importance) * storage
    scores = []
    for m in candidates:
        score = (quality_importance * m.quality_ndcg10 - 
                 (1 - quality_importance) * m.storage_gb / 10)  # normalize storage
        scores.append((m, score))
    
    best_method, _ = max(scores, key=lambda x: x[1])
    return best_method

# Scenario 1: 1B scale, strict latency
scenario1 = select_method_for_scenario(
    corpus_size=1e9,
    latency_budget_ms=10,
    quality_importance=0.3,
    frontier=frontier
)
print(f"Scenario 1 (1B scale, 10ms): {scenario1.method}")

# Scenario 2: 100M scale, moderate latency, quality-focused
scenario2 = select_method_for_scenario(
    corpus_size=1e8,
    latency_budget_ms=100,
    quality_importance=0.8,
    frontier=frontier
)
print(f"Scenario 2 (100M scale, 100ms, quality-focused): {scenario2.method}")

# Scenario 3: Small scale, offline batch (no latency constraint)
scenario3 = select_method_for_scenario(
    corpus_size=1e6,
    latency_budget_ms=1e6,  # no constraint
    quality_importance=1.0,  # maximize quality
    frontier=frontier
)
print(f"Scenario 3 (1M scale, offline): {scenario3.method}")
```

### 실험 3 — 2-Stage Pipeline Simulation

```python
def two_stage_pipeline(
    corpus_size: int,
    first_stage: RetrievalResult,
    second_stage: RetrievalResult,
    top_k_first: int = 1000,
    top_k_final: int = 10
) -> Tuple[float, float, float]:
    """
    Simulate 2-stage retrieval pipeline
    Returns: (total_latency_ms, expected_quality, total_storage_gb)
    """
    # First stage: typically ANN-based
    # Assume ANN overhead is included in latency
    latency_stage1 = first_stage.latency_ms
    
    # Second stage: reranking on top-K
    # Approximate: if single-stage latency is T for corpus_size,
    # then T * (top_k_first / corpus_size) for top_k_first docs
    if second_stage.method == "Cross-Encoder (rerank 1K)":
        # Cross-encoder: approximately linear in num docs
        latency_stage2 = second_stage.latency_ms * (top_k_first / 1000)
    else:
        latency_stage2 = 0  # if second stage is MaxSim, already fast
    
    total_latency = latency_stage1 + latency_stage2
    
    # Quality: assume first stage recalls ~80% of top-K,
    # second stage improves top-10 to near-optimal
    quality_estimate = 0.95 * second_stage.quality_ndcg10  # slight discount for ranking error
    
    # Storage: sum of both stages
    total_storage = first_stage.storage_gb + second_stage.storage_gb
    
    return (total_latency, quality_estimate, total_storage)

# Example: DPR + cross-encoder for 1B scale
latency, quality, storage = two_stage_pipeline(
    corpus_size=1e9,
    first_stage=methods[0],  # DPR
    second_stage=methods[-1],  # Cross-Encoder
    top_k_first=1000,
    top_k_final=10
)
print(f"\n2-Stage (DPR + Cross-Encoder):")
print(f"  Total latency: {latency:.0f} ms")
print(f"  Expected quality: {quality:.3f}")
print(f"  Total storage: {storage:.1f} GB")
```

### 실험 4 — Quality vs Storage Trade-off Visualization

```python
import matplotlib.pyplot as plt

# Extract coordinates for visualization
storage_vals = [m.storage_gb for m in methods]
quality_vals = [m.quality_ndcg10 for m in methods]
labels = [m.method for m in methods]
colors = ['red' if m in frontier else 'gray' for m in methods]

plt.figure(figsize=(10, 6))
plt.scatter(storage_vals, quality_vals, c=colors, s=200, alpha=0.7)

for i, label in enumerate(labels):
    plt.annotate(label, (storage_vals[i], quality_vals[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

# Draw Pareto frontier convex hull (approximate)
frontier_storage = sorted([m.storage_gb for m in frontier])
frontier_quality = sorted([m.quality_ndcg10 for m in frontier])

plt.plot(frontier_storage, frontier_quality, 'b--', alpha=0.5, label='Pareto Frontier')
plt.xlabel('Storage (GB)')
plt.ylabel('Quality (nDCG@10)')
plt.title('Retrieval Methods: Quality vs Storage Trade-off')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('pareto_quality_storage.png')
plt.close()

print("Visualization saved as pareto_quality_storage.png")
```

---

## 🔗 실전 활용

| 상황 | 추천 방법 | 이유 |
|------|----------|------|
| 스타트업, 작은 corpus (<100M) | ColBERT (PLAID) | 좋은 quality-latency, 단순한 운영 |
| 생산 검색 엔진 (1B scale) | DPR (ANN) + optional cross-encoder rerank | 확장성 + latency 보장 |
| Legal/Finance (정확성 최우선) | Ensemble (DPR + ColBERT + Cross-encoder) via RRF | 최고 quality, cost 고려 X |
| 모바일 APP (극한 latency) | DPR + fast approximation (no rerank) | 간단 + 빠름 |
| RAG for LLM (depth > breadth) | ColBERT single-stage | 다양한 관점 포착 (per-token) |

---

## ⚖️ 가정과 한계

1. **Quality 측정**: nDCG@10 만 사용 — recall@K, MRR, Precision@K 등 다른 metric 에서는 순서 바뀔 수 있음.

2. **Latency 측정**: 단순화된 모델 — 실제로는 캐시, 병렬화, 배치 처리 등이 영향.

3. **Storage 비용**: 단순히 GB 수 — 실제로는 DRAM vs SSD, 압축, replication (고가용성) 등 다름.

4. **Quality 측정 dataset**: MS MARCO 기준 — 다른 domain (medical, scientific) 에서는 상대 순서 다를 수 있음.

5. **Hardware 의존성**: GPU 에서의 latency vs CPU-only 에서는 다름.

6. **Training cost**: "Pareto frontier" 는 모델이 fully trained 한 상태만 고려 — fine-tuning cost 무시.

---

## 📌 핵심 정리

$$
\boxed{\text{Pareto optimality: no single method dominates all dimensions (storage, quality, latency)}}
$$

| Method | Storage | Quality | Latency (1B scale) | 적성 |
|--------|---------|---------|-------------------|------|
| DPR | ⭐⭐⭐ (3GB) | ⭐⭐ (0.35) | ⭐⭐⭐ (10ms) | First-stage, extreme scale |
| E5 | ⭐⭐⭐ (3GB) | ⭐⭐⭐ (0.40) | ⭐⭐⭐ (10ms) | First-stage, good balance |
| ColBERT | ⭐⭐ (4.6GB) | ⭐⭐⭐ (0.39) | ⭐⭐ (50ms) | 100M scale single-stage |
| Cross-Encoder | ⭐⭐⭐ (0) | ⭐⭐⭐⭐ (0.42) | ⭐ (10s linear) | Reranking (10-1K docs) |
| **2-Stage (DPR+XE)** | **⭐⭐⭐** | **⭐⭐⭐⭐** | **⭐⭐** | **Billion-scale optimal** |

> **핵심**: Storage · Quality · Latency 는 근본적 trade-off. Pareto frontier 에서의 위치에 따라, 상황에 맞는 방법을 선택. "최고 품질" 이 아니라 "제약 조건에서의 최적" 을 추구.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: DPR (nDCG 0.35) 에서 ColBERT (nDCG 0.39) 로 업그레이드 시, storage 를 1.6GB 더 써야 한다. 이 trade-off 가 "가치있는가"? 정량화하라.

<details>
<summary>해설</summary>

비용-편익 분석:
- Quality gain: 0.39 - 0.35 = 0.04 (4% improvement in nDCG)
- Storage cost: 4.6 - 3.0 = 1.6 GB
- Latency cost: 50 - 10 = 40 ms

단위당 이득:
- Per GB: 0.04 / 1.6 = 0.025 nDCG per GB
- Per ms: 0.04 / 40 = 0.001 nDCG per ms

상황에 따라:
- Storage 싸고 latency budget 넉넉: ColBERT 선택
- Mobile / edge 환경: DPR 유지
- 중요한 것은 절대값 아니라 **제약 조건** 과의 매칭
</details>

**문제 2 (심화)**: Ensemble method (DPR + ColBERT + Cross-Encoder) 를 RRF 로 합치면, quality 는 각각의 max 에 근접할 수 있는가? Storage 와 latency cost 는?

<details>
<summary>해설</summary>

Quality:
- 각 방법의 ranking 이 다르면 → RRF 는 consensus ranking 생성 → quality ≈ 0.40-0.41 (max 0.42 보다 약간 낮음)
- 각 방법의 ranking 이 비슷하면 → RRF 는 "더 confident" ranking → quality ≈ max에 가까움

Storage:
- 3개 모두 index → 3.0 + 4.6 + 0 = 7.6 GB

Latency:
- Parallel: max(10, 50, negligible) = 50 ms (좋음)
- Sequential: 10 + 50 = 60 ms

Practical: Ensemble 은 quality ~0.40 (max 0.42 는 어렵지만 single-method 보다 robust) 이고 storage 7.6GB, latency 50ms → 중대형 서비스에서는 가치있음 (Pareto frontier 근처).
</details>

**문제 3 (논문 비평)**: "새로운 retrieval 방법이 기존 방법을 이기려면, 3차원 (storage, quality, latency) 에서 모두 개선하거나, 적어도 2개는 개선하고 1개는 동일해야 한다" 는 주장이 타당한가?

<details>
<summary>해설</summary>

타당성:
- Pareto optimality 정의상 "새 방법이 의미" → 기존 frontier 의 점들을 dominate 해야 함
- 만약 모든 3개를 악화시키면 → dominated → 의미 없음
- 1개만 개선, 2개 악화 → 여전히 dominated

그러나 예외:
- 특정 domain (예: domain-specific quality) 에 대해서만 비교 → frontier 가 달라질 수 있음
- 비용 (cost of training, computation cost) 을 추가 dimension 으로 보면 → 더 복잡
- 운영 편의성 (모델 size, framework support) 도 고려 → trade-off 재평가

결론: "원칙적으로 타당" 하지만, "상황과 정의에 따라 유동적"
</details>

---

<div align="center">

[◀ 이전 (03. ColBERTv2 PLAID)](./03-colbertv2-plaid.md) · [📚 README](../README.md) · [다음 ▶ (Ch4-01. Exact NN 한계)](../ch4-ann/01-exact-nn-limits.md)

</div>
