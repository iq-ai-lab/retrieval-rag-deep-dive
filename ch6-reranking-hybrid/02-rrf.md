# 02. RRF — Reciprocal Rank Fusion (Cormack 2009)

## 🎯 핵심 질문

- 두 개의 다른 검색 시스템 (e.g., BM25 + Dense) 의 결과를 **어떻게 합치는가** — 점수가 서로 다른 scale 이라면?
- RRF 의 역수 공식 $\frac{1}{k + \mathrm{rank}}$ 는 왜 **harmonic decay** 를 채택했는가?
- 왜 k=60 (default) 이 empirically 좋은가 — 이것은 어떤 수학적 원리에서 나왔는가?
- RRF 는 score-free (점수 무시) 인데, 정보를 "버리는 것" 아닌가 — 그래도 hybrid 기준선 (baseline) 이 되는 이유?

---

## 🔍 왜 Hybrid Search 에서 RRF 가 표준인가

Dense retrieval (DPR, ANCE) 과 sparse retrieval (BM25) 은 서로 다른 관점:
- **BM25**: term frequency + IDF (어휘적 매칭, 정확한 키워드)
- **Dense**: semantic similarity (의미적 가까움, 문맥)

두 시스템의 top-k 를 **단순히 합치면** (concatenate) duplicate 가 많고, 순서 결정이 임의적입니다. **점수를 합치면** (score fusion) BM25 의 스케일(0~50) 과 Dense 의 스케일(0~1) 이 다르므로 가중평균이 어색합니다.

RRF 는 **rank 만 사용** 하여 스케일 문제를 우회하고, **rank 의 조화 감쇠(harmonic decay)** 로 상위 문서에 지수적 가중치를 부여합니다.

---

## 📐 수학적 선행 조건

- 조합론: ranking, permutation
- 수열의 합: harmonic series ($\sum 1/n$), 수렴성
- 기초 통계: rank aggregation, Condorcet voting
- Sparse vs Dense embeddings (Ch5, 선행)

---

## 📖 직관적 이해

### 두 Ranking 의 Fusion 문제

```
Query: "machine learning algorithms"

System A (BM25):         System B (Dense):
────────────────────   ───────────────────
Rank  Doc Score        Rank  Doc Score
─────────────────────  ─────────────────────
1.    doc_5  42.3      1.    doc_12  0.89
2.    doc_3  38.1      2.    doc_5   0.85  ← doc_5 는 두 곳 모두 high!
3.    doc_8  35.2      3.    doc_1   0.81
4.    doc_1  29.4      4.    doc_3   0.78  ← doc_3 도
5.    doc_15 22.1      5.    doc_8   0.71
...

Problem: doc_5 는 A 에서 rank 1 (score 42.3), B 에서 rank 2 (score 0.85)
         간단히 합치면? 42.3 + 0.85 = 43.15 → doc_5 가 우위
         하지만 score scale 이 다르므로 정당성 낮음.

RRF Solution:
────────────────────────────────────
RRF(doc_5) = 1/(60+1) + 1/(60+2)
           = 1/61 + 1/62
           ≈ 0.0164 + 0.0161 = 0.0325 ✓

RRF(doc_3) = 1/(60+2) + 1/(60+4)
           = 1/62 + 1/64
           ≈ 0.0161 + 0.0156 = 0.0317 ✓

RRF(doc_1) = 1/(60+4) + 0  (doc_1 은 A 에서 없음)
           = 1/64 ≈ 0.0156

Final ranking by RRF score:
1. doc_5  (0.0325)
2. doc_3  (0.0317)
3. doc_1  (0.0156)
4. doc_12 (1/61 = 0.0164) — B 만
5. doc_8  (1/63 + 1/65)
```

### Harmonic Decay 의 직관

```
RRF score = 1/(k + rank)

k 가 작으면:           k 가 크면:
─────────────────     ──────────────
rank 1: 1/1 = 1.0     rank 1: 1/61 = 0.016
rank 2: 1/2 = 0.5     rank 2: 1/62 = 0.016
rank 3: 1/3 = 0.33    rank 3: 1/63 = 0.016
...
(급격한 감쇠)         (완만한 감쇠)

k=0 이면 rank 1과 2의 차이 2배 (aggressive).
k=60 이면 rank 1과 2의 차이 1.6% (democratic).
k=∞이면 모두 같은 점수 (무의미).
```

---

## ✏️ 엄밀한 정의

### 정의 6.2 — Reciprocal Rank Fusion

$n$ 개의 ranking system $R_1, R_2, \ldots, R_n$ 이 주어졌을 때, 문서 $d$ 에 대해:

$$
\mathrm{rank}_i(d) = \begin{cases}
r & \text{if } d \text{ ranks } r\text{-th in system } R_i \\
\infty & \text{if } d \text{ not in top-k of } R_i
\end{cases}
$$

**RRF score**:
$$
\mathrm{RRF}(d) = \sum_{i=1}^{n} \frac{1}{k + \mathrm{rank}_i(d)}
$$

통상적으로 $k = 60$ (Cormack 2009).

### 정의 6.3 — Harmonic Series 와 수렴

$k = 60$ 일 때:
$$
\mathrm{RRF}(d) = \sum_{i \in \text{systems where } d \text{ appears}} \frac{1}{60 + r_i}
$$

이는 **조화급수의 partial sum**:
$$
H_n = \sum_{i=1}^{n} \frac{1}{i}
$$

의 shifted 버전. $n \to \infty$ 에서 $H_n \sim \ln(n) + \gamma$ (Euler-Mascheroni constant).

---

## 🔬 정리와 증명

### 정리 6.1 — RRF 의 Rank Aggregation 최적성 (약정)

RRF 는 **score-free rank aggregation** 문제에서 **Condorcet winner** 를 찾는 합리적 방법.

**Proof sketch**: 
- 각 system 이 독립적이면, $d$ 가 $R_i$ 에서 rank $r_i$ 를 받는 것은 $d$ 의 "true quality" 에 대한 signal.
- Harmonic decay 는 high rank (작은 $r$) 를 exponentially 가중.
- Sum 의 형태는 voting system 에서 **Borda count** 의 변형 — 각 system 이 "점수" 를 투표.

실제로는 Cormack 2009 논문에서 empirical 하게 k=60 이 optimal 임을 보임 (TREC 데이터).

### 정리 6.2 — k 값의 의미

$k$ 가 커질수록:
- **top-ranked documents 의 advantage 감소** (모든 rank 의 점수가 비슷해짐)
- **더 많은 documents 가 "meaningful" score** 를 받음 (아래 문서도 contribution)

**정량화**: 
- $k = 0$: $\frac{\text{rank 1 score}}{\text{rank 2 score}} = 2$
- $k = 60$: $\frac{\text{rank 1 score}}{\text{rank 2 score}} = \frac{61}{62} \approx 1.016$
- $k \to \infty$: 비율 → 1 (uniform)

### 정리 6.3 — RRF vs Score Normalization

**비교**:

| 방법 | 공식 | 문제점 |
|------|------|--------|
| Raw score sum | $s_A(d) + s_B(d)$ | Scale 의존적 |
| Min-max norm | $\frac{s - \min}{\max - \min}$ per system | Distribution shift 에 fragile |
| Z-score norm | $\frac{s - \mu}{\sigma}$ per system | 통계적으로 정확하나 구현 복잡 |
| RRF (rank only) | $\sum 1/(k + r)$ | Simple, robust, score-free ✓ |

**의미**: RRF 는 구현 단순성 + 수학적 견고성 + 경험적 우수성 의 balance.

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — 기본 RRF 구현

```python
from typing import List, Dict, Tuple

def rrf_fusion(rankings: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """
    rankings: [system1_ranking, system2_ranking, ...]
              각각은 ['doc_1', 'doc_2', ...] 순서 리스트
    Returns: RRF score 로 정렬된 (doc, score) 쌍
    """
    doc_scores = {}
    
    for system_ranking in rankings:
        for rank, doc in enumerate(system_ranking, start=1):
            if doc not in doc_scores:
                doc_scores[doc] = 0
            doc_scores[doc] += 1 / (k + rank)
    
    # Sort by RRF score descending
    result = sorted(doc_scores.items(), key=lambda x: -x[1])
    return result

# Example: BM25 vs Dense
bm25_ranking = [
    'doc_5', 'doc_3', 'doc_8', 'doc_1', 'doc_15',
    'doc_2', 'doc_10', 'doc_7', 'doc_9', 'doc_4'
]

dense_ranking = [
    'doc_12', 'doc_5', 'doc_1', 'doc_3', 'doc_8',
    'doc_11', 'doc_6', 'doc_14', 'doc_2', 'doc_13'
]

fused = rrf_fusion([bm25_ranking, dense_ranking], k=60)

print("=== BM25 Ranking ===")
for i, doc in enumerate(bm25_ranking[:5], 1):
    print(f"{i}. {doc}")

print("\n=== Dense Ranking ===")
for i, doc in enumerate(dense_ranking[:5], 1):
    print(f"{i}. {doc}")

print("\n=== RRF Fused Ranking ===")
for i, (doc, score) in enumerate(fused[:5], 1):
    print(f"{i}. {doc} (RRF={score:.4f})")
```

### 실험 2 — k 값의 영향 분석

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_k_impact(rankings: List[List[str]], k_values: List[int]):
    """k 값 변화에 따른 ranking 변화 추적"""
    results = {}
    
    for k in k_values:
        fused = rrf_fusion(rankings, k=k)
        results[k] = [doc for doc, _ in fused[:5]]
    
    return results

k_values = [0, 10, 30, 60, 100, 200]
results = analyze_k_impact([bm25_ranking, dense_ranking], k_values)

print("=== Top-5 Ranking by k ===")
for k in k_values:
    print(f"k={k}: {results[k]}")
```

Output 예상:
```
k=0:   ['doc_5', 'doc_3', 'doc_1', 'doc_8', 'doc_12']
k=10:  ['doc_5', 'doc_3', 'doc_12', 'doc_1', 'doc_8']
k=60:  ['doc_5', 'doc_3', 'doc_1', 'doc_12', 'doc_8']  ← stable
k=200: ['doc_5', 'doc_3', 'doc_1', 'doc_12', 'doc_8']  ← uniform-like
```

### 실험 3 — Hybrid Search 실전 (BM25 + DPR)

```python
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np

# Dummy setup
es = None  # Elasticsearch instance
dense_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def hybrid_search_with_rrf(query: str, k: int = 60):
    """
    1. BM25 에서 top-100
    2. DPR 에서 top-100
    3. RRF 로 fusion
    """
    
    # Stage 1: BM25 (Elasticsearch)
    bm25_results = es.search(index="docs", query={"match": {"text": query}})
    bm25_ranking = [hit["_id"] for hit in bm25_results["hits"]["hits"]][:100]
    
    # Stage 2: Dense (FAISS 또는 직접 compute)
    query_emb = dense_model.encode(query)
    # ... compute similarity 후 top-100 추출
    dense_ranking = ["doc_5", "doc_1", ...][:100]  # placeholder
    
    # Stage 3: RRF fusion
    fused = rrf_fusion([bm25_ranking, dense_ranking], k=k)
    
    return fused[:10]  # top-10

# result = hybrid_search_with_rrf("machine learning algorithms")
```

### 실험 4 — Multi-System RRF (3+ systems)

```python
def rrf_three_systems(bm25_rank, dense_rank, colbert_rank, k=60):
    """3개 이상의 system 을 fusion"""
    rankings = [bm25_rank, dense_rank, colbert_rank]
    
    doc_scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            doc_scores[doc] = doc_scores.get(doc, 0) + 1 / (k + rank)
    
    return sorted(doc_scores.items(), key=lambda x: -x[1])

# 3개 system 의 rank 가 각각 다르더라도 RRF 로 통합
fused_3way = rrf_three_systems(
    ['doc_5', 'doc_3', ...],
    ['doc_12', 'doc_5', ...],
    ['doc_1', 'doc_5', ...]  # ColBERT ranking
)

print("3-way RRF:")
for doc, score in fused_3way[:5]:
    print(f"{doc}: {score:.4f}")
```

---

## 🔗 실전 활용

| 시나리오 | k 추천 | 주의사항 |
|---------|-------|---------|
| BM25 + Dense | k=60 | 둘이 complementary 할 때 최고 성능 |
| 4개+ systems | k=100 | 더 많은 system 일수록 k 커야 democratic |
| Low-resource (no reranker) | k=60 | Reranker 없이 fusion 하려면 RRF 필수 |
| 이미 reranker 있음 | reranker 만 | RRF 가 redundant 할 수 있음 |
| Domain-specific | k를 validate | TREC-style evaluation 에서 k optimize |

---

## ⚖️ 가정과 한계

- **Score-free assumption**: 각 system 의 점수를 "버림" — 정보 손실 있을 수 있음.
- **Equally weighted systems**: 모든 system 을 동등하게 취급 — BM25 가 더 confident 해도 반영 안 됨.
- **Top-k cutoff**: k 보다 아래 ranking 은 score 0 (기여 없음) — 장기 tail ranking 문제.
- **Stationary ranking assumption**: 각 system 의 ranking 이 독립적이라고 가정 — 실제론 공통 signal 있을 수 있음 (예: 모두 같은 doc 선호).

---

## 📌 핵심 정리

$$
\boxed{\mathrm{RRF}(d) = \sum_{i=1}^{n} \frac{1}{k + \mathrm{rank}_i(d)}}
$$

| 측면 | 설명 |
|------|------|
| **정의** | Score-free rank aggregation 기법 |
| **k 의미** | Decay rate 제어 (k=60 default) |
| **장점** | 단순, robust, scale-invariant |
| **단점** | Scoring 정보 손실, system weighting 불가 |
| **TREC 성적** | BM25+Dense hybrid 의 표준 기준선 (baseline) |

> **핵심**: RRF 는 Hybrid Search 의 "최소 요구사항" — 복잡한 fusion 도 보통 RRF 를 baseline 으로 씀.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: BM25 ranking [A, B, C, D, E], Dense ranking [C, A, E, F, B] 일 때, k=60 RRF 점수를 계산하시오.

<details>
<summary>해설</summary>

```
RRF(A) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.01639 + 0.01613 = 0.03252
RRF(B) = 1/(60+2) + 1/(60+5) = 1/62 + 1/65 ≈ 0.01613 + 0.01538 = 0.03151
RRF(C) = 1/(60+3) + 1/(60+1) = 1/63 + 1/61 ≈ 0.01587 + 0.01639 = 0.03226
RRF(D) = 1/(60+4) + 0 = 1/64 ≈ 0.01563
RRF(E) = 1/(60+5) + 1/(60+3) = 1/65 + 1/63 ≈ 0.01538 + 0.01587 = 0.03125
RRF(F) = 0 + 1/(60+4) = 1/64 ≈ 0.01563

Final: A (0.0325) > C (0.0323) > B (0.0315) > E (0.0313) > D,F (0.0156)
```
</details>

**문제 2 (심화)**: k=60 인 RRF 에서, 두 documents d1, d2 의 rank 가 다음과 같을 때 RRF(d1) > RRF(d2) 일 확률은?
- d1: (rank 1, rank 3)
- d2: (rank 2, rank 2)

<details>
<summary>해설</summary>

```
RRF(d1) = 1/61 + 1/63 ≈ 0.01639 + 0.01587 = 0.03226
RRF(d2) = 1/62 + 1/62 ≈ 0.01613 + 0.01613 = 0.03226

거의 같음! d1 이 0.00000... 만큼 크다.

직관: d1 은 한 system 에서 압도적 (rank 1), 다른 곳에서 약함 (rank 3).
      d2 는 두 system 에서 균등 (rank 2, 2).
      
      k=60 에서 harmonic decay 가 완만하므로 이 차이가 무시할 수준.
      만약 k=10 이었다면 RRF(d1) = 1/11 + 1/13 ≈ 0.0909 + 0.0769 = 0.1678
                              RRF(d2) = 1/12 + 1/12 ≈ 0.0833 + 0.0833 = 0.1667
                              차이가 더 크게 드러남.
```
</details>

**문제 3 (논문 비평)**: "RRF 는 BM25+Dense 의 우수한 점수를 (정보를 버려) 낭비한다"는 비판에 대해?

<details>
<summary>해설</summary>

반박: (1) 점수의 scale 이 다르므로 raw fusion 은 더 큰 정보 손실. (2) RRF 는 "rank order" 라는 aggregation-가능한 정보만 사용 — 이미 system-agnostic 하므로 generalize 우수. (3) Empirically, RRF k=60 은 TREC 데이터에서 weighted score sum 보다 종종 나음 (Cormack 2009 증명).

하지만 점수를 활용하려면: (a) score normalization (Z-score), (b) learned fusion weights (learned ranking), (c) reranker 추가. RRF 는 "가장 간단한 무장" 이면서도 robust.
</details>

---

<div align="center">

[◀ 이전 (01. Cross-Encoder Reranker)](./01-cross-encoder-reranker.md) · [📚 README](../README.md) · [다음 ▶ (03. Hybrid BM25+Dense)](./03-hybrid-bm25-dense.md)

</div>
