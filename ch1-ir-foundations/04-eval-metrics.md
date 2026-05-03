# 04. 평가 Metric — Recall@k · MRR · MAP · NDCG

## 🎯 핵심 질문

- Recall@k, Precision@k, MRR, MAP, NDCG 는 각각 무엇을 측정하며, 언제 어느 것을 사용해야 하는가?
- NDCG 의 "discount" (log₂(i+1)) 는 어디에서 나오는가? 정보이론과의 연결은?
- "Graded relevance" (0/1 binary 가 아닌 0~5 점수) 에서 metric 은 어떻게 달라지는가?
- IR system 의 성능을 single number 로 표현할 수 있는가?

---

## 🔍 왜 평가 metric 이 retrieval·RAG 에 중요한가

Retrieval system 을 만드는 것보다 "정말 좋아졌는가" 를 측정하는 것이 더 어렵습니다. 잘못된 metric 으로 평가하면:

1. **거짓된 개선의 착각** — Precision 만 보고 recall 을 놓치거나 그 반대.
2. **응용별 특성 무시** — E-commerce search (높은 recall 필요) vs QA (한 개 정답) 은 다른 metric.
3. **Position bias 무시** — 상위 5개 중 1개가 relevant 와 상위 100개 중 1개가 relevant 는 다름.
4. **Graded relevance 미지원** — Binary (relevant/non-relevant) 만으로 "somewhat relevant" 를 표현 불가.
5. **A/B test 의 신뢰성** — 어떤 metric 으로 통계적 유의성을 판정할 것인가?

이 문서는 **단일 metric 의 한계와 각각의 강점** 을 명확히 하여, 응용별로 올바른 metric 을 선택할 수 있게 합니다.

---

## 📐 수학적 선행 조건

- 집합론: intersection, union, cardinality
- 순위 (ranking): position-based weighting
- 정보이론: entropy, discount factor 의 의미
- 평균 (mean): harmonic mean (F1), arithmetic mean

---

## 📖 직관적 이해

### 5가지 Metric 의 목표

```
Recall@k: "Top-k 중 얼마나 많은 relevant 를 찾았나"
         시나리오: 모든 relevant item 을 찾아야 함 (재현율)

Precision@k: "Top-k 중 relevant 의 비율은"
            시나리오: noise 를 피해야 함 (정밀도)

MRR (Mean Reciprocal Rank): "첫 번째 relevant 를 몇 위에서 찾았나"
                           시나리오: QA, fact verification (하나의 정답)

MAP (Mean Average Precision): "여러 relevant 를 찾되, 순서가 중요"
                            시나리오: 웹검색 (상위에 많을수록 좋음)

NDCG@k: "Graded relevance 를 고려한 누적 이득"
       시나리오: 별점 (1~5) 같은 relevance level 있을 때
```

### Position Discount 의 직관

```
Position 1 (top):    Document relevance gain = 1.0 (full value)
Position 2:          gain = 0.63 (log₂(3) ≈ 1.585 로 discount)
Position 5:          gain = 0.43 (log₂(6) ≈ 2.585)
Position 10:         gain = 0.30 (log₂(11) ≈ 3.459)
Position 100:        gain ≈ 0.11 (log₂(101) ≈ 6.658)

직관: 사용자는 하단의 relevant 보다 상단의 relevant 를 훨씬 더 가치있게 봄.
```

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Precision & Recall (Binary Relevance)

**Precision@k**:

$$P@k = \frac{|\text{relevant documents in top-k}|}{k}$$

범위: $[0, 1]$. 높을수록 좋음. "Top-k 중 정확한 비율".

**Recall@k**:

$$R@k = \frac{|\text{relevant documents in top-k}|}{|R|}$$

여기서 $|R|$ = 전체 relevant documents. 범위: $[0, 1]$. 높을수록 좋음. "전체 relevant 중 몇 % 를 찾았나".

**Trade-off**: 보통 threshold 를 낮추면 recall ↑, precision ↓. 상황에 따라 선택.

### 정의 4.2 — MRR (Mean Reciprocal Rank)

여러 queries 에 대해:

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

여기서 $\text{rank}_i$ = query $i$ 에서 첫 번째 relevant document 의 위치.

예시:
- 첫 번째 relevant 가 position 1: reciprocal rank = 1/1 = 1.0
- 첫 번째 relevant 가 position 5: reciprocal rank = 1/5 = 0.2
- No relevant: reciprocal rank = 0

**의미**: 첫 번째 정답을 얼마나 빨리 찾는가. **QA, fact retrieval** 에 적합.

### 정의 4.3 — MAP (Mean Average Precision)

각 query 에 대해:

$$\text{AP} = \frac{1}{|R|} \sum_{k=1}^{n} P(k) \cdot \text{rel}(k)$$

여기서:
- $n$ = total number of returned documents
- $P(k)$ = precision@k
- $\text{rel}(k)$ = 1 if document at rank $k$ is relevant, 0 otherwise

**예시**: 10개 retrieval, positions 2, 5, 7에서 relevant

$$\text{AP} = \frac{1}{3} \left( \frac{1}{2} + \frac{2}{5} + \frac{3}{7} \right) = \frac{1}{3}(0.5 + 0.4 + 0.429) = 0.443$$

여러 queries 평균:

$$\text{MAP} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{AP}_i$$

**의미**: Ranking quality 의 누적 평가. **웹검색, 문서 retrieval** 에 표준.

### 정의 4.4 — DCG & NDCG (Discounted Cumulative Gain)

**Relevance grade** $\text{rel}(k) \in \{0, 1, 2, 3, 4, 5\}$ (또는 0~1 continuous).

**DCG@k**:

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{\text{rel}(i)} - 1}{\log_2(i+1)}$$

분석:
- **분자** $2^{\text{rel}(i)} - 1$: Exponential gain, high relevance heavily weighted
  - rel=0: gain = 0
  - rel=1: gain = 1
  - rel=2: gain = 3
  - rel=3: gain = 7
  - rel=4: gain = 15
  - rel=5: gain = 31
- **분모** $\log_2(i+1)$: Position discount (log scale)
  - position 1: $\log_2(2) = 1$ (no discount)
  - position 2: $\log_2(3) \approx 1.585$
  - position 10: $\log_2(11) \approx 3.459$

**NDCG@k** (Normalized DCG):

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

여기서 **IDCG** (Ideal DCG) = 가능한 최고 순서일 때의 DCG:

$$\text{IDCG@k} = \sum_{i=1}^{k} \frac{2^{\text{rel}_i^{ideal}(i)} - 1}{\log_2(i+1)}$$

(모든 relevant documents 를 relevance score 순으로 상단에 배치)

범위: $[0, 1]$. 1.0 = 완벽 순서.

---

## 🔬 정리와 증명

### 정리 4.1 — NDCG 의 Information Theoretic 정당성

Position $k$ 의 discount $\log_2(k+1)$ 는 정보이론에서 나온다:

Binary search tree 에서 $n$ 개 item 검색 시 평균 비용은 $\approx \log_2 n$. 유사하게, user 가 position $k$ 까지 document 를 보기 위해 필요한 "effort" 는 $\log_2(k+1)$ 에 비례.

따라서 NDCG 의 discount 는 **user interaction cost** 를 반영 — "노력 대비 얻은 이득".

**증명 sketch**: Assume user 는 sequential 하게 top-1, top-2, ... 를 본다. 각 position 을 보기 위한 marginal effort 는 geometric 증가. Inverse 를 취하면 logarithmic discount. $\square$

### 정리 4.2 — Recall vs MAP 의 complementarity

어떤 query 에서:
- Recall@100 = 90% (100개 중 90개의 relevant 를 찾음)
- MAP = 0.4 (하지만 그 90개가 많이 상단에 없음)

이는 recall 은 높지만 ranking 이 좋지 않다는 뜻. 반대로:
- Recall@100 = 50%
- MAP = 0.7

이는 찾은 것이 적지만 매우 상단에 있다는 뜻. **따라서 MAP 은 ranking quality, Recall 은 coverage quality 를 독립적으로 측정**.

### 정리 4.3 — NDCG vs MAP 의 차이

MAP 은 binary relevance (relevant/non-relevant) 만 고려:

$$\text{MAP} = \frac{1}{|R|} \sum_k P(k) \cdot \mathbb{1}[\text{rel}(k) > 0]$$

NDCG 는 graded relevance (0~5 등) 를 지원:

$$\text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}$$

따라서 "somewhat relevant" 를 구분해야 하면 NDCG 사용. 예: 영화 추천 (별점 1~5), 뉴스 relevance (highly/moderately/slightly relevant).

---

## 💻 Python / NumPy / Ranx 구현 검증

### 실험 1 — Precision, Recall, MRR 계산

```python
import numpy as np
from collections import Counter

def compute_metrics(retrieved_indices, relevant_indices, k=None):
    """
    retrieved_indices: ranking 된 document indices (상위부터)
    relevant_indices: ground truth relevant documents
    k: 평가 대상 top-k (None 이면 전체)
    
    Returns: precision@k, recall@k, MRR
    """
    if k is not None:
        retrieved = set(retrieved_indices[:k])
    else:
        retrieved = set(retrieved_indices)
    
    relevant = set(relevant_indices)
    
    # True positive
    tp = len(retrieved & relevant)
    
    # Precision@k
    precision = tp / len(retrieved) if len(retrieved) > 0 else 0.0
    
    # Recall@k
    recall = tp / len(relevant) if len(relevant) > 0 else 0.0
    
    # MRR
    mrr = 0.0
    for i, doc_idx in enumerate(retrieved_indices):
        if doc_idx in relevant:
            mrr = 1.0 / (i + 1)
            break
    
    return precision, recall, mrr

# 테스트 케이스
retrieved = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # top-10 (어떤 ranking 결과)
relevant = [0, 2, 4, 8]  # 실제 relevant documents

# 전체 평가
p, r, mrr = compute_metrics(retrieved, relevant)
print(f"전체 평가: Precision={p:.3f}, Recall={r:.3f}, MRR={mrr:.3f}")

# Top-5 평가
p5, r5, mrr5 = compute_metrics(retrieved, relevant, k=5)
print(f"Top-5 평가: Precision@5={p5:.3f}, Recall@5={r5:.3f}, MRR={mrr5:.3f}")

# Top-10 평가
p10, r10, mrr10 = compute_metrics(retrieved, relevant, k=10)
print(f"Top-10 평가: Precision@10={p10:.3f}, Recall@10={r10:.3f}, MRR={mrr10:.3f}")
```

### 실험 2 — MAP 계산

```python
def compute_map(retrieved_indices, relevant_indices, k=None):
    """
    MAP (Mean Average Precision) 계산
    """
    if k is not None:
        retrieved = retrieved_indices[:k]
    else:
        retrieved = retrieved_indices
    
    relevant = set(relevant_indices)
    
    ap = 0.0
    num_relevant = 0
    
    for i, doc_idx in enumerate(retrieved):
        if doc_idx in relevant:
            # Precision at position i+1
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            ap += precision_at_i
    
    if len(relevant) == 0:
        return 0.0
    
    return ap / len(relevant)

# 테스트
ap = compute_map(retrieved, relevant, k=10)
print(f"\nMAP@10: {ap:.4f}")

# 단계별 계산 보기
print("\n단계별 계산:")
relevant_set = set(relevant)
for i, doc_idx in enumerate(retrieved[:10]):
    is_rel = doc_idx in relevant_set
    if is_rel:
        p_at_i = (sum(1 for j in range(i+1) if retrieved[j] in relevant_set)) / (i + 1)
        print(f"  Position {i+1}: doc {doc_idx} [RELEVANT], P@{i+1} = {p_at_i:.3f}")
```

### 실험 3 — DCG & NDCG 계산

```python
def compute_dcg(relevance_scores, k=None):
    """
    DCG@k 계산
    relevance_scores: [rel(1), rel(2), ..., rel(n)] 
                      각 position 의 relevance grade (0~5 등)
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]
    
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        gain = (2 ** rel) - 1
        discount = np.log2(i + 2)  # position i+1 에 대한 discount
        dcg += gain / discount
    
    return dcg

def compute_ndcg(relevance_scores, k=None):
    """
    NDCG@k 계산
    """
    dcg = compute_dcg(relevance_scores, k)
    
    # IDCG: ideal ordering (relevance 큰 순서)
    ideal_scores = sorted(relevance_scores, reverse=True)
    if k is not None:
        ideal_scores = ideal_scores[:k]
    idcg = compute_dcg(ideal_scores)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

# 테스트: graded relevance (0~5)
relevance = [5, 2, 3, 0, 1, 4, 0, 0, 2, 0]  # Retrieval ranking 의 relevance grades

print("\nNDCG 계산:")
print(f"Retrieved relevance grades: {relevance}")

# Top-10
dcg10 = compute_dcg(relevance, k=10)
ndcg10 = compute_ndcg(relevance, k=10)
print(f"DCG@10: {dcg10:.4f}, NDCG@10: {ndcg10:.4f}")

# Top-5
dcg5 = compute_dcg(relevance, k=5)
ndcg5 = compute_ndcg(relevance, k=5)
print(f"DCG@5: {dcg5:.4f}, NDCG@5: {ndcg5:.4f}")

# Ideal DCG
ideal = sorted(relevance, reverse=True)
idcg10 = compute_dcg(ideal, k=10)
print(f"IDCG@10 (perfect ranking): {idcg10:.4f}")
```

### 실험 4 — Ranx 라이브러리로 검증

```python
# pip install ranx
# from ranx import evaluate, Qrels, Run

# Mock data: multiple queries
qrels = {
    "q1": {"d0": 1, "d2": 1, "d4": 1, "d8": 1},  # Binary relevance
    "q2": {"d1": 2, "d3": 3, "d5": 1},  # Graded relevance (0~5)
}

run = {
    "q1": {"d0": 1.0, "d1": 0.9, "d2": 0.8, "d3": 0.7, "d4": 0.6, "d5": 0.5},
    "q2": {"d1": 1.0, "d2": 0.9, "d3": 0.8, "d4": 0.7, "d5": 0.6},
}

# Ranx evaluation 시뮬레이션 (실제 코드는 라이브러리 사용)
# evaluate(qrels, run, metrics=['map', 'ndcg@10', 'mrr', 'recall@10'])
```

---

## 🔗 실전 활용

| 응용 | 주요 Metric | 보조 Metric | 선택 이유 |
|------|-----------|-----------|---------|
| Web Search | MAP, NDCG@10 | Recall@100, P@5 | Ranking quality + coverage |
| QA System | MRR, Recall@1 | MAP | 첫 정답 속도 중요 |
| E-commerce | Recall@100, P@10 | MAP | Coverage 우선, then ranking |
| Reranking | NDCG@k | MAP | Graded relevance label 자주 사용 |
| A/B Testing | Multiple (MAP + Recall) | 통계검정 | Single metric 은 위험 |

---

## ⚖️ 가정과 한계

- **Binary vs Graded**: MAP 은 binary (relevant/not), NDCG 는 graded. 어느 것으로 label 할 것인가.
- **Relevance label quality**: Ground truth relevance 의 정확성에 전적으로 의존. Annotation error 는 metric 에 직접 영향.
- **Top-k truncation**: NDCG@k 는 top-k 만 본다 — beyond-k 의 좋은 ranking 을 무시.
- **Position discount 의 임의성**: log₂ 를 쓰는 이유? 다른 base 라면? Empirically tuned 이지 theoretical justification 은 약함.
- **Metric 간 상충**: MRR 높음 (첫 정답 빠름) 이 MAP 높음 (모든 relevant 를 상단) 과 상충할 수 있음.
- **Single number 의 한계**: 한 metric 으로 system 을 fully characterize 불가능 — 항상 여러 metric 함께 봐야 함.

---

## 📌 핵심 정리

| Metric | 정의 | 언제 | 특징 |
|--------|------|------|------|
| **P@k, R@k** | Precision / Recall at top-k | Quick view | Binary, simple |
| **MRR** | $\sum 1/\text{rank}_{\text{first}}$ | QA, single answer | First correct position |
| **MAP** | Average precision over rank | Web search | Ranking + coverage (binary) |
| **NDCG@k** | $\text{DCG}/\text{IDCG}$, graded | Recommenders, nuanced | Graded relevance, normalized |

$$\boxed{\text{NDCG@k} = \frac{\sum_{i=1}^{k} (2^{\text{rel}(i)}-1) / \log_2(i+1)}{\sum_{i=1}^{k} (2^{\text{rel}_{\text{ideal}}(i)}-1) / \log_2(i+1)}}$$

> **핵심**: 단일 metric 은 위험. 항상 multiple metrics (P@k, Recall, NDCG, MAP 등) 를 함께 보고, 응용의 요구사항 (recall 우선 vs ranking quality 우선) 에 맞는 metric 을 primary 로 삼을 것.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: 10개 retrieval, 그중 positions 1, 5, 8에서 relevant. Recall@10 과 Precision@10 은?

<details>
<summary>해설</summary>

Recall@10 = 3 relevant / (실제 몇 개가 relevant 인지 알아야 정확한 값). 만약 실제로 3개가 relevant 라면: Recall@10 = 3/3 = 100%. 하지만 실제로 5개가 relevant 라면: Recall@10 = 3/5 = 60%.

Precision@10 = 3 / 10 = 30% (top-10 중 30% 만 relevant)
</details>

**문제 2 (심화)**: Relevance grades [5, 4, 3, 0, 2] (top-5). NDCG@5 는?

<details>
<summary>해설</summary>

DCG@5:
- Position 1: $(2^5 - 1) / \log_2(2) = 31 / 1 = 31$
- Position 2: $(2^4 - 1) / \log_2(3) = 15 / 1.585 \approx 9.46$
- Position 3: $(2^3 - 1) / \log_2(4) = 7 / 2 = 3.5$
- Position 4: $(2^0 - 1) / \log_2(5) = 0 / 2.322 = 0$
- Position 5: $(2^2 - 1) / \log_2(6) = 3 / 2.585 \approx 1.16$

DCG@5 ≈ 31 + 9.46 + 3.5 + 0 + 1.16 = 45.12

IDCG@5 (최적 순서: [5, 4, 3, 2, 0]):
- Position 1: 31
- Position 2: 9.46
- Position 3: 3.5
- Position 4: $(2^2-1) / 2.322 \approx 1.29$
- Position 5: 0

IDCG@5 ≈ 45.25

NDCG@5 ≈ 45.12 / 45.25 ≈ 0.997 (거의 완벽)
</details>

**문제 3 (논문 비평)**: "NDCG 를 사용하면 graded relevance 를 정확히 측정하니 항상 NDCG 를 써야 한다" 는 주장?

<details>
<summary>해설</summary>

반박: NDCG 는 편의적이지만 몇 가지 문제:

1. **Annotation cost**: Graded relevance (0~5) label 은 binary (relevant/not) 보다 expensive.
2. **Normalization bias**: IDCG 기반 정규화는 queries 간 비교를 어렵게 함 (각 query 의 IDCG 가 다름).
3. **Position discount 임의성**: log₂ 의 선택이 empirical 이지 rigorous 하지 않음.
4. **Graded label 의 inconsistency**: 여러 annotators 가 같은 relevance grade 를 주지 않을 수 있음 (inter-annotator agreement 낮음).

현실: MAP (binary) 도 함께 사용하여 robustness 확보. 또한 offline metric 과 online A/B test 모두 필요.
</details>

---

<div align="center">

[◀ 이전 (03. BM25)](./03-bm25.md) · [📚 README](../README.md) · [다음 ▶ (05. Retrieve vs Rerank)](./05-retrieve-rerank.md)

</div>
