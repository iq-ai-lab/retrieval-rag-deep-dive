# 05. Retrieval vs Ranking — Two-Stage Pipeline

## 🎯 핵심 질문

- "Two-stage pipeline" (retrieval + reranking) 이 "single powerful ranker" 보다 왜 나은가? 효율 때문인가, 아니면 이론적 이유가 있는가?
- Stage 1 (retrieval) 의 recall bound 가 stage 2 (ranking) 의 최고 성능을 언제나 제한하는가?
- BM25 → dense embedding → cross-encoder cascading 이 표준인 이유는?
- Trade-off 의 정확한 지점은 무엇인가? (latency vs quality)

---

## 🔍 왜 Two-Stage Pipeline 이 retrieval·RAG 의 필수인가

많은 구현자들이 "더 좋은 모델 하나" 를 찾으려고 합니다. 하지만 real-world RAG 는 거의 항상 **multi-stage pipeline** 을 사용합니다. 이는 단순 효율성이 아닌, 근본적인 이유가 있습니다.

1. **Recall bound theorem** — Stage 1 의 recall 이 100% 미만이면, stage 2 는 그 이상의 recall 을 절대 달성 불가능. 이는 수학적 필연성.
2. **Efficiency-Quality trade-off 의 정량화** — 각 stage 의 모델 복잡도, 계산량, latency 의 trade-off 를 정밀하게 분석 가능.
3. **Cascading 의 이론적 정당성** — Simple + fast (BM25) 에서 complex + slow (cross-encoder) 로 점진적 refinement 는 최적.
4. **Interpretability** — 각 stage 의 역할이 명확 (retrieve = coverage, rerank = quality). Single ranker 는 둘을 모두 해야 해 trade-off 불명확.
5. **Practical success** — BEIR (Thakur et al. 2021) 에서 best systems 는 모두 multi-stage (dense + rerank, hybrid 등).

---

## 📐 수학적 선행 조건

- 집합론: intersection, upper/lower bound
- 확률론: independence, union bound
- 복잡도 분석: time complexity, scaling
- Information Retrieval basics (Ch1-01~04)

---

## 📖 직관적 이해

### Two-Stage 의 동기: "Recall Bound"

```
Stage 1 (Fast Retriever):
  1000개 문서 중 recall@100 = 80%
  → 100개 중 80개 만 relevant 포함, 20개 는 놓침

Stage 2 (Accurate Reranker):
  Stage 1 이 준 100개 만 봄
  → 아무리 좋아도 놓친 20개 는 순서에 영향 불가능

따라서 stage 2 의 최고 recall ≤ stage 1 의 recall
```

### Cascading Strategy: Cost-Quality Frontier

```
비용 (latency) ────────────→

품질   
  │
  │      (Single powerful    (Too expensive!
  │       ranker - infeasible)
  │    
  │      ✓ Cross-encoder
  │          (expensive, best quality)
  │        ╱
  │      ╱
  │    ╱  Reranking
  │  ╱     (moderate cost)
  │╱
  ├─────┬─────┬────────────
         Retrieval   Hybrid
         (fast,      (dense+sparse,
         decent)     balanced)

선택: Query 특성과 SLA (Latency budget) 에 따라
  - High-speed search: BM25 단독
  - Balanced: BM25 + light reranker
  - Quality-critical: Dense retrieval + cross-encoder
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Two-Stage Retrieval Pipeline

**Stage 1 (Retriever)**: Fast, approximate retrieval function

$$\mathcal{R}_1(q) = \{\text{top-}k \text{ documents by } f_1(q,d)\}$$

Query $q$ 에 대해 상위 $k$ 개 document 반환. $k$ 는 보통 50~1000.

**Stage 2 (Reranker)**: Expensive, accurate ranking function

$$\mathcal{R}_2(q) = \text{rerank}(\mathcal{R}_1(q) \text{ using } f_2(q,d))$$

$\mathcal{R}_1(q)$ 내에서만 documents 를 재순서화.

**Final retrieval**:

$$\mathcal{R}_{\text{final}}(q) = \text{top-}k' \text{ documents from } \mathcal{R}_2(q)$$

일반적으로 $k' \ll k$ (e.g., $k=100, k'=10$).

**System latency**:

$$L_{\text{total}} = L_1 + L_2$$

where $L_1$ = stage 1 시간 (전체 N 개 documents 대상), $L_2$ = stage 2 시간 (k 개만 대상).

### 정의 5.2 — Recall Bound

**Stage 1 recall@k**:

$$\text{Recall}_1@k = \frac{|\{\text{relevant docs in top-}k \text{ of } \mathcal{R}_1(q)\}|}{|R(q)|}$$

**Stage 2 의 maximum achievable recall** (even with perfect ranker):

$$\text{Recall}_2^{\max}@k' = \min\left(\text{Recall}_1@k, 1\right)$$

**Proof**: $\mathcal{R}_2$ 는 $\mathcal{R}_1$ 내의 documents 만 rerank 할 수 있으므로, $\mathcal{R}_1$ 이 놓친 relevant documents 는 영원히 lost.

### 정의 5.3 — Efficiency Metrics

**Throughput**:

$$T = \frac{\text{queries processed}}{\text{time}}$$

두 stage 에서:

$$T_{\text{total}} = \frac{1}{L_1 + L_2}$$

**Recall vs Quality trade-off** (고정 latency budget 하에서):

- Stage 1 recall 을 높이려면: 더 큰 $k$ 필요 → $L_1$ 증가 → $k'$ 감소 → quality 저하
- Stage 1 recall 을 낮추면: 작은 $k$ → $L_1$ 감소 → $k'$ 증가 → quality 향상

최적점은 $k$ 선택으로 결정.

---

## 🔬 정리와 증명

### 정리 5.1 — Two-Stage Recall Bound

**Theorem**: Stage 1 의 top-$k$ retrieval recall 을 $r_1$ 이라 하자. Stage 2 는 아무리 좋은 reranker 를 사용해도 recall 은 최대 $r_1$ 이다.

$$\text{Recall}_2^{\max} \leq \text{Recall}_1$$

**Proof**: 

$R(q)$ = 전체 relevant documents 집합, $D_1(q)$ = stage 1 반환 문서 집합.

$$\text{Recall}_1 = \frac{|R(q) \cap D_1(q)|}{|R(q)|}$$

Stage 2 는 $D_1(q)$ 내에서만 재순서화하므로, stage 2 반환 집합 $D_2(q) \subseteq D_1(q)$.

따라서:

$$\text{Recall}_2 = \frac{|R(q) \cap D_2(q)|}{|R(q)|} \leq \frac{|R(q) \cap D_1(q)|}{|R(q)|} = \text{Recall}_1$$

최고의 경우는 $D_2(q) = D_1(q)$ 일 때 (모든 retrieved documents 를 유지), 즉 $\text{Recall}_2^{\max} = \text{Recall}_1$. $\square$

### 정리 5.2 — Two-Stage 의 Efficiency 이득

**Scenario**: 

- $N$ = total documents
- Stage 1 complexity: $O(N)$ (linear scan, e.g., BM25)
- Stage 2 complexity: $O(k \cdot d^2)$ (e.g., cross-encoder, $d$ = embedding dim)

**Stage 1 only** (single powerful ranker with complexity $O(N \cdot d^2)$):

$$L_{\text{single}} = O(N \cdot d^2)$$

**Two-stage**:

$$L_{\text{two-stage}} = O(N) + O(k \cdot d^2) = O(N + k \cdot d^2)$$

비교:

$$\frac{L_{\text{single}}}{L_{\text{two-stage}}} = \frac{N \cdot d^2}{N + k \cdot d^2} \approx \frac{N}{k}$$

$k \ll N$ 이면 (보통 $N = 10^7, k = 100$), two-stage 는 $10^5$ 배 빠르다!

**Trade-off**: 하지만 recall bound 때문에 quality 는 stage 1 에 의존. 따라서 stage 1 의 recall 이 충분해야 함 (보통 recall@100 > 80%).

### 정리 5.3 — Optimal $k$ 의 결정

Latency budget $T_{\max}$ 고정 시, optimal $k$ 는:

$$k^* = \arg\max_k \text{NDCG}(k)$$

subject to $L_1(k) + L_2(k) \leq T_{\max}$.

일반적으로:

$$L_1(k) \propto k \quad \text{(linear in k)}$$
$$L_2(k) \propto 1 \quad \text{(independent of k, or weakly dependent)}$$

따라서 increasing $k$ 는 stage 1 cost 만 증가. 반면 stage 2 quality 는 $k$ 증가에 (recall bound 때문에) diminishing return.

Empirical practice: $k$ 를 binary search 또는 grid search 로 결정 (latency 제약 하에서).

---

## 💻 Python / SimPy / PyTorch 구현 검증

### 실험 1 — Recall Bound 시뮬레이션

```python
import numpy as np
from collections import defaultdict

def simulate_two_stage(
    n_docs=10000,
    n_queries=100,
    n_relevant_per_query=50,
    k_retrieve=100,
    stage1_recall_rate=0.8,  # Stage 1 recall@k
):
    """
    Two-stage pipeline 시뮬레이션
    Returns: stage1_recall, stage2_max_recall
    """
    stage1_recalls = []
    stage2_max_recalls = []
    
    for q_idx in range(n_queries):
        # 각 query 에 대해 relevant documents 생성
        relevant_docs = set(np.random.choice(n_docs, n_relevant_per_query, replace=False))
        
        # Stage 1: approximate retrieval (stage1_recall_rate 의 relevant 만 retrieval)
        n_retrieved_relevant = max(1, int(n_relevant_per_query * stage1_recall_rate))
        retrieved_docs = set(np.random.choice(
            list(relevant_docs), 
            size=min(n_retrieved_relevant, len(relevant_docs)), 
            replace=False
        ))
        
        # Add some false positives (non-relevant docs)
        n_false_positives = max(0, k_retrieve - len(retrieved_docs))
        false_positives = set(np.random.choice(
            [d for d in range(n_docs) if d not in relevant_docs],
            size=n_false_positives,
            replace=False
        ))
        stage1_set = retrieved_docs | false_positives
        
        # Metrics
        stage1_recall = len(retrieved_docs) / len(relevant_docs)
        stage1_recalls.append(stage1_recall)
        
        # Stage 2: perfect reranking (stage 1 set 내에서 모든 relevant 를 상단에)
        stage2_max_recall = len(retrieved_docs) / len(relevant_docs)
        stage2_max_recalls.append(stage2_max_recall)
    
    return np.mean(stage1_recalls), np.mean(stage2_max_recalls)

# 다양한 k 값에 대해 시뮬레이션
k_values = [10, 50, 100, 200, 500]
stage1_recalls = []
stage2_recalls = []

for k in k_values:
    r1, r2 = simulate_two_stage(k_retrieve=k, stage1_recall_rate=0.7)
    stage1_recalls.append(r1)
    stage2_recalls.append(r2)
    print(f"k={k}: Stage1 recall={r1:.3f}, Stage2 max recall={r2:.3f}")

# 그래프
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(k_values, stage1_recalls, 'o-', label='Stage 1 recall', linewidth=2)
plt.plot(k_values, stage2_recalls, 's--', label='Stage 2 max recall (upper bound)', linewidth=2)
plt.xlabel('k (number of retrieved documents)')
plt.ylabel('Recall')
plt.title('Recall Bound: Stage 2 is bounded by Stage 1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 실험 2 — Latency 모델링

```python
def estimate_latency(
    n_docs=10000,
    k=100,
    stage1_per_doc_time_ns=100,  # BM25: 100ns per doc
    stage2_per_doc_time_ms=10,   # Cross-encoder: 10ms per doc
):
    """
    Two-stage 의 전체 latency 추정
    """
    # Stage 1: full scan
    stage1_time_ms = (n_docs * stage1_per_doc_time_ns) / 1e6
    
    # Stage 2: k documents 만
    stage2_time_ms = k * stage2_per_doc_time_ms / 1e3
    
    total_time_ms = stage1_time_ms + stage2_time_ms
    
    return stage1_time_ms, stage2_time_ms, total_time_ms

# 다양한 시나리오
scenarios = [
    {"n_docs": 1000, "k": 50, "name": "Small collection"},
    {"n_docs": 10000, "k": 100, "name": "Medium collection"},
    {"n_docs": 1000000, "k": 100, "name": "Large collection"},
]

print("\nLatency analysis:")
print(f"{'Scenario':<20} {'Stage1(ms)':<12} {'Stage2(ms)':<12} {'Total(ms)':<12}")
for scenario in scenarios:
    l1, l2, total = estimate_latency(
        n_docs=scenario["n_docs"],
        k=scenario["k"]
    )
    print(f"{scenario['name']:<20} {l1:<12.3f} {l2:<12.3f} {total:<12.3f}")
```

### 실험 3 — Retriever 성능 비교

```python
class SimpleRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.vocab = set()
        for doc in documents:
            self.vocab.update(doc.lower().split())
    
    def bm25_retrieve(self, query, k):
        """BM25 기반 retrieval (간단 버전)"""
        query_terms = set(query.lower().split())
        scores = []
        for doc_idx, doc in enumerate(self.documents):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            scores.append((doc_idx, overlap))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_idx for doc_idx, _ in scores[:k]]

class SimpleReranker:
    def rerank(self, query, documents, k=10):
        """Cross-encoder 기반 reranking (시뮬레이션)"""
        # 실제로는 BERT 기반 유사도 계산
        # 여기서는 query-doc 의 overlap 기반 (간단 버전)
        query_tokens = set(query.lower().split())
        scores = []
        for idx, doc in enumerate(documents):
            doc_tokens = set(doc.lower().split())
            # Similarity = overlap ratio
            sim = len(query_tokens & doc_tokens) / max(len(query_tokens | doc_tokens), 1)
            scores.append((idx, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:k]]

# 테스트 데이터
docs = [
    "machine learning basics",
    "deep learning neural networks",
    "machine learning algorithms",
    "natural language processing nlp",
    "machine learning is fun",
    "computer vision images",
    "machine deep learning",
    "neural network training",
] * 10  # 80 documents

retriever = SimpleRetriever(docs)
reranker = SimpleReranker()

query = "machine learning"

# Stage 1
retrieved_indices = retriever.bm25_retrieve(query, k=20)
retrieved_docs = [docs[i] for i in retrieved_indices]

print(f"Query: '{query}'")
print(f"\nStage 1 (BM25) - Top 5 of {len(retrieved_indices)} retrieved:")
for i, idx in enumerate(retrieved_indices[:5]):
    print(f"  {i+1}. {docs[idx]}")

# Stage 2
reranked_indices = reranker.rerank(query, retrieved_docs, k=5)
print(f"\nStage 2 (Rerank) - Top 5 of {len(retrieved_docs)} reranked:")
for i, idx in enumerate(reranked_indices[:5]):
    actual_doc_idx = retrieved_indices[idx]
    print(f"  {i+1}. {docs[actual_doc_idx]}")
```

### 실험 4 — Trade-off 분석: Latency vs Recall vs Quality

```python
def analyze_tradeoff(n_docs=100000):
    """
    Different k 값에서 latency, recall, quality trade-off 분석
    """
    results = []
    
    for k in [50, 100, 500, 1000, 5000]:
        # Latency (simplified model)
        stage1_latency = 50  # ms (constant, BM25 효율)
        stage2_latency = min(k * 0.01, 1000)  # 각 doc 0.01ms (cross-encoder)
        total_latency = stage1_latency + stage2_latency
        
        # Recall (simulated: k ↑ → recall ↑ with saturation)
        recall = min(1.0, 0.5 + 0.4 * np.log(k) / np.log(5000))
        
        # Quality (simulated: recall ↑ → quality ↑ with diminishing return)
        quality = recall ** 0.5  # Diminishing return
        
        results.append({
            'k': k,
            'latency_ms': total_latency,
            'recall': recall,
            'quality': quality,
        })
    
    return results

results = analyze_tradeoff()

print("\nTrade-off Analysis:")
print(f"{'k':<8} {'Latency(ms)':<15} {'Recall':<10} {'Quality':<10}")
for r in results:
    print(f"{r['k']:<8} {r['latency_ms']:<15.2f} {r['recall']:<10.3f} {r['quality']:<10.3f}")

# Pareto frontier: latency vs quality
print("\nPareto frontier:")
for r in results:
    if r['latency_ms'] < 100:
        status = "✓ Feasible"
    elif r['latency_ms'] < 200:
        status = "△ Acceptable"
    else:
        status = "✗ Too slow"
    print(f"k={r['k']}: latency={r['latency_ms']:.0f}ms, quality={r['quality']:.3f} [{status}]")
```

---

## 🔗 실전 활용

| 시나리오 | 추천 Pipeline | 이유 |
|---------|--------------|------|
| Web search (high QPS) | BM25 → Light reranker | Fast, high recall |
| QA systems | Dense retrieval → Cross-encoder | Semantic match → fine-grained ranking |
| E-commerce search | Hybrid (BM25 + dense) → Reranker | Coverage (BM25) + semantic (dense) |
| Document retrieval | BM25 → BM25 (different params) → Dense | Progressive refinement |
| Recommendation | Dense retrieval (candidate) → Cross-encoder | Efficiency first |
| Low-latency (P95<50ms) | BM25 단독 또는 Dense 단독 | No budget for reranking |
| High-quality (offline) | BM25 → Dense → Cross-encoder → LLM | Multiple stages |

---

## ⚖️ 가정과 한계

- **Independent stages**: Stage 1과 stage 2의 점수가 independent 하다고 가정 — 실제로는 correlation 있을 수 있음 (redundant scoring).
- **Stage 1 coverage**: Stage 1 recall 이 충분해야만 stage 2가 의미 있음. Stage 1이 너무 aggressive 하면 recovery 불가능.
- **No multi-path**: 한 번 stage 1을 통과하면 다시 돌아갈 수 없음 — query refinement 나 feedback loop 없음.
- **Static ranking**: 각 stage의 ranking function이 고정 — 예시) BM25의 IDF는 document collection 기반이므로 실시간 업데이트 시 복잡도 증가.
- **Recall bound의 엄격함**: 이론상 bound 이지만, 실제로는 stage 1과 stage 2가 "다른 관점"에서 판단할 수 있어 보상 가능 (e.g., BM25는 어휘 기반, dense는 의미 기반 → 서로 보완 가능).

---

## 📌 핵심 정리

$$\boxed{\text{Recall}_2^{\max} \leq \text{Recall}_1 \quad \Rightarrow \quad \text{Two-stage 는 필연적}}$$

| Stage | 역할 | 특징 | 복잡도 |
|-------|------|------|--------|
| **1 (Retrieval)** | Recall maximization | Fast, approximate | $O(N)$ |
| **2 (Reranking)** | Quality refinement | Accurate, expensive | $O(k \cdot \text{model size})$ |

**Latency vs Quality Trade-off**:

$$L_{\text{total}} = O(N) + O(k \cdot d^2)$$

$k$ 값 선택이 최적점 결정. 보통 empirical tuning 필요.

> **핵심**: Two-stage pipeline은 단순 효율성이 아니라, recall bound의 수학적 필연성에서 비롯됨. "Perfect single ranker" 는 존재할 수 없고, 현실의 제약 (latency, computational cost) 하에서는 two-stage (또는 multi-stage) 가 최적. 이를 이해하는 것이 Ch2-05 (DPR), Ch3-02 (ColBERT), Ch6-03 (reranking) 의 설계 동기를 설명함.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: Stage 1 recall@100 = 70%, stage 2 가 이 100개 중 top-10을 선택. Stage 2 의 최고 recall 은?

<details>
<summary>해설</summary>

Stage 2의 최고 recall = stage 1의 recall = 70%. 왜냐하면 stage 1이 놓친 30%의 relevant documents는 stage 2가 볼 수 없기 때문. 따라서 stage 2가 top-10 을 아무리 완벽하게 선택해도 전체 recall 은 최대 70% 에 머무름 (100개 중 70개만 relevant 이므로).
</details>

**문제 2 (심화)**: Stage 1 complexity $O(N)$, stage 2 complexity $O(k^2)$ 일 때, $N = 10^6, k = 100$ 으로 설정. (a) 두 stage 의 상대 비용, (b) optimal k 는?

<details>
<summary>해설</summary>

(a) Stage 1: $O(10^6) = 10^6$. Stage 2: $O(100^2) = 10^4$. 비율 = $10^6 / 10^4 = 100$ (stage 1이 100배 비쌈).

따라서 stage 2 의 복잡도는 무시할 수 있고, latency 는 stage 1 이 dominant. k 를 증가시켜도 latency 는 거의 변하지 않음.

(b) Optimal k 는 stage 2 계산이 stage 1 과 비슷해지는 지점: $k^2 \approx N$ → $k \approx \sqrt{N} = \sqrt{10^6} = 1000$. 하지만 stage 1 complexity 가 $O(N)$ 이면 이론상 이점이 크지 않으니, 실무에서는 recall-latency trade-off 로 결정. 보통 $k = 100 \sim 500$ 이 관례.
</details>

**문제 3 (논문 비평)**: "Single powerful cross-encoder 가 BM25 + cross-encoder dual stage 보다 훨씬 좋으니 굳이 two-stage 를 쓸 이유가 없다" 는 주장?

<details>
<summary>해설</summary>

반박: 

1. **Recall bound theorem**: Single cross-encoder 는 아무리 좋아도 latency 때문에 N 개 전체 document 를 볼 수 없음 (실시간 처리 불가). 따라서 recall bottleneck.

2. **Computational cost**: Cross-encoder 를 N 개 전체에 apply 하면 latency > 1초 (practical 아님). 반면 BM25 (O(N), 매우 빠름) → top-100 만 cross-encoder 적용하면 수십 ms 내 완료.

3. **Empirical evidence**: BEIR benchmark (Thakur et al. 2021) 에서 best systems 은 모두 multi-stage (BM25/dense retrieval + reranker). Single cross-encoder 는 recall 부족으로 top ranking 달성 못함.

4. **Zero-shot capability**: Cross-encoder 는 fine-tuning 필요. 새로운 도메인에서는 BM25 가 더 robust (zero-shot).

결론: "좋은 모델 하나" 가 "적절한 pipeline" 을 이기지 못함. 이는 information retrieval 의 본질적 한계, 모델 성능의 문제가 아님.
</details>

---

<div align="center">

[◀ 이전 (04. 평가 Metric)](./04-eval-metrics.md) · [📚 README](../README.md) · [다음 ▶ (Ch2-01. BM25 의 한계)](../ch2-dense-retrieval/01-bm25-limits.md)

</div>
