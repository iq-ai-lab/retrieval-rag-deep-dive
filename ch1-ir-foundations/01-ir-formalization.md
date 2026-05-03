# 01. Information Retrieval 의 정식화

## 🎯 핵심 질문

- Information Retrieval 에서 "query", "collection", "relevant" 의 정확한 수학적 정의는 무엇인가?
- Boolean model · Vector Space Model · Probabilistic Retrieval 의 3가지 패러다임이 어떻게 같은 문제를 다르게 해결하는가?
- Recall 과 Precision 의 trade-off 가 정확히 어떤 최적화 문제에서 발생하는가?
- "retrieval effectiveness" 를 정량화하는 근본적인 방법은 무엇인가?

---

## 🔍 왜 정식화가 retrieval·RAG 에 중요한가

RAG (Retrieval-Augmented Generation) 시스템은 retrieval 단계의 성능에 전적으로 의존합니다. 그런데 많은 구현자들이 "더 큰 embedding 모델", "더 많은 데이터" 같은 경험적 해결책만 시도합니다. 이는 문제 정의 자체를 놓친 것입니다.

1. **문제의 엄밀한 정의 없이는 해결 불가능** — query와 document를 수학적 공간에 표현해야만 retrieval 의 근본 한계와 trade-off 를 이해 가능.
2. **3가지 패러다임의 본질 차이를 아는 것은 기법 선택의 근거** — Boolean (정확 매칭) vs VSM (유사도 점수) vs Probabilistic (확률 모델) 은 각각 다른 가정과 용도를 가짐.
3. **recall vs precision 의 이론적 필연성** — 이 trade-off 는 정보 이론의 기본에서 나오며, 이를 이해해야 two-stage pipeline (Ch1-05) 의 정당성이 보임.
4. **RAG 의 상한선 결정** — dense retrieval 은 아무리 좋아도 stage 1 recall 에 bound. 이 bound 를 정식화해야 시스템 설계 가능.

---

## 📐 수학적 선행 조건

- 선형대수: vector space, inner product, norm
- 확률론 기초: probability, conditional probability, independence
- 집합론: intersection, union, cardinality
- (선택) 정보이론: entropy, self-information

---

## 📖 직관적 이해

### Query-Collection-Relevance 의 3차 관계

```
Query q = "machine learning basics"
     ↓ (matching problem)
Document Collection D = {d₁, d₂, ..., dₙ}
     ↓ (relevance judgment)
Relevant Set R(q) = {d₂, d₅, d₁₂, ...}
     
각 패러다임이 다르게 R(q) 를 추정:
- Boolean: "machine" AND "learning" 에서 matched ∩ 의 크기?
- VSM: cos(q, dᵢ) > θ 인 dᵢ?
- Prob: P(relevant | q, d) > 0.5 인 d?
```

### Precision-Recall Trade-off 의 직관

```
Threshold 를 높이면 (엄격):
  Precision ↑  (retrieved 된 것이 모두 relevant)
  Recall ↓     (놓친 relevant document 증가)

Threshold 를 낮추면 (관대):
  Precision ↓  (false positive 증가)
  Recall ↑     (더 많은 relevant 포함)

최적점은 응용에 따라:
- Search engine: recall 우선 (relevant 놓치는 게 더 손해)
- 의료 진단: precision 우선 (false positive 의 risk 큼)
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Information Retrieval 의 기본 요소

**Query space** $\mathcal{Q}$: 사용자 정보 요구를 표현하는 공간. 일반적으로 문자열 또는 벡터.

$$q \in \mathcal{Q}$$

**Document collection** $D$: $|D| = N$ 개의 문서 집합.

$$D = \{d_1, d_2, \ldots, d_N\}$$

**Relevance relation** $\text{Rel} \subseteq \mathcal{Q} \times D$: 진정한 relevance (ground truth). 쿼리 $q$ 에 대해 relevant 문서 집합:

$$R(q) = \{d \in D : (q, d) \in \text{Rel}\}$$

**Retrieval function** $f: \mathcal{Q} \times D \to \mathbb{R}$: 각 $(q, d)$ 쌍에 점수 할당. 이를 이용해 추정된 relevant 집합:

$$\hat{R}(q, k) = \{d \in D : \text{rank}_f(d|q) \leq k\}$$

(상위 $k$ 개 문서, 내림차순 정렬)

### 정의 1.2 — Boolean Model

Query 를 Boolean 식으로 표현. 각 문서를 term 의 presence/absence 벡터로 인코드.

**Query**: $q = t_1 \text{ AND } (t_2 \text{ OR } \neg t_3)$ (Boolean 조합)

**Document vector**: $d_i = (x_1, x_2, \ldots, x_m) \in \{0, 1\}^m$ (term occurrence)

**Retrieval function**:

$$f_{\text{Bool}}(q, d_i) = \begin{cases} 1 & \text{if } d_i \text{ satisfies } q \\ 0 & \text{otherwise} \end{cases}$$

**특징**: deterministic matching, no ranking, exact retrieval only.

### 정의 1.3 — Vector Space Model (VSM)

Query 와 document 를 $m$-차원 벡터로 표현 (term frequency 기반).

**Representation**:

$$q = (q_1, q_2, \ldots, q_m), \quad d_i = (d_{i,1}, d_{i,2}, \ldots, d_{i,m}) \in \mathbb{R}^m$$

여기서 $d_{i,j}$ 는 term $j$ 의 weight (e.g., term frequency 또는 TF-IDF).

**Retrieval function** (cosine similarity):

$$f_{\text{VSM}}(q, d_i) = \frac{q \cdot d_i}{\|q\| \cdot \|d_i\|} = \frac{\sum_{j=1}^m q_j d_{i,j}}{\sqrt{\sum_j q_j^2} \cdot \sqrt{\sum_j d_{i,j}^2}}$$

$\in [0, 1]$ (normalized dot product). **특징**: ranking 가능, soft relevance, dense representation.

### 정의 1.4 — Probabilistic Retrieval Model

문서와 쿼리 사이의 relevance 를 확률 변수로 모델링.

**Relevance variable** $R \in \{0, 1\}$: $R=1$ 이면 relevant, $R=0$ 이면 non-relevant.

**Retrieval function**:

$$f_{\text{Prob}}(q, d_i) = P(R=1 | q, d_i)$$

Bayes rule 로:

$$P(R=1|q,d) = \frac{P(q|R=1, d) P(d|R=1) P(R=1)}{P(q)}$$

Simplifying (independence assumption):

$$f(q, d) \propto \frac{P(q|R=1, d)}{P(q|R=0, d)}$$

**특징**: probabilistic interpretation, modular assumptions, principled ranking.

---

## 🔬 정리와 증명

### 정리 1.1 — Recall-Precision Trade-off 의 필연성

Retrieval threshold $\theta$ 를 정하면:

$$\text{Precision}(\theta) = \frac{|\hat{R}(\theta) \cap R|}{|\hat{R}(\theta)|}$$

$$\text{Recall}(\theta) = \frac{|\hat{R}(\theta) \cap R|}{|R|}$$

$\theta$ 를 상향 조정 (더 엄격) 하면 $|\hat{R}(\theta)|$ 감소 → Precision 증가, Recall 감소 (단조성).

**증명**: $|\hat{R}(\theta) \cap R| \leq \min(|\hat{R}(\theta)|, |R|)$. $\theta$ 증가 시 $|\hat{R}(\theta)|$ 감소이므로 true positives 는 같거나 감소. 따라서:

$$\text{Precision}(\theta) = \frac{|\hat{R}(\theta) \cap R|}{|\hat{R}(\theta)|}$$

분모가 감소하면 분자가 같거나 감소하므로 정밀도는 일반적으로 증가. 한편:

$$\text{Recall}(\theta) = \frac{|\hat{R}(\theta) \cap R|}{|R|}$$

분모 $|R|$ 는 고정, 분자는 감소 → recall 감소. $\square$

### 정리 1.2 — VSM 의 거리 보존성

Query 와 document 를 같은 벡터 공간에 embedding 했을 때, cosine similarity 는 각도에만 의존:

$$f_{\text{VSM}}(q, d_i) = \cos(\angle(q, d_i)) = \cos\theta$$

**의미**: 벡터의 크기 (document length) 는 관계없고 방향만 중요. 이는 length normalization 이 자동 수행됨을 의미.

**증명**: cosine similarity 정의에서 norm 으로 정규화하므로, 같은 방향이면 크기 무관. 이는 term frequency 정규화 효과를 만드는 핵심. (Ch1-02 에서 자세히) $\square$

### 정리 1.3 — Probabilistic Ranking Principle (PRP)

Retrieval function 으로 $P(R=1|q,d)$ 를 사용하면, 이 확률로 내림차순 정렬한 것이 최적 ranking.

**정의 (최적성)**: Expected number of relevant documents 를 최대화하려면, 각 $d_i$ 의 inclusion decision 을 그 relevance probability 로 하면 됨.

**증명**: Top-$k$ 를 정하면 기댓값:

$$E[\text{relevant}|k] = \sum_{d \in \hat{R}(k)} P(R=1|d,q)$$

이를 최대화하려면 $P(R=1|d,q)$ 로 정렬 후 상위 $k$ 개 선택. $\square$

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — Boolean Model 구현 및 검증

```python
import numpy as np
from collections import Counter

class BooleanRetriever:
    """Boolean Information Retrieval 바닥부터"""
    
    def __init__(self, docs):
        """docs: list of strings"""
        self.docs = docs
        self.vocab = set()
        for doc in docs:
            self.vocab.update(doc.lower().split())
        self.vocab = sorted(self.vocab)
        self.term2idx = {t: i for i, t in enumerate(self.vocab)}
        
        # Binary document vectors
        self.doc_vectors = []
        for doc in docs:
            terms = set(doc.lower().split())
            vec = np.array([1.0 if t in terms else 0.0 
                           for t in self.vocab])
            self.doc_vectors.append(vec)
    
    def retrieve(self, query_string):
        """AND/OR 로 간단한 Boolean query 처리"""
        # 간단 버전: 모든 term 이 문서에 있어야 함
        query_terms = set(query_string.lower().split())
        matches = []
        for i, doc_vec in enumerate(self.doc_vectors):
            query_indices = [self.term2idx[t] 
                            for t in query_terms 
                            if t in self.term2idx]
            if len(query_indices) == 0:
                continue
            # 모든 query term 이 doc 에 있는가?
            all_match = all(doc_vec[idx] == 1.0 for idx in query_indices)
            if all_match:
                matches.append(i)
        return matches

# 테스트
docs = [
    "machine learning basics",
    "deep learning neural networks",
    "machine learning algorithms",
    "natural language processing",
]
retriever = BooleanRetriever(docs)
result = retriever.retrieve("machine learning")
print(f"Boolean retrieval for 'machine learning': {result}")
# 예상: [0, 2] (indices of docs containing both terms)
```

### 실험 2 — Vector Space Model (TF + Cosine Similarity)

```python
class VSMRetriever:
    """Vector Space Model — TF + Cosine Similarity"""
    
    def __init__(self, docs):
        self.docs = docs
        self.vocab = set()
        for doc in docs:
            self.vocab.update(doc.lower().split())
        self.vocab = sorted(self.vocab)
        self.term2idx = {t: i for i, t in enumerate(self.vocab)}
        
        # TF vectors (raw term frequency)
        self.doc_vectors = []
        for doc in docs:
            terms = doc.lower().split()
            tf = np.zeros(len(self.vocab))
            for term in terms:
                if term in self.term2idx:
                    tf[self.term2idx[term]] += 1.0
            # L2 normalize
            tf = tf / (np.linalg.norm(tf) + 1e-10)
            self.doc_vectors.append(tf)
    
    def retrieve(self, query_string, k=None):
        """
        Query 에 대해 cosine similarity 로 ranking
        Returns: (doc_indices, scores)
        """
        query_terms = query_string.lower().split()
        q_vec = np.zeros(len(self.vocab))
        for term in query_terms:
            if term in self.term2idx:
                q_vec[self.term2idx[term]] += 1.0
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        
        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            cos_sim = np.dot(q_vec, doc_vec)  # cosine similarity
            scores.append((i, cos_sim))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        if k is not None:
            scores = scores[:k]
        
        return scores

vsm = VSMRetriever(docs)
results = vsm.retrieve("machine learning", k=2)
print(f"VSM retrieval: {results}")
# 예상: [(0, 0.408), (2, 0.408), ...] (상위 2 개)
```

### 실험 3 — Precision & Recall 계산

```python
def compute_metrics(retrieved_indices, relevant_indices, k=None):
    """
    retrieved_indices: retrieval function 이 반환한 document indices
    relevant_indices: ground truth relevant documents
    k: top-k 기준 (None 이면 전체)
    
    Returns: (precision, recall)
    """
    if k is not None:
        retrieved = set(retrieved_indices[:k])
    else:
        retrieved = set(retrieved_indices)
    
    relevant = set(relevant_indices)
    
    # True positive
    tp = len(retrieved & relevant)
    
    # Precision: TP / (TP + FP) = TP / |retrieved|
    precision = tp / len(retrieved) if len(retrieved) > 0 else 0.0
    
    # Recall: TP / (TP + FN) = TP / |relevant|
    recall = tp / len(relevant) if len(relevant) > 0 else 0.0
    
    return precision, recall

# 예시: retrieval 이 [0, 1, 2, 3] 을 반환했고,
# 실제로는 [0, 2] 가 relevant
retrieved = [0, 1, 2, 3]
relevant = [0, 2]

p, r = compute_metrics(retrieved, relevant, k=2)
print(f"Precision@2: {p:.2f}, Recall@2: {r:.2f}")
# Precision@2 = 1/2 = 0.50 (상위 2 개 중 1 개가 relevant)
# Recall@2 = 1/2 = 0.50 (relevant 2 개 중 1 개를 찾음)

p, r = compute_metrics(retrieved, relevant, k=4)
print(f"Precision@4: {p:.2f}, Recall@4: {r:.2f}")
# Precision@4 = 2/4 = 0.50
# Recall@4 = 2/2 = 1.00 (모든 relevant 를 찾음)
```

### 실험 4 — Probabilistic Ranking (P(R=1|q,d) 시뮬레이션)

```python
class ProbabilisticRetriever:
    """
    Probabilistic IR: 각 document 의 relevance probability 를 학습
    (실제로는 복잡한 모델이지만, 여기서는 간단한 예)
    """
    
    def __init__(self, docs, ground_truth_rel):
        """
        docs: list of strings
        ground_truth_rel: (query_id, doc_id, is_relevant) 의 list
        """
        self.docs = docs
        self.vocab = set()
        for doc in docs:
            self.vocab.update(doc.lower().split())
        self.vocab = sorted(self.vocab)
        
        # 간단한 확률 추정: term presence 에 따른 relevance probability
        # P(term_j present | relevant) vs P(term_j present | not relevant)
        self.p_term_given_rel = {}
        self.p_term_given_nonrel = {}
        
        for term in self.vocab:
            rel_count = 0
            rel_with_term = 0
            nonrel_count = 0
            nonrel_with_term = 0
            
            for q_id, doc_id, is_rel in ground_truth_rel:
                has_term = term in self.docs[doc_id].lower()
                if is_rel:
                    rel_count += 1
                    if has_term:
                        rel_with_term += 1
                else:
                    nonrel_count += 1
                    if has_term:
                        nonrel_with_term += 1
            
            # Laplace smoothing
            self.p_term_given_rel[term] = (rel_with_term + 1) / (rel_count + 2)
            self.p_term_given_nonrel[term] = (nonrel_with_term + 1) / (nonrel_count + 2)
    
    def compute_relevance_prob(self, query_string, doc_id):
        """
        P(R=1 | q, d) 를 term-based 모델로 근사
        Naive Bayes: P(R=1) ∝ ∏_j P(term_j | R=1)
        """
        query_terms = set(query_string.lower().split())
        
        # P(relevant) 와 P(non-relevant) 동등하다고 가정 (prior 무시)
        prob_rel = 1.0
        prob_nonrel = 1.0
        
        doc_text = self.docs[doc_id].lower()
        doc_terms = set(doc_text.split())
        
        for term in self.vocab:
            has_term = term in doc_terms
            if term in query_terms:  # query term 만 고려
                if has_term:
                    prob_rel *= self.p_term_given_rel[term]
                    prob_nonrel *= self.p_term_given_nonrel[term]
                else:
                    prob_rel *= (1 - self.p_term_given_rel[term])
                    prob_nonrel *= (1 - self.p_term_given_nonrel[term])
        
        # Posterior: P(R=1 | d, q) = P(d|R=1) / (P(d|R=1) + P(d|R=0))
        if prob_rel + prob_nonrel == 0:
            return 0.5
        return prob_rel / (prob_rel + prob_nonrel)
    
    def retrieve(self, query_string, k=None):
        """Relevance probability 로 ranking"""
        scores = []
        for doc_id in range(len(self.docs)):
            prob = self.compute_relevance_prob(query_string, doc_id)
            scores.append((doc_id, prob))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        if k is not None:
            scores = scores[:k]
        return scores

# 간단한 ground truth 로 학습
ground_truth = [
    (0, 0, 1),  # query 0, doc 0, relevant
    (0, 1, 0),  # query 0, doc 1, non-relevant
    (0, 2, 1),  # query 0, doc 2, relevant
    (0, 3, 0),  # query 0, doc 3, non-relevant
]

prob_retriever = ProbabilisticRetriever(docs, ground_truth)
results = prob_retriever.retrieve("machine learning", k=2)
print(f"Probabilistic retrieval: {results}")
# 예상: [(0 or 2, ~0.7), (다른 doc, ~0.3)]
```

---

## 🔗 실전 활용

| 시나리오 | 추천 모델 | 이유 |
|---------|---------|------|
| 특정 키워드 정확 매칭 필요 (특허검색) | Boolean | Precise control, no false positives |
| 일반 검색 (구글, 스택오버플로우) | VSM + TF-IDF | Balance of ranking + speed |
| 대규모 데이터 (relevance feedback 데이터 충분) | Probabilistic | Principled ranking with learning |
| Dense embedding (BERT 등) | VSM (cosine) | Leverages learned semantics |
| Two-stage pipeline (recall → rerank) | Hybrid | Boolean/VSM for recall, then learn-to-rank |

---

## ⚖️ 가정과 한계

- **Boolean**: "relevant" 를 binary (0/1) 로 가정. 실제로는 graded relevance.
- **VSM**: Term independence 가정 — 실제로 terms 는 상관관계 있음.
- **Probabilistic**: 단순 가정 (naive Bayes) 으로 실제 복잡성 놓칠 수 있음.
- **공통**: Query interpretation 이 사용자 의도를 완벽 반영한다고 가정 (query ambiguity 무시).
- **Collection size**: 모든 모델이 정적 collection 가정 — 동적 문서 추가 시 인덱싱 재구성 필요.

---

## 📌 핵심 정리

$$\boxed{\text{Retrieval} = \underbrace{\text{Representation}}_{\text{Query, Doc}} + \underbrace{\text{Matching}}_{\text{Scoring function}} + \underbrace{\text{Ranking}}_{\text{Top-k}}}$$

| 패러다임 | 표현 | 매칭 함수 | 특징 |
|--------|------|---------|------|
| Boolean | {0,1} vector | Exact matching | Precise, no ranking |
| VSM | TF vector | Cosine similarity | Soft relevance, fast |
| Probabilistic | Term occurrence | P(R\|q,d) | Principled, learnable |

> **핵심**: Information Retrieval 을 정식화하는 순간, recall-precision trade-off 의 필연성과 각 패러다임의 강약점이 명확해짐. 이것이 Ch1-02~05 의 구체적 기법들 (TF-IDF, BM25, metric) 을 정당화하는 기초.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: 100개 문서 컬렉션. 쿼리 "python programming" 에 대해 Boolean retriever 가 5개, VSM retriever 가 20개 를 반환. 이 중 실제 relevant 는 8개. 각각의 precision, recall 은?

<details>
<summary>해설</summary>

공통으로 relevant 인 문서 수를 모르므로 최악/최선 시나리오로 계산.

Boolean (5개 반환):
- 최선: all 5 are relevant → P = 5/5 = 100%, R = 5/8 = 62.5%
- 최악: 1개만 relevant → P = 1/5 = 20%, R = 1/8 = 12.5%

VSM (20개 반환):
- 최선: all 8 relevant in top 20 → P = 8/20 = 40%, R = 8/8 = 100%
- 최악: 1개만 relevant → P = 1/20 = 5%, R = 1/8 = 12.5%

실제로는 각 모델의 ranking order 를 알아야 정확한 계산 가능. 일반적으로 VSM 은 높은 recall 경향 (더 많이 반환), Boolean 은 높은 precision (정확한 matching).
</details>

**문제 2 (심화)**: Cosine similarity 로 계산한 VSM 에서, 두 문서 $d_1 = (1, 0, 1, 0)$ 와 $d_2 = (1, 0, 1, 0)$ 의 similarity 는 1.0. 그런데 $d_3 = (100, 0, 100, 0)$ 도 같은 similarity 1.0. 이것이 VSM 에서 "length normalization" 이라 불리는 이유는?

<details>
<summary>해설</summary>

Cosine similarity 정의:

$$\cos(d_1, d_3) = \frac{d_1 \cdot d_3}{\|d_1\| \|d_3\|} = \frac{1 \cdot 100 + 0 + 1 \cdot 100 + 0}{\sqrt{2} \cdot \sqrt{20000}}$$

$$= \frac{200}{\sqrt{2} \cdot 100\sqrt{2}} = \frac{200}{200} = 1.0$$

따라서 $d_1$ 과 $d_3$ 은 벡터 크기 (document length) 무관하게 같은 similarity. 이는 norm 으로 정규화하기 때문. 즉, cosine similarity 는 **자동으로 document length bias 를 제거** — 긴 문서가 항상 높은 score 를 받지 않음. 이를 "length normalization" 의 기하학적 설명.
</details>

**문제 3 (논문 비평)**: "VSM 은 term independence 를 가정하지만 실제로 많은 terms 는 상관관계가 있다. 따라서 VSM 은 부정확하다" 는 주장에 대해 반박/지지?

<details>
<summary>해설</summary>

반박: VSM 의 각 차원 (term) 이 독립적 축으로 작동한다는 것일 뿐, terms 간 correlation 은 벡터 공간에서 자동 포착. 예를 들어 "machine" 과 "learning" 이 자주 함께 나타나면, 같은 documents 에서 두 좌표가 높아짐 → cosine similarity 계산 시 이미 반영. Term independence 가정은 확률 모델 (Naive Bayes) 의 것이지, 벡터 내적의 것이 아님.

지지 (부분): 하지만 정말 중요한 semantic relation (e.g., "car" 와 "automobile") 은 다른 차원으로 취급. 이는 VSM 의 한계 → Dense embedding (BERT 등) 이 semantic similarity 를 더 잘 포착하는 이유. 즉, 선형 VSM 은 synthetic correlation 은 못하고, 학습된 embedding 은 가능.
</details>

---

<div align="center">

[◀ 이전 (README)](../README.md) · [📚 README](../README.md) · [다음 ▶ (02. TF-IDF 와 Vector Space Model)](./02-tf-idf.md)

</div>
