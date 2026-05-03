# 02. TF-IDF 와 Vector Space Model

## 🎯 핵심 질문

- TF (Term Frequency) 와 IDF (Inverse Document Frequency) 를 각각 독립적으로 동기화할 수 있는가? 아니면 이들이 같은 정보 원천에서 나오는가?
- IDF = $\log(N / \text{df}(t))$ 가 정보이론의 self-information $\log(1/p(t))$ 와 수학적으로 동등한가?
- Length normalization 이 TF-IDF 에 필요한 이유는? Cosine similarity 만으로는 충분한가?
- TF-IDF 가 BM25 에 비해 왜 주로 baseline 이 아닌 피처로만 쓰이는가?

---

## 🔍 왜 TF-IDF 가 retrieval·RAG 에 중요한가

TF-IDF 는 1960년대 Salton 의 논문 이후 60년을 정보검색의 "default" 로 군림했습니다. 비록 modern RAG 는 dense embedding 을 주로 사용하지만, TF-IDF 가 왜 작동하는지 이해하는 것이 필수적입니다.

1. **정보이론적 정당성** — IDF 는 단순히 경험적 가중치가 아니라, term 의 **self-information** 을 계산하는 것. 이는 entropy 와 직결.
2. **BM25 의 기초** — BM25 (Ch1-03) 는 TF-IDF 의 상위호환이지만, TF-IDF 의 선형 가정을 비선형으로 개선한 것일 뿐. 기본 구조는 동일.
3. **Dense retrieval 의 보완** — Dense embedding (Ch2) 과 비교 시 sparse (TF-IDF) 의 강점: interpretable, zero-shot, 계산 효율. 현대 RAG 는 두 가지를 hybrid 로 사용.
4. **길이 정규화의 필연성** — 이를 통해 "긴 문서 bias" 를 제거하는 이론을 배우는 것이 Ch3 의 "late interaction" 으로 이어짐.

---

## 📐 수학적 선행 조건

- 벡터 공간: norm, inner product, cosine similarity
- 로그 성질: $\log(a/b) = \log a - \log b$
- 확률론: probability, frequency → probability
- 정보이론 기초: entropy, self-information

---

## 📖 직관적 이해

### TF-IDF 의 직관: "흔한 단어는 덜 중요"

```
Term frequency (TF):
  "machine" 이 문서에 3번 나타남 → TF = 3
  직관: 더 자주 나타날수록 관련 문서일 확률 ↑

Inverse Document Frequency (IDF):
  "machine" 은 1000개 문서 중 500개에 나타남 → df = 500 → IDF = log(1000/500) = log 2
  "deep" 은 1000개 문서 중 10개에만 나타남 → df = 10 → IDF = log(1000/10) = log 100
  직관: "deep" 이 훨씬 distinctive (정보가 많음)

TF-IDF 조합:
  TF("machine") × IDF("machine") vs TF("deep") × IDF("deep")
  → "deep" 이 더 높은 가중치를 받음 (더 유용한 정보)
```

### IDF 와 Information Theory 의 연결

```
Probability 관점:
  P(t) = df(t) / N  (term t 가 random document 에 나타날 확률)
  
Self-information (bits):
  I(t) = log₂(1/P(t)) = log₂(N/df(t))
  → "term t 가 나타나는 것이 얼마나 놀라운가?"
  
IDF (nats, 보통 natural log):
  IDF(t) = ln(N/df(t))  (정확히 I(t) 의 natural log 버전)
```

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Term Frequency (TF)

**Raw term frequency**:
$$\text{TF}(t, d) = \#\{t \text{ appears in } d\}$$

(문서 $d$ 에서 term $t$ 의 출현 횟수)

**Normalized TF** (length-bias 제거):
$$\text{TF}_{\text{norm}}(t, d) = \frac{\text{TF}(t, d)}{|d|_{\text{terms}}}$$

여기서 $|d|_{\text{terms}}$ 는 문서 $d$ 의 총 term 수.

**Log-scaled TF** (sublinear scaling):
$$\text{TF}_{\text{log}}(t, d) = 1 + \log(\text{TF}(t, d))$$

(tf 가 커도 무한정 커지지 않음 — saturation 효과)

### 정의 2.2 — Inverse Document Frequency (IDF)

**Document frequency**:
$$\text{df}(t) = |\{d \in D : t \text{ occurs in } d\}|$$

(term $t$ 가 포함된 문서의 개수)

**IDF** (standard):
$$\text{IDF}(t) = \log\left(\frac{N}{\text{df}(t)}\right)$$

여기서 $N = |D|$ 는 총 문서 수. 로그 base 는 보통 natural log ($\ln$) 또는 base 10.

**Smoothed IDF** (zero-division 방지):
$$\text{IDF}_{\text{smooth}}(t) = \log\left(\frac{N}{\text{df}(t) + 1}\right) + 1$$

### 정의 2.3 — TF-IDF 벡터와 Cosine Similarity

**TF-IDF 스코어** (term $t$, document $d$):
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**Document vector** (모든 term 에 대한 벡터):
$$\mathbf{d} = \left(\text{TF-IDF}(t_1, d), \text{TF-IDF}(t_2, d), \ldots, \text{TF-IDF}(t_m, d)\right) \in \mathbb{R}^m$$

(sparse vector — 많은 좌표가 0)

**Query vector**:
$$\mathbf{q} = \left(\text{TF-IDF}(t_1, q), \text{TF-IDF}(t_2, q), \ldots, \text{TF-IDF}(t_m, q)\right)$$

**Retrieval score** (cosine similarity, L2 normalized):
$$s(q, d) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \cdot \|\mathbf{d}\|_2} = \frac{\sum_t \text{TF-IDF}(t,q) \cdot \text{TF-IDF}(t,d)}{\sqrt{\sum_t \text{TF-IDF}(t,q)^2} \cdot \sqrt{\sum_t \text{TF-IDF}(t,d)^2}}$$

---

## 🔬 정리와 증명

### 정리 2.1 — IDF 와 Self-Information 의 동등성

$$\text{IDF}(t) = \log\left(\frac{N}{\text{df}(t)}\right) = \log\left(\frac{1}{p_t}\right)$$

여기서 $p_t = \frac{\text{df}(t)}{N}$ 는 term $t$ 의 경험적 확률 (document-level).

이는 정보이론의 self-information 정의와 일치:
$$I(t) = \log(1/P(t)) = -\log P(t)$$

**의미**: term 이 드물수록 (작은 $p_t$) IDF 가 크다 = 그 term 이 나타나는 것이 더 "놀랍고" 따라서 정보가 많다.

**증명**: Direct substitution. $p_t = \text{df}(t)/N$ 로 정의하면,
$$\text{IDF}(t) = \log(N/\text{df}(t)) = \log(1/p_t) = -\log p_t = I(t) \quad \square$$

### 정리 2.2 — Cosine Similarity 의 Length Invariance

TF-IDF 벡터에 cosine similarity 를 적용하면, 문서의 길이 (총 term 개수) 에 무관하게 의미 있는 유사도를 얻는다.

**증명**: Cosine similarity 정의에서,
$$\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|\|\mathbf{d}\|}$$

분자와 분모 모두 document length 에 영향을 받지만, **정규화로 상쇄** 된다. 즉, $\mathbf{d}$ 를 $\alpha \mathbf{d}$ 로 바꾸면 (모든 좌표를 $\alpha$배):

$$\cos(\mathbf{q}, \alpha\mathbf{d}) = \frac{\mathbf{q} \cdot (\alpha\mathbf{d})}{\|\mathbf{q}\| \cdot \|\alpha\mathbf{d}\|} = \frac{\alpha(\mathbf{q} \cdot \mathbf{d})}{\|\mathbf{q}\| \cdot |\alpha| \|\mathbf{d}\|} = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

따라서 스케일 불변. 실제로 TF-IDF 에서 긴 문서는 모든 좌표가 작아서 (term frequency 정규화) 자동으로 보상.

더 정밀하게는, 문서 길이를 $L_d = \sum_t \text{TF}(t,d)$ 라 하면, normalized TF 를 사용하면:

$$\text{TF}_{\text{norm}}(t,d) = \frac{\text{TF}(t,d)}{L_d}$$

이 경우 모든 term 가중치가 비례적으로 감소 → L2 norm 도 감소 → cosine 은 불변. $\square$

### 정리 2.3 — TF 의 Sublinearity

Raw TF 보다 log-scaled TF ($1 + \log(\text{TF})$) 를 사용하면 더 좋은 성능을 보인다.

**직관**: term 이 5번 나타나는 것이 1번 나타나는 것보다 "5배" 중요하지는 않음. 오히려 saturation 효과 있음.

**경험적 정당성**: Salton & McGill (1983) 의 실험에서 sublinear scaling 이 raw TF 보다 우월. 이는 인간의 인식도 logarithmic (Weber's law 같은) 이기 때문.

$$\text{relevance}_{\text{perceived}}(\text{freq}) \propto \log(\text{freq})$$

수학적 증명은 아니지만, 정보이론과 심리학에서 지지. $\square$

---

## 💻 Python / NumPy / Scikit-learn 구현 검증

### 실험 1 — TF-IDF 직접 구현

```python
import numpy as np
from collections import defaultdict
import math

class TFIDFRetriever:
    """TF-IDF 바닥부터 구현"""
    
    def __init__(self, documents):
        """
        documents: list of strings (각 문서)
        """
        self.documents = documents
        self.n_docs = len(documents)
        
        # Vocabulary 구축
        self.vocab = set()
        for doc in documents:
            self.vocab.update(doc.lower().split())
        self.vocab = sorted(list(self.vocab))
        self.term2idx = {t: i for i, t in enumerate(self.vocab)}
        self.n_terms = len(self.vocab)
        
        # IDF 계산
        self.idf = np.zeros(self.n_terms)
        for term_idx, term in enumerate(self.vocab):
            # Document frequency
            df = sum(1 for doc in documents if term in doc.lower().split())
            # IDF with smoothing
            self.idf[term_idx] = math.log((self.n_docs + 1) / (df + 1)) + 1
        
        # TF-IDF 벡터 사전 계산
        self.tfidf_vectors = []
        for doc in documents:
            vec = self._compute_tfidf(doc)
            self.tfidf_vectors.append(vec)
    
    def _compute_tfidf(self, text):
        """Text → TF-IDF vector (sparse)"""
        terms = text.lower().split()
        tf = np.zeros(self.n_terms)
        
        # TF 계산
        for term in terms:
            if term in self.term2idx:
                tf[self.term2idx[term]] += 1.0
        
        # Normalize by document length
        tf = tf / (len(terms) + 1e-10)
        
        # TF-IDF
        tfidf = tf * self.idf
        
        # L2 normalize
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm
        
        return tfidf
    
    def retrieve(self, query, k=None):
        """
        Query → top-k documents by cosine similarity
        Returns: [(doc_idx, score), ...]
        """
        q_vec = self._compute_tfidf(query)
        
        scores = []
        for doc_idx, doc_vec in enumerate(self.tfidf_vectors):
            # Cosine similarity
            cos_sim = np.dot(q_vec, doc_vec)
            scores.append((doc_idx, cos_sim))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if k is not None:
            scores = scores[:k]
        
        return scores

# 테스트
docs = [
    "machine learning basics",
    "deep learning neural networks",
    "machine learning algorithms",
    "natural language processing",
    "machine learning is fun",
]

retriever = TFIDFRetriever(docs)
results = retriever.retrieve("machine learning", k=3)
print("TF-IDF retrieval results:")
for doc_idx, score in results:
    print(f"  Doc {doc_idx}: '{docs[doc_idx]}' (score: {score:.4f})")
```

### 실험 2 — Scikit-learn TfidfVectorizer 검증

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Scikit-learn 사용
vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
tfidf_matrix = vectorizer.fit_transform(docs)

# Query 변환
query = "machine learning"
q_vec = vectorizer.transform([query])

# Cosine similarity
similarities = cosine_similarity(q_vec, tfidf_matrix).flatten()

# Top-k
top_k = 3
top_indices = np.argsort(-similarities)[:top_k]

print("\nScikit-learn TF-IDF:")
for idx in top_indices:
    print(f"  Doc {idx}: '{docs[idx]}' (score: {similarities[idx]:.4f})")
```

### 실험 3 — IDF 의 정보이론적 검증

```python
# IDF = log(1/p_t) 인지 검증

def compute_statistics(documents):
    """각 term 의 df, probability, IDF 계산"""
    all_terms = set()
    for doc in documents:
        all_terms.update(doc.lower().split())
    
    n_docs = len(documents)
    stats = {}
    
    for term in sorted(all_terms):
        df = sum(1 for doc in documents if term in doc.lower().split())
        p_t = df / n_docs
        idf = math.log(n_docs / df)
        self_info = -math.log(p_t)  # log(1/p_t)
        
        stats[term] = {
            'df': df,
            'p_t': p_t,
            'idf': idf,
            'self_information': self_info,
            'difference': abs(idf - self_info)
        }
    
    return stats

stats = compute_statistics(docs)
print("\nIDF vs Self-Information:")
print(f"{'Term':<15} {'DF':<4} {'p_t':<6} {'IDF':<8} {'Self-Info':<10} {'Diff':<8}")
for term, s in list(stats.items())[:5]:
    print(f"{term:<15} {s['df']:<4} {s['p_t']:<6.3f} {s['idf']:<8.3f} {s['self_information']:<10.3f} {s['difference']:<8.3f}")

# log(N/df) 와 log(1/p_t) 가 일치함을 보임
```

### 실험 4 — Length Normalization 효과

```python
# 길이가 다른 문서들에서 길이 정규화의 효과

docs_varied = [
    "machine learning",
    "machine learning machine learning",  # 2배 반복
    "machine learning " * 5,  # 5배 반복
    "deep learning",
    "machine learning is great",
]

tfidf_varied = TFIDFRetriever(docs_varied)

# Query "machine"
q = "machine"
results = tfidf_varied.retrieve(q, k=len(docs_varied))

print("\nLength normalization effect:")
print(f"{'Doc':<40} {'Length':<8} {'Score':<8}")
for doc_idx, score in results:
    doc = docs_varied[doc_idx]
    length = len(doc.split())
    # Truncate long doc for display
    doc_display = (doc[:35] + "...") if len(doc) > 35 else doc
    print(f"{doc_display:<40} {length:<8} {score:<8.4f}")

# 결과: 긴 문서도 normalized score 로 인해 상대적으로 낮지 않음
# (하지만 내용 구성이 같다면 비슷한 score)
```

---

## 🔗 실전 활용

| 시나리오 | TF 종류 | IDF 종류 | 추가 최적화 |
|---------|--------|---------|-----------|
| 일반 검색 (sparse) | Normalized | Standard log | L2 normalization |
| 매우 긴 문서 | Sublinear log(1+tf) | Smoothed IDF | aggressive normalization |
| 매우 짧은 쿼리 | Raw count | Standard | Boost query IDF |
| Dense embedding 과 hybrid | - | - | TF-IDF 를 dense 와 선형결합 |
| 실시간 검색 (Elasticsearch) | BM25 (대체) | 내장 BM25 | Aggregation 불필요 |

---

## ⚖️ 가정과 한계

- **Bag-of-words**: 단어 순서 무시. "machine learning" 과 "learning machine" 동일.
- **Term independence**: 각 term 이 독립적으로 가중치 결정 — semantic relation 없음.
- **Static collection**: IDF 는 컬렉션 기준. 문서 추가 시 IDF 변경.
- **No semantic similarity**: "car" 와 "automobile" 은 다른 term으로 취급 (dense embedding 과 대조).
- **Sparse representation**: 대규모 vocabulary 시 메모리/계산 비효율.
- **Length bias 는 부분적만 제거**: Cosine + TF-IDF normalization 으로는 매우 긴 문서의 bias 를 완전히 제거할 수 없음 (BM25 의 parameter $b$ 와 차이).

---

## 📌 핵심 정리

$$\boxed{\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\left(\frac{N}{\text{df}(t)}\right) = \text{TF}(t,d) \times I(t)}$$

| 구성요소 | 정의 | 의미 |
|---------|------|------|
| TF | term 빈도 (정규화) | "이 문서에서 얼마나 중요한가" |
| IDF | $\log(1/p_t)$ | "전체 컬렉션에서 얼마나 distinctive 한가" |
| L2 norm | $\\| \mathbf{d} \\|$ | 문서 길이 무시, 방향만 비교 |

> **핵심**: TF-IDF + cosine similarity 는 simple 하지만 강력. 60년 동안의 이유는 이론 (정보이론) 과 실무가 만나는 지점. BM25 (Ch1-03) 는 이를 개선한 확률론적 모델이지만 기본 구조는 동일.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: 1000개 문서 컬렉션. 단어 "python" 은 250개 문서에 나타남. IDF(python) 은? (자연로그 사용)

<details>
<summary>해설</summary>

$$\text{IDF}(\text{python}) = \ln(1000 / 250) = \ln(4) \approx 1.386$$

Smoothed 버전 (분모에 +1):
$$\text{IDF}_{\text{smooth}}(\text{python}) = \ln(1001 / 251) + 1 \approx 1.374 + 1 = 2.374$$

Smoothing 은 zero-division 을 피하고, 매우 흔한 단어 (df = N) 의 IDF 를 0 이 아닌 작은 값으로 만듦.
</details>

**문제 2 (심화)**: TF-IDF 벡터 두 개: $\mathbf{d}_1 = (1, 0, 0.5)$ 와 $\mathbf{d}_2 = (100, 0, 50)$. Cosine similarity 는? 왜 크기 무관인지 설명.

<details>
<summary>해설</summary>

$$\cos(\mathbf{d}_1, \mathbf{d}_2) = \frac{\mathbf{d}_1 \cdot \mathbf{d}_2}{\|\mathbf{d}_1\| \|\mathbf{d}_2\|}$$

$$\mathbf{d}_1 \cdot \mathbf{d}_2 = 1 \cdot 100 + 0 \cdot 0 + 0.5 \cdot 50 = 100 + 25 = 125$$

$$\|\mathbf{d}_1\| = \sqrt{1 + 0 + 0.25} = \sqrt{1.25}$$

$$\|\mathbf{d}_2\| = \sqrt{10000 + 0 + 2500} = \sqrt{12500} = 100\sqrt{1.25}$$

$$\cos(\mathbf{d}_1, \mathbf{d}_2) = \frac{125}{\sqrt{1.25} \cdot 100\sqrt{1.25}} = \frac{125}{100 \cdot 1.25} = \frac{125}{125} = 1.0$$

두 벡터가 **방향이 완전히 같음** (스케일 다름). Cosine similarity 는 **방향만 비교** 하므로 크기 무관. 즉, $(1, 0, 0.5)$ 와 $(100, 0, 50)$ 은 같은 방향 (모든 좌표가 100배).
</details>

**문제 3 (논문 비평)**: "IDF 가 log(N/df) 이면, 매우 드문 단어 (df=1) 의 IDF 는 log(N) 으로 엄청 크다. 이는 노이즈 데이터 (typo, spam) 까지 과중하게 가중화하지 않나?" 에 대한 반박.

<details>
<summary>해설</summary>

그렇다 — 이것이 TF-IDF 의 알려진 한계 중 하나. 몇 가지 해결책:

1. **Min DF threshold**: 너무 드문 단어는 아예 vocabulary 에서 제외. 예: "df >= 5".
2. **Smoothed IDF**: $\log((N+1)/(df+1)) + 1$ 처럼 smoothing 으로 극값 감소.
3. **BM25**: Ch1-03 에서 IDF 를 $\log((N-df+0.5)/(df+0.5))$ 로 변형 → saturation 효과.
4. **Dense retrieval**: BERT 같은 learned embedding 은 rare word 를 자동으로 semantic clustering (예: typo 와 올바른 철자를 비슷한 embedding 에 배치).

실제로 Elasticsearch, Lucene 같은 production search 는 BM25 를 사용하거나 IDF cutoff 를 둠.
</details>

---

<div align="center">

[◀ 이전 (01. IR 정식화)](./01-ir-formalization.md) · [📚 README](../README.md) · [다음 ▶ (03. BM25)](./03-bm25.md)

</div>
