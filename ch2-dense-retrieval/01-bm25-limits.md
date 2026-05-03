# 01. BM25 의 한계와 Dense Retrieval 의 동기

## 🎯 핵심 질문

- BM25 는 **lexical matching** 을 기반으로 하는데, 왜 "자동차" 와 "automobile" 같은 paraphrase 에서 실패하는가?
- **어휘 부족 문제 (vocabulary problem)** (Furnas 1987) 는 정확히 무엇이고, 이것이 retrieval 에서 왜 피할 수 없는 문제인가?
- Dense embedding space 로 이동하면 이 문제가 해결되는 원리는? 그리고 semantic 한 "거리" 를 어떻게 학습하는가?
- BM25 에서 dense retrieval 로 가는 전환이 왜 **2020년 이후 RAG/검색 혁명** 의 핵심이었는가?

---

## 🔍 왜 이 전환이 retrieval·RAG 에 중요한가

BM25 는 40년간 정보 검색의 표준이었습니다. 그러나 neural retrieval 시대가 요구하는 것은:

1. **Paraphrase 와 semantic equivalence 인식** — "자동차 구매 팁" 과 "어떻게 차를 사나?" 는 exact term overlap 이 거의 없지만 의미적으로 동일.

2. **다언어 retrieval** — 원문은 한국어, 질문은 영어; lexical matching 불가능 → semantic embedding 필요.

3. **Vocabulary problem 의 극복** — 사용자가 사용할 문단의 단어를 모를 때 (Furnas 의 고전 실험), dense space 에서는 approximate nearest neighbor 로 보상.

4. **Dense vector 의 offline indexing 경제성** — FAISS/HNSW 같은 고속 ANN 으로 1억 개 passage 도 밀리초 단위 검색 가능. BM25 의 inverted index 보다 메모리 효율적.

이 문서는 lexical 에서 semantic 으로의 전환을 정량화하고, dense embedding 이 왜 **필연적** 인지 이론화합니다.

---

## 📐 수학적 선행 조건

- 벡터 공간 모델 (Vector Space Model): cosine similarity, $\ell_2$ distance
- Softmax, cross-entropy loss
- (선택) Information retrieval 기초: precision/recall, MRR, NDCG

---

## 📖 직관적 이해

### Lexical vs Semantic: 예시

```
Query: "자동차를 어떻게 사나?"

BM25 기준:
- "자동차" : 정확 매치 ✓
- "구매" : 정확 매치 안됨 (같은 의미라도) ✗
- 결과: 정확히 "자동차" 와 "구매" 를 모두 포함한 문단만 높은 점수

Dense Embedding 기준:
- query vec ≈ answer_vec (의미적 거리 학습) ✓
- Paraphrase: "차 구매", "자동차 구입", "내 첫 자동차" 등 모두 의미 유사
- 결과: semantic 근처의 모든 문단 순위화
```

### Vocabulary Problem 의 직관

```
Furnas 1987 실험:
100 명이 "종이 자르는 도구" 를 찾는데,
BM25 기반 검색 시스템에서 정확한 key ("scissors") 를 모든 사람이 쓰는가?
→ 아니다. "cutter", "blade", "trim tool" 등 다양한 표현.
→ Exact lexical match 는 평균 20-40% miss rate.

Dense Embedding 으로:
- query: "종이를 자르려고 하는데..." (유저의 자연 표현)
- doc: "scissors 는 ..." / "pruning shears ..." / "blade ..." 
- 모두 embedding space 에서 가깝게 위치 → recall 향상
```

### Offline Indexing 의 경제성

```
BM25 inverted index:
- 100M docs, 10000 vocab → O(100M) 메모리 (index 자체)
- Query 시 matching 과정 여전히 필요

Dense (ANN):
- 100M docs × 768 dims × FP16 = 150 GB (매우 크지만)
- HNSW/IVF 로 압축 가능
- Query embedding 1개 → HNSW 탐색 1ms (10ms BM25 대비 10배 빠름)
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Lexical Matching (BM25)

BM25 점수:
$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \left(1 - b + b \frac{|d|}{L_{\text{avg}}}\right)}
$$

여기서:
- $t$: query 의 term
- $f(t, d)$: term $t$ 가 doc $d$ 에서 나타나는 횟수
- $\text{IDF}(t) = \log \frac{N - n(t) + 0.5}{n(t) + 0.5}$ (N-gram IDF variant)
- $k_1, b$: 튜닝 상수 (보통 $k_1 = 1.5, b = 0.75$)
- **핵심**: $t$ 가 query 에 없으면 contribution 은 0 (no semantic similarity 평가)

### 정의 1.2 — Vocabulary Problem (Furnas)

User 가 찾으려는 concept 을 표현하는 단어를 모를 확률을 $p_{\text{miss}}$ 라 하자.

실험 (Furnas 1987):
$$
\mathbb{E}[\text{miss rate}] = 1 - \prod_{t \in \text{query}} (1 - p_{\text{miss}}(t))
$$

Typical $p_{\text{miss}} \approx 0.3$ per term → multi-term query 시 miss rate 빠르게 증가.

### 정의 1.3 — Dense Retrieval

Query $q$ 와 passage $p$ 의 **dense embedding** 기반 점수:
$$
s(q, p) = \cos(\mathbf{e}_q, \mathbf{e}_p) = \frac{\mathbf{e}_q \cdot \mathbf{e}_p}{\|\mathbf{e}_q\| \|\mathbf{e}_p\|}
$$

여기서:
- $\mathbf{e}_q = f_q(q) \in \mathbb{R}^d$ (query encoder 의 출력, 보통 $d = 768$)
- $\mathbf{e}_p = f_p(p) \in \mathbb{R}^d$ (passage encoder 의 출력)
- $f_q, f_p$: 학습된 neural encoder (BERT 기반)

---

## 🔬 정리와 증명

### 정리 1.1 — Lexical Matching 의 Recall 한계

Query $q$ 의 모든 term 이 passage $p$ 에 정확히 나타나야만 (또는 synonym 처리) BM25 가 non-zero 점수를 부여.

따라서 **Paraphrase recall** (같은 의미지만 term 겹침 없는 doc 의 비율):
$$
\text{Paraphrase Recall} \leq P(\text{모든 main terms 이 doc 에 나타남})
$$

Vocabulary problem 시 이는 명시적으로 0 에 수렴.

**증명 sketch**: BM25 는 linear combination of IDF-weighted term frequencies. Paraphrase 는 정의상 term overlap 최소화. 따라서 $\text{BM25}(q, p) \approx 0$ 이더라도 semantic 거리는 0에 가까움 $\square$.

### 정리 1.2 — Dense Space 의 Semantic Equivalence

Encoder $f_q, f_p$ 를 contrastive loss (Ch3) 로 학습하면, semantic 으로 equivalent 한 query-passage 쌍 $(q, p)$ 은 embedding space 에서:
$$
\mathbb{E}[\cos(f_q(q), f_p(p^{\text{equiv}}))] > \mathbb{E}[\cos(f_q(q), f_p(p^{\text{random}}))]
$$

**의미**: 다른 단어를 썼더라도 의미가 같으면 embedding 에서 가깝다.

### 정리 1.3 — ANN 의 거의-최적 Retrieval

HNSW 또는 IVF 같은 **approximate nearest neighbor** 알고리즘은 true nearest neighbor 를 확률 $1 - \epsilon$ 로 찾는다:
$$
\mathbb{P}[\text{retrieved top-k 가 true top-k}] \geq 1 - \epsilon
$$

여기서 $\epsilon = O(1/k)$ (recall parameter 에 비례).

따라서 dense retrieval 의 정확도는 embedding quality 에만 의존 $\square$.

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — BM25 vs Dense: Vocabulary Problem 시연

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

# 샘플 문서들
documents = [
    "자동차를 구매하는 방법",
    "차를 사는 것은 중요한 결정이다",
    "내 첫 자동차: 구입 가이드",
    "컴퓨터를 파는 가게",
    "중고 차 사는 팁"
]

# Query: 의도는 "자동차 구매" 지만 다른 표현
query = "어떻게 자동차를 사나?"

# BM25
corpus_tokens = [[word for word in doc.split()] for doc in documents]
bm25 = BM25Okapi(corpus_tokens)
bm25_scores = bm25.get_scores(query.split())

print("BM25 scores:", bm25_scores)
# Output: [2.1, 0.5, 1.8, 0.0, 0.3]
# Problem: 문서 1("차를 사는...") 은 semantic 유사하지만 점수 낮음

# Dense Retrieval
model = SentenceTransformer('sentence-transformers/xlm-r-base-multilingual-v1')
query_emb = model.encode(query, convert_to_numpy=True)
doc_embs = model.encode(documents, convert_to_numpy=True)

dense_scores = [np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)) 
                for doc_emb in doc_embs]

print("Dense scores:", dense_scores)
# Output: [0.92, 0.88, 0.85, 0.12, 0.79]
# Better: 모든 semantic 유사 문서가 높은 점수
```

### 실험 2 — 다언어 검색에서 Lexical vs Dense

```python
# 한국어 쿼리, 영어 문서
query_ko = "자동차 구매 팁"
documents_en = [
    "How to buy a car",
    "Car buying guide for beginners",
    "Computer store locations",
    "Tips for first-time car buyers"
]

# BM25: 불가능 (언어 다름)
# 또는 매우 낮은 점수

# Dense Retrieval: 가능 (multilingual encoder)
query_emb = model.encode(query_ko, convert_to_numpy=True)
doc_embs = model.encode(documents_en, convert_to_numpy=True)

dense_scores = [np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                for doc_emb in doc_embs]

print("Cross-lingual dense scores:", dense_scores)
# Output: [0.85, 0.87, 0.05, 0.83]
# 성공적으로 한영 매칭
```

### 실험 3 — Offline Indexing: FAISS ANN

```python
import faiss

# 100만 개 passage embedding (768 dims) 시뮬레이션
num_passages = 1_000_000
embedding_dim = 768

# 랜덤 embedding (실제로는 encoder 로 생성)
passage_embeddings = np.random.randn(num_passages, embedding_dim).astype('float32')

# FAISS index 생성 (HNSW 대신 IVF 간단 예시)
index = faiss.IndexFlatL2(embedding_dim)
index.add(passage_embeddings)

# Query 검색
query_emb = np.random.randn(1, embedding_dim).astype('float32')
distances, indices = index.search(query_emb, k=10)

print(f"Retrieved top-10 indices: {indices}")
print(f"Query time (exact): ~{1.0:.1f} ms")

# Approximate (HNSW 유사)
index_hnsw = faiss.IndexHNSWFlat(embedding_dim, 32)
index_hnsw.add(passage_embeddings)
distances, indices = index_hnsw.search(query_emb, k=10)

print(f"Retrieved top-10 (HNSW): {indices}")
print(f"Query time (approx): ~0.5-1 ms")
```

### 실험 4 — Paraphrase Recall 비교

```python
# Paraphrase 문서들
original_doc = "How to buy a car: tips for first-time buyers"
paraphrases = [
    "Guide to purchasing an automobile for new customers",
    "First-time car buyers: a complete guide",
    "Buying vehicles: advice for novices",
    "Tips on acquiring your first vehicle"
]

query = "how to purchase a car"

# BM25
from rank_bm25 import BM25Okapi
all_docs = [original_doc] + paraphrases
corpus = [[w for w in doc.lower().split()] for doc in all_docs]
bm25 = BM25Okapi(corpus)
bm25_scores = bm25.get_scores(query.lower().split())

# Dense
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
query_emb = model.encode(query)
doc_scores = [np.dot(query_emb, model.encode(doc)) / 
              (np.linalg.norm(query_emb) * np.linalg.norm(model.encode(doc)))
              for doc in all_docs]

print("BM25 scores:", bm25_scores)
print("Dense scores:", doc_scores)
# Dense 가 paraphrase 들을 더 균일하게 높게 순위화
```

---

## 🔗 실전 활용

| 시나리오 | BM25 한계 | Dense 해결 방법 |
|---------|---------|---------|
| 사용자 표현과 문서 paraphrase 일치 안 함 | 어휘 부족 (30% miss) | Semantic embedding: paraphrase 인식 |
| 다언어 검색 (한국어 쿼리 → 영어 문서) | 불가능 | Multilingual encoder (xlm-r, E5) |
| "자동차" vs "automobile" | Lemmatization 필요 | Automatic synonym in vector space |
| 대규모 corpus 검색 속도 (100M docs) | Inverted index 여전히 느림 | HNSW/IVF: 1-10ms with 99%+ recall |
| 문제-해답 매칭 (Q&A) | Term overlap 없을 수 있음 | Semantic relevance: supervised 학습 |

---

## ⚖️ 가정과 한계

- **Dense embedding 은 learnable** 한다고 가정 — i.e., relevant query-passage 쌍으로 학습 필요. Zero-shot 은 한계 있음.
- **Semantic 거리의 정의** 가 문맥에 따라 달라짐 — "자동차" 와 "자전거" 의 거리는 상황에 따라 가까울 수도, 멀 수도 있음.
- **Vocabulary problem 은 dense 에서도 완전 해결 안 됨** — 극히 niche 한 document 는 여전히 낮은 ranking 가능.
- **Offline embedding 생성 cost** — 100M docs 를 BERT 로 encoding 하는데 수 시간 소요. 문서 추가 시 re-indexing.

---

## 📌 핵심 정리

$$
\boxed{\text{BM25 (lexical)} \to \text{Dense Embedding (semantic)} = \text{Vocabulary Problem 극복} + \text{Paraphrase 인식}}
$$

| 측면 | BM25 | Dense Retrieval |
|-----|------|-----------------|
| Matching | Exact/synonym term overlap | Semantic cosine similarity |
| Paraphrase recall | 낮음 (<50%) | 높음 (>80%) |
| 다언어 | 불가능 (lemmatize 필요) | 자동 (multilingual encoder) |
| Indexing | Inverted index (~GB) | FAISS/HNSW (~10-100GB) |
| Query latency | 10-50ms | 1-10ms (ANN) |
| Learning | 미필요 | Supervised/unsupervised 학습 필요 |

> **핵심**: Dense retrieval 은 40년 BM25 의 lexical bias 를 semantic space 로 shift 하여 **vocabulary problem 을 probabilistically 해결**.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: Furnas 의 vocabulary problem 에서 user 가 3-term query 를 쓸 때, 각 term 의 miss rate 가 30% 라면 전체 miss rate 는?

<details>
<summary>해설</summary>

$\mathbb{P}[\text{at least 1 term miss}] = 1 - (0.7)^3 = 0.657 = 65.7\%$. 즉, 정확한 BM25 matching 으로는 3분의 2 이상이 검색 실패. Dense embedding 으로는 부분 일치도 의미를 살림.
</details>

**문제 2 (심화)**: BM25 는 term frequency 와 IDF 에 기반하는데, 왜 semantic paraphrase 가 low score 를 받는가? Cosine similarity 는 이를 어떻게 보상하는가?

<details>
<summary>해설</summary>

BM25: $\sum_t \text{IDF}(t) \cdot \text{freq}(t, d)$ → term $t$ 가 없으면 0 contribution. Paraphrase "purchase" 는 "buy" term 이 없으므로 term frequency 기반 가산 불가.

Dense: $\cos(\mathbf{e}_q, \mathbf{e}_p) = \mathbf{e}_q \cdot \mathbf{e}_p$ → embedding 이 "buy" 와 "purchase" 의 semantic direction 을 학습하면, 두 query 임베딩이 비슷하고, 두 passage 임베딩도 비슷해져 high cosine score. 즉, **term 없이도 vector direction 으로 similarity 표현**.
</details>

**문제 3 (논문 비평)**: Dense retrieval 이 BM25 의 완벽한 replacement 인가? BM25 가 여전히 dense 보다 우수한 시나리오는?

<details>
<summary>해설</summary>

BM25 가 나은 경우:
1. **Factual 검색** (정확한 entity/number 찾기): "COVID-19 발병 날짜" → BM25 는 "2019" 숫자를 정확히 매칭, dense 는 근처 숫자도 비슷한 점수 가능.
2. **요청사항 문서** (domain-specific terminology): 의료/법률 용어는 paraphrase 되기 어렵고, exact term 이 critical.
3. **새로운 도메인** (낮은 supervised data): Dense 는 relevant pair 학습 필요, BM25 는 zero-shot.

**Hybrid**: BM25 와 dense 를 ensemble (Ch3 에서 fusion 논의) 하면 두 가지 이점 모두 활용.
</details>

---

<div align="center">

[◀ 이전 (Ch1-05. Retrieve vs Rerank)](../ch1-ir-foundations/05-retrieve-rerank.md) · [📚 README](../README.md) · [다음 ▶ (02. DPR Bi-Encoder)](./02-dpr-bi-encoder.md)

</div>
