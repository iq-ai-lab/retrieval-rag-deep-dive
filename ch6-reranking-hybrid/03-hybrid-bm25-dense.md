# 03. Hybrid BM25+Dense — SPLADE 와 통합 모델

## 🎯 핵심 질문

- BM25 (sparse, 키워드 기반) 와 Dense (semantic, 임베딩 기반) 는 왜 **complementary** 한가?
- BEIR zero-shot 분석에서 BM25 가 어떤 query/document 유형에 강하고, Dense 가 약한가?
- SPLADE (sparse PLaD Enhanced) 는 BERT 의 MLM head 를 어떻게 활용하여 **sparse expansion** 을 학습하는가?
- 단일 모델로 lexical + semantic 을 통합하는 방법 — SPLADE vs ColBERT vs dense + learnable weights?

---

## 🔍 왜 Hybrid 인가 — BM25 와 Dense 의 Analysis

BEIR (Benchmark-IR) zero-shot evaluation:
- **Dense only** (e.g., mDPR): NDCG@10 = 0.40 (평균)
- **BM25 only**: NDCG@10 = 0.42
- **BM25 + Dense (RRF)**: NDCG@10 = 0.50 (+20% vs dense, +19% vs BM25)

**분석**:
1. **BM25 강점**: 정확한 키워드 (예: 인명, 약 이름), 동음이의어 (context-free 때문에 오히려 advantage).
2. **Dense 강점**: 의미적 문맥 (예: "자동차 속도" vs "자동차 엔진"), paraphrase 및 의역.
3. **Complementary**: Failure case 가 겹치지 않음 — 하나 떨어지면 다른 하나가 보충.

SPLADE 는 이 두 장점을 **단일 모델** 로 통합하려는 시도.

---

## 📐 수학적 선행 조건

- BERT MLM (Masked Language Model) — Ch5 배경
- Sparse vector representation (BOW, TF-IDF)
- Dense embeddings (DPR, ANCE)
- KL divergence, attention weights 해석
- Ranking metrics (NDCG, MAP, MRR)

---

## 📖 직관적 이해

### BM25 vs Dense vs Hybrid

```
Query: "How to cook pasta"

┌─────────────────────────────────────────────────┐
│ Document: "Boiling water is essential for      │
│ cooking noodles and pasta products. ..."       │
└─────────────────────────────────────────────────┘

(1) BM25 분석:
    - IDF(cook): 4.2  ✓
    - IDF(pasta): 3.8 ✓
    - IDF(boiling): 2.1 ✓
    - BM25 score ≈ 8.5 (high)
    → "cook" 와 "pasta" 모두 explicit

(2) Dense (DPR) 분석:
    query_emb @ doc_emb ≈ 0.72 (moderate)
    → "noodles" 는 "pasta" 와 의미 가까우나,
      embedding 은 query "cook" 와의 semantic gap 측정
    → score 가 BM25 보다 낮을 수 있음

(3) Hybrid (SPLADE):
    - BERT MLM: "cook" → attend to "boiling", "cooking"
    - "pasta" → attend to "noodles", "pasta"
    - Sparse expansion + dense pooling
    → Top-5 안에 rank up
```

### SPLADE 의 Sparse Expansion

```
Original query: ["cook", "pasta"]
                 (2 tokens)

SPLADE expansion (MLM head weights):
    "cook" → expands to ["cook", "boil", "cooking",
                         "heat", "prepare", ...]
    "pasta" → expands to ["pasta", "noodles",
                          "spaghetti", "carbs", ...]

Sparse representation (learned weights):
    cook: 0.95
    boil: 0.67
    cooking: 0.72
    pasta: 0.93
    noodles: 0.58
    ...
    
= concatenated sparse vector ✓ (BM25-compatible)

+ also dense [CLS] embedding (semantic) ✓
```

---

## ✏️ 엄밀한 정의

### 정의 6.4 — Sparse vs Dense Retrieval

**Sparse retrieval** (BM25):
$$
\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \text{TF}(t, d) \cdot \text{len\_norm}(d)
$$

$t$ = term (word), explicit term matching required.

**Dense retrieval** (DPR):
$$
\text{score}(q, d) = \text{emb}(q)^\top \text{emb}(d)
$$

$\text{emb}()$ = neural embedding (continuous vector, latent semantics).

### 정의 6.5 — SPLADE (Sparse Lexical And Dense Embedding)

Query $q$, document $d$ 에 대해:

**Sparse expansion** (MLM-based):
$$
\vec{v}_q = \text{BERT}\_\text{MLM}([CLS] q [SEP]) \quad \text{(vocab-size vector)}
$$

**Learning**: MLM logits 에 log-softmax + learned temperature.

**Dense pooling**:
$$
\vec{e}_q = \text{pooling}(\text{BERT\_hidden}([CLS] q [SEP]))
$$

**Hybrid score**:
$$
\text{score}_\text{SPLADE}(q,d) = \alpha \cdot (\vec{v}_q^\top \vec{v}_d) + (1-\alpha) \cdot (\vec{e}_q^\top \vec{e}_d)
$$

또는 두 손실을 동시에 optimize (ranking loss for dense, BM25 ranking loss for sparse).

### 정의 6.6 — BEIR Benchmark 구성

**22개 diverse datasets**:
- Trec-COVID (medical), DBpedia (entity), Scifact (scientific)
- Trec-News (news articles), MS MARCO (web search), Natural Questions (QA)
- 각 domain 에서 zero-shot (training data 없이) 평가

**Metrics**: NDCG@10, nDCG@100, MRR@10 등.

---

## 🔬 정리와 증명

### 정리 6.4 — BM25 와 Dense 의 Complementarity

**Claim**: BM25 와 Dense 의 failure cases 가 서로 orthogonal 하다 (대부분).

**Empirical evidence** (Thakur et al., BEIR):
- BM25 우위 domain: Named entity (인명, 지명), technical keywords (약 이름)
- Dense 우위 domain: Paraphrase-heavy, semantic reasoning (예: "car speed" vs "vehicle acceleration")
- Correlation 분석: 두 시스템의 ranking order 가 낮은 상관 (ρ < 0.6) — truly complementary

**정량화**: RRF fusion 의 gain (+20%) 는 두 시스템의 complementarity 를 증명.

### 정리 6.5 — SPLADE 의 수렴성

SPLADE 학습 시 **joint loss**:
$$
\mathcal{L} = \lambda \mathcal{L}_\text{sparse} + (1-\lambda) \mathcal{L}_\text{dense}
$$

여기서:
- $\mathcal{L}_\text{sparse}$ = BM25-style ranking loss (e.g., listwise)
- $\mathcal{L}_\text{dense}$ = contrastive loss (InfoNCE)

**수렴 성질**: 
- $\lambda$ 가 0 에 가까우면 dense-only convergence (좋은 dense 임베딩)
- $\lambda$ 가 1 에 가까우면 sparse-only convergence (좋은 sparse expansion)
- Optimal $\lambda$ ≈ 0.5~0.7 (둘 balance)

### 정리 6.6 — SPLADE vs BM25+RRF (Pareto)

**성능 비교** (MS MARCO 기준):
```
Model           | NDCG@10 | Latency | Size
────────────────┼─────────┼─────────┼──────
BM25            | 0.280   | 1 ms    | 0 MB (inverted index)
Dense (ANCE)    | 0.330   | 50 ms   | 500 MB (embeddings)
BM25+ANCE (RRF) | 0.380   | 51 ms   | 500 MB
SPLADE v2       | 0.375   | 30 ms   | 50 MB (sparse vectors)
```

**Pareto frontier**:
- Latency-sensitive: BM25+RRF (51ms, 0.380) > SPLADE v2 (30ms, 0.375)
- Storage-sensitive: SPLADE v2 (50MB) << Dense (500MB)
- Accuracy-only: BM25+RRF 약간 우위 (0.380 vs 0.375)

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — BEIR Evaluation Framework

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval

# Dataset 다운로드
dataset = "trec-covid"  # 또는 "dbpedia", "scifact", ...
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
data_path = util.download_and_unzip(url, "./datasets")

# 데이터 로드
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# BM25 평가
model = models.BM25()
retrieval_results = EvaluateRetrieval(model).retrieve(corpus, queries)

# Evaluation
ndcg, mrr, recall = EvaluateRetrieval(model).evaluate(
    qrels, retrieval_results, k_values=[10, 100]
)
print(f"BM25 - NDCG@10: {ndcg['NDCG@10']:.3f}")
```

### 실험 2 — Dense Retrieval (Sentence-Transformers)

```python
from sentence_transformers import SentenceTransformer
import faiss

# Pre-trained dense model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Corpus 를 embedding
corpus_embeddings = model.encode(
    [corpus[cid]['text'] for cid in corpus],
    batch_size=64, show_progress_bar=True, convert_to_numpy=True
)

# FAISS index
index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
index.add(corpus_embeddings)

# Query 와 retrieval
def retrieve_dense(query: str, k: int = 100):
    query_embedding = model.encode(query, convert_to_tensor=False)
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    return indices[0], distances[0]

# BEIR evaluation 통합
dense_results = {}
for qid, query in queries.items():
    indices, scores = retrieve_dense(query, k=100)
    dense_results[qid] = {cid: float(score) for cid, score in zip(
        [list(corpus.keys())[i] for i in indices], scores
    )}

# Evaluation
evaluator = EvaluateRetrieval(model=None)
ndcg_dense = evaluator.evaluate(qrels, dense_results, k_values=[10])
print(f"Dense - NDCG@10: {ndcg_dense['NDCG@10']:.3f}")
```

### 실험 3 — RRF Fusion (BM25 + Dense)

```python
def rrf_fusion(bm25_results, dense_results, k=60):
    """BM25 와 Dense 의 RRF fusion"""
    fused = {}
    
    for qid in bm25_results:
        doc_scores = {}
        
        # BM25 contribution
        for rank, (cid, score) in enumerate(
            sorted(bm25_results[qid].items(), key=lambda x: -x[1])[:100], 1
        ):
            doc_scores[cid] = doc_scores.get(cid, 0) + 1 / (k + rank)
        
        # Dense contribution
        for rank, (cid, score) in enumerate(
            sorted(dense_results[qid].items(), key=lambda x: -x[1])[:100], 1
        ):
            doc_scores[cid] = doc_scores.get(cid, 0) + 1 / (k + rank)
        
        # Sort by fused score
        fused[qid] = dict(
            sorted(doc_scores.items(), key=lambda x: -x[1])
        )
    
    return fused

fused_results = rrf_fusion(bm25_results, dense_results)

# Evaluation
ndcg_fused = evaluator.evaluate(qrels, fused_results, k_values=[10])
print(f"BM25+Dense (RRF) - NDCG@10: {ndcg_fused['NDCG@10']:.3f}")
```

### 실험 4 — SPLADE 구현 (간소화)

```python
import torch
from transformers import AutoTokenizer, AutoModel

class SPLADERetriever:
    def __init__(self, model_name="naver/splade-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def encode_sparse(self, texts: list):
        """Sparse vector 생성"""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True,
                               truncation=True, max_length=256)
        
        with torch.no_grad():
            output = self.model(**inputs)
            # MLM logits (vocab size)
            logits = output.logits[0]  # [seq_len, vocab_size]
            # log-softmax 로 sparse representation
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Max pooling over sequence
            sparse_vec, _ = torch.max(log_probs, dim=0)
            # ReLU 로 음수는 0
            sparse_vec = torch.relu(sparse_vec)
        
        return sparse_vec.cpu().numpy()
    
    def encode_dense(self, texts: list):
        """Dense embedding 생성"""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True,
                               truncation=True, max_length=256)
        
        with torch.no_grad():
            output = self.model(**inputs)
            # [CLS] embedding
            cls_embedding = output.last_hidden_state[:, 0]
        
        return cls_embedding.cpu().numpy()

# Usage
splade = SPLADERetriever()
query = "How to cook pasta"
doc = "Boiling water is essential for cooking noodles..."

sparse_q = splade.encode_sparse([query])
dense_q = splade.encode_dense([query])
sparse_d = splade.encode_sparse([doc])
dense_d = splade.encode_dense([doc])

# Hybrid score
alpha = 0.6
hybrid_score = alpha * (sparse_q @ sparse_d.T)[0, 0] + \
               (1 - alpha) * (dense_q @ dense_d.T)[0, 0]
print(f"SPLADE score: {hybrid_score:.4f}")
```

---

## 🔗 실전 활용

| 상황 | 추천 방식 | 이유 |
|------|----------|------|
| 기존 BM25 인프라 | BM25+Dense (RRF) | 구현 단순, inverted index 재사용 |
| 정확도 최우선 | SPLADE v3 | 단일 모델, dense+sparse 최적화 |
| Latency 중요 (< 50ms) | SPLADE | Dense 보다 빠름, 정확도 손실 적음 |
| Domain-specific corpus | Fine-tuned Dense + BM25 RRF | Domain 특성 반영, complementarity 확보 |
| 매우 큰 corpus (1B+) | BM25만 | 임베딩 저장 불가능, sparse inverted index 만 사용 |

---

## ⚖️ 가정과 한계

- **BEIR 대표성**: Zero-shot BEIR 는 평균적 성능 — specific domain 에서는 다를 수 있음.
- **Complementarity 가정**: BM25 와 Dense 의 failure case 가 항상 disjoint 하지 않음.
- **SPLADE 학습 cost**: Joint training 이 복잡 — inference 는 빠르지만 학습/fine-tuning 어려움.
- **Scaling**: 매우 큰 corpus 에서는 SPLADE 의 sparse vector 도 (vocab size 크기) 메모리 문제 가능.

---

## 📌 핵심 정리

| 방식 | 장점 | 단점 | BEIR NDCG@10 |
|------|------|------|--------------|
| BM25 | 빠름, 저장량 적음 | 의미 약함 | 0.42 |
| Dense | 의미 이해, paraphrase | 느림, 저장량 큼 | 0.40 |
| BM25+Dense (RRF) | Complementarity | 두 배 인프라 | 0.50 |
| SPLADE | 단일 모델, 양쪽 장점 | 학습 어려움 | 0.49 |

> **핵심**: Hybrid search 는 단일 modality (BM25 또는 Dense) 대비 +20% 향상. SPLADE 는 같은 성능을 단일 모델로 달성.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: BEIR 에서 BM25 = 0.42, Dense = 0.40, RRF = 0.50 일 때, complementarity 의 정도는?

<details>
<summary>해설</summary>

Gain 분석:
- RRF vs BM25: (0.50 - 0.42) / 0.42 ≈ 19% 상대 향상
- RRF vs Dense: (0.50 - 0.40) / 0.40 ≈ 25% 상대 향상
- 만약 독립적 (truly orthogonal) 이면 gain 이 더 클 것 (예: 0.55+)
- 실제 0.50 은 "충분히 complementary 하지만 일부 overlap" 을 의미

"True complementarity" 정량 방법: 
$\rho = \text{corr}(\text{BM25 ranking}, \text{Dense ranking}) < 0.5$ (대체로)
</details>

**문제 2 (심화)**: SPLADE 의 sparse expansion MLM logits 에 log-softmax 를 적용하는 이유는?

<details>
<summary>해설</summary>

(1) **Normalization**: MLM logits 의 scale 은 batch, 모델 크기에 의존 → log-softmax 로 정규화하면 확률 해석 가능.

(2) **Sparse penalty**: exp(log-softmax) 후 threshold 로 작은 값들을 0 으로 처리 → sparse vector 형성 (대부분 zero entry).

(3) **Learning stability**: Raw logits 에 relu 는 불안정 → log-softmax 로 bounded 범위에서 학습.

만약 log-softmax 없이 relu(logits) 만 하면 sparse 가 덜 효과적 (모든 양수 logits 가 살아남).
</details>

**문제 3 (논문 비평)**: "SPLADE 는 BM25+Dense 를 이기지 못한다 (SPLADE v2: 0.375 < RRF: 0.380)" 는 주장에 대한 반박?

<details>
<summary>해설</summary>

반박 (1): Latency 고려 — SPLADE 30ms vs RRF 51ms. Per-query latency 가 중요 (배치 처리 환경과 다름).

반박 (2): 저장량 — SPLADE 50MB (sparse vocab vectors) vs RRF 500MB (dense embeddings). 확장성 우위.

반박 (3): End-to-end 시스템 — RRF 는 "two models + fusion" 복잡도. SPLADE 는 단일 모델 배포 단순.

따라서 "정확도만" 비교는 불공평 — context (latency, memory, complexity) 를 고려하면 SPLADE 는 trade-off 관점에서 경쟁력 있음.

최신 SPLADE v3 (2023) 은 정확도도 RRF 수준 (0.38+) 달성.
</details>

---

<div align="center">

[◀ 이전 (02. RRF)](./02-rrf.md) · [📚 README](../README.md) · [다음 ▶ (04. LLM-as-Reranker)](./04-llm-as-reranker.md)

</div>
