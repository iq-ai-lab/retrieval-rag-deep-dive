# 02. Bi-Encoder Architecture — DPR (Karpukhin 2020)

## 🎯 핵심 질문

- DPR 이 **two separate BERT encoder** (query encoder vs passage encoder) 를 분리한 이유는 정확히 무엇인가?
- Bi-encoder 의 점수 함수 $s(q,p) = f_q(q)^\top f_p(p)$ 는 왜 효율적인가? Cross-encoder 와의 trade-off 는?
- Offline indexing 의 경제성이 정확히 어떻게 계산되는가 — 검색 시 어떤 연산이 필요 없어지는가?
- DPR 이 Natural Questions 와 TriviaQA 에서 달성한 정확도 향상 (dense vs BM25) 은 정량적으로 무엇인가?

---

## 🔍 왜 Bi-Encoder 가 dense retrieval 의 표준이 되었는가

DPR (Dense Passage Retrieval) 은 2020년 Facebook AI 에서 발표한 landmark paper 입니다. 핵심 기여:

1. **두 encoder 의 명시적 분리** — Query 와 passage 는 다른 분포를 따르고, 따라서 별도로 학습하는 것이 효율적.

2. **Offline embedding caching** — Passage embedding 은 학습 완료 후 한 번만 계산. Query embedding 은 inference 시점에만 계산. 따라서 100M passage 를 미리 embedding 하고 FAISS 로 indexing → millisecond-scale 검색 가능.

3. **Supervised training 전략** — NQ, TriviaQA 같은 labeled retrieval 데이터로 supervised contrastive learning (InfoNCE). BM25 보다 20-30% 절대 정확도 향상.

이 문서는 DPR 의 architecture, offline indexing 경제, training 전략, 그리고 실제 성과를 엄밀히 정식화합니다.

---

## 📐 수학적 선행 조건

- BERT / Transformer 기초: [CLS] token, self-attention
- Vector 거리: inner product, cosine similarity
- Supervised learning: cross-entropy, triplet loss (informal)
- (선택) Information retrieval metrics: MRR, NDCG

---

## 📖 직각의 이해

### Bi-Encoder 의 구조 비유

```
시나리오: 백만 명 중에서 특정 사람을 찾는 경매인

[ Cross-Encoder (대회 심사위원) ]
- 경매인이 입장할 때마다, 심사위원이 그 사람과 매칭 점수를 직접 계산
- 정확하지만, 1000만 쌍 모두 점수를 매겨야 함 (O(N) cross-product)
- 검색은 느림

[ Bi-Encoder (미리 작성된 신원 카드) ]
- 모든 사람의 신원 카드(embedding)를 미리 작성
- 경매인(query embedding)과 신원 카드들을 비교 → 빠른 매칭
- O(N) 사전 계산 + O(1) 내적 비교 = offline 경제성
```

### Offline Indexing 의 흐름

```
Training Phase (한 번):
  Query encoder: f_q(q)       ← BERT + 학습
  Passage encoder: f_p(p)     ← BERT + 학습

Offline Indexing Phase (한 번):
  모든 passage p_1, ..., p_M 에 대해:
    embedding[i] = f_p(p_i)   ← M 번 encoding (GPU 병렬 가능)
  FAISS 에 저장

Inference Phase (쿼리마다):
  query embedding = f_q(q)    ← 1 번 encoding (fast)
  top-k = FAISS.search(query_embedding, k=100)  ← 1ms
  반환
```

### Dense vs Sparse 메모리 모델

```
BM25 Inverted Index:
- Vocab size 100K, Avg doc len 200
- Memory: ~1-10 GB (sparse, 압축 가능)

Dense FAISS Index:
- 100M passages, 768 dims, FP16
- Memory: 100M × 768 × 2 bytes = 150 GB
- BUT: HNSW 압축으로 5-10× 축소 가능

실제 deployment:
- FAISS (PQ 또는 HNSW): 10-50 GB for 100M passages
- + query encoder weight: ~400 MB
```

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Bi-Encoder 아키텍처

**Query Encoder**:
$$
\mathbf{q} = f_q(q) \in \mathbb{R}^d
$$
여기서 $f_q$ 는 BERT 의 [CLS] token 의 출력 (또는 mean pooling):
$$
\mathbf{q} = \text{pool}([\text{embedding of } q])
$$

**Passage Encoder**:
$$
\mathbf{p} = f_p(p) \in \mathbb{R}^d
$$
마찬가지로 BERT [CLS] 또는 mean pooling.

**Retrieval Score**:
$$
s(q, p) = \mathbf{q}^\top \mathbf{p} = \langle \mathbf{q}, \mathbf{p} \rangle
$$

따라서 top-k retrieval:
$$
p_1, \ldots, p_k = \arg\top_k s(q, p) \quad \forall p \in \text{corpus}
$$

### 정의 2.2 — Offline Indexing 복잡도

**구성 단계**:
$$
\text{Precompute: } \mathbf{p}_i = f_p(p_i) \quad \forall i = 1, \ldots, M \quad [\text{O}(M \cdot d)]
$$

**검색 단계**:
$$
\text{Query: } \mathbf{q} = f_q(q) \quad [O(1)]
$$
$$
\text{ANN search: } \text{top-k closest } \mathbf{p}_i \text{ to } \mathbf{q} \quad [O(\log M) \text{ HNSW 또는 } O(M/c) \text{ IVF}]
$$

**vs Cross-Encoder**:
$$
\text{Cross-Encoder score: } s(q, p_i) = f_{\text{cross}}(q, p_i) \quad [O(M \cdot d)]
$$

따라서 **offline indexing 이득**:
$$
\frac{\text{Cross-Encoder inference cost}}{\text{Bi-Encoder inference cost}} = \frac{O(M \cdot d)}{O(\log M + d)} \approx \frac{M}{K} \approx 100\text{-}1000\times
$$

### 정의 2.3 — Supervised Training

Labeled triplet $(q, p^+, p^-)$ 에 대해 (positive = relevant passage, negative = irrelevant):

InfoNCE loss (Ch3 에서 자세히):
$$
L = -\log \frac{e^{s(q, p^+) / \tau}}{\sum_j e^{s(q, p_j) / \tau}}
$$

여기서 $p_j$ 는 batch 내의 모든 passage (in-batch negatives).

---

## 🔬 정리와 증명

### 정리 2.1 — Bi-Encoder 의 Linear Separability

Dense embedding space 에서 relevant query-passage 쌍 $(q^{(i)}, p^{(i)+})$ 이 contrastive loss 로 충분히 학습되면:
$$
\mathbb{E}[\mathbf{q}^{(i)} \cdot \mathbf{p}^{(i)+}] - \mathbb{E}[\mathbf{q}^{(i)} \cdot \mathbf{p}^{(i)-}] \geq \Delta > 0
$$

따라서 threshold 기반 이진 분류 (relevant/irrelevant) 는 hyperplane $\mathbf{q} \cdot \mathbf{p} = \theta$ 로 **linearly separable**.

**증명 sketch**: InfoNCE loss (Ch3) 는 positive 와 negative 를 최대한 밀어낸다. Sufficient margin $\Delta$ 가 존재하면 linear classifier 가 두 집단을 분리 가능 $\square$.

### 정리 2.2 — ANN Search 의 Recall-Efficiency Trade-off

FAISS HNSW 인덱스에서:
$$
\text{Recall@k (top-k 정확도)} = \frac{\text{\# correctly retrieved}}{\text{total relevant}}
$$

**Complexity**:
$$
T_{\text{search}} = O(\log M + \epsilon \cdot d)
$$

여기서 $\epsilon$ 는 graph 탐색의 branching factor. Recall 과 latency 는:
$$
\text{Recall} \approx 1 - e^{-\epsilon \sqrt{d}} \quad (\text{asymptotic})
$$

**결과**: 99% recall@10 달성 시, T ≈ 1-5 ms (M = 100M, d = 768 기준).

### 정리 2.3 — Mean Pooling vs [CLS] Token

Query/passage embedding 을 구성하는 두 방식:

**[CLS] pooling**:
$$
\mathbf{p} = h_{\text{[CLS]}} \in \mathbb{R}^{d_h}
$$

**Mean pooling**:
$$
\mathbf{p} = \frac{1}{L} \sum_{i=1}^L h_i
$$

DPR 은 두 방식 모두 실험 — mean pooling 이 약 1-2% 높은 MRR. **이유**: mean pooling 은 passage 의 모든 word 의 semantic 정보를 평균화하여 더 robust $\square$.

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — DPR Bi-Encoder 모델 로드 및 Encoding

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Pre-trained DPR encoder 로드 (Hugging Face)
query_encoder_name = "facebook/dpr-question_encoder-single-nq-base"
passage_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"

query_tokenizer = AutoTokenizer.from_pretrained(query_encoder_name)
query_encoder = AutoModel.from_pretrained(query_encoder_name)

passage_tokenizer = AutoTokenizer.from_pretrained(passage_encoder_name)
passage_encoder = AutoModel.from_pretrained(passage_encoder_name)

# Query embedding
query = "자동차는 어떻게 사나?"
query_inputs = query_tokenizer(query, return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    query_outputs = query_encoder(**query_inputs)
    query_embedding = query_outputs.pooler_output  # [1, 768]

print(f"Query embedding shape: {query_embedding.shape}")

# Passage embedding
passage = "자동차 구매 가이드: 첫 번째 차를 사는 방법"
passage_inputs = passage_tokenizer(passage, return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    passage_outputs = passage_encoder(**passage_inputs)
    passage_embedding = passage_outputs.pooler_output  # [1, 768]

# Retrieval score
score = torch.matmul(query_embedding, passage_embedding.T).item()
print(f"Similarity score: {score:.4f}")
```

### 실험 2 — Offline Indexing with FAISS

```python
import faiss

# 100만 개 passage embedding 시뮬레이션
num_passages = 1_000_000
embedding_dim = 768

# 실제로는 encoder 로 생성하지만, 여기선 시뮬레이션
passage_embeddings = np.random.randn(num_passages, embedding_dim).astype('float32')

# L2 normalize (cosine similarity 를 위해)
faiss.normalize_L2(passage_embeddings)

# FAISS index 생성 (HNSW)
index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 neighbors per node
index.add(passage_embeddings)

print(f"Index size: {index.ntotal} passages")

# Query 검색
query_embedding = np.random.randn(1, embedding_dim).astype('float32')
faiss.normalize_L2(query_embedding)

distances, indices = index.search(query_embedding, k=100)

print(f"Top-1 passage index: {indices[0, 0]}")
print(f"Top-10 distances: {distances[0, :10]}")
```

### 실험 3 — Bi-Encoder vs Cross-Encoder 비용 비교

```python
import time

# Bi-Encoder: offline precomputation
print("=== Bi-Encoder ===")
t_offline = time.time()
# 모든 passage embedding (병렬화 가능)
passage_embeddings = []
batch_size = 32
for batch_idx in range(0, num_passages // batch_size):
    batch_passages = [f"passage {i}" for i in range(batch_size)]
    batch_inputs = passage_tokenizer(batch_passages, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = passage_encoder(**batch_inputs)
        embeddings = outputs.pooler_output
        passage_embeddings.append(embeddings.cpu())
        
t_offline_end = time.time()
print(f"Offline indexing time: {t_offline_end - t_offline:.2f}s")

# Inference
t_infer = time.time()
query_embedding = query_encoder(**query_inputs).pooler_output
# FAISS search (simulated, ~1ms)
top_k_scores = torch.matmul(query_embedding, torch.cat(passage_embeddings).T)
t_infer_end = time.time()
print(f"Per-query inference time: {(t_infer_end - t_infer)*1000:.2f}ms")

# Cross-Encoder: no offline compute, but slow inference
print("\n=== Cross-Encoder ===")
# Would need to run encoder on (query, passage) pairs for ALL passages
# Estimated: 100M pairs × 50ms/pair = 5M seconds = 58 days!
print("Estimated per-query time: ~50ms (for 100 passages) to hours (for 1M)")
```

### 실험 4 — Mean Pooling vs [CLS] Comparison

```python
# Mean pooling vs [CLS] token comparison
def mean_pool(last_hidden_states, attention_mask):
    """Mean pool last hidden states."""
    # Expand attention mask for broadcasting
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.shape).float()
    # Sum and divide by number of non-padding tokens
    sum_hidden = (last_hidden_states * expanded_mask).sum(dim=1)
    sum_mask = expanded_mask.sum(dim=1)
    mean_embedding = sum_hidden / sum_mask.clamp(min=1e-9)
    return mean_embedding

# Query embedding
query_inputs = query_tokenizer(query, return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    outputs = query_encoder(**query_inputs, output_hidden_states=True)
    cls_embedding = outputs.pooler_output  # [1, 768]
    mean_embedding = mean_pool(outputs.last_hidden_state, query_inputs['attention_mask'])

print(f"CLS shape: {cls_embedding.shape}")
print(f"Mean pool shape: {mean_embedding.shape}")

# Similarity comparison
similarities_cls = []
similarities_mean = []

# 여러 passage 에 대해
for passage in passages:
    passage_inputs = passage_tokenizer(passage, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        p_outputs = passage_encoder(**passage_inputs, output_hidden_states=True)
        p_cls = p_outputs.pooler_output
        p_mean = mean_pool(p_outputs.last_hidden_state, passage_inputs['attention_mask'])
        
        sim_cls = torch.matmul(cls_embedding, p_cls.T).item()
        sim_mean = torch.matmul(mean_embedding, p_mean.T).item()
        
        similarities_cls.append(sim_cls)
        similarities_mean.append(sim_mean)

print(f"Mean similarity (CLS): {np.mean(similarities_cls):.4f}")
print(f"Mean similarity (Mean pool): {np.mean(similarities_mean):.4f}")
```

---

## 🔗 실전 활용

| 상황 | 추천 방식 | 이유 |
|-----|---------|-----|
| 대규모 corpus (>1M docs) | Bi-Encoder + FAISS | Offline indexing 경제 (1ms/query) |
| 실시간 업데이트 필요 | Bi-Encoder + in-memory | Passage embedding 재계산 빠름 |
| 작은 corpus (<10K) | Cross-Encoder | 정확도 이득 (3-5% MRR), latency 무시할 수 있음 |
| Reranker 로 2단계 | Bi-Encoder (1단계) + Cross-Encoder (2단계) | 속도와 정확도 절충 |
| Few-shot learning | Bi-Encoder (zero-shot) | 다양한 도메인 사전학습 모델 |

---

## ⚖️ 가정과 한계

- **Offline embedding 은 static** — Document 내용 변경 시 재계산 필요. 실시간 업데이트는 모든 passage embedding 을 on-the-fly 계산해야 하므로 불가능.

- **Query-passage 분포 차이** — DPR 은 query 와 passage 의 encoder 를 분리했는데, 실제로 두 분포가 얼마나 다른지는 경험적 (dataset 의존).

- **FAISS ANN 의 approximate 성** — Top-k 의 일부를 놓칠 수 있음 (99% recall@10, 1% miss 가능). Hybrid 전략으로 보완 필요.

- **[CLS] vs mean pooling 선택** — 데이터셋마다 최적이 다를 수 있음. 일반화는 어렵다.

- **Embedding 차원이 크다** — 768 dims 는 메모리 과도. Quantization (Ch3) 으로 FP32 → INT8 (4배 압축) 하면 정확도 2-3% 하락.

---

## 📌 핵심 정리

$$
\boxed{s(q, p) = f_q(q)^\top f_p(p) \quad \Rightarrow \quad \text{offline cache } \mathbf{p} + \text{fast inference}}
$$

| 측면 | Bi-Encoder (DPR) | Cross-Encoder |
|-----|------------------|--------------|
| Offline cost | O(M · d) (한 번) | 0 |
| Per-query cost | O(d + ANN search) ~1-10ms | O(M · d) ~seconds |
| 정확도 | ~87-90% MRR (NQ) | ~93-95% (3-5% 높음) |
| Scalability | 100M+ | <100K |
| 메모리 | 100-200 GB | Query encoder 만 |
| Use case | Large-scale retrieval | Reranking, small corpus |

> **핵심**: Bi-encoder 의 offline caching 이 dense retrieval 을 **web-scale** 로 만들었음.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: 100만 개 passage, 각 768 dims, FP32 기준. FAISS 에 저장하는데 필요한 메모리는?

<details>
<summary>해설</summary>

$100M \times 768 \times 4 \text{ bytes} = 300$ GB (uncompressed). HNSW 또는 PQ 압축으로 10-20% 정도만 저장 가능 (50% recall@10 목표): ~30-60 GB. 실제 deployment 에서는 quantization (FP16) 로 절반 (150 GB), 또는 int8 로 4분의 1 (75 GB) 가능.
</details>

**문제 2 (심화)**: DPR 이 query 와 passage encoder 를 별도로 두는 대신, 통합 encoder $f(q \text{ or } p)$ 를 공유할 수 없는 이유는? (실제로 시도된 적 있음)

<details>
<summary>해설</summary>

Shared encoder 는 query 와 passage 의 representation space 를 강제로 한 개로 압축. Query 는 "자동차 사는 법?" 같은 5단어, passage 는 "자동차 구입 완벽 가이드..." 같은 100단어 → 길이 분포가 다름. 따라서 별도 encoder 가 각 분포에 특화. Empirical 로 separate encoder 가 +3-5% MRR.
</details>

**問題 3 (논文 비평)**: FAISS ANN search 에서 1% 의 top-passage 를 놓치는 것이 final retrieval accuracy 에 얼마나 영향을 미치는가?

<details>
<summary>해설</summary>

Top-1 retrieval (exact answer 있을 확률) 에서는 매우 심각 (1% 직접 loss). 하지만 top-100 retrieval 에서는 완화됨 (놓친 passage 가 2-5위였을 확률이 낮음). Hybrid 전략: exact 또는 high-threshold ANN 을 사용하면 99.9% recall@100 달성 가능. DPR paper 에서 실제 실험: 99% recall@100 으로도 MRR 손실 <1%.
</details>

---

<div align="center">

[◀ 이전 (01. BM25 한계)](./01-bm25-limits.md) · [📚 README](../README.md) · [다음 ▶ (03. InfoNCE In-Batch)](./03-infonce-in-batch.md)

</div>
