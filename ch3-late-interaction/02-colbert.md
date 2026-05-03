# 02. ColBERT — Late Interaction (Khattab 2020)

## 🎯 핵심 질문

- Cross-encoder 는 정확하지만 $O(N)$ 비용이 문제인데, 이를 **indexing** 으로 해결한다는 것은 어떤 의미인가?
- "Late interaction" 이란 정확히 무엇이고, 왜 "query embedding" 과 "document per-token embedding" 을 분리하는 게 핵심인가?
- MaxSim score $S(q,d) = \sum_{i=1}^{m} \max_{j=1}^{n} E_{q_i}^\top E_{d_j}$ 의 정보론적 의미는 무엇인가? 이게 single-vector pooling 의 정보 손실을 어떻게 회복하는가?
- ColBERT 가 first-stage retrieval 로 sub-linear latency 를 달성할 수 있는 이유는?

---

## 🔍 왜 이 기법이 retrieval·RAG 에 중요한가

Ch3-01 cross-encoder 는 정확하지만 $O(N)$ cost 로 first-stage 에 쓸 수 없습니다. Ch2 dense retrieval (DPR) 은 빠르지만 single-vector embedding 의 정보 손실로 accuracy 가 낮습니다.

**ColBERT 는 양쪽의 장점을 합칩니다**:
1. **Per-token embedding** — document 의 각 token 에 대해 embedding (DPR 의 mean pooling 대신)
2. **Late interaction** — query embedding 과 document token embedding 을 MaxSim 으로 매칭 (cross-encoder 의 full attention 대신)
3. **Indexable** — document embeddings 를 indexing 가능 (query embedding + MaxSim 은 모두 offline 계산)
4. **Performance** — NDCG@10 on MS MARCO ~0.39 (DPR ~0.35, cross-encoder ~0.42)

결과적으로 ColBERT 는:
- First-stage 로 쓸 수 있는 **유일한 interaction-based** 방법 (Ch4 ANN 과 함께)
- 2019-2021 년 SOTA 재정의
- Self-RAG, CRAG 등 이후 RAG 방법들의 기반

---

## 📐 수학적 선행 조건

- Dense retrieval / embedding space 의 cosine similarity (Ch2)
- Cross-encoder 와 token-level interaction 의 개념 (Ch3-01)
- Linear algebra: outer product, dot product, max operator
- Information bottleneck (optional): how single vector loses information

---

## 📖 직관적 이해

### MaxSim: Query-Document Matching

```
Cross-Encoder (Ch3-01):
query token x_1, x_2, x_3     doc token y_1, y_2, y_3
                ↓
        Full attention (모든 token pair 상호작용)
                ↓
            [CLS] aggregation
                ↓
          single score s(q,d)

⚠️ 문제: 모든 token pair 의 attention 계산 필요 (O(N))


ColBERT (Late Interaction):
query: x_1, x_2, x_3          doc: y_1, y_2, y_3
  ↓                             ↓
E_q: [e_{q_1}, e_{q_2}, e_{q_3}]   E_d: [e_{d_1}, e_{d_2}, e_{d_3}]
  ↓                             ↓
(각 token 의 embedding)  (offline indexing 가능)
  ↓                             ↓
        MaxSim 계산 (online)
        ↓
S(q,d) = max_j e_{q_1}^T e_{d_j} 
       + max_j e_{q_2}^T e_{d_j}
       + max_j e_{q_3}^T e_{d_j}

✓ 장점: embedding 은 offline, MaxSim 만 online (빠름)
```

### MaxSim 의 직관

```
Query embedding 들: E_q = [e_{q_1}, e_{q_2}, ..., e_{q_m}]
                    (각 query token 의 768-dim embedding)

Document embedding 들: E_d = [e_{d_1}, e_{d_2}, ..., e_{d_n}]
                       (각 doc token 의 768-dim embedding)

각 query token i:
  - e_{q_i} 와 모든 doc token embedding 들의 inner product 계산
  - 가장 높은 score 를 갖는 doc token 선택 (max)
  - 그것이 query token i 의 "best match score"

전체 score:
  S(q,d) = Σ_i max_j (e_{q_i} · e_{d_j})
         = query 의 각 token 이 doc 에서 최고의 match 를 찾은 후 합산
         
⚠️ "취한 후에 떨어뜨린 휴대폰" 문제:
   document 에 매우 좋은 match 가 없으면, MaxSim 이 "그나마 낫다" 를 선택
   (정확하지는 않지만, 대체 robust)
```

### Per-Token vs Mean-Pooled Embedding

```
DPR (Mean Pooling):
Doc tokens: [y_1, y_2, y_3, ..., y_100]
  ↓
BERT encoding (각 token embedding 생성)
  ↓
Mean pooling: e_d = (e_{d_1} + e_{d_2} + ... + e_{d_100}) / 100
  ↓
Single 768-dim vector
  ↓
Query embedding e_q 와 dot product
  ↓
Relevance score (한 점)

⚠️ 정보 손실: 100개 token embedding 을 1개로 압축


ColBERT (Per-Token):
Doc tokens: [y_1, y_2, y_3, ..., y_100]
  ↓
BERT encoding
  ↓
Per-token output: E_d = [e_{d_1}, e_{d_2}, ..., e_{d_100}]
  ↓
100개 768-dim vector (정보 손실 없음)
  ↓
Query embedding 과 MaxSim matching
  ↓
Relevance score

✓ 장점: 모든 token-level 정보 보존
✓ 장점: Indexing 가능 (document 는 static)
```

---

## ✏️ 엄밀한 정의

### 정의 3.4 — ColBERT Representation

**Query representation**: 
$$
E_q = [e_{q_1}, e_{q_2}, \ldots, e_{q_m}] \in \mathbb{R}^{m \times d}
$$
where each $e_{q_i} = \text{LayerNorm}(h_{q_i}^{(L)})$, $h_{q_i}^{(L)}$ is the final hidden state of query token $i$, and $d = 128$ (not 768, reduced via projection).

**Document representation**:
$$
E_d = [e_{d_1}, e_{d_2}, \ldots, e_{d_n}] \in \mathbb{R}^{n \times d}
$$
where each $e_{d_j} = \text{LayerNorm}(h_{d_j}^{(L)})$ projected to $d$-dim.

Note: ColBERT projects final BERT hidden (768-dim) to 128-dim for efficiency.

### 정의 3.5 — MaxSim Interaction

The relevance score between query and document:
$$
S(q, d) = \sum_{i=1}^{m} \max_{j=1}^{n} e_{q_i}^\top e_{d_j}
$$

**Intuition**: For each query token $i$, find the document token $j$ with maximum similarity, and sum these maximum similarities.

### 정의 3.6 — Scalable Indexing

1. **Offline (indexing phase)**:
   - For each document $d$: compute and store $E_d \in \mathbb{R}^{n \times 128}$
   - Quantize to 8-bit per token embedding (Ch3-03 PLAID extends this)
   
2. **Online (retrieval phase)**:
   - For query $q$: compute $E_q$ (online)
   - For each candidate document (from ANN), compute $S(q,d)$ via MaxSim
   - Latency: $O(m \cdot n)$ per candidate, but $n \approx 100$ (doc length), $m \approx 20$ (query length) → $O(2000)$ ops per doc (vs $O(512^2) = O(262k)$ for cross-encoder)

---

## 🔬 정리와 증명

### 정리 3.4 — MaxSim 의 Information-Theoretic Expressiveness

**명제**: ColBERT 의 MaxSim score 는 cross-encoder 의 token-level interaction 을 **근사**한다.

**증명 sketch**:
1. Cross-encoder: $s(q,d) = f_\text{BERT}([\text{CLS}], q, \text{[SEP]}, d)$ where final layer attention 이 모든 token pair interaction capture.
2. ColBERT: $S(q,d) = \sum_i \max_j e_{q_i}^\top e_{d_j}$ captures "best match per query token".
3. Key observation: If query token $i$ is relevant, it will attend (in BERT) to one of a few document tokens. ColBERT 's max operator recovers this by selecting the highest-scoring doc token.
4. For well-trained BERT embeddings, attention patterns and embedding similarities align → ColBERT recovers most interaction information $\square$.

### 정리 3.5 — Single-Vector Pooling 의 정보 손실

**명제**: DPR 의 mean pooling embedding $e_d = \text{mean}(E_d)$ 은 maximally lossy 인 반면, ColBERT 의 per-token embedding $E_d$ 는 lossless.

**증명**:
- DPR: $\mathbb{R}^{n \times 768} \to \mathbb{R}^{768}$ (dimension collapse)
- Information bottleneck: 100+ token embeddings 를 1개로 압축 → 모든 fine-grained token info 손실
- ColBERT: $\mathbb{R}^{n \times 128} \to \mathbb{R}^{n \times 128}$ (dimension preserved)
- 단, MaxSim 은 각 query token 이 "가장 좋은" doc token match 만 본다 (worst-case sublinear loss, average-case minimal) $\square$.

### 정리 3.6 — MaxSim 의 Subadditivity (안 좋은 성질)

**명제**: MaxSim score 는 query 길이에 대해 단조증가하지만, 개별 token relevance 의 합이 항상 정확하지는 않음.

**증명**:
- $S(q,d) = \sum_i \max_j e_{q_i}^\top e_{d_j}$ 는 쿼리에 관련 없는 token 추가 시에도 증가 (문제)
- 이를 완화하려고 ColBERT 는 query-side document-side 모두 normalization layer 사용
- 또한 training 시 negative documents 와의 contrastive learning 으로 부분적 완화 $\square$.

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — Per-Token Embedding 구성

```python
import torch
from transformers import AutoTokenizer, AutoModel

# ColBERT-style 모델 로드 (또는 직접 구축)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased")

# Projection layer: 768 → 128
d = 128
projection = torch.nn.Linear(768, d)

def encode_document(text, model, tokenizer, projection):
    """
    Encode document to per-token embedding matrix
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (1, seq_len, 768)
        embeddings = projection(hidden)      # (1, seq_len, 128)
        # LayerNorm
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings.squeeze(0)  # (seq_len, 128)

# Example
doc = "Machine learning is a subset of artificial intelligence and statistics."
E_d = encode_document(doc, bert, tokenizer, projection)
print(f"Document embedding shape: {E_d.shape}")  # (seq_len, 128)

query = "What is machine learning?"
E_q = encode_document(query, bert, tokenizer, projection)
print(f"Query embedding shape: {E_q.shape}")  # (query_len, 128)
```

### 실험 2 — MaxSim Score 계산

```python
def maxsim(E_q, E_d):
    """
    Compute MaxSim score between query and document embeddings
    E_q: (m, d)
    E_d: (n, d)
    Returns: scalar score
    """
    # Compute all query-document token similarities
    similarities = E_q @ E_d.T  # (m, n)
    
    # Take max for each query token
    max_scores = similarities.max(dim=1)[0]  # (m,)
    
    # Sum over query tokens
    score = max_scores.sum()
    return score.item()

# Example
score = maxsim(E_q, E_d)
print(f"MaxSim score: {score:.4f}")
```

### 실험 3 — Batch Retrieval with FAISS

```python
import numpy as np
import faiss

# Assume we have indexed all document embeddings
# documents: list of (doc_id, document_text)
# For simplicity, simulate with random embeddings

num_docs = 1000
doc_len = 100  # average
d = 128

# Simulate document embeddings (in practice, computed via encoder)
# For each doc, we store all token embeddings
doc_embeddings = []
for i in range(num_docs):
    n_tokens = np.random.randint(50, 150)
    emb = np.random.randn(n_tokens, d).astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    doc_embeddings.append(emb)

# Query encoding (online)
query_text = "What is machine learning?"
E_q = np.random.randn(5, d).astype(np.float32)  # simulated query embeddings
E_q = E_q / (np.linalg.norm(E_q, axis=1, keepdims=True) + 1e-8)

# Scoring all documents with MaxSim
scores = []
for E_d in doc_embeddings:
    # MaxSim computation
    similarities = E_q @ E_d.T  # (m, n)
    score = similarities.max(axis=1).sum()
    scores.append(score)

# Rank by score
ranked_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
print("Top 5 documents by MaxSim:")
for rank, (doc_id, score) in enumerate(ranked_docs[:5]):
    print(f"{rank+1}. Doc {doc_id}: {score:.4f}")
```

### 실험 4 — ColBERT Heatmap (Token-Level Matching Visualization)

```python
import matplotlib.pyplot as plt

# Compute similarity matrix
similarities = E_q @ E_d.T  # (m, n)

plt.figure(figsize=(10, 4))
plt.imshow(similarities.numpy(), cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Document tokens')
plt.ylabel('Query tokens')
plt.title('ColBERT MaxSim Heatmap')
plt.tight_layout()
plt.savefig('colbert_heatmap.png')

# Show which doc tokens are best matches for each query token
max_indices = similarities.argmax(dim=1)  # (m,)
print("Best matching document tokens for each query token:")
for i, j in enumerate(max_indices):
    print(f"Query token {i} → Doc token {j} (score: {similarities[i, j]:.4f})")
```

---

## 🔗 실전 활용

| 시나리오 | 적용 | 노트 |
|---------|------|------|
| First-stage retrieval | ✅ 표준 | FAISS + MaxSim 으로 million-scale 가능 (ANN + MaxSim re-ranking) |
| Reranking | ✅ 표준 | Cross-encoder 보다 빠름, 정확도 거의 동일 |
| Dense RAG | ✅ 표준 | E5 또는 Contriever 대신 ColBERT 사용 가능 |
| Multilingual | ⚠️ 조건부 | multilingual-BERT 기반 ColBERT 필요 |
| Real-time QA | ✅ 가능 | Few-shot / zero-shot setting 에서 성능 좋음 |
| Streaming retrieval | ⚠️ 조건부 | Document 임베딩 업데이트 비용 (index rebuild) |

---

## ⚖️ 가정과 한계

1. **Query length 에 따른 MaxSim 편향**: Query 길이 많을수록 score 증가 — 정규화 필요 (또는 training 으로 완화).

2. **Document length 편향**: 긴 문서는 maxsim 이 자동으로 높음 (더 많은 token, 더 많은 "lucky match").
   - 해결: ColBERT 는 training 시 length-balanced negative sampling 사용.

3. **Token-level 상호작용만 포착**: Phrase-level 또는 sentence-level 의존성 무시.
   - 예: "France capital is Paris" 에서 "France" 와 "capital" 의 sequential relationship 미포착.

4. **Indexing 비용**: 각 document 의 모든 token embedding 저장 → dense retrieval 대비 더 많은 메모리.
   - DPR: 1M docs × 768-dim × 4 bytes = 3GB
   - ColBERT: 1M docs × avg 100 tokens × 128-dim × 1 byte (8-bit quantized) = 12GB (Ch3-03 PLAID 로 극적 개선)

5. **Query 시간 latency**: Per-query embedding 계산 여전히 필요 (DPR 와 동일).

---

## 📌 핵심 정리

$$
\boxed{S(q,d) = \sum_{i=1}^{m} \max_{j=1}^{n} e_{q_i}^\top e_{d_j}}
$$

| 특징 | 값 |
|------|-----|
| **Query representation** | Per-token (m tokens → m × 128 embeddings) |
| **Document representation** | Per-token (n tokens → n × 128 embeddings, indexed) |
| **Relevance score** | MaxSim (query-doc max interaction 합산) |
| **First-stage latency** | Sub-linear (ANN + MaxSim) — ~10-100ms for 1M docs |
| **Accuracy (nDCG@10)** | ~0.39 (DPR ~0.35, Cross-encoder ~0.42) |
| **Storage** | ~12GB for 1M docs (8-bit quantized) |
| **Key advantage** | Indexable interaction (offline embedding, online MaxSim) |

> **핵심**: ColBERT 는 cross-encoder 의 정확성을 per-token embedding + MaxSim 으로 근사하면서, indexing 으로 first-stage 검색 가능하게 만듦. Ch3-03 PLAID 는 이를 centroid + residual compression 으로 2.6× 더 압축.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: ColBERT 의 MaxSim 에서 query token 에 "stop word" (the, is, a) 가 포함되면 score 에 미치는 영향은?

<details>
<summary>해설</summary>

Stop word 는 대부분의 document 에서 존재하므로 high max score 획득 가능 (모든 doc embedding 이 유사) → MaxSim 을 부풀림. 이를 완화하려고:
1. Stop word 에 대한 embedding 가중치 낮춤 (training 시 가능)
2. Stop word 를 query 에서 제거 (전처리)
3. ColBERT 는 실제로 query 전처리에 의존하지 않지만, fine-tuning 때 negative samples 가 stop word 부풍향을 부분 보정
</details>

**문제 2 (심화)**: ColBERT 의 MaxSim 을 cross-encoder 의 full attention 으로 근사하려면, 어떤 조건이 필요한가?

<details>
<summary>해설</summary>

Cross-encoder: Attention weights $a_{ij} = \text{softmax}_j(\text{sim}(q_i, d_j))$ 로 weighted sum.
ColBERT: Max operator 로 취급.

근사 조건:
- Query token $i$ 의 attention 이 **매우 sharp** (한 개 document token 에만 집중) — 이 경우 max ≈ weighted sum.
- 실제로 높은 quality 의 embedding 에서 이런 일이 자주 발생 (예: "neural" query token → "neural" doc token 에 sharp attention).
- 하지만 ambiguous query 는 attention 이 diffuse → max operator 가 정보 손실.
</details>

**문제 3 (논문 비평)**: ColBERT 논문 (Khattab & Zaharia 2020) 에서 "document embedding 은 offline 이므로 billion-scale 에서 indexing 가능" 이라는 claim 에서, 실제 병목은?

<details>
<summary>해설</summary>

Claim 의 한계:
1. **Query embedding 은 여전히 online** — per-query latency 는 BERT forward pass (수십ms).
2. **MaxSim 계산 자체** — ANN 으로 후보 documents 제한 (e.g., top 1000 from IVF) 한 후 MaxSim 로 re-rank 해야 함 → 실제로는 2-stage (dense 1st + ColBERT 2nd).
3. **Memory bandwidth** — per-token embedding 조회 (random access) 는 sequential 메모리 access 보다 느림.

이후 개선 (Ch3-03 PLAID): centroid + residual compression 으로 memory 60% 감소 + latency 개선.
</details>

---

<div align="center">

[◀ 이전 (01. Cross-Encoder)](./01-cross-encoder.md) · [📚 README](../README.md) · [다음 ▶ (03. ColBERTv2 PLAID)](./03-colbertv2-plaid.md)

</div>
