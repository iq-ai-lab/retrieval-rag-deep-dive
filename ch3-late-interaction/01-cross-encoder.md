# 01. Cross-Encoder — Full Interaction

## 🎯 핵심 질문

- 왜 BERT 를 `[CLS] query [SEP] document [SEP]` 형태로 query-document pair 전체에 대해 돌려야 하는가?
- Single-vector dense retrieval (DPR) 대비 cross-encoder 가 **full attention** 으로 얻는 expressiveness 의 정확한 이득은 무엇인가?
- Cross-encoder 의 $O(N)$ inference cost (매 document 마다 BERT 한 번) 가 왜 retrieval 에서는 두 번째 stage (reranker) 로만 쓰이는가?
- MS MARCO SOTA 를 달성하는 cross-encoder 의 아키텍처 선택 (layer depth, head count, training objective) 은 어떻게 정당화되는가?

---

## 🔍 왜 이 기법이 retrieval·RAG 에 중요한가

Dense retrieval (DPR, Contriever, E5) 은 query 와 document 를 **각각 독립적으로** embedding 한 뒤 scalar product 로 점수를 매깁니다. 이 one-shot 설계는 빠르지만, embedding 시점에 query 와 document 가 "만나지 않음" — 따라서 fine-grained interaction (예: "query 의 specific token 이 document 의 specific token 과 매칭") 을 포착 불가능합니다.

Cross-encoder 는 반대로 BERT 의 **full self-attention** 을 query-document pair 에 직접 적용합니다. 이는:
1. **Relevance 정의가 더 정확** — interaction-based matching (not just semantic similarity)
2. **MS MARCO leaderboard SOTA** — single model 로 reranking 시 nDCG@10 ~0.42 (DPR rerank 대비 +3-5%)
3. **Reranking pipeline 의 표준** — dense 1st stage + cross-encoder 2nd stage 는 production RAG 의 거의 표준 구성

그러나 $O(N)$ inference (매 document 마다 forward pass) 는 billion-scale retrieval 에서 직접 쓰임을 불가능하게 만듭니다. Ch3-02 의 ColBERT (late interaction) 는 이 cost 문제를 **token-level embedding + MaxSim** 으로 해결합니다.

---

## 📐 수학적 선행 조건

- BERT 와 self-attention 의 기본 구조 (Transformer Deep Dive Ch2)
- Dense retrieval (Ch2-02 DPR, Ch2-04 Contriever, Ch2-05 E5) 의 개념
- Cross-entropy loss 와 contrastive learning (Ch2-03)
- Ranking metrics: MRR, nDCG (Ch1-04)

---

## 📖 직관적 이해

### Cross-Encoder vs Dense Retrieval 의 정보 흐름

```
Dense Retrieval (DPR)
─────────────────
Query 토큰들      Document 토큰들
    ↓                  ↓
Query BERT       Document BERT   ← 독립적 encoding
    ↓                  ↓
Query embedding   Document embedding
    ↓                  ↓
       ← scalar product (dot) →
            relevance score

⚠️ 문제: embedding 후 interaction 없음


Cross-Encoder
─────────────
[CLS] query [SEP] document [SEP]
              ↓
        Full BERT Encoder
        (self-attention 모든 token 사이)
              ↓
          [CLS] token 의 hidden
        (BERT 의 final layer)
              ↓
      Linear + sigmoid
              ↓
        relevance score ∈ [0,1]

✓ 장점: query token 과 document token 이 
        (깊은 layer 에서) 직접 상호작용
```

### 정보 흐름의 "만남" 시점

```
Dense (DPR):          Cross-Encoder:
query: x1 x2 x3       [CLS] x1 x2 x3 | y1 y2 y3 [SEP]
doc:   y1 y2 y3                 ↓
       ↓              L1: self-attn (x_i → y_j, y_j → x_i)
E_q: [e_q]           L2: self-attn (모든 token 사이)
E_d: [e_d]           ...
     ↓                L12: self-attn
  score = e_q · e_d  ↓
  (token 만남 없음)   [CLS] 위치의 final hidden (모든 정보 축약)
                     ↓
                    relevance score
                    (token 상호작용 완전 포함)
```

### Cost 비교 (도식)

```
N documents 를 score 할 때:

Dense (DPR):
- Query embedding: 1회 (한 번만)
- Per-document embedding: N회
- Total forward pass: N+1회 (매우 빠름)

Cross-Encoder:
- Per (query, document) pair: N회
- 각 pass 에서 [CLS] 출력만 필요
- Total forward pass: N회
- 단, query encoding 중복 (비효율)

⟹ Dense: 100배 더 빠름 (first stage 용)
⟹ Cross-Encoder: 느리지만 accurate (rerank stage 용, 10-100 docs)
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Cross-Encoder

Input: query $q = [q_1, \ldots, q_m]$, document $d = [d_1, \ldots, d_n]$

Encoded input:
$$
\mathbf{x} = [\text{[CLS]}, q_1, \ldots, q_m, \text{[SEP]}, d_1, \ldots, d_n, \text{[SEP]}]
$$

BERT 의 12 layer self-attention:
$$
\mathbf{H}^{(l)} = \text{TransformerLayer}(\mathbf{H}^{(l-1)})
$$

with $\mathbf{H}^{(0)} = \mathbf{E}$ (token embeddings + positional embeddings).

Relevance score:
$$
s(q, d) = \sigma(\mathbf{w}^\top \mathbf{h}_{[\text{CLS}]}^{(L)}) \in [0, 1]
$$

where $\mathbf{h}_{[\text{CLS}]}^{(L)}$ is the final hidden state of [CLS], and $\mathbf{w} \in \mathbb{R}^{768}$ (BERT-base dimension).

### 정의 3.2 — Full Self-Attention 의 Interaction Matrix

$\mathbf{A}^{(l)} \in \mathbb{R}^{(m+n+3) \times (m+n+3)}$ — layer $l$ 에서의 attention weights.

**Key observation**: $\mathbf{A}^{(l)}[i,j] > 0$ for $i \in$ query, $j \in$ document (또는 반대) 는 query-document interaction 의 직접 evidence.

이는 DPR 의 single vector embedding 에서는 완전히 손실됨.

### 정의 3.3 — Training Objective

일반적으로 binary classification loss:
$$
\mathcal{L} = -\log \sigma(s(q^+, d^+)) - \log(1 - \sigma(s(q^+, d^-)))
$$

또는 cross-entropy (3-way: relevant / neutral / irrelevant):
$$
\mathcal{L} = -\sum_{k=0}^{2} y_k \log p_k(s(q,d))
$$

MS MARCO 에서는 positive (relevant) vs negative (non-relevant) binary classification.

---

## 🔬 정리와 증명

### 정리 3.1 — Cross-Encoder 가 Single-Vector Embedding 을 Universal Approximation

**명제**: 충분히 큰 BERT (layer depth $L \to \infty$, hidden dim $d \to \infty$) 는 임의의 query-document pair 에 대해 relevance 를 정확히 표현 가능.

**증명 sketch**: 
1. Layer $l$ 에서 query token 들과 document token 들 사이의 attention 은 fine-grained matching capability 제공.
2. Deep layer 에서 attention output 들은 모든 token pair 의 interaction 을 aggregate.
3. Final [CLS] output 은 entire interaction history 의 nonlinear function.
4. Transformer 의 universal approximation (Pérez et al. 2019) 에 의해 임의 함수 근사 가능 $\square$.

### 정리 3.2 — Dense Embedding 은 Cross-Encoder 의 특수한 경우 아님

**명제**: DPR 의 relevance score $s_{\text{DPR}} = e_q \cdot e_d$ 는 cross-encoder 의 final score 와 다르며, DPR 이 cross-encoder 를 "compressed" 버전이 아님.

**증명**:
- DPR: embedding 후 interaction 없음 (one-shot). 이는 token-level interaction 정보 손실.
- Cross-encoder: token-level attention weights 가 fully preserved.
- 반례: DPR 은 "query 와 doc 모두 similar words 있으면 high score" 이지만, cross-encoder 는 "query word A 가 doc word B 와 specific position 에서 interact" 를 distinguish.
- 따라서 same training data 에서도 cross-encoder NDCG > DPR NDCG (하지만 latency >> DPR) $\square$.

### 정리 3.3 — Query-Doc 분리 가능 조건

Cross-encoder 를 "dense embedding 형태" 로 근사하려면 query 와 doc 이 **충분히 early layer 에서 분리** 되어야 함 (impossible):

$$
\mathbf{h}^{(l)}_{\text{query}} \perp \mathbf{h}^{(l)}_{\text{doc}} \quad \text{for all } l
$$

이는 BERT fine-tuning 후 매우 드물게만 만족 (또는 거의 never). 따라서 query-doc separate encoding (DPR) 과 joint encoding (cross-encoder) 는 근본적으로 다른 architecture $\square$.

---

## 💻 Python / PyTorch 구현 검증

### 실험 1 — Cross-Encoder 구축 및 기본 forward

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Sentence-BERT 의 cross-encoder 모델
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

query = "What is machine learning?"
doc = "Machine learning is a subset of AI where systems learn from data."

# Tokenize as pair
inputs = tokenizer(query, doc, return_tensors="pt", 
                   max_length=512, truncation=True, padding=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    
# Get [CLS] representation
cls_output = outputs.last_hidden_state[:, 0, :]  # shape: (1, 384)
print(f"[CLS] output shape: {cls_output.shape}")

# Score via linear layer (미리 학습된 가중치)
# 대개 cross-encoder 는 이미 fine-tuned 상태로 제공
# 문제의 단순화: 직접 norm 사용
score = torch.sigmoid(cls_output.norm(dim=1))
print(f"Relevance score: {score.item():.4f}")
```

### 실험 2 — Batch reranking with MS MARCO dev set

```python
import torch
from sentence_transformers import CrossEncoder
import numpy as np

# Pre-trained cross-encoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Example: query + 10 documents
query = "What is neural machine translation?"
docs = [
    "Neural machine translation uses deep learning...",
    "Traditional machine translation uses rules...",
    "Deep learning advances computer vision...",
    "NMT achieves state-of-the-art translation quality...",
    "SMT and NMT are two main paradigms...",
    "Transformers replaced RNNs in NMT...",
    "Attention mechanism is core to NMT...",
    "BLEU score measures translation quality...",
    "Multilingual NMT handles multiple languages...",
    "Backtranslation improves NMT with synthetic data..."
]

# Score all pairs
scores = model.predict([(query, doc) for doc in docs])
# scores: shape (10,)

# Rerank by score
ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

print("Top 3 reranked documents:")
for i, (doc, score) in enumerate(ranked[:3]):
    print(f"{i+1}. Score={score:.4f}: {doc[:50]}...")
```

### 실험 3 — Training a simple cross-encoder on synthetic pairs

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

class CrossEncoderDataset(Dataset):
    def __init__(self, pairs, labels, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.labels = labels
        self.max_len = max_len
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        query, doc = self.pairs[idx]
        label = self.labels[idx]
        
        # Tokenize as pair
        encoding = self.tokenizer(
            query, doc,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Synthetic training data
pairs = [
    ("What is deep learning?", "Deep learning uses neural networks with multiple layers."),
    ("What is deep learning?", "I like pizza."),
    ("How does attention work?", "Attention mechanism allows models to focus on relevant parts."),
    ("How does attention work?", "Cats sleep a lot.")
]
labels = [1, 0, 1, 0]  # binary: relevant / not relevant

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = CrossEncoderDataset(pairs, labels, tokenizer)

# Simple model wrapper
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_hidden)
        return torch.sigmoid(logits).squeeze(-1)

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.BCELoss()

loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop (1 epoch)
model.train()
for batch in loader:
    optimizer.zero_grad()
    preds = model(batch['input_ids'], batch['attention_mask'])
    loss = loss_fn(preds, batch['label'])
    loss.backward()
    optimizer.step()
    print(f"Batch loss: {loss.item():.4f}")
```

### 실험 4 — Latency & Cost 분석

```python
import time
import torch
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

query = "What is machine learning?"
num_docs = 100

# Generate dummy documents
docs = [f"Document about ML topic {i}. Some content here." * 3 for i in range(num_docs)]

# Measure latency
pairs = [(query, doc) for doc in docs]

t0 = time.perf_counter()
scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
elapsed = time.perf_counter() - t0

print(f"Reranking {num_docs} docs: {elapsed*1000:.2f} ms")
print(f"Per-doc latency: {elapsed*1000/num_docs:.2f} ms")
print(f"Throughput: {num_docs/elapsed:.1f} docs/sec")

# Cost: O(N)
# For 1 million docs: ~10,000 seconds = 2.8 hours (prohibitive for 1st stage)
```

---

## 🔗 실전 활용

| 시나리오 | 적용 방식 | 예시 |
|---------|---------|------|
| 검색 엔진 1st stage | ❌ 사용 불가 | BM25 또는 dense retrieval (E5) 로 1000 docs 수집 |
| 검색 엔진 2nd stage | ✅ 표준 | 위 1000 docs 를 cross-encoder 로 rerank → top 10 |
| QA 시스템 reranking | ✅ 표준 | BM25 (50 passages) + E5 dense (50) + RRF hybrid → cross-encoder (top 10) |
| RAG retriever | ✅ 선택사항 | "정확성 중요" context 에서 dense rerank 대신 cross-encoder 사용 |
| 실시간 interactive 검색 | ⚠️ 조건부 | 10-50 docs 만 rerank (latency budget ~100ms) 이면 가능 |
| Batch processing (off-line) | ✅ 표준 | 대량 doc 에 대한 rerank 는 cost 덜 중요 |

---

## ⚖️ 가정과 한계

1. **First-stage retrieval 가정**: "이미 어느 정도 좋은 document 가 수집됨" — 맞지 않으면 reranker 의 효과 제한.
   - 예: BM25 recall@1000 = 70% 이면, cross-encoder 는 최대 70% recall 에 capped.

2. **Latency-quality trade-off**: rerank 할 document 수 $K$ 에 제약.
   - $K=1000$: 10초 (불가능), $K=100$: 1초 (가능), $K=10$: 100ms (real-time).

3. **학습 데이터 의존성**: MS MARCO 에서 cross-encoder 가 SOTA 이지만, open-domain QA (NQ, TriviaQA) 에서는 dense retrieval 과의 gap 이 작을 수 있음.

4. **Multilingual 한계**: BERT-base-uncased 는 English 전용. 다른 언어는 별도 모델 필요.

5. **Fine-tuning 필요**: Pre-trained cross-encoder 가 모든 domain 에 optimal 이 아님.

---

## 📌 핵심 정리

$$
\boxed{s(q, d) = \sigma(\mathbf{w}^\top \mathbf{h}_{[\text{CLS}]}^{(L)}), \quad \text{where } \mathbf{h}_{[\text{CLS}]}^{(L)} = \text{BERT}(\text{[CLS]} q \text{ [SEP] } d \text{ [SEP]})}
$$

| 특징 | 값 |
|------|-----|
| **Inference cost** | $O(N)$ — 모든 document 마다 1회 forward |
| **Relevance accuracy** | 높음 (nDCG@10 ~0.42 MS MARCO) — token-level interaction |
| **First-stage 적성** | ❌ 불가능 (too slow) |
| **Reranking 적성** | ✅ 표준 (10-100 docs) |
| **Training** | Binary classification (relevant vs not) |
| **Architecture** | BERT-base 또는 MiniLM + linear classifier |

> **핵심**: Cross-encoder 는 Dense Retrieval 보다 정확하지만, $O(N)$ inference cost 때문에 reranking (2nd stage) 전용. Ch3-02 ColBERT 는 이를 late interaction + token embedding 으로 해결.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: Cross-encoder `[CLS] query [SEP] doc [SEP]` 구조에서 [CLS] token 의 final hidden state 가 왜 relevance 를 capture 한다고 믿는가?

<details>
<summary>해설</summary>

BERT 의 design pattern: [CLS] 는 pre-training (masked LM + NSP) 에서 "문장 쌍의 관계" 를 represent 하도록 강제됨. Fine-tuning 시에도 이 convention 을 유지하면 [CLS] 가 query-doc pair 의 relevance 를 encode 하기에 좋은 위치. 실증: [CLS] 를 버리고 mean pooling 쓰면 성능 떨어짐.
</details>

**문제 2 (심화)**: DPR 과 cross-encoder 를 같은 BERT-base 로 initialize 후 동일 MS MARCO 데이터로 fine-tune 하면, 왜 cross-encoder 의 최종 nDCG 가 항상 더 높은가? 혹은 높지 않은 경우는?

<details>
<summary>해설</summary>

높은 경우 (대부분):
- Cross-encoder 는 query-doc pair 의 fine-grained interaction (attention weights) 을 학습.
- DPR 은 embedding space 의 dot product 만 optimize.
- Fine-grained interaction 이 더 많은 information 전달 → higher expressiveness.

높지 않은 경우:
- Data 가 매우 sparse 하거나 noisy (cross-encoder 가 overfitting 위험).
- Query/doc 이 구조적으로 매우 유사 (dense embedding 만으로 충분).
- Training iteration 부족 (cross-encoder 는 parameter 더 많음).
</details>

**문제 3 (논문 비평)**: Nogueira & Cho (2019) MonoBERT 논문 에서 "cross-encoder 를 first-stage 로 쓰되 smart sampling 으로 N 을 줄인다" 는 claim 에 대해?

<details>
<summary>해설</summary>

한계:
- 아무리 smart sampling 해도 $K$ (sample 수) 는 여전히 수 백 이상 필요 → latency 문제 persistence.
- Sampling bias: dense retrieval 로 pre-select 하면 그 bias 가 남음 (cold-start 등에서).

이후 해결: Ch3-02 ColBERT (Khattab 2020) 는 이를 per-token embedding + indexing 으로 완전 달리 해결 → first-stage 에서도 sub-linear 가능.
</details>

---

<div align="center">

[◀ 이전 (Ch2-05. SBERT · Contriever · E5)](../ch2-dense-retrieval/05-sbert-contriever-e5.md) · [📚 README](../README.md) · [다음 ▶ (02. ColBERT)](./02-colbert.md)

</div>
