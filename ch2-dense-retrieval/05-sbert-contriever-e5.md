# 05. SBERT · Contriever · E5 — Supervised → Unsupervised → Weakly-Supervised

## 🎯 핵심 질문

- **SBERT (Sentence-BERT, Reimers & Gupta 2019)** 는 어떻게 Siamese BERT 로 supervised contrastive learning 을 처음 실현했는가? NLI/STS 데이터의 역할은?
- **Contriever (Izacard et al. 2021)** 는 왜 labeled data 없이 (unsupervised) 학습 가능한가? Random cropping + MoCo queue 의 메커니즘은?
- **E5 (Wang et al. 2022)** 의 "weakly-supervised" 는 정확히 무엇인가? Web corpus 와 instruction prefix 의 역할은?
- 세 방법의 진화 축 — supervised → unsupervised → weakly-supervised — 이 정보 검색 연구의 어떤 개선을 대표하는가?

---

## 🔍 왜 이 세 가지가 Dense Retrieval 의 진화를 대표하는가

Dense retrieval 의 주요 과제는 **labeled retrieval 데이터의 부족**입니다. DPR 과 ANCE 는 모두 NQ, TriviaQA, MS MARCO 같은 labeled 데이터에 의존합니다.

SBERT → Contriever → E5 의 진화는 이 문제를 점진적으로 해결합니다:

1. **SBERT (2019)**: Supervised — NLI (Natural Language Inference) + STS (Semantic Textual Similarity) 데이터로 학습. 일반적이지만 domain-specific 데이터 필요.

2. **Contriever (2021)**: Unsupervised — Document 의 random cropping 으로 자동 positive pair 생성. MoCo (Momentum Contrast) 로 안정적 training. Labeled data 완전 불필요.

3. **E5 (2022)**: Weakly-Supervised — Web corpus (검색 log, passages) 와 instruction prefix ("search_query:" vs "search_document:") 로 pseudo-labeling. Unsupervised 와 supervised 의 장점 결합.

4. **이후 발전**: BGE, UAE, jina-embeddings 등이 E5 의 패러다임 (weak supervision + instruction) 을 확장.

이 문서는 세 방법의 수학적 정의, 학습 메커니즘, 그리고 성능 비교를 다룹니다.

---

## 📐 수학적 선행 조건

- Siamese architecture
- Contrastive learning (Ch3)
- Momentum update, exponential moving average
- Data augmentation (cropping, noise)

---

## 📖 직관적 이해

### 세 가지 학습 패러다임의 Data Requirement

```
[ SBERT (Supervised) ]
Labeled triplets: (anchor, positive, negative)
Data source:
  - NLI: entailment/contradiction 쌍
  - STS: 0-5 점수의 similarity 쌍
  - 요구: 수만 개의 high-quality labeled pairs
예: ("The cat sat on the mat", "A cat is sitting", 0.9 similarity)

[ Contriever (Unsupervised) ]
Unlabeled documents: p_1, p_2, ...
Self-supervision:
  - random crop (1단계): p_i → p_i^{crop1}, p_i^{crop2}
  - 두 crop 이 같은 document → positive pair
  - 다른 document 의 crop → negative
  - 요구: 순수한 document 텍스트 (label 불필요)
예: ("The cat sat on the mat" 의 첫번째 절반, 두번째 절반) → 자동 positive

[ E5 (Weakly-Supervised) ]
Web corpus + pseudo-labeling:
  - Query-document 쌍 (검색 log): (search query, clicked document)
  - Instruction prefix: query 는 "search_query:" 로, document 는 "search_document:" 로
  - 모델은 prefix 로부터 역할 (query vs doc) 학습
  - 요구: Web 규모 데이터 (but no manual labeling)
예: ("search_query: 자동차 구매", "search_document: 자동차 구입 가이드", 1.0 match)
```

### SBERT vs Contriever vs E5 의 아키텍처

```
[ SBERT ]
Input: (sentence1, sentence2)
         ↓ BERT encoder
    (emb1, emb2)
         ↓ mean pooling
    (vec1, vec2)
         ↓ cosine similarity + loss
    Label: 0-5 similarity score

[ Contriever ]
Input: document → two random crops
      p_i → [crop1, crop2]
         ↓ same encoder
    (emb1, emb2)  ← different forward passes
         ↓ momentum contrast (queue)
    comparison with queue of negatives
         ↓ contrastive loss
    crop1 와 crop2 는 positive (같은 doc)
    queue 의 다른 doc crops → negatives

[ E5 ]
Input: (query, document)
    "search_query: 자동차 구매"
    "search_document: 자동차 구입 방법"
         ↓ same BERT encoder (shared)
    (query_emb, doc_emb)
         ↓ cosine similarity + loss
    Label: Web 에서 co-occurrence (clicked) → positive
    모델은 prefix 로 역할 구별
```

### 데이터 효율성 비교

```
SBERT:
  - 학습 데이터: STS (~5.7K pairs) + NLI (~570K triplets)
  - 의존: 수동 라벨링
  - 장점: Domain-specific fine-tuning 가능
  - 단점: 새로운 도메인 = 새로운 라벨링 필요

Contriever:
  - 학습 데이터: English Wikipedia + CC-News (document 만 필요)
  - 자동 pseudo-positive: random cropping
  - 장점: 라벨링 불필요, 대규모 가능
  - 단점: Document 내부 cropping → query-document 관계 배우지 못함

E5:
  - 학습 데이터: Web corpus + search log (query-doc 쌍)
  - 자동 라벨링: co-occurrence (clicked)
  - 장점: Query-document 관계 학습 + 대규모
  - 단점: Web 데이터의 noise (모든 click 이 relevant 아님)
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — SBERT (Siamese BERT)

**Architecture**:
- 동일한 BERT 인코더 $f$ 를 두 개의 입력에 적용
- $\mathbf{u} = f(u) \in \mathbb{R}^d$ (sentence 1)
- $\mathbf{v} = f(v) \in \mathbb{R}^d$ (sentence 2)

**Pooling**: Mean pooling over token embeddings:
$$
\mathbf{u} = \frac{1}{n} \sum_{i=1}^{n} h_i
$$

**Loss** (여러 가지 가능):

1. **Cosine-similarity + MSE** (STS 데이터):
$$
L = \text{MSE}(\cos(\mathbf{u}, \mathbf{v}), y)
$$
여기서 $y \in [0, 1]$ 는 similarity label.

2. **Triplet Loss** (NLI 데이터):
$$
L = \max(0, \|(\mathbf{a} - \mathbf{p})\|_2 - \|(\mathbf{a} - \mathbf{n})\|_2 + \epsilon)
$$
여기서 $a$ = anchor, $p$ = positive, $n$ = negative, $\epsilon$ = margin.

### 정의 5.2 — Contriever (Unsupervised Learning)

**Self-supervised positive pairs**:

Document $p$ 에서 random cropping (비율 $r$) 으로 두 버전:
$$
p^{(1)} = \text{crop}(p, r), \quad p^{(2)} = \text{crop}(p, r)
$$

**Embedding 계산**:
$$
\mathbf{p}^{(1)} = f_{\theta}(p^{(1)}), \quad \mathbf{p}^{(2)} = f_{\phi}(p^{(2)})
$$

여기서 $f_{\phi}$ 는 momentum encoder: $\phi = \tau \phi + (1-\tau) \theta$ (exponential moving average, $\tau \approx 0.999$).

**MoCo Loss** (Momentum Contrast):
$$
L = -\log \frac{\exp(\mathbf{p}^{(1)} \cdot \mathbf{p}^{(2)} / \tau)}{\exp(\mathbf{p}^{(1)} \cdot \mathbf{p}^{(2)} / \tau) + \sum_{k \in \text{queue}} \exp(\mathbf{p}^{(1)} \cdot \mathbf{k} / \tau)}
$$

여기서 queue $\mathcal{Q}$ 는 이전 iterations 의 negative embeddings (size = 65536).

**핵심**: Document 내부 cropping 으로 자동 positive; queue 의 다른 documents 는 hard negatives (같은 batch 내 비슷한 docs 아님).

### 정의 5.3 — E5 (Text Embedding by Contrastive Learning)

**Input Format** (instruction prefix):
$$
\text{Query: "search\_query: } q \text{"} \\
\text{Document: "search\_document: } p \text{"}
$$

**Shared Encoder**:
$$
\mathbf{q} = f([\text{search\_query:} | q]), \quad \mathbf{p} = f([\text{search\_document:} | p])
$$

**Weakly-supervised Labels**:
- Positive: Web corpus 에서 co-occurrence (clicked pairs)
- 전체 Web 로부터 $(q, p^+)$ 쌍 수집

**Loss** (InfoNCE with in-batch negatives):
$$
L = -\log \frac{\exp(s(\mathbf{q}, \mathbf{p}^+) / \tau)}{\exp(s(\mathbf{q}, \mathbf{p}^+) / \tau) + \sum_{j \neq i} \exp(s(\mathbf{q}, \mathbf{p}_j^+) / \tau)}
$$

여기서 $\mathbf{p}_j^+$ (batch 내 다른 query 의 positive documents) 를 in-batch negatives 로 사용.

---

## 🔬 정리와 증명

### 정리 5.1 — Contriever 의 Random Cropping 으로의 Positive Pair 자동 생성

Document 길이 $n$, cropping ratio $r$ (보통 $r = 0.2$ ~ $0.5$) 일 때:

$$
P(\text{두 crop 이 meaningful overlap 가지나}) \geq 1 - O(1/n) \quad \text{as } n \to \infty
$$

**증명 sketch**: 길이 $n$ 의 document, 각 crop 은 길이 $rn$. Overlap 이 0 일 확률:
$$
P(\text{no overlap}) = \frac{(1-r)n}{n} \to 1-r > 0 \quad \text{(small)}
$$

충분히 긴 document (보통 Wikipedia, $n > 100$) 에서는 거의 항상 meaningful content 공유 $\square$.

### 정리 5.2 — Momentum Encoder 의 Stability

Momentum encoder $\phi_t$ 의 update:
$$
\phi_t = \tau \phi_{t-1} + (1 - \tau) \theta_t
$$

$\tau = 0.999$ 일 때, momentum encoder 는 현재 encoder $\theta_t$ 로부터 약:
$$
\text{lag} \approx \frac{1}{1-\tau} = 1000 \text{ iterations}
$$

따라서 **queue 에 저장된 embeddings 은 매우 stale** (1000 iterations 이전). 하지만:

$$
\mathbb{E}[\text{embedding 변화 over 1000 iters}] = O(\sqrt{1000 \cdot \text{learning rate}}) \ll 1
$$

Smooth learning trajectory 이면 staleness 무시할 수 있음 $\square$.

### 정리 5.3 — E5 의 Instruction-based Role Awareness

Instruction prefix "search_query:" vs "search_document:" 가 있을 때, BERT 는 자동으로:

1. Query-specific tokens 에 더 높은 weight (query encoder processing)
2. Document-specific tokens 에 더 높은 weight (document encoder processing)

**증명**: 대규모 데이터에서 prefix 와 역할의 상관이 강하면, attention mechanism 이 prefix 에 응답하여 representation 을 분화. Fine-grained 증명은 attention probe 실험 (Jawahar et al.) 로 검증됨.

**결론**: Single encoder 여도, instruction 만으로 query/document 를 구분 가능 $\square$.

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — SBERT 학습 (Triplet Loss)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class SBERTTripletModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, sentences):
        """
        sentences: list of strings
        output: [len(sentences), hidden_dim]
        """
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True,
                               truncation=True, max_length=128)
        outputs = self.bert(**inputs)
        # Mean pooling
        embeddings = (outputs.last_hidden_state * 
                     inputs['attention_mask'].unsqueeze(-1)).sum(dim=1) / \
                    inputs['attention_mask'].sum(dim=1, keepdim=True)
        return embeddings

def triplet_loss(anchor, positive, negative, margin=0.5):
    """Triplet loss: L = max(0, d(a, n) - d(a, p) + margin)"""
    dist_ap = torch.norm(anchor - positive, p=2, dim=1)
    dist_an = torch.norm(anchor - negative, p=2, dim=1)
    loss = torch.clamp(dist_an - dist_ap + margin, min=0).mean()
    return loss

# Training example
model = SBERTTripletModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Sample data
anchors = ["The cat sat on the mat"]
positives = ["A cat is sitting"]
negatives = ["The dog ran outside"]

for epoch in range(10):
    model.train()
    
    anchor_embs = model(anchors)
    pos_embs = model(positives)
    neg_embs = model(negatives)
    
    loss = triplet_loss(anchor_embs, pos_embs, neg_embs, margin=0.5)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### 실험 2 — Contriever 의 Random Cropping + MoCo

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrieveModel(nn.Module):
    def __init__(self, hidden_dim=768, queue_size=65536):
        super().__init__()
        self.encoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Momentum encoder
        self.momentum_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.tau = 0.999
        self._init_momentum_encoder()
        
        # MoCo queue
        self.register_buffer('queue', torch.randn(queue_size, hidden_dim))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        # Normalize queue
        self.queue = F.normalize(self.queue, dim=1)
    
    def _init_momentum_encoder(self):
        for param_q, param_k in zip(self.encoder.parameters(), 
                                    self.momentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Exponential moving average update"""
        for param_q, param_k in zip(self.encoder.parameters(),
                                   self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.tau + param_q.data * (1 - self.tau)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace oldest
        if ptr + batch_size <= len(self.queue):
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Wrap around
            remain = len(self.queue) - ptr
            self.queue[ptr:] = keys[:remain]
            self.queue[:batch_size - remain] = keys[remain:]
        
        ptr = (ptr + batch_size) % len(self.queue)
        self.queue_ptr[0] = ptr
    
    def forward(self, x_q, x_k, temperature=0.07):
        """
        x_q: query version (main encoder)
        x_k: key version (momentum encoder)
        """
        # Query features
        q = F.normalize(self.encoder(x_q), dim=1)
        
        # Key features
        with torch.no_grad():
            self._update_momentum_encoder()
            k = F.normalize(self.momentum_encoder(x_k), dim=1)
            self._dequeue_and_enqueue(k)
        
        # MoCo loss
        # Positive: inner product of q and k
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(1)  # [B, 1]
        
        # Negative: with all queue elements
        l_neg = torch.einsum('nc,kc->nk', q, self.queue)  # [B, queue_size]
        
        # Logits: [B, 1+queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1) / temperature
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

# Simulation
model = ContrieveModel(hidden_dim=768, queue_size=1024)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
for step in range(100):
    # Random cropping (simulated by different input)
    x_q = torch.randn(32, 768)  # crop 1
    x_k = torch.randn(32, 768)  # crop 2 (from same doc, but simulated here)
    
    loss = model(x_q, x_k)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 20 == 0:
        print(f"Step {step}: MoCo loss = {loss.item():.4f}")
```

### 실험 3 — E5 with Instruction Prefix

```python
from sentence_transformers import SentenceTransformer
import torch

# E5 모델 로드 (실제 사용)
model = SentenceTransformer('intfloat/e5-base')

# Query and document with instruction prefix
queries = [
    "search_query: 자동차를 어떻게 사나?",
    "search_query: BERT 모델은 무엇인가"
]

documents = [
    "search_document: 자동차 구매 완벽 가이드",
    "search_document: BERT: Pre-training of Deep Bidirectional Transformers",
    "search_document: 자동차 유지 비용"
]

# Encode
query_embeddings = model.encode(queries, convert_to_tensor=True)
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# Compute similarity
similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T)

print("Similarity matrix:")
print(similarity_matrix)
print("\nFor query 0 (자동차), top doc: ", similarity_matrix[0].argmax())
print("For query 1 (BERT), top doc: ", similarity_matrix[1].argmax())
```

### 실험 4 — 세 가지 방법 성능 비교

```python
import numpy as np
from sentence_transformers import SentenceTransformer, util

# 세 가지 모델
sbert = SentenceTransformer('all-MiniLM-L6-v2')  # SBERT-like
contriever = SentenceTransformer('facebook/contriever')  # Contriever
e5 = SentenceTransformer('intfloat/e5-base')  # E5

# Test queries
test_queries = [
    "How to buy a car",
    "What is BERT",
    "Python programming"
]

# Test corpus
test_corpus = [
    "Guide to purchasing vehicles",
    "BERT: Bidirectional Encoder Representations from Transformers",
    "Introduction to Python programming",
    "Car maintenance tips",
    "Deep learning tutorial"
]

models = {'SBERT': sbert, 'Contriever': contriever, 'E5': e5}
results = {}

for model_name, model in models.items():
    print(f"\n=== {model_name} ===")
    
    if model_name == 'E5':
        # E5 requires instruction prefix
        query_embeddings = model.encode(
            [f'search_query: {q}' for q in test_queries],
            convert_to_tensor=True
        )
        doc_embeddings = model.encode(
            [f'search_document: {d}' for d in test_corpus],
            convert_to_tensor=True
        )
    else:
        query_embeddings = model.encode(test_queries, convert_to_tensor=True)
        doc_embeddings = model.encode(test_corpus, convert_to_tensor=True)
    
    # Compute similarities
    similarity = util.cos_sim(query_embeddings, doc_embeddings)
    
    for i, query in enumerate(test_queries):
        top_k_idx = torch.argsort(similarity[i], descending=True)[:3]
        print(f"Query: '{query}'")
        for j, idx in enumerate(top_k_idx):
            print(f"  {j+1}. {test_corpus[idx]} (score: {similarity[i, idx]:.4f})")
```

---

## 🔗 실전 활용

| 상황 | 추천 모델 | 이유 |
|------|---------|------|
| Domain-specific fine-tuning (NLI/STS 데이터 있음) | SBERT | Triplet loss / MSE loss 로 fine-tune 가능 |
| Zero-shot (labeled data 없음) | Contriever | Unsupervised, 대규모 unlabeled corpus 에서 학습 |
| 대규모 web-scale 적용 | E5 | Weakly-supervised, Web corpus 에서 자동 학습 |
| 다언어 검색 | E5-multilingual | Instruction 기반 구조 다언어 확장 용이 |
| 매우 작은 데이터셋 | SBERT fine-tuned | Supervised signal 이 작을 때 안정적 |
| 계속 증가하는 corpus | Contriever | Re-indexing 없이도 새 documents 추가 가능 |

---

## ⚖️ 가정과 한계

- **SBERT**: NLI/STS 데이터에 의존 — domain shift 시 성능 하락. Triplet loss 는 hyperparameter sensitive.

- **Contriever**: Random cropping 은 long documents 에서만 meaningful (짧은 text 는 overlap 부족). Document 내 중요 정보 위치에 따라 성능 편차.

- **E5**: Web corpus 의 noise (모든 click 이 relevant 아님) — 자동 라벨 품질 낮을 수 있음. Instruction prefix 는 모델의 capacity 사용 (일부 layer 가 prefix processing 에 투여).

- **일반화**: 세 모델 모두 English-centric data 에서 학습 — cross-lingual transfer 성능 하락.

---

## 📌 핵심 정리

| 방법 | Data Type | Learning | 데이터 규모 | Domain Shift |
|-----|----------|---------|-----------|-------------|
| SBERT | Labeled (NLI/STS) | Supervised | ~600K pairs | Fine-tune 필요 |
| Contriever | Unlabeled (docs) | Unsupervised | ~millions | 강함 (zero-shot) |
| E5 | Weakly-labeled (web) | Weak Supervision | ~billions | 강함 (general) |

$$
\boxed{\text{SBERT (supervised)} \to \text{Contriever (unsupervised)} \to \text{E5 (weak)} = \text{Label 의존성 감소}}
$$

> **핵심**: Dense retrieval 의 진화 = **labeled data 의존성 제거 + scale-up capability 확대**. E5 이후, 사실상 모든 dense retrieval 은 weak supervision 기반.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: Contriever 의 random cropping 에서, 길이 100 tokens 의 document 를 50% cropping rate 로 자르면, 두 crop 이 완전히 다른 문장일 확률은?

<details>
<summary>해설</summary>

완전히 다를 확률은 매우 낮음 (거의 0). 왜냐하면:
- 첫 crop: tokens 1-50
- 두 번째 crop: 매 iteration 다른 위치 (1-50, 25-75, 50-100 등)
- 100 tokens 에서 두 개의 50-token 구간이 겹치지 않을 확률: 거의 0 (최대 50개 구간, 대부분 overlap)

따라서 Contriever 의 가정 (같은 doc 의 두 crop = positive) 은 충분히 valid. 실제로 길이 짧은 text (<100 tokens) 에서는 random cropping 이 덜 효과적.
</details>

**問題 2 (심화)**: E5 의 instruction prefix ("search_query:" vs "search_document:") 가 없다면, 동일한 encoder 로 query 와 document 를 구분할 수 있을까?

<details>
<summary>해설</summary>

이론적으로는 불가능 (information 부족). 실제로:
- Prefix 없이: query 와 document embedding 이 같은 space 에서 뒤섞일 가능성
- Prefix 있이: BERT 의 attention mechanism 이 "search_query:" 를 보고, query-specific representation 생성
- Ablation study (E5 논문): prefix 제거 시 ~2-3% 성능 하락

따라서 prefix 는 **명시적 신호** (role indicator) 로서 중요. 다른 방법: query 와 doc 를 separate encoder 로 (DPR 처럼) 하면 prefix 필요 없음.

E5 의 선택 (single encoder + prefix) 는:
- 메모리 효율 (parameter 절반)
- Inference cost 낮음 (하나의 encoder weight 만 load)
- But: query/doc 분화 약함
</details>

**問題 3 (논문 비평)**: Contriever 는 document cropping 으로 학습하는데, DPR 은 label 있는 QA pairs 로 학습한다. E5 가 둘 다 능가하는 이유는?

<details>
<summary>해설</summary>

세 가지 이유:

1. **Scale**: E5 는 Web corpus (수십억 pairs) 에서 학습 → DPR (NQ/TriviaQA ~600K) 보다 10000배 큼. Scale advantage 명백.

2. **Task alignment**: E5 는 query-document 관계 학습 (QA 와 유사). Contriever 는 document 내부 cropping (query 와 무관) → query 의 특성 (질문체 등) 을 모를 수 있음.

3. **Weak signal의 로버스트성**: Web click 은 noisy (false positives/negatives) 하지만, 대규모 → averaging out effect. DPR 의 labeled data 는 clean 하지만 작음.

**결론**: E5 = (Contriever 의 scale) + (DPR 의 task alignment) + (weak label 의 효율) = 최고 성능. 이후 모든 dense retrieval 은 E5 스타일 weak supervision 을 표준으로 채택.
</details>

---

<div align="center">

[◀ 이전 (04. Hard Negative Mining)](./04-hard-negatives.md) · [📚 README](../README.md) · [다음 ▶ (Ch3-01. Cross-Encoder)](../ch3-late-interaction/01-cross-encoder.md)

</div>
