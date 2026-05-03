# 01. Vanilla RAG (Lewis 2020)

## 🎯 핵심 질문

- 왜 생성 모델(LLM)에 외부 지식을 직접 연결해야 하는가? — Parametric knowledge vs non-parametric knowledge 의 trade-off
- RAG 의 핵심 공식 $p(y|x) = \sum_z p(y|z,x) p(z|x)$ 에서 retriever $p(z|x)$ 와 generator $p(y|z,x)$ 를 어떻게 공동 학습하는가?
- RAG-Sequence vs RAG-Token 의 수학적·실전적 차이는?
- Vanilla RAG 가 이후 RETRO·REALM·Self-RAG 의 출발점인 이유는?

---

## 🔍 왜 이 기법이 retrieval·RAG 에 중요한가

LLM 의 knowledge cutoff 와 hallucination 문제를 푸는 가장 직관적 방법: **retrieve 한 문서 조각(passage)을 prompt 에 concat 하고, 그것을 바탕으로 답변 생성**. 
- 2020년 Lewis 논문 이전: retrieval 과 generation 이 완전히 분리 (dense retriever + seq2seq generator, pipeline 방식)
- Vanilla RAG: end-to-end 학습으로 두 모듈의 joint optimization
- 이후 대부분의 RAG 기법은 Vanilla RAG 의 retriever/generator architecture 를 상속하면서, **retrieval 품질 개선** 또는 **generator 의 adaptive 선택** 으로 발전

실제 LangChain, LlamaIndex 등 실무 RAG 시스템은 Vanilla RAG 의 구조를 기본으로 확장.

---

## 📐 수학적 선행 조건

- 기본 확률론: conditional probability, marginal likelihood
- 정보 검색 기초: dense retrieval, DPR (Ch4-02)
- Seq2seq 모델: encoder-decoder (BART, T5)
- 최적화: EM algorithm (optional, EM 으로도 유도 가능)

---

## 📖 직관적 이해

### Vanilla RAG 의 구조도

```
입력 x ("Paris 의 대통령은?")
  ↓
┌─────────────────────────────────────┐
│ Retriever (DPR): p(z|x)              │
│ - BERT encoder 로 query/doc encode   │
│ - dense vector 비교 (cosine sim)     │
│ - top-k passage 선택                 │
└─────────────────────────────────────┘
  ↓
z_1, z_2, ..., z_k (예: "Paris is...")
  ↓
┌─────────────────────────────────────┐
│ Generator: p(y|z,x) = p(y|z+x)       │
│ - BART/T5 decoder                    │
│ - concat(x, z_i) 를 처리             │
│ - 답변 token 순차 생성               │
└─────────────────────────────────────┘
  ↓
y ("Emmanuel Macron")
```

### 학습 목표: 경계 우도(marginal likelihood)

```
원본 목표 (generator only):
p(y|x) = P(y | context x)

Vanilla RAG 목표 (marginal):
p(y|x) = Σ p(y|z,x) × p(z|x)
         z∈Z
         
의미: 모든 가능한 passage z 를 고려한, 
      weighted average generation 확률
      
최적: retriever 가 정답에 관련 z 를 높은 확률로 선택
     → 해당 z 와 조합 시 generator 가 정답 y 를 높은 확률로 생성
```

---

## ✏️ 엄밀한 정의

### 정의 5.1.1 — RAG 모델 구성

**Retriever**: DPR (Dense Passage Retrieval)
$$
p(z|x) = \frac{\exp(\alpha_1(x) \cdot \alpha_2(z)) / \tau}{\sum_{z' \in Z} \exp(\alpha_1(x) \cdot \alpha_2(z')) / \tau}
$$

- $\alpha_1$: query encoder (BERT), $\alpha_2$: passage encoder (BERT)
- $\tau$: temperature (보통 1)
- $Z$: corpus 의 모든 passage

**Generator**: seq2seq (BART)
$$
p(y|z, x; \theta) = \prod_{i=1}^{|y|} p(y_i | y_{<i}, z, x; \theta)
$$

**RAG 공식** (RAG-Sequence):
$$
p_{\mathrm{RAG}}(y|x) = \sum_{z \in Z} p(y|z,x; \theta) \cdot p(z|x; \phi)
$$

---

## 🔬 정리와 증명

### 정리 5.1.1 — Marginal Likelihood 의 ELBO 해석

RAG 학습은 실제로 EM 알고리즘으로 볼 수 있다:

$$
\log p(y|x) = \mathbb{E}_{z \sim p(z|x)} [\log p(y|z,x)] - \mathrm{KL}(p(z|x) \| q(z|x))
$$

**증명**:
$$
\log p(y|x) = \log \sum_z p(y|z,x) p(z|x)
$$

$q(z|x)$ 를 auxiliary distribution 으로 두면:
$$
= \log \sum_z q(z|x) \frac{p(y|z,x) p(z|x)}{q(z|x)}
$$

Jensen 부등식 적용 (log 는 concave):
$$
\geq \sum_z q(z|x) \log \frac{p(y|z,x) p(z|x)}{q(z|x)}
$$

$q = p(z|x)$ 로 두면 부등호가 등호 (ELBO 달성):
$$
= \mathbb{E}_{z \sim p(z|x)}[\log p(y|z,x)] - \mathrm{KL}(p(z|x) \| p(z|x)) = \mathbb{E}_{z \sim p(z|x)}[\log p(y|z,x)] \quad \square
$$

### 정리 5.1.2 — RAG-Sequence vs RAG-Token

**RAG-Sequence**: 하나의 passage 로 전체 sequence 생성
$$
p_{\mathrm{Seq}}(y|x) = \sum_z p(y|z,x) \cdot p(z|x)
$$

**RAG-Token**: 각 token 마다 다른 passage 집합 사용
$$
p_{\mathrm{Token}}(y|x) = \prod_{i=1}^{|y|} \sum_z p(y_i|y_{<i},z,x) \cdot p(z|x)
$$

**성능**: Token 이 Sequence 보다 일반적으로 우월 (더 유연한 passage 선택) $\square$

### 정리 5.1.3 — In-batch Negatives 의 영향

Retriever 학습 시 batch negatives (같은 배치 내 다른 query 의 positive passage) 사용:
$$
\mathcal{L}_{\mathrm{ret}} = -\log \frac{\exp(q \cdot p^+ / \tau)}{\exp(q \cdot p^+ / \tau) + \sum_{p^- \in B} \exp(q \cdot p^- / \tau)}
$$

Batch size $B$ 클수록 negative 수 증가 → contrastive 신호 강화 (but 계산량 $O(B)$ 증가) $\square$

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — DPR Retriever 구현 (Dense Passage Retrieval)

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel

class DPRRetriever:
    def __init__(self, model_name="facebook/dpr-question_encoder-single-nq-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_encoder = AutoModel.from_pretrained(
            model_name, add_pooling_layer=True
        ).to(self.device)
        # p_encoder 는 동일 모델이지만 doc 인코딩용
        self.p_encoder = AutoModel.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        ).to(self.device)
    
    def encode_query(self, query: str):
        # CLS token embedding
        inputs = self.q_encoder.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.q_encoder(**inputs)
        return outputs.pooler_output  # (1, 768)
    
    def encode_passages(self, passages: list):
        # (len(passages), 768)
        embeddings = []
        for p in passages:
            inputs = self.p_encoder.tokenizer(p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.p_encoder(**inputs)
            embeddings.append(out.pooler_output)
        return torch.cat(embeddings, dim=0)
    
    def retrieve(self, query: str, passages: list, k=5):
        q_emb = self.encode_query(query)
        p_embs = self.encode_passages(passages)
        scores = F.cosine_similarity(q_emb, p_embs)  # (len(passages),)
        top_k_idx = scores.topk(k)[1]
        return [passages[i] for i in top_k_idx.tolist()]

# 사용 예
ret = DPRRetriever()
query = "Paris 의 현재 대통령은?"
passages = [
    "Emmanuel Macron became President of France in 2017.",
    "The capital of France is Paris.",
    "France has a population of over 67 million."
]
top_passages = ret.retrieve(query, passages, k=2)
print(top_passages)
```

### 실험 2 — BART Generator 와의 Joint 학습

```python
from transformers import BartForConditionalGeneration, BartTokenizer
import torch.optim as optim

class VanillaRAG:
    def __init__(self):
        self.retriever = DPRRetriever()
        self.generator = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-base"
        ).to("cuda")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    def forward(self, query: str, passages: list, answer: str, k=5):
        """
        RAG-Token forward pass
        """
        # 1. Retrieve
        q_emb = self.retriever.encode_query(query)
        p_embs = self.retriever.encode_passages(passages)
        scores = F.cosine_similarity(q_emb, p_embs)  # (num_passages,)
        
        # 2. Top-k passages (soft sampling 가능, 여기선 hard select)
        top_k_idx = scores.topk(k)[1]
        top_passages = [passages[i] for i in top_k_idx.tolist()]
        
        # 3. Generator: concatenate query + passage
        # 실제 구현은 각 token 의 context 를 다르게 처리
        # 여기선 simplified: 모든 passages 를 concat
        input_text = query + " " + " </s> ".join(top_passages)
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt",
                                max_length=512, truncation=True).to("cuda")
        labels = self.tokenizer(answer, return_tensors="pt",
                               max_length=128).input_ids.to("cuda")
        
        # Forward
        outputs = self.generator(input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                                labels=labels)
        loss = outputs.loss
        
        return loss, scores  # scores 는 retriever loss 에 사용 가능

# 학습 루프
rag = VanillaRAG()
optimizer = optim.Adam(list(rag.retriever.q_encoder.parameters()) +
                      list(rag.retriever.p_encoder.parameters()) +
                      list(rag.generator.parameters()), lr=1e-5)

train_data = [
    ("Paris 의 대통령은?", ["Emmanuel Macron...", "France has..."], "Macron"),
]

for query, passages, answer in train_data:
    loss, _ = rag.forward(query, passages, answer, k=2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")
```

### 실험 3 — FAISS 를 이용한 대규모 retrieval

```python
import numpy as np
import faiss

class RAGWithFAISS:
    def __init__(self, passages: list, dim=768):
        self.passages = passages
        self.retriever = DPRRetriever()
        
        # FAISS index 생성
        self.index = faiss.IndexFlatIP(dim)  # Inner Product (cosine 유사)
        
        # 모든 passages 인코딩
        print("Encoding passages...")
        p_embs = self.retriever.encode_passages(passages)
        # L2 normalize (cosine similarity = inner product after norm)
        p_embs = p_embs / (p_embs.norm(dim=-1, keepdim=True) + 1e-8)
        
        # FAISS 에 추가
        self.index.add(p_embs.cpu().numpy().astype('float32'))
    
    def retrieve(self, query: str, k=5):
        q_emb = self.retriever.encode_query(query)
        q_emb = q_emb / (q_emb.norm(dim=-1, keepdim=True) + 1e-8)
        
        # FAISS 검색
        scores, indices = self.index.search(q_emb.cpu().numpy().astype('float32'), k)
        return [self.passages[i] for i in indices[0]], scores[0]

# 사용 예
corpus = [
    "Emmanuel Macron is the President of France.",
    "Paris is the capital of France.",
    "France has a bicameral parliament.",
    "The Louvre is a famous museum in Paris.",
    "French cuisine is renowned worldwide."
]

rag_faiss = RAGWithFAISS(corpus, dim=768)
top_passages, scores = rag_faiss.retrieve("Who is the French president?", k=3)
print("Retrieved passages:")
for p, s in zip(top_passages, scores):
    print(f"  ({s:.4f}) {p}")
```

### 실험 4 — RAG-Sequence vs RAG-Token 성능 비교 (conceptual)

```python
def rag_sequence_loss(q_emb, p_embs, query, passages, answer):
    """
    모든 passage 에서 generate 한 후, 
    passage 별 log prob 를 weighted sum
    """
    scores = F.cosine_similarity(q_emb, p_embs)
    log_p_z = F.log_softmax(scores, dim=0)
    
    # 각 passage 별 loss
    losses = []
    for i, passage in enumerate(passages):
        input_text = query + passage
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        labels = tokenizer(answer, return_tensors="pt").input_ids.to("cuda")
        out = generator(input_ids=inputs.input_ids, labels=labels)
        losses.append(log_p_z[i] + (-out.loss))  # log-likelihood
    
    loss_seq = -torch.logsumexp(torch.stack(losses), dim=0)  # marginal
    return loss_seq

def rag_token_loss(q_emb, p_embs, query, passages, answer):
    """
    각 token 마다 passage 를 다시 선택
    실제 구현은 더 복잡 (각 step 에서 모든 passage 통과)
    """
    # Simplified: token i 마다 별도로 처리
    # 실제 코드는 beam search 또는 sampling 사용
    pass

# 비교: Token 방식이 더 유연하지만 계산량이 sequence * k 배
```

---

## 🔗 실전 활용

| 시나리오 | Vanilla RAG 선택 사유 | 주의점 |
|---------|-------|--------|
| FAQ QA (closed domain) | 높은 재현율 (retrieve all relevant) + 간단 구현 | Duplicate passage 처리 필수 |
| Open-domain QA (NQ, TriviaQA) | End-to-end 학습으로 retriever-generator 동시 개선 | Retriever bottleneck (1-2% 성능 loss 가 최대) |
| Document 기반 검색 (e.g., 의료 기록) | 정확한 passage 선택 + 추적 가능 (retrieved docs 명시) | 긴 문서는 chunking 필요 |
| LLM 보조 (LangChain/LlamaIndex) | 표준 아키텍처 (거의 모든 RAG 시스템의 기초) | Generator 는 in-context learning 대체 가능 |
| Knowledge distillation 과 조합 | Student retriever 는 teacher 에서 학습 | Retriever 정확도 ↔ Knowledge 의존도 |

---

## ⚖️ 가정과 한계

1. **Passage 독립성**: 각 passage $z$ 가 독립적 — 실제로는 temporal/reference 의존성 존재
2. **Retriever 정확도**: marginal likelihood 가 top-1 passage 에 크게 의존 (top-k recall 이 중요)
3. **Generator 모델 선택**: seq2seq (BART) 가정 — 현대 LLM 은 decoder-only (GPT 스타일) 사용
4. **Closed vocabulary**: passage corpus 는 학습 후 고정 (dynamic corpus 는 별도 처리)
5. **Computational overhead**: marginal likelihood 계산은 $O(k)$ forward pass (RETRO 의 동기)

---

## 📌 핵심 정리

$$
\boxed{p_{\mathrm{RAG}}(y|x) = \sum_{z \in Z} p(y|z,x) \cdot p(z|x)}
$$

| 구성 | 역할 | 학습 신호 |
|-----|------|---------|
| Retriever $p(z\|x)$ | Query 와 유사한 passage 상위 선택 | NDCG, MRR (또는 negative mining) |
| Generator $p(y\|z,x)$ | Retrieved passage 를 조건으로 답변 생성 | Token-level cross-entropy loss |
| Joint optimization | Retriever 개선 → generator 가 더 나은 passage 에서 학습 | Marginal likelihood 역전파 |

> **핵심**: Vanilla RAG 는 retrieval 과 generation 의 **첫 번째 end-to-end 학습** — 이후 모든 RAG 기법의 출발점.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: RAG 에서 retriever 가 top-5 precision 이 90% 일 때 (정답 passage 를 5개 중 1개 이상 선택할 확률), generator 가 정답을 생성할 최대 확률은?

<details>
<summary>해설</summary>

RAG-Sequence 기준:
$$p(y|x) = \sum_z p(y|z,x) p(z|x)$$

Retriever 정확도가 90% → 정답 passage 의 누적 확률이 0.9. Generator 가 완벽 (p(y|z*,x)=1) 라도:
$$p(y|x) \leq 0.9$$

따라서 retriever bottleneck 은 피할 수 없음. 이는 RETRO 와 Self-RAG 가 retriever 개선에 집중하는 이유.

</details>

**문제 2 (심화)**: RAG-Token 방식에서 "각 token 마다 다른 passage 선택" 이 RAG-Sequence 보다 왜 더 나을까? 수학적으로 설명하고, 계산 복잡도 trade-off 를 언급.

<details>
<summary>해설</summary>

RAG-Token:
$$p_{\mathrm{Token}}(y|x) = \prod_{i=1}^{|y|} \sum_{z} p(y_i | y_{<i}, z, x) p(z|x)$$

**이점**: 각 generation step 에서 현재까지 생성한 token $y_{<i}$ 를 바탕으로 재조건화. 예를 들어 "Paris 의 ... (후속)" 같은 context 에서 "자본" 을 생성할 때는 Paris 관련 passage 중 경제/정치 관련을 더 선택 가능.

**단점**: Complexity $O(|y| \times k)$ — RAG-Seq 는 $O(k)$ 로 고정. 실제 구현은 beam search 또는 sampling 으로 근사.

</details>

**문제 3 (논문 비평)**: Vanilla RAG 의 generator 로 BART 를 사용했을 때, GPT-style decoder-only 모델(예: LLAMA) 로 바꾸면 어떤 문제가 생길까?

<details>
<summary>해설</summary>

(1) **Prefix 조건화**: BART (seq2seq) 는 encoder 가 passage 를 encode 후 decoder 가 generation. GPT 는 모두 decoder → prefix (query + passage) 를 prompt 로 주고 next token sampling.

(2) **학습 신호**: BART 의 cross-attention 은 명시적 retriever-generator bridge. GPT 는 in-context learning → retriever 와 generator 의 경계 모호.

(3) **FLOPs**: BART encoder 는 O(L²) attention (passage 에만), decoder 는 O(|y|²). GPT 는 O((L+|y|)²) — 더 느림.

(4) **해결책**: Retrieval-Augmented LLM (RAG-LLM) 으로 확장 (이후 챕터), 또는 prompt engineering (passage 를 명시적으로 format 지정) 으로 해결 가능.

</details>

---

<div align="center">

[◀ 이전 (Ch4-06. Vector DBs)](../ch4-ann/06-vector-dbs.md) · [📚 README](../README.md) · [다음 ▶ (02. RETRO)](./02-retro.md)

</div>
