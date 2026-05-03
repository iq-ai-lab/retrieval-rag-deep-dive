# 01. Cross-Encoder Reranker — MonoBERT · MonoT5

## 🎯 핵심 질문

- Dense retriever (e.g., DPR) 는 수백만 문서를 embedding 하는데, 상위 k 개 후보 중 "정말 관련 있는 것" 을 어떻게 구분하는가?
- MonoBERT 와 MonoT5 의 근본적 차이 — 분류 vs 생성 — 는 reranking 정확도에 어떤 영향을 주는가?
- 왜 MS MARCO 에서 two-stage (retrieval + reranking) 이 single dense 보다 +20-30% MRR 을 얻는가?
- DuoT5 의 pairwise comparison 은 ListNet ranking loss 와 어떻게 다른가?

---

## 🔍 왜 Cross-Encoder Reranker 가 RAG 에 필수인가

Dense retrieval (DPR, ColBERT, BGE) 은 **BM25 보다 효율적** 이지만 top-k 내 정확한 순위 결정에는 약합니다:

1. **Recall-Precision 트레이드오프** — k 를 크게 (e.g., 100) 하면 모든 답을 포함하지만 rank 5 내 정답이 보장 안 됨.
2. **In-batch mixing** — retriever 의 batch embedding 은 non-contextual; "질문 q" 에 대해 "문서 d" 의 상관도를 정확히 계산하지 않음.
3. **모더은 LLM 은 top-5 품질에 민감** — 상위 doc 의 정확도 1% 차이가 최종 QA 답변의 3-5% 정확도 변화.

Cross-encoder 는 **질문과 문서를 함께 encode** 하여 각 (q, d) 쌍의 관련도를 직접 계산합니다.

---

## 📐 수학적 선행 조건

- Transformer attention mechanism (Ch0)
- Binary classification (softmax, BCE loss)
- Sequence-to-sequence generation (T5 기초)
- Ranking metrics: MRR, NDCG, MAP (Ch6-00)

---

## 📖 직관적 이해

### Two-Stage Pipeline

```
Query: "Albert Einstein 의 주요 업적은?"

┌─────────────────────────────────┐
│  Stage 1: Dense Retriever       │
│  (e.g., DPR, BGE-large)         │
│  Speed: 빠름, Recall 중심        │
└────────────┬────────────────────┘
             │ top-100 docs
             ▼
    ┌─────────────────┐
    │ Doc 1-100 (후보들)
    │ ├─ Doc 41: "Einstein relativity" (rank 41)
    │ ├─ Doc 7:  "Photoelectric effect"
    │ ├─ Doc 99: "역사적 배경"
    │ └─ ...
    └────────┬────────┘
             │
┌────────────▼──────────────────────┐
│  Stage 2: Cross-Encoder Reranker  │
│  (MonoBERT, MonoT5)               │
│  Speed: 느림, Precision 중심       │
└────────────┬──────────────────────┘
             │
             ▼
    Top-5 reranked:
    1. Doc 7  (relevance=0.95)
    2. Doc 41 (relevance=0.89)
    3. ...
```

### MonoBERT vs MonoT5

```
Input: [CLS] "Einstein 업적" [SEP] "상대성이론은..." [SEP]

MonoBERT (분류):
  [CLS] embedding → FC(2) → softmax(relevant, irrelevant)
  Output: 0.92 (probability of relevant)

MonoT5 (생성):
  Encoder: Input 전체
  Decoder: generate "true" or "false"
  Output: P("true" | input) ≈ 0.95
```

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Cross-Encoder

Query $q$, document $d$ 에 대해 **cross-encoder** $f_\theta$ 는:
$$
s(q, d) = f_\theta([CLS] \, q \, [SEP] \, d \, [SEP])
$$

여기서 $s \in [0, 1]$ (또는 unbounded).

**MonoBERT**: $s(q,d) = \text{softmax}(\mathbf{w}^\top h_{[CLS]})_{\text{relevant}}$, $h_{[CLS]}$ = BERT [CLS] representation.

**MonoT5**: $s(q,d) = P_\theta(\text{"true"} | q, d)$, T5 생성 확률.

### 정의 6.2 — Reranking Loss

**Classification (MonoBERT)**:
$$
\mathcal{L}_{\text{clf}} = -\sum_{(q,d,y)} y \log s(q,d) + (1-y) \log(1-s(q,d))
$$

**Generation (MonoT5)**:
$$
\mathcal{L}_{\text{gen}} = -\log P(\text{"true"} | q, d^+) - \log P(\text{"false"} | q, d^-)
$$

### 정의 6.3 — DuoT5 (Pairwise)

두 문서 $d_1, d_2$ 를 T5 로 비교:
$$
\text{input} = [CLS] \, q \, [SEP] \, d_1 \, [SEP] \, d_2 \, [SEP]
$$
$$
P(\text{d1 > d2}) = P_\theta(\text{"yes"} | \text{input})
$$

---

## 🔬 정리와 증명

### 정리 6.1 — Cross-Encoder 의 Contextual Ranking

Cross-encoder 의 relevance score 는 **joint representation** 에 기반하므로, 동일 문서도 다른 질문에서 상이한 점수를 받음:

$$
s(q_1, d) \neq s(q_2, d) \quad \text{in general}
$$

대비: Dense retriever 는 $d$ 의 embedding 이 고정 ($\text{score} = q \cdot d$, query-specific scaling 만 적용).

**의미**: Cross-encoder 가 더 정확한 ranking 을 가능하게 하지만 inference cost 가 높음 ($O(k)$ forward passes for top-k docs).

### 정리 6.2 — MS MARCO 에서의 MRR Gain

MS MARCO 100K dev set 기준:
- Dense only (DPR): MRR@10 = 0.73
- Dense + MonoT5 rerank (top 100): MRR@10 = 0.88 (+20.5%)

**증명 스케치**: Retriever 의 rank 버그 (recall loss) 를 reranker 가 보정. Top-100 내에 정답이 있으면, reranker 는 ~95% 확률로 상위 5 내 위치.

### 정리 6.3 — MonoBERT vs MonoT5 의 Calibration

MonoT5 의 생성 확률 $P(\text{"true"} | q,d)$ 는 **더 calibrated** (true 확률과 일치율 높음) 했으나 계산상 느림.

**실증** (Nogueira 2020): MonoT5-3B > MonoBERT-base on BEIR zero-shot, 하지만 모델 크기가 15배.

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — MonoBERT Reranker 구현 및 inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load MonoBERT (CE-DistilRoBERTa-base 사용)
model_name = "cross-encoder/mmarco-MiniLMv2-L12-H384-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def rerank_monbert(query: str, docs: list, batch_size=32):
    """
    query: 검색 질문
    docs: 문서 리스트
    Returns: (doc, score) 쌍 정렬된 리스트
    """
    scores = []
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        # Tokenize (query, doc) pairs
        pairs = [[query, doc] for doc in batch_docs]
        inputs = tokenizer(pairs, padding=True, truncation=True,
                          max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            logits = model(**inputs).logits
            # sigmoid 스코어
            batch_scores = torch.sigmoid(logits[:, 1]).cpu().numpy()
        scores.extend(batch_scores)
    
    # Sort by score
    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
    return ranked

# 예시
query = "Albert Einstein 의 주요 업적은?"
docs = [
    "상대성이론은 물리학의 기초를 재정의했다.",
    "아인슈타인은 1879년 독일에서 태어났다.",
    "광전효과의 발견으로 노벨상을 수상했다.",
]
ranked = rerank_monbert(query, docs)
for doc, score in ranked:
    print(f"{score:.3f} | {doc}")
```

### 실험 2 — MonoT5 Reranker 와 생성 확률

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# MonoT5 (castorini/monot5-base-msmarco)
tokenizer = T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco")
model = T5ForConditionalGeneration.from_pretrained(
    "castorini/monot5-base-msmarco"
)
model.eval()

def rerank_monot5(query: str, docs: list):
    """Generate "true"/"false" probability"""
    scores = []
    for doc in docs:
        # Input format: "Query: [query] Document: [doc]"
        input_text = f"Query: {query} Document: {doc}"
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        
        with torch.no_grad():
            # "true" token logit
            outputs = model(input_ids=inputs, decoder_input_ids=inputs[:, :1])
            logits = outputs.logits[0, 0, :]
            
            # True label ID (depends on tokenizer, usually 1176 for T5)
            true_id = tokenizer.encode("true")[0]
            false_id = tokenizer.encode("false")[0]
            
            true_logit = logits[true_id]
            false_logit = logits[false_id]
            
            # Softmax
            score = torch.softmax(
                torch.stack([true_logit, false_logit]), dim=0
            )[0].item()
        
        scores.append(score)
    
    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
    return ranked

ranked_t5 = rerank_monot5(query, docs)
for doc, score in ranked_t5:
    print(f"{score:.3f} | {doc}")
```

### 실험 3 — Two-Stage Pipeline (Retrieval + Reranking)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Stage 1: Dense Retriever (BGE-small)
retriever = SentenceTransformer("BAAI/bge-small-en-v1.5")

# DB documents
db_docs = [
    "상대성이론 (특수 · 일반)은 시공간의 기본 구조를 설명한다.",
    "광전효과는 양자의 관점에서 빛과 전자의 상호작용을 설명한다.",
    "아인슈타인은 1921년 노벨상을 받았다.",
    "브라운 운동에 대한 분자적 설명을 제시했다.",
    "우주 상수의 개념을 도입했다.",
]

query = "상대성이론의 의미"

# Encode
doc_embeddings = retriever.encode(db_docs, convert_to_tensor=True)
query_embedding = retriever.encode(query, convert_to_tensor=True)

# Top-k retrieval
from torch.nn.functional import cosine_similarity
sims = cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings)[0]
top_k_indices = torch.topk(sims, k=3).indices.tolist()
top_k_docs = [db_docs[i] for i in top_k_indices]

print("=== Stage 1: Retrieval ===")
for idx in top_k_indices:
    print(f"{sims[idx]:.3f} | {db_docs[idx]}")

# Stage 2: Reranking
print("\n=== Stage 2: Reranking (MonoT5) ===")
ranked = rerank_monot5(query, top_k_docs)
for doc, score in ranked:
    print(f"{score:.3f} | {doc}")
```

### 실험 4 — DuoT5 Pairwise Comparison

```python
def rerank_duot5(query: str, docs: list):
    """
    T5 를 이용한 pairwise comparison.
    모든 쌍을 비교하여 그래프 기반으로 최종 순서 결정 (복잡함).
    여기서는 간단한 voting 예시.
    """
    # DuoT5 의 복잡함을 고려해 MonoT5 로 근사
    # 실제 DuoT5 는 permutation-based reranking (RankZephyr 같은 LLM 방식으로 진화)
    return rerank_monot5(query, docs)

# 실전: 실제 DuoT5 는 상업용 API (Cohere Rerank, jina reranker) 로 사용
```

---

## 🔗 실전 활용

| 시나리오 | 추천 Reranker | 이유 |
|---------|--------------|------|
| 빠른 prototype | MonoBERT-base | 가볍고 배포 쉬움, MRR@10 ~80% |
| 최고 정확도 원함 | MonoT5-3B | MS MARCO 에서 최고 성능, batch 가능 |
| Long document 처리 | ColBERT v2 + MonoT5 | ColBERT 가 long context 우월 |
| 실시간 (P95 latency < 100ms) | TinyBERT reranker | 증류 모델, 배치 size 적어도 무방 |
| Cost-sensitive | BM25 두 번째 pass | 가장 저렴, 의외로 경쟁력 있음 |

---

## ⚖️ 가정과 한계

- **Top-k 가정**: Reranker 는 top-k 내에만 효과 — recall 이 낮으면 (정답이 top-100 내 없음) 무용지물.
- **Latency 비용**: 100 docs × 0.01 sec/doc ≈ 1 sec — serving 시 배치 처리 필수.
- **학습 데이터 의존성**: MonoBERT/T5 는 MS MARCO (English wiki+passages) 에 pretrain — 다른 언어/domain 에 fine-tune 필요.
- **질문 길이 제약**: Transformer 의 512 토큰 limit — very long documents 는 truncation.

---

## 📌 핵심 정리

$$
\boxed{s(q, d) = f_\theta([CLS] \, q \, [SEP] \, d \, [SEP])}
$$

| 모델 | 방식 | 장점 | 단점 |
|------|------|------|------|
| MonoBERT | 분류 (softmax) | 빠름, 가벼움 | 확률 보정 (calibration) 약함 |
| MonoT5 | 생성 ("true"/"false") | 정확도 높음, 보정 우수 | 느림 (T5 decoding) |
| DuoT5 | Pairwise 비교 | 이론적으로 우월 | 계산량 $O(k^2)$, 실용성 낮음 |

> **핵심**: Two-stage (retrieval + reranking) 는 dense-only 대비 +20-30% MRR 향상 — RAG 에서 필수 불가결.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: 1000개 문서 중 top-100 을 retriever 로 찾은 후 MonoBERT reranker 로 top-5 를 다시 순서 짓는 경우의 수는? Inference time 은?

<details>
<summary>해설</summary>

경우의 수: $100 \times 99 \times 98 \times 97 \times 96$ (상이 순열, 하지만 보통 점수 기반). Inference time: 100 문서 × ~0.001 sec/doc ≈ 0.1 sec (batch 처리 시). GPU 배치면 더 빠름.
</details>

**문제 2 (심화)**: Retriever 가 top-100 내 정답을 놓칠 확률이 5% 일 때, reranker 가 그것을 "1등으로 만들 수 없다"는 의미는? 이것이 two-stage 의 근본 한계는?

<details>
<summary>해설</summary>

정답이 top-100 에 없으면 reranker 는 다른 문서 중 최고 순위를 매길 수 밖에 없음 (recall 을 높일 수 없음). 따라서 전체 시스템의 MRR 은 retriever 의 recall 에 상한 (bottleneck). 해결책: retriever 의 k 를 키우거나 (latency 증가), retriever 자체 성능을 올림 (학습).
</details>

**문제 3 (논문 비평)**: "MonoT5 는 MonoBERT 보다 항상 좋다" 는 주장의 문제점?

<details>
<summary>해설</summary>

(1) 모델 크기 차이 (T5-base 대 MonoBERT-base 는 parameter 수 비슷하지만, T5-3B 는 15배 큼). (2) Decoding latency — MonoT5 는 token-by-token 생성이므로 batch 효율이 MonoBERT 보다 낮을 수 있음. (3) 몇몇 in-domain task (e.g., scientific papers 의 domain-specific reranking) 에서는 fine-tune 된 MonoBERT 가 더 효율적. 따라서 "가성비" 관점에서는 domain, latency SLO 에 따라 다름.
</details>

---

<div align="center">

[◀ 이전 (Ch5-06. FiD)](../ch5-rag/06-fid.md) · [📚 README](../README.md) · [다음 ▶ (02. RRF)](./02-rrf.md)

</div>
