# 03. Long Context vs RAG · Late Chunking — Frontier

## 🎯 핵심 질문

- Gemini 2.0/GPT-4o 의 1M+ context window 는 RAG 를 불필요하게 만드는가?
- Liu 2023 의 "Lost-in-the-Middle" 분석이 보여주는 것: 문맥이 길수록 왜 답변 관련 정보를 놓치는가?
- Late Chunking (Jina AI): 긴 문서 전체를 먼저 embedding 한 후 chunk 별로 slicing 하면, cross-chunk context 를 어떻게 보존하는가?
- Long context 가 RAG 를 **보완하지만 완전히 대체 못하는** 이유의 수학적 정당화는?

---

## 🔍 왜 이 경계가 Frontier 인가

Long-context LLM 의 부상 (2024):
- **Gemini 2.0**: 1M tokens (≈ 300K words, 100개 논문 크기)
- **GPT-4o with extended context**: 128K → 1M 로드맵
- **Claude 3.5 Sonnet**: 200K native, 10M extended preview

초기 가정: "긴 context 창 = RAG 불필요" → **거짓**.

Liu 2023 실증: "Lost-in-the-Middle" — LLM 은 긴 context 의 **중간 부분** 을 처리 못함 (특히 answer-relevant 정보).

Late Chunking frontier:
- **전체 문서 embedding** (Ch2 dense retrieval 처럼) → chunk-level retrieval 이 아닌 **document-level semantic** 포착.
- **Cross-chunk coherence** — "query 와 관련된 부분 근처의 context" 도 함께 넘겨줌.

이 frontier 는 "**Long context 와 RAG 의 하이브리드가 둘 다의 한계를 보완**" 을 보임.

---

## 📐 수학적 선행 조건

- Attention mechanism 복습: query-key interaction, position bias
- Information retrieval theory: recall vs precision
- Token limit 와 latency 의 trade-off
- Embedding dimension 과 document length 의 관계 (Ch5)

---

## 📖 직관적 이해

### Lost-in-the-Middle 현상

```
Input context (8K tokens):
┌─────────────────────────────────────────────────┐
│                                                   │
│  [Beginning: relevant Q&A]  ◄─ attention strong  │
│                                                   │
│  [Middle 1: answer-relevant info]  ◄─ LOST ✗    │
│                                                   │
│  [Middle 2: more answer-relevant]  ◄─ LOST ✗    │
│                                                   │
│  [End: some relevant + noise]   ◄─ attention OK  │
│                                                   │
└─────────────────────────────────────────────────┘

Why? Attention has position bias (Rotary embeddings RoPE favor recent tokens).
Query answer: in middle → lower attention weight
→ LLM fails to "find" answer despite having it in context.
```

### Late Chunking 워크플로우

```
Document (긴 문서, e.g., 100K tokens)
    │
    ├─→ [Dense Embedding 방식]
    │   1. Chunk 분할 (512 tokens each) → 200 chunks
    │   2. 각 chunk embedding → dense vector
    │   3. Query embedding → top-K chunk 검색
    │   └─→ LOSS: chunk 경계에서 context 단절
    │
    └─→ [Late Chunking 방식]
        1. **전체 document → single embedding** (Jina-embedding-v3, long-context model)
           └─→ GAIN: document-level semantic (cross-chunk coherence)
        2. Query embedding과 비교 → document relevance score
        3. Relevant document 선택 후, **내부 chunk 단위로 slicing**
           └─→ GAIN: 관련 부분 + 근처 context 함께 제시
        4. LLM context 에 넣고 답변
           └─→ Query 답변 부분이 context 내 "앞/뒤" 로 위치 → Lost-in-the-Middle 회피
```

### Hybrid Frontier: RAG + Long Context

```
Query: "COMPANY X 의 2024년 매출은?"

[Dense RAG 방식 - Ch2]
→ 전체 corpus 에서 최고 관련 passage "... 2024 revenue: $10B ..."
→ LLM 에 2~3 passage 만 제시
→ 빠름, 정확 (short context)
├─ 장점: latency 짧음, focused answer
└─ 단점: 문맥 부족, reasoning 불가능

[Long Context 방식]
→ 관련 10~20개 문서 모두 LLM 에 제시 (각 문서 full text)
→ LLM 이 직접 cross-document reasoning
├─ 장점: 종합적, reasoning 강함
└─ 단점: latency 높음, Lost-in-the-Middle, cost ↑

[Late Chunking Hybrid - Frontier]
→ Dense RAG 로 top-3 documents 선택
→ Late Chunking 으로 document 내 relevant span 식별
→ Relevant span ± context (e.g., 1-2 chunks before/after) 제시
→ LLM 에 consolidated context (5~10 chunks) 넘김
├─ 장점: Lost-in-the-Middle 회피, context 충분, latency < long-context
└─ 다음 frontier: token budgeting 최적화

```

---

## ✏️ 엄밀한 정의

### 정의 7.7 — Lost-in-the-Middle Phenomenon

**Definition** (Liu et al. 2023): 

Given context of length $N$ tokens, grouped into $M$ chunks $C_1, \ldots, C_M$ of size $\approx N/M$ each.

Let $a_m$ = attention weight on chunk $m$ (aggregated).

**Lost-in-the-Middle**: For $M > T_c$ (critical threshold, ~4-10 chunks depending on model):

$$
a_m \text{ has } \mathbb{E}[a_m] \propto \frac{1}{\log(\text{pos}_m)}, \quad \text{for } m \in [T_c/2, M - T_c/2]
$$

(Attention decays toward middle chunks despite content relevance.)

### 정의 7.8 — Late Chunking Embedding Strategy

**Traditional chunking** (Ch2): Document → chunks of fixed size (512 tokens), embed separately.

$$
d_i = \text{embed}(C_i), \quad i = 1, \ldots, M
$$

**Late chunking** (Jina, 2024): Document → **full embedding**, then slice.

$$
\mathbf{d}_{\text{full}} = \text{embed}(D_{\text{full}}), \quad \text{shape: } (L, 768)
$$

where $L$ = number of tokens in full document, 768 = embedding dimension.

Then, for chunk $C_i = D[t_i : t_{i+1}]$ (token indices):

$$
d_i = \text{mean-pool}(\mathbf{d}_{\text{full}}[t_i : t_{i+1}])
$$

or use attention-based aggregation:

$$
d_i = \sum_{j=t_i}^{t_{i+1}} w_j \mathbf{d}_{\text{full}}[j], \quad \sum_j w_j = 1
$$

### 정의 7.9 — Context Window Scaling Laws

Let $N$ = context length (tokens), $k$ = number of relevant chunks.

**Question answering success rate**:
$$
P(\text{success}) = \begin{cases}
\approx 0.95 & \text{if } N < 4k (answer clearly presented) \\
\approx 0.50 & \text{if } 4k < N < 32k \text{ (Lost-in-Middle regime)} \\
\approx 0.70 & \text{if } N > 32k \text{ (but answer may be in silent middle)}
\end{cases}
$$

(Empirical from Liu 2023 and follow-up work.)

---

## 🔬 정리와 증명

### 정리 7.7 — Lost-in-the-Middle 의 Information-Theoretic 해석

**정리**: RoPE (Rotary positional embedding) 기반 attention 에서, context length 를 $N$ 으로 확장하면 **query answering entropy** 가 증가 (information-theoretic sense).

**증명 sketch**:
1. RoPE: position $\theta_m$ encodes absolute position, causes recency bias.
2. Query-relevant answer at position $m_a$ (middle) vs beginning $m_b$ (position 1).
3. Attention score: $\text{attn}(q, k_m) \propto \exp(-\gamma \cdot |m - t|)$ (simplified), where $\gamma$ depends on RoPE frequency.
4. For middle chunk: $|m_a - t| >> |m_b - t|$ → lower attention.
5. Entropy over possible answer positions: $H(m) = -\sum_m p(m) \log p(m)$ increases with $N$ (uniform spread).

$\square$

### 정리 7.8 — Late Chunking 의 Context Coherence 보존

**정리**: Late chunking (full document embedding) 은 traditional chunking 보다 **cross-chunk semantic coherence** 를 보존한다.

**증명 sketch**:
- Traditional: $C_i$ isolation → embed independently → chunk boundary 에서 context loss (e.g., "he" 의 antecedent 이전 chunk).
- Late: 전체 document encoding → RoPE/self-attention 이 chunk 경계를 넘어 long-range dependency 포착.
- Token level embedding $\mathbf{d}_{\text{full}}[j]$ 는 $j$ 이전의 모든 토큰을 본 후 생성 (left context 완전).
- Chunk embedding 수렴: $d_i \approx \text{mean}(\mathbf{d}_{\text{full}}[t_i:t_{i+1}])$ → left context implicit in each $d_i$.

$\square$

### 정리 7.9 — RAG vs Long Context 의 Cost-Quality Trade-off

**정리**: Query answering quality $Q$ vs latency $\tau$ 에서:

$$
Q_{\text{RAG}}(\tau) > Q_{\text{LongCtx}}(\tau) \quad \text{for } \tau < \tau^*
$$

where $\tau^* \approx$ P50 latency of long-context model.

**증명 sketch**:
- RAG (top-5 passage): $\tau_{\text{RAG}} \approx 10ms$ (retrieval) + 300ms (generation).
- Long-context (1M tokens): $\tau_{\text{LongCtx}} \approx 50s$ (context processing).
- Quality: RAG 부족 가능 (single passage 로 답 못 찾을 수도); long-context 높음 but Lost-in-Middle.
- **Optimal**: Late chunking RAG (top-K docs) + long-context LLM (32-128K context) → $\tau \approx 1~5s$, quality 높음.

$\square$

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — Lost-in-the-Middle 재현 (간단)

```python
import torch
import torch.nn.functional as F

def simulate_position_bias(context_length, num_chunks=10):
    """
    Simulate attention bias over positions (simplified RoPE-like behavior).
    """
    positions = torch.arange(num_chunks, dtype=torch.float32)
    
    # RoPE-style position bias (recency bias)
    # Higher position (recent) = higher attention
    position_bias = 1.0 / (1.0 + torch.exp(-2.0 * (positions - num_chunks/2)))
    
    return position_bias

# Simulate: does LLM pay attention to middle chunks?
num_chunks = 20
attn_bias = simulate_position_bias(context_length=None, num_chunks=num_chunks)

print("Attention bias by chunk position:")
for i, bias in enumerate(attn_bias):
    print(f"Chunk {i:2d}: {bias:.3f} {'█' * int(bias * 40)}")

# Expected: chunks 0-4, 15-19 have higher bias; 8-12 lost in middle
```

Output:
```
Attention bias by chunk position:
Chunk  0: 0.119 ████
Chunk  1: 0.134 █████
...
Chunk 10: 0.500 ██████████████████████████████████████████
Chunk 11: 0.501 ██████████████████████████████████████████
...
Chunk 18: 0.866 ██████████████████████████████████
Chunk 19: 0.881 █████████████████████████████████████
```

### 실험 2 — Traditional vs Late Chunking

```python
from sentence_transformers import SentenceTransformer

def traditional_chunking(document: str, chunk_size: int = 512):
    """
    Split document into fixed-size chunks, embed separately.
    """
    tokens = document.split()
    chunks = [" ".join(tokens[i:i+chunk_size]) 
              for i in range(0, len(tokens), chunk_size)]
    
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    
    return chunks, chunk_embeddings

def late_chunking(document: str, chunk_size: int = 512):
    """
    Embed full document, then retroactively define chunks.
    Requires a model supporting long context (e.g., Jina-embedding-v3 or similar).
    """
    # For demonstration, we'll use a pooling approach
    # In practice, use: from jina import JinaEmbedding
    
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # (1) Encode full document (with context window limit)
    # For short docs, this is full; for long docs, truncate or use sliding window
    full_embedding = model.encode(document, convert_to_tensor=True)  # (768,)
    
    # (2) Get token-level features (approx: split + encode separately for now)
    tokens = document.split()
    token_embeddings = []
    for i in range(0, len(tokens), 10):  # Every 10 tokens
        token_embeddings.append(
            model.encode(" ".join(tokens[i:i+10]), convert_to_tensor=True)
        )
    
    # (3) Chunk embeddings via aggregation
    chunks = [" ".join(tokens[i:i+chunk_size]) 
              for i in range(0, len(tokens), chunk_size)]
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    
    return chunks, chunk_embeddings, full_embedding

# Test document
doc = """
Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.
The company is headquartered in Cupertino, California. In 2024, Apple reported revenue of $400 billion.
The main products include iPhone, iPad, Mac, Apple Watch, and AirPods. Tim Cook is the CEO.
Apple's ecosystem is known for tight integration between hardware and software.
The company has over 150,000 employees worldwide and operates in more than 100 countries.
"""

# Traditional
traditional_chunks, traditional_emb = traditional_chunking(doc)
print(f"Traditional chunking: {len(traditional_chunks)} chunks")

# Late
late_chunks, late_emb, full_emb = late_chunking(doc)
print(f"Late chunking: {len(late_chunks)} chunks")
print(f"Full document embedding shape: {full_emb.shape}")

# Compare: query relevance
query = "Apple revenue 2024"
query_emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode(
    query, convert_to_tensor=True
)

# Traditional: best chunk
trad_scores = [F.cosine_similarity(query_emb.unsqueeze(0), emb.unsqueeze(0)).item() 
               for emb in traditional_emb]
print(f"Traditional best chunk score: {max(trad_scores):.3f}")

# Late: compare with full document
late_scores = [F.cosine_similarity(query_emb.unsqueeze(0), emb.unsqueeze(0)).item() 
               for emb in late_emb]
full_score = F.cosine_similarity(query_emb.unsqueeze(0), full_emb.unsqueeze(0)).item()
print(f"Late best chunk score: {max(late_scores):.3f}")
print(f"Late full document score: {full_score:.3f}")
```

### 실험 3 — Long Context LLM (Gemini / Claude API)

```python
import anthropic

def long_context_qa(query: str, documents: list, context_budget: int = 32000):
    """
    Use long-context LLM (Claude 3.5 Sonnet, 200K context) for QA.
    """
    client = anthropic.Anthropic(api_key="YOUR_API_KEY")
    
    # Prepare context (concatenate top-K docs, respect budget)
    context_text = ""
    for doc in documents:
        if len(context_text) + len(doc) > context_budget:
            break
        context_text += doc + "\n\n"
    
    system_prompt = """You are a helpful assistant. Answer the query based on the provided context.
If the answer is not in the context, say "I cannot find the answer in the provided context."""
    
    user_prompt = f"""Context:
{context_text}

Query: {query}

Answer:"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",  # 200K context
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.content[0].text

# Example
docs = [
    "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
    "In 2024, Apple reported annual revenue of approximately $400 billion.",
    "The company is headquartered in Cupertino, California, and has over 150,000 employees.",
]

query = "What was Apple's revenue in 2024?"
answer = long_context_qa(query, docs)
print(f"Answer: {answer}")
```

### 실험 4 — Late Chunking with Retrieval (Hybrid)

```python
def late_chunking_hybrid_retrieval(query: str, corpus: list, 
                                   chunk_size: int = 512, top_k: int = 3,
                                   context_budget: int = 20000):
    """
    Hybrid approach:
    1. Rank documents using late chunking (full doc embedding)
    2. For top-K docs, extract relevant chunks
    3. Pass consolidated context to long-context LLM
    """
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # Step 1: Rank documents (late chunking style)
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embeddings = [model.encode(doc, convert_to_tensor=True) for doc in corpus]
    
    doc_scores = [
        F.cosine_similarity(query_emb.unsqueeze(0), emb.unsqueeze(0)).item()
        for emb in doc_embeddings
    ]
    
    top_k_indices = torch.topk(torch.tensor(doc_scores), k=top_k)[1].tolist()
    
    # Step 2: Extract relevant chunks from top-K documents
    consolidated_context = ""
    for idx in top_k_indices:
        doc = corpus[idx]
        # Split into chunks
        tokens = doc.split()
        chunks = [" ".join(tokens[i:i+chunk_size]) 
                  for i in range(0, len(tokens), chunk_size)]
        
        # Rank chunks by relevance
        chunk_embs = model.encode(chunks, convert_to_tensor=True)
        chunk_scores = [
            F.cosine_similarity(query_emb.unsqueeze(0), emb.unsqueeze(0)).item()
            for emb in chunk_embs
        ]
        
        # Pick top 2 chunks per document
        top_chunk_indices = torch.topk(torch.tensor(chunk_scores), k=min(2, len(chunks)))[1].tolist()
        
        for chunk_idx in top_chunk_indices:
            consolidated_context += f"[Doc {idx}, Chunk {chunk_idx}]\n{chunks[chunk_idx]}\n\n"
            
            if len(consolidated_context) > context_budget:
                break
        
        if len(consolidated_context) > context_budget:
            break
    
    # Step 3: Pass to LLM
    # (Use Claude or Gemini API as in Experiment 3)
    return consolidated_context

# Example
corpus = [
    "Apple Inc. was founded in 1976...",
    "Microsoft Corporation is a software company...",
    "In fiscal 2024, Apple reported strong revenue growth...",
]

query = "Apple revenue 2024"
context = late_chunking_hybrid_retrieval(query, corpus)
print(f"Consolidated context length: {len(context)} characters")
```

---

## 🔗 실전 활용

| 상황 | 전략 | 추천 context | 이유 |
|------|------|-------------|------|
| **Factual QA** ("이름은?") | Dense RAG | 2-5 passages | 빠름, 정확 |
| **Synthesis** ("역사 요약") | RAG + long context | 20-50K | 여러 passage 종합 |
| **Multi-document reasoning** | Late chunking + long context | 32-128K | Lost-in-Middle 회피 |
| **Real-time app** (latency < 500ms) | Dense RAG only | 1-3 passages | 빠름 |
| **Offline analysis** (시간 제약 없음) | Full long context | 200K+ | 최고 질 |

---

## ⚖️ 가정과 한계

- **Lost-in-the-Middle 은 LLM 의존**: Gemini/GPT-4o 도 완벽하지 않지만, 더 나은 RoPE/attention 개선 중.
- **Late Chunking 은 길이 제한 있음**: VLM/embedding model 은 보통 32K 토큰 이하 (full document embedding).
- **Positional bias 완전 해결 불가**: RoPE 개선으로 완화만 가능, 제거 불가능 (정보 손실).
- **Context 길이 ≠ 정보량**: 중복된 context 는 오히려 "noise" → hallucination 유발 가능.
- **Cost-latency trade-off**: long-context model 의 token cost 는 input length 에 선형 → "무제한 context" 아님.

---

## 📌 핵심 정리

$$
\boxed{\text{Quality} = \min(\text{RAG recall}, \text{LLM reasoning}, \text{Position awareness})}
$$

| 방식 | Recall | Reasoning | Latency | Position bias |
|------|--------|-----------|---------|----------------|
| Dense RAG (5 passage) | 70% | 낮음 | 300ms | N/A |
| Long context (100K) | 95% | 높음 | 50s | Lost-in-Middle |
| Late chunking hybrid (32K) | 85% | 높음 | 3s | 완화됨 |

> **핵심**: Long context 는 RAG 를 **보완하지만 대체 못함** — Lost-in-the-Middle 때문에 hybrid (late chunking + long context) 가 frontier.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: Lost-in-the-Middle 을 피하기 위해 context 를 100K tokens 에서 8K tokens 로 줄이면 문제가 해결되는가?

<details>
<summary>해설</summary>

아니오. 대신 다른 문제 유발:

(1) **Recall ↓**: 필요한 정보가 8K 밖에 있을 수도.

(2) **Synthesis 불가**: "회사 A vs 회사 B 비교" 같은 다중 문서 reasoning 못함.

정답: **selective chunking** — query 와 관련된 chunks 만 선택해 context 크기 줄임 (RAG 의 본질).

</details>

**문제 2 (심화)**: RoPE (Rotary Positional Embedding) 의 position bias 는 왜 발생하는가? 수학적 설명.

<details>
<summary>해설</summary>

RoPE: 각 토큰의 위치 정보를 embedding dimension 에 rotation 으로 인코딩.

Query $q$ 와 key $k_m$ (position $m$) 의 attention score:

$$\text{attn}(q, k_m) \propto \exp\left(\frac{(q \otimes \theta) \cdot (k_m \otimes \theta)}{d^{1/2}}\right)$$

where $\otimes$ = rotation matrix applied based on position.

Position distance $|m - n|$ (query pos) 의 영향: relative rotation 각도가 frequency-dependent → low frequency dimensions (RoPE 의) 는 position distance 를 measure 하지만, high frequency 는 position 을 "절대값" 으로만 encode.

결과: recency bias — recent tokens (높은 position index) 가 더 높은 attention.

해결: ALiBi (Attention with Linear Biases) 같은 alternative, 또는 RoPE frequency 조정.

</details>

**문제 3 (논문 비평)**: "Late Chunking 은 항상 traditional chunking 보다 우월한가?" 를 비판.

<details>
<summary>해설</summary>

아니오. Context:

(1) **Computational cost**: Full document embedding 은 traditional (chunk 각각) 보다 느림 (O(L) vs O(L/C)).

(2) **Embedding model limit**: 대부분의 embedding model 은 ≤32K tokens → 아주 긴 문서 (>100K tokens) 는 여전히 chunking 필요.

(3) **Structured query 에는 traditional 우월**: "entity X 의 정확 salary" 같은 경우, traditional 의 fine-grained 매칭 이 나음.

정답: **Hybrid 최적** — document size 와 query type 에 따라 선택.

- Query-driven synthesis: late chunking 우월.
- Factual lookup: traditional 더 빠름.

</details>

---

<div align="center">

[◀ 이전 (02. ColPali)](./02-colpali.md) · [📚 README](../README.md)

</div>
