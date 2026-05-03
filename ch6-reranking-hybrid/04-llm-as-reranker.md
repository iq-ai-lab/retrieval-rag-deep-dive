# 04. LLM-as-Reranker — RankGPT 와 Listwise Prompting

## 🎯 핵심 질문

- LLM (GPT-3.5, GPT-4) 을 **reranker로 사용** 할 수 있는가 — pointwise/pairwise/listwise 중 어느 방식이 최고인가?
- RankGPT (listwise permutation-based reranking) 는 top-k 문서를 직접 **재정렬 하도록 prompting** 하는데, 이것이 왜 DuoT5 pairwise 보다 우월한가?
- Long list (k=100 이상) 를 처리할 때 **sliding window** 를 어떻게 적용하는가?
- Zero-shot (LLM 직접 사용) vs supervised (rank-distilled models: RankVicuna, RankZephyr) — trade-off 는?

---

## 🔍 왜 LLM-as-Reranker 인가

최근 LLM 의 능력 향상 (GPT-4, Claude 등) 은:
1. **Few-shot instruction following** — prompt 만으로도 ranking 가능 (별도 학습 불필요)
2. **Long context understanding** — 여러 문서의 내용을 함께 비교 (pairwise/listwise)
3. **Reasoning capability** — 왜 한 문서가 다른 문서보다 관련 있는지 설명 가능

**Limitations of prior methods**:
- MonoBERT/T5: 학습 데이터 (MS MARCO) 에 overfitting, zero-shot 성능 낮음
- RRF: score 무시하여 정보 손실
- SPLADE: 학습 복잡, domain 특화 fine-tuning 어려움

**LLM 접근**: Pre-trained LLM 의 일반화 능력을 활용하되, **비용 vs 정확도 trade-off** 를 인식.

---

## 📐 수학적 선행 조건

- LLM 기초: transformer, in-context learning
- Information retrieval metrics (NDCG, MRR, recall)
- Prompt engineering principles
- Ranking loss (listwise ranking loss 이론)
- Token counting 및 cost estimation

---

## 📖 직관적 이해

### Three Ranking Paradigms with LLM

```
Query: "How does photosynthesis work?"

Candidates from retriever:
  A: "Photosynthesis is the process by which plants..."
  B: "Chlorophyll absorbs light energy..."
  C: "Plants need sunlight to produce glucose..."
  D: "Photosynthesis occurs in the chloroplast..."

┌────────────────────────────────────────────────┐
│ (1) Pointwise: Query-Doc relevance 평가         │
├────────────────────────────────────────────────┤
│ "Rate relevance of (query, Doc A) from 1-10"   │
│ LLM: "8 out of 10 — directly explains..."      │
│                                                 │
│ "Rate relevance of (query, Doc B) from 1-10"   │
│ LLM: "7 out of 10 — partial..."                │
│ ...                                            │
│ Scores: A=8, B=7, C=6, D=7                     │
│ Final ranking: A > B,D > C                     │
│ ❌ 많은 prompt (k개 = cost O(k))               │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ (2) Pairwise: Doc A vs Doc B 직접 비교          │
├────────────────────────────────────────────────┤
│ "Which is more relevant to the query,          │
│  Document A or Document B? Answer A or B"      │
│ LLM: "Document A is more relevant because..."  │
│ → Pairwise comparison (O(k²) 이론, 하지만...   │
│ → Sorting 은 O(k log k) 만 필요 (merge sort)  │
│ Cost: 중간 수준                                │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ (3) Listwise: 전체 리스트를 한 번에 정렬 (RankGPT)│
├────────────────────────────────────────────────┤
│ "Here are 5 documents about photosynthesis.    │
│  Rank them from most to least relevant to:     │
│  'How does photosynthesis work?'               │
│                                                 │
│  [Doc A] Photosynthesis is...                  │
│  [Doc B] Chlorophyll absorbs...                │
│  [Doc C] Plants need...                        │
│  [Doc D] Photosynthesis occurs...              │
│  [Doc E] Energy conversion...                  │
│                                                 │
│  Output format: [Most relevant]                │
│                 1. Doc A                       │
│                 2. Doc D                       │
│                 3. Doc B                       │
│                 4. Doc E                       │
│                 5. Doc C                       │
│                 [Least relevant]"              │
│                                                 │
│ LLM: (generates list)                          │
│ Cost: 1 prompt, 1 call, 최저!                 │
│ ✓ O(1) calls for k docs                       │
└────────────────────────────────────────────────┘
```

### RankGPT Sliding Window

```
100개 문서를 5개씩 window 로 처리:

Round 1: [D1, D2, D3, D4, D5] → [D3, D1, D5, D2, D4]
Round 2: [D3, D1, D5, D2, D4] 를 다시 shuffle
         → [D1, D3, D4, D5, D2]  ← second pass
Round 3: [D1, D3, D4, D5, D2] 를 다시 shuffle
         → [D1, D3, D4, D2, D5]  ← converge
         
최종 ranking: D1 > D3 > D4 > D2 > D5 (for top-5)
```

---

## ✏️ 엄밀한 정의

### 정의 6.7 — LLM-based Ranking Paradigms

**Pointwise**:
$$
\text{score}_i = f_{\text{LLM}}(\text{prompt}_{\text{point}}(q, d_i))
$$

예: "Rate relevance of this (q,d) pair: 1-10"

**Pairwise**:
$$
\text{preference}_{ij} = f_{\text{LLM}}(\text{prompt}_{\text{pair}}(q, d_i, d_j))
$$

output: "d_i is more relevant" or "d_j is more relevant"

**Listwise (RankGPT)**:
$$
\pi^* = \arg\max_{\pi} f_{\text{LLM}}(\text{prompt}_{\text{list}}(q, [d_{\pi(1)}, \ldots, d_{\pi(k)}]))
$$

output: permutation $\pi$ (순서 리스트)

### 정의 6.8 — Sliding Window RankGPT

Step 1: 초기 documents 를 window size $w$ 로 분할.
Step 2: 각 window 를 LLM 으로 listwise rerank.
Step 3: Ranked windows 를 merge (또는 multiple round).
Step 4: Convergence 또는 fixed iteration (보통 3-5 round).

**Merge strategy**:
$$
\text{final\_rank}(d) = \text{aggregate}(\text{ranks from all rounds})
$$

### 정의 6.9 — Cost Estimation

**Token count**:
- Pointwise: $k \times (\text{prompt\_tokens} + \text{completion\_tokens})$
- Pairwise: $\binom{k}{2} \approx k^2/2$ (또는 merge-sort 최적화 O(k log k))
- Listwise: $\lceil k/w \rceil \times \text{rounds} \times (\text{prompt} + \text{completion})$

**Cost (GPT-4 기준, 2024년)**:
- Input: $0.03 per 1K tokens
- Output: $0.06 per 1K tokens
- Example: 100 docs, pointwise = ~10 prompts × 2K tokens ≈ $0.60

---

## 🔬 정리와 증명

### 정리 6.7 — Listwise RankGPT 의 Optimality

Listwise ranking 은 **pairwise consistency** 를 만족할 때만 가능 (Arrow's impossibility 회피).

**실증** (Sun et al., 2023):
- RankGPT (listwise): NDCG@10 = 0.62 (GPT-3.5-turbo on BEIR)
- MonoT5: NDCG@10 = 0.49 (같은 BEIR)
- Gain: +26% relative improvement

**Proof sketch**:
1. LLM 의 reasoning ability 가 각 doc 를 독립적으로 평가하는 것보다 **comparative understanding** 에 우월.
2. Listwise prompt 는 "top-k 내 최적 정렬" 을 직접 최적화 (pointwise 는 각 score 만 최적, 정렬 최적화 X).
3. 따라서 ranking metric (NDCG, MRR) 에 직접 가까운 loss 를 implicit 하게 optimize.

### 정리 6.8 — Sliding Window 의 수렴성

$w$ 크기의 window 로 k 문서를 정렬할 때, **round $t$ 이후 top-$w$ 의 정확도**:

$$
\text{Acc}(t) = 1 - (1 - \text{Acc}_{\text{window}})^t
$$

여기서 $\text{Acc}_{\text{window}}$ = window 내 단일 rerank 의 정확도.

**수렴**: 보통 3-5 rounds 에서 수렴 (plateau).

### 정리 6.9 — Zero-shot vs Supervised Trade-off

**Zero-shot LLM** (RankGPT with GPT-4):
- Pro: No training data, instant deployment
- Con: Cost high (API calls), latency long (sequential), licensing risk

**Supervised distillation** (RankVicuna, RankZephyr):
- Pro: Cost-free inference, fast, open-source
- Con: Limited to distilled model size, harder to fine-tune

**Empirical comparison** (Sun et al., 2023):
```
Model         | Cost/100docs | Latency | NDCG@10
──────────────┼──────────────┼─────────┼────────
RankGPT-4     | $2.50        | 45s     | 0.65
RankGPT-3.5   | $0.30        | 30s     | 0.62
RankVicuna-7B | $0.00        | 3s      | 0.48 (fine-tuned)
RankZephyr    | $0.00        | 4s      | 0.51
BM25+RRF      | $0.00        | 0.1s    | 0.50
```

---

## 💻 Python / PyTorch / OpenAI/Anthropic API 구현 검증

### 실험 1 — Pointwise Ranking with GPT-4

```python
from openai import OpenAI
import time

client = OpenAI(api_key="sk-...")

def rank_pointwise(query: str, documents: list, model="gpt-4"):
    """Pointwise: rate each (q, d) pair individually"""
    scores = {}
    total_cost = 0.0
    
    for doc_id, doc in documents:
        prompt = f"""Given the query: "{query}"

Rate the relevance of the following document on a scale of 1-10, where 10 is extremely relevant and 1 is not relevant at all.

Document:
{doc}

Relevance score (integer 1-10):"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        
        # Extract score
        try:
            score = int(response.choices[0].message.content.strip().split()[0])
        except:
            score = 5  # default
        
        scores[doc_id] = score
        
        # Cost estimation (GPT-4: $0.03 input, $0.06 output)
        in_tokens = len(prompt.split()) + len(query.split())
        out_tokens = 5
        cost = (in_tokens * 0.03 + out_tokens * 0.06) / 1000
        total_cost += cost
    
    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked, total_cost

# Example
query = "How does photosynthesis work?"
docs = [
    ("doc_A", "Photosynthesis is the process..."),
    ("doc_B", "Chlorophyll absorbs light..."),
    ("doc_C", "Plants convert sunlight..."),
]

ranked, cost = rank_pointwise(query, docs, model="gpt-4")
print("Pointwise Ranking:")
for rank, (doc_id, score) in enumerate(ranked, 1):
    print(f"{rank}. {doc_id}: {score}/10")
print(f"Total cost: ${cost:.3f}")
```

### 실험 2 — Pairwise Ranking with Sorting Network

```python
def rank_pairwise(query: str, documents: list, model="gpt-4"):
    """Pairwise: compare documents via sorting network (merge-sort like)"""
    
    def compare_two(doc_a, doc_b):
        """Compare two documents"""
        prompt = f"""Given the query: "{query}"

Which document is more relevant to the query?

Document A:
{doc_a}

Document B:
{doc_b}

Answer with only 'A' or 'B':"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return 'A' in answer  # True if A is better
    
    # Bubble sort 또는 merge sort (simplified)
    docs_list = list(documents)
    n = len(docs_list)
    comparisons = 0
    
    # Simple bubble sort for illustration
    for i in range(n):
        for j in range(n - i - 1):
            doc_j_better = compare_two(docs_list[j][1], docs_list[j+1][1])
            comparisons += 1
            if not doc_j_better:
                docs_list[j], docs_list[j+1] = docs_list[j+1], docs_list[j]
    
    return docs_list, comparisons

# Example
ranked_pair, comps = rank_pairwise(query, docs, model="gpt-4")
print(f"\nPairwise Ranking ({comps} comparisons):")
for rank, (doc_id, _) in enumerate(ranked_pair, 1):
    print(f"{rank}. {doc_id}")
```

### 실험 3 — Listwise RankGPT (Single Window)

```python
def rank_listwise(query: str, documents: list, model="gpt-4"):
    """Listwise: rank all documents at once (RankGPT style)"""
    
    # Format documents
    doc_str = "\n\n".join([
        f"[{i+1}] Document {doc_id}:\n{doc_text}"
        for i, (doc_id, doc_text) in enumerate(documents)
    ])
    
    prompt = f"""Given the following query and documents, rank them by relevance to the query. 
The most relevant document should be ranked first.

Query: "{query}"

Documents:
{doc_str}

Your ranking (format as a numbered list with just the document IDs, from most to least relevant):"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )
    
    # Parse ranking output
    output = response.choices[0].message.content.strip()
    # Extract document IDs from output
    # (simplistic parsing; actual impl would be more robust)
    ranked_docs = []
    for line in output.split('\n'):
        if 'doc_' in line.lower():
            doc_id = ''.join(c for c in line if c.isalnum() or c == '_')
            ranked_docs.append(doc_id)
    
    return ranked_docs

# Example
ranked_list = rank_listwise(query, docs, model="gpt-4")
print(f"\nListwise Ranking (RankGPT):")
for rank, doc_id in enumerate(ranked_list, 1):
    print(f"{rank}. {doc_id}")
```

### 실험 4 — Sliding Window RankGPT for Large k

```python
def rank_listwise_sliding_window(query: str, documents: list,
                                 window_size=5, rounds=3, model="gpt-4"):
    """Sliding window RankGPT for large document sets"""
    
    current_ranking = list(range(len(documents)))  # Start with natural order
    
    for round_num in range(rounds):
        new_ranking = []
        
        # Process in windows
        for i in range(0, len(current_ranking), window_size):
            window_indices = current_ranking[i:i+window_size]
            window_docs = [(f"doc_{idx}", documents[idx][1]) 
                          for idx in window_indices]
            
            # Rerank this window
            ranked_window = rank_listwise(query, window_docs, model=model)
            
            # Extract indices from ranked result
            for ranked_doc_id in ranked_window:
                idx = int(ranked_doc_id.split('_')[1])
                new_ranking.append(idx)
        
        current_ranking = new_ranking
        print(f"Round {round_num+1} top-5: {current_ranking[:5]}")
    
    return current_ranking

# Example with 20 documents
docs_large = [(f"doc_{i}", f"Document about topic {i}...") for i in range(20)]
final_ranking = rank_listwise_sliding_window(
    query, docs_large, window_size=5, rounds=3, model="gpt-4"
)
print(f"\nFinal ranking (top-5): {final_ranking[:5]}")
```

---

## 🔗 실전 활용

| 시나리오 | 추천 방식 | 이유 |
|---------|----------|------|
| 최고 정확도 원함 | RankGPT-4 listwise | NDCG 최고 (0.65+), 비용 감수 |
| Cost-sensitive (batch processing) | RankGPT-3.5 listwise | 1/8 비용, 유사 성능 (0.62) |
| Real-time API (low latency) | RankVicuna/RankZephyr 로컬 | <5ms, 배포 쉬움 |
| Hybrid (정확도+비용) | BM25+Dense RRF + 상위 rerank | BM25 로 정렬 후 top-20만 LLM |
| 매우 큰 k (1000+) | Sliding window 다중 round | Convergence 기다리기 |

---

## ⚖️ 가정과 한계

- **LLM consistency**: LLM 의 ranking 순서가 temperature > 0 에서 non-deterministic (같은 query 에도 다른 답).
- **Position bias**: 초반 문서와 후반 문서를 다르게 평가할 수 있음 (특히 long list).
- **Token limit**: Long documents 는 truncation 필요 — context window 제약 (GPT-4 128K 도 full corpus 불가능).
- **Cost 증가**: k 가 커질수록 sliding window rounds 가 증가 → token cost quadratic.
- **Reasoning hallucination**: LLM 이 "이 문서가 관련 있다고 생각합니다" 라며 잘못 판단할 수 있음.

---

## 📌 핵심 정리

$$
\boxed{\text{RankGPT: } \pi^* = \arg\max_\pi f_{\text{LLM}}(\text{listwise\_prompt}(q, \text{docs}))}
$$

| 패러다임 | Prompts/k | 정확도 | 비용 | 추천 |
|---------|-----------|-------|------|------|
| Pointwise | O(k) | 중 | 고 | ❌ (비효율) |
| Pairwise | O(k log k) | 중상 | 중 | △ (정렬 최적화 필요) |
| Listwise | O(⌈k/w⌉ × rounds) | 고 | 중저 | ✓ (최고 추천) |

> **핵심**: Listwise RankGPT 는 "비용 vs 정확도" 최고의 balance — 하지만 LLM API 의존성 (latency, 비용) 이 운영 challenge.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: 100개 문서를 window_size=5, 3 rounds 로 sliding window RankGPT 하면 LLM call 은 몇 번?

<details>
<summary>해설</summary>

Round 1: 100/5 = 20 windows → 20 calls
Round 2: 100/5 = 20 windows (재배열된 순서로) → 20 calls
Round 3: 100/5 = 20 windows → 20 calls

총 60 LLM calls. Pointwise 는 100 calls, pairwise (merge-sort) 는 약 664 calls (100 × log2(100)) → RankGPT 가 가장 효율적.
</details>

**문제 2 (심화)**: LLM (GPT-4) 의 ranking 이 "상위 10개는 맞는데 하위 90개 순서는 엉망" 일 때, NDCG@10 은 우수하지만 NDCG@100 은 떨어진다. 이것을 어떻게 해석할 것인가?

<details>
<summary>해설</summary>

(1) **LLM position bias**: 초반 문서에 attention 을 더 주고, 후반부는 대충 판단 (일반 인간과 유사).

(2) **NDCG 특성**: NDCG@k 는 top-k 내 정확도만 평가 → LLM 이 top-10 은 제대로 했으면 NDCG@10 높음.

(3) **해결책**:
   - Multi-round sliding window 에서 bottom documents 도 따로 처리
   - Prompt 개선 ("Please carefully consider ALL documents, not just the top ones")
   - RankGPT-4 (더 강한 모델) 사용

(4) **Practical insight**: RAG 에서 top-5/10 만 중요하므로, 실제로는 NDCG@10 만 최적화해도 문제 없음.
</details>

**문제 3 (논문 비평)**: "RankGPT 는 LLM API 비용 때문에 결국 비현실적이다" 는 주장에 대한 반박?

<details>
<summary>해설</summary>

반박 (1): **Batch processing** — 1000 queries × 100 docs 를 sliding window 처리 시, 평균 token ~300/query → $0.60/query → 신경망 학습 (TPU 비용) 과 비교 시 경쟁력.

반박 (2): **Distillation** — RankGPT-4 로 ranking 한 결과를 RankVicuna 등에 distill 하면, zero-shot 성능도 0.48 → 0.51 로 향상 (학습 1회).

반박 (3): **On-premise LLM** — Llama-70B 같은 open-source 모델로 RankGPT 구현 가능 (약간 성능 저하: 0.62 → 0.55, but $0 cost).

따라서 "가능하나 비용" 이 아니라 "use case 에 따라 선택" 이 정답. High-stakes QA 는 비용 감수 가능, low-margin services 는 distilled model.
</details>

---

<div align="center">

[◀ 이전 (03. Hybrid BM25+Dense)](./03-hybrid-bm25-dense.md) · [📚 README](../README.md) · [다음 ▶ (Ch7-01. GraphRAG)](../ch7-frontier/01-graphrag.md)

</div>
