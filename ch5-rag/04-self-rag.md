# 04. Self-RAG (Asai 2024)

## 🎯 핵심 질문

- 모든 질문이 retrieval 이 필요한 것은 아니다 — 언제 retrieve 할지, 어떤 retrieved document 가 유용한지를 LLM 이 스스로 판단하는 방법은?
- **Reflection token** ([Retrieve], [IsREL], [IsSUP], [IsUSE]) 를 vocabulary 에 추가하고, GPT-4 로 label 을 생성한 후 fine-tuning 하는 원리?
- Token-level decoding 에서 reflection score 를 기반으로 **beam 을 branching** 하는 dynamic tree search 는?
- "Critic + Generator 의 통합" 으로 왜 hallucination 을 줄일 수 있는가?

---

## 🔍 왜 이 기법이 retrieval·RAG 에 중요한가

이전 기법들의 한계:
- Vanilla RAG, RETRO, REALM: **모든 input 에 항상 retrieval** → 불필요한 retrieval 오버헤드, 때로 retrieved doc 가 방해
- 예: "What is 2+2?" 같은 간단한 질문도 retrieval → irrelevant docs 에서 confusion 증가

Self-RAG (Asai 2024) 의 혁신: **Adaptive retrieval + In-context critic**
- LLM 이 [Retrieve] token 을 생성하면 retrieval 수행 (필요할 때만)
- [IsREL], [IsSUP] 같은 reflection token 으로 retrieved document 의 relevance·supportiveness 평가
- **Dynamic decoding**: beam search 에서 reflection score 를 기반으로 경로 선택 → 높은 품질의 경로만 유지

결과: 
- Retrieval 횟수 40% 감소 (필요 없는 경우 스킵)
- 같은 throughput 에서 quality 향상 (beam 의 효율적 사용)
- 현재 가장 강력한 adaptive RAG 시스템

---

## 📐 수학적 선행 조건

- LLM generation (decoding, beam search)
- Beam search 알고리즘 (tree expansion)
- Scoring function (log probability)
- Information retrieval 기초

---

## 📖 직관적 이해

### Self-RAG 의 Reflection Tokens

```
표준 LLM generation:
  "Question: What is 2+2?"
  → "The answer is 4"
  (retrieval 여부 모름, always use doc 가능)

Self-RAG generation:
  "Question: What is 2+2?"
  → "[Retrieve: No] The answer is 4 [IsREL: N/A]"
  (LLM 이 자체 판단: 이 질문은 knowledge retrieval 불필요)
  
  vs
  
  "Question: Who is the current president of France?"
  → "[Retrieve: Yes] <search result: Emmanuel Macron...> 
      The current president is Emmanuel Macron [IsREL: Yes] [IsSUP: Yes]"
  (LLM 이 판단: 이 질문은 최신 정보 필요, retrieve 함)
```

### Dynamic Decoding with Reflection Scores

```
Beam search tree:
            [START]
             /  \
         Gen    Gen
          /       \
      [Retrieve?] [Retrieve?]
       /   \         /  \
     Yes   No      Yes  No
     /      \      /     \
  Search  Continue Search Continue
    |      |        |      |
 [IsREL?] [...]  [IsREL?] [...]
  Y/N/N=  score1  Y/N/N=  score2
  
점수 = P(token) × reflection_score
      × (1 if is_relevant else 0.5)
      
Beam 에서 점수 높은 경로만 유지 → quality control
```

---

## ✏️ 엄밀한 정의

### 정의 5.4.1 — Reflection Token Vocabulary Extension

원본 vocabulary $\mathcal{V}$ 에 4개의 special tokens 추가:

$$
\mathcal{V}' = \mathcal{V} \cup \{[\text{Retrieve}], [\text{IsREL}], [\text{IsSUP}], [\text{IsUSE}]\}
$$

각 token 의 의미:
- $[\text{Retrieve}]$: Yes/No — 다음 retrieval 수행 여부
- $[\text{IsREL}]$: Yes/No/Partial — retrieved document 가 query 에 relevant 한지
- $[\text{IsSUP}]$: Yes/No/Partial — retrieved document 가 생성 답변을 support 하는지
- $[\text{IsUSE}]$: Yes/No — 최종 답변이 useful/correct 한지

### 정의 5.4.2 — Token-level Supervised Fine-tuning

주어진 (query, answer) pair, retrieved document $d$ 에 대해:

**GPT-4 로 label 생성**:
$$
[\text{Retrieve}]_{\text{label}}, [\text{IsREL}]_{\text{label}}, [\text{IsSUP}]_{\text{label}}, [\text{IsUSE}]_{\text{label}} = \text{GPT4}(q, d, a)
$$

**Fine-tuning loss** (각 reflection token 위치에서):
$$
\mathcal{L}_{\text{SFT}} = -\sum_{t} \log p(r_t | r_{<t}, q, d)
$$

여기서 $r_t \in \{\text{Yes}, \text{No}, \text{Partial}\}$ 는 ground-truth reflection token.

**전체 loss**:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{generation}} + \lambda \mathcal{L}_{\text{reflection}}
$$

### 정의 5.4.3 — Dynamic Tree Search Decoding

Beam width $k$, reflection weight $\alpha$ 에 대해:

$$
\text{Score}(s_t) = \log p(y_t | y_{<t}) + \alpha \cdot R(y_t)
$$

여기서 $R(y_t)$ 는 reflection score:
$$
R(y_t) = \begin{cases}
1.0 & \text{if } y_t = [\text{Retrieve}] \text{ and model confident} \\
0.8 & \text{if } [\text{IsREL}] = \text{Yes} \\
0.5 & \text{if } [\text{IsREL}] = \text{Partial} \\
0.0 & \text{if } [\text{IsREL}] = \text{No}
\end{cases}
$$

각 step 에서 top-$k$ 개 경로만 유지 (branching factor $= k$).

---

## 🔬 정리와 증명

### 정리 5.4.1 — Adaptive Retrieval 의 Efficiency Gain

Self-RAG 가 모든 input 에 항상 retrieval 하는 baseline 대비 $c$ 배 빠른 이유:

$$
\text{Speedup} = \frac{\text{Latency}_{\text{baseline}}}{\text{Latency}_{\text{self-rag}}} = \frac{1}{1 - (1-p_r) \cdot r}
$$

- $p_r$: retrieval 을 수행할 확률 (전체 input 의 40%)
- $r$: retrieval latency 의 비율 (생성 시간 대비)

**예**: $p_r = 0.4, r = 0.3$ → $\text{Speedup} = 1 / (1 - 0.6 \times 0.3) = 1.23×$ (23% 빠름)

**더 큰 효과**: Retrieved doc 이 답변에 해를 끼치는 경우 제거 → generation quality 향상 (indirect speedup) $\square$

### 정리 5.4.2 — Reflection Token 의 Prediction Accuracy

Self-RAG 모델이 fine-tuning 후 reflection token 을 정확히 predict 하는 확률:

$$
\text{Accuracy}([\text{IsREL}]) \approx 0.92, \quad \text{Accuracy}([\text{IsSUP}]) \approx 0.89, \quad \text{Accuracy}([\text{IsUSE}]) \approx 0.85
$$

**의미**: LLaMA-7B fine-tuned on reflection data → GPT-4 label 과 92% agreement 달성. 이는 모델이 **internal critic** 으로서 충분히 작동함을 의미 $\square$

### 정리 5.4.3 — Beam Branching 의 Quality-Latency Trade-off

Beam width $k$, reflection weight $\alpha$ 에 대해:

$$
\text{Quality}(k, \alpha) \uparrow, \quad \text{Latency}(k) \uparrow
$$

**Optimal point**: $k=4$ (각 step 에서 4개 경로 유지), $\alpha = 0.5$ (reflection 와 likelihood 의 균형)

- $k=1$ (greedy): 빠르지만 quality 낮음 (reflection 무시)
- $k=8$: quality 높지만 latency 2배 증가
- $k=4$: optimal (quality 95%, latency overhead 20%) $\square$

---

## 💻 Python / PyTorch 구현 검증

### 실험 1 — Reflection Token 추가 및 Fine-tuning

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim

class SelfRAGTokenizer:
    def __init__(self, base_tokenizer_name="meta-llama/Llama-2-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
        
        # Reflection tokens 추가
        self.reflection_tokens = {
            "[Retrieve]": "<retrieve>",
            "[IsREL]": "<isrel>",
            "[IsSUP]": "<issup>",
            "[IsUSE]": "<isuse>",
        }
        
        # Tokenizer 에 특수 토큰 추가
        self.tokenizer.add_tokens(list(self.reflection_tokens.values()))
        
    def encode_with_reflection(self, text: str, reflection_labels: dict):
        """
        text: original generation
        reflection_labels: {'retrieve': 'yes', 'isrel': 'yes', ...}
        """
        # Insert reflection tokens
        augmented_text = text
        if 'retrieve' in reflection_labels:
            augmented_text += f" {self.reflection_tokens['[Retrieve]']} {reflection_labels['retrieve']}"
        if 'isrel' in reflection_labels:
            augmented_text += f" {self.reflection_tokens['[IsREL]']} {reflection_labels['isrel']}"
        
        return self.tokenizer.encode(augmented_text, return_tensors="pt")

class SelfRAGModel:
    def __init__(self, base_model_name="meta-llama/Llama-2-7b-hf"):
        self.tokenizer = SelfRAGTokenizer(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            vocab_size=len(self.tokenizer.tokenizer)
        )
    
    def fine_tune_on_reflection(self, examples: list):
        """
        examples: [(query, answer, retrieved_doc, reflection_labels), ...]
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for query, answer, doc, labels in examples:
            # Construct input: query + doc + answer + reflection tokens
            input_text = f"Query: {query}\nDocument: {doc}\nAnswer: {answer}"
            
            # Encode with reflection labels
            input_ids = self.tokenizer.encode_with_reflection(input_text, labels)
            
            # Forward pass
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Loss: {loss.item():.4f}")

# 사용 예
rag = SelfRAGModel()
examples = [
    ("Who is the president of France?",
     "Emmanuel Macron",
     "Emmanuel Macron became President of France in 2017.",
     {"retrieve": "yes", "isrel": "yes", "issup": "yes"}),
]
# rag.fine_tune_on_reflection(examples)  # In practice, use real data
```

### 실험 2 — Adaptive Retrieval 결정

```python
class AdaptiveRetriever:
    def __init__(self, model: SelfRAGModel):
        self.model = model
        self.retriever = None  # Placeholder for actual retriever (FAISS, etc.)
    
    def should_retrieve(self, query: str, top_k: int = 1):
        """
        Model 의 [Retrieve] token 출력을 기반으로 retrieval 여부 결정
        """
        # Generate tokens until [Retrieve] decision
        input_ids = self.model.tokenizer.tokenizer.encode(
            f"Query: {query}",
            return_tensors="pt"
        )
        
        # Model forward (temperature = 0.7 for some variance)
        with torch.no_grad():
            logits = self.model.model(input_ids).logits
        
        # Get next token probability
        next_token_logits = logits[0, -1, :] / 0.7  # temperature sampling
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # Check if [Retrieve] token is likely
        retrieve_token_id = self.model.tokenizer.tokenizer.encode("[Retrieve]")[0]
        retrieve_prob = probs[retrieve_token_id].item()
        
        return retrieve_prob > 0.5
    
    def retrieve_if_needed(self, query: str, documents: list):
        """
        Adaptive retrieval: decide and retrieve if necessary
        """
        if self.should_retrieve(query):
            # Simple BM25-like retrieval (placeholder)
            # In practice, use FAISS or similar
            scores = [len(set(query.split()) & set(doc.split())) for doc in documents]
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:1]
            return [documents[i] for i in top_idx]
        else:
            return []  # No retrieval needed

# 사용 예
retriever = AdaptiveRetriever(rag)
query1 = "What is 2+2?"
query2 = "Who is the current president of France?"

retrieved1 = retriever.retrieve_if_needed(query1, [])
retrieved2 = retriever.retrieve_if_needed(query2, ["Emmanuel Macron is the president of France."])

print(f"Query 1 retrieval: {len(retrieved1)} docs")  # Expected: 0
print(f"Query 2 retrieval: {len(retrieved2)} docs")  # Expected: 1
```

### 실험 3 — Dynamic Beam Search with Reflection Scores

```python
import heapq

class ReflectionScoredBeamSearch:
    def __init__(self, model: SelfRAGModel, k: int = 4, alpha: float = 0.5):
        self.model = model
        self.k = k  # beam width
        self.alpha = alpha  # reflection weight
    
    def reflection_score(self, token: str):
        """
        Reflection token 에 기반한 score
        """
        if token in ["yes", "[IsREL]=Yes"]:
            return 0.8
        elif token in ["partial", "[IsREL]=Partial"]:
            return 0.5
        elif token in ["no", "[IsREL]=No"]:
            return 0.0
        else:
            return 1.0  # Regular token: neutral
    
    def beam_search(self, query: str, max_length: int = 50):
        """
        Dynamic beam search with reflection scores
        """
        input_ids = self.model.tokenizer.tokenizer.encode(query, return_tensors="pt")
        
        # Initialize beam: (score, sequence, input_ids)
        beams = [(0.0, [], input_ids.clone())]
        
        for step in range(max_length):
            candidates = []
            
            for score, seq, ids in beams:
                if ids.shape[1] > 512:  # Max seq len
                    candidates.append((score, seq, ids))
                    continue
                
                # Model forward
                with torch.no_grad():
                    logits = self.model.model(ids).logits
                
                # Top-k tokens
                topk_logits, topk_indices = torch.topk(logits[0, -1, :], k=self.k)
                topk_probs = torch.softmax(topk_logits, dim=-1).log()
                
                for token_id, log_prob in zip(topk_indices, topk_probs):
                    token = self.model.tokenizer.tokenizer.decode([token_id])
                    
                    # Compute score: likelihood + reflection
                    token_score = log_prob.item()
                    refl_score = self.reflection_score(token)
                    combined_score = token_score + self.alpha * refl_score
                    
                    new_score = score + combined_score
                    new_seq = seq + [token]
                    new_ids = torch.cat([ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    
                    candidates.append((new_score, new_seq, new_ids))
            
            # Keep top-k
            candidates.sort(key=lambda x: -x[0])
            beams = candidates[:self.k]
        
        # Return best sequence
        best_score, best_seq, _ = beams[0]
        return " ".join(best_seq)

# 사용 예
beam_search = ReflectionScoredBeamSearch(rag, k=4, alpha=0.5)
query = "Who is the president of France?"
generated = beam_search.beam_search(query, max_length=30)
print(f"Generated: {generated}")
```

### 실험 4 — Reflection Label 생성 (GPT-4 API 사용)

```python
# 실제 구현은 OpenAI API 필요
# 여기는 mock implementation

class ReflectionLabelGenerator:
    def __init__(self):
        self.gpt4_prompt_template = """Given:
Query: {query}
Retrieved Document: {doc}
Generated Answer: {answer}

Evaluate:
- IsRelevant (Is the document relevant to the query?): Yes/Partial/No
- IsSupporting (Does the document support the answer?): Yes/Partial/No
- IsUseful (Is the answer useful and correct?): Yes/No

Return only: IsREL=<Y/P/N>, IsSUP=<Y/P/N>, IsUSE=<Y/N>"""
    
    def generate_labels(self, query: str, doc: str, answer: str):
        """
        Mock: return hardcoded labels
        Real implementation: call GPT-4 API
        """
        # This would call: response = openai.ChatCompletion.create(...)
        # For now, return example labels
        return {
            "retrieve": "yes" if len(query) > 5 else "no",
            "isrel": "yes" if any(w in doc.lower() for w in query.lower().split()) else "no",
            "issup": "yes",
            "isuse": "yes",
        }

# 사용 예
label_gen = ReflectionLabelGenerator()
labels = label_gen.generate_labels(
    "Who is the president of France?",
    "Emmanuel Macron is the current president of France.",
    "Emmanuel Macron"
)
print(f"Labels: {labels}")
```

---

## 🔗 실전 활용

| 시나리오 | Self-RAG 선택 사유 | 주의점 |
|---------|-------|--------|
| Real-time QA (latency critical) | Adaptive retrieval 로 40% 오버헤드 절감 | Reflection token accuracy < 100% → occasional errors |
| Knowledge-heavy domain (medical, legal) | [IsSUP] 로 hallucination 제거, 신뢰도 증가 | Fine-tuning data (GPT-4 label) 필요 |
| Multi-turn conversation | Retrieve 여부를 context-aware 하게 결정 | Conversation history 의존성 (long context 챌린지) |
| Low-latency serving | Beam width = 2-3 으로 quality-latency trade-off | 매우 복잡한 질문은 약할 수 있음 |

---

## ⚖️ 가정과 한계

1. **Fine-tuning data**: GPT-4 로 reflection label 생성 (비싼 연산, 시간 소요)
2. **Reflection token accuracy**: 92% 정도 → 8% 의 잘못된 판단 가능
3. **Beam search overhead**: Dynamic branching 은 여전히 sequential (parallel 안됨)
4. **Domain shift**: 특정 domain 에 fine-tuned 모델은 다른 domain 에서 reflection accuracy 하락
5. **Generalization**: 매우 다른 query pattern 에는 적응성 부족

---

## 📌 핵심 정리

| 구성 | 역할 |
|------|------|
| [Retrieve] | Query 에 retrieval 필요 여부 결정 |
| [IsREL] | Retrieved document 의 relevance 평가 |
| [IsSUP] | Document 가 생성 답변을 support 하는지 판단 |
| [IsUSE] | 최종 답변의 유용성 평가 |

$$
\boxed{\text{Score}(s_t) = \log p(y_t) + \alpha \cdot R(y_t)}
$$

> **핵심**: Self-RAG 는 **LLM 스스로 retrieval 과 quality 를 판단** — "always retrieve" 의 비효율을 "adaptive retrieve" 로 개선. 현재 가장 실용적 RAG 시스템.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: Self-RAG 에서 [Retrieve] token 을 "No" 로 결정했다면, [IsREL]·[IsSUP] 는 어떻게 처리하는가?

<details>
<summary>해설</summary>

[Retrieve]="No" 이면 following reflection tokens 는 무시 (또는 "[IsREL]=N/A" 로 마크). 

Decoder 의 로직:
```python
if token == "[Retrieve]=No":
    # Skip retrieval, continue generation without doc context
    skip_reflection_tokens = True
elif token == "[Retrieve]=Yes":
    # Perform retrieval, expect [IsREL], [IsSUP], [IsUSE]
    skip_reflection_tokens = False
```

이렇게 하면 불필요한 retrieval 을 skip 하면서도 model 의 consistency 유지.

</details>

**문제 2 (심화)**: Self-RAG 의 beam search 에서 reflection weight $\alpha$ 를 어떻게 설정해야 하는가? Task-dependent 한가?

<details>
<summary>해설</summary>

**일반적 설정**: $\alpha = 0.5$ (log likelihood 와 reflection score 의 균형)

**Task-dependent 조정**:
- **Factoid QA (NQ, TriviaQA)**: $\alpha = 0.7$ (reflection 을 더 중시 → hallucination 감소)
- **Open-domain generation (Wikipedia)**: $\alpha = 0.3$ (likelihood 중시 → fluency 향상)
- **Medical/Legal**: $\alpha = 0.8$ (reflection 매우 중시 → safety critical)

**AutoTune 방법**: 
Development set 에서 ROUGE/EM 를 max 하는 $\alpha$ 를 grid search. 일반적으로 $\alpha \in [0.3, 0.8]$.

</details>

**문제 3 (논문 비평)**: "Self-RAG 가 40% retrieval 비용 절감" 이라는 주장이 실제로 end-to-end latency 에서 의미가 있는가?

<details>
<summary>해설</summary>

**이론**:
- Retrieval latency = 100ms (FAISS search)
- Generation latency = 500ms (LLM decoding)
- Baseline (always retrieve) = 600ms

Self-RAG (40% retrieval):
- 60% 의 input 은 generation only = 500ms
- 40% 의 input 은 retrieve + generate = 600ms
- 평균 = 0.6 × 500 + 0.4 × 600 = 540ms

**Speed up = 600 / 540 = 1.11×** (11% 개선)

**비판**:
- 11% 개선은 미미 (0.4 의 retrieval 비율이 낮아서)
- Retrieval latency 가 더 크면 (예: API call) → 더 큰 이득

**더 큰 효과**:
- Reflection tokens 로 hallucination 감소 → **Quality 향상** (이것이 main contribution)
- Speed up 은 secondary benefit

</details>

---

<div align="center">

[◀ 이전 (03. REALM · Atlas)](./03-realm-atlas.md) · [📚 README](../README.md) · [다음 ▶ (05. CRAG)](./05-crag.md)

</div>
