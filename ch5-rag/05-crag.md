# 05. CRAG — Corrective RAG (Yan 2024)

## 🎯 핵심 질문

- Retrieved document 가 충분하지 않을 때 (confidence 낮음) 어떻게 web search 로 fallback 할 것인가?
- **Retrieval evaluator** (T5 기반 confidence model) 를 어떻게 훈련하고, 그 신호를 dynamically 활용하는가?
- "Knowledge refinement" (passage 수준의 filtering 및 선택) 로 quality 를 어떻게 높일 것인가?
- CRAG 가 Self-RAG 의 reflection 과 다르게, **외부 source (web)** 를 활용하는 방식의 장점은?

---

## 🔍 왜 이 기법이 retrieval·RAG 에 중요한가

Self-RAG 의 한계:
- Reflection token 은 **internal critic** (모델이 자신의 판단)
- Hallucination 여전히 가능 (모델이 높은 confidence 로 잘못된 판단)
- Fixed corpus 에만 retrieval (새로운 정보 반영 불가)

CRAG (Corrective RAG, Yan 2024) 의 혁신: **Confidence-based Routing**
- Retrieval evaluator 로 retrieved document 의 "actual utility" 평가
- **세 가지 경로로 분기**:
  1. High confidence → 표준 retrieval (corpus-based)
  2. Low confidence → web search (real-time, 새 정보)
  3. Ambiguous → 둘 다 (ensemble)
- Knowledge refinement 로 passage 수준에서 quality 제어

결과:
- Corpus 기반 retrieval 의 한계 극복 (web search 추가)
- Self-RAG 대비 hallucination 40% 감소
- **현실적 application** (최신 정보 필요, 신뢰도 중요한 domain)

---

## 📐 수학적 선행 조건

- Information retrieval 평가 (NDCG, MRR)
- Confidence/uncertainty 모델링
- Multi-task learning (T5)
- Document ranking (BM25, dense retrieval)

---

## 📖 직관적 이해

### CRAG 의 3-Path Routing

```
Query: "Who is the current president of Germany? (2024)"

Retrieved docs: ["Angela Merkel was the chancellor of Germany"]
                  ↓
           [Retrieval Evaluator]
                  ↓
        Confidence = 0.3 (very low, outdated)
                  ↓
        ┌─────────┬─────────┬─────────┐
        │         │         │         │
      High      Low      Ambiguous
     (>0.7)   (<0.5)     (0.5-0.7)
        │         │         │
    Use Retrieved  Web Search   Both
        │         │         │
        └─────────┼─────────┘
                  ↓
   Knowledge Refinement
   (passage filtering,
    strip-level selection)
                  ↓
        [Final augmented context]
                  ↓
        Generator: "Olaf Scholz is the current..."
```

### Retrieval Evaluator (T5)

```
T5 backbone (encoder-decoder):

Input: 
  "Question: Who is the president of Germany?
   Retrieved: Angela Merkel was the chancellor of Germany"

Output (3-class):
  - Supported (정답이 passage 에 명시됨): P = 0.7
  - Partially Supported (관련 있지만 정확하지 않음): P = 0.2
  - Not Supported (관련 없음): P = 0.1

Decision:
  if Supported > 0.7: use retrieved
  elif Not Supported > 0.5: web search
  else: both
```

---

## ✏️ 엄밀한 정의

### 정의 5.5.1 — Retrieval Evaluator

**Model**: T5-base encoder-decoder

**Input format**:
$$
\text{input} = \text{"Question: } q \text{ Retrieved: } d \text{"}
$$

**Output**: 3-class probability distribution
$$
p_{\text{eval}}(c | q, d) = \text{softmax}(\text{logits}_{\text{T5}})
$$

여기서 $c \in \{\text{Supported}, \text{Partially}, \text{Not}\}$

**Training data**: 
- Supported: (q, d, a) 에서 $d$ 의 정보로 $a$ 를 생성 가능
- Partially: 관련은 있지만 partial
- Not: 무관

### 정의 5.5.2 — Confidence-based Routing

**Confidence score**:
$$
\text{Conf}(q, d) = \max_c p_{\text{eval}}(c | q, d)
$$

**Routing 결정**:
$$
\text{Path} = \begin{cases}
\text{Retrieve} & \text{if } \text{Conf} > \tau_h \text{ and } p(\text{Supported}) > \tau_s \\
\text{Web} & \text{if } \text{Conf} < \tau_l \text{ or } p(\text{Not}) > \tau_n \\
\text{Hybrid} & \text{otherwise}
\end{cases}
$$

- $\tau_h = 0.7$ (high), $\tau_l = 0.5$ (low)
- $\tau_s = 0.5$ (supported), $\tau_n = 0.5$ (not supported)

### 정의 5.5.3 — Knowledge Refinement (Strip-level)

Retrieved documents 에서:

1. **Passage-level filtering**: Evaluator confidence < threshold 인 passage 제거

2. **Strip-level extraction**: 각 passage 에서 query-relevant sentence 만 추출
   $$
   \text{strips} = \{\text{sent} \in d : \text{relevance}(q, \text{sent}) > \theta\}
   $$
   여기서 relevance 는 BM25 또는 semantic similarity

3. **Final context**:
   $$
   C = \text{concat}(\text{strips}_1, \text{strips}_2, \ldots)
   $$
   (top-k strips by score)

---

## 🔬 정리와 증명

### 정리 5.5.1 — Web Search 의 효과

Query 가 outdated/factual information 필요할 때:

$$
\text{EM}_{\text{CRAG}}(q) \geq \text{EM}_{\text{Retrieval-only}}(q)
$$

**증명 스케치**:
- Retrieval-only 가 outdated document $d_{\text{old}}$ 만 가지면 → EM = 0
- CRAG 가 evaluator 로 $d_{\text{old}}$ 를 reject 하고 web search → 최신 정보 $d_{\text{new}}$ 획득 → EM > 0

**실험값**: CRAG 는 outdated query 에서 +18% EM 향상 (Self-RAG 는 그대로) $\square$

### 정리 5.5.2 — Evaluator Accuracy 와 Final Quality

Evaluator 의 accuracy 를 $\eta$ (현실: ~85%), routing decision 이 correct 일 확률을 $\rho$ 라 하면:

$$
\text{Quality}(\text{final}) \propto \eta + (1-\eta) \cdot \text{fallback\_penalty}
$$

**의미**: Evaluator 가 85% 정확하면, 15% 의 error 가 있어도 web search fallback 으로 부분 복구 가능. 이는 Self-RAG 의 reflection (internal only) 보다 우월 $\square$

### 정리 5.5.3 — Knowledge Refinement 의 Efficiency

Strip-level extraction 으로 context length 감소:

$$
\text{Length}_{\text{refined}} \ll \text{Length}_{\text{full}}
$$

**예**: Full document = 500 tokens, relevant strips = 50 tokens (10%)
- 생성 latency: $O(\text{Length})$ 따라 10배 개선 가능
- Quality: 거의 동일 (irrelevant sentence 제거로 noise 감소) $\square$

---

## 💻 Python / PyTorch 구현 검증

### 실험 1 — Retrieval Evaluator (T5) 훈련

```python
import torch
from transformers import T5ForSequenceClassification, T5Tokenizer
import torch.optim as optim

class RetrievalEvaluator:
    def __init__(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # T5 를 sequence classification 으로 (3 classes)
        self.model = T5ForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )
        self.class_labels = ["Supported", "Partially", "Not"]
    
    def evaluate(self, query: str, retrieved_doc: str):
        """
        Evaluate: does retrieved_doc support the query?
        """
        input_text = f"Question: {query} Retrieved: {retrieved_doc}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1)[0]
        conf = probs.max().item()
        pred_class = self.class_labels[probs.argmax().item()]
        
        return {
            "confidence": conf,
            "prediction": pred_class,
            "probabilities": {
                self.class_labels[i]: probs[i].item() for i in range(3)
            }
        }
    
    def train(self, train_examples: list, lr=1e-5, epochs=3):
        """
        train_examples: [(query, doc, label), ...]
        label: 0=Supported, 1=Partially, 2=Not
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for query, doc, label in train_examples:
                input_text = f"Question: {query} Retrieved: {doc}"
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                labels = torch.tensor([label])
                
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 사용 예
evaluator = RetrievalEvaluator()

# Evaluate an example
result = evaluator.evaluate(
    "Who is the current president of Germany?",
    "Angela Merkel was the chancellor of Germany from 2005 to 2021."
)
print(f"Result: {result}")
# Expected: confidence ~0.2-0.3, prediction="Not" (outdated)
```

### 실험 2 — Confidence-based Routing

```python
class CRAGRouter:
    def __init__(self, evaluator: RetrievalEvaluator, web_search_fn=None):
        self.evaluator = evaluator
        self.web_search_fn = web_search_fn or self.dummy_web_search
        
        # Routing thresholds
        self.tau_high = 0.7
        self.tau_low = 0.5
        self.tau_supported = 0.5
        self.tau_not = 0.5
    
    def dummy_web_search(self, query: str):
        """Placeholder for actual web search (e.g., DuckDuckGo, Google)"""
        return [f"Web result for: {query}"]
    
    def route(self, query: str, retrieved_docs: list):
        """
        Route based on evaluator confidence and class distribution
        """
        path = None
        selected_docs = []
        
        for doc in retrieved_docs:
            result = self.evaluator.evaluate(query, doc)
            conf = result["confidence"]
            pred = result["prediction"]
            probs = result["probabilities"]
            
            if conf > self.tau_high and probs["Supported"] > self.tau_supported:
                # High confidence in corpus retrieval
                path = "Retrieve"
                selected_docs.append(doc)
            elif conf < self.tau_low or probs["Not"] > self.tau_not:
                # Low confidence: web search
                path = "Web"
                web_results = self.web_search_fn(query)
                selected_docs.extend(web_results)
            else:
                # Ambiguous: both
                path = "Hybrid"
                selected_docs.append(doc)
                web_results = self.web_search_fn(query)
                selected_docs.extend(web_results)
        
        if path is None:
            path = "Web"  # Default to web search if all rejected
        
        return {
            "path": path,
            "documents": selected_docs,
            "num_docs": len(selected_docs)
        }

# 사용 예
router = CRAGRouter(evaluator)

query = "What is the capital of France in 2024?"
retrieved = ["Paris is the capital of France."]

routing_result = router.route(query, retrieved)
print(f"Routing path: {routing_result['path']}")
print(f"Selected docs: {len(routing_result['documents'])}")
```

### 실험 3 — Knowledge Refinement (Strip-level Selection)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class KnowledgeRefinement:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000)
    
    def extract_strips(self, query: str, document: str, num_strips: int = 3):
        """
        Extract most relevant sentences (strips) from document
        """
        # Split document into sentences
        import re
        sentences = re.split(r'[.!?]+', document)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) == 0:
            return []
        
        # Compute relevance: BM25-like (simplified with TF-IDF)
        all_texts = [query] + sentences
        try:
            tfidf_matrix = self.tfidf.fit_transform(all_texts)
            query_vec = tfidf_matrix[0]
            sent_vecs = tfidf_matrix[1:]
            
            # Cosine similarity
            similarities = (query_vec @ sent_vecs.T).toarray().flatten()
            
            # Top-k strips
            top_indices = np.argsort(similarities)[-num_strips:][::-1]
            strips = [sentences[i] for i in top_indices]
            
            return strips
        except:
            # Fallback if TF-IDF fails
            return sentences[:num_strips]
    
    def refine_context(self, query: str, documents: list, num_strips_per_doc: int = 2):
        """
        Refine multiple documents into concise strips
        """
        all_strips = []
        
        for doc in documents:
            strips = self.extract_strips(query, doc, num_strips_per_doc)
            all_strips.extend(strips)
        
        # Deduplicate and return
        unique_strips = list(set(all_strips))
        return unique_strips[:5]  # Max 5 strips

# 사용 예
refiner = KnowledgeRefinement()

query = "Who is the president of France?"
documents = [
    "Emmanuel Macron is the current president of France. He was elected in 2017. France is located in Western Europe. Paris is the capital.",
    "The French government is led by the Prime Minister. Emmanuel Macron serves as the President. France has a bicameral parliament."
]

refined_strips = refiner.refine_context(query, documents, num_strips_per_doc=2)
print("Refined context:")
for strip in refined_strips:
    print(f"  - {strip}")
```

### 실험 4 — CRAG Full Pipeline

```python
class CRAG:
    def __init__(self, evaluator, router, refiner, generator):
        self.evaluator = evaluator
        self.router = router
        self.refiner = refiner
        self.generator = generator  # LLM generator
    
    def generate_with_crag(self, query: str, retrieved_docs: list):
        """
        Full CRAG pipeline:
        1. Retrieve
        2. Evaluate & Route
        3. Refine knowledge
        4. Generate
        """
        # Step 1: Routing
        routing_result = self.router.route(query, retrieved_docs)
        selected_docs = routing_result["documents"]
        
        # Step 2: Knowledge refinement
        refined_strips = self.refiner.refine_context(query, selected_docs)
        context = " ".join(refined_strips)
        
        # Step 3: Generate
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        # In practice: use actual LLM (e.g., T5, GPT)
        # answer = self.generator.generate(prompt)
        
        return {
            "routing_path": routing_result["path"],
            "context": context,
            "num_strips": len(refined_strips),
            # "answer": answer
        }

# 사용 예
crag_pipeline = CRAG(evaluator, router, refiner, generator=None)

result = crag_pipeline.generate_with_crag(
    "Who is the president of France in 2024?",
    ["Emmanuel Macron has been president since 2017.",
     "France is a country in Europe."]
)

print(f"Routing path: {result['routing_path']}")
print(f"Context length (strips): {result['num_strips']}")
```

---

## 🔗 실전 활용

| 시나리오 | CRAG 선택 사유 | 주의점 |
|---------|-------|--------|
| Real-time news QA (latest info 필수) | Web search fallback 으로 최신 정보 반영 | Web API latency (100-500ms) 추가 |
| Medical/Legal (hallucination intolerant) | Evaluator confidence 로 신뢰도 control | Evaluator 훈련 data 충분해야 함 |
| Long-context summarization | Strip-level refinement 로 context 압축 | Relevant sentence 추출 정확도 중요 |
| Fact-checking (SNI 업무) | Hybrid mode 로 corpus + web 동시 활용 | 충돌하는 정보 처리 필요 |

---

## ⚖️ 가정과 한계

1. **Evaluator accuracy**: 85% 정도 → 15% error rate (perfect 아님)
2. **Web search latency**: 100-500ms → real-time application 에서 bottleneck 가능
3. **Strip extraction accuracy**: BM25/TF-IDF 기반 → semantic understanding 부족 가능
4. **Corpus freshness**: Pretraining 데이터는 고정 → 매우 최신 정보는 web search 필수
5. **Cost**: 각 query 에 evaluator forward pass + 가능한 web search → compute/API cost 증가

---

## 📌 핵심 정리

| 구성 | 역할 |
|------|------|
| Retrieval Evaluator (T5) | Retrieved doc 의 utility 평가 |
| Confidence-based Router | 세 가지 경로 선택 (Retrieve/Web/Hybrid) |
| Knowledge Refinement | Passage → strip-level 로 context 압축 |
| Web Search Fallback | Outdated/insufficient 시 최신 정보 |

$$
\boxed{\text{Path} = \begin{cases}
\text{Retrieve} & \text{if } \text{Conf} > \tau_h \\
\text{Web} & \text{if } \text{Conf} < \tau_l \\
\text{Hybrid} & \text{else}
\end{cases}}
$$

> **핵심**: CRAG 는 **evaluator-driven adaptive routing** 으로 corpus 의 한계 극복 — Self-RAG (internal) 에서 CRAG (external web) 로 진화.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: CRAG 의 "Hybrid" 경로에서 corpus-based retrieval 과 web search 결과를 어떻게 합치는가?

<details>
<summary>해설</summary>

두 가지 방식:

1. **Sequential (pipeline)**: 먼저 corpus, 안 되면 web
   - Latency: corpus (fast) + web (slow) 순차
   - Quality: web 이 corpus 를 override 가능

2. **Parallel (ensemble)**: 둘 다 동시에 수행, 결과 merge
   - Latency: max(corpus, web) = web latency 지배
   - Quality: 더 많은 정보, ensemble robustness

CRAG 논문: 기본적으로 sequential, 하지만 ambiguous 일 때만 parallel 추가

</details>

**문제 2 (심화)**: Evaluator 를 T5 base (110M params) 로 훈련했을 때, larger model (T5-large, 770M) 을 쓰면 성능이 얼마나 향상되는가? Compute cost 는?

<details>
<summary>해설</summary>

**성능 향상**:
- T5-base evaluator accuracy: ~85%
- T5-large evaluator accuracy: ~90% (5% 개선)
- T5-3B evaluator accuracy: ~92% (diminishing returns)

**Compute cost**:
- T5-base forward: ~10ms (GPU)
- T5-large forward: ~30ms (3배)
- T5-3B forward: ~100ms (10배)

**Trade-off decision**:
- Latency critical (QA, search): T5-base (10ms 추가 overhead, 최소)
- Accuracy critical (medical, legal): T5-large (30ms, 5% 개선 worth)
- Offline processing: T5-3B (92% accuracy, 시간 비용 무시)

</details>

**문제 3 (논문 비평)**: "CRAG 는 web search 로 outdated query 문제를 해결한다" 는 주장에서, web search 자체가 hallucination/misinformation 을 가져올 수 있지 않은가?

<details>
<summary>해설</summary>

**유효한 비판**:
- Web 정보는 항상 정확하지 않음 (fake news, outdated pages, scams)
- CRAG 의 evaluator 는 corpus 에 대해서만 훈련 (web content 에는 untested)

**방어**:
1. Web search 도 평판 기반 ranking (Google, Bing 은 이미 필터링)
2. CRAG 의 evaluator 를 web results 에도 적용 가능 (transfer learning)
3. Multi-source ensemble 으로 conflicting info 감지

**개선 방안**:
- Evaluator 를 web-aware 하게 재훈련 (web source 신뢰도 포함)
- Fact-checking module 추가 (knowledge base cross-reference)
- Multi-source voting (여러 web source 일치 확인)

결론: CRAG 는 corpus 한계 극복이 main value, web quality 는 separate concern.

</details>

---

<div align="center">

[◀ 이전 (04. Self-RAG)](./04-self-rag.md) · [📚 README](../README.md) · [다음 ▶ (06. FiD)](./06-fid.md)

</div>
