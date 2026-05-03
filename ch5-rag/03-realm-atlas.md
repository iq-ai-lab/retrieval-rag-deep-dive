# 03. REALM · Atlas — End-to-End Trained RAG

## 🎯 핵심 질문

- RETRO 와 달리 **retriever 를 학습 가능**하게 만드는 방법은? (RETRO 의 retriever 는 frozen)
- **Inverse cloze task** (문장 내 passage 를 masked 토큰으로 복구) 를 pretraining 으로 사용하는 원리?
- MLM (Masked Language Modeling) gradient 가 어떻게 retriever 개선 신호를 제공하는가?
- Atlas (2024) 는 few-shot learning 에서 왜 T5+Contriever 조합으로 SOTA 를 달성하는가?

---

## 🔍 왜 이 기법이 retrieval·RAG 에 중요한가

이전 기법들의 한계:
- Vanilla RAG: retriever & generator 공동 학습이지만, 여전히 **curriculum 나쁨** (초기 retriever 품질 낮음)
- RETRO: in-layer retrieval 로 효율적이지만, **retriever 는 frozen** (사전학습된 BERT, 학습 불가)

REALM (2020) 의 혁신: **Inverse cloze task** 로 retriever 를 사전학습
- Passage 내 random span 를 mask → MLM loss 로 학습
- **Gradient flow**: MLM loss → retriever 를 통해 역전파 (어떤 passage 를 retrieve 했을 때 MLM 이 쉬워지는지 학습)

Atlas (2024) 의 진화: 
- Contriever (unsupervised dense retriever) + T5 (generator)
- Few-shot in-context learning 으로 prompt 활용
- REALM 의 아이디어를 현대 LLM scale 로 확장

---

## 📐 수학적 선행 조건

- EM 알고리즘 (latent variable 의 likelihood 최적화)
- Masked Language Modeling (BERT/RoBERTa)
- Gradient 추적 (어느 passage 선택이 loss 에 영향)
- Contrastive learning (Contriever 기초)

---

## 📖 직관적 이해

### REALM 의 학습 구조

```
원본 문서: "Paris is the capital of France."

Inverse Cloze Task:
  Mask random span: "Paris is the [MASK] of France."
  
REALM 의 과정:
  1. "Paris is the" 까지를 query 로 사용
  2. Corpus 에서 가장 유사 document 검색
       → "Paris, capital of France, is a major city."
  3. Retrieved document + masked sentence 를 MLM 으로 학습
       → gradient 가 retriever 를 개선 
           (capital 을 복구하기 좋은 doc 을 선택하도록)

이는 Vanilla RAG 의 "어떤 passage 가 답변에 도움?" 과 비슷하지만,
사전학습 레벨에서 dense retriever 를 끝까지 학습 가능하게 함
```

### Atlas 의 현대적 접근

```
Atlas = Contriever (retriever) + T5 (generator)

사전학습 (Unsupervised):
  - Contriever: BERT 기반, contrastive loss 로 dense vector 학습
  - 여러 데이터셋에서 (Wikipedia, C4 등) 재현 (in-batch negatives)

Fine-tuning (Supervised):
  - Few-shot: 예시 (query, answer) 를 prompt 로 제공
  - T5: retrieve doc + in-context examples 를 받고 답변 생성

SOTA 달성:
  - NQ, TriviaQA 에서 few-shot 으로 full-supervised 와 비슷 성능
  - 매개변수: 11B (매우 큼) 이지만, few-shot 의 flexibility
```

---

## ✏️ 엄밀한 정의

### 정의 5.3.1 — Inverse Cloze Task (REALM)

주어진 문서 $d = (d_1, \ldots, d_L)$ 에서:

1. **Span masking**: 길이 $\ell$ 의 random span $s$ 를 선택하고 mask token 으로 대체
   $$d' = (d_1, \ldots, d_{i-1}, [MASK], d_{i+\ell}, \ldots, d_L)$$

2. **Salient span query**: span 이전의 context $q = (d_1, \ldots, d_{i-1})$ 을 query 로 사용

3. **Retrieval-augmented MLM**: 
   $$\mathcal{L}_{\text{REALM}} = -\log p(s | [MASK], \text{RetrieveD}(q))
$$
   
   여기서 $\text{RetrieveD}(q)$ 는:
   $$
   \text{RetrieveD}(q) = \arg\max_d \mathrm{Score}(q, d)
$$

4. **두 모듈의 gradient**:
   - MLM loss 역전파: $\frac{\partial \mathcal{L}}{\partial d}$ (generator 개선)
   - 같은 loss: $\frac{\partial \mathcal{L}}{\partial \mathrm{Score}}$ (retriever 개선, 어떤 doc 을 retrieve 했을 때 loss 가 작아지는지)

### 정의 5.3.2 — Atlas: Retriever + Generator

**Retriever** (Contriever):
$$
p(d|q) = \frac{\exp(\mathrm{Enc}_q(q) \cdot \mathrm{Enc}_d(d) / \tau)}{\sum_{d'} \exp(\mathrm{Enc}_q(q) \cdot \mathrm{Enc}_d(d') / \tau)}
$$

**Generator** (T5):
$$
p(y|q, d_1, \ldots, d_k) = \prod_{i=1}^{|y|} p(y_i | y_{<i}, q, d_1, \ldots, d_k)
$$

**Joint loss**:
$$
\mathcal{L} = -\log p(y|q) = -\log \sum_{d_1, \ldots, d_k} p(y|q, d_1, \ldots, d_k) \prod_{j=1}^k p(d_j|q)
$$

---

## 🔬 정리와 증명

### 정리 5.3.1 — Inverse Cloze 로 학습된 Retriever 의 의미

REALM 의 retriever 는 **MLM loss 를 최소화하기 좋은 document 를 선택**:

$$
\frac{\partial \mathcal{L}_{\text{MLM}}}{\partial \mathrm{Score}(q, d)} = \frac{\partial \mathcal{L}}{\partial d_{\text{retrieved}}} \cdot \frac{\partial d_{\text{retrieved}}}{\partial \mathrm{Score}}
$$

**의미**: 만약 document $d$ 를 retrieve 했을 때 MLM loss 가 작아지면, retriever score 도 증가하도록 gradient 흐름 → **utility-driven retrieval** (실제로 도움이 되는 document 선택).

이는 **Vanilla RAG 의 generator gradient** 와 비슷하지만, **사전학습 단계에서부터** 적용 가능 $\square$

### 정리 5.3.2 — Few-shot Learning 에서의 Atlas

T5 모델의 in-context learning 능력이 있으면:

$$
p(y|q, \text{examples}) \geq p(y|q) \quad \text{(대부분의 경우)}
$$

**증명 스케치**:
- T5 는 multitask 학습 (SuperGLUE, GLUE, etc.) 으로 in-context understanding 이미 습득
- Retrieved document $d$ 에 few-shot examples 를 추가해도 "instruction-following" 으로 처리
- Atlas 에서는 prompt format: `{examples} Question: {q} Document: {d} Answer:` 로 구성

**결과**: Full-supervised REALM 과 유사 성능, few-shot 에서 유연함 $\square$

### 정리 5.3.3 — Contriever 의 Contrastive 학습

Atlas 의 retriever (Contriever) 는 다음 contrastive loss 로 학습:

$$
\mathcal{L}_{\text{contra}} = -\log \frac{\exp(q^+ \cdot d^+ / \tau)}{\sum_{(d,d') \in B} \exp(q^+ \cdot d' / \tau)}
$$

여기서 $d^+$ 는 positive document (정답 포함), $d'$ 는 in-batch negatives.

**Scaling**: Batch size $B$ 크면 negatives 많음 → stronger training signal (but FLOPs 증가) $\square$

---

## 💻 Python / PyTorch 구현 검증

### 실험 1 — Inverse Cloze Task 구현

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

class InverseCloseTask:
    def __init__(self, model_name="roberta-base", mask_len=5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.mask_len = mask_len
    
    def create_masked_example(self, document: str):
        """
        Random span 을 mask 하고, query 생성
        """
        tokens = document.split()
        
        # Random span 선택
        start = torch.randint(0, len(tokens) - self.mask_len, (1,)).item()
        masked_tokens = tokens.copy()
        masked_span = masked_tokens[start:start+self.mask_len]
        masked_tokens[start:start+self.mask_len] = ["[MASK]"] * self.mask_len
        
        # Query: mask 이전
        query_tokens = tokens[:start]
        query = " ".join(query_tokens)
        
        # Masked document
        masked_doc = " ".join(masked_tokens)
        
        # Answer: masked span
        answer = " ".join(masked_span)
        
        return query, masked_doc, answer
    
    def compute_mlm_loss(self, masked_doc: str, answer: str):
        """MLM loss 계산"""
        inputs = self.tokenizer(masked_doc, return_tensors="pt")
        labels = self.tokenizer(answer, return_tensors="pt").input_ids
        
        # Mask 토큰의 위치
        mask_token_idx = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero()[0, 1]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # MLM logits 에서 answer 의 첫 토큰 확률
        logits = outputs.logits[0, mask_token_idx, :]
        target_id = labels[0, 0]
        loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), target_id.unsqueeze(0))
        
        return loss.item()

# 사용 예
ict = InverseCloseTask()
doc = "Paris is the capital of France and is located in the north-central part of the country"
query, masked_doc, answer = ict.create_masked_example(doc)
loss = ict.compute_mlm_loss(masked_doc, answer)
print(f"Query: {query}")
print(f"Masked doc: {masked_doc}")
print(f"Answer: {answer}")
print(f"MLM loss: {loss:.4f}")
```

### 실험 2 — Retriever + Generator 공동 학습 (Simplified Atlas)

```python
from transformers import T5ForConditionalGeneration, AutoModel

class SimpleAtlas:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Retriever: Contriever-like (simplified: use BERT CLS)
        self.retriever = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        # Generator: T5
        self.generator = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
    
    def encode_query(self, query: str):
        inputs = self.retriever.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.retriever(**inputs)
        return outputs.pooler_output  # (1, 768)
    
    def encode_document(self, doc: str):
        inputs = self.retriever.tokenizer(doc, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.retriever(**inputs)
        return outputs.pooler_output
    
    def retrieve_and_generate(self, query: str, documents: list, answer: str, k=2):
        """
        1. Query encode
        2. Document 들과 유사도 계산
        3. Top-k doc retrieve
        4. Generator 에 전달 및 loss 계산
        """
        q_emb = self.encode_query(query)
        
        # Encode documents
        doc_embs = []
        for doc in documents:
            doc_embs.append(self.encode_document(doc))
        doc_embs = torch.cat(doc_embs, dim=0)
        
        # Retrieve top-k
        scores = torch.cosine_similarity(q_emb, doc_embs)
        top_k_idx = scores.topk(k)[1]
        top_docs = [documents[i] for i in top_k_idx.tolist()]
        
        # Generator: concat query + top docs
        input_text = f"Question: {query} Context: {' '.join(top_docs)}"
        inputs = self.generator.tokenizer(input_text, return_tensors="pt",
                                         max_length=512, truncation=True).to(self.device)
        labels = self.generator.tokenizer(answer, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.generator(input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                                labels=labels)
        
        return outputs.loss, scores

# 사용 예
atlas = SimpleAtlas()
query = "Who is the president of France?"
documents = [
    "Emmanuel Macron has been the President of France since 2017.",
    "The capital of France is Paris, located on the Seine River.",
    "France is known for its culture, wine, and fashion."
]
answer = "Emmanuel Macron"

loss, scores = atlas.retrieve_and_generate(query, documents, answer, k=2)
print(f"Generation loss: {loss.item():.4f}")
```

### 실험 3 — Contriever 기반 Dense Retrieval

```python
from sentence_transformers import SentenceTransformer

class ContrieverRetriever:
    def __init__(self, model_name="facebook/contriever"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: list):
        """Encode multiple texts"""
        return self.model.encode(texts, convert_to_tensor=True)
    
    def retrieve(self, query: str, documents: list, k=5):
        """Retrieve top-k documents"""
        q_emb = self.encode([query])
        doc_embs = self.encode(documents)
        
        scores = torch.nn.functional.cosine_similarity(q_emb, doc_embs)
        top_k_idx = scores.topk(k)[1]
        
        return [documents[i] for i in top_k_idx.tolist()], scores[top_k_idx].tolist()

# 사용 예
contriever = ContrieverRetriever()
query = "What is the capital of France?"
documents = [
    "Paris is the capital city of France.",
    "France has a population of about 67 million.",
    "The Eiffel Tower is located in Paris.",
    "French cuisine is famous worldwide.",
    "The Louvre Museum is in Paris."
]

retrieved, scores = contriever.retrieve(query, documents, k=3)
for doc, score in zip(retrieved, scores):
    print(f"({score:.4f}) {doc}")
```

### 실험 4 — Few-shot Prompting with Retrieved Docs (Atlas style)

```python
class FewShotAtlas:
    """Few-shot in-context learning with retrieved documents"""
    
    def __init__(self):
        self.generator = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.tokenizer = self.generator.tokenizer
        self.contriever = ContrieverRetriever()
    
    def format_few_shot_prompt(self, examples: list, query: str, documents: list):
        """
        examples: [(q1, a1), (q2, a2), ...]
        """
        prompt = ""
        for q, a in examples:
            prompt += f"Q: {q}\nA: {a}\n\n"
        
        doc_context = " ".join(documents)
        prompt += f"Context: {doc_context}\n"
        prompt += f"Q: {query}\nA:"
        
        return prompt
    
    def generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.generator.generate(inputs.input_ids, max_length=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# 사용 예
few_shot = FewShotAtlas()

examples = [
    ("What is the capital of France?", "Paris"),
    ("Who is the president of Germany?", "Olaf Scholz"),
]

query = "What is the capital of Germany?"
documents = ["Berlin is the capital city of Germany, located in the eastern part of the country."]

prompt = few_shot.format_few_shot_prompt(examples, query, documents)
answer = few_shot.generate(prompt)
print(f"Generated answer: {answer}")
```

---

## 🔗 실전 활용

| 시나리오 | REALM/Atlas 선택 사유 | 주의점 |
|---------|-------|--------|
| Domain-specific pretraining | Inverse cloze 로 retriever 를 corpus 에 적응 | Compute-heavy (gradient through retrieval) |
| Few-shot QA (NQ, TriviaQA) | T5 의 in-context learning + dense retrieval | 11B 모델 필수 (작은 모델은 few-shot 약함) |
| Rapid domain adaptation | Retriever + generator 공동 fine-tuning 가능 | Cold-start: 초기 retriever 품질 중요 |
| Knowledge updates | New documents 추가 시 retriever 재학습 (REALM 방식) | Incremental learning 비용 |

---

## ⚖️ 가정과 한계

1. **Inverse Cloze 의 representativeness**: Masked span 복구가 모든 task 와 대응되지 않음 (예: factoid QA 는 다름)
2. **Gradient through retrieval**: Top-k selection 은 discrete 연산 → straight-through estimator 등 필요 (논문에서 실제 구현은 복잡함)
3. **Scale 의존성**: Atlas 는 11B 모델 필수 — 작은 모델 (< 1B) 에서 few-shot 효과 미미
4. **Index freshness**: Pretraining 후 corpus 추가 시 (예: 2024 news) retriever 재학습 필수
5. **Compute cost**: End-to-end 학습은 매우 비쌈 (Atlas 의 11B 를 처음부터 학습은 수주 소요)

---

## 📌 핵심 정리

| 기법 | Retriever | Generator | 학습 방식 |
|------|-----------|-----------|---------|
| REALM | MLM-gradient driven | BERT+MLM | Inverse cloze pretraining |
| Atlas | Contriever (contrastive) | T5 | Few-shot fine-tuning |
| 공통점 | End-to-end learnable | Conditional generation | Marginal likelihood |

$$
\boxed{\mathcal{L} = -\log p(y|q) = -\log \sum_d p(y|q,d) p(d|q)}
$$

> **핵심**: REALM 과 Atlas 는 **retriever 를 joint training 으로 학습 가능**하게 한 첫 번째 대규모 시스템 — 이후 모든 학습 가능 RAG 의 기초.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: Inverse Cloze Task 에서 span 을 random 하게 mask 하는 이유는? (hint: curriculum learning)

<details>
<summary>해설</summary>

Random masking 은 모든 context length 에서 같은 학습 신호를 제공. 만약 항상 document 의 특정 부분만 mask 하면 (예: 항상 끝) → retriever 는 그 부분을 복구하기 좋은 document 만 선택 학습 (biased).

Random 으로 하면 → retriever 는 **다양한 context 에서 정보를 찾을 수 있는** 일반적 능력 습득.

이는 implicit curriculum: 짧은 span 은 denser 한 정보 (harder), 긴 span 은 broader (easier).

</details>

**문제 2 (심화)**: REALM 의 gradient-through-retrieval 을 실제로 구현할 때, top-1 document 선택이 discrete 연산인데 어떻게 역전파하는가?

<details>
<summary>해설</summary>

Vanilla backprop 은 discrete operation 을 통과 불가. REALM 논문의 실제 구현:

1. **Gumbel-softmax**: Document selection 을 soft 하게 만들어 gradient 통과 가능하게 함
   $$d = \text{softmax}(\text{scores} + \text{Gumbel}(0,1) / \tau)$$
   $\tau \to 0$ 으로 annealing 하면 one-hot 에 가까워짐.

2. **Straight-through estimator**: Forward pass 에서 one-hot select, backward pass 에서 soft select
   $$d_{\text{forward}} = \text{onehot}(\arg\max \text{scores})$$
   $$d_{\text{backward}} = \text{softmax}(\text{scores})$$

3. **Stop-gradient**: Retrieval score 만 학습하고, document embedding 은 고정 (REALM 이 선택한 방식)

실제로는 (3) 을 많이 사용 (가장 안정적).

</details>

**문제 3 (논문 비평)**: "Atlas 는 11B 매개변수로 few-shot 에서 full-supervised 와 비슷하다" 는 주장이 공정한가? 다른 모델 (GPT-3.5, LLaMA) 과의 비교는?

<details>
<summary>해설</summary>

**공정한 측면**:
- Atlas 의 강점: retrieval-augmented + T5 의 multitask fine-tuning → explicit knowledge + flexibility
- Few-shot in-context learning 은 LLaMA, GPT-3.5 도 잘 함

**공정하지 않은 측면**:
- Atlas 11B 는 사전학습 + fine-tuning (supervised QA data 사용) → full-supervised 와 비슷하게 비교하는 것이 맞음
- 하지만 GPT-3.5 (175B, 매우 큼) 와 비교하면 Atlas 의 효율성이 두드러짐
- 공정한 비교: 같은 scale 의 retrieval-free 모델 (T5-11B without retrieval) 과 비교

**결론**: Atlas 는 "같은 크기의 non-retrieval 모델 대비" 우월 (due to retrieval), "절대 성능" 은 GPT-scale 에 못미침 (이는 향후 개선 과제).

</details>

---

<div align="center">

[◀ 이전 (02. RETRO)](./02-retro.md) · [📚 README](../README.md) · [다음 ▶ (04. Self-RAG)](./04-self-rag.md)

</div>
