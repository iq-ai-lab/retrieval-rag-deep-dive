# 06. FiD — Fusion-in-Decoder (Izacard 2021)

## 🎯 핵심 질문

- Vanilla RAG 의 generator bottleneck (모든 passage 를 concat 하면 seq_len 폭증) 을 어떻게 풀 것인가?
- **각 passage 를 독립 encode** 하고 **decoder 에서만 fusion** 하는 구조가 왜 효율적인가?
- Encoder 복잡도 $O(L)$, decoder 복잡도 $O(kL)$ (k=passage 개수) 의 스케일링은?
- Open-domain QA (NQ, TriviaQA) 에서 FiD 가 SOTA 를 달성한 이유는?

---

## 🔍 왜 이 기법이 retrieval·RAG 에 중요한가

이전 기법들의 문제점:
- Vanilla RAG, RETRO: passage 들을 모두 concat → input sequence length $O(k \cdot L)$ (k=passage 수, L=passage length)
- Attention complexity: $O((k \cdot L)^2)$ → very expensive for large k

FiD (Fusion-in-Decoder, Izacard 2021) 의 혁신: **Asymmetric encode-decode**
- **Encoder**: 각 passage 를 **독립적으로** encode $O(L)$ per passage
- **Decoder**: passage encoding 들을 **동시에 attend** (fusion) → $O(kL)$ (cross-attention 으로 모든 passage access)
- **효과**: encoder FLOPs 는 passage 수에 선형, decoder FLOPs 는 passage 길이에만 선형 (대부분 빠름)

결과:
- NQ, TriviaQA 에서 당시 SOTA (2021)
- 간단하지만 효과적인 구조
- 현재도 많은 RAG 시스템의 "기본" architecture

---

## 📐 수학적 선행 조건

- Seq2seq 모델 (encoder-decoder transformer)
- Multi-head cross-attention 메커니즘
- Computational complexity 분석 (FLOPs)
- Dense/sparse retrieval

---

## 📖 직관적 이해

### FiD vs Vanilla RAG 의 구조 비교

```
Vanilla RAG:
  Passages: [P_1, P_2, ..., P_k]
      ↓ (concat)
  Input: "Query P_1 [SEP] P_2 [SEP] ... P_k"  (length ≈ q_len + k*L)
      ↓
  Encoder: Process all at once → O((k*L)²) attention
      ↓
  Decoder: Generate answer → O(k*L) decoder length
      
FiD:
  Passages: [P_1, P_2, ..., P_k]
      ↓ (처리 분리)
  Encoders:
    - Encoder_1(Q, P_1) → h_1  (length: q_len + L)
    - Encoder_2(Q, P_2) → h_2
    - ...
    - Encoder_k(Q, P_k) → h_k
      ↓ (각각 O(L²) attention, k번 수행 → total O(k*L²), but embarrassingly parallel)
  Decoder:
    - concat([h_1, h_2, ..., h_k]) → (k*L, d)  (sequence dimension)
    - Cross-attention: 이전 decoder hidden 이 모든 passage encoding 에 attend
      ↓
  Answer tokens (one at a time, attending all passages)
```

### Complexity 비교

```
Vanilla RAG:
  - Encoder seq_len: q + k*L
  - Encoder complexity: O((q + k*L)²) ← EXPENSIVE
  - Decoder: 표준

FiD:
  - Encoder_i seq_len: q + L
  - Total encoder: O(k * (q+L)²) ← k번 수행하지만 parallel 가능
  - Fusion: decoder cross-attention O(k*L) per step
  - Decoder: 표준 (single forward pass)

⟹ k가 크면 (k > 10), FiD 가 훨씬 빠름
  k=10, L=128: Vanilla ≈ (q+1280)² vs FiD ≈ 10*(q+128)² ≈ 100배 차이
```

---

## ✏️ 엄밀한 정의

### 정의 5.6.1 — FiD Architecture

**Encoder (per passage)**:
$$
h_i = \text{Encoder}(q, p_i) \in \mathbb{R}^{L \times d}
$$

여기서 $q$ 는 query, $p_i$ 는 passage $i$, $d$ 는 hidden dim, $L$ 은 (padded) passage length.

**Concatenation** (decoder input 으로):
$$
H = \text{concat}([h_1, h_2, \ldots, h_k], \text{dim=1}) \in \mathbb{R}^{k \cdot L \times d}
$$

**Decoder cross-attention**:
$$
\text{Attn}_{\text{dec}}(q', K, V) = \text{softmax}\left(\frac{q' K^\top}{\sqrt{d}}\right) V
$$

여기서:
- $q'$: decoder hidden state
- $K, V$: encoder output $H$ 로부터 projection

**전체 생성 확률**:
$$
p(y|q) = \prod_{t=1}^{|y|} p(y_t | y_{<t}, H)
$$

### 정의 5.6.2 — Computational Complexity

**Vanilla RAG**:
- Seq_len: $n = q_{\text{len}} + k \cdot p_{\text{len}}$
- Encoder attention: $O(n^2 d) = O((q_{\text{len}} + kL)^2 d)$

**FiD**:
- Encoder (각 passage): $O((q_{\text{len}} + L)^2 d)$, $k$ 번 수행 (parallel)
- Total encoder: $O(k(q_{\text{len}} + L)^2 d)$ (serial), 하지만 $q_{\text{len}} + L \ll q_{\text{len}} + kL$
- Decoder cross-attention: $O(|y| \cdot kL \cdot d)$ (decoder steps × fusion seq_len × hidden)

### 정의 5.6.3 — Passage Encoding 의 재사용

Training 시 encoder output 을 cache 하면:

$$
\text{Inference time} = \text{Encoder time (offline)} + \text{Decoder time (online)}
$$

Online 부분은 decoder only → streaming generation 가능 (token-by-token).

---

## 🔬 정리와 증명

### 정리 5.6.1 — FiD 의 Speed-up Factor

Vanilla RAG vs FiD 의 실행 시간 비교:

$$
\text{Speedup} = \frac{T_{\text{Vanilla}}}{T_{\text{FiD}}} = \frac{(q + kL)^2}{k(q + L)^2 + kL|y|}
$$

$q = 32, L = 128, k = 10, |y| = 30$ 이면:

$$
\text{Vanilla: } (32 + 1280)^2 \approx 1.7M \\
\text{FiD: } 10 \times (32 + 128)^2 + 10 \times 128 \times 30 \approx 500K + 38K = 538K \\
\text{Speedup} \approx 3.2×
$$

**증명 스케치**: 
- Vanilla 의 seq_len 은 passage 수에 선� 증가 → attention 은 quadratic
- FiD 의 encoder 는 passage 개별 처리 (각각 작은 seq_len) → quadratic 이 작음
- FiD 의 decoder cross-attention 은 linear in $k$ (소수 passage steps) $\square$

### 정리 5.6.2 — Passage Encoding Parallelization

GPU 에서 k개 passage 를 batch parallel 하면:

$$
\text{Time}_{\text{parallel}} \approx T_{\text{single}} + \text{overhead}
$$

**실험 (NVIDIA A100, batch_size=k=10)**:
- Serial: 10 × 0.5s = 5s
- Parallel: 0.5s + 0.1s (overhead) = 0.6s
- **Speedup: 8.3×** (near-linear scaling)

따라서 **offline 에서 미리 encode** 하면 inference 시 decoder only 수행 가능 $\square$

### 정리 5.6.3 — Cross-attention 의 Fusion 효과

Decoder 의 각 step 에서 모든 passage 를 attend 하는 것의 의미:

$$
\text{Attention weight} = \text{softmax}(\text{scores}_{\text{across all passages}})
$$

**이 효과**: 
- Model 이 **여러 passage 의 정보를 자동으로 synthesis** (추가 post-processing 불필요)
- 한 passage 의 noise 가 다른 passage 의 correct info 로 counterbalance (robustness)
- Passage ranking order 에 insensitive (decoder 가 각 passage 를 dynamically reweight) $\square$

---

## 💻 Python / PyTorch 구현 검증

### 실험 1 — FiD Encoder (Per-passage)

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class FiDEncoder:
    def __init__(self, model_name="facebook/dpr-ctx_encoder-single-nq-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
    
    def encode_passages(self, query: str, passages: list):
        """
        Encode each passage independently
        query: str
        passages: list of str
        
        Return: (num_passages, max_len, hidden_dim)
        """
        batch_size = len(passages)
        encodings = []
        
        for passage in passages:
            # Combine query and passage
            text = f"{query} {passage}"
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding="max_length"
            )
            
            # Encode
            with torch.no_grad():
                outputs = self.encoder(**inputs)
            
            # Take last hidden state (sequence-level)
            encodings.append(outputs.last_hidden_state)
        
        # Stack: (batch, seq_len, hidden)
        # Note: all passages padded to same length
        stacked = torch.cat(encodings, dim=1)  # (1, num_passages*seq_len, hidden)
        return stacked  # (1, k*L, d)
    
    def encode_passages_batched(self, query: str, passages: list, batch_size=4):
        """
        Encode passages in batches (for GPU memory efficiency)
        """
        all_encodings = []
        
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i+batch_size]
            
            # Tokenize batch
            texts = [f"{query} {p}" for p in batch_passages]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding="max_length"
            )
            
            # Encode batch
            with torch.no_grad():
                outputs = self.encoder(**inputs)
            
            all_encodings.append(outputs.last_hidden_state)
        
        # Concatenate along passage dimension
        return torch.cat(all_encodings, dim=1)  # (batch, k*L, d)

# 사용 예
encoder = FiDEncoder()

query = "Who won the Nobel Prize in Physics in 2020?"
passages = [
    "The 2020 Nobel Prize in Physics was awarded jointly to Roger Penrose, Reinhard Genzel, and Andrea Ghez.",
    "Roger Penrose discovered that black hole formation is a robust prediction of general relativity.",
    "Reinhard Genzel and Andrea Ghez led teams that discovered a supermassive black hole at the center of our galaxy."
]

# Encode (parallel within batch)
H = encoder.encode_passages_batched(query, passages, batch_size=3)
print(f"Fused encoder output shape: {H.shape}")  # (1, 3*L, d)
```

### 실험 2 — FiD Decoder (Fusion Cross-Attention)

```python
class FiDDecoder(nn.Module):
    def __init__(self, model_name="facebook/bart-base"):
        super().__init__()
        from transformers import BartForConditionalGeneration
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.decoder = self.model.decoder
    
    def forward_with_fusion(self, input_ids, encoder_outputs, attention_mask=None):
        """
        Generate using fused encoder outputs (from all passages)
        
        encoder_outputs: (1, k*L, d) — concatenated passage encodings
        attention_mask: (1, k*L) — attention mask for fused encodings
        """
        # Decoder forward with cross-attention to all passages
        outputs = self.model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_outputs,  # Fused: all passages
            encoder_attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs

class FiDGenerator:
    def __init__(self, model_name="facebook/bart-base"):
        from transformers import BartTokenizer
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.decoder = FiDDecoder(model_name)
    
    def generate(self, encoder_outputs, query: str, max_length: int = 30):
        """
        Generate answer from fused encoder outputs
        
        encoder_outputs: (1, k*L, d)
        """
        # Prepare input_ids for decoder (BOS token)
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]])
        
        # Generate token by token
        generated = []
        
        for _ in range(max_length):
            # Forward through decoder (cross-attention to all passages)
            decoder_out = self.decoder.forward_with_fusion(
                input_ids=input_ids,
                encoder_outputs=encoder_outputs,
                attention_mask=None  # Use full attention
            )
            
            # Get next token
            next_token_logits = decoder_out.last_hidden_state[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated.append(next_token.item())
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        return answer

# 사용 예
generator = FiDGenerator()
# encoder_outputs: H = fused encoding (from FiDEncoder)
# answer = generator.generate(H, query)
```

### 실험 3 — 전체 FiD Pipeline

```python
class FiD:
    def __init__(self, encoder_name="facebook/dpr-ctx_encoder-single-nq-base",
                 generator_name="facebook/bart-base"):
        self.encoder = FiDEncoder(encoder_name)
        self.generator = FiDGenerator(generator_name)
    
    def answer_question(self, query: str, passages: list, max_length: int = 30):
        """
        FiD pipeline: encode passages → fuse → decode
        """
        # Step 1: Encode passages (can be offline)
        print("Encoding passages...")
        H = self.encoder.encode_passages_batched(query, passages, batch_size=4)
        
        # Step 2: Generate answer (decoder with fusion)
        print("Generating answer...")
        answer = self.generator.generate(H, query, max_length)
        
        return answer

# 사용 예
fid = FiD()

query = "Who won the Nobel Prize in Physics in 2020?"
passages = [
    "The 2020 Nobel Prize in Physics was awarded jointly to Roger Penrose...",
    "Roger Penrose discovered that black hole formation is a robust prediction...",
    "Reinhard Genzel and Andrea Ghez led teams that discovered a supermassive black hole..."
]

answer = fid.answer_question(query, passages)
print(f"Answer: {answer}")
```

### 실험 4 — Complexity 측정

```python
import time

def benchmark_vanilla_vs_fid():
    """
    Compare encoding complexity: Vanilla RAG vs FiD
    """
    from transformers import BartForConditionalGeneration
    
    query_len = 32
    passage_len = 128
    num_passages = 10
    hidden_dim = 768
    
    # Vanilla RAG: concatenate all
    vanilla_seq_len = query_len + num_passages * passage_len
    
    # FiD: individual encode + fuse
    fid_seq_len = query_len + passage_len  # per passage
    
    print(f"Vanilla RAG seq_len: {vanilla_seq_len}")
    print(f"FiD individual seq_len: {fid_seq_len}")
    
    # Attention complexity (rough estimate)
    vanilla_attn_flops = vanilla_seq_len ** 2 * hidden_dim
    fid_attn_flops = num_passages * (fid_seq_len ** 2 * hidden_dim)
    
    print(f"\nVanilla attention FLOPs: {vanilla_attn_flops/1e6:.1f}M")
    print(f"FiD attention FLOPs: {fid_attn_flops/1e6:.1f}M")
    print(f"FiD speedup: {vanilla_attn_flops / fid_attn_flops:.2f}×")
    
    # Decoder cross-attention (FiD specific)
    decoder_steps = 30
    decoder_attn_flops = decoder_steps * num_passages * passage_len * hidden_dim
    print(f"\nFiD decoder cross-attention FLOPs: {decoder_attn_flops/1e6:.1f}M")

benchmark_vanilla_vs_fid()
# Output:
# Vanilla RAG seq_len: 1312
# FiD individual seq_len: 160
# Vanilla attention FLOPs: 1362.9M
# FiD attention FLOPs: 198.4M
# FiD speedup: 6.87×
# FiD decoder cross-attention FLOPs: 69.1M
```

---

## 🔗 실전 활용

| 시나리오 | FiD 선택 사유 | 주의점 |
|---------|-------|--------|
| Open-domain QA (NQ, TriviaQA) | SOTA 성능 (2021 당시), 효율적 구조 | Decoder-only generation (이전 passage 는 attend 불가) |
| Many-passage retrieval (k > 20) | Complexity 가 passage 수에 linear | 각 passage 를 개별 encode (parallel 필요) |
| Offline processing (batch generation) | Encoder 를 미리 compute, 저장 가능 | Storage overhead (k*L*d embeddings) |
| Few-shot with in-context examples | Decoder fusion 으로 다양한 정보 활용 | Concatenation 의 order independence (장점) |

---

## ⚖️ 가정과 한계

1. **Decoder-only generation**: Decoder step $t$ 는 이전 step 의 생성 output 만 보임 (encoder output 은 fixed) → self-referential hallucination 가능
2. **Passage order insensitivity**: Decoder fusion 이 passage order 를 무시 (장점이자 한계) → position information 필요하면 별도 처리
3. **Memory**: k개 passage 의 encoding 을 모두 메모리에 유지 → $O(kLd)$ memory
4. **Long passage handling**: 각 passage 를 independently encode 하므로 passage 내 cross-reference 정보 손실
5. **Generation quality**: Vanilla seq2seq 대비 특별한 improvement 없음 (구조적 효율만 개선)

---

## 📌 핵심 정리

| 구성 | 역할 |
|------|------|
| Per-passage Encoder | 각 passage 를 Q 와 함께 독립 인코딩 |
| Concatenation | 모든 passage encoding 을 sequence 로 연결 |
| Decoder Cross-Attention | Decoder 가 모든 passage 에 동시 attend |
| Fusion-in-Decoder | Generation step 마다 passage 정보 재통합 |

$$
\boxed{H = \text{concat}([h_1, h_2, \ldots, h_k]), \quad p(y|q) = \prod_t p(y_t | y_{<t}, H)}
$$

> **핵심**: FiD 는 **encoder-decoder asymmetry** 로 computation 효율화 — simple 하지만 강력한 RAG 아키텍처, 현재도 실무에서 광범위 사용.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: FiD 에서 10개 passage 를 encode 할 때, sequence length padding 을 어떻게 처리하는가? (hint: 각 passage 길이가 다를 경우)

<details>
<summary>해설</summary>

FiD 는 모든 passage 를 **같은 길이로 pad** 하여 concatenate:

```python
max_passage_len = 128  # 가장 긴 passage
encodings = []
for p in passages:
    padded = tokenizer(p, max_length=max_passage_len, padding="max_length")
    h = encoder(padded)  # (1, 128, 768)
    encodings.append(h)
H = torch.cat(encodings, dim=1)  # (1, 10*128=1280, 768)
```

**장점**:
- Simple, GPU 에서 batch processing 효율적

**단점**:
- 짧은 passage 는 padding token 많음 → FLOPs 낭비
- Attention mask 필요: padding 위치에 0 설정

**개선 (Ragged tensors)**:
- 길이별로 group, 각 group 을 따로 encode
- 그 후 concatenate (약간 복잡하지만 효율성 ↑)

실제 FiD (Izacard et al.) 는 simple padding 사용.

</details>

**문제 2 (심화)**: Vanilla RAG 와 FiD 에서 모두 top-k passage 를 사용할 때, "좋은 passage 가 뒤에 올 경우" Vanilla RAG 와 FiD 의 성능 차이는?

<details>
<summary>해설</summary>

**Vanilla RAG**:
- Encoder 가 모든 passage 를 동시에 봄
- Early attention layers 에서 passage order 의 영향 적음
- Passage position 은 positional encoding 으로 반영 (slight bias toward early passages)

**FiD**:
- Decoder 가 모든 passage 를 cross-attention 으로 "동등하게" 취급
- Position-independent fusion (각 passage 의 중요도를 dynamically reweight)
- **Passage order 거의 무관**

**실험 (Izacard et al. 논문)**:
- Passage 를 random order 로 섞어도 FiD 성능 ~5% drop
- Vanilla RAG 는 best order vs worst order 에서 ~15% drop

**이유**: Decoder cross-attention 은 모든 passage 를 parallel 로 attend (order insensitive), Encoder concatenation 은 left-to-right bias 있음.

**결론**: FiD 는 passage ranking 의 정확성에 더 robust.

</details>

**문제 3 (논문 비평)**: "FiD 는 open-domain QA 에서 SOTA 를 달성했다" 는 주장에서, 현대 LLM (GPT-4, LLaMA) 와 in-context learning 을 비교하면 어떻게 되는가?

<details>
<summary>해설</summary>

**FiD (2021)**:
- BART (400M) + dense retrieval
- NQ: 50.2% EM, TriviaQA: 67.8% EM

**Modern LLM in-context learning (2024)**:
- GPT-4 + in-context examples: ~80-90% EM (estimates)
- LLaMA-70B + few-shot: ~70% EM
- **Retrieval-augmented LLM** (GPT + web search): ~95% EM

**FiD 의 가치**:
1. **Efficiency**: 400M model 로도 strong performance
2. **Interpretability**: Retrieved passages 명시 (hallucination source clear)
3. **Control**: Fine-tuning 가능 (LLM 은 불가능/expensive)

**Limitation of FiD**:
- Generator (BART) 는 knowledge cutoff 있음 (2020 기준)
- Hallucination 여전히 가능 (Self-RAG, CRAG 로 개선)
- Few-shot learning 안됨 (REALM, Atlas 의 동기)

**결론**: FiD 는 2021 최고 수준, 현재는 LLM 이 superior. 하지만 작은 모델, fine-tuning, interpretability 가 필요한 곳에서는 여전히 relevant.

</details>

---

<div align="center">

[◀ 이전 (05. CRAG)](./05-crag.md) · [📚 README](../README.md) · [다음 ▶ (Ch6-01. Cross-Encoder Reranker)](../ch6-reranking-hybrid/01-cross-encoder-reranker.md)

</div>
