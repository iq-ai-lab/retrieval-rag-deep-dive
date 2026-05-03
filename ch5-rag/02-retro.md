# 02. RETRO — Chunked Cross-Attention (Borgeaud 2022)

## 🎯 핵심 질문

- Vanilla RAG 의 bottleneck (최대 k 개 passage 만 처리, 모든 token 에 동일 passage 사용) 을 어떻게 풀 것인가?
- 64-token chunk 단위로 retrieval 해서 **chunk-level 정보** 를 transformer layer 내부에 직접 통합하는 구조는?
- "Chunked Cross-Attention" 수학이 standard attention 과 어떻게 다른가?
- 2T token 을 사전학습한 RETRO 가 같은 scale 의 dense decoder 보다 왜 "25배 작은 모델로 비슷한 성능"을 낼 수 있는가?

---

## 🔍 왜 이 기법이 retrieval·RAG 에 중요한가

Vanilla RAG 의 한계:
1. **Passage 수 제한**: top-k (보통 k=5-10) 개만 처리 → retriever quality 에 과도하게 의존
2. **Token-level 적응성 부족**: 모든 output token 이 같은 passage set 사용 → fine-grained matching 불가능
3. **추가 모듈 구조**: retriever 와 generator 가 분리되어 있어 end-to-end training 어려움

RETRO (Retrieval-Enhanced Transformer) 의 혁신:
- **Chunk 단위 retrieval**: 64-token chunk 에 대해 nearest neighbor 검색 (top-2 chunks)
- **Chunked Cross-Attention**: transformer 의 각 layer 내부에 retrieved chunks 를 dynamic cross-attention → 모든 layer 에서 retrieval 정보 활용
- **Scale efficiency**: 2T token 학습 시 대략 25배 작은 모델 (Retro-3B ≈ Baseline-70B)

현대 RAG 시스템은 RETRO 의 in-layer retrieval 개념을 많이 차용.

---

## 📐 수학적 선행 조건

- Transformer attention 메커니즘 (Query, Key, Value)
- Dense retrieval (DPR, BM25)
- Nearest neighbor search (approximate 또는 exact)
- Cross-attention (decoder-encoder attention 개념)

---

## 📖 직관적 이해

### RETRO 의 아키텍처

```
입력 시퀀스 (길이 L = 2048)
  ↓
청크 분할: [chunk_0, chunk_1, ..., chunk_n] (각 64 tokens)
  ↓
각 청크에 대해:
┌─────────────────────────────────────┐
│ Retrieval (offline index 검색)       │
│ - chunk 의 prefix 를 query 로 사용   │
│ - corpus 에서 top-2 similar chunks   │
└─────────────────────────────────────┘
  ↓
Retrieved chunks: [Z_1, Z_2]
  ↓
┌──────────────────────────────────────────┐
│ Transformer layers (with CCA)             │
│ layer 0:  attention + CCA(query, chunks)  │
│ layer 1:  attention + CCA(query, chunks)  │
│ ...                                        │
│ layer L: attention + CCA(query, chunks)   │
└──────────────────────────────────────────┘
  ↓
Output tokens
```

### Chunked Cross-Attention 직관

```
Standard Self-Attention:
  Query, Key, Value 모두 같은 시퀀스에서
  
RETRO Chunked Cross-Attention:
  Query: 현재 시퀀스 (encoding)
  Key, Value: Retrieved chunks (외부 지식)
  
효과: 매 layer 마다 retrieved context 를 
      "명시적으로 mix-in" → layer 의 표현력 향상
```

---

## ✏️ 엄밀한 정의

### 정의 5.2.1 — Chunk 와 Retrieval

Input sequence $x = (x_1, \ldots, x_L)$ 를 **chunks** 로 분할:
$$
\text{Chunk}_i = (x_{64(i-1)+1}, \ldots, x_{64i}), \quad i = 1, \ldots, \lceil L/64 \rceil
$$

각 chunk $c_i$ 에 대해 **prefix** (처음 32 tokens) 를 retrieval query 로 사용하고, corpus 에서 top-2 nearest chunks $z_i^{(1)}, z_i^{(2)}$ 를 검색.

### 정의 5.2.2 — Chunked Cross-Attention (CCA)

표준 self-attention:
$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

Chunked Cross-Attention (각 layer $\ell$ 에서):
$$
\text{CCA}_\ell(h, Z) = \text{softmax}\left(\frac{h \cdot W_Q^\ell \cdot (W_K^\ell Z)^\top}{\sqrt{d}}\right) W_V^\ell Z
$$

- $h$: 현재 시퀀스의 hidden state (layer $\ell-1$ output)
- $Z$: retrieved chunks 의 concatenation (각 chunk 에서 top-2)
- $W_Q, W_K, W_V$: learnable projection matrices

### 정의 5.2.3 — Layer 내 병합

RETRO layer $\ell$ 의 출력:
$$
h^{(\ell)} = \text{Norm}(h^{(\ell-1)} + \text{SelfAttn}(h^{(\ell-1)}))
$$
$$
h^{(\ell)} = \text{Norm}(h^{(\ell)} + \text{CCA}(h^{(\ell)}, Z))
$$
$$
h^{(\ell)} = \text{Norm}(h^{(\ell)} + \text{FFN}(h^{(\ell)}))
$$

---

## 🔬 정리와 증명

### 정리 5.2.1 — Chunked Cross-Attention 의 효율성

RETRO-3B (2T tokens 학습) 의 성능이 Baseline-70B (2T tokens, retrieval 없음) 과 유사한 이유:

**증명 스케치**:
1. **파라미터 효율**: $3B \ll 70B$ — 약 23배 차이
2. **Effective context**: RETRO 는 retrieved chunks 로 "virtual context" 확장 (2개 chunks × 64 tokens = 추가 128 tokens per step)
3. **Knowledge reuse**: Top-2 chunks 검색 시, 이전 학습과정에서 유사 패턴을 본 모델이 더 효율적 학습 가능

정량화: Scaling law $L_{\mathrm{val}} = a(N + M_r)^{-\alpha}$ 에서 $M_r$ = retrieved token 효과:
$$
L_{\mathrm{RETRO}}(3B + \text{Retrieved}) \approx L_{\mathrm{Baseline}}(70B)
$$

이는 retrieval 이 약 23배 scaling law 개선을 주는 것으로 해석 가능 $\square$

### 정리 5.2.2 — Nearest Neighbor 의 정확도 요구사항

Chunk $c_i$ 의 prefix 로 검색할 때, top-k recall 이 generation 품질에 직접 영향:

$$
\text{Perplexity}(c_i | Z_i) \leq \text{Perplexity}(c_i) - \alpha \cdot \mathrm{Recall@k}
$$

**실험값**: RETRO 논문에서 top-2 recall ≈ 60% → perplexity 약 15% 개선. Top-10 recall ≈ 80% → 약 20% 개선.

따라서 **approximate nearest neighbor** (FAISS IVF) 가 아닌 **exact KNN** 필요 (inference 시 recall 을 위해) $\square$

### 정리 5.2.3 — Memory/Compute Trade-off

CCA 계산량: retrieved chunks 수 $k=2$, chunk length $=64$ 일 때,
$$
\text{FLOPs}_{\text{CCA}} = O(L \cdot d \cdot 2 \cdot 64) = O(128Ld)
$$

Standard self-attention:
$$
\text{FLOPs}_{\text{Self}} = O(L^2 d)
$$

**Trade-off**: CCA 는 $O(L)$ 에 가깝지만, retrieval 오버헤드 (offline index search) 존재. Inference 시 token per second 는 추가 latency 감수 $\square$

---

## 💻 Python / PyTorch 구현 검증

### 실험 1 — Chunk 분할 및 Prefix 추출

```python
import torch
import torch.nn as nn

class ChunkRetriever:
    def __init__(self, chunk_size=64, prefix_size=32):
        self.chunk_size = chunk_size
        self.prefix_size = prefix_size
    
    def chunk_sequence(self, x: torch.Tensor):
        """
        x: (batch, seq_len, d)
        return: list of chunks (batch, chunk_size, d)
        """
        B, L, d = x.shape
        chunks = []
        for i in range(0, L, self.chunk_size):
            end = min(i + self.chunk_size, L)
            chunks.append(x[:, i:end, :])
        return chunks
    
    def get_prefix(self, chunk: torch.Tensor):
        """chunk: (batch, chunk_size, d)"""
        return chunk[:, :self.prefix_size, :]  # (batch, prefix_size, d)
    
    def prefix_embedding(self, prefix: torch.Tensor):
        """prefix 의 CLS token 또는 mean pooling"""
        return prefix.mean(dim=1)  # (batch, d)

# 사용 예
seq = torch.randn(2, 2048, 768)  # batch=2, seq_len=2048, d=768
retriever = ChunkRetriever(chunk_size=64, prefix_size=32)
chunks = retriever.chunk_sequence(seq)
print(f"Number of chunks: {len(chunks)}")  # 2048/64 = 32

prefix_emb = retriever.prefix_embedding(retriever.get_prefix(chunks[0]))
print(f"Prefix embedding shape: {prefix_emb.shape}")  # (2, 768)
```

### 실험 2 — Chunked Cross-Attention 구현

```python
class ChunkedCrossAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=12, k_retrieved=2, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_retrieved = k_retrieved
        self.chunk_size = chunk_size
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, h, Z_retrieved):
        """
        h: query (batch, seq_len, d_model)
        Z_retrieved: retrieved chunks (batch, seq_len, k*chunk_size, d_model)
                     또는 flattened
        """
        B, L, d = h.shape
        
        # Query projection
        Q = self.W_Q(h)  # (B, L, d)
        
        # Key, Value projection on retrieved chunks
        K = self.W_K(Z_retrieved)  # (B, L, k*chunk_size, d)
        V = self.W_V(Z_retrieved)
        
        # Attention weights (simplified, single-head)
        # 실제로는 multi-head 필요
        scores = torch.matmul(Q.unsqueeze(2), K.transpose(-2, -1)) / (d ** 0.5)
        # scores: (B, L, 1, k*chunk_size)
        
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V).squeeze(2)  # (B, L, d)
        
        output = self.out_proj(context)
        return output

# 사용 예
h = torch.randn(2, 2048, 768)
Z = torch.randn(2, 2048, 128, 768)  # 64 tokens * 2 chunks
cca = ChunkedCrossAttention(d_model=768, num_heads=12)
output = cca(h, Z)
print(f"CCA output shape: {output.shape}")  # (2, 2048, 768)
```

### 실험 3 — FAISS 를 이용한 Chunk Retrieval

```python
import faiss
import numpy as np

class ChunkIndexRetrieval:
    def __init__(self, d_model=768):
        self.d_model = d_model
        self.index = None
        self.chunk_db = []
    
    def build_index(self, chunks_embeddings: np.ndarray):
        """
        chunks_embeddings: (num_chunks, d_model)
        """
        # L2 norm
        faiss.normalize_L2(chunks_embeddings)
        
        # IVF index (approximate nearest neighbor)
        nlist = max(1, len(chunks_embeddings) // 100)  # cluster 수
        quantizer = faiss.IndexFlatL2(self.d_model)
        self.index = faiss.IndexIVFFlat(quantizer, self.d_model, nlist, faiss.METRIC_L2)
        self.index.train(chunks_embeddings)
        self.index.add(chunks_embeddings)
    
    def retrieve(self, prefix_embeddings: np.ndarray, k=2):
        """
        prefix_embeddings: (batch*num_chunks, d_model)
        return: (batch*num_chunks, k) indices
        """
        faiss.normalize_L2(prefix_embeddings)
        distances, indices = self.index.search(prefix_embeddings, k)
        return indices

# 사용 예 (offline 에서 미리 실행)
# chunks_emb: 전체 corpus 의 chunk embeddings
# retrieval = ChunkIndexRetrieval()
# retrieval.build_index(chunks_emb.cpu().numpy())
```

### 실험 4 — RETRO Layer 의 전체 파이프라인

```python
class RETROLayer(nn.Module):
    def __init__(self, d_model=768, num_heads=12, ffn_dim=3072):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cca = ChunkedCrossAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, h, Z_retrieved):
        """
        h: (batch, seq_len, d_model)
        Z_retrieved: (batch, seq_len, k*chunk_size, d_model)
        """
        # Self-attention
        h_sa, _ = self.self_attn(h, h, h)
        h = self.norm1(h + h_sa)
        
        # Chunked Cross-Attention
        h_cca = self.cca(h, Z_retrieved)
        h = self.norm2(h + h_cca)
        
        # FFN
        h_ffn = self.ffn(h)
        h = self.norm3(h + h_ffn)
        
        return h

# 사용 예
layer = RETROLayer(d_model=768)
h = torch.randn(2, 2048, 768)
Z = torch.randn(2, 2048, 128, 768)
h_out = layer(h, Z)
print(f"RETRO layer output: {h_out.shape}")
```

---

## 🔗 실전 활용

| 시나리오 | RETRO 선택 사유 | 주의점 |
|---------|-------|--------|
| 사전학습 (domain-specific) | Chunk-level retrieval 로 knowledge 효율화 | Offline index 구축 필수 (시간 소비) |
| Long document understanding | In-layer retrieval 로 context 확장 | Retrieved chunks 의 quality 중요 (top-2 recall) |
| Knowledge-heavy tasks (QA, factoid) | Effective scaling: 작은 모델 + retrieved context | Token-per-second 는 느림 (retrieval overhead) |
| Multi-domain training | Chunk DB 를 여러 domain 에 공유 | Domain shift 시 retrieval quality 하락 가능 |

---

## ⚖️ 가정과 한계

1. **Retrieval 정확도 가정**: Top-2 recall 이 ~60% 이상 필요 — 낮으면 학습 부진
2. **Chunk 크기 고정**: 64 tokens 는 임의 선택 — task 별로 최적화 필요
3. **Prefix 기반 검색**: 32-token prefix 는 짧을 수 있음 — long document 에서는 context 부족
4. **Offline retrieval**: Inference 시 retrieved chunks 는 고정 (streaming/adaptive retrieval 불가)
5. **Indexing cost**: 2T token corpus 에 대해 chunk embedding + FAISS index 구축 시 며칠 소요

---

## 📌 핵심 정리

$$
\boxed{h^{(\ell)} = \text{Norm}(h^{(\ell-1)} + \text{CCA}(h^{(\ell-1)}, Z^{(\ell)}))}
$$

| 구성 | 설명 |
|------|------|
| Chunk-based Retrieval | 64-token chunks, top-2 nearest (prefix-based) |
| In-layer Integration | 각 transformer layer 에 CCA 추가 |
| Scaling Efficiency | 3B model + retrieved context ≈ 70B baseline |
| Compute/Quality Trade-off | Retrieved chunks cost $O(Ld)$ (negligible) but retrieval latency |

> **핵심**: RETRO 는 **retrieval 을 모델의 내부 메커니즘으로 승격** — 이후 모든 retrieval-augmented 모델의 영감.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: 64-token chunk 에서 prefix 를 32 tokens 로 설정한 이유는? (hint: precision-recall trade-off)

<details>
<summary>해설</summary>

32 tokens (약 6-8 단어) 는 query 의 **주요 의미**를 충분히 포함하면서도, **중복 검색** (유사 prefix 의 false positives) 을 피하는 균형점. 더 짧으면 (예: 8 tokens) → recall 높지만 precision 낮음. 더 길면 (예: 64 tokens) → precision 높지만 모든 정보를 담지 못함 (prefix 만 선택했으므로).

실제로 논문에서는 32/64 비율을 실험적으로 검증 (ablation).

</details>

**문제 2 (심화)**: "RETRO 의 chunk 단위 retrieval 이 dense passage 전체 retrieval 보다 왜 효율적인가?" 를 dense retriever 의 입장에서 비판.

<details>
<summary>해설</summary>

**RETRO 의 장점**:
- Chunk-level 은 fine-grained matching (64 tokens = 한 주제)
- Prefix 기반 query 는 encoding cost 감소 (전체 passage 대신 32 tokens)
- Top-2 수준의 낮은 k 로 충분 (높은 정확도)

**비판**:
- 32-token prefix 는 context 불충분할 수 있음 (예: ambiguous referent "그것은...")
- Chunk DB 는 학습 후 고정 → dynamic knowledge 반영 불가 (REALM 과 달리)
- IVF index 는 approximate → exact nearest neighbor 와 차이 (논문에서는 exact KNN 사용, 비싼 계산)

**해결책**: Atlas (03장) 는 이 문제들을 retriever 의 end-to-end 학습으로 개선.

</details>

**문제 3 (논문 비평)**: RETRO 가 2T tokens 로 학습된 baseline (non-retrieval) 과 비교했을 때, "같은 데이터 양" 에서 공정한 비교인가? Compute 비용 관점에서.

<details>
<summary>해설</summary>

**공정하지 않은 측면**:
- RETRO 의 training 시간 은 retrieval overhead (chunk retrieval, CCA 계산) 로 인해 약 2-3배 느림
- Effective "training FLOPs" 는 baseline 보다 훨씬 많을 가능성
- 즉, RETRO-3B (2T tokens, 느린 training) vs Baseline-70B (2T tokens, 빠른 training) 는 **wall-clock 시간** 에서 비슷할 수도 있음

**공정한 비교**:
- "같은 FLOPs" 기준으로 비교해야 정확함
- 논문에서는 언급이 있지만, 명시적 FLOPs 카운트는 부족

**의의**: 그럼에도 "retrieval 이 scaling 효율성 개선" 이라는 핵심은 유효.

</details>

---

<div align="center">

[◀ 이전 (01. Vanilla RAG)](./01-vanilla-rag.md) · [📚 README](../README.md) · [다음 ▶ (03. REALM · Atlas)](./03-realm-atlas.md)

</div>
