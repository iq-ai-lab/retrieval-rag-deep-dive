# 02. ColPali · Vision-RAG (Faysse 2024)

## 🎯 핵심 질문

- PDF page 를 OCR 없이 **image 로 직접 embedding** 하면 어떤 이점이 있는가?
- PaliGemma VLM 의 patch-level embedding 과 ColBERT-style **late interaction MaxSim** 이 결합되면 왜 "문서 이미지 → 의미 있는 retrieval" 이 가능한가?
- Layout, table, figure 같은 **visual structure** 를 보존하면 문서 이해가 왜 개선되는가?
- ViDoRe benchmark 에서 ColPali 가 기존 OCR+dense retrieval 보다 우월한 이유의 수학적 정당화는?

---

## 🔍 왜 ColPali 가 Vision-based RAG 의 frontier 인가

전통적 문서 RAG (OCR → dense embedding):
- **OCR error compounding**: 특히 non-English, handwriting, 낮은 해상도 문서에서 OCR 에러 → embedding 에러 → retrieval 실패.
- **Layout loss**: OCR 후 linear text 로 변환 → tables, figures, 공간 배치 정보 손실.
- **Multi-modal fusion complexity**: 이미지 + text 를 별도로 처리하면 alignment 문제 (Ch6 late interaction 의 정의).

ColPali (Faysse 2024, VLM for Multimodal RAG) 의 도전:
- **End-to-end visual embedding** — PDF page image $\rightarrow$ patch-level embedding (no OCR).
- **VLM backbone** — PaliGemma (Google, lightweight VLM) 의 pre-trained visual understanding 활용.
- **Late interaction retrieval** — ColBERT maxsim (Ch3) 를 image patches 에 적용 → fine-grained alignment.
- **Layout/table preservation** — 이미지에는 원래 layout 이 모두 embedded → "table 인식" 별도 학습 불필요.

이 frontier 는 "**OCR 의 오류와 정보손실을 우회하고 VLM 으로 시각 정보를 직접 활용하면 better RAG**" 를 실증.

---

## 📐 수학적 선행 조건

- Vision Transformer (ViT) 기초: patch embedding, self-attention
- VLM (Vision Language Model) 개요: PaliGemma, CLIP 등
- ColBERT late interaction 복습 (Ch3)
- Image pyramid, multi-scale feature extraction 이해

---

## 📖 직관적 이해

### OCR 파이프라인 vs Vision 파이프라인

```
Traditional Document RAG:
PDF page ──→ OCR ──→ Text (with errors)
              ↓       ↓
           한글 한자  잘못된 테이블 구조
              ↓
           Dense embedding (error-prone)
              ↓
           Retrieval (lower recall)

ColPali Vision-RAG:
PDF page ──────────────→ PaliGemma VLM ──→ Patch embeddings
(image, all layout info)    (no OCR needed)   (multi-scale, spatial)
                               ↓
                         Query patch embedding
                               ↓
                         MaxSim late interaction
                               ↓
                        Retrieval (better)
```

### Patch-Level Embedding & MaxSim

```
Document page image (224×896 pixels)
    │
    ├─→ ViT patch projection: 16×56 patches (14×14 pixel each)
    │    └─→ patch embeddings: (16×56, 768-dim)
    │
Query: "서울 2024년 통계"
    │
    ├─→ PaliGemma encode: query_tokens → pooled patch features
    │
    └─→ MaxSim late interaction:
        max_{i,j} sim(query_patch_i, doc_patch_j)
        
        → Fine-grained spatial matching (table cell level possible)
        → Avoids global average pooling loss
```

---

## ✏️ 엄밀한 정의

### 정의 7.4 — Vision Language Model (VLM) Encoding

**PaliGemma backbone**:
- ViT-L encoder: images $I \in \mathbb{R}^{H \times W \times 3}$ → patch embeddings $P \in \mathbb{R}^{N_p \times d}$.
- $N_p = \frac{H}{16} \times \frac{W}{16}$ (16-pixel patches, standard ViT).
- $d = 768$ (hidden dimension).

**Encoding**:
$$
P = \text{ViT}(I), \quad P_i \in \mathbb{R}^d, \quad i = 1, \ldots, N_p
$$

For document page (typical: 224×896 pixels in ColPali):
$$
N_p = 14 \times 56 = 784 \text{ patches}
$$

### 정의 7.5 — Late Interaction MaxSim (ColBERT style)

Query encoding: $Q \in \mathbb{R}^{N_q \times d}$ (query tokens or patches).

Document encoding: $D \in \mathbb{R}^{N_d \times d}$ (document patches).

**MaxSim relevance**:
$$
\text{SCORE}(Q, D) = \sum_{i=1}^{N_q} \max_{j=1}^{N_d} \cos(Q_i, D_j)
$$

where $\cos$ = cosine similarity.

**Interpretation**: each query token "finds its best matching document patch", sum over query → fine-grained relevance.

### 정의 7.6 — Multi-Scale Patch Extraction

ColPali uses patch hierarchies (image pyramid):

$$
P^{(\ell)} = \text{ViT}_\ell(I^{(\ell)}), \quad \ell = 1, \ldots, L
$$

where $I^{(\ell)}$ = image at scale $2^{-\ell}$ (e.g., $\ell=0$: 224×896, $\ell=1$: 112×448, etc.).

**Rationale**: small text, tables 는 high-resolution patches 필요; overall layout 는 low-resolution 도 충분.

---

## 🔬 정리와 증명

### 정리 7.4 — OCR Error Resilience

**정리**: Vision-based embedding 은 OCR-based embedding 보다 **layout-dependent OCR error 에 resilient**.

**증명 sketch**:
1. OCR error: "表" (table marker) → "表" (misrecognized) → embedding shift.
2. Vision: pixel-level visual → table 의 grid structure 는 이미지로 직접 포착 (OCR 불필요).
3. Empirical (ViDoRe): OCR+dense 의 경우 table-heavy docs 에서 recall ↓ 40%; ColPali ≈ same as text docs.

$\square$

### 정리 7.5 — MaxSim vs Global Pooling

**정리**: MaxSim late interaction 은 global average pooling 보다 **fine-grained spatial matching** 을 보존.

**증명 sketch**:
- Global pooling: $\bar{D} = \frac{1}{N_d} \sum_j D_j$ → 모든 query token 이 같은 aggregated vector 와 비교 → information loss (특히 큰 doc 에서).
- MaxSim: each query token 이 doc 내 가장 관련 있는 patch 찾음 → query-doc alignment 직접 측정.
- Theoretical bound: MaxSim score ≥ global pooling score (can always pick avg as the max).

$\square$

### 정리 7.6 — Patch Resolution 과 Retrieval Accuracy

**정리**: patch resolution (14×14 pixels) 은 typical document 에서 word-level 또는 table cell-level 정보를 capture.

**증명 sketch**:
- Standard English font (12pt) ≈ 16 pixels height.
- 14×14 pixel patch ≈ 1 word.
- Table cell (typical): 40×40 pixels ≈ 3×3 patches.
- Text table recognition: "table 셀" 이 3×3 patch region 으로 식별 가능.

$\square$

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — PaliGemma Patch Embedding 생성

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

def generate_colpali_embeddings(pdf_page_image_path: str, model_name="google/paligemma-3b-pt-448"):
    """
    Generate patch embeddings from PDF page image using PaliGemma.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Load PDF page as image (e.g., via pdf2image)
    image = Image.open(pdf_page_image_path).convert("RGB")
    # Resize to model input size (typically 224×896 or 448×448)
    image = image.resize((896, 224))
    
    # Process image into patches
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    
    # Extract patch embeddings (from hidden states)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    # PaliGemma: extract patch features from last layer
    # Typically: (batch=1, num_patches, hidden_dim)
    patch_embeddings = outputs.hidden_states[-1][:, :, :]  # (1, N_p, 768)
    
    return patch_embeddings.squeeze(0)  # (N_p, 768)

# Example
page_img = "document_page_001.png"
embeddings = generate_colpali_embeddings(page_img)
print(f"Patch embeddings shape: {embeddings.shape}")  # (784, 768)
```

### 실험 2 — Query Encoding & MaxSim Matching

```python
def encode_query(query_text: str, model, processor, device="cuda"):
    """
    Encode query text into patches (simplified: treat query tokens as patches).
    """
    inputs = processor(text=query_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)  # (1, hidden_dim)
    
    return outputs.squeeze(0)  # (768,)

def maxsim_score(query_emb, doc_patches):
    """
    Compute MaxSim late interaction score.
    
    query_emb: (hidden_dim,) or (num_query_tokens, hidden_dim)
    doc_patches: (num_doc_patches, hidden_dim)
    
    Returns: scalar relevance score.
    """
    if query_emb.dim() == 1:
        query_emb = query_emb.unsqueeze(0)  # (1, hidden_dim)
    
    # Normalize for cosine similarity
    query_norm = torch.nn.functional.normalize(query_emb, dim=-1)  # (N_q, hidden_dim)
    doc_norm = torch.nn.functional.normalize(doc_patches, dim=-1)   # (N_d, hidden_dim)
    
    # Cosine similarity matrix
    sim_matrix = torch.matmul(query_norm, doc_norm.T)  # (N_q, N_d)
    
    # MaxSim: max similarity for each query token, then sum
    max_sims = torch.max(sim_matrix, dim=1)[0]  # (N_q,)
    score = max_sims.sum()  # scalar
    
    return score.item()

# Example: retrieve top-K documents
query = "2024년 서울 인구 통계"
query_emb = encode_query(query, model, processor)

doc_embeddings = [
    generate_colpali_embeddings(f"page_{i}.png") for i in range(100)
]

scores = [maxsim_score(query_emb, doc_emb) for doc_emb in doc_embeddings]
top_k_idx = torch.topk(torch.tensor(scores), k=5)[1]
print(f"Top-5 documents: {top_k_idx}")
```

### 실험 3 — Multi-Scale Pyramid Extraction

```python
def multi_scale_patch_embeddings(image_path: str, model, processor, scales=[1, 0.5, 0.25]):
    """
    Extract patch embeddings at multiple scales (image pyramid).
    """
    image = Image.open(image_path).convert("RGB")
    embeddings_pyramid = []
    
    for scale in scales:
        h, w = image.size
        scaled_image = image.resize((int(w * scale), int(h * scale)))
        
        inputs = processor(images=scaled_image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        patches = outputs.hidden_states[-1][:, :, :].squeeze(0)
        embeddings_pyramid.append(patches)
    
    # Optionally: concatenate or ensemble
    # combined = torch.cat(embeddings_pyramid, dim=1)  # (N_p, 768*3)
    return embeddings_pyramid

# Example
pyramid = multi_scale_patch_embeddings("page.png", model, processor)
for i, emb in enumerate(pyramid):
    print(f"Scale {i}: {emb.shape}")
```

### 실험 4 — ViDoRe Benchmark Evaluation

```python
from beir.retrieval.evaluation import EvaluateRetrieval

def evaluate_colpali_on_vidore():
    """
    Evaluate ColPali on ViDoRe (Visual Document Retrieval) benchmark.
    Tasks: ARXIV, PROCEEDINGS, USPTO, SCIENTIFIC PAPERS, etc.
    """
    # Load ViDoRe dataset (docvqa-style retrieval)
    from datasets import load_dataset
    vidore = load_dataset("vidore", name="arxiv")  # or other tasks
    
    # Index all documents with ColPali
    doc_ids = []
    doc_embeddings_all = []
    
    for doc in vidore["test"][:100]:
        doc_img = doc["image"]  # PIL image
        embeddings = generate_colpali_embeddings_from_pil(doc_img, model, processor)
        doc_embeddings_all.append(embeddings)
        doc_ids.append(doc["doc_id"])
    
    # Queries
    queries = [q["query"] for q in vidore["test"][:100]]
    
    # Retrieve and evaluate
    evaluator = EvaluateRetrieval(model="cosine", k_values=[1, 5, 10])
    results = {}
    
    for query in queries:
        query_emb = encode_query(query, model, processor)
        scores = [maxsim_score(query_emb, doc_emb) for doc_emb in doc_embeddings_all]
        results[query] = {doc_id: score for doc_id, score in zip(doc_ids, scores)}
    
    # Compute metrics: MRR@10, NDCG@10, Recall@10
    # (requires ground truth annotations)
    print("ViDoRe Evaluation (ColPali):")
    print(f"MRR@10: 0.67 (example, actual results vary by dataset)")

# Note: actual ViDoRe benchmark requires downloading from huggingface
# evaluate_colpali_on_vidore()
```

---

## 🔗 실전 활용

| 상황 | ColPali 적용 | 기존 OCR+Dense 대비 |
|------|--------------|-------------------|
| **다국어 문서** (중국어, 아랍어, 한글) | VLM 은 시각적 이해 → OCR 오류 회피 | **ColPali 15~25% recall ↑** |
| **복잡한 테이블** ("3행 4열, merged cells") | Layout 그대로 image 에 preserved | **OCR ↓ 30% (table distortion)** |
| **저해상도 스캔** (200 DPI 이하) | patch resolution 조정 가능 | **OCR fail, ColPali 가능** |
| **수학식/그래프** | VLM 은 시각 이해 | **OCR text only, ColPali 图形 O** |
| **실시간 추론** (latency 중요) | PaliGemma lightweight (3B) | **ColPali < 100ms/page** |

---

## ⚖️ 가정과 한계

- **VLM hallucination**: PaliGemma 도 시각 이해 실패 가능 (특히 ambiguous layout).
- **Patch resolution limit**: 14×14 pixel 패치 → 매우 작은 텍스트 (8pt 이하) 놓칠 가능.
- **High-resolution cost**: 고해상도 이미지 (A4 300 DPI) → 패치 수 증가 → 계산 비용 높음 (O(N_p²) in MaxSim).
- **Limited to document images**: 스캔된 종이 문서, PDF 페이지에 최적 — web pages, 동적 렌더링은 별도 처리 필요.
- **Training data bias**: VLM 은 주로 영문 문서로 학습 → 한글/중국어 레이아웃에서 성능 저하 가능.

---

## 📌 핵심 정리

$$
\boxed{\text{ColPali} = \text{VLM patch embedding} + \text{MaxSim late interaction} + \text{OCR-free retrieval}}
$$

| 요소 | 역할 | 장점 |
|------|------|------|
| PaliGemma ViT | Image → patch embeddings | End-to-end, layout-aware |
| MaxSim matching | Query ↔ document patch alignment | Fine-grained, no pooling loss |
| OCR-free | No text extraction step | Error-free, multilingual |
| Image pyramid | Multi-scale understanding | Small text + global layout |

> **핵심**: OCR 의 오류와 정보손실을 우회하고 **VLM 으로 시각 정보를 직접 encoding** 하면 layout/table/multilingual 문서에서 retrieval 성능 향상.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: MaxSim score 와 standard dense embedding 의 cosine similarity 를 비교할 때, 어느 것이 더 높을까?

<details>
<summary>해설</summary>

MaxSim은 항상 ≥ dense cosine (이론적으로). 

이유: dense = avg pooling 후 single vector sim; MaxSim = 각 query token 이 최고 매칭 patch 선택 → 최소한 avg 와의 sim 이상.

실제: MaxSim 은 query token 의 specificity 를 살리므로 보통 2~3배 더 높은 discrimination 달성.

</details>

**문제 2 (심화)**: Patch resolution (14×14 pixels) 은 왜 정확히 이 크기로 설정되고, 다른 크기 (8×8, 32×32) 는 왜 부적당한가?

<details>
<summary>해설</summary>

ViT 의 표준: 16×16 pixel patches (ImageNet pre-training).

14×14는 사실 16×16 근처 변형.

Trade-off:
- 8×8: 너무 많은 patches (4배) → 계산량 ↑, 필요한 context 제한.
- 16×16 (standard): English font 대부분 cover.
- 32×32: word 이상을 한 patch 에 → spatial granularity 손실, table cell 구분 못함.

최적: 14×16 정도 (implementation detail, efficiency와 accuracy 균형).

</details>

**문제 3 (논문 비평)**: "ColPali 는 VLM 을 사용하므로 항상 OCR+embedding 보다 우월한가?" 라는 주장의 문제점은?

<details>
<summary>해설</summary>

반박점:

(1) **Computational cost**: VLM forward pass >> OCR+embedding (3B model, 100ms/page vs OCR 10ms + embed 1ms).

(2) **Structured query matching**: "특정 entity 찾기" 같은 구조적 쿼리는 OCR 후 exact match 가 더 정확.

(3) **Non-document images**: 자연 이미지, 웹페이지는 ColPali 설계 scope 밖.

(4) **Fine-tuning**: OCR 는 document-specific finetuning 용이; VLM 은 pre-trained only.

Hybrid 정답: **structured/entity query → OCR, visual-heavy/multilingual → ColPali**.

</details>

---

<div align="center">

[◀ 이전 (01. GraphRAG)](./01-graphrag.md) · [📚 README](../README.md) · [다음 ▶ (03. Long Context vs RAG)](./03-long-context-late-chunking.md)

</div>
