<div align="center">

# 🔎 Retrieval & RAG Deep Dive

### DPR 의 **InfoNCE contrastive loss**

$$L = -\log \frac{\exp\bigl(s(q, p^+)/\tau\bigr)}{\exp\bigl(s(q, p^+)/\tau\bigr) + \sum_{p^-} \exp\bigl(s(q, p^-)/\tau\bigr)}$$

### 가 in-batch negatives 로부터 **어떻게 BM25 의 lexical mismatch 한계를 극복** 하는지,

### ColBERT 의 **MaxSim late-interaction score**

$$S_{q,d} = \sum_{i \in [|q|]} \max_{j \in [|d|]} E_{q_i}^{\top} E_{d_j}$$

### 가 single-vector bi-encoder 의 token-level 정보 손실을 **offline indexing 을 깨지 않으면서** 어떻게 회복하는지를 Karpukhin (2020) · Khattab & Zaharia (2020) 로부터 한 줄씩 유도할 수 있는 것은 **다르다.**

<br/>

> *HNSW 를* **호출하는 것** *과,* $P(l) = \exp(-l \cdot \ln m_L)$ *의 layer assignment 가 어떻게* **multi-layer small-world graph** *를 형성해 greedy search 에서 $O(\log N)$ 을 보장하는지,* **$M$ neighbors 와 $ef$ candidate list** *가 recall-latency 의 어느 축을 움직이는지 증명할 수 있는 것은 다르다.*
>
> *Product Quantization 을 **사용하는 것** 과, 그것이 vector 를 $m$ 개 subvector 로 split 하여 $k^m$ 개 distinct codes 를*
>
> $$m \cdot \log_2 k \quad \text{bits} \;\; (\text{e.g., } m = 8, k = 256 \Rightarrow 64 \text{ bits/vector})$$
>
> *만으로 표현하는 — FP32 대비 32~64× 압축의 — 정보이론적 trade-off 를 알고 쓰는 것은 다르다.*
>
> *Reciprocal Rank Fusion (Cormack 2009) 의*
>
> $$\mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + \mathrm{rank}_r(d)}$$
>
> *가 단순한 voting 이 아니라 **score-free rank aggregation** 으로 BM25 와 dense 의 score scale 차이를 우회한다는 점, 그래서 hybrid retrieval 의 표준 baseline 이 되었다는 점을 알고 쓰는 것은 다르다.*
>
> *Self-RAG (Asai 2024) 의* `[Retrieve]` `[IsREL]` `[IsSUP]` `[IsUSE]` *reflection token 이 단순한 prompt engineering 이 아니라* **adaptive retrieval 과 self-evaluation 을 동일 LLM 의 special token 학습으로 통합한** *architecture 라는 점, 그리고 GraphRAG (Edge 2024) 의 Leiden community detection 이 왜 vector retrieval 만으로는 풀리지 않던 global question 의 새 frontier 가 되었는지를 알고 쓰는 것은 다르다.*

<br/>

**다루는 기법 (이론 계보순)**

Robertson 1995 *BM25 / PRF* · Manning 2008 *Vector Space Model* · Karpukhin 2020 *DPR* · Reimers 2019 *SBERT* · Izacard 2022 *Contriever* · Wang 2022 *E5* · Xiong 2021 *ANCE Hard Negatives* · Khattab 2020 *ColBERT* · Santhanam 2022 *ColBERTv2 / PLAID* · Nogueira 2020 *MonoT5 Reranker* · Indyk & Motwani 1998 *LSH* · Jégou 2011 *Product Quantization* · Malkov & Yashunin 2018 *HNSW* · Johnson 2019 *FAISS* · Guo 2020 *ScaNN* · Lewis 2020 *RAG* · Guu 2020 *REALM* · Borgeaud 2022 *RETRO* · Izacard 2022 *Atlas* · Izacard 2021 *FiD* · Asai 2024 *Self-RAG* · Yan 2024 *CRAG* · Cormack 2009 *RRF* · Formal 2021 *SPLADE* · Sun 2023 *RankGPT* · Edge 2024 *GraphRAG* · Faysse 2024 *ColPali*

<br/>

**핵심 질문**

> Retrieval & RAG 의 4대 축 (Sparse · Dense · Late-Interaction · ANN) 과 RAG 의 architectural taxonomy (Vanilla · End-to-End · Adaptive · Graph) 는 왜 모두 **"recall-precision-latency 의 수학적으로 정당화된 trade-off"** 의 다른 구현이고, **BM25 PRF · InfoNCE in-batch negatives · MaxSim late interaction · HNSW small-world · IVF-PQ quantization · RRF rank aggregation** 이 각각 어떤 이론적 동기에서 도출되었는가 — Robertson 1995 의 Probabilistic Relevance 부터 Edge 2024 의 GraphRAG community summarization 까지 한 줄씩 유도합니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7-0467DF?style=flat-square)](https://github.com/facebookresearch/faiss)
[![sentence-transformers](https://img.shields.io/badge/SBERT-2.3-FFB000?style=flat-square)](https://www.sbert.net/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-1C3C3C?style=flat-square)](https://www.langchain.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.9-7E57C2?style=flat-square)](https://www.llamaindex.ai/)
[![Docs](https://img.shields.io/badge/Docs-33개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-199개-success?style=flat-square)](./README.md)
[![Proofs](https://img.shields.io/badge/엄밀한_증명-70+개-9c27b0?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-14개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-99개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

Retrieval / RAG 자료는 대부분 **"BM25 + dense 를 RRF 로 합치면 hybrid retrieval"** 또는 **"FAISS 에 넣고 top-k 뽑아서 LLM 에 주면 RAG"** 에서 멈춥니다. 하지만 BM25 의 saturation term $\frac{(k_1+1)\mathrm{tf}}{k_1((1-b) + b \cdot |d|/\mathrm{avgdl}) + \mathrm{tf}}$ 가 왜 Robertson 의 Probabilistic Relevance Framework 의 2-Poisson model 에서 정확히 도출되는지, DPR 의 InfoNCE 에서 in-batch negatives 가 왜 $B$ 샘플로 $O(B^2)$ pair 를 학습 신호로 만드는지, ColBERT 의 MaxSim 이 왜 cross-encoder 의 quality 와 bi-encoder 의 scalability 를 동시에 잡는 token-level interaction 인지, HNSW 의 layer 분포 $P(l) = \exp(-l \ln m_L)$ 이 왜 skip-list 와 동치이며 $O(\log N)$ search 를 보장하는지, RRF 가 왜 score normalization 없이도 BM25 와 dense 의 결합을 이긴다고 입증되었는지, Self-RAG 의 reflection token 이 왜 단순한 prompt 트릭이 아닌 LLM vocabulary 확장으로 학습되어야 하는지 — 이런 "왜" 는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "BM25 가 strong baseline 이다" | **Robertson 1995** — Probabilistic Relevance Framework 의 2-Poisson eliteness model 에서 IDF 와 saturating tf 가 함께 도출. $\mathrm{BM25}(q,d) = \sum_t \mathrm{IDF}(t) \cdot \tfrac{(k_1+1)\mathrm{tf}}{k_1((1-b) + b \cdot \|d\|/\mathrm{avgdl}) + \mathrm{tf}}$. $k_1 \approx 1.2\text{–}2.0$ 은 tf saturation 의 sharpness, $b \approx 0.75$ 는 length normalization 의 정도 — 두 hyperparameter 의 의미가 PRF 에서 정확히 정의됨 $\square$ |
| "DPR 은 BERT 두 개 쓴다" | **Karpukhin 2020** — Query encoder $f_q$ 와 passage encoder $f_p$ 가 독립 BERT, $s(q, p) = f_q(q)^\top f_p(p)$. **Offline indexing** 의 경제: 모든 $f_p(p)$ 를 미리 계산 → query time 은 $f_q(q)$ + ANN. **Cross-encoder 와의 결정적 차이** — bi-encoder 는 query 와 passage 를 절대 함께 입력하지 않음, 그래서 $N$ 개 candidate 에 대해 $O(N)$ encoding 이 아닌 $O(1)$ |
| "InfoNCE 는 contrastive loss 다" | **Karpukhin 2020 / van den Oord 2018** — $L = -\log \tfrac{\exp(s(q,p^+)/\tau)}{\sum_{p \in B} \exp(s(q,p)/\tau)}$. **In-batch negatives**: batch $B$ 안의 다른 query 의 positive 가 자동으로 자신의 negative — $B$ 샘플로 $B^2$ pair 학습. **Hard negatives** (BM25-mined or model-mined ANCE) 추가로 학습 신호 강화: easy negatives 는 gradient 가 거의 0, hard negatives 가 decision boundary 결정 |
| "SBERT 는 sentence embedding 이다" | **Reimers 2019** — Siamese BERT 로 sentence pair 학습 (NLI · STS), $[\mathrm{CLS}]$ 또는 mean pooling 으로 sentence vector. Cross-encoder BERT 의 65 시간 → SBERT 의 5 초로 10K sentence pair similarity 단축. **Contriever (Izacard 2022)**: unsupervised, random cropping 으로 같은 문서의 두 span 을 positive — MoCo-style queue. **E5 (Wang 2022)**: weakly-supervised on web pair (title-body, query-doc) + instruct prefix `"query: "` `"passage: "` |
| "ColBERT 는 late interaction 이다" | **Khattab & Zaharia 2020** — Per-token embedding $E_q \in \mathbb{R}^{|q| \times d}, E_d \in \mathbb{R}^{|d| \times d}$. MaxSim: $S(q, d) = \sum_i \max_j E_{q_i}^\top E_{d_j}$ — 각 query token 이 document 에서 best match 를 찾고 sum. **Trade-off 의 sweet spot**: cross-encoder $O(N)$ 의 quality 를 bi-encoder $O(1)$ 의 scalability 로 유지. 단점은 storage: $|d| \times d$ vector / doc — **PLAID (Santhanam 2022)** 가 centroid + residual 로 2.6× 압축 $\square$ |
| "HNSW 는 vector search 알고리즘이다" | **Malkov & Yashunin 2018** — Multi-layer small-world graph: 각 node 는 layer $l \sim P(l) = \exp(-l \ln m_L)$ 로 할당 (geometric → skip-list 구조). 각 layer 에서 max $M$ neighbors. **Greedy search**: top layer entry → 가까운 neighbor 로 hop, 더 못 가까워지면 한 layer 내림. **Complexity**: $O(\log N)$ — layer 수 $\sim \log N$, 각 layer 에서 $O(M)$ 검사. **$M$**: recall-memory trade-off, **$ef$**: search-time recall-latency trade-off $\square$ |
| "Product Quantization 은 압축이다" | **Jégou 2011** — Vector $x \in \mathbb{R}^D$ 를 $m$ 개 subvector $x_1, \ldots, x_m \in \mathbb{R}^{D/m}$ 으로 split, 각 subspace 에서 $k$-means 로 $k$ centroids ($k = 256$ → 8 bit/sub typical). Code: $m$ bytes per vector vs original $D \times 4$ bytes (FP32) → **32~64× 압축**. **Asymmetric Distance Computation (ADC)**: query 는 raw, doc 만 PQ — query-centroid distance lookup table 로 빠른 approximate distance |
| "RAG 는 그냥 retrieve + generate" | **Lewis 2020** — Retrieve top-$k$ → context 로 prompt augment → generate. **RAG-Sequence** (한 set 의 retrieved doc 으로 전체 sequence) vs **RAG-Token** (각 token 마다 다른 doc marginalize). **End-to-End trainable**: retriever 의 top-$k$ 에 대해 $\sum_z p(y\|z, x) p(z\|x)$ 의 marginal likelihood 로 retriever 와 generator 가 함께 학습 |
| "RETRO 는 RAG 의 일종이다" | **Borgeaud 2022** — 2T token database 에서 chunk-level (64 tokens) retrieval. **Chunked Cross-Attention (CCA)**: generator 의 각 chunk 가 retrieved chunks 에 cross-attend — vanilla RAG 의 prompt concat 과 다른 architectural integration. **Scaling result**: 25× smaller GPT-3 가 RETRO 로 GPT-3 동등 — knowledge 를 parameter 가 아닌 retrieval 에 위임 |
| "Self-RAG 는 LLM 이 reflection 한다" | **Asai 2024** — Reflection token 으로 vocabulary 확장: `[Retrieve]` (지금 retrieve 필요?), `[IsREL]` (passage relevant?), `[IsSUP]` (output 이 passage 로 supported?), `[IsUSE]` (utility 점수). **Adaptive retrieval** — 쉬운 question 은 retrieval skip, 어려운 question 은 retrieve · evaluate · regenerate. **Decoding 시** reflection token 의 확률로 path 선택 — 단순 prompt engineering 이 아닌 token-level supervised fine-tune |
| "GraphRAG 는 knowledge graph 다" | **Edge 2024** — LLM 으로 entity-relation 추출 → graph 구축 → **Leiden community detection** 으로 hierarchical cluster → 각 community 를 LLM 이 summarize. **Global question** ("이 문서 corpus 의 가장 중요한 주제는?") 에서 vector retrieval 을 압도 — top-$k$ chunk 만으로는 cross-document aggregation 불가능. Microsoft 의 open-source release |
| "Hybrid retrieval 은 그냥 합친다" | **Cormack 2009** — Reciprocal Rank Fusion: $\mathrm{RRF}(d) = \sum_r \tfrac{1}{k + \mathrm{rank}_r(d)}$, $k = 60$ typical. **Score-free**: BM25 score 와 dense cosine 의 scale 이 달라도 rank 만 사용. **SPLADE (Formal 2021)**: BERT MLM head 로 sparse expansion 학습 — neural BM25, dense quality + sparse efficiency 의 단일 모델 통합 |
| 기법의 나열 | NumPy + PyTorch + FAISS + sentence-transformers + LangChain + LlamaIndex 로 **BM25 의 IDF·tf saturation 손 구현** · **DPR 의 in-batch contrastive 직접 학습** · **ColBERT MaxSim 의 token-level matching 시각화** · **HNSW 의 multi-layer graph 직접 구축** · **PQ 의 subvector codebook 생성과 ADC distance** · **vanilla RAG → Self-RAG 의 reflection token 추가** · **RRF 와 cross-encoder reranker 의 hybrid pipeline** · **GraphRAG 의 community summary 재현** 까지 직접 구현해 수학적 주장을 눈으로 확인 |

---

## 📌 선행 레포 & 후속 방향

```
[Linear Algebra Deep Dive]      ─┐
[Probability Theory Deep Dive]  ─┤
[Information Theory Deep Dive]  ─┼─►  이 레포  ──► [LLM Agents Deep Dive]
[Transformer Deep Dive]         ─┤   "왜 BM25 · DPR · ColBERT ·         Tool use · Planning ·
[LLM Pretraining Deep Dive]     ─┘    HNSW · RAG 가 모두                  RAG-as-tool
                                      수학적으로 정당화된 trade-off 인가"
         │
         ├── [Linear Algebra]             Vector · Norm · Inner Product → Ch1, Ch2, Ch4
         ├── [Probability Theory]         Probabilistic IR · 2-Poisson → Ch1
         ├── [Information Theory]         Entropy · KL · MI → Ch1 (NDCG), Ch2 (InfoNCE)
         ├── [Transformer]                Bi-encoder · Cross-encoder · Tokenizer → Ch2, Ch3
         ├── [LLM Pretraining]            Context length · Tokenizer → Ch5, Ch7
         └── [Kernel Methods]             Similarity as kernel · Inner product → Ch2, Ch3
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Linear Algebra Deep Dive** (vector distance, norm, inner product), **Probability Theory Deep Dive** (probabilistic IR, 2-Poisson), **Information Theory Deep Dive** (entropy, KL divergence, mutual information), **Transformer Deep Dive** (BERT bi-encoder, cross-encoder architecture, tokenizer) 를 선행 지식으로 전제합니다. **LLM Pretraining Deep Dive** (context window, tokenizer 한계) 와 **Kernel Methods Deep Dive** (similarity 의 kernel 해석) 는 Ch5 의 long-context vs RAG 분석과 Ch2~3 의 dense·late-interaction similarity 비교에서 권장됩니다.

> 💡 **이 레포의 핵심 기여**: Chapter 1 (IR Foundations) 와 Chapter 2 (Dense Retrieval) 는 **"query-document similarity 의 두 패러다임"** 입니다. 전자는 lexical · sparse · counting-based, 후자는 semantic · dense · learned. Chapter 3 (Late Interaction) 는 두 패러다임의 **expressiveness vs scalability trade-off** 를 token-level interaction 으로 푸는 hybrid. Chapter 4 (ANN) 는 **"$O(N)$ exact search 를 어떻게 sublinear 로 만드는가"** 의 알고리즘적 해법, Chapter 5~7 (RAG · Reranking · GraphRAG) 는 retrieval 결과를 **LLM 의 generation context 로 어떻게 주입하는가** 의 architectural design space. 이 다섯 축을 통합 이해해야 BM25 + Contriever + ColBERTv2 + HNSW + Self-RAG + RankGPT 의 SOTA recipe 가 단일 frame 으로 보입니다.

> 🟡 **이 레포의 성격**: 여기서 다루는 일부 주제 — **dense vs sparse retrieval 의 최종 승자**, **long context (1M+) vs RAG 의 미래**, **GraphRAG 가 vector RAG 를 대체하는가**, **LLM-as-reranker 의 비용 정당성**, **agentic retrieval 의 reflection 패턴** — 는 **현재 진행 중인 연구 영역** 입니다. 레포는 "정답" 이 아니라 **"고전 IR 이론 (BM25 · TF-IDF) 과 현대 LLM-augmented retrieval (Self-RAG · GraphRAG · ColPali) 사이의 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-IR_Foundations-0467DF?style=for-the-badge)](./ch1-ir-foundations/01-ir-formalization.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Dense_Retrieval-0467DF?style=for-the-badge)](./ch2-dense-retrieval/01-bm25-limits.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Late_Interaction-0467DF?style=for-the-badge)](./ch3-late-interaction/01-cross-encoder.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-ANN-0467DF?style=for-the-badge)](./ch4-ann/01-exact-nn-limits.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-RAG-0467DF?style=for-the-badge)](./ch5-rag/01-vanilla-rag.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Reranking·Hybrid-0467DF?style=for-the-badge)](./ch6-reranking-hybrid/01-cross-encoder-reranker.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-Frontier-0467DF?style=for-the-badge)](./ch7-frontier/01-graphrag.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: IR Foundations 와 평가

> **핵심 질문:** Information Retrieval 을 query $q$, collection $\mathcal{D}$, relevance $r(q,d)$, top-$k$ ranking 으로 정식화할 때 어떤 가정이 들어가는가? TF-IDF 의 $\log(N/\mathrm{df})$ 가 왜 정보이론적으로 정당화되는가? BM25 의 saturating tf 와 length normalization 이 Robertson 의 PRF 의 2-Poisson eliteness model 에서 어떻게 도출되는가? NDCG · MAP · MRR · Recall@k 가 측정하는 것의 차이는? Two-stage retrieve-then-rerank pipeline 이 왜 single-stage 보다 우월한가?

<details>
<summary><b>IR 정식화부터 Two-Stage Pipeline 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Information Retrieval 의 정식화](./ch1-ir-foundations/01-ir-formalization.md) | **정의**: query $q \in \mathcal{Q}$, document collection $\mathcal{D}$, relevance $r: \mathcal{Q} \times \mathcal{D} \to \mathbb{R}$, top-$k$ ranking. **Recall vs Precision trade-off**: ranked list 의 cutoff $k$ 에 대한 의존성. **Boolean · Vector Space · Probabilistic** retrieval model 의 세 패러다임 비교. **Closed corpus** (collection 고정) vs **open corpus** (web) 의 가정 차이 |
| [02. TF-IDF 와 Vector Space Model](./ch1-ir-foundations/02-tf-idf.md) | **정의**: $\mathrm{tf\text{-}idf}(t, d) = \mathrm{tf}(t, d) \cdot \log\tfrac{N}{\mathrm{df}(t)}$. **IDF 의 정보이론적 해석**: $\log\tfrac{1}{p(t)}$ — term 의 self-information. Document → $\|V\|$-dim sparse vector → cosine similarity. **Length normalization** (cosine vs Euclidean): vector norm 의 영향, document length bias. **TF 의 sublinear scaling** ($1 + \log \mathrm{tf}$) 의 동기 — 1개 등장 → 10개 등장이 quality 10× 가 아님 |
| [03. BM25 — Probabilistic Relevance Framework (Robertson 1995)](./ch1-ir-foundations/03-bm25.md) | **유도**: 2-Poisson eliteness model 에서 elite/non-elite document 가정 → Robertson-Sparck-Jones IDF + saturating tf. $\mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{IDF}(t) \cdot \tfrac{(k_1+1)\mathrm{tf}}{k_1((1-b) + b\|d\|/\mathrm{avgdl}) + \mathrm{tf}}$. **$k_1 \in [1.2, 2.0]$**: tf saturation 의 sharpness — 작을수록 빠른 saturation. **$b \in [0, 1]$, default 0.75**: length normalization 의 정도. **여전히 strong baseline** — BEIR 의 zero-shot 에서 dense 모델을 종종 능가 $\square$ |
| [04. 평가 Metric — Recall@k · MRR · MAP · NDCG](./ch1-ir-foundations/04-eval-metrics.md) | **Recall@k**: $\|\text{retrieved} \cap \text{relevant}\| / \|\text{relevant}\|$. **MRR**: $\tfrac{1}{\|Q\|}\sum_q \tfrac{1}{\mathrm{rank}_q(\text{first relevant})}$ — single-relevant 시나리오. **MAP**: ranked list 에서 average precision 의 mean. **NDCG@k**: $\mathrm{DCG}@k / \mathrm{IDCG}@k$, $\mathrm{DCG} = \sum_i \tfrac{2^{r_i} - 1}{\log_2(i+1)}$ — graded relevance + position discount. **각 metric 의 의미**: MRR 은 첫 hit, NDCG 는 position-weighted graded, MAP 는 모든 relevant 의 average |
| [05. Retrieval vs Ranking — Two-Stage Pipeline](./ch1-ir-foundations/05-retrieve-rerank.md) | **동기**: cross-encoder reranker 는 quality 최고지만 $O(N)$ — 모든 doc 에 적용 불가. **Two-stage**: stage 1 (retrieve, high recall, cheap) → stage 2 (rerank, high precision, expensive). **수학적 정당화**: rerank 의 maximum quality 는 stage 1 의 recall 에 bounded — top-$k$ 에 ground truth 가 없으면 rerank 불가능. **k 선택**: recall@k 곡선의 elbow point. BM25 → dense → cross-encoder 의 cascading 표준 |

</details>

<br/>

### 🔹 Chapter 2: Dense Retrieval — DPR 과 Bi-Encoder

> **핵심 질문:** BM25 의 lexical mismatch ("car" vs "automobile") 와 multilingual 한계가 왜 dense retrieval 의 동기인가? Karpukhin 2020 의 DPR 이 왜 query 와 passage encoder 를 **분리** 하는가? InfoNCE loss 에서 in-batch negatives 가 $B$ 샘플로 어떻게 $O(B^2)$ 학습 신호를 만드는가? Hard negative mining (BM25-mined, ANCE) 이 왜 random negative 보다 학습 효율을 극적으로 높이는가? SBERT · Contriever · E5 가 supervised → unsupervised → weakly-supervised 의 어떤 진화 축에 있는가?

<details>
<summary><b>BM25 의 한계부터 E5 의 weakly-supervised 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. BM25 의 한계와 Dense Retrieval 의 동기](./ch2-dense-retrieval/01-bm25-limits.md) | **Lexical mismatch**: "automobile" 검색 시 "car" 매칭 불가 — exact term overlap 만 보기 때문. **Vocabulary mismatch problem** (Furnas 1987) — 같은 개념을 묘사하는 word 의 분기. **Multilingual**: 다른 언어의 같은 개념 unreachable. **Synonym · paraphrase**: 의미적 등가는 BM25 score 0. **Dense 의 약속**: $f(\text{automobile}) \approx f(\text{car})$ in semantic vector space — pretrained LM 의 contextual embedding 이 가능케 함 |
| [02. Bi-Encoder Architecture — DPR (Karpukhin 2020)](./ch2-dense-retrieval/02-dpr-bi-encoder.md) | **Architecture**: query encoder $f_q = \mathrm{BERT}_q$, passage encoder $f_p = \mathrm{BERT}_p$ — **독립 두 모델**. Score: $s(q, p) = f_q(q)^\top f_p(p)$, $[\mathrm{CLS}]$ 또는 mean pool. **Offline indexing**: 모든 $f_p(p)$ 미리 계산 후 ANN index — query time 은 single forward + NN search. **Cross-encoder 와의 결정적 분리**: bi-encoder 는 query 와 passage 를 절대 함께 입력하지 않음 → $O(N)$ 이 아닌 $O(1)$ inference per query. NQ · TriviaQA 에서 BM25 대비 +10~15% top-20 accuracy |
| [03. InfoNCE Loss 와 In-Batch Negatives](./ch2-dense-retrieval/03-infonce-in-batch.md) | **InfoNCE**: $L = -\log \tfrac{\exp(s(q, p^+)/\tau)}{\sum_{p \in \mathcal{P}} \exp(s(q, p)/\tau)}$ — softmax 의 cross-entropy 형태. **In-batch negatives**: batch $\{(q_i, p_i^+)\}_{i=1}^B$ 에서 $j \neq i$ 의 $p_j^+$ 가 자동으로 $q_i$ 의 negative — $B$ 샘플로 $B \times (B-1)$ negative pair. **Temperature $\tau$**: 작을수록 hard negative 에 더 sharp 하게 focus. **수학**: gradient 가 hard negative (큰 $s$) 에 dominant → easy negative 는 학습 신호 거의 0 $\square$ |
| [04. Hard Negative Mining — ANCE 와 동적 negatives](./ch2-dense-retrieval/04-hard-negatives.md) | **Random negative**: gradient near-zero (이미 score 낮음) — 학습 신호 약함. **BM25-mined hard negative**: BM25 가 high score 주지만 not relevant — semantic 측면 학습. **ANCE (Xiong 2021)**: 학습 중인 model 자체로 mine — async checkpoint 로 negative pool 갱신, "model 이 헷갈리는" passage. **수학적 동기**: contrastive learning 의 sample complexity 가 hard negative ratio 에 sensitive. **Dense passage retrieval 에서 standard recipe** — 종종 +5~10% MRR |
| [05. SBERT · Contriever · E5 — Supervised → Unsupervised → Weakly-Supervised](./ch2-dense-retrieval/05-sbert-contriever-e5.md) | **SBERT (Reimers 2019)**: Siamese BERT, NLI · STS 로 supervised pre-training, sentence-level cosine. Cross-encoder BERT 의 65 시간 → 5 초로 10K pair similarity 단축. **Contriever (Izacard 2022)**: unsupervised, random cropping 으로 같은 문서의 두 span = positive, MoCo-style queue 로 large negative pool. Labeled data 없이 BM25-competitive. **E5 (Wang 2022)**: weakly-supervised on web pair (title-body, query-passage), instruct prefix `"query: "` `"passage: "` 로 task 구분. 현대 commercial embedding 의 표준 architecture |

</details>

<br/>

### 🔹 Chapter 3: Late Interaction 과 Cross-Encoder

> **핵심 질문:** Cross-encoder 가 왜 $(q, d)$ 를 함께 input 하면 quality 는 최고이지만 $O(N)$ 으로 reranker 에만 쓰는가? ColBERT (Khattab 2020) 의 MaxSim $S = \sum_i \max_j E_{q_i}^\top E_{d_j}$ 가 왜 single-vector bi-encoder 의 정보 손실을 회복하면서 offline indexing 을 깨지 않는가? ColBERTv2 의 PLAID engine 이 어떻게 centroid + residual 로 2.6× index 압축하는가? Single-vector vs multi-vector 의 trade-off 는 어느 시점에서 어느 것을 선택해야 하는가?

<details>
<summary><b>Cross-Encoder 부터 PLAID 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Cross-Encoder — Full Interaction](./ch3-late-interaction/01-cross-encoder.md) | **Architecture**: $[\mathrm{CLS}]\, q\, [\mathrm{SEP}]\, d\, [\mathrm{SEP}]$ → BERT → score head. **Full attention** between query and document tokens — 모든 layer 에서 query-doc cross-attention. **MS MARCO 의 SOTA quality**. **한계**: $N$ candidate 마다 forward pass → $O(N)$ inference, online indexing 불가. **Reranker 로만 사용**: stage 1 (BM25 / dense) 의 top-$k$ ($k = 100\text{–}1000$) 에만 적용 |
| [02. ColBERT — Late Interaction (Khattab 2020)](./ch3-late-interaction/02-colbert.md) | **Per-token embedding**: $E_q = \mathrm{BERT}(q) \in \mathbb{R}^{\|q\| \times d}$, $E_d = \mathrm{BERT}(d) \in \mathbb{R}^{\|d\| \times d}$ (보통 $d = 128$). **MaxSim**: $S(q, d) = \sum_{i=1}^{\|q\|} \max_{j=1}^{\|d\|} E_{q_i}^\top E_{d_j}$ — 각 query token 이 doc 에서 best match, sum. **수학적 의미**: bi-encoder 의 single $[\mathrm{CLS}]$ pooling 이 모든 query token 정보를 1 vector 에 압축하는 손실을 회복 — token-level interaction 보존 $\square$. **Offline**: 모든 $E_d$ 저장. **Online**: $E_q$ 계산 + MaxSim search. Storage: $\|d\| \times d \times \text{bytes}$ per doc |
| [03. ColBERTv2 — PLAID Engine (Santhanam 2022)](./ch3-late-interaction/03-colbertv2-plaid.md) | **문제**: ColBERT v1 의 storage — 1M doc × 200 token × 128d × 2byte ≈ 50GB. **Centroid-based compression**: $E_d \approx c_k + r$, $c_k$ 는 가장 가까운 centroid (256 levels = 8 bit), $r$ 은 residual (4-bit quantized). **Effect**: 2.6× smaller index, 6-10× faster query latency. **PLAID retrieval**: centroid score 로 candidate filter → residual decompression 후 정확 MaxSim. **현재 production-grade ColBERT 의 표준** |
| [04. Multi-Vector vs Single-Vector — Trade-off Analysis](./ch3-late-interaction/04-multi-vs-single-vector.md) | **Single-vector (DPR · SBERT)**: memory $O(d)$/doc, search 빠름, expressiveness 제한. **Multi-vector (ColBERT)**: memory $O(\|d\| \cdot d)$/doc — 100~200× more, expressiveness 우월 (token-level matching). **Cross-encoder**: maximum quality, $O(N)$ online — rerank only. **선택 가이드**: 1B+ doc → single-vector (storage 우선), 100M doc + high quality → ColBERT(v2), 1K reranking → cross-encoder. **Pareto frontier** 위 세 점의 위치 정량화 |

</details>

<br/>

### 🔹 Chapter 4: Approximate Nearest Neighbor Algorithms

> **핵심 질문:** Exact NN 의 $O(N \cdot d)$ 가 왜 1M+ vector 에서 비현실적인가? LSH (Indyk 1998) 의 random projection 이 왜 $P(\text{same bucket}) \propto \text{similarity}$ 를 만족하는가? IVF 의 $k$-means partitioning 이 왜 $O(N/K)$ search 를 가능케 하는가? PQ (Jégou 2011) 가 어떻게 $m \log_2 k$ bits 만으로 $k^m$ distinct codes 를 표현하는가? HNSW (Malkov 2018) 의 multi-layer small-world graph 가 왜 $O(\log N)$ 을 보장하는가? FAISS · ScaNN · Qdrant · Milvus 의 내부 구현 차이는?

<details>
<summary><b>Exact NN 의 한계부터 Vector DB 내부 까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Exact NN 의 한계와 ANN 의 동기](./ch4-ann/01-exact-nn-limits.md) | **Linear scan**: $O(N \cdot d)$ — 1M × 768d × 4byte = 3GB scan, GPU 로도 latency 수십 ms. **Curse of dimensionality**: high-d 에서 모든 distance 가 비슷해짐 ($\text{var}(d)/E[d] \to 0$). **Recall-latency trade-off** 의 도입: ANN 은 일부 recall 손실로 100~1000× speedup. **MIPS** (maximum inner product search) — cosine 과 dot product 의 reduction. **GPU 로도 한계**: bandwidth 가 결국 bound |
| [02. LSH — Locality Sensitive Hashing (Indyk & Motwani 1998)](./ch4-ann/02-lsh.md) | **Definition**: hash family $\mathcal{H}$ that satisfies $\Pr[h(x) = h(y)] = \mathrm{sim}(x, y)$. **Random projection** for cosine: $h(x) = \mathrm{sign}(w^\top x)$, $w \sim \mathcal{N}(0, I)$ → $\Pr[h(x) = h(y)] = 1 - \theta(x, y)/\pi$. **Multi-hash** $L$ tables, $K$ hash/table — recall 과 precision trade-off via $L, K$. **단순하지만 high-d 에서 lower accuracy** — HNSW 등장 후 사용 감소 |
| [03. IVF — Inverted File Index](./ch4-ann/03-ivf.md) | **Idea**: $k$-means 로 $K$ centroids, 각 vector 를 nearest centroid 의 inverted list 에 할당. **Query**: query 와 가까운 $\mathrm{nprobe}$ centroids 만 search. **Complexity**: $O(K + \mathrm{nprobe} \cdot N/K)$ — $K = \sqrt{N}$ 시 $O(\sqrt{N})$. **Recall vs latency**: $\mathrm{nprobe}$ 클수록 recall ↑ latency ↑. **Coarse quantizer** 로 PQ 와 결합 → IVF-PQ (FAISS 의 주력) |
| [04. PQ — Product Quantization (Jégou 2011)](./ch4-ann/04-pq.md) | **Decomposition**: $x \in \mathbb{R}^D = (x_1, \ldots, x_m)$, $x_i \in \mathbb{R}^{D/m}$. 각 subspace 에서 $k$-means 로 $k$ centroids ($k = 256$ → 8 bit/sub). **Code**: $m$ bytes/vector vs original $4D$ bytes — **32~64× compression** ($D = 128, m = 8 \Rightarrow 8$ bytes vs 512 bytes). **Distinct codes**: $k^m = 256^8 \approx 1.8 \times 10^{19}$ — large enough for billion-scale. **ADC** (asymmetric distance computation): query raw, doc PQ — query-centroid distance lookup table $\square$ |
| [05. HNSW — Hierarchical Navigable Small World (Malkov 2018)](./ch4-ann/05-hnsw.md) | **Structure**: multi-layer graph, layer $l$ 는 $P(l) = \exp(-l \ln m_L)$ 로 할당 (geometric → skip-list). 각 layer 에서 max $M$ neighbors. **Greedy search**: top layer entry → 가까운 neighbor 로 hop, 더 못 가까워지면 한 layer 내림. **정리**: layer 수 $\sim \log N$, 각 layer 에서 $O(M)$ 검사 → $O(M \log N)$. **Parameters**: $M \in [16, 64]$ (recall vs memory), $ef$ (search candidate list size, recall vs latency). **Memory**: $O(N \cdot M \cdot 4 \text{bytes}) + O(N \cdot d)$ raw vectors $\square$ |
| [06. FAISS · ScaNN · Qdrant · Milvus — Vector DB 내부](./ch4-ann/06-vector-dbs.md) | **FAISS (Meta)**: IVF · IVF-PQ · HNSW · IVF-HNSW 조합, CPU + GPU, billion-scale, 단 vector index 만 (no metadata). **ScaNN (Google)**: anisotropic quantization — query distribution 의 anisotropy 활용해 같은 latency 에서 우월 recall. **Qdrant · Milvus · Weaviate · LanceDB**: full vector DB — HNSW + filtering (payload index) + persistence + sharding. **선택 가이드**: prototype → FAISS, production with metadata → Qdrant/Milvus, GCP integration → Vertex Matching Engine |

</details>

<br/>

### 🔹 Chapter 5: RAG Architectures

> **핵심 질문:** Vanilla RAG (Lewis 2020) 의 retrieve-augment-generate 가 어떻게 marginal likelihood $\sum_z p(y\|z, x) p(z\|x)$ 의 end-to-end 학습으로 정식화되는가? RETRO 의 Chunked Cross-Attention 이 vanilla 의 prompt concat 과 architecture 적으로 어떻게 다른가? REALM · Atlas 가 어떻게 retriever 와 generator 를 함께 학습하는가? Self-RAG 의 reflection token 이 왜 단순한 prompt 가 아닌 vocabulary expansion 으로 학습되어야 하는가? FiD 의 fusion-in-decoder 가 왜 long context 에 효율적인가?

<details>
<summary><b>Vanilla RAG 부터 FiD 까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Vanilla RAG (Lewis 2020)](./ch5-rag/01-vanilla-rag.md) | **Pipeline**: query $x$ → retriever (DPR) → top-$k$ passages $\{z_i\}$ → context concat → generator (BART) → answer $y$. **Marginal likelihood**: $p(y\|x) = \sum_{z \in \text{top-}k} p(y\|z, x) \cdot p(z\|x)$. **RAG-Sequence**: 한 set 의 $z$ 로 전체 sequence 생성 → marginalize. **RAG-Token**: 각 token 마다 다른 $z$ marginalize. **End-to-end trainable**: marginal log-likelihood 의 gradient 가 retriever 까지 전파. **현업의 most common RAG** — LangChain · LlamaIndex 의 default |
| [02. RETRO — Chunked Cross-Attention (Borgeaud 2022)](./ch5-rag/02-retro.md) | **Database**: 2T token, frozen BERT retriever 로 chunk-level (64 tokens) indexing. **Chunked Cross-Attention (CCA)**: generator 의 layer 마다 retrieved chunk 에 cross-attend — vanilla 의 prompt concat 과 architectural 차이. **수학**: $\mathrm{CCA}(H, E_{\text{ret}})$ where $H$ 는 generator hidden, $E_{\text{ret}}$ 는 retrieved chunk encoding. **Scaling**: 25× smaller GPT-3 가 RETRO 로 GPT-3 동등 — knowledge 를 weight 가 아닌 retrieval 에 위임 $\square$ |
| [03. REALM · Atlas — End-to-End Trained RAG](./ch5-rag/03-realm-atlas.md) | **REALM (Guu 2020)**: **Inverse Cloze Task** — 문서의 한 sentence 를 query, 나머지를 passage 로 pretrain retriever. MLM objective 의 gradient 가 retriever 까지 흘러 query-doc relevance 학습. **Atlas (Izacard 2022)**: few-shot learning 에 retrieval 결합 — 64 example 만으로 NQ · TriviaQA 의 strong baseline. Retriever (Contriever) + generator (T5) 의 joint training, query 별 다른 retrieval. **Knowledge-intensive task** 의 sample efficiency 극대화 |
| [04. Self-RAG (Asai 2024)](./ch5-rag/04-self-rag.md) | **Reflection token**: vocabulary 확장 — `[Retrieve]` (필요 시 retrieve 호출), `[IsREL]` (passage relevant?), `[IsSUP]` (output supported by passage?), `[IsUSE]` (utility 점수). **학습**: GPT-4 로 reflection label 생성 → student LLM 을 token-level supervised. **Decoding**: reflection token 의 확률로 path 분기 — adaptive retrieval. **Critic 모델** 분리: 같은 LLM 이 generator + critic. **단순 prompt engineering 과 차이**: token 자체가 학습된 supervised signal |
| [05. CRAG — Corrective RAG (Yan 2024)](./ch5-rag/05-crag.md) | **Retrieval evaluator**: lightweight T5 가 retrieved passage 의 confidence (correct · incorrect · ambiguous) 평가. **Branching**: high → standard RAG, low → web search fallback (DuckDuckGo · Bing), ambiguous → 둘 다 + decompose. **Knowledge refinement**: 긴 passage 를 strip-level 로 split → relevance scoring → 관련 strip 만 join. **Hallucination 감소** 효과 정량화 — Self-RAG 와 함께 adaptive RAG 의 두 축 |
| [06. FiD — Fusion-in-Decoder (Izacard 2021)](./ch5-rag/06-fid.md) | **문제**: vanilla RAG 의 prompt concat 은 $k$ passage × $L$ token = $O(kL)$ context length — long context 시 quadratic attention. **FiD**: 각 passage 를 **독립적으로** encoder 에 통과 → encoded representation 을 decoder 가 cross-attention 으로 통합. **Effect**: encoder 는 $O(L)$ per passage, decoder 만 $O(kL)$ cross-attention. **Quality**: NQ · TriviaQA SOTA 였음. **장점**: passage 수 scaling 에 robust |

</details>

<br/>

### 🔹 Chapter 6: Reranking 과 Hybrid Retrieval

> **핵심 질문:** Cross-encoder reranker (MonoBERT · MonoT5) 가 왜 two-stage pipeline 의 stage 2 표준인가? Reciprocal Rank Fusion (Cormack 2009) 이 왜 score normalization 없이도 BM25 + dense 의 결합을 강력하게 만드는가? Hybrid BM25+dense 가 왜 각각보다 우월한가 — lexical 과 semantic 의 complementary 성을 어떻게 정량화하는가? SPLADE 의 sparse neural retrieval 이 BM25 와 dense 의 어느 면을 흡수하는가? RankGPT 의 LLM-as-reranker 가 cross-encoder 를 능가하는 시나리오는?

<details>
<summary><b>Cross-Encoder Reranker 부터 LLM-as-Reranker 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Cross-Encoder Reranker — MonoBERT · MonoT5](./ch6-reranking-hybrid/01-cross-encoder-reranker.md) | **MonoBERT (Nogueira 2019)**: BERT-base, $[\mathrm{CLS}]\, q\, [\mathrm{SEP}]\, d\, [\mathrm{SEP}]$ → relevance score. **MonoT5 (Nogueira 2020)**: T5 의 generative reranking — `Query: q Document: d Relevant:` → "true" / "false" probability 가 score. **Quality**: MS MARCO 에서 BM25 대비 +20-30% MRR. **Pipeline**: stage 1 (BM25 / dense, top-100~1000) → MonoBERT/T5 (top-10). **DuoT5**: pairwise reranking 으로 listwise 효과 |
| [02. RRF — Reciprocal Rank Fusion (Cormack 2009)](./ch6-reranking-hybrid/02-rrf.md) | **정의**: $\mathrm{RRF}(d) = \sum_{r \in R} \tfrac{1}{k + \mathrm{rank}_r(d)}$, $k = 60$ default. **Score-free**: BM25 score range 와 dense cosine range 의 normalization 불필요 — rank 만 사용. **수학적 성질**: harmonic-like decay → top rank 에 큰 가중. **놀라운 결과**: 단순한 RRF 가 score normalization + linear combination 을 보통 이김. BM25 + dense 의 hybrid retrieval 의 표준 baseline $\square$ |
| [03. Hybrid BM25+Dense — SPLADE 와 통합 모델](./ch6-reranking-hybrid/03-hybrid-bm25-dense.md) | **Complementary nature**: BM25 는 exact match · rare term · multi-word phrase 에 강함. Dense 는 paraphrase · synonym · cross-lingual 에 강함. **BEIR 분석**: zero-shot 에서 BM25 가 종종 dense 를 이김 (out-of-domain), in-domain 은 dense 우월. **SPLADE (Formal 2021)**: BERT MLM head 로 sparse lexical expansion 학습 — 출력이 vocabulary 위 sparse vector (BM25-style) 이지만 neural-trained quality. **Single model 로 lexical + semantic 통합** |
| [04. LLM-as-Reranker — RankGPT 와 listwise prompting](./ch6-reranking-hybrid/04-llm-as-reranker.md) | **RankGPT (Sun 2023)**: GPT-3.5/4 에 candidate list 를 입력 → permutation 출력 (listwise ranking). **Sliding window** for long candidate list. **Zero-shot** RankGPT-4 가 supervised SOTA cross-encoder 와 동등/우월. **비용**: API call latency 와 token cost — production 에서는 distillation (RankVicuna · RankZephyr) 로 small model 에 transfer. **언제 LLM 쓰나**: top-30 reranking 의 critical 정확도 필요, latency budget 충분 |

</details>

<br/>

### 🔹 Chapter 7: Advanced — GraphRAG · Multimodal · Frontier

> **핵심 질문:** GraphRAG (Edge 2024) 의 LLM-generated knowledge graph + Leiden community detection 이 왜 vector retrieval 만으로는 풀리지 않던 global question 의 새 frontier 인가? ColPali (Faysse 2024) 가 어떻게 OCR 없이 document page 를 이미지로 직접 retrieve 하는가? 1M+ context length 시대에 RAG 는 여전히 필요한가 — late chunking 과 long-context reranking 의 frontier 는?

<details>
<summary><b>GraphRAG · ColPali · Long Context (3개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. GraphRAG (Edge 2024)](./ch7-frontier/01-graphrag.md) | **Pipeline**: corpus → LLM (entity-relation extraction) → knowledge graph $G = (V, E)$ → **Leiden community detection** (modularity 최적화 → hierarchical clusters) → 각 community 를 LLM 이 summarize. **Global question** ("이 corpus 의 가장 중요한 주제 5 개?") 에서 vector RAG 압도 — top-$k$ chunk 만으로는 cross-document aggregation 불가능. **Local question** 은 vector RAG 와 hybrid. **Microsoft open-source release**, 2024 의 RAG frontier |
| [02. ColPali · Vision-RAG (Faysse 2024)](./ch7-frontier/02-colpali.md) | **Setup**: PDF page → image (no OCR) → **PaliGemma** VLM encoder → patch-level embedding. ColBERT-style **late interaction** between query token embedding 과 page patch embedding. **MaxSim**: $S(q, p) = \sum_i \max_j E_{q_i}^\top E_{p_j}$ (page 의 patch 단위). **장점**: OCR error 없음, layout · table · figure 의 visual context 보존, scanned document 처리. **ViDoRe benchmark** 에서 OCR + text retriever pipeline 압도 |
| [03. Long Context vs RAG · Late Chunking — Frontier](./ch7-frontier/03-long-context-late-chunking.md) | **Long context**: Gemini · GPT-4o 의 1M+ token. **Lost in the middle (Liu 2023)**: 중간 위치의 정보가 양 끝보다 약하게 활용 — context length 가 retrieval 을 완전 대체하지 못함. **Late Chunking (Jina AI 2024)**: 긴 문서 전체를 먼저 embedding (long-context encoder) → 그 다음 chunk 로 slice. Cross-chunk context 보존, chunk 경계의 정보 손실 회피. **결론**: long context 는 RAG 를 보완 (retrieval 정확도 ↓ 시 fallback) 하지만 대체하지 못함 — 둘의 hybrid frontier |

</details>

---

> 🆕 **2026-04 최신 업데이트**: Ch1-03 의 BM25 유도에 2-Poisson eliteness model 의 단계별 derivation 추가, Ch2-03 의 InfoNCE gradient 분석에서 hard negative 의 dominance 증명 강화, Ch3-02 의 ColBERT MaxSim 의 expressiveness 정리를 single-vector pooling 의 information bottleneck 관점으로 재정리, Ch4-04 의 PQ 와 Ch4-05 의 HNSW 의 complexity 증명을 명시적으로 분리, Ch5-04 의 Self-RAG reflection token 의 학습 protocol 을 step-by-step 으로 보강, Ch7-01 의 GraphRAG 에 Leiden algorithm 의 modularity 최적화 derivation 추가했습니다. **11-섹션 문서 골격이 전체 33개 문서에서 일관**됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현** 을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$ 로 종결되는 엄밀한 증명 또는 `results/` 하의 학습 곡선·plot 을 확인할 수 있습니다.

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **TF-IDF 의 정보이론적 정당화** | $\mathrm{IDF}(t) = \log(N/\mathrm{df}(t))$ 가 term 의 self-information $\log(1/p(t))$ 와 동치 | [Ch1-02](./ch1-ir-foundations/02-tf-idf.md) |
| **BM25 PRF Derivation** | 2-Poisson eliteness model 에서 saturating tf 와 length normalization 도출 | [Ch1-03](./ch1-ir-foundations/03-bm25.md) |
| **NDCG Position Discount** | $\mathrm{DCG} = \sum_i (2^{r_i} - 1)/\log_2(i+1)$ 의 graded relevance + log discount 정당화 | [Ch1-04](./ch1-ir-foundations/04-eval-metrics.md) |
| **Two-Stage Recall Bound** | Stage 2 의 max quality 는 stage 1 의 recall@k 에 bounded | [Ch1-05](./ch1-ir-foundations/05-retrieve-rerank.md) |
| **DPR Bi-Encoder Decomposition** | $s(q,p) = f_q(q)^\top f_p(p)$ — query 와 passage 의 absolute decoupling | [Ch2-02](./ch2-dense-retrieval/02-dpr-bi-encoder.md) |
| **InfoNCE In-Batch Negatives** | $B$ 샘플로 $B(B-1)$ negative pair, hard negative 가 gradient dominant | [Ch2-03](./ch2-dense-retrieval/03-infonce-in-batch.md) |
| **ColBERT MaxSim Expressiveness** | Single-vector pooling 의 정보 손실을 token-level interaction 으로 회복 | [Ch3-02](./ch3-late-interaction/02-colbert.md) |
| **PLAID Centroid Compression** | Centroid + residual 로 ColBERT 인덱스 2.6× 압축, exact MaxSim 보존 | [Ch3-03](./ch3-late-interaction/03-colbertv2-plaid.md) |
| **LSH Locality Property** | $\Pr[h(x) = h(y)] = 1 - \theta(x,y)/\pi$ — random projection 의 cosine 보존 | [Ch4-02](./ch4-ann/02-lsh.md) |
| **PQ Compression Bound** | $m \log_2 k$ bits 로 $k^m$ distinct codes — 32-64× FP32 대비 압축 | [Ch4-04](./ch4-ann/04-pq.md) |
| **HNSW Logarithmic Search** | Multi-layer + skip-list 분포 → $O(M \log N)$ greedy search | [Ch4-05](./ch4-ann/05-hnsw.md) |
| **RAG Marginal Likelihood** | $p(y\|x) = \sum_z p(y\|z,x) p(z\|x)$ — retriever 와 generator 의 end-to-end 학습 | [Ch5-01](./ch5-rag/01-vanilla-rag.md) |
| **RETRO Scaling Equivalence** | 25× smaller model 이 retrieval 로 GPT-3 동등 — knowledge externalization | [Ch5-02](./ch5-rag/02-retro.md) |
| **Self-RAG Reflection Tokens** | Vocabulary 확장으로 adaptive retrieval + self-evaluation 을 token-level 학습 | [Ch5-04](./ch5-rag/04-self-rag.md) |
| **RRF Score-Free Aggregation** | Rank-only fusion 이 score normalization + linear combination 을 능가 | [Ch6-02](./ch6-reranking-hybrid/02-rrf.md) |
| **SPLADE Lexical-Semantic Unification** | BERT MLM head 로 sparse vocabulary expansion — single-model lexical+semantic | [Ch6-03](./ch6-reranking-hybrid/03-hybrid-bm25-dense.md) |
| **GraphRAG Community Aggregation** | Leiden modularity 최적화 → hierarchical summary 가 global question 의 frontier | [Ch7-01](./ch7-frontier/01-graphrag.md) |
| **ColPali OCR-Free Retrieval** | VLM late interaction 이 OCR pipeline 의 error compounding 회피 | [Ch7-02](./ch7-frontier/02-colpali.md) |
| **Lost-in-the-Middle Bound** | 1M context 도 retrieval 을 완전 대체 못함 — late chunking 의 hybrid 정당화 | [Ch7-03](./ch7-frontier/03-long-context-late-chunking.md) |

> 💡 **챕터별 문서·정리/정의 수** (실측):
>
> | 챕터 | 문서 수 | 정리·정의 |
> |------|---------|------------|
> | Ch1 IR Foundations | 5 | 32 |
> | Ch2 Dense Retrieval | 5 | 30 |
> | Ch3 Late Interaction | 4 | 26 |
> | Ch4 ANN Algorithms | 6 | 37 |
> | Ch5 RAG Architectures | 6 | 33 |
> | Ch6 Reranking · Hybrid | 4 | 23 |
> | Ch7 Frontier | 3 | 18 |
> | **합계** | **33** | **199** |
>
> 추가로 **70+ 엄밀한 $\square$ 증명 + 99 연습문제 (모두 해설 포함) + 132 NumPy/PyTorch/FAISS/LangChain 실험 코드 (`### 실험 N` 형식, 33 문서 × 4 실험)**.
>
> Ch3, Ch6, Ch7 은 **3~4 문서** 로 구성 — late interaction · reranking · frontier 는 mature 주제만 다룸 (Chapter 1·2·5 의 5~6 문서, Ch4 의 6 문서와 의도적 차이).

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
torch==2.1.0
transformers==4.36.0
sentence-transformers==2.3.0   # SBERT · Contriever · E5 (Ch2)
rank-bm25==0.2.2               # BM25 baseline (Ch1)
faiss-cpu==1.7.4               # ANN (Ch4)
# faiss-gpu==1.7.4             # GPU FAISS (선택)
hnswlib==0.7.0                 # HNSW 단독 (Ch4-05)
nmslib==2.1.1                  # alternative ANN
pyserini==0.22.1               # Lucene-based BM25 + dense (Ch1, Ch6)
ranx==0.3.16                   # IR metric (NDCG · MAP · MRR) (Ch1-04)
beir==2.0.0                    # BEIR benchmark (Ch6, Ch7)
colbert-ai==0.2.19             # ColBERT v2 / PLAID (Ch3)
langchain==0.1.0               # RAG pipeline (Ch5)
langchain-community==0.0.13
llama-index==0.9.40            # alternative RAG framework (Ch5)
chromadb==0.4.22               # local vector DB (Ch4-06)
qdrant-client==1.7.0           # production vector DB (Ch4-06)
networkx==3.2                  # GraphRAG (Ch7-01)
python-louvain==0.16           # Leiden / Louvain community (Ch7-01)
matplotlib==3.8.0
seaborn==0.13.0
tqdm==4.66.0
jupyter==1.0.0
# 선택 사항
openai==1.10.0                 # LLM-as-reranker · GraphRAG (Ch6, Ch7)
anthropic==0.15.0              # Claude reranker
cohere==4.39                   # Cohere reranker
splade==0.1.0                  # SPLADE (Ch6-03)
```

```bash
# 환경 설치 (CUDA 12.1 기준)
pip install numpy==1.26.0 scipy==1.11.0 torch==2.1.0 \
            transformers==4.36.0 sentence-transformers==2.3.0 \
            rank-bm25==0.2.2 faiss-cpu==1.7.4 hnswlib==0.7.0 \
            pyserini==0.22.1 ranx==0.3.16 beir==2.0.0 \
            colbert-ai==0.2.19 langchain==0.1.0 llama-index==0.9.40 \
            chromadb==0.4.22 qdrant-client==1.7.0 \
            networkx==3.2 python-louvain==0.16 \
            matplotlib==3.8.0 seaborn==0.13.0 tqdm==4.66.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 ① — BM25 바닥부터 (Ch1-03)
import numpy as np
from collections import Counter

class BM25:
    """Robertson 1995 PRF 직접 구현"""
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.N = len(corpus)
        self.doc_lens = np.array([len(d) for d in corpus])
        self.avgdl = self.doc_lens.mean()
        self.tfs = [Counter(d) for d in corpus]
        self.df = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, t):
        df = self.df.get(t, 0)
        return np.log((self.N - df + 0.5) / (df + 0.5) + 1)   # Robertson-Sparck-Jones

    def score(self, query, idx):
        s = 0.0
        for t in query:
            tf = self.tfs[idx].get(t, 0)
            if tf == 0: continue
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[idx] / self.avgdl)
            s += self.idf(t) * num / den
        return s

# 대표 실험 ② — DPR 의 InfoNCE in-batch negatives (Ch2-03)
import torch, torch.nn as nn, torch.nn.functional as F

def dpr_loss(q_emb, p_emb_pos, p_emb_neg=None, tau=1.0):
    """L = -log exp(s(q, p+)/τ) / Σ exp(s(q, p)/τ)"""
    B = q_emb.size(0)
    scores = q_emb @ p_emb_pos.T / tau                  # [B, B], diag = positives
    if p_emb_neg is not None:                           # hard negatives 추가
        s_neg = torch.einsum('bd,bkd->bk', q_emb, p_emb_neg) / tau
        scores = torch.cat([scores, s_neg], dim=1)
    labels = torch.arange(B, device=q_emb.device)       # diag = correct pair
    return F.cross_entropy(scores, labels)

# 대표 실험 ③ — ColBERT MaxSim (Ch3-02)
def colbert_maxsim(E_q, E_d):
    """S = Σ_i max_j (E_qi · E_dj) — token-level late interaction"""
    sim = E_q @ E_d.T                                   # [|q|, |d|], normalized 가정
    return sim.max(dim=-1).values.sum()                 # max over doc, sum over query

# 대표 실험 ④ — HNSW 단순 구현 (Ch4-05)
class SimpleHNSW:
    """Multi-layer 생략한 single-layer greedy insert + search 골격"""
    def __init__(self, M=16):
        self.M = M
        self.graph = {}; self.data = {}

    def _d(self, a, b): return np.linalg.norm(a - b)

    def insert(self, idx, vec):
        self.data[idx] = vec; self.graph[idx] = []
        if len(self.data) == 1: return
        cands = sorted([(self._d(vec, self.data[j]), j) for j in self.data if j != idx])
        for d, j in cands[:self.M]:
            self.graph[idx].append((j, d))
            self.graph[j].append((idx, d))
            self.graph[j] = sorted(self.graph[j], key=lambda x: x[1])[:self.M]

    def search(self, q, k=10):
        if not self.data: return []
        entry = next(iter(self.data))
        visited = {entry}
        cands = [(self._d(q, self.data[entry]), entry)]
        results = []
        while cands:
            cands.sort()
            d_curr, curr = cands.pop(0)
            results = sorted(results + [(d_curr, curr)])[:k]
            for n, _ in self.graph[curr]:
                if n in visited: continue
                visited.add(n)
                cands.append((self._d(q, self.data[n]), n))
        return results

# 대표 실험 ⑤ — Product Quantization (Ch4-04)
from sklearn.cluster import KMeans

class ProductQuantizer:
    """x ∈ R^D → m subvectors → m bytes/vector (k=256)"""
    def __init__(self, m=8, k=256):
        self.m, self.k = m, k
        self.centroids = None

    def fit(self, X):
        N, D = X.shape
        assert D % self.m == 0
        sub_D = D // self.m
        self.centroids = []
        for i in range(self.m):
            km = KMeans(n_clusters=self.k, n_init=10).fit(X[:, i*sub_D:(i+1)*sub_D])
            self.centroids.append(km.cluster_centers_)

    def encode(self, X):
        N, D = X.shape; sub_D = D // self.m
        codes = np.zeros((N, self.m), dtype=np.int32)
        for i in range(self.m):
            X_sub = X[:, i*sub_D:(i+1)*sub_D]
            d = ((X_sub[:, None] - self.centroids[i][None]) ** 2).sum(-1)
            codes[:, i] = d.argmin(-1)
        return codes

# 대표 실험 ⑥ — Reciprocal Rank Fusion (Ch6-02)
def rrf(rankings, k=60):
    """RRF(d) = Σ 1/(k + rank_r(d)) — score-free hybrid"""
    scores = {}
    for ranking in rankings:               # ranking: list of doc_id, sorted by score desc
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

# 대표 실험 ⑦ — Vanilla RAG pipeline (Ch5-01)
class VanillaRAG:
    def __init__(self, retriever, llm):
        self.retriever, self.llm = retriever, llm

    def answer(self, question, k=5):
        passages = self.retriever.search(question, top_k=k)
        ctx = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
        prompt = f"Answer based on context.\n\nContext:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
        return self.llm.generate(prompt)
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격** 으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 기법이 retrieval·RAG 에 중요한가** | 해당 이론·기법이 search/RAG 의 어떤 핵심 한계를 푸는지 |
| 3 | 📐 **수학적 선행 조건** | LA · Prob · Info · Transformer 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | Vector space · graph · inverted index · MaxSim · RRF 의 시각화 |
| 5 | ✏️ **엄밀한 정의** | TF-IDF · BM25 · InfoNCE · MaxSim · HNSW layer · PQ codebook 등 |
| 6 | 🔬 **정리와 증명** | BM25 PRF 도출 · InfoNCE gradient · MaxSim expressiveness · HNSW complexity · RRF 우월성 등 |
| 7 | 💻 **Python / PyTorch / FAISS / 도메인별 라이브러리 구현 검증** | 4 가지 실험 (`### 실험 1` ~ `### 실험 4`) — toy corpus · MS MARCO · BEIR · pipeline ablation. 문서별로 적합한 라이브러리 사용 (BM25 → rank-bm25 / Pyserini · ANN → FAISS / hnswlib · Reranker → transformers · RAG → LangChain / LlamaIndex · 평가 → ranx · GraphRAG → networkx) |
| 8 | 🔗 **실전 활용** | Production RAG system · cost · domain mismatch · query distribution — 환경별 선택 가이드 |
| 9 | ⚖️ **가정과 한계** | Closed corpus · IID query · single-modality · annotation cost 등 |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 ($\boxed{}$ 핵심 수식 + Pareto frontier 위치) |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 기초 / 심화 / 논문 비평 의 3 문제, `<details>` 펼침 해설 |

> 📚 **연습문제 총 99개** (33 문서 × 3 문제): **기초 / 심화 / 논문 비평** 의 3-tier 구성, 모든 문제에 `<details>` 펼침 해설 포함. BM25 의 $k_1, b$ 손 튜닝부터 InfoNCE in-batch gradient 직접 도출, ColBERT MaxSim 의 expressiveness 증명, HNSW 의 layer 분포로 $O(\log N)$ derivation, PQ 의 codebook 직접 구축, Self-RAG 의 reflection token 학습 protocol 분석, GraphRAG 의 Leiden modularity 계산까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 500~600줄 (정의·증명·코드·연습문제 포함) 기준 **약 60분~1시간 30분**. 전체 33문서는 약 **35~45시간** 상당 (BM25/DPR/HNSW/PQ 직접 구현 · BEIR 실험 · GraphRAG community 재현 포함 시 60시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "RAG 는 만들어봤지만 BM25 + dense 가 왜 작동하는지 이론적으로 이해하고 싶다" — 입문 투어 (1주, 약 12~14시간)</b></summary>

<br/>

```
Day 1  Ch1-01  IR 정식화
       Ch1-03  BM25 (PRF derivation)
Day 2  Ch1-04  평가 metric (NDCG · MRR)
       Ch1-05  Two-stage pipeline
Day 3  Ch2-02  DPR bi-encoder
       Ch2-03  InfoNCE in-batch negatives
Day 4  Ch3-02  ColBERT late interaction
       Ch4-05  HNSW
Day 5  Ch4-04  Product Quantization
       Ch5-01  Vanilla RAG
Day 6  Ch5-04  Self-RAG (reflection token)
       Ch6-02  RRF
Day 7  Ch6-04  LLM-as-reranker
       Ch7-01  GraphRAG
```

</details>

<details>
<summary><b>🟡 "BM25 + Dense + ColBERT + Hybrid 의 통합 이론을 정복한다" — 이론 집중 (2주, 약 24~28시간)</b></summary>

<br/>

```
1주차 — IR Foundation · Dense · Late Interaction · ANN
  Day 1    Ch1-01~02   IR 정식화 + TF-IDF
  Day 2    Ch1-03~05   BM25 + metric + two-stage
  Day 3    Ch2-01~02   BM25 한계 + DPR
  Day 4    Ch2-03~05   InfoNCE + hard neg + SBERT/Contriever/E5
  Day 5    Ch3-01~02   Cross-encoder + ColBERT
  Day 6    Ch3-03~04   PLAID + multi/single-vector trade-off
  Day 7    Ch4-01~03   Exact NN 한계 + LSH + IVF

2주차 — ANN · RAG · Hybrid · Frontier
  Day 1    Ch4-04~06   PQ + HNSW + Vector DBs
  Day 2    Ch5-01~02   Vanilla RAG + RETRO
  Day 3    Ch5-03~04   REALM/Atlas + Self-RAG
  Day 4    Ch5-05~06   CRAG + FiD
  Day 5    Ch6-01~02   Cross-encoder reranker + RRF
  Day 6    Ch6-03~04   Hybrid BM25+Dense + LLM-as-reranker
  Day 7    Ch7-01~03   GraphRAG + ColPali + Long context vs RAG
```

</details>

<details>
<summary><b>🔴 "Retrieval & RAG 의 수학을 완전 정복한다" — 전체 정복 (8주, 약 35~45시간 + 재현 실험 15~20시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — IR Foundations
         → BM25 의 PRF 2-Poisson eliteness derivation 손 유도
         → NDCG 의 graded relevance + log discount 정당화
         → Two-stage pipeline 의 recall bound 증명
         → MS MARCO dev set 위에서 BM25 baseline 재현 (MRR@10 ≈ 0.18)

2주차   Chapter 2 전체 — Dense Retrieval
         → DPR contrastive training (NQ, BERT-base) 직접 학습
         → InfoNCE 의 in-batch + hard negative gradient 분석
         → Contriever 의 unsupervised cropping 재현
         → SBERT vs E5 의 embedding 시각화 (t-SNE)

3주차   Chapter 3 전체 — Late Interaction
         → Cross-encoder vs ColBERT 의 latency-quality 측정 (BEIR)
         → ColBERT MaxSim 의 token-level matching heatmap
         → ColBERTv2 PLAID 의 centroid + residual 직접 구현
         → Single vs multi-vector 의 storage-quality Pareto

4주차   Chapter 4 전체 — ANN Algorithms
         → LSH 의 random projection collision 검증
         → IVF + PQ 직접 구현 (1M vector × 768d)
         → HNSW 의 multi-layer graph 직접 구축, $O(\log N)$ 측정
         → FAISS · ScaNN · Qdrant 의 동일 데이터 위 recall-latency 비교

5주차   Chapter 5 전체 — RAG Architectures
         → Vanilla RAG (Lewis 2020) 의 marginal likelihood 학습 재현
         → RETRO 의 Chunked Cross-Attention layer 분석
         → Self-RAG 의 reflection token 학습 protocol 재현
         → CRAG 의 retrieval evaluator + web fallback 구현

6주차   Chapter 6 전체 — Reranking · Hybrid
         → MonoT5 reranker 학습 (MS MARCO triple)
         → RRF + cross-encoder 의 cascading pipeline
         → SPLADE 의 sparse expansion 재현
         → RankGPT 의 listwise prompting + sliding window

7주차   Chapter 7 전체 — Frontier
         → GraphRAG 의 entity-relation extraction + Leiden community
         → ColPali 의 page image embedding + MaxSim retrieval
         → Long context vs RAG 의 lost-in-the-middle 측정

8주차   종합 — End-to-End Production RAG
         → BEIR 14-task zero-shot eval (BM25 vs Contriever vs E5 vs ColBERTv2)
         → Hybrid pipeline 구축: Pyserini BM25 + E5 dense + RRF + MonoT5 rerank
         → GraphRAG + vector RAG 의 hybrid (local + global question)
         → 종합 토론: "현업의 best-practice RAG 가 BM25 baseline 을 왜 여전히 포함하는가"
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | Vector · Norm · Inner Product · SVD | **Ch1-02** (TF-IDF), **Ch2** (cosine), **Ch4** (PCA · PQ) |
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | Probabilistic IR · 2-Poisson · Bayes | **Ch1-03** (BM25 PRF) |
| [information-theory-deep-dive](https://github.com/iq-ai-lab/information-theory-deep-dive) | Entropy · KL · Mutual Information | **Ch1-02** (IDF), **Ch1-04** (NDCG), **Ch2-03** (InfoNCE) |
| [transformer-deep-dive](https://github.com/iq-ai-lab/transformer-deep-dive) | Attention · BERT · Tokenizer | **Ch2-02** (DPR), **Ch3-01** (cross-encoder), **Ch3-02** (ColBERT) |
| [llm-pretraining-deep-dive](https://github.com/iq-ai-lab/llm-pretraining-deep-dive) | Context window · Tokenizer 한계 | **Ch5-06** (FiD context), **Ch7-03** (long context) |
| [kernel-methods-deep-dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive) | Similarity as Kernel · Inner Product | **Ch2-02** (cosine kernel), **Ch3-02** (MaxSim) |
| [efficient-ml-deep-dive](https://github.com/iq-ai-lab/efficient-ml-deep-dive) | Quantization · KV cache · FlashAttention | **Ch3-03** (PLAID quantization), **Ch4-04** (PQ) |
| [llm-agents-deep-dive](https://github.com/iq-ai-lab/llm-agents-deep-dive) *(다음)* | Tool use · Planning · Multi-step reasoning | RAG-as-tool, agentic retrieval |

> 💡 이 레포는 **"BM25 · DPR · ColBERT · HNSW · RAG 가 모두 recall-precision-latency-cost 의 수학적으로 정당화된 trade-off 의 다른 구현이고, BM25 PRF · InfoNCE · MaxSim · HNSW small-world · IVF-PQ · RRF 가 왜 각각의 이론적 동기를 갖는가"** 에 집중합니다. Linear Algebra 에서 vector 와 SVD 를, Probability 에서 PRF 와 Bayes 를, Information 에서 entropy 와 KL 을, Transformer 에서 BERT 와 attention 을 익힌 후 오면 Chapter 2 (DPR-Contriever-E5 일직선) 와 Chapter 3 (ColBERT-PLAID) 의 derivation 이 훨씬 자연스럽습니다. **LLM Agents Deep Dive** 와 함께 보면 Ch5 의 Self-RAG · CRAG 가 agentic reasoning 의 retrieval 측면임이 선명해집니다.

---

## 📖 Reference

### 📚 IR Foundations · Textbook
- **Introduction to Information Retrieval** (Manning, Raghavan, Schütze, 2008) — 표준 교과서
- **The Probabilistic Relevance Framework: BM25 and Beyond** (Robertson & Zaragoza, 2009) — **BM25 정식**
- **Okapi at TREC-3** (Robertson, Walker, Jones, Hancock-Beaulieu, Gatford, 1995) — **BM25 효시**
- **A Statistical Interpretation of Term Specificity** (Spärck Jones, 1972) — **IDF 효시**
- **Search Engines: Information Retrieval in Practice** (Croft, Metzler, Strohman, 2009)
- **A Vocabulary Problem in Human-System Communication** (Furnas, Landauer, Gomez, Dumais, 1987) — vocabulary mismatch

### 🎯 Dense Retrieval
- **Dense Passage Retrieval for Open-Domain Question Answering** (Karpukhin, Oğuz, Min, Lewis, Wu, Edunov, Chen, Yih, 2020) — **DPR**
- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** (Reimers & Gurevych, 2019)
- **Unsupervised Dense Information Retrieval with Contrastive Learning** (Izacard, Caron, Hosseini, Riedel, Bojanowski, Joulin, Grave, 2022) — **Contriever**
- **Text Embeddings by Weakly-Supervised Contrastive Pre-training** (Wang, Yang, Huang, Jiao, Yang, Jiang, Majumder, Wei, 2022) — **E5**
- **Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval** (Xiong et al., 2021) — **ANCE**
- **Representation Learning with Contrastive Predictive Coding** (van den Oord, Li, Vinyals, 2018) — **InfoNCE**
- **SimCSE: Simple Contrastive Learning of Sentence Embeddings** (Gao, Yao, Chen, 2021)

### 🧩 Late Interaction · Cross-Encoder
- **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT** (Khattab & Zaharia, 2020)
- **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction** (Santhanam, Khattab, Saad-Falcon, Potts, Zaharia, 2022) — **PLAID**
- **PLAID: An Efficient Engine for Late Interaction Retrieval** (Santhanam et al., 2022)
- **Multi-Stage Document Ranking with BERT** (Nogueira, Yang, Cho, Lin, 2019) — **MonoBERT**
- **Document Ranking with a Pretrained Sequence-to-Sequence Model** (Nogueira, Jiang, Pradeep, Lin, 2020) — **MonoT5**

### 🔢 ANN · Vector Search
- **Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality** (Indyk & Motwani, 1998) — **LSH**
- **Product Quantization for Nearest Neighbor Search** (Jégou, Douze, Schmid, 2011)
- **Optimized Product Quantization** (Ge, He, Ke, Sun, 2013) — **OPQ**
- **Efficient and Robust Approximate Nearest Neighbor Search Using HNSW** (Malkov & Yashunin, 2018) — **HNSW**
- **Billion-scale similarity search with GPUs** (Johnson, Douze, Jégou, 2019) — **FAISS**
- **Accelerating Large-Scale Inference with Anisotropic Vector Quantization** (Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar, 2020) — **ScaNN**
- **DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node** (Subramanya et al., 2019)

### 🧠 RAG Architectures
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020) — **RAG 효시**
- **REALM: Retrieval-Augmented Language Model Pre-Training** (Guu, Lee, Tung, Pasupat, Chang, 2020)
- **Improving Language Models by Retrieving from Trillions of Tokens** (Borgeaud et al., 2022) — **RETRO**
- **Atlas: Few-shot Learning with Retrieval Augmented Language Models** (Izacard et al., 2022)
- **Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering** (Izacard & Grave, 2021) — **FiD**
- **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** (Asai, Wu, Wang, Sil, Hajishirzi, 2024)
- **Corrective Retrieval Augmented Generation** (Yan, Xu, Wang, Wang, Wang, Wang, 2024) — **CRAG**
- **Retrieval-Augmented Generation for Large Language Models: A Survey** (Gao et al., 2024)

### 🔀 Reranking · Hybrid
- **Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods** (Cormack, Clarke, Buettcher, 2009) — **RRF**
- **SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking** (Formal, Piwowarski, Clinchant, 2021)
- **SPLADEv2: Sparse Lexical and Expansion Model for Information Retrieval** (Formal et al., 2021)
- **Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents** (Sun et al., 2023) — **RankGPT**
- **RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models** (Pradeep, Sharifymoghaddam, Lin, 2023)

### 🌐 Frontier · GraphRAG · Multimodal
- **From Local to Global: A Graph RAG Approach to Query-Focused Summarization** (Edge et al., 2024) — **GraphRAG**
- **From Louvain to Leiden: guaranteeing well-connected communities** (Traag, Waltman, van Eck, 2019) — Leiden algorithm
- **ColPali: Efficient Document Retrieval with Vision Language Models** (Faysse et al., 2024)
- **Lost in the Middle: How Language Models Use Long Contexts** (Liu et al., 2023)
- **Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models** (Günther et al., 2024) — Jina AI

### 📊 Benchmarks · Datasets
- **MS MARCO: A Human Generated MAchine Reading COmprehension Dataset** (Bajaj et al., 2016)
- **Natural Questions: a Benchmark for Question Answering Research** (Kwiatkowski et al., 2019)
- **TriviaQA: A Large Scale Distantly Supervised Challenge Dataset** (Joshi, Choi, Weld, Zettlemoyer, 2017)
- **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models** (Thakur, Reimers, Rücklé, Srivastava, Gurevych, 2021)
- **LoTTE: Long-Tail Topic-stratified Evaluation for IR** (Santhanam et al., 2022)
- **KILT: a Benchmark for Knowledge Intensive Language Tasks** (Petroni et al., 2021)
- **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering** (Yang et al., 2018)

### 🛠️ Implementation · Libraries
- **FAISS** (Johnson et al., 2019) — Meta vector search
- **Pyserini** (Lin et al., 2021) — Lucene + dense retrieval
- **sentence-transformers** (Reimers & Gurevych, 2019) — SBERT 생태계
- **ColBERT-AI** — official ColBERT v2 / PLAID
- **LangChain** — RAG framework, multi-LLM
- **LlamaIndex** — RAG framework, indexing focus
- **Chroma · Qdrant · Milvus · Weaviate · LanceDB** — vector DBs
- **ranx** — IR metric (NDCG · MAP · MRR · Recall) 표준 구현
- **BEIR** — zero-shot IR benchmark suite

---

<div align="center">

**⭐️ 도움이 되셨다면 Star 를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"BM25 + Dense Hybrid 를 호출하는 것과 — Robertson 1995 의 BM25 가 PRF 의 2-Poisson eliteness model 에서 정확히 도출됨을 증명 · Karpukhin 2020 의 DPR InfoNCE 에서 in-batch negatives 가 어떻게 $B$ 샘플로 $O(B^2)$ 학습 신호를 만드는지 손 유도 · Khattab & Zaharia 2020 의 ColBERT MaxSim 이 single-vector pooling 의 정보 손실을 token-level interaction 으로 회복함을 정량화 · Malkov & Yashunin 2018 의 HNSW 가 small-world graph 의 layer 분포로 어떻게 $O(\log N)$ 을 보장하는지 derive · Jégou 2011 의 PQ 가 어떻게 $m \log_2 k$ bits 만으로 $k^m$ distinct codes 를 표현하는지 분석 · Cormack 2009 의 RRF 가 왜 score normalization 없이도 hybrid retrieval 의 표준 baseline 이 되는지 증명 — 그리고 Asai 2024 의 Self-RAG 의 reflection token 이 왜 단순 prompt 가 아닌 vocabulary expansion 으로 학습되어야 하는지 — 이 모든 '왜' 를 직접 유도할 수 있는 것은 다르다"*

</div>
