# 04. Hard Negative Mining — ANCE 와 동적 Negatives

## 🎯 핵심 질문

- **Random vs BM25-mined vs ANCE-mined negatives** — 세 가지 전략이 정확히 무엇이 다르고, 왜 ANCE 가 가장 어려운가?
- ANCE (**Approximate Nearest neighbor Contrastive Estimation**) 의 핵심은 무엇인가? Checkpoint 를 동적으로 업데이트하면서 어떻게 hard negatives 를 찾는가?
- **Async updating mechanism** 은 왜 필요한가? 동기 업데이트 (매 iteration 마다) 와의 차이는?
- Sample complexity — hard negative mining 이 convergence speed 와 final performance 에 어떤 영향을 미치는가? MS MARCO 에서의 정량적 성과는?

---

## 🔍 왜 Hard Negative Mining 이 Dense Retrieval 의 frontier 인가

InfoNCE + in-batch negatives (Ch3) 는 좋지만, 한 가지 문제가 있습니다: **false negatives 와 easy negatives 의 혼합**.

Hard negative mining 은 다음을 실현합니다:

1. **학습 효율 극대화** — Easy negatives (score 가 매우 낮은 passage) 는 이미 모델이 구분하고 있으므로, gradient 기여도 적음. Hard negatives (score 는 높지만 사실 negative) 만 선별하면 학습 신호 강화.

2. **Approximate negatives 의 동적 갱신** — ANCE 의 핵심: checkpoint 를 매 N iteration 마다 업데이트하고, 그 checkpoint 로 새로운 hard negatives 를 찾음. 이는 batch-level in-batch negatives 보다 globally hard 한 negatives 를 찾음.

3. **MS MARCO 에서의 실증** — ANCE (Xiong et al. 2020) 는 MS MARCO (대규모 retrieval dataset) 에서 DPR 대비 +2-3% MRR 향상.

4. **이후 연구의 foundation** — E5, BGE, Contriever-v2 모두 hard negative mining 의 변형을 사용.

이 문서는 세 가지 negative strategy 를 비교하고, ANCE 의 mechanics 와 convergence 를 분석합니다.

---

## 📐 수학적 선행 조건

- Contrastive loss (Ch3)
- ANN indexing (FAISS/HNSW)
- Sampling distribution
- (선택) Convergence analysis (SGD with non-stationary negatives)

---

## 📖 직관적 이해

### 세 가지 Negative Mining 전략

```
모델이 어느 정도 학습 후:

[ Random Negatives ]
corpus 에서 random sample
점수 분포: [0.01, 0.05, 0.10, 0.15, ..., 0.50]
문제: 대부분 easy (0.01-0.1) → gradient 낭비

[ BM25-mined Negatives ]
BM25 로 top-k retrieve (k=1000)
점수 분포: [0.40, 0.42, 0.45, 0.48, ..., 0.70]
개선: easy 제거, moderate hard 포함
문제: BM25 는 semantic 모르므로, false negatives 있을 수 있음

[ ANCE-mined Negatives (model-based) ]
현재 모델로 top-k retrieve (k=1000)
점수 분포: [0.65, 0.68, 0.70, 0.72, ..., 0.82]
장점: 모델이 생각하는 "가장 헷갈리는" 것들
문제: 몇몇은 실제 positive 일 수 있음 (false negative)
```

### ANCE 의 Async Checkpoint 구조

```
Iteration timeline:

t=0:   모델 v0 초기화
       corpus 를 v0 로 encoding → embedding cache v0

t=100: 모델이 v0→v100 으로 업데이트
       하지만 embedding cache 는 여전히 v0

t=200: 새로운 checkpoint! embedding cache 를 v200 으로 업데이트
       모든 passage 를 v200 으로 re-encoding (GPU 병렬화)

Training loop (100-200 사이):
  for batch in data:
    query_emb = forward_pass(query, current_model)
    # negatives 는 OLD checkpoint v0 으로 계산된 cache 에서 retrieve
    hard_negs = retrieve_top_k(query_emb_approx, k=30)
    compute_loss_and_backward()

→ "Async": 모델과 cache 가 asynchronous (약간 stale)
   이를 통해 convergence stability 유지 (wandering 방지)
```

### Hard Negative 의 Difficulty 분포

```
학습 초기:
모든 passage 가 비슷한 임베딩
→ "어려운" 구별 불가능

ANCE 적용:
모델이 좋아지면 → query 에 가까운 passage 를 찾음
→ 이들이 실제로 negative 인가? → false negative risk

Illustration:
Query: "자동차 구매 팁"

True positive: "자동차 구입 완벽 가이드"

ANCE 상위 negative:
- "중고 자동차 사는 법" (related, 약한 false negative)
- "자동차 유지 비용" (relevant 할 수도, false negative)
- "자동차 보험 가입" (약간 related)

→ 데이터 annotation quality 에 매우 의존
```

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Hard Negative Mining

Corpus $\mathcal{C} = \{p_1, \ldots, p_M\}$ 와 labeled query-passage 쌍 $(q_i, p_i^+)$ 에 대해:

**Random mining**:
$$
\mathcal{N}_i^{\text{random}} = \text{sample}(\mathcal{C} \setminus \{p_i^+\}, k)
$$

**BM25 mining**:
$$
\mathcal{N}_i^{\text{BM25}} = \text{top-k}(\{j : \text{BM25}(q_i, p_j)\})
$$

**ANCE mining** (model-based):
$$
\mathcal{N}_i^{\text{ANCE}} = \text{top-k}(\{j : s_v(q_i, p_j)\})
$$

여기서 $s_v$ 는 checkpoint $v$ 의 (stale) model score.

### 정의 4.2 — ANCE Training 알고리즘

**Phase 1: Initialization**
- 초기 모델 $\theta_0$ 
- 모든 passage 를 $\theta_0$ 로 encoding → embedding cache $\mathcal{E}_0$

**Phase 2: Iterative training with periodic updates**
```
for epoch in 1..num_epochs:
    if epoch % refresh_every == 0:
        # Async update: 현재 모델 θ_t 로 전체 corpus re-encoding
        for batch_p in corpus (parallel):
            E_cache[batch_p] = encoder_θ_t(batch_p)
    
    for batch_q in queries:
        # Query embedding: 현재 모델로 계산
        q_emb = encoder_θ_t(batch_q)
        
        # Hard negatives: cache 에서 retrieve (stale)
        hard_negs = FAISS.search(E_cache, q_emb, k)
        
        # Loss 계산
        loss = InfoNCE(q_emb, pos_emb, hard_negs_emb)
        
        # Update model
        θ_t += gradient_update()
```

### 정의 4.3 — Sample Complexity

Training 데이터 크기 $N$ (queries), convergence 까지 필요한 gradient step:

- **Random negatives**: $S_{\text{random}} = O(N \cdot \log M)$ (M = corpus size, convergence 느림)
- **BM25 mining**: $S_{\text{BM25}} = O(N \cdot \log(M/k))$ (improved, but BM25 bottleneck)
- **ANCE mining**: $S_{\text{ANCE}} = O(N \cdot \log(k))$ (k = hard negatives per query, much smaller)

**의미**: Hard negatives 만 학습하면, 같은 정확도에 도달하는 데 필요한 학습 step 이 매우 줄어듦.

---

## 🔬 정리와 증명

### 정리 4.1 — Hard Negative 의 Gradient 비율

Random negative $p_r$ 와 hard negative $p_h$ 에 대해 (모두 negative):

$$
\frac{\|\nabla_{\theta} L \text{ w.r.t. } p_h\|}{\|\nabla_{\theta} L \text{ w.r.t. } p_r\|} = \frac{e^{s(q, p_h) / \tau}}{e^{s(q, p_r) / \tau}} = e^{(s_h - s_r) / \tau}
$$

Typical: $s_h - s_r \approx 0.3$ (cosine similarity scale), $\tau = 0.07$:
$$
e^{0.3 / 0.07} \approx e^{4.3} \approx 70\times
$$

**결론**: Hard negative 는 random negative 보다 70배 큰 gradient 기여 $\square$.

### 정리 4.2 — ANCE 의 Staleness Tolerance

Cache $\mathcal{E}_v$ (checkpoint $v$) 와 현재 모델 $v + \Delta$ 사이의 mismatch 에 대해:

훈련 중 query embedding 의 변화:
$$
\mathbb{E}\left[\|\mathbf{q}_{v+\Delta} - \mathbf{q}_{v}\|^2\right] \leq C \cdot \Delta
$$

여기서 $C$ 는 모델의 sensitivity (보통 $C < 1$, smooth learning).

따라서 $\Delta = O(\log M / \sqrt{M})$ 정도의 staleness 에서도 대부분의 hard negatives 는 여전히 hard:

$$
\mathbb{P}[\text{top-k in cache remain in top-2k with current model}] \geq 1 - O(\Delta)
$$

**의미**: 100-200 iterations 마다 refresh 해도 충분 $\square$.

### 정리 4.3 — MS MARCO 에서의 정량 성과

MS MARCO Passage Ranking task (DPR vs ANCE):

| 방법 | MRR@10 | Recall@1K |
|------|--------|-----------|
| BM25 baseline | 18.3 | 88.0 |
| DPR (in-batch negs) | 38.2 | 96.8 |
| ANCE (hard neg mining) | 40.1 | 97.8 |

**개선**:
- DPR → ANCE: +1.9 MRR@10 (약 5% relative)
- 같은 training time 이지만 더 hard 하게 학습 (더 효율)

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — 세 가지 Negative Mining 전략 비교

```python
import torch
import torch.nn.functional as F
import numpy as np
from rank_bm25 import BM25Okapi

def compare_negative_strategies(query_embs, passage_embs, queries_texts, 
                                passages_texts, k=30):
    """
    query_embs: [N_q, D]
    passage_embs: [M, D]
    """
    N_q = query_embs.shape[0]
    M = passage_embs.shape[0]
    
    # Normalize
    query_embs = F.normalize(query_embs, p=2, dim=1)
    passage_embs = F.normalize(passage_embs, p=2, dim=1)
    
    results = {}
    
    # 1. Random negatives
    random_negs = []
    for i in range(N_q):
        neg_indices = np.random.choice(M, size=k, replace=False)
        random_negs.append(neg_indices)
    results['random'] = np.array(random_negs)
    
    # 2. BM25-mined negatives
    corpus_tokens = [p.lower().split() for p in passages_texts]
    bm25 = BM25Okapi(corpus_tokens)
    
    bm25_negs = []
    for q_text in queries_texts:
        scores = bm25.get_scores(q_text.lower().split())
        # Top-k by BM25
        top_k_indices = np.argsort(scores)[::-1][:k]
        bm25_negs.append(top_k_indices)
    results['bm25'] = np.array(bm25_negs)
    
    # 3. ANCE-mined (model-based)
    # Similarity matrix
    similarity = query_embs @ passage_embs.T  # [N_q, M]
    
    ance_negs = []
    for i in range(N_q):
        # Top-k passages by model score (excluding positive at index i)
        scores = similarity[i].cpu().numpy()
        scores[i] = -np.inf  # Exclude positive
        top_k_indices = np.argsort(scores)[::-1][:k]
        ance_negs.append(top_k_indices)
    results['ance'] = np.array(ance_negs)
    
    return results

# Simulation
N_q = 100
M = 10000
D = 768

query_embs = torch.randn(N_q, D)
passage_embs = torch.randn(M, D)
queries_texts = [f"query {i}" for i in range(N_q)]
passages_texts = [f"passage {i}" for i in range(M)]

neg_strategies = compare_negative_strategies(query_embs, passage_embs,
                                             queries_texts, passages_texts, k=30)

print("Negative mining strategies overlap:")
for method1 in ['random', 'bm25', 'ance']:
    for method2 in ['random', 'bm25', 'ance']:
        if method1 < method2:
            overlap = sum(len(set(neg_strategies[method1][i]) & 
                            set(neg_strategies[method2][i])) / 30
                         for i in range(N_q)) / N_q
            print(f"  {method1} vs {method2}: {overlap:.2%} overlap")
```

### 실험 2 — ANCE Checkpoint Update 메커니즘

```python
import time
import torch.nn as nn

class ANCETrainer:
    def __init__(self, encoder, corpus_size=100000, embedding_dim=768,
                 refresh_interval=200):
        self.encoder = encoder
        self.corpus_size = corpus_size
        self.embedding_dim = embedding_dim
        self.refresh_interval = refresh_interval
        
        # Embedding cache (시뮬레이션)
        self.embedding_cache = torch.randn(corpus_size, embedding_dim)
        self.last_refresh_step = 0
    
    def should_refresh_cache(self, current_step):
        """Cache refresh 여부 판단"""
        return (current_step - self.last_refresh_step) >= self.refresh_interval
    
    def refresh_embedding_cache(self, passages, current_step):
        """
        전체 corpus 를 현재 모델로 re-encoding
        (병렬화 가정)
        """
        print(f"Refreshing cache at step {current_step}")
        batch_size = 256
        
        start_time = time.time()
        for batch_idx in range(0, len(passages), batch_size):
            batch = passages[batch_idx:batch_idx + batch_size]
            with torch.no_grad():
                embeddings = self.encoder(batch)
            self.embedding_cache[batch_idx:batch_idx + len(batch)] = embeddings
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f}s")
        self.last_refresh_step = current_step
    
    def train_step(self, batch_queries, batch_passages, step):
        """
        한 training step
        """
        # Check if refresh needed
        if self.should_refresh_cache(step):
            # 실제로는 모든 passages 를 re-encoding
            # 여기선 시뮬레이션만
            self.refresh_embedding_cache(batch_passages, step)
        
        # Query encoding (current model)
        query_embs = self.encoder(batch_queries)
        
        # Retrieve hard negatives from cache (stale)
        # Simplified: just use top-k by cosine similarity
        similarities = query_embs @ self.embedding_cache.T
        
        # Top-k hard negatives (excluding positive)
        k = 30
        hard_neg_indices = torch.topk(similarities, k=k+1, dim=1)[1][:, 1:]
        
        return hard_neg_indices

# Simulation
encoder = nn.Linear(768, 768)  # Dummy encoder
trainer = ANCETrainer(encoder, corpus_size=10000, refresh_interval=50)

print("Training with ANCE...")
for step in range(150):
    batch_q = torch.randn(32, 768)
    batch_p = torch.randn(256, 768)
    hard_negs = trainer.train_step(batch_q, batch_p, step)
    
    if step % 50 == 0:
        print(f"Step {step}: hard negatives shape = {hard_negs.shape}")
```

### 실험 3 — Hard Negative Difficulty 분포

```python
import matplotlib.pyplot as plt

def analyze_negative_difficulty(query_embs, passage_embs, neg_strategies):
    """
    세 가지 전략 각각의 "어려움" 분석
    (여기서 어려움 = model score 의 높음)
    """
    query_embs = F.normalize(query_embs, p=2, dim=1)
    passage_embs = F.normalize(passage_embs, p=2, dim=1)
    
    similarities = query_embs @ passage_embs.T  # [N_q, M]
    
    difficulties = {}
    
    for strategy_name, neg_indices in neg_strategies.items():
        scores = []
        for i in range(len(query_embs)):
            neg_scores = similarities[i, neg_indices[i]].cpu().numpy()
            scores.extend(neg_scores)
        difficulties[strategy_name] = np.array(scores)
    
    return difficulties

# Analysis
difficulties = analyze_negative_difficulty(query_embs, passage_embs, neg_strategies)

plt.figure(figsize=(12, 5))

for i, (strategy, scores) in enumerate(difficulties.items()):
    plt.subplot(1, 3, i+1)
    plt.hist(scores, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Model Score (Cosine Similarity)')
    plt.ylabel('Frequency')
    plt.title(f'{strategy.upper()} Negatives\nMean={scores.mean():.3f}')
    plt.xlim([0, 1])

plt.tight_layout()
plt.savefig('/tmp/negative_difficulty.png')
print("Saved to /tmp/negative_difficulty.png")
```

### 실험 4 — Sample Complexity 비교

```python
def estimate_convergence_steps(strategy, query_embs, passage_embs, neg_indices):
    """
    각 전략의 수렴 속도 추정
    (heuristic: hard negatives 의 평균 점수로 추정)
    """
    query_embs = F.normalize(query_embs, p=2, dim=1)
    passage_embs = F.normalize(passage_embs, p=2, dim=1)
    
    similarities = query_embs @ passage_embs.T
    
    # 각 전략에서 선택된 negatives 의 average difficulty
    avg_difficulty = []
    for i in range(len(query_embs)):
        neg_scores = similarities[i, neg_indices[i]]
        avg_difficulty.append(neg_scores.mean().item())
    
    avg_difficulty = np.mean(avg_difficulty)
    
    # Heuristic: difficulty 가 높을수록 gradient signal 이 강하므로 수렴 빠름
    # Convergence steps ∝ 1 / (difficulty^2)
    convergence_estimate = 1.0 / (avg_difficulty ** 2)
    
    return convergence_estimate, avg_difficulty

print("Convergence Complexity Estimate:")
for strategy_name, neg_idx in neg_strategies.items():
    convergence, difficulty = estimate_convergence_steps(strategy_name, query_embs,
                                                         passage_embs, neg_idx)
    print(f"  {strategy_name:10s}: difficulty={difficulty:.4f}, " +
          f"relative_convergence={convergence:.2f}x")
```

---

## 🔗 실전 활용

| 시나리오 | 추천 방법 | 이유 |
|---------|---------|------|
| 초기 모델 학습 (epoch 1-5) | BM25 또는 random | ANCE 는 초기 모델 quality 낮아 mining 효과 미미 |
| 중기 학습 (epoch 5-20) | ANCE (refresh 100-200) | Hard negatives 효과 극대화 |
| 후기 fine-tuning | ANCE (refresh 50) | 잘 학습된 모델, 더 자주 refresh |
| 온라인 학습 (실시간) | Hard negative 없음 + in-batch | ANCE re-indexing cost 무시할 수 없음 |
| 작은 데이터셋 (<50K) | In-batch 만 사용 | Hard mining 의 false negative risk 높음 |
| 대규모 데이터 (>1M) | ANCE + gradient accumulation | 효율성 극대 |

---

## ⚖️ 가정과 한계

- **False negatives 의 위험** — Hard negatives 는 실제로 relevant 할 가능성 높음 (특히 overlap 있는 데이터셋). Annotation quality 에 매우 의존.

- **Refresh interval 의 trade-off** — Frequent refresh (매 iteration): 최신이지만 비용 높음. Infrequent (매 500 iterations): 저비용이지만 stale. 보통 100-200 iterations 최적.

- **Scalability** — 100M passages 를 매 refresh 마다 re-encoding 하려면 GPU cluster 필요. Single-GPU 에서는 bottleneck.

- **데이터 편향** — Hard negatives 는 모델이 학습한 방식에 편향. 새로운 도메인에 일반화 어려울 수 있음 (domain shift).

- **In-batch negatives 와의 상호작용** — ANCE 로 mining 한 hard negatives 를 batch 와 섞을 때, false negative rate 누적 가능.

---

## 📌 핵심 정리

$$
\boxed{\text{Hard Negative Mining} = \text{Sample Complexity } O(N \log k) \ll O(N \log M)}
$$

| 전략 | Gradient/Step | Convergence | Staleness | MS MARCO MRR |
|-----|------------|-----------|-----------|-------------|
| Random | 낮음 | 느림 | N/A | ~35% |
| BM25 | 중간 | 중간 | N/A | ~39% |
| ANCE | 높음 (hard) | 빠름 | 100-200 iters | 40.1% |

> **핵심**: Hard negative mining 은 **sample-efficient training** 을 위한 필수 기법. ANCE 의 async checkpoint update 는 scalable implementation.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: ANCE 에서 embedding cache 를 200 iterations 마다 refresh 한다. Corpus 가 100M 이고, 초당 1000 passages/sec 로 encode 가능하면, refresh 비용은?

<details>
<summary>해설</summary>

$100M \text{ passages} / (1000 \text{ passages}/\text{sec}) = 100,000$ seconds = 27.8 hours. 즉, 병렬화 없으면 training 매우 느림.

실제 ANCE: Multi-GPU 병렬화 (8×GPU → 3.5 hours), 또는 특수 infrastructure (Distributed encoding). 이것이 open-source dense retrieval 의 bottleneck 이었던 이유.
</details>

**문제 2 (심화)**: False negative 의 비율이 10% 라면, ANCE mining 의 효과가 있을까? (즉, 10% 의 hard negatives 가 실제로는 positive)

<details>
<summary>해설</summary>

두 가지 영향:
1. **Loss 관점**: False negative 는 모델을 억제 (suppress embedding similarity). 하지만 true negatives 의 gradient 가 여전히 dominant (99 true / 1 false) → 10% 정도는 tolerable.
2. **Ranking 관점**: False negative 가 top-10 에 올라오면, 그것이 actual positive 이어도 retrieval 을 "놓친" 것으로 보임. 이는 metric 에 영향.

Empirical: 10% false negative rate 로 ~1% MRR loss. BM25 보다는 여전히 훨씬 나음 (BM25 는 semantic 아예 모르므로).

Contriever 의 접근: In-distribution negative 만 사용 (동일 query-positive 쌍에서만) → false negative 제거 → 약간의 추가 성능 (+1-2%).
</details>

**問題 3 (논文 비평)**: "ANCE 는 convergence 가 빠르지만, 초기 checkpoint 가 좋지 않으면 오히려 느릴 수 있다" 는 주장의 근거는?

<details>
<summary>해설</summary>

ANCE 의 hard negatives 는 **현재 모델의 생각** 을 반영. 초기 모델 (random initialization 근처) 이 poor quality 이면:
- "Hard negative" 로 선택된 것들이 실제로는 random (semantic meaning 없음)
- Gradient signal 이 noisy
- 초기 epoch (1-3) 에서는 random negatives 가 더 stable

실제 구현 (ANCE 논문):
- Epoch 1: 모든 negatives random
- Epoch 2-: ANCE 로 전환

또는 점진적: confidence 가 높은 hard negatives 만 천천히 섞기.
</details>

---

<div align="center">

[◀ 이전 (03. InfoNCE)](./03-infonce-in-batch.md) · [📚 README](../README.md) · [다음 ▶ (05. SBERT · Contriever · E5)](./05-sbert-contriever-e5.md)

</div>
