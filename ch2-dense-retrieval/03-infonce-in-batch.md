# 03. InfoNCE Loss 와 In-Batch Negatives

## 🎯 핵심 질문

- **InfoNCE (Information NCE, Oord 2018)** 손실 함수는 정확히 무엇이고, contrastive learning 에서 왜 standard 가 되었는가?
- **In-batch negatives** 전략은 어떻게 작동하는가? Batch size $B$ 개의 passage 에서 갑자기 $B(B-1)$ 개의 negative pair 가 생기는 이유는?
- Temperature parameter $\tau$ 의 역할은? $\tau$ 가 작아지면 왜 hard negative 에 집중하게 되는가?
- Gradient 분석 — in-batch negatives 에서 **어떤 passage 들이 loss 를 dominate** 하고, 그것이 sampling strategy 에 미치는 영향은?

---

## 🔍 왜 InfoNCE 와 in-batch negatives 가 Dense Retrieval 의 핵심인가

DPR 의 학습에서 InfoNCE loss 와 in-batch negatives 는 명시적으로 등장하지 않았지만, 이후 모든 dense retrieval 방법 (Contriever, E5, etc.) 의 foundation 입니다:

1. **Noise Contrastive Estimation (NCE) 의 한계 극복** — Traditional NCE 는 negative sampling 에 computationally expensive. In-batch negatives 는 batch 의 다른 passage 들을 "free" negative 로 재활용.

2. **Batch 효율성** — Batch size $B$ 에서 $B$ 개의 positive pair 와 동시에 $O(B^2)$ negative pair 학습. 이는 negative sampling rate 를 기하급수적으로 높인다.

3. **Temperature scheduling 의 효과** — $\tau$ 를 조정하여 training 의 초반 (high $\tau$, soft) 에서 후반 (low $\tau$, hard) 로 transition. 이는 curriculum learning 효과.

4. **Gradient dynamics 분석** — Hard negative (false negative 일 확률 높음) 가 gradient 를 지배하면, sampling strategy 를 바꾸어야 한다는 신호 → Hard negative mining (Ch4).

이 문서는 InfoNCE 의 정식화, in-batch negatives 의 mechanics, 그리고 gradient analysis 를 다룹니다.

---

## 📐 수학적 선행 조건

- 확률론: softmax, cross-entropy, KL divergence
- Calculus: gradient, Hessian (선택)
- Sampling: importance sampling, sampling bias

---

## 📖 직관적 이해

### Contrastive Learning 의 비유

```
목표: 고양이와 개를 분류하는 모델 학습

Cross-Entropy (분류 기반):
- "이 동물은 고양이? 개?" → 2-class classification
- Binary decision boundary

Contrastive Learning (embedding 기반):
- "이 동물의 임베딩을 다른 고양이들과는 가깝게, 개들과는 멀게"
- Embedding space 에서의 clustering

Info NCE:
- Positive: 같은 범주의 다른 사진 (같은 고양이 사진들)
- Negative: 다른 범주 사진 (개들)
- Loss: "고양이 사진끼리는 가까워, 개와는 멀어" 를 확률로 표현
```

### In-Batch Negatives 의 구조

```
Batch size B = 4 (간단 예시):

Query embeddings:        q_1, q_2, q_3, q_4
Positive passages:       p_1+, p_2+, p_3+, p_4+
(in-batch) negatives:    p_2+, p_3+, p_4+ (for q_1)
                         p_1+, p_3+, p_4+ (for q_2)
                         ...

Pair visualization:
q_1 vs p_1+ (positive)  ✓  score high
q_1 vs p_2+ (negative)  ✗  score low
q_1 vs p_3+ (negative)  ✗  score low
q_1 vs p_4+ (negative)  ✗  score low

→ 1 query, 1 positive, 3 negatives
→ Per-batch: 4 positives, 4*(4-1)=12 negatives 동시 학습

Total pairs ∼ B(B-1) instead of B
```

### Temperature 의 역할

```
τ (temperature) 가 softmax 를 어떻게 조정:

High τ (e.g., 0.5):
  softmax(score/0.5) → smooth distribution
  모든 negative 가 비슷한 loss contribution
  curriculum 초기 (global structure 학습)

Low τ (e.g., 0.05):
  softmax(score/0.05) → sharp distribution
  highest-scoring negative 가 loss 지배
  curriculum 후반 (hard negative focus)

Visualization:
      τ=0.5          τ=0.05
    ▁▂▃▄▅▆▇         ▁▁▁▁▁██
  softmax output (smooth vs peaky)
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — InfoNCE Loss

Positive pair $(q, p^+)$ 와 negative set $\{p_1^-, p_2^-, \ldots, p_{K}^-\}$ 에 대해:

$$
L_{\text{InfoNCE}} = -\log \frac{\exp(s(q, p^+) / \tau)}{\exp(s(q, p^+) / \tau) + \sum_{i=1}^{K} \exp(s(q, p_i^-) / \tau)}
$$

여기서:
- $s(q, p) = f_q(q)^\top f_p(p)$ : inner product score
- $\tau$ : temperature parameter ($\tau > 0$)
- 분자: positive 의 확률 (높아야 함)
- 분모: positive + all negatives (확률 분포)

**재해석** (분류로):
$$
L = \text{CrossEntropy}([\text{positive_score}, \text{negative_scores}])
$$

Binary classification: "이 passage 가 positive 인가?" → softmax 확률.

### 정의 3.2 — In-Batch Negatives

Batch size $B$ 에서:
$$
\text{Positives}: (q_i, p_i^+) \quad \text{for } i = 1, \ldots, B
$$

$$
\text{In-batch negatives}: \{p_j^+ : j \neq i\} \quad \text{for } q_i
$$

따라서 각 query $q_i$ 에 대해:
- 1 positive: $p_i^+$
- $B-1$ negatives: $p_j^+$ where $j \neq i$ (다른 query 의 positive passage)

**Loss** (in-batch):
$$
L_i = -\log \frac{\exp(s(q_i, p_i^+) / \tau)}{\exp(s(q_i, p_i^+) / \tau) + \sum_{j \neq i} \exp(s(q_i, p_j^+) / \tau)}
$$

### 정의 3.3 — Gradient 와 Hard Negative Dominance

Loss $L$ 에 대해 score $s_i$ 의 gradient:

$$
\frac{\partial L}{\partial s_i} = \frac{\partial}{\partial s_i} \left[ -\log \frac{e^{s^+/\tau}}{e^{s^+/\tau} + \sum_j e^{s_j^-/\tau}} \right]
$$

계산하면:
$$
\frac{\partial L}{\partial s^+} = \frac{-1}{\tau} \left( 1 - p(s^+ \text{ is max}) \right)
$$

$$
\frac{\partial L}{\partial s_i^-} = \frac{1}{\tau} p_i = \frac{1}{\tau} \cdot \frac{e^{s_i^-/\tau}}{\text{partition}}
$$

**Hard negative** (high $s_i^-$): $p_i$ 가 크므로 gradient magnitude 가 크다 → loss 를 dominate.

---

## 🔬 정리와 증명

### 정리 3.1 — In-Batch Negatives 의 Effective Batch Size

Batch size $B$ 일 때, in-batch negatives 를 사용하면:

$$
\text{# negative pairs} = B(B-1)
$$

따라서 **effective negative sampling rate** (naive negative sampling 대비):
$$
\frac{B(B-1)}{B} = B - 1 \times
$$

**증명**: 각 query $q_i$ 에 대해 $B-1$ 개의 negative (다른 query 의 positive). 총 $B$ queries → $B \times (B-1)$ pairs $\square$.

### 정리 3.2 — Temperature Annealing 의 효과

Training epoch $t$ 에서 $\tau(t)$ 를 감소시키면:

$$
\mathbb{P}[\text{model learns hard negatives}] = 1 - e^{-\Delta s(t) / \tau(t)}
$$

여기서 $\Delta s(t)$ 는 positive 와 hardest negative 의 점수 차이.

- $\tau \to \infty$ (초기): $\mathbb{P} \to \Delta s / \infty \approx 0$ (무시)
- $\tau \to 0$ (후기): $\mathbb{P} \to 1$ (반드시 학습)

**결과**: Low $\tau$ schedule 은 curriculum learning 역할 $\square$.

### 정리 3.3 — Hard Negative 의 Gradient Magnitude

score $s^-$ (negative) 의 gradient:

$$
\left\| \nabla_{s^-} L \right\| = \frac{1}{\tau} \cdot \frac{e^{s^-/\tau}}{\text{partition}}
$$

High $s^-$ (hard negative) 일 때:
$$
e^{s^-/\tau} \gg e^{s_i^-/\tau} \quad \Rightarrow \quad \left\| \nabla L \right\| \text{ concentrate on hard } s^-
$$

따라서 **hard negatives 만 loss 를 지배** (soft negatives 는 무시) → sampling strategy 변경 필요.

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — InfoNCE Loss 구현

```python
import torch
import torch.nn.functional as F

def infonce_loss(query_embs, passage_embs, temperature=0.07):
    """
    query_embs: [B, D]
    passage_embs: [B, D]
    
    Positive pair: (q_i, p_i)
    Negatives: {p_j for j != i} (in-batch)
    """
    # Normalize embeddings
    query_embs = F.normalize(query_embs, p=2, dim=1)
    passage_embs = F.normalize(passage_embs, p=2, dim=1)
    
    # Compute similarity matrix [B, B]
    # sim[i, j] = query_i · passage_j
    logits = query_embs @ passage_embs.T
    
    # Scaled by temperature
    logits = logits / temperature
    
    # Labels: diagonal (positive pairs)
    labels = torch.arange(len(query_embs), device=query_embs.device)
    
    # Cross-entropy: positive should have class label = i
    loss = F.cross_entropy(logits, labels)
    
    return loss

# Example
B = 32
D = 768
query_embs = torch.randn(B, D)
passage_embs = torch.randn(B, D)

loss = infonce_loss(query_embs, passage_embs, temperature=0.07)
print(f"InfoNCE loss: {loss.item():.4f}")
```

### 실험 2 — In-Batch Negatives 의 효과 시각화

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def infonce_with_analysis(query_embs, passage_embs, temperature=0.07):
    """InfoNCE 와 함께 per-example 분석"""
    B = query_embs.shape[0]
    query_embs = F.normalize(query_embs, p=2, dim=1)
    passage_embs = F.normalize(passage_embs, p=2, dim=1)
    
    logits = query_embs @ passage_embs.T / temperature
    
    # Softmax 확률
    probs = F.softmax(logits, dim=1)
    
    # 각 query 에 대해, positive passage 의 확률
    positive_probs = torch.diag(probs)
    
    # 각 query 에 대해, top-k negative 의 확률
    top_k_negatives = []
    for i in range(B):
        neg_probs = probs[i, :i].tolist() + probs[i, i+1:].tolist()  # exclude positive
        top_k_negatives.append(sorted(neg_probs, reverse=True)[:5])
    
    return positive_probs.detach().cpu().numpy(), np.array(top_k_negatives)

# Simulation
B = 64
D = 768
query_embs = torch.randn(B, D)
passage_embs = torch.randn(B, D)

pos_probs, neg_probs = infonce_with_analysis(query_embs, passage_embs, temperature=0.07)

print(f"Mean positive probability: {pos_probs.mean():.4f}")
print(f"Max negative probability: {neg_probs.max():.4f}")
print(f"Rank-1 recall (pos > all neg): {(pos_probs >= neg_probs[:, 0]).sum() / B:.2%}")
```

### 실험 3 — Temperature 의 효과

```python
temperatures = [0.01, 0.05, 0.07, 0.1, 0.2, 0.5]
losses = []

for tau in temperatures:
    loss = infonce_loss(query_embs, passage_embs, temperature=tau)
    losses.append(loss.item())
    print(f"τ={tau}: loss={loss.item():.4f}")

# Visualization
plt.figure(figsize=(8, 5))
plt.plot(temperatures, losses, marker='o', linewidth=2)
plt.xlabel('Temperature τ')
plt.ylabel('InfoNCE Loss')
plt.xscale('log')
plt.grid(True)
plt.title('Temperature vs Loss')
plt.savefig('/tmp/temperature_loss.png')
print("Saved to /tmp/temperature_loss.png")
```

### 실험 4 — Hard Negative Dominance (Gradient Analysis)

```python
import torch
import torch.nn.functional as F

def compute_gradient_magnitude(query_embs, passage_embs, temperature=0.07):
    """
    각 passage (negative) 에 대한 gradient magnitude 계산
    """
    query_embs = F.normalize(query_embs, p=2, dim=1).requires_grad_(True)
    passage_embs = F.normalize(passage_embs, p=2, dim=1).requires_grad_(True)
    
    logits = query_embs @ passage_embs.T / temperature
    labels = torch.arange(len(query_embs), device=query_embs.device)
    loss = F.cross_entropy(logits, labels)
    
    loss.backward()
    
    # passage gradient magnitude
    grad_magnitude = passage_embs.grad.norm(dim=1)
    
    return grad_magnitude.detach()

B = 32
D = 768
query_embs = torch.randn(B, D)
passage_embs = torch.randn(B, D)

grad_mag = compute_gradient_magnitude(query_embs, passage_embs, temperature=0.07)

# Top passages by gradient magnitude (hard negatives)
top_k = 5
top_indices = torch.topk(grad_mag, k=top_k).indices
print(f"Top {top_k} hard negatives (by gradient): {top_indices.tolist()}")

# Visualization
plt.figure(figsize=(10, 5))
plt.bar(range(B), grad_mag.numpy())
plt.xlabel('Passage Index')
plt.ylabel('Gradient Magnitude')
plt.title('Hard Negative Dominance (higher = harder)')
plt.savefig('/tmp/hard_negatives.png')
print("Saved to /tmp/hard_negatives.png")
```

---

## 🔗 실전 활용

| 상황 | 추천 전략 | 이유 |
|-----|---------|------|
| Large batch (B=512) | In-batch negatives | $B(B-1) \approx 260K$ pairs 자동으로 학습 |
| Small batch (B=8) | Hard negative mining | In-batch negatives 부족 (28 pairs) |
| Early training | High τ (e.g., 0.5) | Soft objective, global structure |
| Late training | Low τ (e.g., 0.05) | Hard negatives 에 집중 |
| Fine-tuning | Mixed strategy | 초기 high τ → 점진적 낮춤 (annealing) |
| Domain shift | Reset negatives | In-batch negatives 가 domain-specific 아님 |

---

## ⚖️ 가정과 한계

- **In-batch negatives 는 "false negatives"일 수 있다** — 다른 query 의 positive passage 가 현재 query 와도 relevant 할 수 있음. 이는 upper bound on loss (stronger training signal) 이지만 정확도 저하 가능.

- **Temperature scheduling 은 hyperparameter** — Fixed $\tau$ vs annealing 의 최적값은 dataset/model-dependent. 권장: $\tau \in [0.05, 0.1]$ 범위.

- **Hard negative dominance 는 샘플링 편향 신호** — Gradient 가 hard negatives 에 집중되면, curriculum 을 계속 진행하거나 mining-based 방법으로 전환해야 함 (Ch4).

- **In-batch negatives 는 distributed training 에 영향** — Multi-GPU 시, 각 GPU 의 batch 만 negatives 로 사용하면 effective negative size 줄어듦. 이를 보정하려면 gradient synchronization 필요.

---

## 📌 핵심 정리

$$
\boxed{L_{\text{InfoNCE}} = -\log \frac{e^{s(q,p^+)/\tau}}{e^{s(q,p^+)/\tau} + \sum_j e^{s(q,p_j^-)/\tau}}}
$$

| 요소 | 효과 |
|-----|------|
| In-batch negatives | $B(B-1)$ negative pairs (free, batch-level parallelism) |
| Temperature τ | Low = hard focus (후기), High = soft (초기) |
| Hard negatives | Gradient dominate (sampling strategy 변경 신호) |
| Cross-entropy | Probabilistic positive vs negative classification |

> **핵심**: InfoNCE + in-batch negatives + temperature annealing = Dense retrieval 의 **standard training recipe**.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: Batch size 32, temperature 0.07 에서 학습 중. 한 query 의 positive passage 가 다른 query 의 positive 일 확률은?

<details>
<summary>해설</summary>

이것이 "false negative" 문제. 이상적으로는 다른 query 의 positive 가 현재 query 와 무관하지만, 실제로는:
- Overlapping topics (QA dataset): ~10-30% false negative rate
- Diverse topics (web corpus): ~1-5% false negative rate

DPR 논문에서는 이를 무시하고 in-batch negatives 로 충분히 학습됨을 보였음.
</details>

**문제 2 (심화)**: Hard negative 의 gradient magnitude 가 soft negative 의 100배라면, 왜 soft negatives 도 중요한가? (즉, gradient 만으로는 충분하지 않은 이유)

<details>
<summary>해설</summary>

두 가지 이유:
1. **Regularization effect**: Soft negatives 도 작은 gradient 기여 → embedding space 의 "boundary" 를 smooth 하게 유지. Hard negatives 만 학습하면 overfitting (false negative 에 대한 penalty 과다).
2. **Early stage learning**: Training 초반에는 hard negatives 자체를 정의하기 어려움 (모든 passage 가 비슷한 점수). 점진적으로 curriculum (soft → hard) 로 진행.

따라서 temperature annealing: 초기 high τ (모든 negative 균등) → 후기 low τ (hard 집중).
</details>

**문제 3 (논문 비평)**: Contriever (Izacard 2021) 는 in-batch negatives 대신 "다른 배치의 passage 를 negative 로 쓰지 말 것 (false negatives 제거)" 를 제안했다. 이것이 성능을 얼마나 향상시킬까?

<details>
<summary>해설</summary>

Contriever 의 ablation: in-batch negatives (원본 DPR 스타일) vs query-passage-only negatives (false negative 제거).

결과: ~1-2% MRR 향상 (NQ 에서 ~34% → 35.5%). 크지 않은 개선이지만:
- False negative rate 가 높은 domain (overlapping QA) 에서는 더 큼 (3-5%)
- Large batch (B=512) 에서는 무시할 수준 (in-batch negatives 충분)
- Small batch (B=32) 에서 의미 있음

Contriever 는 추가로 unsupervised (random cropping, MoCo) 로 학습하므로 false negative 제거의 효과 분석 어려움.
</details>

---

<div align="center">

[◀ 이전 (02. DPR Bi-Encoder)](./02-dpr-bi-encoder.md) · [📚 README](../README.md) · [다음 ▶ (04. Hard Negative Mining)](./04-hard-negatives.md)

</div>
