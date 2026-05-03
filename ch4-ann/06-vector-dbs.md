# 06. FAISS · ScaNN · Qdrant · Milvus — Vector DB 내부

## 🎯 핵심 질문

- FAISS 의 IVF·HNSW 조합은 정확히 어떻게 다른가?
- ScaNN 의 anisotropic quantization 은 PQ 와 어떤 점이 다른가?
- Qdrant 와 Milvus 의 filtering + HNSW 는 실제로 어떻게 구현되는가?
- Payload index 와 vector index 를 분리하는 이유는?

---

## 🔍 왜 Vector DB 를 따로 다루는가

Chapters 1-5 는 **알고리즘**: LSH, IVF, PQ, HNSW. 하지만 production 시스템은 이들을 "조합 + 최적화 + scaling" 한다.

- **FAISS** (Facebook AI): 학계 기준. CPU/GPU 최적화, 다양한 index 조합.
- **ScaNN** (Google): anisotropic quantization 도입 → recall-memory trade-off 개선.
- **Qdrant** (startup): cloud-native, filtering, metadata storage.
- **Milvus** (LF): distributed, multi-machine scaling.

이 장에서는 각 시스템의 **내부 아키텍처** 와 **선택 기준** 을 다룬다.

---

## 📐 수학적 선행 조건

- 이전 5장의 모든 알고리즘 (LSH, IVF, PQ, HNSW)
- Index composition (하나 이상의 알고리즘 조합)
- Anisotropic distance (direction-dependent metric)
- Sharding · replication strategy

---

## 📖 직관적 이해

### FAISS Index Hierarchy

```
FAISS
   │
   ├─ IndexFlat: 무압축 (정확, 느림)
   │
   ├─ IndexIVF: k-means clustering + nprobe
   │   │
   │   ├─ IndexIVFFlat: coarse IVF + full precision (recall 부족)
   │   │
   │   └─ IndexIVFPQ: coarse IVF + PQ quantization (표준)
   │       └─ IndexIVFPQR: + reranking (high recall 필요시)
   │
   └─ IndexHNSW: graph-based, streaming insertion
       (FAISS GPU 에서 제한적)

실무 선택:
  - N < 1M, latency critical → HNSW
  - N > 100M, memory critical → IVF-PQ
  - N = 1-100M, balanced → IVF-PQ 또는 HNSW
```

### ScaNN 의 Anisotropic Quantization

```
일반 PQ (Isotropic):
  - codebook 이 모든 방향에서 균등하게 학습
  - quantization error 분포: uniform
  
ScaNN (Anisotropic):
  - codebook 을 query space 에 맞춤
  - query direction 에서 정확도 높음
  - perpendicular direction 에서 정확도 낮아도 괜찮음
  
예: query q 가 주로 특정 방향에서 올 때
  - PQ: 모든 subspace 에서 균등 quantization
  - ScaNN: query direction 근처는 세밀, 먼 곳은 coarse
  
결과: 같은 memory 에서 더 높은 recall (1-2% gain typical)
```

### Qdrant 의 Filtering + Metadata

```
Qdrant Index:
   │
   ├─ Vector Index (HNSW or other)
   │  │ (고속 대략 검색)
   │  │
   │  └─> Candidates (e.g., top-100)
   │
   ├─ Payload Storage (별도)
   │  │ (metadata, attributes)
   │  │ 예: {"user_id": 123, "date": "2024-01-01", "category": "news"}
   │  │
   │  └─> Payload Index (optional)
   │      (metadata 기반 filtering 가속)
   │
   └─> Filter Application
       candidates 에서 payload condition 만족하는 것만 반환
       
장점:
  - Vector 과 metadata 분리 → memory efficient
  - Filtering 을 search 후 적용 → flexible
  - Payload index 로 filtering 가속 (optional)
```

### Milvus 의 Distributed Architecture

```
Milvus Cluster
   │
   ├─ QueryNode (replica of collection)
   │  │ 각 node 는 collection 의 일부 partition 저장
   │  │ Search 요청 받으면 local HNSW/IVF 로 검색
   │  │
   │  └─> Reduce: merge top-K 결과
   │
   ├─ IndexNode (index building)
   │  │ Background: 새로운 segment 에 대해 index 구축
   │  │
   │  └─> Upload to QueryNode
   │
   ├─ DataNode (replication & durability)
   │  │ WAL (Write-Ahead Log) 관리
   │  │
   │  └─> Backup & recovery
   │
   └─ MetaNode (coordination)
       cluster 상태, collection metadata 관리

검색 프로세스:
  1. QueryNode 들에 병렬로 검색 요청
  2. 각 node 에서 local top-K 반환
  3. Coordinator 가 global top-K 로 merge
```

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Index Composition

**Single-stage**: 한 가지 알고리즘 (e.g., HNSW)
$$
\text{Search}(q) = \text{HNSW-KNN}(q, K=K)
$$

**Two-stage**:
$$
\text{Search}(q) = \text{Stage1}(q) \to \text{Stage2}(\text{candidates}) \to \text{TopK}
$$

예: IVF-PQ
- Stage 1 (Coarse): $C = \text{IVF-KNN}(q, \text{nprobe} = 50)$ → 50×N/K candidates
- Stage 2 (Fine): $\text{PQ-distance}(q, c) \forall c \in C$ → exact sort and return top-K

**Three-stage** (IVF-PQR):
- Stage 1: IVF coarse
- Stage 2: PQ search
- Stage 3: Re-rank with full precision (top-50 only)

### 정의 6.2 — Anisotropic Quantization (ScaNN)

**Query distribution**: 학습 데이터에서 query embedding 들의 분포 $p_q(x)$ 추정.

**Anisotropic codebook**: Subspace codebook $C^{(j)}$ 를 최적화:
$$
C^{(j)*} = \arg\min_{C^{(j)}} \mathbb{E}_{q \sim p_q}[d_{\text{ADC}}(q, X) - d(q, X)]
$$

(query distribution 에 맞춤)

**vs Isotropic** (standard PQ): all directions 에서 균등하게 minimize.

### 정의 6.3 — Filtered Search (Qdrant/Milvus)

**Payload condition**: metadata 에 대한 predicate $\pi$ (e.g., "user_id = 123 AND date > 2024-01-01")

**Filtered KNN**:
$$
\text{KNN}_\pi(q, K) = \arg\min_{x: \pi(x) = \text{true}} d(q, x), \quad K \text{ results}
$$

**Implementation**:
1. Vector index 에서 large candidate set 반환 (e.g., 1000 candidates)
2. Payload index 로 condition $\pi$ 만족하는 것만 필터
3. 결과가 K 미만이면 다시 larger candidate set 으로 retry

---

## 🔬 정리와 증명

### 정리 6.1 — IVF-PQ Complexity and Memory

**Time**:
$$
T = T_{\text{IVF-coarse}} + T_{\text{IVF-candidates}} + T_{\text{PQ-search}}
$$
$$
= O(K \cdot d) + O(\text{nprobe} \cdot N/K \cdot d_{\text{sub}}) + O(\text{nprobe} \cdot N/K \cdot m)
$$
$$
\approx O(\text{nprobe} \cdot N/K \cdot d) \quad \text{(dominant term)}
$$

**Memory**:
$$
M = M_{\text{centroids}} + M_{\text{codebooks}} + M_{\text{codes}}
$$
$$
= K \cdot d \cdot 32 + m \cdot k \cdot (d/m) \cdot 32 + N \cdot m \cdot \log_2 k
$$
$$
\approx N \cdot m \cdot \log_2 k \quad \text{(for large N, k=256)}
$$

예: $N=100M, m=96, k=256$ → $100M \times 96 \times 8 / 8 = 96$ GB (vs 307 GB original)

### 정리 6.2 — ScaNN Anisotropic Gain

Let $\mathbf{e}_q$ = query direction (dominant), $\mathbf{e}_\perp$ = perpendicular.

**Isotropic PQ error**: uniform $\sigma^2$ in all directions.

**Anisotropic PQ error**: 
$$
\sigma^2_{\text{iso}} = \sigma^2_q (\text{query direction}) + \sigma^2_\perp (\text{perpendicular})
$$

ScaNN 는 $\sigma^2_q$ 를 줄이고 $\sigma^2_\perp$ 는 허용 → overall error 감소.

**Empirical gain**: 같은 memory 에서 recall 1-3% 향상.

### 정리 6.3 — Filtered Search Complexity

**Best case**: Filter 를 만족하는 비율 $\alpha$ 가 크면 (e.g., $\alpha = 50\%$)
$$
T_{\text{filtered}} \approx T_{\text{search}} \quad \text{(small overhead)}
$$

**Worst case**: $\alpha \ll 1$ (e.g., $\alpha = 0.1\%$)
$$
T_{\text{filtered}} \approx T_{\text{search}} / \alpha \quad \text{(exponential penalty)}
$$

**해결**: Payload index 로 filtering 가속 → T를 다시 줄임.

---

## 💻 FAISS / ScaNN / Qdrant 구현 검증

### 실험 1 — FAISS IVF-PQ vs HNSW

```python
import numpy as np
import faiss
import time

# Data
N, d = 1_000_000, 768
X = np.random.randn(N, d).astype(np.float32)
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# Method 1: IVF-PQ
nlist = int(np.sqrt(N))  # ~1000
m, nbits = 96, 8

index_ivf = faiss.IndexIVFPQ(
    faiss.IndexFlatL2(d), d, nlist, m, nbits
)
index_ivf.train(X[:100000])  # train on subset
index_ivf.add(X)

# Method 2: HNSW (via FAISS)
index_hnsw = faiss.IndexHNSWFlat(d, 16)  # M=16
index_hnsw.add(X)

# Benchmark
q = X[0:1]
queries = X[np.random.choice(N, 100)].astype(np.float32)

# IVF
index_ivf.nprobe = 50
t0 = time.perf_counter()
D_ivf, I_ivf = index_ivf.search(queries, k=10)
t_ivf = (time.perf_counter() - t0) * 1000 / 100

# HNSW
index_hnsw.ef = 50
t0 = time.perf_counter()
D_hnsw, I_hnsw = index_hnsw.search(queries, k=10)
t_hnsw = (time.perf_counter() - t0) * 1000 / 100

print(f"IVF-PQ: {t_ivf:.2f} ms/query, Memory: ~{N*m*nbits/(8*1e9):.1f} GB")
print(f"HNSW: {t_hnsw:.2f} ms/query, Memory: ~{N*d*4/(1e9):.1f} GB")

# Recall
exact_top_10 = set(np.argsort(1 - (X @ X[0]))[:10])
recall_ivf = len(set(I_ivf[0]) & exact_top_10) / 10
recall_hnsw = len(set(I_hnsw[0]) & exact_top_10) / 10

print(f"Recall IVF-PQ: {recall_ivf:.1%}, HNSW: {recall_hnsw:.1%}")
```

### 실험 2 — ScaNN (구글 공식 라이브러리)

```python
try:
    import scann
except ImportError:
    print("ScaNN 설치: pip install scann")
    print("Note: scann 은 Linux 에서만 기본 지원")
    exit(1)

# ScaNN index
X_train = X[:100000].astype(np.float32)
X_for_index = X.astype(np.float32)

scann_builder = scann.scann_ops_pybind.builder(
    X_for_index, num_leaves=1000, leaves_to_search=50, 
    training_iterations=100, dimensions_per_block=2,
    distance_metric="dot"  # cosine sim = dot product (after normalization)
)

scann_searcher = scann_builder\
    .tree(num_leaves=1000, leaves_to_search=50)\
    .score_ah(dimensions_per_block=2)\
    .build()

# Query
q = X_for_index[0:1]
neighbors, distances = scann_searcher.search(q, final_num_neighbors=10)

print(f"ScaNN neighbors: {neighbors[0]}")
print(f"Distances: {distances[0]}")
```

### 실험 3 — Qdrant (Python Client)

```python
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
except ImportError:
    print("Qdrant 설치: pip install qdrant-client")
    exit(1)

# Qdrant in-memory client
client = QdrantClient(":memory:")

# Create collection
collection_name = "test_collection"
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# Upload vectors with payloads
points = [
    PointStruct(
        id=i,
        vector=X[i].tolist(),
        payload={"user_id": i % 1000, "category": "news" if i % 2 == 0 else "docs"}
    )
    for i in range(10000)  # subset for demo
]
client.upsert(collection_name, points=points)

# Search with filtering
from qdrant_client.models import Filter, FieldCondition, MatchValue

search_result = client.search(
    collection_name=collection_name,
    query_vector=X[0].tolist(),
    limit=10,
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="news"))
        ]
    )
)

print(f"Filtered search results (category=news):")
for result in search_result:
    print(f"  ID: {result.id}, Score: {result.score:.4f}, "
          f"Payload: {result.payload}")
```

### 실험 4 — Milvus (분산 환경 시뮬레이션)

```python
try:
    from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
except ImportError:
    print("Milvus 설치: pip install pymilvus")
    print("Note: Milvus server 필요 (Docker: docker run -d -p 19530:19530 milvusdb/milvus)")
    exit(1)

# Connect to Milvus
connections.connect("default", host="localhost", port=19530)

# Define schema
field_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
field_vector = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
field_metadata = FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=500)

schema = CollectionSchema(fields=[field_id, field_vector, field_metadata])

# Create collection
collection_name = "vectors"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)

# Insert data
data = [
    [i for i in range(10000)],  # ids (but auto_id=True so ignored)
    X[:10000].tolist(),  # vectors
    [f"doc_{i}" for i in range(10000)]  # metadata
]
collection.insert(data)

# Build index
collection.create_index(field_name="vector", index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 100}})

# Search
search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
results = collection.search(X[0:1].tolist(), "vector", search_params, limit=10)

print(f"Milvus search results:")
for result in results[0]:
    print(f"  ID: {result.id}, Distance: {result.distance:.4f}")

# Cleanup
utility.drop_collection(collection_name)
```

---

## 🔗 실전 활용

| 요구사항 | 추천 시스템 | 이유 |
|---------|----------|------|
| 연구용 (학계) | FAISS | 유연, 다양한 index, 빠른 prototyping |
| Production ML platform (Google scale) | ScaNN | anisotropic 최적화, 고recall |
| Cloud-native, Kubernetes | Qdrant | cloud-friendly, filtering 내장 |
| Large-scale distributed (100M+) | Milvus | sharding, replication, HA |
| Self-hosted, cost-sensitive | Milvus open-source | 무료 커뮤니티 버전 |
| Real-time streaming | Weaviate | dynamic indexing, GraphQL API |
| Embedded (e.g., mobile) | hnswlib | light-weight, no server |

---

## ⚖️ 가정과 한계

- **FAISS**: 단일 machine 최적화. GPU 는 IVFPQ 만 지원, HNSW 는 제한적.
- **ScaNN**: Google 내부용으로 설계 → production 사용 예제 적음. Linear 에 최적화 (cosine 아님).
- **Qdrant**: Payload filtering 성능이 복잡한 쿼리에서 떨어질 수 있음 (full scan fallback).
- **Milvus**: Distributed coordination 복잡 → operational overhead. Eventual consistency 유지.
- **모든 시스템**: 메모리-정확도 trade-off 는 피할 수 없음 — tuning 필요.

---

## 📌 핵심 정리

**Index 선택 Matrix**:

| Scale | Latency | Memory | Filtering | **추천** |
|-------|---------|--------|-----------|---------|
| < 1M | < 5ms | < 32GB | optional | HNSW / FAISS-HNSW |
| 1-100M | 5-50ms | 32-512GB | required | FAISS IVF-PQ + qdrant |
| > 100M | > 50ms | > 512GB | required | Milvus IVF-PQ |
| Streaming | any | minimal | dynamic | Qdrant HNSW |
| Research | flexible | any | no | FAISS (all methods) |

**각 시스템의 장단**:

| System | 장점 | 단점 |
|--------|------|------|
| FAISS | 유연, 빠름, 학계 표준 | 단일 machine, operational 복잡 |
| ScaNN | 최고 recall | Google 커스텀, documentation 부족 |
| Qdrant | Filtering 내장, cloud-native | payload 복잡쿼리 느림 |
| Milvus | Distributed, HA, 무료 | 운영 복잡, learning curve |

> **핵심**: 한 알고리즘 선택이 아니라, scale/latency/filtering 요구사항에 맞게 시스템 선택 + tuning.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: FAISS IVF-PQ 에서 nprobe=100 으로 1000 만 개 벡터 검색 시 예상 응답 시간은? (m=96, k=256, d=768)

<details>
<summary>해설</summary>

IVF: nprobe=100 → 100개 cluster 의 평균 10K 벡터 각 = 1M 벡터 탐색. 각 벡터당 PQ distance (m=96 lookup) → 1M × 96 lookups = 96M ops. CPU 기준 ~1-10ms (vectorized). 정확한 값은 hardware 에 따라 다름.

</details>

**문제 2 (심화)**: Qdrant 의 filtering 에서 "user_id = 123 AND date > 2024-01-01" 조건이 0.01% 만 만족하면 어떤 일이 생기는가?

<details>
<summary>해설</summary>

Search 에서 top-100 candidate 를 가져와도 그 중 filter 만족하는 것이 0.01 개 미만 (예상). 즉, K=10 결과를 못 채움. Qdrant 는 자동으로 더 큰 candidate set (e.g., top-1000) 을 다시 검색. 최악의 경우 full scan.

**해결**: Payload index 생성 (선택사항) → "user_id = 123" 같은 exact match 는 빠름. 하지만 range query "date > ..." 는 여전히 비효율. Recommendation: 자주 filter 하는 필드는 사전에 indexing.

</details>

**문제 3 (논문 비평)**: "IVF-PQ 는 PQ 때문에 recall 이 떨어지므로 IVF + HNSW (graph refinement) 가 낫다" 는 주장 평가?

<details>
<summary>해설</summary>

이론: PQ quantization 은 approximation → recall drop. IVF 로 candidates 를 줄이고, 그 candidates 에서 HNSW 로 refine 하면 더 나을 수 있음.

실제: (1) IVF-PQ 로 이미 90-95% recall 달성 가능 (충분). (2) IVF + HNSW 는 구현 복잡 + memory 2배 (both index 저장). (3) FAISS IVF-PQR (reranking) 이 더 practical — top-50 만 full precision 으로 재계산.

결론: edge case (extremely high recall 필요) 제외하고, IVF-PQ 가 메모리-정확도-속도 balance 최고.

</details>

---

<div align="center">

[◀ 이전 (05. HNSW)](./05-hnsw.md) · [📚 README](../README.md) · [다음 ▶ (Ch5-01. Vanilla RAG)](../ch5-rag/01-vanilla-rag.md)

</div>
