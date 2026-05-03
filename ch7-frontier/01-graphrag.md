# 01. GraphRAG (Edge 2024)

## 🎯 핵심 질문

- Corpus 에서 entity 와 relationship 을 추출하여 knowledge graph $G = (V, E)$ 로 변환하면 retrieval 이 어떻게 달라지는가?
- Community detection (Leiden algorithm) 을 통한 hierarchical summarization 이 왜 "global question" 의 답변 품질을 개선하는가?
- 기존 vector RAG 와의 차이: dense retrieval 은 쿼리-문서 similarity 에만 의존하지만, GraphRAG 는 어떤 새로운 정보 흐름을 활용하는가?
- Microsoft 의 GraphRAG 구현에서 Leiden modularity 최적화의 수학적 정당화는?

---

## 🔍 왜 GraphRAG 가 Retrieval 의 frontier 인가

기존 dense vector RAG (Ch2, Ch3) 의 한계:
- **Local coherence only** — 문서의 특정 passage 와 쿼리의 유사도만 측정 → "전체 도메인의 흐름이나 구조 파악" 불가능.
- **Cold start 문제** — 코퍼스 내 entity 간 의존성을 명시적으로 학습하지 않음.
- **Aggregation quality** — 여러 passage 를 종합할 때 명시적 context fusion 없이 simple concatenation.

GraphRAG 의 도전:
- **Structured extraction** — corpus → LLM entity/relation extraction → graph construction
- **Community-driven summarization** — modularity 기반 clustering → each community 별 hierarchical summary
- **Dual-mode retrieval** — local question (entity 기반) vs global question (community summary 기반)

이 frontier 는 "**knowledge structure 를 명시적으로 build 하면 complex question 해결 성능이 올라간다**" 는 가설을 실증.

---

## 📐 수학적 선행 조건

- Graph theory 기초: degree, modularity, connected components
- Spectral clustering, Louvain/Leiden algorithm 개요
- LLM 기반 정보 추출의 recall/precision trade-off
- KG embedding 기초 (TransE, 등은 선택)

---

## 📖 직관적 이해

### RAG 의 진화: Vector → Graph

```
(1) Dense Vector RAG
    Query ──┐
            ├──→ Embedding space ──→ Top-K similar passage ──→ LLM answer
    Corpus ─┘     (cosine similarity)

(2) GraphRAG
    Query ──────────┐
                    ├──→ Knowledge Graph
                    │       (E.g., "Apple" ── CEO:Tim Cook ── headquarter:Cupertino)
                    │
    Corpus ─────────┤    ├──→ [Local Retrieval] 
                    │    │    entity-based QA
                    │    │
                    │    └──→ [Global Retrieval]
                    │         community summary aggregation
                    │
                    └──────→ LLM(local + global context) ──→ answer
```

### Community Detection (Modularity 최적화)

```
Knowledge Graph G = (V, E)
    │
    ├─ Nodes (entities): Apple, Steve Jobs, iPhone, iOS, ...
    │
    ├─ Edges (relations): [founder], [product], [platform], ...
    │
    └─→ Leiden Algorithm
        Find partition C = {C1, C2, ...} maximizing modularity:
        Q = Σ_{c} [edges_within_c / total_edges - (degree_c / total_degree)²]
        
        → Each community Ci: mini knowledge graph 를 represent
        → Summarize each community (LLM): "업계 리더 Apple 과 그 제품 생태계"
        → Higher-level aggregation (communities of communities)
```

---

## ✏️ 엄밀한 정의

### 정의 7.1 — Knowledge Graph Construction

**Input**: Corpus $\mathcal{D} = \{d_1, \ldots, d_n\}$ (documents).

**Process**:
1. **Entity & Relation Extraction**: LLM (e.g., ChatGPT) processes each document with prompt:
   ```
   "Extract all entities and their relationships. Format: (entity1, relation, entity2)"
   ```
   → Output: triples $\{(e_1, r, e_2), \ldots\}$ per document.

2. **Graph Merge**: 
   $$G = (V, E) \text{ where } V = \bigcup_{\text{docs}} \text{entities}, \quad E = \bigcup_{\text{docs}} \text{edges}$$

3. **Deduplication**: String similarity (e.g., LevenshteinDistance < 0.8) → merge nodes.

**Output**: Knowledge graph $G$ with $|V|$ nodes, $|E|$ edges.

### 정의 7.2 — Leiden Community Detection

**Modularity** of partition $C = \{C_1, \ldots, C_k\}$:
$$
Q(C) = \sum_{i=1}^{k} \left[ \frac{m_i}{m} - \left(\frac{\sum_{v \in C_i} d_v}{2m}\right)^2 \right]
$$

where:
- $m_i$ = edges within community $C_i$
- $m$ = total edges
- $d_v$ = degree of node $v$

**Leiden Algorithm** (Traag et al. 2019):
- Iteratively moves nodes to maximize local modularity (faster than Louvain, allows refinement pass).
- Output: hierarchical partition tree $T$ (can be multi-level).

### 정의 7.3 — Hierarchical Summarization

For each community $C_i$ at level $\ell$:

$$
\text{Summary}(C_i) = \text{LLM}_{\text{summarize}}\left( \text{entities}(C_i), \text{relations}(C_i), \text{source docs} \right)
$$

**Two-mode retrieval**:
- **Local**: $Q_{\text{local}}$ → retrieve entities + incident edges → LLM answer.
- **Global**: $Q_{\text{global}}$ → retrieve summaries of relevant communities → LLM answer.

---

## 🔬 정리와 증명

### 정리 7.1 — Leiden Modularity 수렴성

**정리**: Leiden algorithm 은 한정된 iteration 내에 **local maximum modularity** 에 도달한다.

**증명 sketch**: 
1. Each node's move 를 "greedy locally optimal" 하게 함.
2. Community refinement pass 로 "ghost edges" 제거.
3. Modularity $Q(C^{(t)})$ 는 monotone increasing, bounded above by 1 → convergence.

$\square$

### 정리 7.2 — Community Size 와 Summary Quality 의 Trade-off

**정리**: 너무 fine-grained community (많은 작은 cluster) 는 LLM 요약의 context 부족, 너무 coarse (큰 cluster) 는 정보 압축 손실.

**증명 sketch**:
- Let $|C_i|$ = size of community $i$, $\ell_i$ = average path length within $C_i$.
- Summary quality $\propto \ell_i$ (longer paths = more structure to explain).
- LLM context limit $\approx 8K$ tokens → optimal $|C_i| \in [50, 500]$ empirically.

$\square$

### 정리 7.3 — Global Aggregation vs Local Retrieval

**정리** (MS GraphRAG paper): Global question 에 대해, global retrieval (community summary aggregation) 은 local retrieval (passage-level) 보다 **높은 coverage** 를 달성.

**증명 sketch**:
- Global Q: "Tell me the key innovations of the company over time" (requires long-range entity context).
- Local retrieval: top-K passage 에 답이 흩어져 있을 가능성 high (poor aggregation).
- Global: community summarization 은 temporal/causal structure 를 명시적으로 encode → better synthesis.

$\square$

---

## 💻 Python / PyTorch / FAISS 구현 검증

### 실험 1 — Entity & Relation 추출 (LLM-based)

```python
import json
from openai import OpenAI

def extract_entities_relations(text: str, model="gpt-4o-mini"):
    """
    Extract entities and relations from text using LLM.
    Returns: list of (entity1, relation, entity2) triples.
    """
    client = OpenAI(api_key="YOUR_KEY")  # or use OPENAI_API_KEY env var
    
    prompt = f"""
Extract all entities and their relationships from the text.
Format: JSON list of {{"entity1": "...", "relation": "...", "entity2": "..."}}

Text:
{text}
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    try:
        triples = json.loads(response.choices[0].message.content)
        return triples
    except:
        return []

# Example: Wikipedia snippet about Apple
text = """
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
The company is headquartered in Cupertino, California. Tim Cook is the CEO.
Apple produces the iPhone, iPad, and Mac computers.
"""

triples = extract_entities_relations(text)
print(json.dumps(triples, indent=2))
# [{"entity1": "Apple Inc.", "relation": "founded_by", "entity2": "Steve Jobs"}, ...]
```

### 실험 2 — Knowledge Graph Construction & Leiden Community Detection

```python
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import json

def build_kg(triples: list) -> nx.Graph:
    """Build knowledge graph from triples."""
    G = nx.Graph()
    for triple in triples:
        e1, rel, e2 = triple['entity1'], triple['relation'], triple['entity2']
        G.add_edge(e1, e2, relation=rel)
    return G

def leiden_clustering(G: nx.Graph):
    """
    Use python-louvain (Louvain, faster approx of Leiden) for community detection.
    For exact Leiden, use leiden package (install: pip install leiden).
    """
    try:
        from leiden import leiden_find_partition
        import leidenalg
        
        # Convert networkx to igraph
        import igraph as ig
        edge_list = list(G.edges())
        G_ig = ig.Graph()
        G_ig.add_vertices(len(G.nodes()))
        G_ig.vs["name"] = list(G.nodes())
        G_ig.add_edges(edge_list)
        
        partition = leiden_find_partition(G_ig, leidenalg.ModularityOptimizer())
        return partition
    except ImportError:
        # Fallback: use greedy modularity (networkx built-in)
        communities = list(greedy_modularity_communities(G))
        return {node: i for i, comm in enumerate(communities) for node in comm}

# Build sample KG
triples = [
    {"entity1": "Apple Inc.", "relation": "founded_by", "entity2": "Steve Jobs"},
    {"entity1": "Steve Jobs", "relation": "born_in", "entity2": "San Francisco"},
    {"entity1": "Apple Inc.", "relation": "headquarters", "entity2": "Cupertino"},
    {"entity1": "Apple Inc.", "relation": "product", "entity2": "iPhone"},
    {"entity1": "iPhone", "relation": "released", "entity2": "2007"},
]

G = build_kg(triples)
print(f"KG: {len(G.nodes())} nodes, {len(G.edges())} edges")

communities = leiden_clustering(G)
print(f"Communities: {communities}")
```

### 실험 3 — Hierarchical Summarization (LLM)

```python
def summarize_community(G: nx.Graph, community: list, model="gpt-4o-mini"):
    """
    Generate a natural language summary of a community.
    """
    # Extract subgraph
    subG = G.subgraph(community)
    edges_info = []
    for e1, e2, data in subG.edges(data=True):
        rel = data.get('relation', 'related_to')
        edges_info.append(f"({e1}) --[{rel}]--> ({e2})")
    
    entities_str = ", ".join(community)
    edges_str = "\n".join(edges_info)
    
    prompt = f"""
Summarize the following knowledge sub-graph in 1-2 sentences:
Entities: {entities_str}
Relations:
{edges_str}

Summary:
"""
    
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    return response.choices[0].message.content

# Example
comm1 = ["Apple Inc.", "Steve Jobs", "San Francisco"]
summary = summarize_community(G, comm1)
print(f"Community summary: {summary}")
# "Steve Jobs founded Apple Inc. in San Francisco."
```

### 실험 4 — Dual-mode Retrieval (Local vs Global)

```python
def local_retrieval(G: nx.Graph, query: str, top_k=5):
    """
    Local retrieval: find entities matching query, return incident edges.
    """
    query_lower = query.lower()
    matching_nodes = [n for n in G.nodes() if query_lower in n.lower()]
    
    edges_info = []
    for node in matching_nodes:
        for neighbor in G.neighbors(node):
            rel = G[node][neighbor].get('relation', 'related_to')
            edges_info.append(f"({node}) --[{rel}]--> ({neighbor})")
    
    return edges_info[:top_k]

def global_retrieval(G: nx.Graph, communities: dict, query: str, top_k=3):
    """
    Global retrieval: return community summaries most relevant to query.
    (Simplified: just return largest communities)
    """
    comm_sizes = {}
    for comm_id, nodes in enumerate(set(communities.values())):
        comm_nodes = [n for n, c in communities.items() if c == comm_id]
        comm_sizes[comm_id] = len(comm_nodes)
    
    top_comms = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [f"Community {cid} (size {size})" for cid, size in top_comms]

# Test
query = "Apple"
print("Local:", local_retrieval(G, query))
print("Global:", global_retrieval(G, communities, query))
```

---

## 🔗 실전 활용

| 상황 | GraphRAG 적용 | 기존 Vector RAG 대비 |
|------|---------------|-------------------|
| **Structured QA** (entity-centric: "CEO 누구?") | Local retrieval 충분 | 유사 또는 약간 우월 |
| **Synthesis Query** ("회사 역사 요약") | Global + Local 조합 필수 | **GraphRAG 1.5~2.0배 우월** |
| **Cross-domain reasoning** ("다양한 회사 비교") | Community aggregation 활용 | **GraphRAG 명확히 우월** |
| **Cold-start corpus** | Entity/relation 추출 cost 높음 | Dense embeddings 빠름 |
| **Korean text** | LLM extraction quality 문제 가능 | 한국어 embedding model 강함 |

---

## ⚖️ 가정과 한계

- **LLM extraction quality**: Entity/relation 추출의 recall/precision 의존 — hallucination 가능.
- **Scalability**: $|V| \times |E|$ graph 에서 Leiden 은 $O(E \log E)$ but large corpus (>1M docs) 에서 bottleneck.
- **Language coverage**: MS 구현은 주로 English — Korean 코퍼스에서는 extraction accuracy 검증 필요.
- **Community interpretation**: Community 가 의미 있는 semantic cluster 임을 보장 못함.
- **Summary staleness**: 정적 KG summary → corpus update 시 재계산 필요 (비용 높음).

---

## 📌 핵심 정리

$$
\boxed{\text{GraphRAG} = \text{Entity/Relation Extraction} + \text{Leiden Community Detection} + \text{Hierarchical Summarization}}
$$

| 단계 | 담당 | 수학 |
|------|------|------|
| Extraction | LLM | Prompt engineering (no closed form) |
| Community | Leiden algorithm | Modularity maximization (NP-hard, greedy approx) |
| Summarization | LLM | Context-aware abstractive summarization |
| Retrieval | Hybrid (local + global) | Entity matching + community relevance ranking |

> **핵심**: Vector RAG 는 passage-level similarity 에만 의존하지만, GraphRAG 는 **entity-relation structure 를 명시적으로 build** 하여 global question 에 강점.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**: Entity extraction 에서 "Apple (회사)" vs "Apple (과일)" 의 disambiguation 을 어떻게 처리할까?

<details>
<summary>해설</summary>

(1) Context window — 같은 문서 내에서 disambiguation hint (e.g., "Apple Inc. 는 기술회사" vs "apple 은 과일").

(2) External knowledge — WIKIDATA, DBpedia 같은 KG 와 링크 후 중복 제거.

(3) LLM prompt engineering — "Extract company entities only" 같은 제약.

실제 MS GraphRAG 는 (1)+(3) 조합; LLM 기반 disambiguator 도 고려 중.

</details>

**문제 2 (심화)**: Leiden modularity 최적화가 왜 NP-hard 이고, greedy approximation 의 ratio bound 는?

<details>
<summary>해설</summary>

최대 modularity 를 갖는 partition 찾기는 NP-complete (Brandes et al. 2008). Leiden (및 Louvain) 은 greedy 로 $O(E \log E)$ 에 local maximum 도달.

Approximation ratio: 최적 vs greedy 의 비율은 graph structure 에 따라 다름. 실증적으로 대부분 95~99% 근처.

</details>

**문제 3 (논문 비평)**: "GraphRAG 는 vector RAG 을 완전히 대체할 수 있는가?" 라는 주장을 평가.

<details>
<summary>해설</summary>

아니오. 이유:

(1) **Cost**: LLM extraction × corpus size + Leiden computation (O(E log E)) >> dense embedding (O(d) per doc).

(2) **Hallucination risk**: Entity/relation 추출 에러 → KG 자체가 잘못됨 → 못 복구.

(3) **Hybrid이 optimal**: Local question → dense retrieval 빠름; global question → graph 우월.

실제 frontier (2024): **GraphRAG + Dense Embedding hybrid** — entity-aware dense retrieval 로 둘의 장점 결합 (e.g., ColBERT with entity anchors).

</details>

---

<div align="center">

[◀ 이전 (Ch6-04. LLM-as-Reranker)](../ch6-reranking-hybrid/04-llm-as-reranker.md) · [📚 README](../README.md) · [다음 ▶ (02. ColPali)](./02-colpali.md)

</div>
