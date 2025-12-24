# Local vs Cloud Vector Storage

## Learning Objectives
- Compare local and cloud deployment models for vector databases
- Understand data locality, latency, and cost trade-offs
- Evaluate when to choose local vs cloud solutions
- Plan migration paths between deployment models

## Why This Matters

Where you deploy your vector database affects everything:
- **Performance**: Network latency vs local disk access
- **Cost**: Self-managed infrastructure vs managed services
- **Scalability**: Vertical limits vs horizontal scaling
- **Operations**: Your maintenance burden vs provider SLA

Understanding these trade-offs helps you make the right decision for your specific use case and avoid costly migrations later.

## The Concept

### Deployment Models Overview

```
LOCAL (Embedded/Self-Hosted)           CLOUD (Managed Service)
----------------------------           ----------------------
Chroma (in-process)                    Pinecone
Chroma (persistent)                    Weaviate Cloud
Qdrant (self-hosted)                   Qdrant Cloud
Milvus (self-hosted)                   Zilliz (managed Milvus)
pgvector (PostgreSQL)                  Supabase (pgvector)
```

### Local Vector Databases

#### Embedded Mode (In-Process)

The database runs within your application process.

```python
import chromadb

# Ephemeral (in-memory only)
client = chromadb.Client()

# Persistent (saved to disk)
client = chromadb.PersistentClient(path="./chroma_data")
```

**Pros**:
- Zero network latency
- No external dependencies
- Simple development setup
- Free (no service costs)

**Cons**:
- Limited by machine resources
- No horizontal scaling
- Data loss risk if not persisted properly
- Single-machine concurrency limits

#### Self-Hosted Server Mode

The database runs as a separate service you manage.

```python
import chromadb

# Connect to self-hosted Chroma server
client = chromadb.HttpClient(
    host="localhost",
    port=8000
)
```

**Pros**:
- Can run on dedicated hardware
- Multiple applications can connect
- More control over configuration
- Still no external service costs

**Cons**:
- You manage infrastructure
- You handle backups and recovery
- Security is your responsibility
- Scaling requires manual intervention

### Cloud Vector Databases

Managed services where the provider handles infrastructure.

```python
# Example: Pinecone
import pinecone

pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
index = pinecone.Index("my-index")

# Example: Weaviate Cloud
import weaviate

client = weaviate.Client(
    url="https://your-cluster.weaviate.network",
    auth_client_secret=weaviate.AuthApiKey(api_key="your-key")
)
```

**Pros**:
- Managed infrastructure (no ops burden)
- Automatic scaling
- Built-in high availability
- Global distribution options
- Professional support

**Cons**:
- Usage-based costs (can be significant)
- Network latency
- Vendor lock-in
- Data leaves your infrastructure
- Less configuration control

### Comparison Table

| Factor | Local (Embedded) | Local (Self-Hosted) | Cloud (Managed) |
|--------|-----------------|-------------------|-----------------|
| **Latency** | Microseconds | Milliseconds | 10-100+ ms |
| **Setup** | Minutes | Hours | Minutes |
| **Scaling** | Vertical only | Manual horizontal | Automatic |
| **Cost** | Compute only | Compute + Ops | Per-query/storage |
| **Ops Burden** | Low | High | None |
| **Data Privacy** | Full control | Full control | Depends on provider |
| **Availability** | App-dependent | Your SLA | Provider SLA |

### Data Locality Considerations

#### Privacy and Compliance

```
LOCAL: Data never leaves your infrastructure
├── Full GDPR compliance control
├── No third-party data processing
├── Audit trail under your control
└── Required for sensitive industries (healthcare, finance)

CLOUD: Data processed by provider
├── Check provider certifications (SOC2, HIPAA, etc.)
├── Data residency options (region selection)
├── Encryption at rest and in transit
└── Review data processing agreements
```

#### Latency Patterns

```
LOCAL EMBEDDED
--------------
App → Vector DB (same process)
Latency: ~100 microseconds

LOCAL SERVER  
------------
App → Network → Vector DB Server → Network → App
Latency: 1-5 milliseconds (local network)

CLOUD
-----
App → Internet → Cloud Provider → Processing → Internet → App
Latency: 20-200 milliseconds (varies by region)
```

### Cost Analysis

#### Local Costs

```python
# Local cost calculation
local_costs = {
    "compute": {
        "small": {"instance": "t3.medium", "monthly": 30},
        "medium": {"instance": "m5.xlarge", "monthly": 150},
        "large": {"instance": "r5.2xlarge", "monthly": 400}
    },
    "storage": {
        "per_gb_month": 0.10  # EBS/SSD
    },
    "ops": {
        "hours_per_week": 2,  # Maintenance, monitoring
        "hourly_rate": 100
    }
}

def calculate_local_monthly_cost(instance_size, storage_gb, include_ops=True):
    cost = local_costs["compute"][instance_size]["monthly"]
    cost += storage_gb * local_costs["storage"]["per_gb_month"]
    if include_ops:
        cost += local_costs["ops"]["hours_per_week"] * 4 * local_costs["ops"]["hourly_rate"]
    return cost

# Example: Medium instance, 100GB storage, with ops
print(calculate_local_monthly_cost("medium", 100, True))  # $150 + $10 + $800 = $960
```

#### Cloud Costs

```python
# Cloud cost calculation (example rates, vary by provider)
cloud_costs = {
    "pinecone": {
        "pod_per_hour": 0.10,    # Standard pod
        "queries_per_million": 2.00,
        "storage_per_gb": 0.025
    },
    "weaviate_cloud": {
        "base_monthly": 25,
        "per_million_objects": 5
    }
}

def calculate_cloud_monthly_cost(vectors_millions, queries_millions, storage_gb):
    # Pinecone example
    pod_hours = 24 * 30  # One pod running 24/7
    pod_cost = pod_hours * cloud_costs["pinecone"]["pod_per_hour"]
    query_cost = queries_millions * cloud_costs["pinecone"]["queries_per_million"]
    storage_cost = storage_gb * cloud_costs["pinecone"]["storage_per_gb"]
    return pod_cost + query_cost + storage_cost

# Example: 1M vectors, 10M queries/month, 50GB storage
print(calculate_cloud_monthly_cost(1, 10, 50))  # $72 + $20 + $1.25 = $93.25
```

### Decision Framework

```
START
  │
  ▼
┌─────────────────────────────┐
│ Data sensitivity high?      │
│ (Healthcare, finance, etc.) │
└─────────────────────────────┘
          │
    Yes ──┼── No
          │
          ▼
   ┌──────────────┐     ┌────────────────────┐
   │ LOCAL ONLY   │     │ Query volume high? │
   │ (compliance) │     │ (>10M queries/mo)  │
   └──────────────┘     └────────────────────┘
                                  │
                         Yes ──┼── No
                                  │
                                  ▼
           ┌────────────────┐     ┌─────────────────┐
           │ CLOUD MANAGED  │     │ Team has ops    │
           │ (scale + SLA)  │     │ experience?     │
           └────────────────┘     └─────────────────┘
                                         │
                                  Yes ──┼── No
                                         │
                                         ▼
                    ┌──────────────┐     ┌───────────────┐
                    │ LOCAL        │     │ CLOUD         │
                    │ SELF-HOSTED  │     │ (lower ops)   │
                    └──────────────┘     └───────────────┘
```

## Code Example

Abstraction layer for seamless local/cloud switching:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class VectorStoreInterface(ABC):
    """Abstract interface for vector store operations."""
    
    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict]
    ):
        pass
    
    @abstractmethod
    def query(
        self,
        embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]):
        pass


class LocalChromaStore(VectorStoreInterface):
    """Local Chroma implementation."""
    
    def __init__(self, path: str = "./chroma_data", collection_name: str = "default"):
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def add(self, ids, embeddings, documents, metadatas):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def query(self, embedding, n_results=5, where=None):
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    
    def delete(self, ids):
        self.collection.delete(ids=ids)


class CloudPineconeStore(VectorStoreInterface):
    """Pinecone cloud implementation."""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        import pinecone
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)
    
    def add(self, ids, embeddings, documents, metadatas):
        # Pinecone requires vectors as list of tuples
        vectors = []
        for id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            meta_with_doc = {**meta, "_document": doc}
            vectors.append((id, emb, meta_with_doc))
        self.index.upsert(vectors=vectors)
    
    def query(self, embedding, n_results=5, where=None):
        results = self.index.query(
            vector=embedding,
            top_k=n_results,
            filter=where,
            include_metadata=True
        )
        # Convert to Chroma-like format for consistency
        return {
            "ids": [[m.id for m in results.matches]],
            "documents": [[m.metadata.get("_document", "") for m in results.matches]],
            "metadatas": [[{k: v for k, v in m.metadata.items() if k != "_document"} 
                          for m in results.matches]],
            "distances": [[m.score for m in results.matches]]
        }
    
    def delete(self, ids):
        self.index.delete(ids=ids)


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create(config: Dict[str, Any]) -> VectorStoreInterface:
        store_type = config.get("type", "local")
        
        if store_type == "local":
            return LocalChromaStore(
                path=config.get("path", "./chroma_data"),
                collection_name=config.get("collection", "default")
            )
        elif store_type == "pinecone":
            return CloudPineconeStore(
                api_key=config["api_key"],
                environment=config["environment"],
                index_name=config["index_name"]
            )
        else:
            raise ValueError(f"Unknown store type: {store_type}")


# Usage - easily switch between local and cloud
config_local = {
    "type": "local",
    "path": "./my_vectors",
    "collection": "documents"
}

config_cloud = {
    "type": "pinecone",
    "api_key": "your-api-key",
    "environment": "us-west1-gcp",
    "index_name": "my-index"
}

# Development: use local
store = VectorStoreFactory.create(config_local)

# Production: switch to cloud by changing config
# store = VectorStoreFactory.create(config_cloud)

# Same API regardless of backend
store.add(
    ids=["doc1"],
    embeddings=[[0.1, 0.2, 0.3]],
    documents=["Hello world"],
    metadatas=[{"source": "test"}]
)

results = store.query([0.1, 0.2, 0.3], n_results=5)
```

## Key Takeaways

1. **Local embedded is best for development** - zero setup, zero cost
2. **Self-hosted gives control but requires ops investment**
3. **Cloud managed is best for production scale** - when budget allows
4. **Data privacy requirements may dictate the choice**
5. **Build abstraction layers** for flexibility to switch later
6. **Cost analysis should include ops time**, not just infrastructure

## Additional Resources

- [Chroma Deployment Guide](https://docs.trychroma.com/deployment) - Local and server deployment options
- [Pinecone Architecture](https://docs.pinecone.io/docs/architecture) - Understanding managed vector DB architecture
- [Vector Database Comparison](https://thenewstack.io/comparing-vector-databases/) - Comprehensive comparison article
