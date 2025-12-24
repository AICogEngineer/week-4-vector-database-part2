# Cloud Deployment Considerations

## Learning Objectives
- Understand security requirements for cloud vector databases
- Plan network architecture for cloud deployments
- Evaluate managed service options and their trade-offs
- Implement production-ready cloud configurations

## Why This Matters

Moving from local development to cloud production involves more than just connecting to a different endpoint. Security, networking, and service selection can make or break your deployment. Getting these right ensures your RAG system is secure, performant, and cost-effective.

## The Concept

### Security Considerations

#### Authentication and Authorization

Cloud vector databases require proper access control:

```python
# API Key Authentication (basic)
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="your-cloud-host",
    port=8000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        chroma_client_auth_credentials="your-api-key"
    )
)

# OAuth/JWT Authentication (enterprise)
# Typically configured at the provider level
```

**Best Practices**:
- Never hardcode API keys in source code
- Use environment variables or secrets managers
- Rotate keys regularly
- Use least-privilege access policies

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Load from environment
api_key = os.getenv("VECTOR_DB_API_KEY")
if not api_key:
    raise ValueError("VECTOR_DB_API_KEY environment variable not set")
```

#### Data Encryption

```
AT REST                          IN TRANSIT
-------                          ----------
Data stored encrypted            TLS 1.2+ for all connections
Provider-managed keys            Certificate validation
Customer-managed keys (CMK)      VPN/Private Link options
Encryption key rotation          mutual TLS (mTLS) for high security
```

```python
# Example: Verifying TLS is used
import ssl
import certifi

# Ensure HTTPS is used
client = chromadb.HttpClient(
    host="https://your-cloud-host",  # Note HTTPS
    port=443,
    ssl=True
)
```

### Networking Architecture

#### Public Endpoint (Simple)

```
Your Application → Internet → Cloud Vector DB
                     ↑
              TLS encrypted
```

```python
# Public endpoint configuration
config = {
    "host": "your-cluster.vectordb.io",
    "port": 443,
    "ssl": True
}
```

**Pros**: Simple setup
**Cons**: Exposed to internet, higher latency

#### Private Link / VPC Peering (Secure)

```
Your VPC                    Provider VPC
────────                    ────────────
Your App → Private Link → Vector DB
           (no internet)
```

```python
# Private endpoint configuration
config = {
    "host": "your-cluster.private.vectordb.io",
    "port": 443,
    "ssl": True,
    "vpc_id": "vpc-12345"  # Provider-specific
}
```

**Pros**: No internet exposure, lower latency
**Cons**: More complex setup, requires VPC configuration

#### Region Selection

Choose regions close to your application and users:

```python
# Region considerations
regions = {
    "us-east-1": {
        "latency_from_ny": "5ms",
        "compliance": ["SOC2", "HIPAA"],
        "cost_multiplier": 1.0
    },
    "eu-west-1": {
        "latency_from_london": "10ms",
        "compliance": ["GDPR", "SOC2"],
        "cost_multiplier": 1.1
    },
    "ap-southeast-1": {
        "latency_from_singapore": "5ms",
        "compliance": ["SOC2"],
        "cost_multiplier": 1.15
    }
}

def select_region(user_location: str, compliance_needs: list) -> str:
    """Select optimal region based on requirements."""
    # Logic to select best region
    pass
```

### Managed Service Options

#### Pinecone

```python
import pinecone

# Initialize
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west1-gcp"  # Region
)

# Create index
pinecone.create_index(
    name="production-index",
    dimension=1536,
    metric="cosine",
    pods=1,
    pod_type="p1.x1"  # Performance tier
)

# Get index
index = pinecone.Index("production-index")
```

**Features**: Fully managed, global distribution, enterprise security
**Pricing**: Pod-based + queries

#### Weaviate Cloud

```python
import weaviate

client = weaviate.Client(
    url="https://your-cluster.weaviate.network",
    auth_client_secret=weaviate.AuthApiKey(
        api_key=os.getenv("WEAVIATE_API_KEY")
    ),
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # For vectorization
    }
)

# Create schema
class_obj = {
    "class": "Document",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002"
        }
    }
}
client.schema.create_class(class_obj)
```

**Features**: Built-in vectorization, GraphQL API, modules
**Pricing**: Capacity-based tiers

#### Qdrant Cloud

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)
```

**Features**: Rust performance, filtering, payload indexing
**Pricing**: Resource-based

### Production Configuration

```python
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class CloudVectorDBConfig:
    """Production configuration for cloud vector database."""
    
    # Connection
    provider: str  # pinecone, weaviate, qdrant
    host: str
    api_key: str
    environment: Optional[str] = None
    
    # Security
    use_ssl: bool = True
    verify_ssl: bool = True
    
    # Performance
    connection_timeout: int = 30
    request_timeout: int = 60
    max_retries: int = 3
    
    # Reliability
    enable_health_checks: bool = True
    health_check_interval: int = 30
    
    @classmethod
    def from_environment(cls, provider: str) -> 'CloudVectorDBConfig':
        """Load configuration from environment variables."""
        prefix = provider.upper()
        
        return cls(
            provider=provider,
            host=os.environ[f"{prefix}_HOST"],
            api_key=os.environ[f"{prefix}_API_KEY"],
            environment=os.getenv(f"{prefix}_ENVIRONMENT"),
            use_ssl=os.getenv(f"{prefix}_USE_SSL", "true").lower() == "true",
            connection_timeout=int(os.getenv(f"{prefix}_CONN_TIMEOUT", "30")),
            request_timeout=int(os.getenv(f"{prefix}_REQ_TIMEOUT", "60"))
        )


class ProductionVectorClient:
    """Production-ready vector database client with retries and health checks."""
    
    def __init__(self, config: CloudVectorDBConfig):
        self.config = config
        self._client = self._create_client()
        self._healthy = True
    
    def _create_client(self):
        """Create provider-specific client."""
        if self.config.provider == "pinecone":
            return self._create_pinecone_client()
        elif self.config.provider == "qdrant":
            return self._create_qdrant_client()
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _create_pinecone_client(self):
        import pinecone
        pinecone.init(
            api_key=self.config.api_key,
            environment=self.config.environment
        )
        return pinecone
    
    def _create_qdrant_client(self):
        from qdrant_client import QdrantClient
        return QdrantClient(
            url=self.config.host,
            api_key=self.config.api_key,
            timeout=self.config.request_timeout
        )
    
    def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            if self.config.provider == "pinecone":
                self._client.list_indexes()
            elif self.config.provider == "qdrant":
                self._client.get_collections()
            self._healthy = True
            return True
        except Exception:
            self._healthy = False
            return False
    
    def query_with_retry(self, *args, **kwargs):
        """Query with automatic retry on failure."""
        import time
        
        for attempt in range(self.config.max_retries):
            try:
                return self._query(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
    
    def _query(self, *args, **kwargs):
        """Provider-specific query implementation."""
        # Implementation depends on provider
        pass


# Usage
config = CloudVectorDBConfig.from_environment("PINECONE")
client = ProductionVectorClient(config)

# Health check
if client.health_check():
    print("Vector database is healthy")
else:
    print("Vector database health check failed")
```

### Monitoring and Observability

```python
import logging
import time
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_db")


def monitor_latency(operation_name: str):
    """Decorator to monitor operation latency."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                latency = (time.time() - start) * 1000
                logger.info(f"{operation_name} completed in {latency:.2f}ms")
                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                logger.error(f"{operation_name} failed after {latency:.2f}ms: {e}")
                raise
        return wrapper
    return decorator


class MonitoredVectorStore:
    """Vector store with built-in monitoring."""
    
    def __init__(self, client):
        self.client = client
        self.metrics = {
            "queries": 0,
            "inserts": 0,
            "errors": 0,
            "total_latency_ms": 0
        }
    
    @monitor_latency("vector_query")
    def query(self, *args, **kwargs):
        self.metrics["queries"] += 1
        return self.client.query(*args, **kwargs)
    
    @monitor_latency("vector_insert")
    def add(self, *args, **kwargs):
        self.metrics["inserts"] += 1
        return self.client.add(*args, **kwargs)
    
    def get_metrics(self) -> dict:
        return self.metrics.copy()
```

## Code Example

Complete cloud deployment setup:

```python
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    """Cloud vector database configuration."""
    provider: str
    api_key: str
    host: str
    region: Optional[str] = None
    index_name: str = "default"
    
    # Security
    use_private_endpoint: bool = False
    verify_ssl: bool = True
    
    # Reliability
    timeout_seconds: int = 30
    max_retries: int = 3


class CloudVectorDB(ABC):
    """Abstract cloud vector database interface."""
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def upsert(self, vectors: list):
        pass
    
    @abstractmethod
    def query(self, vector: list, top_k: int = 5) -> list:
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        pass


class PineconeDB(CloudVectorDB):
    """Pinecone cloud implementation."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.index = None
    
    def connect(self):
        import pinecone
        
        pinecone.init(
            api_key=self.config.api_key,
            environment=self.config.region
        )
        
        self.index = pinecone.Index(self.config.index_name)
        logger.info(f"Connected to Pinecone index: {self.config.index_name}")
    
    def upsert(self, vectors: list):
        if not self.index:
            raise RuntimeError("Not connected")
        
        self.index.upsert(vectors=vectors)
    
    def query(self, vector: list, top_k: int = 5) -> list:
        if not self.index:
            raise RuntimeError("Not connected")
        
        results = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        return results.matches
    
    def health_check(self) -> bool:
        try:
            import pinecone
            pinecone.list_indexes()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


def create_cloud_db(config: CloudConfig) -> CloudVectorDB:
    """Factory function for cloud vector databases."""
    providers = {
        "pinecone": PineconeDB,
        # Add other providers as needed
    }
    
    provider_class = providers.get(config.provider)
    if not provider_class:
        raise ValueError(f"Unknown provider: {config.provider}")
    
    db = provider_class(config)
    db.connect()
    return db


# Production usage
if __name__ == "__main__":
    # Load configuration from environment
    config = CloudConfig(
        provider="pinecone",
        api_key=os.environ["PINECONE_API_KEY"],
        host="https://api.pinecone.io",
        region=os.environ.get("PINECONE_REGION", "us-west1-gcp"),
        index_name=os.environ.get("PINECONE_INDEX", "production")
    )
    
    # Create and use database
    db = create_cloud_db(config)
    
    if db.health_check():
        print("Cloud vector database is ready")
    else:
        print("Health check failed - check configuration")
```

## Key Takeaways

1. **Never hardcode credentials** - use environment variables or secrets managers
2. **Use private endpoints when possible** - reduces exposure and latency
3. **Select regions close to your users** - minimize latency
4. **Implement retry logic** - cloud services can have transient failures
5. **Monitor everything** - track latency, errors, and usage
6. **Plan for failover** - consider multi-region deployments for critical systems

## Additional Resources

- [Pinecone Security](https://docs.pinecone.io/docs/security) - Enterprise security features
- [Weaviate Cloud Setup](https://weaviate.io/developers/wcs) - Managed service documentation
- [AWS Private Link](https://aws.amazon.com/privatelink/) - Private connectivity options
