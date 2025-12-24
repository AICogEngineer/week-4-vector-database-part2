# Vector Database Best Practices

## Learning Objectives
- Apply production best practices for vector database management
- Implement robust data management and backup strategies
- Set up monitoring and alerting for production systems
- Plan for maintenance and operational excellence

## Why This Matters

Building a working prototype is easy. Running a production vector database that serves millions of queries reliably is an entirely different challenge. These best practices, learned from real-world deployments, help you avoid common pitfalls and build systems that scale.

## The Concept

### Data Management

#### Collection Organization

```python
# Anti-pattern: One collection for everything
collection = client.create_collection("all_documents")
# Problems: Hard to manage, can't scale different data types independently

# Best practice: Separate collections by domain/lifecycle
collections = {
    "user_content": client.create_collection("user_content"),
    "product_catalog": client.create_collection("product_catalog"),
    "support_docs": client.create_collection("support_docs"),
    "archived": client.create_collection("archived")
}
```

**Collection Naming Conventions**:
```python
# Pattern: {domain}_{type}_{version}
"products_embeddings_v2"
"support_articles_v1"
"user_queries_v1"

# Include version for safe migrations
# Use lowercase with underscores
# Be descriptive but concise
```

#### ID Strategy

```python
# Anti-pattern: Sequential integers
ids = ["1", "2", "3"]  # Collision risk, no meaning

# Anti-pattern: Random UUIDs only
ids = [str(uuid.uuid4())]  # Hard to debug, no context

# Best practice: Meaningful composite IDs
def create_document_id(source: str, chunk_index: int, version: str = "v1") -> str:
    """Create a meaningful, unique document ID."""
    import hashlib
    
    # Create deterministic ID from content attributes
    base = f"{source}_{chunk_index}"
    hash_suffix = hashlib.md5(base.encode()).hexdigest()[:8]
    
    return f"{version}_{source}_{chunk_index}_{hash_suffix}"

# Result: "v1_mlguide_0_a1b2c3d4"
# - Sortable by source
# - Debuggable (know where it came from)
# - Unique (hash prevents collisions)
```

#### Data Lifecycle

```python
from datetime import datetime, timedelta


class DataLifecycleManager:
    """Manage document lifecycle in vector database."""
    
    def __init__(self, collection, retention_days: int = 365):
        self.collection = collection
        self.retention_days = retention_days
    
    def add_with_lifecycle(
        self,
        id: str,
        embedding: list,
        document: str,
        metadata: dict
    ):
        """Add document with lifecycle metadata."""
        enhanced_metadata = {
            **metadata,
            "_created_at": datetime.utcnow().isoformat(),
            "_expires_at": (
                datetime.utcnow() + timedelta(days=self.retention_days)
            ).isoformat()
        }
        
        self.collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[enhanced_metadata]
        )
    
    def cleanup_expired(self) -> int:
        """Remove expired documents."""
        now = datetime.utcnow().isoformat()
        
        # Find expired documents
        expired = self.collection.get(
            where={"_expires_at": {"$lt": now}},
            include=["metadatas"]
        )
        
        if expired["ids"]:
            self.collection.delete(ids=expired["ids"])
        
        return len(expired["ids"])
    
    def archive_old_documents(self, archive_collection, days: int = 180):
        """Move old documents to archive collection."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        old_docs = self.collection.get(
            where={"_created_at": {"$lt": cutoff}},
            include=["embeddings", "documents", "metadatas"]
        )
        
        if old_docs["ids"]:
            # Copy to archive
            archive_collection.add(
                ids=old_docs["ids"],
                embeddings=old_docs["embeddings"],
                documents=old_docs["documents"],
                metadatas=old_docs["metadatas"]
            )
            
            # Delete from primary
            self.collection.delete(ids=old_docs["ids"])
        
        return len(old_docs["ids"])
```

### Backup and Recovery

```python
import json
import os
from pathlib import Path
from datetime import datetime


class VectorDBBackup:
    """Backup and restore vector database collections."""
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup_collection(self, collection, collection_name: str) -> str:
        """Create a backup of a collection."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{collection_name}_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        # Get all data
        all_data = collection.get(
            include=["embeddings", "documents", "metadatas"]
        )
        
        # Save in chunks for large collections
        chunk_size = 10000
        total = len(all_data["ids"])
        
        for i in range(0, total, chunk_size):
            end = min(i + chunk_size, total)
            chunk_data = {
                "ids": all_data["ids"][i:end],
                "embeddings": all_data["embeddings"][i:end] if all_data["embeddings"] else None,
                "documents": all_data["documents"][i:end] if all_data["documents"] else None,
                "metadatas": all_data["metadatas"][i:end] if all_data["metadatas"] else None
            }
            
            chunk_file = backup_path / f"chunk_{i // chunk_size}.json"
            with open(chunk_file, "w") as f:
                json.dump(chunk_data, f)
        
        # Save metadata
        meta_file = backup_path / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump({
                "collection_name": collection_name,
                "timestamp": timestamp,
                "total_documents": total,
                "chunk_count": (total + chunk_size - 1) // chunk_size
            }, f)
        
        return str(backup_path)
    
    def restore_collection(self, backup_path: str, collection) -> int:
        """Restore a collection from backup."""
        backup_path = Path(backup_path)
        
        # Load metadata
        with open(backup_path / "metadata.json") as f:
            meta = json.load(f)
        
        restored = 0
        for chunk_file in sorted(backup_path.glob("chunk_*.json")):
            with open(chunk_file) as f:
                chunk_data = json.load(f)
            
            collection.add(
                ids=chunk_data["ids"],
                embeddings=chunk_data["embeddings"],
                documents=chunk_data["documents"],
                metadatas=chunk_data["metadatas"]
            )
            restored += len(chunk_data["ids"])
        
        return restored
    
    def list_backups(self, collection_name: str = None) -> list:
        """List available backups."""
        backups = []
        
        for path in self.backup_dir.iterdir():
            if path.is_dir():
                meta_file = path / "metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                    
                    if collection_name is None or meta["collection_name"] == collection_name:
                        backups.append({
                            "path": str(path),
                            **meta
                        })
        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
```

### Monitoring and Alerting

```python
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vectordb_monitor")


@dataclass
class HealthMetrics:
    """Health metrics for vector database."""
    is_healthy: bool = True
    query_latency_ms: float = 0
    collection_size: int = 0
    last_check: str = ""
    error_message: Optional[str] = None


class VectorDBMonitor:
    """Monitor vector database health and performance."""
    
    def __init__(
        self,
        collection,
        alert_callback: Optional[Callable[[str, str], None]] = None
    ):
        self.collection = collection
        self.alert_callback = alert_callback or self._default_alert
        
        # Thresholds
        self.query_latency_threshold_ms = 500
        self.error_count_threshold = 5
        
        # State
        self.consecutive_errors = 0
        self.last_metrics = HealthMetrics()
    
    def _default_alert(self, level: str, message: str):
        """Default alert handler - logs to console."""
        if level == "critical":
            logger.critical(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)
    
    def check_health(self) -> HealthMetrics:
        """Perform health check."""
        metrics = HealthMetrics()
        metrics.last_check = datetime.utcnow().isoformat()
        
        try:
            # Check query performance
            test_embedding = [0.0] * 384  # Adjust dimension as needed
            
            start = time.time()
            self.collection.query(
                query_embeddings=[test_embedding],
                n_results=1
            )
            metrics.query_latency_ms = (time.time() - start) * 1000
            
            # Check collection size
            metrics.collection_size = self.collection.count()
            
            # Evaluate health
            if metrics.query_latency_ms > self.query_latency_threshold_ms:
                self.alert_callback(
                    "warning",
                    f"High query latency: {metrics.query_latency_ms:.0f}ms"
                )
            
            metrics.is_healthy = True
            self.consecutive_errors = 0
            
        except Exception as e:
            metrics.is_healthy = False
            metrics.error_message = str(e)
            self.consecutive_errors += 1
            
            if self.consecutive_errors >= self.error_count_threshold:
                self.alert_callback(
                    "critical",
                    f"Vector DB unhealthy - {self.consecutive_errors} consecutive errors: {e}"
                )
        
        self.last_metrics = metrics
        return metrics
    
    def get_statistics(self) -> dict:
        """Get collection statistics."""
        return {
            "collection_count": self.collection.count(),
            "last_health_check": self.last_metrics.last_check,
            "is_healthy": self.last_metrics.is_healthy,
            "last_query_latency_ms": self.last_metrics.query_latency_ms
        }


# Usage
def send_slack_alert(level: str, message: str):
    """Send alert to Slack (example)."""
    # In production, integrate with your alerting system
    print(f"[SLACK {level.upper()}] {message}")

monitor = VectorDBMonitor(collection, alert_callback=send_slack_alert)

# Run health check
health = monitor.check_health()
print(f"Healthy: {health.is_healthy}, Latency: {health.query_latency_ms:.2f}ms")
```

### Operational Checklist

```python
PRODUCTION_CHECKLIST = """
BEFORE DEPLOYMENT
-----------------
[ ] Data backup strategy implemented
[ ] Monitoring and alerting configured
[ ] Connection pooling for high concurrency
[ ] API keys stored in secrets manager
[ ] Rate limiting configured
[ ] Error handling and retries implemented

DATA MANAGEMENT
---------------
[ ] Collection naming convention documented
[ ] ID generation strategy defined
[ ] Metadata schema documented
[ ] Data lifecycle (retention/archival) defined
[ ] Duplicate detection implemented

PERFORMANCE
-----------
[ ] Appropriate index type selected (HNSW parameters tuned)
[ ] Batch operations used for bulk data
[ ] Query results limited appropriately
[ ] Include fields minimized
[ ] Connection reuse implemented

SECURITY
--------
[ ] TLS enabled for all connections
[ ] API keys rotated regularly
[ ] Access controls configured
[ ] Audit logging enabled
[ ] Data encryption verified

RELIABILITY
-----------
[ ] Health checks implemented
[ ] Automatic failover configured (if cloud)
[ ] Backup schedule established
[ ] Recovery procedure tested
[ ] Runbook documented

OBSERVABILITY
-------------
[ ] Query latency tracked
[ ] Error rates monitored
[ ] Collection sizes tracked
[ ] Resource utilization monitored
[ ] Alerts configured for anomalies
"""

def print_checklist():
    print(PRODUCTION_CHECKLIST)
```

## Code Example

Production-ready vector database wrapper combining best practices:

```python
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import chromadb

logger = logging.getLogger(__name__)


@dataclass
class VectorDBConfig:
    """Configuration for production vector database."""
    collection_name: str
    persist_directory: str = "./vector_data"
    backup_directory: str = "./backups"
    hnsw_m: int = 16
    hnsw_ef_search: int = 100
    batch_size: int = 1000
    enable_monitoring: bool = True


class ProductionVectorDB:
    """Production-ready vector database with best practices."""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        
        # Initialize client and collection
        self.client = chromadb.PersistentClient(path=config.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": config.hnsw_m,
                "hnsw:search_ef": config.hnsw_ef_search
            }
        )
        
        # Initialize utilities
        self.backup_manager = VectorDBBackup(config.backup_directory)
        self.monitor = VectorDBMonitor(self.collection) if config.enable_monitoring else None
        self.lifecycle = DataLifecycleManager(self.collection)
        
        logger.info(f"Initialized ProductionVectorDB: {config.collection_name}")
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> int:
        """Add documents with batching and lifecycle tracking."""
        total = len(ids)
        added = 0
        
        for i in range(0, total, self.config.batch_size):
            end = min(i + self.config.batch_size, total)
            
            # Enhance metadata with lifecycle info
            enhanced_metas = []
            for meta in metadatas[i:end]:
                enhanced_metas.append({
                    **meta,
                    "_created_at": datetime.utcnow().isoformat(),
                    "_version": "v1"
                })
            
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=enhanced_metas
            )
            added += end - i
        
        logger.info(f"Added {added} documents to {self.config.collection_name}")
        return added
    
    def query(
        self,
        embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """Query with performance tracking."""
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    
    def backup(self) -> str:
        """Create a backup."""
        path = self.backup_manager.backup_collection(
            self.collection,
            self.config.collection_name
        )
        logger.info(f"Backup created: {path}")
        return path
    
    def health_check(self) -> Dict:
        """Run health check."""
        if self.monitor:
            metrics = self.monitor.check_health()
            return {
                "healthy": metrics.is_healthy,
                "latency_ms": metrics.query_latency_ms,
                "collection_size": metrics.collection_size,
                "error": metrics.error_message
            }
        return {"healthy": True, "monitoring_disabled": True}
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "name": self.config.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.config.persist_directory
        }


# Example usage
if __name__ == "__main__":
    config = VectorDBConfig(
        collection_name="production_docs",
        persist_directory="./prod_data"
    )
    
    db = ProductionVectorDB(config)
    
    # Health check
    health = db.health_check()
    print(f"Health: {health}")
    
    # Get stats
    stats = db.get_stats()
    print(f"Stats: {stats}")
```

## Key Takeaways

1. **Organize collections by domain** - separate concerns, scale independently
2. **Use meaningful IDs** - include source info for debugging
3. **Implement lifecycle management** - retention, archival, cleanup
4. **Regular backups are essential** - test your restore procedure
5. **Monitor everything** - latency, errors, collection sizes
6. **Follow the checklist** - systematic approach prevents oversights

## Additional Resources

- [Chroma Production Guide](https://docs.trychroma.com/deployment) - Official deployment recommendations
- [Database Reliability Engineering](https://www.oreilly.com/library/view/database-reliability-engineering/9781491925935/) - General database operations
- [SRE Book (Google)](https://sre.google/sre-book/table-of-contents/) - Site reliability principles
