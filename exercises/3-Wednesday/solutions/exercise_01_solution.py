"""
Exercise 01: Persistent Deployment - Solution

Complete implementation of persistent vector database configuration.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("./chroma_data")
BACKUP_DIR = Path("./backups")
COLLECTION_NAME = "production_docs"


# ============================================================================
# SAMPLE DATA
# ============================================================================

SAMPLE_DOCUMENTS = [
    {"id": "doc_001", "content": "Getting started with machine learning requires understanding basic concepts.", "metadata": {"category": "tutorial", "version": "1.0"}},
    {"id": "doc_002", "content": "Deep learning uses neural networks with multiple layers for complex tasks.", "metadata": {"category": "tutorial", "version": "1.0"}},
    {"id": "doc_003", "content": "Natural language processing enables computers to understand human language.", "metadata": {"category": "reference", "version": "2.0"}},
    {"id": "doc_004", "content": "Vector databases store high-dimensional embeddings for similarity search.", "metadata": {"category": "reference", "version": "2.0"}},
    {"id": "doc_005", "content": "RAG systems combine retrieval and generation for better AI responses.", "metadata": {"category": "tutorial", "version": "2.0"}},
]


# ============================================================================
# SOLUTION IMPLEMENTATIONS
# ============================================================================

def setup_persistent_client(data_dir: Path) -> tuple:
    """
    Set up a persistent Chroma client with proper configuration.
    """
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize persistent client
    client = chromadb.PersistentClient(path=str(data_dir))
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    return client, collection


class BackupManager:
    """
    Manages backups and restores for vector database collections.
    """
    
    def __init__(self, collection, backup_dir: Path, embedder=None):
        self.collection = collection
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or SentenceTransformer('all-MiniLM-L6-v2')
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """
        Create a backup of the collection.
        """
        # Generate backup name if not provided
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_name = f"backup_{timestamp}.json"
        
        if not backup_name.endswith('.json'):
            backup_name += '.json'
        
        # Get all data from collection
        data = self.collection.get(include=["documents", "metadatas"])
        
        # Create backup structure
        backup_data = {
            "collection_name": self.collection.name,
            "created_at": datetime.now().isoformat(),
            "count": len(data["ids"]),
            "ids": data["ids"],
            "documents": data["documents"],
            "metadatas": data["metadatas"]
        }
        
        # Save to file
        backup_path = self.backup_dir / backup_name
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        return backup_name
    
    def restore_backup(self, backup_name: str, clear_existing: bool = True) -> Dict:
        """
        Restore collection from a backup file.
        """
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_name}")
        
        # Load backup
        with open(backup_path) as f:
            data = json.load(f)
        
        # Clear existing data if requested
        if clear_existing:
            existing = self.collection.get()
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        
        # Re-embed documents (embeddings aren't stored in backup)
        if data["documents"]:
            embeddings = self.embedder.encode(data["documents"]).tolist()
            
            # Add back to collection
            self.collection.add(
                ids=data["ids"],
                documents=data["documents"],
                embeddings=embeddings,
                metadatas=data["metadatas"]
            )
        
        return {
            "restored_count": len(data["ids"]),
            "backup_name": backup_name,
            "original_date": data.get("created_at", "unknown")
        }
    
    def list_backups(self) -> List[str]:
        """List all available backup files."""
        if not self.backup_dir.exists():
            return []
        
        backups = [f.name for f in self.backup_dir.glob("*.json")]
        return sorted(backups, reverse=True)  # Most recent first


def migrate_collection(
    source_collection,
    target_collection,
    embedder
) -> Dict:
    """
    Migrate data from source to target collection.
    """
    # Get all data from source
    data = source_collection.get(include=["documents", "metadatas"])
    
    if not data["ids"]:
        return {"migrated_count": 0, "source": source_collection.name, "target": target_collection.name}
    
    # Re-embed documents
    embeddings = embedder.encode(data["documents"]).tolist()
    
    # Add to target collection
    target_collection.add(
        ids=data["ids"],
        documents=data["documents"],
        embeddings=embeddings,
        metadatas=data["metadatas"]
    )
    
    return {
        "migrated_count": len(data["ids"]),
        "source": source_collection.name,
        "target": target_collection.name
    }


@dataclass
class HealthStatus:
    """Health check result."""
    name: str
    status: str  # "OK", "WARNING", "ERROR"
    message: str


class DatabaseHealthCheck:
    """
    Health checking for vector database.
    """
    
    def __init__(self, data_dir: Path, collection):
        self.data_dir = data_dir
        self.collection = collection
    
    def check_storage(self) -> HealthStatus:
        """Check storage health."""
        try:
            # Check directory exists
            if not self.data_dir.exists():
                return HealthStatus(
                    name="Storage",
                    status="ERROR",
                    message=f"Data directory does not exist: {self.data_dir}"
                )
            
            # Check writable
            test_file = self.data_dir / ".health_check"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                return HealthStatus(
                    name="Storage",
                    status="ERROR",
                    message="Data directory is not writable"
                )
            
            # Check disk space (simplified - just check if we can write)
            # In production, you'd use shutil.disk_usage()
            try:
                usage = shutil.disk_usage(self.data_dir)
                gb_free = usage.free / (1024 ** 3)
                
                if gb_free < 1:
                    return HealthStatus(
                        name="Storage",
                        status="WARNING",
                        message=f"Low disk space: {gb_free:.1f} GB available"
                    )
                
                return HealthStatus(
                    name="Storage",
                    status="OK",
                    message=f"{gb_free:.1f} GB available"
                )
            except:
                return HealthStatus(
                    name="Storage",
                    status="OK",
                    message="Directory accessible"
                )
            
        except Exception as e:
            return HealthStatus(
                name="Storage",
                status="ERROR",
                message=str(e)
            )
    
    def check_collection(self) -> HealthStatus:
        """Check collection health."""
        try:
            # Try to count documents
            count = self.collection.count()
            
            return HealthStatus(
                name="Collection",
                status="OK",
                message=f"{count} documents"
            )
            
        except Exception as e:
            return HealthStatus(
                name="Collection",
                status="ERROR",
                message=str(e)
            )
    
    def run_all_checks(self) -> List[HealthStatus]:
        """Run all health checks and return results."""
        return [
            self.check_storage(),
            self.check_collection(),
        ]


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    """Test the persistent deployment setup."""
    print("=" * 60)
    print("Exercise 01: Persistent Deployment - SOLUTION")
    print("=" * 60)
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Clean up from previous runs
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    
    # Test 1: Setup persistent client
    print("\n=== Persistent Client Setup ===")
    
    client, collection = setup_persistent_client(DATA_DIR)
    print(f"[OK] Persistent client initialized at {DATA_DIR}")
    
    # Add sample data
    embeddings = embedder.encode([d["content"] for d in SAMPLE_DOCUMENTS]).tolist()
    collection.add(
        ids=[d["id"] for d in SAMPLE_DOCUMENTS],
        documents=[d["content"] for d in SAMPLE_DOCUMENTS],
        embeddings=embeddings,
        metadatas=[d["metadata"] for d in SAMPLE_DOCUMENTS]
    )
    print(f"[OK] Added {len(SAMPLE_DOCUMENTS)} sample documents")
    
    # Test 2: Backup
    print("\n=== Backup Test ===")
    
    backup_mgr = BackupManager(collection, BACKUP_DIR, embedder)
    backup_name = backup_mgr.create_backup()
    print(f"[OK] Created backup: {backup_name}")
    
    # Verify backup
    backup_path = BACKUP_DIR / backup_name
    with open(backup_path) as f:
        data = json.load(f)
    print(f"[OK] Backup contains {len(data['ids'])} documents")
    
    # List backups
    backups = backup_mgr.list_backups()
    print(f"[OK] Available backups: {backups}")
    
    # Test 3: Restore
    print("\n=== Restore Test ===")
    
    # Clear and restore
    stats = backup_mgr.restore_backup(backup_name)
    print(f"[OK] Restored {stats['restored_count']} documents from backup")
    
    # Verify restore
    print(f"[OK] Collection now has {collection.count()} documents")
    
    # Test 4: Migration
    print("\n=== Migration Test ===")
    
    target_collection = client.get_or_create_collection("migrated_docs")
    migration_stats = migrate_collection(collection, target_collection, embedder)
    print(f"[OK] Migrated {migration_stats['migrated_count']} documents from '{migration_stats['source']}' to '{migration_stats['target']}'")
    
    # Test 5: Health Check
    print("\n=== Health Check ===")
    
    health = DatabaseHealthCheck(DATA_DIR, collection)
    results = health.run_all_checks()
    
    for check in results:
        print(f"{check.name}: [{check.status}] {check.message}")
    
    print("\n" + "=" * 60)
    print("[OK] Persistent deployment complete!")
    print("=" * 60)
    
    # Cleanup
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)


if __name__ == "__main__":
    run_tests()
