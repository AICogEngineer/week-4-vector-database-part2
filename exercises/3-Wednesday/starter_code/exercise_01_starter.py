"""
Exercise 01: Persistent Deployment - Starter Code

Configure a production-ready persistent vector database.

Instructions:
1. Implement each TODO function
2. Run this file to test your implementations
3. Check the expected output in the exercise guide
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("./chroma_data")
BACKUP_DIR = Path("./backups")
COLLECTION_NAME = "production_docs"


# ============================================================================
# SAMPLE DATA (DO NOT MODIFY)
# ============================================================================

SAMPLE_DOCUMENTS = [
    {"id": "doc_001", "content": "Getting started with machine learning requires understanding basic concepts.", "metadata": {"category": "tutorial", "version": "1.0"}},
    {"id": "doc_002", "content": "Deep learning uses neural networks with multiple layers for complex tasks.", "metadata": {"category": "tutorial", "version": "1.0"}},
    {"id": "doc_003", "content": "Natural language processing enables computers to understand human language.", "metadata": {"category": "reference", "version": "2.0"}},
    {"id": "doc_004", "content": "Vector databases store high-dimensional embeddings for similarity search.", "metadata": {"category": "reference", "version": "2.0"}},
    {"id": "doc_005", "content": "RAG systems combine retrieval and generation for better AI responses.", "metadata": {"category": "tutorial", "version": "2.0"}},
]


# ============================================================================
# TODO: IMPLEMENT THESE FUNCTIONS
# ============================================================================

def setup_persistent_client(data_dir: Path) -> tuple:
    """
    Set up a persistent Chroma client with proper configuration.
    
    Tasks:
    1. Create data directory if it doesn't exist
    2. Initialize PersistentClient with the directory
    3. Create or get the collection
    4. Return (client, collection)
    
    Args:
        data_dir: Path for persistent storage
        
    Returns:
        Tuple of (client, collection)
    """
    # TODO: Implement this function
    # Hints:
    # - Use Path.mkdir(parents=True, exist_ok=True)
    # - Use chromadb.PersistentClient(path=str(data_dir))
    # - Use client.get_or_create_collection()
    
    pass  # Remove this and add your implementation


class BackupManager:
    """
    Manages backups and restores for vector database collections.
    """
    
    def __init__(self, collection, backup_dir: Path):
        self.collection = collection
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """
        Create a backup of the collection.
        
        Tasks:
        1. Get all documents, metadata, and IDs from collection
        2. Serialize to JSON format
        3. Save to backup directory with timestamp
        4. Return backup filename
        
        Returns:
            Backup filename
        """
        # TODO: Implement this method
        # Hints:
        # - Use collection.get() to retrieve all data
        # - Include documents, metadatas, and ids
        # - Use json.dump() to save
        
        pass  # Remove this and add your implementation
    
    def restore_backup(self, backup_name: str) -> Dict:
        """
        Restore collection from a backup file.
        
        Tasks:
        1. Load backup JSON
        2. Clear current collection (optional, configurable)
        3. Re-add all documents with embeddings
        4. Return restore statistics
        
        Returns:
            Stats dict with restored counts
        """
        # TODO: Implement this method
        # Note: You'll need to re-embed the documents since we don't store embeddings
        
        pass  # Remove this and add your implementation
    
    def list_backups(self) -> List[str]:
        """List all available backup files."""
        # TODO: Implement this method
        
        pass  # Remove this and add your implementation


def migrate_collection(
    source_collection,
    target_collection,
    embedder
) -> Dict:
    """
    Migrate data from source to target collection.
    
    Tasks:
    1. Get all data from source
    2. Re-embed documents (embeddings may not transfer)
    3. Add to target collection
    4. Return migration statistics
    
    Args:
        source_collection: Collection to migrate from
        target_collection: Collection to migrate to
        embedder: Embedding model
        
    Returns:
        Stats dict with migration counts
    """
    # TODO: Implement this function
    
    pass  # Remove this and add your implementation


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
        """
        Check storage health.
        
        Tasks:
        1. Verify directory exists and is writable
        2. Check available disk space
        3. Return appropriate status
        """
        # TODO: Implement this method
        
        pass  # Remove this and add your implementation
    
    def check_collection(self) -> HealthStatus:
        """
        Check collection health.
        
        Tasks:
        1. Verify collection is accessible
        2. Count documents
        3. Check for any errors
        """
        # TODO: Implement this method
        
        pass  # Remove this and add your implementation
    
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
    print("Exercise 01: Persistent Deployment")
    print("=" * 60)
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Clean up from previous runs
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    
    # Test 1: Setup persistent client
    print("\n=== Persistent Client Setup ===")
    try:
        result = setup_persistent_client(DATA_DIR)
        
        if result is None:
            print("[INFO] setup_persistent_client not implemented yet")
            return
        
        client, collection = result
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
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Test 2: Backup
    print("\n=== Backup Test ===")
    try:
        backup_mgr = BackupManager(collection, BACKUP_DIR)
        backup_name = backup_mgr.create_backup()
        
        if backup_name is None:
            print("[INFO] create_backup not implemented yet")
        else:
            print(f"[OK] Created backup: {backup_name}")
            
            # Verify backup
            backup_path = BACKUP_DIR / backup_name
            if backup_path.exists():
                with open(backup_path) as f:
                    data = json.load(f)
                print(f"[OK] Backup contains {len(data.get('ids', []))} documents")
            
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Test 3: Health Check
    print("\n=== Health Check ===")
    try:
        health = DatabaseHealthCheck(DATA_DIR, collection)
        results = health.run_all_checks()
        
        if not results or results[0] is None:
            print("[INFO] Health checks not implemented yet")
        else:
            for check in results:
                print(f"{check.name}: [{check.status}] {check.message}")
                
    except Exception as e:
        print(f"[ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("[OK] Test complete!")
    print("=" * 60)
    
    # Cleanup
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)


if __name__ == "__main__":
    run_tests()
