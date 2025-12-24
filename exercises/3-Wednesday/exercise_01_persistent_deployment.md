# Exercise 01: Persistent Deployment

## Overview

Local development with in-memory databases is convenient, but production systems need persistent storage with proper data management. In this exercise, you'll configure a production-ready persistent vector database.

## Learning Objectives

- Configure Chroma for persistent storage
- Implement backup and restore functionality
- Create an initialization and recovery workflow
- Handle data migration between environments

## The Scenario

Your team is moving from development to staging. You need to:
1. Set up persistent storage that survives restarts
2. Create a backup strategy for disaster recovery
3. Build a migration tool for moving data between environments

## Your Tasks

### Task 1: Persistent Client Setup (20 min)

Implement `setup_persistent_client()`:
- Configure Chroma's PersistentClient with a data directory
- Ensure directory exists and is writable
- Return client and collection

### Task 2: Backup System (25 min)

Implement `BackupManager`:
- `create_backup()`: Export all data to JSON format
- `restore_backup()`: Import data from backup file
- `list_backups()`: Show available backups

The backup should include:
- All documents and their IDs
- All metadata
- Collection configuration

### Task 3: Data Migration (20 min)

Implement `migrate_collection()`:
- Copy data from source to target collection
- Preserve all metadata
- Handle name conflicts
- Report migration statistics

### Task 4: Health Check (15 min)

Implement `DatabaseHealthCheck`:
- `check_storage()`: Verify disk space and permissions
- `check_collection()`: Validate collection integrity
- `check_embeddings()`: Sample check that embeddings are valid

## Definition of Done

- [_] Persistent client configured and working
- [_] Backup creates valid JSON exports
- [_] Restore successfully recreates data
- [_] Migration between collections works
- [_] Health checks report accurate status

## Testing Your Solution

```bash
cd exercises/3-Wednesday/starter_code
python exercise_01_starter.py
```

Expected output:
```
=== Persistent Deployment Exercise ===

[OK] Persistent client initialized at ./chroma_data

=== Backup Test ===
[OK] Created backup: backup_2024-01-15_10-30-00.json
[OK] Backup contains 50 documents

=== Restore Test ===
[OK] Restored 50 documents from backup

=== Migration Test ===
[OK] Migrated 50 documents from 'source' to 'target'

=== Health Check ===
Storage: [OK] 15.2 GB available
Collection: [OK] 50 documents, no orphans
Embeddings: [OK] Sample validation passed

[OK] Persistent deployment configured successfully!
```

## Stretch Goals (Optional)

1. Add incremental backups (only changed documents)
2. Implement backup compression
3. Add backup verification (compare counts/checksums)
