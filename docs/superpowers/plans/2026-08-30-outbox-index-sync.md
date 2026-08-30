# Week07 Outbox Index Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task with checkbox tracking.

**Goal:** Persist long-term-memory changes and indexing work atomically, then reconcile Milvus through an idempotent, retryable CLI worker.

**Architecture:** MySQL remains authoritative. `memory_outbox` stores only `memory_id` and delivery state. A worker claims an event, rereads MySQL, upserts active memory or deletes inactive/missing memory in Milvus, and records the outcome with a lease token.

**Tech Stack:** Python, SQLAlchemy, Alembic, MySQL, SQLite, PyMilvus, pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-outbox-index-sync-design.md`

## Global Constraints

- Use Windows CMD commands for learner-facing steps; never read or change `.env`.
- Never commit/delete `pytest-of-wurunnan/`.
- MySQL is authoritative; Milvus is rebuildable and only stores vector index data.
- `memory add`/`memory deactivate` enqueue work; they do not call Embedding or Milvus inline.
- Maximum attempts: 3. Lease: 60 seconds. Retry delays: 1 then 2 seconds.
- No queue framework, daemon, FastAPI, authentication, observability, or deployment work.

## File Structure

- `app/models.py`: `MemoryOutbox` ORM model and constants.
- `migrations/versions/<revision>_create_memory_outbox.py`: table and ready-event index.
- `app/long_term_memory.py`: one-transaction memory/event writes and internal `get_by_id`.
- `app/outbox.py`: state repository and reconciliation worker.
- `app/milvus_memory.py`: idempotent vector deletion.
- `main.py`: Outbox CLI commands and dependency wiring.
- `tests/test_database.py`, `tests/test_long_term_memory.py`, `tests/test_outbox.py`, `tests/test_milvus_memory.py`, `tests/test_main.py`: isolated tests.
- `README.md`: architecture, commands, and acceptance steps.

### Task 1: Schema and atomic writes

**Files:** modify `app/models.py`, `app/long_term_memory.py`, `tests/test_database.py`, `tests/test_long_term_memory.py`; create one Alembic migration.

- [ ] Write a failing SQLite test asserting `memory_outbox` has `id`, `memory_id`, `event_type`, `status`, `attempt_count`, `available_at`, `lease_token`, `lease_expires_at`, `last_error`, `processed_at`, `created_at`, `updated_at`, and index `ix_memory_outbox_status_available_id(status, available_at, id)`.
- [ ] Run ` .venv\Scripts\python.exe -m pytest tests\test_database.py -q`; confirm it fails because the table is absent.
- [ ] Add status constants `pending`, `processing`, `succeeded`, `failed`, event `memory.index_requested`, and the `MemoryOutbox` model using the existing UTC and bigint/SQLite conventions.
- [ ] Create migration from `77ca49d48dd1`; upgrade creates table/index and downgrade removes index/table.
- [ ] Write failing tests that `add()` creates one pending event and `deactivate()` creates one additional pending event for the same ID.
- [ ] Implement `session.flush()` after adding memory, add the event before the same single commit; enqueue after soft deactivation before its commit; add internal `get_by_id(memory_id)` without active filtering.
- [ ] Run database and repository tests; commit `feat: add memory outbox schema`.

### Task 2: Idempotent target operation and event state machine

**Files:** modify `app/milvus_memory.py`, `tests/test_milvus_memory.py`; create `app/outbox.py`, `tests/test_outbox.py`.

- [ ] Write a failing test that `MilvusMemoryVectorIndex.delete(101)` ensures collection then calls client delete with `[101]`; run the Milvus test and observe missing-method failure.
- [ ] Implement delete: reject IDs below one, ensure collection, call PyMilvus delete by primary-key IDs without a search.
- [ ] Write failing state tests: only due pending or expired-lease processing events can be claimed; claiming writes a UUID lease token, 60-second lease and increments attempts.
- [ ] Implement `MemoryOutboxRepository.claim_next(now, lease_seconds=60)` with ordered selection and `with_for_update(skip_locked=True)`.
- [ ] Write failing token tests: wrong token cannot finish; correct token clears lease and succeeds; failures store `ExceptionType: message`, schedule 1/2 second backoff, and third failure becomes failed.
- [ ] Implement `mark_succeeded`, `mark_failed`, and `retry_all_failed`; retry resets failed events to pending, attempts zero, immediate availability, no error/lease.
- [ ] Run targeted tests; commit `feat: add outbox event state machine`.

### Task 3: Reconciliation worker

**Files:** modify `app/outbox.py`, `tests/test_outbox.py`.

- [ ] Write a failing test where an active MySQL memory makes `drain(limit=1)` call `MemorySyncService.sync`, succeeds its event, and returns `succeeded=1,retrying=0,failed=0`.
- [ ] Implement `MemoryOutboxWorker(outbox_repository, long_term_memory_repository, memory_sync_service, vector_index, max_attempts=3)` and `drain(limit, now=None)`; reject limit below one.
- [ ] Write failing tests where inactive or absent MySQL memory calls `vector_index.delete(memory_id)` and succeeds, never syncs.
- [ ] Implement that authority check for every claimed event; catch per-event external failures, call `mark_failed`, and continue draining later events.
- [ ] Write failing test where embedding failure leaves a pending event with attempt one/error and next due time; after advancing availability, a successful drain completes it.
- [ ] Run `pytest tests\test_outbox.py tests\test_memory_sync.py -q`; commit `feat: reconcile memory indexes from outbox`.

### Task 4: CLI, docs, and verification

**Files:** modify `main.py`, `tests/test_main.py`, `README.md`.

- [ ] Write failing parser tests for `memory outbox drain --limit 7` and required `memory outbox retry-failed --all`; zero limit and omitted `--all` must fail.
- [ ] Write failing add-command test with factories that raise if Embedding/Milvus/Sync is touched; assert add only saves MySQL/event and prints `已新增长期记忆并创建索引任务：101`.
- [ ] Implement commands: drain default limit 10 and report `成功 <n>，重试 <n>，失败 <n>`; retry-failed only uses database repository and reports requeued count.
- [ ] Write tests proving retry-failed does not construct Milvus, Embeddings, or worker; run CLI/repository/worker test suite.
- [ ] Update README with transactional write path, eventual consistency, four statuses, retry policy, CMD drain/retry commands, and MySQL final filtering guarantee.
- [ ] Run CMD: `.venv\Scripts\python.exe -m pytest -q`, then `git diff --check` and `git status --short`; do not stage prohibited files.
- [ ] With existing private configuration, run migration, add memory, drain, deactivate, drain; simulate Milvus outage and verify task persists, then service recovery drains it.
- [ ] Stage only explicit implementation, migration, tests, README, and plan files; commit `feat: add outbox memory index sync`.

## Plan Self-Review

- Tasks cover schema, atomic commit, idempotent delete, lease claim, retry/failure/replay, worker, CLI, documentation, offline tests, and real validation.
- Function and state names are consistent across tasks; scope excludes unrelated infrastructure.
