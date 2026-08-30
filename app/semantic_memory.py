class SemanticLongTermMemoryService:
    def __init__(
        self,
        embeddings,
        vector_index,
        long_term_memory_repository,
    ):
        self.embeddings = embeddings
        self.vector_index = vector_index
        self.long_term_memory_repository = (
            long_term_memory_repository
        )

    def search_active(
        self,
        user_id: str,
        question: str,
        limit: int = 3,
    ):
        vector = self.embeddings.embed_query(question)

        memory_ids = self.vector_index.search(
            user_id=user_id,
            vector=vector,
            limit=limit,
        )

        memories = (
            self.long_term_memory_repository.list_active_by_ids(
                user_id,
                memory_ids,
            )
        )

        memories_by_id = {
            memory.id: memory
            for memory in memories
        }

        return [
            memories_by_id[memory_id]
            for memory_id in memory_ids
            if memory_id in memories_by_id
        ]