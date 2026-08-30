class MemorySyncService:
    def __init__(self, embeddings, vector_index):
        self.embeddings = embeddings
        self.vector_index = vector_index

    def sync(self, memory):
        vector = self.embeddings.embed_query(memory.content)

        self.vector_index.upsert(
            memory_id=memory.id,
            user_id=memory.user_id,
            vector=vector,
        )