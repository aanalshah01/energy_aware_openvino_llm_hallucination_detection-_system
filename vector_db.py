import chromadb
from chromadb.utils import embedding_functions

class KnowledgeBaseManager:
    def __init__(self, db_path="./chroma_db", collection_name="factual_evidence"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )

    def populate_default_knowledge(self):
        """Seed DB with reference documents if empty."""
        if self.collection.count() == 0:
            documents = [
                "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity.",
                "Marie Curie won two Nobel Prizes: Physics in 1903 and Chemistry in 1911.",
                "Albert Einstein developed the theory of relativity and won the Nobel Prize in Physics in 1921.",
                "Isaac Newton formulated the laws of motion and universal gravitation in Philosophiae Naturalis Principia Mathematica.",
                "The Apollo 11 mission was the spaceflight that first landed humans on the Moon in July 1969.",
                "Neil Armstrong was the first person to walk on the Moon."
            ]
            ids = [f"doc_{i}" for i in range(len(documents))]
            self.collection.add(documents=documents, ids=ids)
            print(f"Populated vector DB with {len(documents)} core documents.")

    def add_documents(self, documents: list[str]):
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        self.collection.add(documents=documents, ids=ids)

    def search_evidence(self, claim: str, n_results: int = 2) -> list[str]:
        """Retrieve top matching ground-truth sentences for a claim."""
        results = self.collection.query(query_texts=[claim], n_results=n_results)
        if results and "documents" in results and len(results["documents"]) > 0:
            return results["documents"][0]
        return []

if __name__ == "__main__":
    kb = KnowledgeBaseManager()
    kb.populate_default_knowledge()
    ev = kb.search_evidence("Who won the Nobel Prize in 1911?")
    print("Retrieved Evidence:", ev)