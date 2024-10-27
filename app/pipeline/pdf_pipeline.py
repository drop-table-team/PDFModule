from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from app.models.schemas import DocumentMetadata
from app.util.llm_provider import LLMProvider


class PDFAnalysisPipeline:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.embeddings = OllamaEmbeddings(
            base_url=llm_provider.llm.base_url, model=llm_provider.llm.model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    async def process_document(self, pages: List[Document]) -> DocumentMetadata:
        """
        Process a PDF document using embeddings for better analysis of larger documents.
        Returns extended metadata including title and both full and short summaries.
        """
        all_text_chunks = []
        for page in pages:
            all_text_chunks.extend(self.text_splitter.split_text(page.page_content))

        vectorstore = FAISS.from_texts(texts=all_text_chunks, embedding=self.embeddings)

        # Generate all metadata in parallel for better performance
        title = await self._generate_title(vectorstore, pages[0].page_content)
        summary = await self._generate_summary(vectorstore, pages[0].page_content)
        short_summary = await self._generate_short_summary(
            vectorstore, pages[0].page_content
        )
        tags = await self._generate_tags(vectorstore, pages[0].page_content)

        return DocumentMetadata(
            title=title, summary=summary, short_summary=short_summary, tags=tags
        )

    async def _generate_title(self, vectorstore: FAISS, example_text: str) -> str:
        """
        Generate a descriptive title for the document.
        """
        query = "Was ist der Haupttitel oder das Hauptthema dieses Dokuments?"
        relevant_chunks = vectorstore.similarity_search(query, k=2)
        combined_text = "\n".join(doc.page_content for doc in relevant_chunks)

        prompt = f"""
        Du bist ein erfahrener Dokumentenanalytiker. Analysiere den folgenden Textauszug und erzeuge einen prägnanten und relevanten Titel für das Dokument.

        Textauszug:
        {combined_text}
        
        # Ausgabeanforderungen
        - Gib nur den Titel zurück, bestehend aus maximal 10 Wörtern.
        - Der Titel muss ohne Einleitung, zusätzliche Texte oder Erklärungen sein.
        """

        return await self.llm_provider.generate(prompt)

    async def _generate_summary(self, vectorstore: FAISS, example_text: str) -> str:
        """
        Generate a detailed summary identifying the document type and its main points.
        """
        query = "Was sind die wichtigsten Inhalte und Hauptpunkte des Dokuments?"
        relevant_chunks = vectorstore.similarity_search(query, k=5)
        combined_text = "\n".join(doc.page_content for doc in relevant_chunks)

        prompt = f"""
        Du bist ein Experte für Inhaltszusammenfassungen. Erstelle auf Basis der folgenden Textabschnitte eine umfassende Zusammenfassung des Dokuments.

        Textauszüge:
        {combined_text}
        
        # Ausgabeanforderungen
        - Gib eine ausführliche Zusammenfassung im Fließtext zurück (200-300 Wörter).
        - Die Zusammenfassung soll den Dokumenttyp, Hauptthemen und Schlüsselpunkte identifizieren und wichtige Ergebnisse zusammenfassen.
        - Gib nur die Zusammenfassung ohne zusätzliche Erklärungen oder Einleitungen zurück.
        """

        return await self.llm_provider.generate(prompt)

    async def _generate_short_summary(
        self, vectorstore: FAISS, example_text: str
    ) -> str:
        """
        Generate a concise summary (2-3 sentences) of the document's key points.
        """
        query = "Was sind die absolut wichtigsten Kernaussagen?"
        relevant_chunks = vectorstore.similarity_search(query, k=3)
        combined_text = "\n".join(doc.page_content for doc in relevant_chunks)

        prompt = f"""
        Erstelle eine kurze Zusammenfassung der zentralen Inhalte auf Basis der folgenden Textauszüge.

        Textauszüge:
        {combined_text}
        
        # Ausgabeanforderungen
        - Fasse die zentralen Inhalte in 2-3 Sätzen zusammen (maximal 50 Wörter).
        - Gib nur die Zusammenfassung zurück, ohne Einleitungen, Anmerkungen oder weitere Erklärungen.
        """

        return await self.llm_provider.generate(prompt)

    async def _generate_tags(self, vectorstore: FAISS, example_text: str) -> List[str]:
        """
        Generate relevant tags based on document content.
        """
        query = "Was sind die Hauptthemen und -inhalte?"
        relevant_chunks = vectorstore.similarity_search(query, k=3)
        combined_text = "\n".join([doc.page_content for doc in relevant_chunks])

        prompt = f"""
        Basierend auf den folgenden Textabschnitten, erzeuge eine Liste relevanter Schlagwörter, die die Hauptthemen und Inhalte des Dokuments beschreiben.

        Textauszüge:
        {combined_text}
        
        # Ausgabeanforderungen
        - Gib nur die Schlagwörter als durch Kommas getrennte Liste zurück (5-8 Schlagwörter).
        - Die Ausgabe soll nur die Schlagwörter enthalten, ohne Einleitungen, Erklärungen oder andere Texte.
        """

        tags_text = await self.llm_provider.generate(prompt)
        tags_text = tags_text.split("\n")[-1]  # Only take the last part of the output
        return [tag.strip() for tag in tags_text.split(",")]
