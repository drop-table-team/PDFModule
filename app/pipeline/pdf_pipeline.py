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
        Basierend auf diesem Textauszug:
        {combined_text}
        
        Generiere einen prägnanten, aussagekräftigen Titel für das Dokument (maximal 10 Wörter).
        Gib nur den Titel zurück, ohne zusätzliche Erklärungen."""

        return await self.llm_provider.generate(prompt)

    async def _generate_summary(self, vectorstore: FAISS, example_text: str) -> str:
        """
        Generate a detailed summary identifying the document type and its main points.
        """
        query = "Was sind die wichtigsten Inhalte und Hauptpunkte des Dokuments?"
        relevant_chunks = vectorstore.similarity_search(query, k=5)
        combined_text = "\n".join(doc.page_content for doc in relevant_chunks)

        prompt = f"""
        Basierend auf den folgenden relevanten Abschnitten des Dokuments:
        {combined_text}
        
        Erstelle eine ausführliche Zusammenfassung (ca. 200-300 Wörter), die:
        1. Den Dokumenttyp identifiziert
        2. Die Hauptthemen und wichtigsten Punkte detailliert beschreibt
        3. Die wichtigsten Schlussfolgerungen oder Ergebnisse zusammenfasst
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
        Basierend auf diesem Textauszug:
        {combined_text}
        
        Fasse die wichtigsten Kernaussagen in 2-3 prägnanten Sätzen zusammen.
        Die Zusammenfassung sollte maximal 50 Wörter umfassen."""

        return await self.llm_provider.generate(prompt)

    async def _generate_tags(self, vectorstore: FAISS, example_text: str) -> List[str]:
        """
        Generate relevant tags based on document content.
        """
        query = "Was sind die Hauptthemen und -inhalte?"
        relevant_chunks = vectorstore.similarity_search(query, k=3)
        combined_text = "\n".join([doc.page_content for doc in relevant_chunks])

        prompt = f"""Basierend auf diesen Abschnitten des Dokuments:
        {combined_text}
        
        Generiere eine Liste von 5-8 relevanten Schlagwörtern, die die Hauptthemen, Inhalte und den Typ des Dokuments beschreiben.
        Gib nur die Schlagwörter getrennt durch Kommas zurück."""

        tags_text = await self.llm_provider.generate(prompt)
        return [tag.strip() for tag in tags_text.split(",")]
