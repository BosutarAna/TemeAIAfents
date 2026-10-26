"""
Tests for Tema 3 - RAG Search

These tests verify that the RAG pipeline works correctly:
1. Chunking splits text properly
2. Files are loaded with correct metadata
3. Search filtering by source works
"""

import pytest
from pathlib import Path

# Import functions from tema3
from teme.tema3_RAG_search import (
    chunk_text,
    load_and_chunk_files,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


class TestChunking:
    """Tests for the chunk_text function"""

    def test_chunk_text_creates_chunks(self):
        """Verify that chunk_text splits text into pieces"""
        text = "A" * 1000  # 1000 characters
        chunks = chunk_text(text, chunk_size=500, overlap=100)

        # Should create multiple chunks
        assert len(chunks) > 1, "Should create more than one chunk for long text"

    def test_chunk_text_returns_dict_with_required_keys(self):
        """Verify each chunk has text, page, and chunk_index"""
        text = "This is a sample text for testing chunking functionality."
        chunks = chunk_text(text, chunk_size=500, overlap=100)

        assert len(chunks) > 0, "Should create at least one chunk"

        # Check first chunk has all required keys
        chunk = chunks[0]
        assert "text" in chunk, "Chunk should have 'text' key"
        assert "page" in chunk, "Chunk should have 'page' key"
        assert "chunk_index" in chunk, "Chunk should have 'chunk_index' key"

    def test_chunk_text_preserves_content(self):
        """Verify that chunked text contains original content"""
        original_text = "Hello world this is a test"
        chunks = chunk_text(original_text, chunk_size=500, overlap=100)

        # The text should be in the first chunk
        assert "Hello" in chunks[0]["text"], "Chunk should contain original text"

    def test_chunk_index_increments(self):
        """Verify chunk_index increases for each chunk"""
        text = "A" * 2000  # Long text to create multiple chunks
        chunks = chunk_text(text, chunk_size=500, overlap=100)

        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i, f"Chunk index should be {i}"

    def test_empty_text_returns_empty_list(self):
        """Verify empty text returns no chunks"""
        chunks = chunk_text("", chunk_size=500, overlap=100)
        assert chunks == [], "Empty text should return empty list"

    def test_overlap_creates_more_chunks(self):
        """Verify overlap creates overlapping chunks"""
        text = "A" * 1000

        # Without overlap
        chunks_no_overlap = chunk_text(text, chunk_size=500, overlap=0)

        # With overlap
        chunks_with_overlap = chunk_text(text, chunk_size=500, overlap=100)

        # Overlap should create more chunks (because we step less each time)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)


class TestLoadFiles:
    """Tests for file loading functionality"""

    def test_load_and_chunk_files_adds_source_metadata(self, tmp_path):
        """Verify that loaded chunks have 'source' metadata with filename"""
        # Create a test file
        test_file = tmp_path / "test_document.txt"
        test_file.write_text("This is test content for the document.")

        # Load and chunk
        chunks = load_and_chunk_files(tmp_path)

        assert len(chunks) > 0, "Should create at least one chunk"

        # Each chunk should have source = filename
        for chunk in chunks:
            assert "source" in chunk, "Chunk should have 'source' key"
            assert chunk["source"] == "test_document.txt", "Source should be filename"

    def test_load_multiple_files_different_sources(self, tmp_path):
        """Verify multiple files get different source metadata"""
        # Create two test files
        file1 = tmp_path / "document1.txt"
        file1.write_text("Content of first document.")

        file2 = tmp_path / "document2.txt"
        file2.write_text("Content of second document.")

        # Load and chunk
        chunks = load_and_chunk_files(tmp_path)

        # Get unique sources
        sources = set(chunk["source"] for chunk in chunks)

        assert "document1.txt" in sources, "Should have document1.txt as source"
        assert "document2.txt" in sources, "Should have document2.txt as source"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """Verify empty directory returns no chunks"""
        chunks = load_and_chunk_files(tmp_path)
        assert chunks == [], "Empty directory should return empty list"

    def test_ignores_unsupported_file_types(self, tmp_path):
        """Verify unsupported file types are skipped"""
        # Create unsupported file
        unsupported = tmp_path / "image.jpg"
        unsupported.write_text("fake image content")

        # Create supported file
        supported = tmp_path / "document.txt"
        supported.write_text("Real text content.")

        chunks = load_and_chunk_files(tmp_path)

        # Should only have chunks from .txt file
        sources = set(chunk["source"] for chunk in chunks)
        assert "document.txt" in sources
        assert "image.jpg" not in sources


class TestMetadata:
    """Tests for metadata handling"""

    def test_page_number_estimation(self):
        """Verify page numbers are estimated based on position"""
        # Create text that spans multiple "pages" (assuming 2000 chars/page)
        text = "A" * 5000  # Should be about 2-3 pages
        chunks = chunk_text(text, chunk_size=500, overlap=100)

        # First chunks should be page 1
        assert chunks[0]["page"] == 1

        # Later chunks should have higher page numbers
        pages = [c["page"] for c in chunks]
        assert max(pages) > 1, "Long text should have multiple pages"

    def test_explicit_page_markers_detected(self):
        """Verify 'Pagina X:' markers are detected"""
        text = "Pagina 5: This is content on page 5."
        chunks = chunk_text(text, chunk_size=500, overlap=100)

        # Should detect page 5 from marker
        assert chunks[0]["page"] == 5, "Should detect page number from 'Pagina X:' marker"
