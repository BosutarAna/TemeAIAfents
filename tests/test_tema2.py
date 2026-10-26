"""Test pentru Tema 2 - Agent simplu"""

from teme.tema2 import search_keyword, list_files


def test_list_files_returns_output():
    """Verifica ca list_files returneaza ceva (nu e gol)."""
    result = list_files()
    assert result is not None
    assert len(result) > 0, "list_files ar trebui sa returneze lista de fisiere"


def test_search_keyword_finds_pytest():
    """Verifica ca unealta de cautare gaseste 'pytest' in fisierele asteptate."""
    result = search_keyword("pytest")

    # Verificam ca rezultatul contine test_tema2.py
    # (stim ca pytest apare in acest fisier - chiar in linia de import)
    assert "test_tema2" in result or "tema2" in result, (
        f"Fisierul cu pytest nu a fost gasit. Output: {result}"
    )
