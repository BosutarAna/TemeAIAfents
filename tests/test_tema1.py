"""Teste pentru Tema 1 - Citire JSON si validare Pydantic"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from teme.tema1_student_validator import StudentValidator, citeste_student


class TestStudentValidator:
    def test_student_valid(self):
        student = StudentValidator(
            nume="Ana Popescu",
            varsta=20,
            email="ana.popescu@example.com",
        )
        assert student.nume == "Ana Popescu"
        assert student.varsta == 20
        assert student.email == "ana.popescu@example.com"

    def test_varsta_negativa(self):
        with pytest.raises(ValidationError):
            StudentValidator(nume="Test", varsta=-1, email="test@example.com")

    def test_varsta_zero(self):
        with pytest.raises(ValidationError):
            StudentValidator(nume="Test", varsta=0, email="test@example.com")

    def test_email_invalid(self):
        with pytest.raises(ValidationError):
            StudentValidator(nume="Test", varsta=20, email="not-an-email")

    def test_camp_lipsa(self):
        with pytest.raises(ValidationError):
            StudentValidator(nume="Test", varsta=20)


class TestCitesteStudent:
    def test_citeste_din_fisier(self, tmp_path: Path):
        date = {"nume": "Ion Ionescu", "varsta": 22, "email": "ion@example.com"}
        fisier = tmp_path / "student.json"
        fisier.write_text(json.dumps(date))

        student = citeste_student(fisier)

        assert student.nume == "Ion Ionescu"
        assert student.varsta == 22
        assert student.email == "ion@example.com"

    def test_fisier_cu_date_invalide(self, tmp_path: Path):
        date = {"nume": "Test", "varsta": -5, "email": "bad"}
        fisier = tmp_path / "student.json"
        fisier.write_text(json.dumps(date))

        with pytest.raises(ValidationError):
            citeste_student(fisier)

    def test_student_json_din_proiect(self):
        fisier = Path(__file__).resolve().parent.parent / "student.json"
        student = citeste_student(fisier)

        assert student.nume == "Ana Popescu"
        assert student.varsta == 20
        assert student.email == "ana.popescu@example.com"