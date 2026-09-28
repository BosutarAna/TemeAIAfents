import json
from pathlib import Path
from pydantic import BaseModel, EmailStr, PositiveInt


class StudentValidator(BaseModel):
    nume: str
    varsta: PositiveInt
    email: EmailStr

def citeste_student(cale_fisier: str|Path) -> StudentValidator:
    with open(cale_fisier) as f:
        date = json.load(f)
    return StudentValidator(**date)

if __name__ == "__main__":
    student = citeste_student("../student.json")
    print(f"student Valid: {student}")