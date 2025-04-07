from pydantic import BaseModel, Field


class HelloRequestModel(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., gt=0)
