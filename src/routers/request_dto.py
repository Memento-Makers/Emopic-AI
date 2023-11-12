from pydantic import BaseModel

class InferenceRequest(BaseModel):
    url: str
    photo_id: int