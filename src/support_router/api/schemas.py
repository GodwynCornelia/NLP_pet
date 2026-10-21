from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr
    top_k: StrictInt = Field(default=3, ge=1, le=5)
    include_similar: StrictBool = True

    @field_validator("text")
    @classmethod
    def valid_text(cls, value):
        value = value.strip()
        if not 1 <= len(value) <= 4000:
            raise ValueError("text must have 1..4000 characters after strip")
        return value


class BatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StrictStr = Field(min_length=1, max_length=200)
    text: Any = None  # Invalid row contents are item-level errors, not whole-batch errors.


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[BatchItem] = Field(min_length=1, max_length=100)
    top_k: StrictInt = Field(default=3, ge=1, le=5)
    include_similar: StrictBool = False

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("Batch IDs must be unique")
        return self


class Prediction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    category: str
    score: float = Field(ge=0, le=1)


class SimilarExample(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    sample_id: str
    text: str
    category: str
    similarity: float = Field(ge=-1, le=1)
    exact_match: bool


class PredictResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    request_id: str
    run_id: str
    predicted_category: str
    confidence: float = Field(ge=0, le=1)
    top_predictions: list[Prediction]
    routing_decision: Literal["auto_route", "manual_review"]
    review_reason: Literal["low_confidence", "auto_routing_disabled"] | None
    threshold: float | None
    similar_examples: list[SimilarExample]
    warnings: list[str]
    latency_ms: float = Field(ge=0)


class BatchResponseItem(BaseModel):
    id: str
    result: PredictResponse | None
    error: dict[str, str] | None

    @model_validator(mode="after")
    def exactly_one(self):
        if (self.result is None) == (self.error is None):
            raise ValueError("Exactly one result/error must be populated")
        return self
