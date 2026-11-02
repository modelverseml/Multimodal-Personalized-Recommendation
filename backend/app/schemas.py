from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    role: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    asin: str
    title: str
    category: str
    subcategory: str
    store: str | None
    price: float | None
    average_rating: float | None
    rating_number: int
    image_url: str | None


class ProductDetail(ProductOut):
    description: str
    features: str


class ProductPage(BaseModel):
    items: list[ProductOut]
    total: int
    page: int
    page_size: int


class ScoredProduct(BaseModel):
    product: ProductOut
    score: float


class ProductWithSimilar(BaseModel):
    product: ProductDetail
    similar: list[ScoredProduct]


class SearchResult(BaseModel):
    query: str
    latency_ms: float
    items: list[ScoredProduct]


class CustomerOut(BaseModel):
    id: int
    external_id: str
    display_name: str
    source: str
    interaction_count: int
    top_category: str | None = None


class HistoryEvent(BaseModel):
    product: ProductOut
    event_type: str
    rating: float | None
    created_at: datetime


class CustomerDetail(CustomerOut):
    history: list[HistoryEvent]
    category_mix: dict[str, float]


class CustomerCreate(BaseModel):
    display_name: str | None = Field(default=None, max_length=128)


class Reason(BaseModel):
    feature: str
    label: str
    contribution: float | None


class Recommendation(BaseModel):
    position: int
    product: ProductOut
    score: float
    retrieval_rank: int | None
    reasons: list[Reason]
    anchor: ProductOut | None


class RecommendationResponse(BaseModel):
    request_id: str
    customer_id: int
    model_version: str
    strategy: str
    history_length: int
    timings: dict[str, float]
    items: list[Recommendation]


class EventCreate(BaseModel):
    customer_id: int
    product_id: int
    event_type: Literal["view", "click", "add_to_cart", "purchase"]
    request_id: str | None = None


class EventOut(BaseModel):
    id: int
    customer_id: int
    product_id: int
    event_type: str
    created_at: datetime
