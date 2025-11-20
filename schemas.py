"""
Database Schemas for Budgeting App

Each Pydantic model represents a MongoDB collection.
Collection name is the lowercase of the class name.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date

class Budget(BaseModel):
    """
    Monthly budget configuration
    Collection: "budget"
    """
    month: str = Field(..., description="YYYY-MM month key")
    limit_amount: float = Field(..., ge=0, description="Monthly spending limit in currency units")

class Transaction(BaseModel):
    """
    Expense transaction record
    Collection: "transaction"
    """
    date: date = Field(..., description="Transaction date")
    amount: float = Field(..., description="Amount spent (positive value)")
    merchant: str = Field(..., description="Merchant or payee")
    category: Optional[str] = Field(None, description="Category if available")
    source: Literal["gpay", "statement", "manual"] = Field("manual", description="Source of the transaction")
    note: Optional[str] = Field(None, description="Optional note")

class WishlistItem(BaseModel):
    """
    Wishlist item the user plans to buy
    Collection: "wishlistitem"
    """
    name: str = Field(..., description="Item name")
    estimated_price: float = Field(..., ge=0, description="Estimated price")
    priority: Literal["High", "Medium", "Low"] = Field("Medium", description="Priority")
    deadline_month: Optional[str] = Field(None, description="Optional YYYY-MM month to target purchase")
    purchased: bool = Field(False, description="Whether this item has been purchased")

class Setting(BaseModel):
    """
    App-level settings (placeholder)
    Collection: "setting"
    """
    key: str
    value: str
