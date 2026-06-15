"""Handles request validation"""
from pydantic import BaseModel
from typing import List


class UserRequest(BaseModel):
    """guides input format"""
    prompt: str 


class ChatResponse(BaseModel):
    """guides output format"""
    response: str
    sources: list