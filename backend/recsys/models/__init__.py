"""
Recommender models package.
"""
from __future__ import annotations

from recsys.models.als import ALSRecommender
from recsys.models.base import BaseRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender

__all__ = [
    "ALSRecommender",
    "BaseRecommender",
    "ContentRecommender",
    "ItemKNNRecommender",
    "PopularityRecommender",
]


