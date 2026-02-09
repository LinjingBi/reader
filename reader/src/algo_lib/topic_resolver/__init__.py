"""
Topic resolver component for algo_lib.
"""

# DO NOT REMOVE THIS VERSION LINE, only bump when you make a change to the topic_resolver code.
__version__ = "0.1.0"

from algo_lib.topic_resolver.models import (
    ClusterInput,
    TopicInput,
    TopicResolveAction,
    TopicResolveOutput,
)
from algo_lib.topic_resolver.resolver import resolve_topic

__all__ = [
    "resolve_topic",
    "TopicResolveAction",
    "TopicInput",
    "ClusterInput",
    "TopicResolveOutput",
]

