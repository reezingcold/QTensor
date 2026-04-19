from __future__ import annotations
from dataclasses import dataclass, field
import itertools

# global counter to give each Index a unique identity
_index_uid_counter = itertools.count()


@dataclass(frozen=True, slots=True)
class Index:
    dim: int
    name: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    prime_level: int = 0
    uid: int = field(default_factory=lambda: next(_index_uid_counter), compare=False)

    
    def __post_init__(self):
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError(f"Index dim must be a positive integer, got {self.dim}")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Index name must be a non-empty string")
        if not isinstance(self.tags, tuple):
            raise TypeError("Index tags must be a tuple of strings")
        if not all(isinstance(tag, str) for tag in self.tags):
            raise TypeError("All tags must be strings")
        if not isinstance(self.prime_level, int) or self.prime_level < 0:
            raise ValueError("prime_level must be a non-negative integer")
    
    # unique hash based on identity
    def __hash__(self) -> int:
        return hash((self.uid, self.prime_level))
        
    def prime(self, n: int = 1) -> Index:
        if n < 0:
            raise ValueError("n must be non-negative")
        
        return Index(dim=self.dim, name=self.name, tags=self.tags, prime_level=self.prime_level + n, uid=self.uid)

    def unprime(self, n: int = 1) -> Index:
        if n < 0:
            raise ValueError("n must be non-negative")
        if self.prime_level < n:
            raise ValueError("Cannot unprime below zero")

        return Index(dim=self.dim, name=self.name, tags=self.tags, prime_level=self.prime_level - n, uid=self.uid)
    
    def rename(self, new_name: str) -> Index:
        if not isinstance(new_name, str) or not new_name:
            raise ValueError("new_name must be a non-empty string")

        return Index(dim=self.dim, name=new_name, tags=self.tags, prime_level=self.prime_level)

    def add_tags(self, *new_tags: str) -> Index:
        if not all(isinstance(tag, str) for tag in new_tags):
            raise TypeError("All tags must be strings")

        return Index(dim=self.dim, name=self.name, tags=self.tags + tuple(new_tags), prime_level=self.prime_level)

    def __repr__(self) -> str:
        prime_marks = "'" * self.prime_level
        # tag_str = f", tags={self.tags}" if self.tags else ""
        # return f"Index({self.name}{prime_marks}, dim={self.dim}, uid={self.uid}{tag_str})"
        return f"({self.name}{prime_marks}|dim={self.dim}|uid={self.uid})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return self.uid == other.uid and self.prime_level == other.prime_level

    def matches(self, other: "Index") -> bool:
        """Return True if two indices have the same structural metadata."""
        if not isinstance(other, Index):
            return False
        return (
            self.dim == other.dim
            and self.name == other.name
            and self.tags == other.tags
            and self.prime_level == other.prime_level
        )