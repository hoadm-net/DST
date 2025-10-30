from typing import Dict, Set

# Minimal default ontology for closed slots commonly used in MultiWOZ
# This is not exhaustive but covers frequent cases for a simple baseline.
default_ontology: Dict[str, Set[str]] = {
    # Restaurant
    "restaurant-pricerange": {"cheap", "moderate", "expensive", "dontcare"},
    "restaurant-area": {"centre", "north", "south", "east", "west", "dontcare"},
    # Hotel
    "hotel-pricerange": {"cheap", "moderate", "expensive", "dontcare"},
    "hotel-area": {"centre", "north", "south", "east", "west", "dontcare"},
    "hotel-parking": {"yes", "no", "dontcare"},
    "hotel-internet": {"yes", "no", "dontcare"},
    "hotel-stars": {"0", "1", "2", "3", "4", "5", "dontcare"},
    # Train
    "train-day": {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "dontcare"},
}

class Ontology:
    def __init__(self, mapping: Dict[str, Set[str]] | None = None) -> None:
        self._map: Dict[str, Set[str]] = dict(mapping or default_ontology)

    def is_closed(self, slot: str) -> bool:
        return slot in self._map

    def values(self, slot: str) -> Set[str]:
        return self._map.get(slot, set())

    def add_values(self, slot: str, values: Set[str]) -> None:
        self._map.setdefault(slot, set()).update(values)
