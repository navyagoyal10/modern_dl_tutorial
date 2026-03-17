"""Multimodal EuroSAT dataset skeleton for A12."""

from __future__ import annotations

from torch.utils.data import Dataset


DESCRIPTIONS = {
    "AnnualCrop": "Aerial view of agricultural fields with annual crop rows",
    "Forest": "Dense tree canopy in a satellite view",
    "HerbaceousVegetation": "Low vegetation and grass in open fields",
    "Highway": "A paved motorway seen from above",
    "Industrial": "Industrial facilities and large buildings",
    "Pasture": "Open grassland used for grazing",
    "PermanentCrop": "Orchards or vineyards from aerial view",
    "Residential": "Residential houses and streets",
    "River": "A river flowing through the terrain",
    "SeaLake": "A sea, lake, or reservoir body",
}


class MultimodalEuroSAT(Dataset):
    """Dataset returning image tensors and tokenized class text."""

    def __init__(
        self,
        image_paths,
        labels,
        class_names,
        image_transform,
        tokenizer,
        max_text_len: int = 64,
    ) -> None:
        """Store data references, transform, tokenizer, and text length."""
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = image_transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Return multimodal sample dict for one index.

        Returns keys:
            image, input_ids, attention_mask, label
        """
        # TODO: load image and apply image transform.
        # TODO: map class label to text description.
        # TODO: tokenize text into input_ids and attention_mask.
        # TODO: return dict with image/text tensors and integer label.
        raise NotImplementedError("Implement MultimodalEuroSAT.__getitem__.")
