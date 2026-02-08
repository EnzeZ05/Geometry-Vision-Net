GVNet is a ResNet-12 episodic few-shot model that predicts multiple polygon “planes” per image, renders each plane into a soft point-in-polygon mask, then merges them with a soft-union to get a single shape signature per sample. It classifies queries by a hybrid logit = α · soft-IoU between merged masks, plus regularizers that push class centroids apart and discourage plane overlap to ensure the boundaries of each object.

# For some reason, the multiplane approach would always shrink into single dot after many epoches, I don't know why. A failed innovation.
