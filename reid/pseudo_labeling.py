"""Global pseudo labels for locally annotated camera-incremental ReID."""

from __future__ import absolute_import

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class PseudoLabelResult:
    labels_by_filename: dict
    num_local_identities: int
    num_matched: int
    num_new: int
    mean_matched_similarity: float


def compute_identity_prototypes(features, local_labels):
    """Average image features according to known camera-local identities."""
    if features.ndim != 2 or features.size(0) == 0:
        raise ValueError("features must be a non-empty [N, D] tensor")
    if features.size(0) != local_labels.numel():
        raise ValueError("features and local_labels must have equal length")

    identities = torch.unique(local_labels, sorted=True)
    prototypes = []
    for label in identities:
        mask = local_labels == label
        prototypes.append(features[mask].mean(dim=0))
    return F.normalize(torch.stack(prototypes), dim=1), identities


def mutual_nearest_assignment(current_prototypes, historical_memory):
    """Map local identities to global IDs through bidirectional matching.

    A local identity receives a historical label only when both sides select
    each other as their cosine nearest neighbour.  Every other identity gets a
    new, globally unique and contiguous label.
    """
    num_current = current_prototypes.size(0)
    if num_current == 0:
        raise ValueError("current_prototypes must not be empty")

    if not historical_memory:
        return torch.arange(num_current, dtype=torch.long), [], []

    historical_labels = sorted(int(label) for label in historical_memory)
    historical_prototypes = torch.cat(
        [historical_memory[label]["mean_features"] for label in historical_labels],
        dim=0,
    ).to(current_prototypes.device)
    historical_prototypes = F.normalize(historical_prototypes, dim=1)
    current_prototypes = F.normalize(current_prototypes, dim=1)

    similarities = current_prototypes.mm(historical_prototypes.t())
    nearest_history = similarities.argmax(dim=1)
    nearest_current = similarities.argmax(dim=0)

    assigned = torch.full((num_current,), -1, dtype=torch.long)
    matched_identities = []
    matched_similarities = []
    for current_index, history_index in enumerate(nearest_history.tolist()):
        if int(nearest_current[history_index]) != current_index:
            continue
        assigned[current_index] = historical_labels[history_index]
        matched_identities.append(current_index)
        matched_similarities.append(float(similarities[current_index, history_index]))

    next_label = max(historical_labels) + 1
    for current_index in range(num_current):
        if assigned[current_index] == -1:
            assigned[current_index] = next_label
            next_label += 1

    return assigned, matched_identities, matched_similarities


def _extract_features(model, data_loader):
    """Extract deterministic mean features while preserving model mode."""
    features = []
    filenames = []
    local_labels = []
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    try:
        with torch.no_grad():
            for images, batch_filenames, batch_labels, _, _ in data_loader:
                outputs = model(images.to(device, non_blocking=True))
                features.append(outputs[0].detach())
                filenames.extend(list(batch_filenames))
                local_labels.extend(batch_labels.tolist())
    finally:
        model.train(was_training)

    if not features:
        raise ValueError("cannot generate pseudo labels for an empty camera")
    features = F.normalize(torch.cat(features, dim=0), dim=1)
    labels = torch.tensor(local_labels, device=features.device, dtype=torch.long)
    return features, labels, filenames


def generate_pseudo_labels(model, data_loader, historical_memory):
    """Associate known local identities with the historical identity space."""
    features, local_labels, filenames = _extract_features(model, data_loader)
    current_prototypes, local_identities = compute_identity_prototypes(
        features, local_labels)
    global_identity_labels, matched_identities, matched_similarities = (
        mutual_nearest_assignment(current_prototypes, historical_memory)
    )
    local_to_global = dict(zip(
        local_identities.cpu().tolist(),
        global_identity_labels.tolist(),
    ))
    image_labels = [local_to_global[int(label)]
                    for label in local_labels.cpu().tolist()]

    labels_by_filename = dict(zip(filenames, image_labels))
    if len(labels_by_filename) != len(filenames):
        raise ValueError("duplicate filenames found while assigning pseudo labels")

    num_identities = current_prototypes.size(0)
    mean_similarity = (sum(matched_similarities) / len(matched_similarities)
                       if matched_similarities else 0.0)
    return PseudoLabelResult(
        labels_by_filename=labels_by_filename,
        num_local_identities=num_identities,
        num_matched=len(matched_identities),
        num_new=num_identities - len(matched_identities),
        mean_matched_similarity=mean_similarity,
    )
