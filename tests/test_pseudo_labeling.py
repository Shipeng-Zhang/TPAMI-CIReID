import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from reid.pseudo_labeling import (
    compute_identity_prototypes,
    generate_pseudo_labels,
    mutual_nearest_assignment,
)


class _FeatureDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([
            [1.0, 0.0], [0.9, 0.1],
            [0.0, 1.0], [0.1, 0.9],
            [-1.0, 0.0],
        ])
        self.labels = [0, 0, 1, 1, 2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], str(index), self.labels[index], 0, 0


class _IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        return (features + self.anchor * 0,)


class PseudoLabelingTest(unittest.TestCase):
    def test_only_mutual_nearest_identities_reuse_history(self):
        current = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ])
        memory = {
            4: {"mean_features": torch.tensor([[1.0, 0.0]])},
            9: {"mean_features": torch.tensor([[0.0, 1.0]])},
        }

        assigned, matched, _ = mutual_nearest_assignment(current, memory)

        self.assertEqual(assigned.tolist(), [4, 9, 10])
        self.assertEqual(matched, [0, 1])

    def test_local_identity_prototypes_are_normalized(self):
        features = torch.tensor([[2.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
        local_labels = torch.tensor([4, 4, 8])

        prototypes, identities = compute_identity_prototypes(
            features, local_labels)

        self.assertEqual(identities.tolist(), [4, 8])
        self.assertTrue(torch.allclose(prototypes.norm(dim=1), torch.ones(2)))

    def test_image_labels_follow_local_identity_assignment(self):
        memory = {
            5: {"mean_features": torch.tensor([[1.0, 0.0]])},
            8: {"mean_features": torch.tensor([[0.0, 1.0]])},
        }
        result = generate_pseudo_labels(
            _IdentityModel(), DataLoader(_FeatureDataset(), batch_size=2), memory)

        self.assertEqual(result.num_local_identities, 3)
        self.assertEqual(result.num_matched, 2)
        self.assertEqual(result.num_new, 1)
        self.assertEqual(
            [result.labels_by_filename[str(index)] for index in range(5)],
            [5, 5, 8, 8, 9],
        )


if __name__ == "__main__":
    unittest.main()
