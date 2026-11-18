import importlib.util
import unittest

from phase2.graph.data import to_homogeneous_graph


class GraphTrainingTests(unittest.TestCase):
    def test_homogeneous_conversion_preserves_edge_types(self):
        class Hetero:
            def to_homogeneous(self, **kwargs):
                self.options = kwargs
                return object()

        data = Hetero()
        to_homogeneous_graph(data)
        self.assertTrue(data.options["add_edge_type"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_training_uses_unweighted_cross_entropy_per_batch(self):
        import torch
        import torch.nn.functional as F
        from phase2.graph.engine import train_one_epoch

        class Batch:
            def __init__(self, x, y):
                self.x, self.y = x, y

            def to(self, device):
                return self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)

            def forward(self, batch):
                return self.linear(batch.x)

        torch.manual_seed(1)
        model, reference = Model(), Model()
        reference.load_state_dict(model.state_dict())
        batches = [Batch(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1])),
                   Batch(torch.tensor([[2., 0.]]), torch.tensor([1]))]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        expected_loss = 0.0
        for batch in batches:
            expected_optimizer.zero_grad()
            loss = F.cross_entropy(reference(batch), batch.y)
            loss.backward()
            expected_optimizer.step()
            expected_loss += loss.item() * batch.y.numel()
        actual_loss = train_one_epoch(model, batches, optimizer, "cpu")
        self.assertAlmostEqual(actual_loss, expected_loss / 3, places=6)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_weighted_cross_entropy_uses_sample_count_even_for_single_class_batches(self):
        import torch
        import torch.nn.functional as F
        from phase2.graph.engine import train_one_epoch

        class Batch:
            def __init__(self, x, y):
                self.x, self.y = x, y

            def to(self, device):
                return self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)

            def forward(self, batch):
                return self.linear(batch.x)

        torch.manual_seed(4)
        model, reference = Model(), Model()
        reference.load_state_dict(model.state_dict())
        # Test both a mixed batch and a single minority-class graph.
        batches = [Batch(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1])),
                   Batch(torch.tensor([[2., 0.]]), torch.tensor([1]))]
        weights = torch.tensor([2.0 / 3.0, 2.0])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        expected_loss = 0.0
        for batch in batches:
            expected_optimizer.zero_grad()
            loss = F.cross_entropy(reference(batch), batch.y, weight=weights, reduction="sum") / batch.y.numel()
            loss.backward()
            expected_optimizer.step()
            expected_loss += loss.item() * batch.y.numel()

        actual_loss = train_one_epoch(model, batches, optimizer, "cpu", class_weights=weights)
        self.assertAlmostEqual(actual_loss, expected_loss / 3, places=6)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_weight_remains_effective_for_single_graph_update(self):
        import torch
        from phase2.graph.engine import train_one_epoch

        class Batch:
            y = torch.tensor([1])

            def to(self, device):
                return self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(1, 2))

            def forward(self, batch):
                return self.logits

        plain, weighted = Model(), Model()
        plain_loss = train_one_epoch(plain, [Batch()], torch.optim.SGD(plain.parameters(), lr=0.1), "cpu")
        weighted_loss = train_one_epoch(weighted, [Batch()], torch.optim.SGD(weighted.parameters(), lr=0.1),
                                        "cpu", class_weights=torch.tensor([0.5, 2.0]))
        self.assertAlmostEqual(weighted_loss, 2 * plain_loss, places=6)
        self.assertTrue(torch.allclose(weighted.logits, 2 * plain.logits, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
