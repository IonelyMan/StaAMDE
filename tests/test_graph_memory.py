import importlib.util
import unittest

from phase2.graph.data import to_homogeneous_graph


class GraphMemoryTests(unittest.TestCase):
    def test_homogeneous_conversion_does_not_copy_unused_edge_types(self):
        class Hetero:
            node_types = ["app", "api"]

            def to_homogeneous(self, **kwargs):
                self.options = kwargs
                return object()

        data = Hetero()
        to_homogeneous_graph(data)
        self.assertFalse(data.options["add_edge_type"])

    def test_existing_homogeneous_graph_drops_unused_edge_types(self):
        class Graph:
            def __init__(self):
                self.edge_index = object()
                self.x = object()
                self.edge_type = object()

        graph = Graph()
        to_homogeneous_graph(graph)
        self.assertFalse(hasattr(graph, "edge_type"))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_micro_batches_match_weighted_full_batch_update(self):
        import torch
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
        full = Model()
        split = Model()
        reference = Model()
        split.load_state_dict(full.state_dict())
        reference.load_state_dict(full.state_dict())
        x = torch.tensor([[1., 0.], [0., 1.], [2., 0.], [0., 2.]])
        y = torch.tensor([0, 1, 1, 1])
        weights = torch.tensor([1., 2.])
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        expected_loss = torch.nn.functional.cross_entropy(
            reference(Batch(x, y)), y, weight=weights, reduction="sum"
        ) / len(y)
        expected_loss.backward()
        reference_optimizer.step()
        full_loss = train_one_epoch(full, [Batch(x, y)], torch.optim.SGD(full.parameters(), lr=0.1),
                                    "cpu", weights, accumulation_steps=1)
        split_loss = train_one_epoch(split, [Batch(x[:2], y[:2]), Batch(x[2:], y[2:])],
                                     torch.optim.SGD(split.parameters(), lr=0.1), "cpu", weights,
                                     accumulation_steps=2)
        self.assertAlmostEqual(full_loss, split_loss, places=6)
        self.assertAlmostEqual(full_loss, expected_loss.item(), places=6)
        for a, b in zip(full.parameters(), split.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
        for actual, expected in zip(full.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_single_graph_update_retains_class_weight(self):
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
                                        "cpu", torch.tensor([0.5, 2.0]))
        self.assertAlmostEqual(weighted_loss, 2.0 * plain_loss, places=6)
        self.assertTrue(torch.allclose(weighted.logits, 2.0 * plain.logits, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
