import unittest

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from phase2.lightgbm.inference import selected_vectorizer_transform


class SelectedVectorizerTransformTests(unittest.TestCase):
    def test_selected_transform_matches_original_for_various_feature_types(self):
        training = [
            {"api:a": 2.5, "api:b": 3.0, "color": "blue", "tags": ["one", "two"]},
            {"api:c": 4.0, "color": "red", "tags": ["two", "three"]},
        ]
        samples = [
            {"api:a": 1.0, "api:b": 7, "color": "red", "tags": ["one", "unknown"], "unseen": 10},
            {"api:c": 2.0, "color": "green", "tags": ["three", "two"]},
            {},
        ]
        for sparse in (True, False):
            vectorizer = DictVectorizer(sparse=sparse)
            vectorizer.fit(training)
            names = vectorizer.get_feature_names_out()
            for indices in (
                np.array([len(names) - 1, 0, 2]),
                np.array([0, 0, len(names) - 1]),
                np.array([-1, 0]),
                np.array([], dtype=np.int64),
                np.arange(len(names)),
            ):
                with self.subTest(sparse=sparse, indices=indices.tolist()):
                    old_vocab = dict(vectorizer.vocabulary_)
                    expected = vectorizer.transform(samples)[:, indices]
                    actual = selected_vectorizer_transform(vectorizer, indices, samples)
                    if sparse:
                        np.testing.assert_array_equal(actual.toarray(), expected.toarray())
                    else:
                        np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(vectorizer.vocabulary_, old_vocab)

    def test_reordered_selected_features_have_same_names_and_values(self):
        vectorizer = DictVectorizer()
        features = [{"a": 1.0, "b": 2.0, "c": 3.0}]
        vectorizer.fit(features)
        indices = np.array([2, 0])
        actual = selected_vectorizer_transform(vectorizer, indices, features)
        np.testing.assert_array_equal(actual.toarray(), [[3.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
