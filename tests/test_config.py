import unittest

from config import Config


class ConfigIsolationTests(unittest.TestCase):
    def test_mutable_settings_are_isolated_per_instance(self):
        first = Config.isolated()
        second = Config.isolated()

        first.DATASET["base_path"] = "/tmp/first-dataset"
        first.DATASET["subdirs"].append("extra")
        first.TRAINING["epochs"] = 1

        self.assertNotEqual(first.DATASET["base_path"], second.DATASET["base_path"])
        self.assertNotIn("extra", second.DATASET["subdirs"])
        self.assertNotIn("extra", Config.DATASET["subdirs"])
        self.assertNotEqual(first.TRAINING["epochs"], second.TRAINING["epochs"])
        self.assertEqual(second.TRAINING["epochs"], Config.TRAINING["epochs"])


if __name__ == "__main__":
    unittest.main()
