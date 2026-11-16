import unittest

from drive_utils import extract_google_drive_file_id


class GoogleDriveReferenceTests(unittest.TestCase):
    FILE_ID = "1zPrsdalDVOjgGr0rGPBXq4HpoeYa9h2E"

    def test_accepts_raw_file_id(self):
        self.assertEqual(extract_google_drive_file_id(self.FILE_ID), self.FILE_ID)

    def test_extracts_id_from_share_link(self):
        link = f"https://drive.google.com/file/d/{self.FILE_ID}/view?usp=sharing"
        self.assertEqual(extract_google_drive_file_id(link), self.FILE_ID)

    def test_extracts_id_from_query_link(self):
        link = f"https://drive.google.com/open?id={self.FILE_ID}"
        self.assertEqual(extract_google_drive_file_id(link), self.FILE_ID)

    def test_rejects_non_drive_url(self):
        with self.assertRaises(ValueError):
            extract_google_drive_file_id(f"https://example.com/{self.FILE_ID}")

    def test_rejects_malformed_drive_link(self):
        with self.assertRaises(ValueError):
            extract_google_drive_file_id("https://drive.google.com/file/d//view")


if __name__ == "__main__":
    unittest.main()
