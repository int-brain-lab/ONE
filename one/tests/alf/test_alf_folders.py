import unittest
from pathlib import Path
import tempfile

import one.alf.folders as folders


class TestALFFolders(unittest.TestCase):
    tempdir = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.session_path = (Path(cls.tempdir.name)
                            .joinpath('fakelab', 'Subjects', 'fakemouse', '1900-01-01', '001'))
        cls.session_path.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_next_num_folder(self):
        self.session_path.rmdir()  # Remove '001' folder
        next_num = folders.next_num_folder(self.session_path.parent)
        self.assertEqual('001', next_num)

        self.session_path.parent.rmdir()  # Remove date folder
        next_num = folders.next_num_folder(self.session_path.parent)
        self.assertEqual('001', next_num)

        self.session_path.parent.joinpath(next_num).mkdir(parents=True)  # Add '001' folder
        next_num = folders.next_num_folder(self.session_path.parent)
        self.assertEqual('002', next_num)

        self.session_path.parent.joinpath('053').mkdir()  # Add '053' folder
        next_num = folders.next_num_folder(self.session_path.parent)
        self.assertEqual('054', next_num)

        self.session_path.parent.joinpath('099').mkdir()  # Add '099' folder
        next_num = folders.next_num_folder(self.session_path.parent)
        self.assertEqual('100', next_num)

        self.session_path.parent.joinpath('999').mkdir()  # Add '999' folder
        with self.assertRaises(AssertionError):
            folders.next_num_folder(self.session_path.parent)

    def test_remove_empty_folders(self):
        root = Path(self.tempdir.name) / 'glob_dir'
        root.mkdir()
        root.joinpath('empty0').mkdir(exist_ok=True)
        root.joinpath('full0').mkdir(exist_ok=True)
        root.joinpath('full0', 'file.txt').touch()
        self.assertTrue(len(list(root.glob('*'))) == 2)
        folders.remove_empty_folders(root)
        self.assertTrue(len(list(root.glob('*'))) == 1)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
