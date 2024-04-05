from unittest import TestCase, main 
from pathlib import Path
import os

class TestFiles(TestCase):
    # Vérifie si ces fichiers sont dans le dossier racine
    def test_root_files(self):
        root_dir_files = os.listdir()
        self.assertIn("streamlit_app.py", root_dir_files)


if __name__  == '__main__':
    main(verbosity=2)
