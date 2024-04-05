from unittest import TestCase, main 
from pathlib import Path
import os

import streamlit as st 
from ultralytics import YOLO
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine

import database
import settings

class TestFiles(TestCase):
    # Vérifie si ces fichiers sont dans le dossier racine
    def test_root_files(self):
        root_dir_files = os.listdir()
        self.assertIn("streamlit_app.py", root_dir_files)


if __name__  == '__main__':
    main(verbosity=2)
