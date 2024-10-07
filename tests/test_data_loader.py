import pytest
from src.data_loader import DataLoader

def test_load_data():
    loader = DataLoader("data/raw/example.csv")
    data = loader.load_data()
    assert data is not None