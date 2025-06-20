from torch_geometric.data import Data

class MyData(Data):
    """Custom Data class to tell DataLoader how to batch matrix and coords attributes."""
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'coords':
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)
