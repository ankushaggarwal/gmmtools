# src/gmmtools/__init__.py
from .gmm import GMM_Custom
from .gmm import (
    CMI_gmms,
    CMI_gmms_MC,
    MI_data_matrix,
    CMI_data
)

__all__ = [
    "GMM_Custom",
    "CMI_gmms",
    "CMI_gmms_MC",
    "MI_data_matrix",
    "CMI_data"
]
