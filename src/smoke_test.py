"""Import the complete competition environment and print key versions."""

from __future__ import annotations

import cvxpy
import matplotlib
import networkx
import numpy as np
import openpyxl
import pandas as pd
import pulp
import scipy
import sklearn
import statsmodels
import sympy
import torch


def main() -> None:
    print("Python environment OK")
    print("Torch:", torch.__version__)
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("SciPy:", scipy.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Statsmodels:", statsmodels.__version__)
    print("OpenPyXL:", openpyxl.__version__)
    print("SymPy:", sympy.__version__)
    print("NetworkX:", networkx.__version__)
    print("CVXPY:", cvxpy.__version__)
    print("PuLP:", pulp.__version__)
    print("CUDA available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
