from functools import partial
from typing import Union

import numpy as np


class CQFSHSVDSampler:
    _verbose: bool

    def __init__(
            self,
            verbose: bool = False,
    ):
        self._verbose = verbose

