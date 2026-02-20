import numpy as np
from Measurements import Measurements
from dumper import update_dump_file
from logger import log_simulation_data

class MolecularDynamics(Measurements):
    def __init__(self,
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)