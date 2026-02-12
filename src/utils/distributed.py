# This file has been created by Ashley and Luna. It provides utility functions for distributing the program with SLURM

import os
import time
from pathlib import Path
import shutil
import numpy as np
import utils.logging
import logging

def distribute(sliceable_array):
    du = DistributedUtils()

    # distribute across multiple tasks by giving each node a slice of a larger array dependent on its array id
    n_files = len(sliceable_array)
    bin_start = du.get_bin_start(n_files)
    bin_end = du.get_bin_end(n_files)
    return sliceable_array[bin_start:bin_end]  # each node only interacts with its own bin


class DistributedUtils:
    """
    Utility functions for running on a distributed system (SLURM in particular)
    """
    def __init__( self, log_level: int = logging.INFO ):
        self.logger = utils.logging.get_logger( __name__, log_level )

    def is_distributed( self ) -> bool:
        return self.get_task_count() != 1

    def get_task_id( self ) -> int:
        return int( os.environ.get( "SLURM_ARRAY_TASK_ID", 0 ) )

    def get_task_count( self ) -> int:
        return int( os.environ.get( "SLURM_ARRAY_TASK_COUNT", 1 ) )

    def get_bin_end( self, n: int ) -> int:
        """
        Function to take a total number n and get the corresponding end of the bin that this node should be dealing with
        """
        return ( n * ( self.get_task_id() + 1 ) ) // self.get_task_count()

    def get_bin_start( self, n: int ) -> int:
        """
        Function to take a total number n and get the corresponding start of the bin that this node should be dealing with
        """
        return ( n * self.get_task_id() ) // self.get_task_count()