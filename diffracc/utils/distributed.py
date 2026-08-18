# This file has been created by Ashley and Luna. It provides utility functions for distributing the program with SLURM
import os

from .logger import LoggingLevels, get_logger


def distribute(sliceable_array):
    """
    Compute the slice of a larger array that this node should be dealing with in a distributed environment. The slice is
    determined by the task ID and the total number of tasks in the distributed environment.

    Parameters
    ----------
    sliceable_array : array-like
        The larger array to be distributed across the nodes. Each node will only interact with its own slice of the
        array, determined by its task ID and the total number of tasks in the distributed environment.

    Returns
    -------
    array-like
        The slice of the array that this node should be dealing with.
    """
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
    def __init__(self, log_level: int = LoggingLevels.INFO.value):
        self.logger = get_logger(__name__, log_level)


    def is_distributed(self) -> bool:
        """
        Check if the program is running in a distributed environment by checking the SLURM_ARRAY_TASK_COUNT environment
        variable.

        Returns
        -------
        bool
            True if the program is running in a distributed environment, False otherwise.
        """
        return self.get_task_count() != 1


    def get_task_id(self) -> int:
        """
        Get the task ID of the current node in a distributed environment by checking the SLURM_ARRAY_TASK_ID environment
        variable. If the program is not running in a distributed environment, it returns 0.

        Returns
        -------
        int
            The task ID of the current node.
        """
        return int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


    def get_task_count(self) -> int:
        """
        Get the total number of tasks in a distributed environment by checking the SLURM_ARRAY_TASK_COUNT environment
        variable. If the program is not running in a distributed environment, it returns 1.

        Returns
        -------
        int
            The total number of tasks in a distributed environment.
        """
        return int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))


    def get_bin_end(self, n: int) -> int:
        """
        Function to take a total number n and get the corresponding end of the bin that this node should be dealing with
        
        Parameters
        ----------
        n : int
            The total number of items to be distributed across the nodes.
        
        Returns
        -------
        int
            The end index of the bin that this node should be dealing with.
        """
        return (n * (self.get_task_id() + 1)) // self.get_task_count()


    def get_bin_start(self, n: int) -> int:
        """
        Function to take a total number n and get the corresponding start of the bin that this node should be dealing
        with
        
        Parameters
        ----------
        n : int
            The total number of items to be distributed across the nodes.

        Returns
        -------
        int
            The start index of the bin that this node should be dealing with.
        """
        return (n * self.get_task_id()) // self.get_task_count()
