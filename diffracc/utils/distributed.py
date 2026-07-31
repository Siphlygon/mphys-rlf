# This file has been created by Ashley and Luna. It provides utility functions for distributing the program with SLURM
import json
import os
from collections.abc import Sequence
from pathlib import Path

from .logger import LoggingLevels, get_logger

_logger = get_logger(__name__)


def distribute(sliceable_array):
    du = DistributedUtils()

    # distribute across multiple tasks by giving each node a slice of a larger array dependent on its array id
    n_files = len(sliceable_array)
    bin_start = du.get_bin_start(n_files)
    bin_end = du.get_bin_end(n_files)
    return sliceable_array[bin_start:bin_end]  # each node only interacts with its own bin


def write_work_plan(plan_path: Path | str, items: Sequence) -> None:
    """
    Write the list of remaining work items to `plan_path`, once, from a single process, so that every distributed worker
    can later read the identical list and split it consistently.

    This enables being able to account for existing files pre-distribution, which would otherwise be a race-condition
    for each worker trying to calculate "what is left to do" itself when it should be set prior to being distributed.

    The plan is written to a temporary sibling and atomically renamed into place, so a worker can never observe a
    half-written plan. It is serialised as JSON so the same helper works for integer indices (sampling) or path strings
    (analysis), and so the plan stays human-inspectable.

    Parameters
    ----------
    plan_path : Path | str
        Where to write the plan. Parent directories are created if necessary.
    items : Sequence
        The remaining work items (JSON-serialisable, e.g. ints or strings), already in the desired deterministic
        order; workers slice this order directly.
    """
    items = list(items)
    plan_path = Path(plan_path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = plan_path.with_name(plan_path.name + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(items, f)
    tmp_path.replace(plan_path)  # atomic on the same filesystem, so readers see all-or-nothing
    _logger.info('Wrote work plan with %i items to %s', len(items), plan_path)


def read_work_plan(plan_path: Path | str) -> list:
    """
    Read the full list of work items written by `write_work_plan`.

    Parameters
    ----------
    plan_path : Path | str
        The plan file to read.

    Returns
    -------
    list
        Every work item in the plan, in the order it was written.
    """
    with open(plan_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_work_plan_slice(plan_path: Path | str) -> list:
    """
    Read a work plan and return only this worker's share of it, using the same contiguous SLURM split as `distribute`.
    Because every worker reads the same immutable file, the shares are disjoint and together cover the whole plan
    exactly once, with no coordination between the independently-scheduled tasks.

    Parameters
    ----------
    plan_path : Path | str
        The plan file to read.

    Returns
    -------
    list
        This worker's contiguous slice of the plan (possibly empty).
    """
    return distribute(read_work_plan(plan_path))


class DistributedUtils:
    """
    Utility functions for running on a distributed system (SLURM in particular)
    """
    def __init__( self, log_level: int = LoggingLevels.INFO.value ):
        self.logger = get_logger( __name__, log_level )


    def is_distributed( self ) -> bool:
        """
        Check if the program is running in a distributed environment by checking the SLURM_ARRAY_TASK_COUNT environment
        variable.

        Returns
        -------
        bool
            True if the program is running in a distributed environment, False otherwise.
        """
        return self.get_task_count() != 1


    def get_task_id( self ) -> int:
        """
        Get the task ID of the current node in a distributed environment by checking the SLURM_ARRAY_TASK_ID environment
        variable. If the program is not running in a distributed environment, it returns 0.

        Returns
        -------
        int
            The task ID of the current node.
        """
        return int( os.environ.get( "SLURM_ARRAY_TASK_ID", 0 ) )


    def get_task_count( self ) -> int:
        """
        Get the total number of tasks in a distributed environment by checking the SLURM_ARRAY_TASK_COUNT environment
        variable. If the program is not running in a distributed environment, it returns 1.

        Returns
        -------
        int
            The total number of tasks in a distributed environment.
        """
        return int( os.environ.get( "SLURM_ARRAY_TASK_COUNT", 1 ) )


    def get_bin_end( self, n: int ) -> int:
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
        return ( n * ( self.get_task_id() + 1 ) ) // self.get_task_count()


    def get_bin_start( self, n: int ) -> int:
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
        return ( n * self.get_task_id() ) // self.get_task_count()
