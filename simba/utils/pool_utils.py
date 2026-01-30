"""
Multiprocessing pool utilities.
"""

import atexit
import multiprocessing
import signal
import sys
from multiprocessing.pool import Pool as PoolClass

from simba.utils.logger_setup import logger


class NoDaemonProcess(multiprocessing.Process):
    """
    Process class that allows spawning child processes.
    """

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonPool(PoolClass):
    """
    Pool that uses non-daemon worker processes.

    This allows workers to spawn child processes, which is necessary for
    implementing hard timeouts.
    """

    def Process(self, *args, **kwds):  # noqa: N802
        proc = super().Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc


# Global pool registry for cleanup
_active_pools = []


def register_pool(pool):
    """
    Register a pool for automatic cleanup on exit.

    Parameters
    ----------
    pool : NoDaemonPool
        Pool to register for cleanup.
    """
    _active_pools.append(pool)


def unregister_pool(pool):
    """
    Unregister a pool from automatic cleanup.

    Parameters
    ----------
    pool : NoDaemonPool
        Pool to unregister.
    """
    try:  # noqa: SIM105
        _active_pools.remove(pool)
    except ValueError:
        pass


def cleanup_pool(pool):
    """
    Clean up a single pool.
    """
    try:
        pool.close()
        pool.terminate()
        pool.join()
    except Exception as e:
        logger.warning(f"Error during pool cleanup: {e}")


def _cleanup_all_pools():
    """Clean up all registered pools on exit."""
    for pool in _active_pools:
        cleanup_pool(pool)


def _signal_handler(sig, frame):
    """Handle interrupt signals by cleaning up pools."""
    logger.info("Received interrupt signal, cleaning up pools...")
    _cleanup_all_pools()
    sys.exit(0)


# Register cleanup handlers
atexit.register(_cleanup_all_pools)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
