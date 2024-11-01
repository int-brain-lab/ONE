"""
(DEPRECATED) Module for identifying and parsing ALF file names.

This module has moved to :mod:`one.alf.path`.
"""
import warnings

from .path import *  # noqa

warnings.warn(
    '`one.alf.files` will be removed in version 3.0. Use `one.alf.path` instead.', FutureWarning)
