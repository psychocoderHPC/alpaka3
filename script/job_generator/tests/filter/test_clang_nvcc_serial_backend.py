# pylint: disable=missing-docstring

"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import io
import unittest

from bashi.globals import ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, CLANG, DEVICE_COMPILER, GCC, HOST_COMPILER, OFF, ON
from utils import parse_bashi_row

from alpaka_bashi.alpaka_filter import (
    AlpakaFilter,
    check_clang_host_compiler_requires_serial_backend_a5,
)


class TestClangNvccSerialBackend(unittest.TestCase):
    """A clang host compiler newer than any nvcc supported clang version cannot be used together
    with nvcc/CUDA. It is a CPU only compiler and must be combined with the serial backend."""

    VALID_ROWS = [
        [(HOST_COMPILER, CLANG, 22), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)],
        [(HOST_COMPILER, CLANG, 23), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)],
        [(HOST_COMPILER, CLANG, 23)],
        [(HOST_COMPILER, GCC, 13), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)],
        [(DEVICE_COMPILER, GCC, 13), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)],
    ]

    def test_valid_check_clang_host_compiler_requires_serial_backend_a5(self):
        for row in self.VALID_ROWS:
            with self.subTest(row=row):
                self.assertTrue(
                    check_clang_host_compiler_requires_serial_backend_a5(parse_bashi_row(row), AlpakaFilter()),
                    f"{row}",
                )

    INVALID_ROWS = [
        [(HOST_COMPILER, CLANG, 23), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)],
    ]

    def test_invalid_check_clang_host_compiler_requires_serial_backend_a5(self):
        for row in self.INVALID_ROWS:
            with self.subTest(row=row):
                reason_msg_func = io.StringIO()
                self.assertFalse(
                    check_clang_host_compiler_requires_serial_backend_a5(
                        parse_bashi_row(row), AlpakaFilter(output=reason_msg_func)
                    ),
                    f"{row}",
                )
                self.assertEqual(
                    reason_msg_func.getvalue(),
                    "Clang 23 is not supported by any nvcc version, therefore the serial backend must be enabled.",
                    f"{row}",
                )
