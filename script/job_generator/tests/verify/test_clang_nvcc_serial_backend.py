# pylint: disable=missing-docstring

"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest

from bashi.globals import ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, CLANG, GCC, HOST_COMPILER, OFF, ON
from bashi.types import ParameterValuePair
from utils import default_remove_test, parse_expected_val_pairs

from alpaka_bashi.verify import remove_clang_unsupported_by_nvcc_without_serial_backend


class TestClangNvccSerialBackend(unittest.TestCase):
    def test_remove_invalid_combinations(self):
        test_param_value_pairs: list[ParameterValuePair] = parse_expected_val_pairs(
            [
                ((HOST_COMPILER, CLANG, 22), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 23), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 23), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, GCC, 13), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
            ]
        )

        expected_results: list[ParameterValuePair] = parse_expected_val_pairs(
            [
                ((HOST_COMPILER, CLANG, 22), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 23), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, GCC, 13), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
            ]
        )

        default_remove_test(
            remove_clang_unsupported_by_nvcc_without_serial_backend,
            test_param_value_pairs,
            expected_results,
            self,
        )
