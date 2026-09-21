# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Static dependency rules for independently evolving modules."""

from scienceflow.foundation.architecture.dependencies import (
    DependencyRule,
    DependencyViolation,
    find_dependency_violations,
)

__all__ = [
    "DependencyRule",
    "DependencyViolation",
    "find_dependency_violations",
]
