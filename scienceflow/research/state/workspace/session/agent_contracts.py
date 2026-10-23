# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Workspace path and session sentinels used by the ScienceAgent adapter."""

from __future__ import annotations

import logging
import re

_logger = logging.getLogger("scienceflow")
_ON_LLM_CALL_UNSET = object()
_HOST_ABSOLUTE_VISIBLE_PATH_RE = re.compile(
    r"(?<![\w.])/(?:home|work|mnt|kaggle|tmp)(?:/[^\s\"'`<>),;]*)?",
)
_WSP_ABSOLUTE_VISIBLE_PREFIX_RE = re.compile(r"/\S+/wsp/[0-9a-f]{32}/")

__all__ = tuple(name for name in globals() if not name.startswith("__"))
