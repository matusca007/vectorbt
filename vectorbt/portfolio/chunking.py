# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Extensions to `vectorbt.utils.chunking`."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.base.chunking import GroupLensMapper, FlexArraySlicer, group_lens_mapper
from vectorbt.portfolio.enums import SimulationOutput
from vectorbt.utils.chunking import ChunkMeta, ArraySlicer
from vectorbt.utils.config import Config
from vectorbt.utils.template import Rep

flex_array_gl_slicer = FlexArraySlicer(1, mapper=group_lens_mapper)
"""Flexible slicer along the column axis based on group lengths."""


def get_init_cash_slicer(ann_args: tp.AnnArgs) -> ArraySlicer:
    """Get slicer for `init_cash` based on cash sharing."""
    cash_sharing = ann_args['cash_sharing']['value']
    if cash_sharing:
        return ArraySlicer(0)
    return ArraySlicer(0, mapper=group_lens_mapper)


def merge_sim_outs(results: tp.List[SimulationOutput],
                   chunk_meta: ChunkMeta,
                   ann_args: tp.AnnArgs,
                   mapper: GroupLensMapper) -> SimulationOutput:
    """Merge chunks of `vectorbt.portfolio.enums.SimulationOutput` instances."""
    for _chunk_meta in chunk_meta:
        _mapped_chunk_meta = mapper.map(_chunk_meta, ann_args)
        results[_chunk_meta.idx].order_records['col'] += _mapped_chunk_meta.start
        results[_chunk_meta.idx].log_records['col'] += _mapped_chunk_meta.start
        results[_chunk_meta.idx].log_records['group'] += _chunk_meta.start
    return SimulationOutput(
        order_records=np.concatenate([r.order_records for r in results]),
        log_records=np.concatenate([r.log_records for r in results]),
    )


merge_sim_outs_config = Config(
    dict(
        merge_func=merge_sim_outs,
        merge_kwargs=dict(
            chunk_meta=Rep("chunk_meta"),
            ann_args=Rep("ann_args"),
            mapper=group_lens_mapper
        )
    )
)
"""Config for merging using `merge_sim_outs`."""
