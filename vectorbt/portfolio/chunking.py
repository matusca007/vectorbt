# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Extensions to `vectorbt.utils.chunking`."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils.chunking import ChunkMeta, ArraySlicer
from vectorbt.base.chunking import GroupLensMapper, group_lens_mapper
from vectorbt.portfolio.enums import SimulationOutput


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
    order_id_start = 0
    log_id_start = 0
    for _chunk_meta in chunk_meta:
        _mapped_chunk_meta = mapper.map(_chunk_meta, ann_args)
        results[_chunk_meta.idx].order_records['id'] += order_id_start
        results[_chunk_meta.idx].log_records['id'] += log_id_start
        results[_chunk_meta.idx].log_records['order_id'] += order_id_start
        results[_chunk_meta.idx].order_records['col'] += _mapped_chunk_meta.start
        results[_chunk_meta.idx].log_records['col'] += _mapped_chunk_meta.start
        results[_chunk_meta.idx].log_records['group'] += _chunk_meta.start
        if len(results[_chunk_meta.idx].order_records) > 0:
            order_id_start = results[_chunk_meta.idx].order_records['id'][-1] + 1
        if len(results[_chunk_meta.idx].log_records) > 0:
            log_id_start = results[_chunk_meta.idx].log_records['id'][-1] + 1
    return SimulationOutput(
        order_records=np.concatenate([r.order_records for r in results]),
        log_records=np.concatenate([r.log_records for r in results]),
    )
