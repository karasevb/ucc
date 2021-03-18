/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_MATH_H_
#define UCC_MATH_H_

#include "config.h"
#include <ucs/sys/math.h>
#include "ucc_datastruct.h"

#define ucc_min(_a, _b) ucs_min((_a), (_b))
#define ucc_max(_a, _b) ucs_max((_a), (_b))

extern size_t ucc_dt_sizes[UCC_DT_USERDEFINED];
static inline size_t ucc_dt_size(ucc_datatype_t dt)
{
    if (dt < UCC_DT_USERDEFINED) {
        return ucc_dt_sizes[dt];
    }
    return 0;
}

static inline ucc_rank_t ucc_ep_map_eval(ucc_ep_map_t map, ucc_rank_t rank)
{
    ucc_rank_t r;
    switch(map.type) {
    case UCC_EP_MAP_FULL:
        r = rank;
        break;
    case UCC_EP_MAP_STRIDED:
        r = map.strided.start + rank*map.strided.stride;
        break;
    case UCC_EP_MAP_ARRAY:
        r = *((ucc_rank_t*)((ptrdiff_t)map.array.map + rank*map.array.elem_size));
        break;
    case UCC_EP_MAP_CB:
        r = (ucc_rank_t)map.cb.cb(rank, map.cb.cb_ctx);
        break;
    default:
        r = -1;
    }
    return r;
}

#endif
