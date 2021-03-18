/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_team.h"
#include "ucc_lib.h"
#include "components/cl/ucc_cl.h"
#include "components/tl/ucc_tl.h"

static ucc_status_t ucc_team_alloc_id(ucc_team_t *team);
static void ucc_team_relase_id(ucc_team_t *team);

void ucc_copy_team_params(ucc_team_params_t *dst, const ucc_team_params_t *src)
{
    dst->mask = src->mask;
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_ORDERING, ordering);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_OUTSTANDING_COLLS,
                            outstanding_colls);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_EP, ep);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_EP_RANGE, ep_range);
    //TODO do we need to copy ep_list ?
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_TEAM_SIZE,
                            team_size);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_SYNC_TYPE,
                            sync_type);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_OOB, oob);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_P2P_CONN, p2p_conn);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_MEM_PARAMS,
                            mem_params);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_TEAM_PARAM_FIELD_EP_MAP, ep_map);
}

static ucc_status_t ucc_team_create_post_single(ucc_context_t *context,
                                                ucc_team_t *team)
{
    ucc_status_t status;
    if ((team->params.mask & UCC_TEAM_PARAM_FIELD_EP) &&
        (team->params.mask & UCC_TEAM_PARAM_FIELD_EP_RANGE) &&
        (team->params.ep_range == UCC_COLLECTIVE_EP_RANGE_CONTIG)) {
        team->rank =
            team->params.ep; //TODO need to make sure we don't exceed rank size
    } else {
        ucc_error(
            "rank value of a process is not provided via ucc_team_params.ep "
            "with ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG. "
            "not supported yet...");
        return UCC_ERR_NOT_SUPPORTED;
    }
    team->cl_teams = ucc_malloc(sizeof(ucc_cl_team_t *) * context->n_cl_ctx);
    if (!team) {
        ucc_error("failed to allocate %zd bytes for cl teams array",
                  sizeof(ucc_cl_team_t *) * context->n_cl_ctx);
        return UCC_ERR_NO_MEMORY;
    }
    team->state = UCC_TEAM_CL_CREATE;
    if (!team->service_team) {
        ucc_base_team_params_t b_params;
        ucc_base_team_t       *b_team;
        status = ucc_tl_context_get(context, UCC_TL_UCP, &context->service_ctx);
        if (UCC_OK != status) {
            ucc_warn("TL UCP context is not available, "
                     "service team can not be created");
            goto error;
        }
        memcpy(&b_params, &team->params, sizeof(ucc_team_params_t));
        b_params.rank     = team->rank;
        b_params.scope    = UCC_CL_LAST + 1; // CORE scopre id - never overlaps with CL type
        b_params.scope_id = 0;
        status = UCC_TL_CTX_IFACE(context->service_ctx)
            ->team.create_post(&context->service_ctx->super, &b_params, &b_team);
        if (UCC_OK != status) {
            ucc_error("tl ucp service team create post failed");
            goto error;
        }
        team->service_team = ucc_derived_of(b_team, ucc_tl_team_t);
        team->state = UCC_TEAM_SERVICE_TEAM;
    }
    team->last_team_create_posted = -1;
    team->status                  = UCC_INPROGRESS;
    return UCC_OK;

error:
    free(team->cl_teams);
    return status;
}

ucc_status_t ucc_team_create_post(ucc_context_h *contexts, uint32_t num_contexts,
                                  const ucc_team_params_t *params,
                                  ucc_team_h *new_team)
{
    ucc_team_t  *team;
    ucc_status_t status;
    if (num_contexts < 1) {
        return UCC_ERR_INVALID_PARAM;
    } else if (num_contexts > 1) {
        ucc_error("team creation from multiple contexts is not supported yet");
        return UCC_ERR_NOT_SUPPORTED;
    }
    team = ucc_malloc(sizeof(ucc_team_t), "ucc_team");
    if (!team) {
        ucc_error("failed to allocate %zd bytes for ucc team",
                  sizeof(ucc_team_t));
        return UCC_ERR_NO_MEMORY;
    }
    team->n_cl_teams   = 0;
    team->num_contexts = num_contexts;
    team->service_team = NULL;
    team->task         = NULL;
    team->id           = ((uint16_t)-1);
    team->contexts =
        ucc_malloc(sizeof(ucc_context_t *) * num_contexts, "ucc_team_ctx");
    if (!team->contexts) {
        ucc_error("failed to allocate %zd bytes for ucc team contexts array",
                  sizeof(ucc_context_t) * num_contexts);
        status = UCC_ERR_NO_MEMORY;
        goto err_ctx_alloc;
    }

    memcpy(team->contexts, contexts, sizeof(ucc_context_t *) * num_contexts);
    ucc_copy_team_params(&team->params, params);
    status    = ucc_team_create_post_single(contexts[0], team);
    *new_team = team;
    return status;

err_ctx_alloc:
    ucc_free(team);
    return status;
}

static inline ucc_status_t
ucc_team_create_service_team(ucc_context_t *context, ucc_team_t *team)
{
    ucc_status_t status;
    status = UCC_TL_CTX_IFACE(context->service_ctx)
        ->team.create_test(&team->service_team->super);
    if (status < 0) {
        team->service_team = NULL;
        ucc_error("failed to create service tl ucp team");
    }
    return status;
}

static inline ucc_status_t
ucc_team_create_cls(ucc_context_t *context, ucc_team_t *team)
{
    int                    i;
    ucc_cl_iface_t        *cl_iface;
    ucc_base_team_t       *b_team;
    ucc_status_t           status;
    ucc_base_team_params_t b_params;

    if (team->last_team_create_posted >= 0) {
        cl_iface = UCC_CL_CTX_IFACE(context->cl_ctx[team->last_team_create_posted]);
        b_team   = &team->cl_teams[team->last_team_create_posted]->super;
        status   = cl_iface->team.create_test(b_team);
        if (status < 0) {
            ucc_debug("failed to create CL %s team", cl_iface->super.name);
            /* TODO: see comment above */
        } else if (status == UCC_INPROGRESS) {
            return status;
        }
    }
    memcpy(&b_params.params, &team->params, sizeof(ucc_team_params_t));
    b_params.rank = team->rank;
    for (i = team->last_team_create_posted + 1; i < context->n_cl_ctx; i++) {
        cl_iface = UCC_CL_CTX_IFACE(context->cl_ctx[i]);
        status   = cl_iface->team.create_post(&context->cl_ctx[i]->super,
                                            &b_params, &b_team);
        if (status != UCC_OK) {
            ucc_debug("failed to create CL %s team", cl_iface->super.name);
            /* TODO: see comment above*/
            continue;
        }
        status = cl_iface->team.create_test(b_team);
        if (status < 0) {
            ucc_debug("failed to create CL %s team", cl_iface->super.name);
            /* TODO: see comment above */
            continue;
        }
        team->cl_teams[team->n_cl_teams++] =
            ucc_derived_of(b_team, ucc_cl_team_t);
        if (status == UCC_INPROGRESS) {
            team->last_team_create_posted = i;
            /* workaround to fix oob allgather issue if multiple teams use it
               simultaneously*/
            return UCC_INPROGRESS;
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_team_create_test_single(ucc_context_t *context,
                                         ucc_team_t    *team)
{
    ucc_status_t status;
    switch (team->state) {
    case UCC_TEAM_SERVICE_TEAM:
        status = ucc_team_create_service_team(context, team);
        if (UCC_OK != status) {
            return status;
        }
        team->state = UCC_TEAM_ALLOC_ID;
    case UCC_TEAM_ALLOC_ID:
        status = ucc_team_alloc_id(team);
        if (UCC_OK != status) {
            return status;
        }
        team->state = UCC_TEAM_CL_CREATE;
    case UCC_TEAM_CL_CREATE:
        status = ucc_team_create_cls(context, team);
        if (UCC_OK != status) {
            return status;
        }
    }
    team->status = UCC_OK;
    /* TODO: add team/coll selection and check if some teams are never
             used after selection and clean them up */
    return UCC_OK;
}

ucc_status_t ucc_team_create_test(ucc_team_h team)
{
    /* we don't support multiple contexts per team yet */
    ucc_assert(team->num_contexts == 1);
    if (team->status == UCC_OK) {
        return UCC_OK;
    }
    return ucc_team_create_test_single(team->contexts[0], team);
}

static ucc_status_t ucc_team_destroy_single(ucc_team_h team)
{
    ucc_cl_iface_t *cl_iface;
    int             i;
    ucc_status_t    status;
    if (team->service_team) {
        if (UCC_OK != (status = UCC_TL_CTX_IFACE(team->contexts[0]->service_ctx)
                       ->team.destroy(&team->service_team->super))) {
            return status;
        }
        team->service_team = NULL;
        ucc_tl_context_put(team->contexts[0]->service_ctx);
    }
    for (i = 0; i < team->n_cl_teams; i++) {
        if (!team->cl_teams[i])
            continue;
        cl_iface = UCC_CL_TEAM_IFACE(team->cl_teams[i]);
        if (UCC_OK !=
            (status = cl_iface->team.destroy(&team->cl_teams[i]->super))) {
            return status;
        }
        team->cl_teams[i] = NULL;
    }
    ucc_free(team);
    return UCC_OK;
}

ucc_status_t ucc_team_destroy_nb(ucc_team_h team)
{
    if (team->status != UCC_OK) {
        ucc_error("team %p is used before team_create is completed", team);
        return UCC_ERR_INVALID_PARAM;
    }

    /* we don't support multiple contexts per team yet */
    ucc_assert(team->num_contexts == 1);
    return ucc_team_destroy_single(team);
}

ucc_status_t ucc_team_destroy(ucc_team_h team)
{
    ucc_status_t status;
    ucc_team_relase_id(team);
    while (UCC_INPROGRESS == (status = ucc_team_destroy_single(team))) {
        ; //TODO call ucc progress here
    }
    return status;
}

static inline int
find_first_set_and_zero(uint64_t *value) {
    int i;
    for (i=0; i<64; i++) {
        if (*value & ((uint64_t)1 << i)) {
            *value &= ~((uint64_t)1 << i);
            return i+1;
        }
    }
    return 0;
}

static inline void
set_id_bit(uint64_t *local, int id) {
    int map_pos = id / 64;
    int pos = id % 64;
    local[map_pos] |= ((uint64_t)1 << (pos-1));
}

static ucc_status_t ucc_team_alloc_id(ucc_team_t *team)
{
    /* at least 1 ctx is always available */
    ucc_context_t   *ctx      = team->contexts[0];
    ucc_tl_iface_t  *tl_iface = UCC_TL_TEAM_IFACE(team->service_team);
    uint64_t        *local, *global;
    ucc_status_t     status;
    int              pos, i;

    if (!ctx->ids.pool) {
        ctx->ids.pool = ucc_malloc(ctx->ids.pool_size*2*sizeof(uint64_t), "ids_pool");
        if (!ctx->ids.pool) {
            ucc_error("failed to allocate %zd bytes for team_ids_pool",
                      ctx->ids.pool_size*2*sizeof(uint64_t));
            return UCC_ERR_NO_MEMORY;
        }
        /* init all bits to 1 - all available */
        memset(ctx->ids.pool, 255, ctx->ids.pool_size*2*sizeof(uint64_t));
    }
    local  = ctx->ids.pool;
    global = ctx->ids.pool + ctx->ids.pool_size;

    if (!team->task) {
        ucc_tl_team_subset_t subset = {
            .map.type   = UCC_EP_MAP_FULL,
            .map.ep_num = team->params.oob.participants,
            .myrank     = team->rank
        };
        status = tl_iface->scoll.allreduce(
            &team->service_team->super, global, local, UCC_DT_UINT64, ctx->ids.pool_size, UCC_OP_BAND,
            subset, &team->task);
        if (status < 0) {
            ucc_error("failed to start service allreduce for team ids pool allocation: %s",
                      ucc_status_string(status));
            return status;
        }
    }
    ucc_context_progress(ctx);
    status = tl_iface->scoll.test(team->task);
    if (status < 0) {
        ucc_error("service allreduce test failure: %s",
                  ucc_status_string(status));
        return status;
    } else if (status != UCC_OK) {
        return status;
    }
    tl_iface->scoll.cleanup(team->task);
    team->task = NULL;
    memcpy(local, global, ctx->ids.pool_size*sizeof(uint64_t));
    pos = 0;
    for (i=0; i<ctx->ids.pool_size; i++) {
        if ((pos = find_first_set_and_zero(&local[i])) > 0) {
            break;
        }
    }
    if (pos > 0) {
        ucc_assert(pos <= 64);
        team->id = (uint16_t)(i*64+pos);
        ucc_info("allocated ID %d for team %p", team->id, team);
    } else {
        team->id = 0;
        ucc_warn("could not allocate team id, whole id space is occupied, "
                 "try increasing UCC_TEAM_IDS_POOL_SIZE");
        return UCC_ERR_NO_RESOURCE;
    }
    ucc_assert(team->id > 0);
    return UCC_OK;
}

static void ucc_team_relase_id(ucc_team_t *team)
{
    ucc_context_t *ctx = team->contexts[0];
    if (1 /* Check id is internal */) {
        set_id_bit(ctx->ids.pool, team->id);
    }
}
