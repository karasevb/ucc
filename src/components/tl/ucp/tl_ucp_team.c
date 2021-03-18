/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_ep.h"
#include "tl_ucp_addr.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_tl_ucp_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_status_t          status = UCC_OK;
    ucc_tl_ucp_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_ucp_context_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super);
    /* TODO: init based on ctx settings and on params: need to check
             if all the necessary ranks mappings are provided */
    self->context_ep_storage = 0;
    self->addr_storage       = NULL;
    self->eps                = NULL;
    self->preconnect_task    = NULL;
    self->size               = params->params.oob.participants;
    self->scope              = params->scope;
    self->scope_id           = params->scope_id;
    self->rank               = params->rank;
    self->id                 = params->id;
    self->seq_num            = 0;
    if (self->context_ep_storage) {
        self->status = UCC_OK;
    } else {
        self->status = UCC_INPROGRESS;
        status       = ucc_tl_ucp_addr_exchange_start(ctx, params->params.oob,
                                                      &self->addr_storage);
        if (status == UCC_INPROGRESS) {
            /* exchange started but not complete return UCC_OK from post */
            status = UCC_OK;
        }
    }
    tl_info(tl_context->lib, "posted tl team: %p", self);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_ucp_team_t)
{
    if (self->addr_storage) {
        ucc_tl_ucp_addr_storage_free(self->addr_storage);
    }
    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_ucp_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_ucp_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_ucp_team_destroy(ucc_base_team_t *tl_team)
{
    ucc_tl_ucp_team_t    *team = ucc_derived_of(tl_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_context_t *ctx  = UCC_TL_UCP_TEAM_CTX(team);
    ucc_status_t          status;
    if (team->eps) {
        status = ucc_tl_ucp_close_eps(ctx, team->eps, team->size);
        if (UCC_INPROGRESS == status) {
            return status;
        } else if (UCC_OK != status) {
            tl_error(team->super.super.context->lib,
                     "failed to close team eps");
            return status;
        }
        ucc_free(team->eps);
    }
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_ucp_team_t)(tl_team);
    return UCC_OK;
}

static ucc_status_t ucc_tl_ucp_team_preconnect(ucc_tl_ucp_team_t *team)
{
    ucc_rank_t src, dst;
    ucc_status_t status;
    int i;
    if (!team->preconnect_task) {
        team->preconnect_task = ucc_tl_ucp_get_task(team);
        team->preconnect_task->tag = 0;
    }
    if (UCC_INPROGRESS == ucc_tl_ucp_test(team->preconnect_task)) {
        return UCC_INPROGRESS;
    }
    for (i = team->preconnect_task->send_posted; i < team->size; i++) {
        src = (team->rank - i + team->size) % team->size;
        dst = (team->rank + i) % team->size;
        status = ucc_tl_ucp_send_nb(NULL, 0, UCC_MEMORY_TYPE_UNKNOWN, src, team,
                                    team->preconnect_task);
        if (UCC_OK != status) {
            return status;
        }
        status = ucc_tl_ucp_recv_nb(NULL, 0, UCC_MEMORY_TYPE_UNKNOWN, dst, team,
                                    team->preconnect_task);
        if (UCC_OK != status) {
            return status;
        }
        if (UCC_INPROGRESS == ucc_tl_ucp_test(team->preconnect_task)) {
            return UCC_INPROGRESS;
        }
    }
    tl_debug(UCC_TL_TEAM_LIB(team), "preconnected tl team: %p, num_eps %d",
             team, team->size);
    ucc_tl_ucp_put_task(team->preconnect_task);
    team->preconnect_task = NULL;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_ucp_team_t    *team = ucc_derived_of(tl_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_context_t *ctx  = UCC_TL_UCP_TEAM_CTX(team);
    ucc_status_t          status;
    if (team->status == UCC_OK) {
        return UCC_OK;
    }
    if (team->addr_storage && (!team->eps)) {
        status = ucc_tl_ucp_addr_exchange_test(team->addr_storage);
        if (UCC_INPROGRESS == status) {
            return UCC_INPROGRESS;
        } else if (UCC_OK != status) {
            return status;
        }
        team->eps = ucc_calloc(sizeof(ucp_ep_h), team->size, "team_eps");
        if (!team->eps) {
            tl_error(tl_team->context->lib,
                     "failed to allocate %zd bytes for team eps",
                     sizeof(ucp_ep_h) * team->size);
            return UCC_ERR_NO_MEMORY;
        }
    }
    if (team->size <= ctx->cfg.preconnect) {
        status = ucc_tl_ucp_team_preconnect(team);
        if (UCC_INPROGRESS == status) {
            return UCC_INPROGRESS;
        } else if (UCC_OK != status) {
            goto err_preconnect;
        }
    }

    tl_info(tl_team->context->lib, "initialized tl team: %p", team);
    team->status = UCC_OK;
    return UCC_OK;

err_preconnect:
    ucc_free(team->eps);
    return status;
}
