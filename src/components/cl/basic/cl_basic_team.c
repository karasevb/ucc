/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_cl_basic_team_t, ucc_base_context_t *cl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_cl_basic_context_t *ctx =
        ucc_derived_of(cl_context, ucc_cl_basic_context_t);
    int                     n_teams = 1, i;
    ucc_status_t            status;

    self->tl_ucp_team  = NULL;
    self->tl_nccl_team = NULL;
    ucc_assert(ctx->tl_ucp_ctx != NULL);
    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_team_t, &ctx->super);
    status = ucc_team_multiple_req_alloc(&self->team_create_req,
                                         UCC_CL_BASIC_NUM_TLS);
    if (UCC_OK != status) {
        cl_error(cl_context->lib, "failed to allocate team req multiple");
        return status;
    }
    self->team_create_req->descs[0].ctx = ctx->tl_ucp_ctx;
    if (self->tl_nccl_team) {
        self->team_create_req->descs[1].ctx = ctx->tl_nccl_ctx;
        n_teams++;
    }
    for (i = 0; i < n_teams; i++) {
        memcpy(&self->team_create_req->descs[i].param, params,
               sizeof(ucc_base_team_params_t));
        self->team_create_req->descs[i].param.scope    = UCC_CL_BASIC;
        self->team_create_req->descs[i].param.scope_id = 0;
    }
    self->team_create_req->n_teams = n_teams;
    status = ucc_tl_team_create_multiple(self->team_create_req);
    if (status < 0) {
        cl_error(cl_context->lib, "failed to post tl team create (%d)",
                 status);
        return status;
    }
    cl_info(cl_context->lib, "posted cl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_basic_team_t)
{
    cl_info(self->super.super.context->lib, "finalizing cl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_basic_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_cl_basic_team_t, ucc_cl_team_t);

ucc_status_t ucc_cl_basic_team_destroy(ucc_base_team_t *cl_team)
{
    ucc_cl_basic_team_t    *team    = ucc_derived_of(cl_team, ucc_cl_basic_team_t);
    ucc_cl_basic_context_t *ctx     = UCC_CL_BASIC_TEAM_CTX(team);
    ucc_status_t            status  = UCC_OK;
    int                     n_teams = team->tl_nccl_team ? 2 : 1;
    if (NULL == team->team_create_req) {
        status = ucc_team_multiple_req_alloc(&team->team_create_req,
                                             UCC_CL_BASIC_NUM_TLS);
        if (UCC_OK != status) {
            cl_error(ctx->super.super.lib, "failed to allocate team req multiple");
            return status;
        }
        team->team_create_req->n_teams       = n_teams;
        team->team_create_req->descs[0].team = team->tl_ucp_team;
        team->team_create_req->descs[1].team = team->tl_nccl_team;
    }
    status = ucc_tl_team_destroy_multiple(team->team_create_req);
    if (UCC_INPROGRESS == status) {
        return status;
    }

    if (team->team_create_req->descs[0].status != UCC_OK) {
        cl_error(ctx->super.super.lib, "ucp team destroy failed (%d)",
                 status);
        status = team->team_create_req->descs[0].status;
    }

    if (n_teams > 1 && (team->team_create_req->descs[1].status != UCC_OK)) {
        cl_error(ctx->super.super.lib, "nccl team destroy failed (%d)",
                 status);
        status = team->team_create_req->descs[1].status;
    }
    ucc_team_multiple_req_free(team->team_create_req);
    UCC_CLASS_DELETE_FUNC_NAME(ucc_cl_basic_team_t)(cl_team);
    return status;
}

ucc_status_t ucc_cl_basic_team_create_test(ucc_base_team_t *cl_team)
{
    ucc_cl_basic_team_t    *team = ucc_derived_of(cl_team, ucc_cl_basic_team_t);
    ucc_cl_basic_context_t *ctx  = UCC_CL_BASIC_TEAM_CTX(team);
    ucc_status_t            status;

    status = ucc_tl_team_create_multiple(team->team_create_req);
    if (status == UCC_OK) {
        team->tl_ucp_team = NULL;
        team->tl_nccl_team = NULL;
        if ((ctx->tl_nccl_ctx != NULL) &&
            (team->team_create_req->descs[1].status == UCC_OK)) {
            team->tl_nccl_team = team->team_create_req->descs[1].team;
            cl_info(ctx->super.super.lib, "initialized nccl team");
        }
        if (team->team_create_req->descs[0].status != UCC_OK) {
            if (team->tl_nccl_team) {
                UCC_TL_CTX_IFACE(ctx->tl_nccl_ctx)
                    ->team.destroy(&team->tl_nccl_team->super);
                team->tl_nccl_team = NULL;
            }
            cl_error(ctx->super.super.lib, "failed to create tl ucp team");
        }
        team->tl_ucp_team = team->team_create_req->descs[0].team;
        status            = team->team_create_req->descs[0].status;
        ucc_team_multiple_req_free(team->team_create_req);
        team->team_create_req = NULL;
    }
    return status;
}
