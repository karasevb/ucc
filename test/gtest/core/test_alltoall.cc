/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, int, int>;
using Param_1 = std::tuple<int, int>;

class test_alltoall : public UccCollArgs, public ucc::test
{
public:
    UccCollCtxVec data_init(int nprocs, ucc_datatype_t dtype,
                                     size_t count)
    {
        UccCollCtxVec ctxs(nprocs);
        for (auto i = 0; i < nprocs; i++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            ctxs[i] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[i]->args = coll;

            coll->mask = 0;
            coll->coll_type = UCC_COLL_TYPE_ALLTOALL;
            coll->src.info.mem_type = mem_type;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dtype;
            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count   = (ucc_count_t)count;
            coll->dst.info.datatype = dtype;

            UCC_CHECK(ucc_mc_alloc(&ctxs[i]->init_buf,
                                   ucc_dt_size(dtype) * count * nprocs,
                                   UCC_MEMORY_TYPE_HOST));
            for (int r = 0; r < nprocs; r++) {
                size_t rank_size = ucc_dt_size(dtype) * count;
                alltoallx_init_buf(r, i,
                                   (uint8_t*)ctxs[i]->init_buf + r * rank_size,
                                   rank_size);
            }

            UCC_CHECK(ucc_mc_alloc(&coll->dst.info.buffer,
                      ucc_dt_size(dtype) * count * nprocs, mem_type));
            if (TEST_INPLACE == inplace) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
                ucc_mc_memcpy(coll->dst.info.buffer, ctxs[i]->init_buf,
                              ucc_dt_size(dtype) * count, mem_type,
                              UCC_MEMORY_TYPE_HOST);
            } else {
                UCC_CHECK(ucc_mc_alloc(&coll->src.info.buffer,
                          ucc_dt_size(dtype) * count * nprocs, mem_type));
                ucc_mc_memcpy(coll->src.info.buffer, ctxs[i]->init_buf,
                              ucc_dt_size(dtype) * count * nprocs, mem_type,
                              UCC_MEMORY_TYPE_HOST);
            }
        }
        return ctxs;
    }
    void data_fini(UccCollCtxVec ctxs)
    {
        for (gtest_ucc_coll_ctx_t* ctx : ctxs) {
            ucc_coll_args_t* coll = ctx->args;
            if (coll->src.info.buffer) { /* no inplace */
                UCC_CHECK(ucc_mc_free(coll->src.info.buffer, mem_type));
            }
            UCC_CHECK(ucc_mc_free(coll->dst.info.buffer, mem_type));
            UCC_CHECK(ucc_mc_free(ctx->init_buf, UCC_MEMORY_TYPE_HOST));
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }
    void data_validate(UccCollCtxVec ctxs)
    {
        std::vector<uint8_t *> dsts(ctxs.size());

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                size_t buf_size =
                        ucc_dt_size(ctxs[r]->args->dst.info.datatype) *
                        (size_t)ctxs[r]->args->dst.info.count * ctxs.size();
                UCC_CHECK(ucc_mc_alloc((void**)&dsts[r], buf_size,
                                       UCC_MEMORY_TYPE_HOST));
                ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info.buffer,
                              buf_size, UCC_MEMORY_TYPE_HOST, mem_type);
            }
        } else {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (uint8_t *)(ctxs[r]->args->dst.info.buffer);
            }
        }
        for (int r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t* coll = ctxs[r]->args;

            for (int i = 0; i < ctxs.size(); i++) {
                size_t rank_size = ucc_dt_size(coll->dst.info.datatype) *
                        (size_t)coll->dst.info.count;
                EXPECT_EQ(0,
                          alltoallx_validate_buf(
                              i, r, (uint8_t*)dsts[r] + rank_size * i,
                              rank_size));
            }
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                ucc_mc_free((void*)dsts[r], UCC_MEMORY_TYPE_HOST);
            }
        }
        return;
    }
};

class test_alltoall_0 : public test_alltoall,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoall_0, single)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    const int count = std::get<2>(GetParam());

    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    this->set_inplace(TEST_NO_INPLACE);
    UccCollCtxVec args = data_init(size, (ucc_datatype_t)dtype, count);
    UccReq    req(team, args);
    req.start();
    req.wait();
    data_validate(args);
    data_fini(args);
}

#ifdef HAVE_CUDA
UCC_TEST_P(test_alltoall_0, single_cuda)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    const int count = std::get<2>(GetParam());

    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    set_inplace(TEST_NO_INPLACE);
    set_mem_type(UCC_MEMORY_TYPE_CUDA);
    UccCollCtxVec args = data_init(size, (ucc_datatype_t)dtype, count);
    UccReq    req(team, args);
    req.start();
    req.wait();
    data_validate(args);
    data_fini(args);
}
#endif

INSTANTIATE_TEST_CASE_P(
    ,
    test_alltoall_0,
    ::testing::Combine(
        ::testing::Values(2,7), // nprocs
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1), // dtype
        ::testing::Values(1,3))); // count
        /* // Not supported
        ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE) // inplace
        */

/* TODO: enable parallel teams once it is supported */
/*
class test_alltoall_1 : public test_alltoall,
        public ::testing::WithParamInterface<Param_1> {};

UCC_TEST_P(test_alltoall_1, multiple)
{
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<0>(GetParam());
    const int count = std::get<1>(GetParam());

    std::vector<UccReq> reqs;
    std::vector<UccCollCtxVec> args;
    for (auto &team : UccJob::getStaticTeams()) {
        UccCollCtxVec arg = data_init(team->procs.size(),
                                       dtype, count);
        args.push_back(arg);
        reqs.push_back(UccReq(team, arg));
    }
    UccReq::startall(reqs);
    UccReq::waitall(reqs);

    for (auto arg : args) {
        data_validate(arg);
        data_fini(arg);
    }
}

INSTANTIATE_TEST_CASE_P(
    ,
    test_alltoall_1,
    ::testing::Combine(
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1), // dtype
        ::testing::Values(1,3,8))); // count
#endif
*/
