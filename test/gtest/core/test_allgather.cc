/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, int, int, gtest_ucc_inplace_t>;

class test_allgather : public UccCollArgs, public ucc::test
{
public:
    UccCollCtxVec data_init(int nprocs, ucc_datatype_t dtype,
                                     size_t count)
    {
        UccCollCtxVec ctxs(nprocs);
        for (auto r = 0; r < nprocs; r++) {

            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));
            ctxs[r] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->mask = 0;
            coll->flags = 0;
            coll->coll_type = UCC_COLL_TYPE_ALLGATHER;
            coll->src.info.mem_type = mem_type;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dtype;
            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count   = (ucc_count_t)count;
            coll->dst.info.datatype = dtype;

            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->init_buf,
                                   ucc_dt_size(dtype) * count,
                                   UCC_MEMORY_TYPE_HOST));
            for (int i = 0; i < ucc_dt_size(dtype) * count; i++) {
                uint8_t *sbuf = (uint8_t*)ctxs[r]->init_buf;
                sbuf[i] = r;
            }

            ctxs[r]->rbuf_size = ucc_dt_size(dtype) * count * nprocs;
            UCC_CHECK(ucc_mc_alloc(&coll->dst.info.buffer, ctxs[r]->rbuf_size,
                      mem_type));
            if (TEST_INPLACE == inplace) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
                ucc_mc_memcpy((void*)((ptrdiff_t)coll->dst.info.buffer +
                                      r * count * ucc_dt_size(dtype)),
                              ctxs[r]->init_buf, ucc_dt_size(dtype) * count,
                              mem_type, UCC_MEMORY_TYPE_HOST);
            } else {
                UCC_CHECK(ucc_mc_alloc(&coll->src.info.buffer,
                          ucc_dt_size(dtype) * count, mem_type));
                ucc_mc_memcpy(coll->src.info.buffer, ctxs[r]->init_buf,
                              ucc_dt_size(dtype) * count, mem_type,
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
                UCC_CHECK(ucc_mc_alloc((void**)&dsts[r], ctxs[r]->rbuf_size,
                                       UCC_MEMORY_TYPE_HOST));
                ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info.buffer,
                              ctxs[r]->rbuf_size, UCC_MEMORY_TYPE_HOST, mem_type);
            }
        } else {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (uint8_t *)(ctxs[r]->args->dst.info.buffer);
            }
        }
        for (int i = 0; i < ctxs.size(); i++) {
            uint8_t *rbuf = dsts[i];
            for (int r = 0; r < ctxs.size(); r++) {
                size_t rank_size = ucc_dt_size((ctxs[r])->args->src.info.datatype) *
                        (ctxs[r])->args->src.info.count;
                for (int i = 0; i < rank_size; i++) {
                    EXPECT_EQ(r, rbuf[r*rank_size + i]);
                }
            }
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                ucc_mc_free((void*)dsts[r], UCC_MEMORY_TYPE_HOST);
            }
        }
    }
};

class test_allgather_0 : public test_allgather,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_allgather_0, single_host)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    const int count = std::get<2>(GetParam());
    const gtest_ucc_inplace_t inplace = std::get<3>(GetParam());

    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    set_inplace(inplace);
    set_mem_type(UCC_MEMORY_TYPE_HOST);
    UccCollCtxVec args = data_init(size, dtype, count);
    UccReq    req(team, args);
    req.start();
    req.wait();
    data_validate(args);
    data_fini(args);
}

#ifdef HAVE_CUDA
UCC_TEST_P(test_allgather_0, single_cuda)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    const int count = std::get<2>(GetParam());
    const gtest_ucc_inplace_t inplace = std::get<3>(GetParam());

    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    set_inplace(inplace);
    set_mem_type(UCC_MEMORY_TYPE_CUDA);
    UccCollCtxVec args = data_init(size, dtype, count);
    UccReq    req(team, args);
    req.start();
    req.wait();
    data_validate(args);
    data_fini(args);
}
#endif

INSTANTIATE_TEST_CASE_P(
    ,
    test_allgather_0,
    ::testing::Combine(
        ::testing::Values(2,7), // nprocs
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1), // dtype
        ::testing::Values(1,3), // count
        ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE)));  // inplace
