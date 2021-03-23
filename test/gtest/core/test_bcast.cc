/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param = std::tuple<int, int, int, int>;

class test_bcast : public UccCollArgs, public ucc::test
{
private:
    int root;
public:
    UccCollCtxVec data_init(int nprocs, ucc_datatype_t dtype, size_t count)
    {
        UccCollCtxVec ctxs(nprocs);
        for (auto r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            ctxs[r] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->mask = 0;
            coll->coll_type = UCC_COLL_TYPE_BCAST;
            coll->src.info.mem_type = mem_type;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dtype;
            coll->root = root;

            ctxs[r]->rbuf_size = ucc_dt_size(dtype) * count;

            UCC_CHECK(ucc_mc_alloc(&coll->src.info.buffer, ctxs[r]->rbuf_size,
                                   mem_type));
            if (r == root) {
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->init_buf, ctxs[r]->rbuf_size,
                                       UCC_MEMORY_TYPE_HOST));
                for (int i = 0; i < ctxs[r]->rbuf_size; i++) {
                    uint8_t *sbuf = (uint8_t*)ctxs[r]->init_buf;
                    sbuf[i] = i;
                }
                ucc_mc_memcpy(coll->src.info.buffer, ctxs[r]->init_buf,
                              ctxs[r]->rbuf_size, mem_type,
                              UCC_MEMORY_TYPE_HOST);
            }
        }
        return ctxs;
    }
    void data_fini(UccCollCtxVec ctxs)
    {
        for (gtest_ucc_coll_ctx_t* ctx : ctxs) {
            ucc_coll_args_t* coll = ctx->args;
            UCC_CHECK(ucc_mc_free(coll->src.info.buffer, mem_type));
            UCC_CHECK(ucc_mc_free(ctx->init_buf, UCC_MEMORY_TYPE_HOST));
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }
    void data_validate(UccCollCtxVec ctxs)
    {
        int root = ctxs[0]->args->root;
        uint8_t *dsts;

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
                UCC_CHECK(ucc_mc_alloc((void**)&dsts, ctxs[root]->rbuf_size,
                                       UCC_MEMORY_TYPE_HOST));
                ucc_mc_memcpy(dsts, ctxs[root]->args->src.info.buffer,
                              ctxs[root]->rbuf_size, UCC_MEMORY_TYPE_HOST, mem_type);
        } else {
            dsts = (uint8_t*)ctxs[root]->args->src.info.buffer;
        }
        for (int r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t* coll = ctxs[r]->args;
            if (coll->root == r) {
                continue;
            }
            for (int i = 0; i < ctxs[r]->rbuf_size; i++) {
                EXPECT_EQ(i, dsts[i]);
                //printf("%d ", dsts[i]);
            }
            //printf("\n");
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            ucc_mc_free((void*)dsts, UCC_MEMORY_TYPE_HOST);
        }
        return;
    }
    void set_root(int _root)
    {
        root = _root;
    }
};

class test_bcast_0 : public test_bcast,
        public ::testing::WithParamInterface<Param> {};

UCC_TEST_P(test_bcast_0, single_host)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    const int count = std::get<2>(GetParam());
    const int root = std::get<3>(GetParam());

    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    set_mem_type(UCC_MEMORY_TYPE_HOST);
    set_root(root);
    UccCollCtxVec args = data_init(size, (ucc_datatype_t)dtype, count);
    UccReq    req(team, args);
    req.start();
    req.wait();
    data_validate(args);
    data_fini(args);
}

#ifdef HAVE_CUDA
UCC_TEST_P(test_bcast_0, single_cuda)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    const int count = std::get<2>(GetParam());
    const int root = std::get<3>(GetParam());

    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    set_mem_type(UCC_MEMORY_TYPE_CUDA);
    set_root(root);
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
    test_bcast_0,
    ::testing::Combine(
        ::testing::Values(2,7), // nprocs
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_UINT32 + 1), // dtype
        ::testing::Values(1,3), // count
        ::testing::Values(0,1))); // root
