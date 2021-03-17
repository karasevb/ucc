/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, int>;

template <class T>
class test_alltoallv : public UccCollArgs, public ucc::test
{
public:
    uint64_t coll_mask;
    uint64_t coll_flags;

    test_alltoallv() : coll_mask(0), coll_flags(0) {}
    UccCollCtxVec data_init(int nprocs, ucc_datatype_t dtype,
                             size_t count) {
        int buf_count;
        UccCollCtxVec ctxs(nprocs);

        for (auto r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            ctxs[r] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->coll_type = UCC_COLL_TYPE_ALLTOALLV;
            coll->mask = coll_mask;
            coll->flags = coll_flags;

            coll->src.info_v.mem_type = mem_type;
            coll->src.info_v.counts = (ucc_count_t*)malloc(sizeof(T) * nprocs);
            coll->src.info_v.datatype = dtype;
            coll->src.info_v.displacements = (ucc_aint_t*)malloc(sizeof(T) * nprocs);

            coll->dst.info_v.mem_type = mem_type;
            coll->dst.info_v.counts = (ucc_count_t*)malloc(sizeof(T) * nprocs);
            coll->dst.info_v.datatype = dtype;
            coll->dst.info_v.displacements = (ucc_aint_t*)malloc(sizeof(T) * nprocs);

            buf_count = 0;
            for (int i = 0; i < nprocs; i++) {
                int rank_count = (nprocs + r - i) * count;
                ((T*)coll->src.info_v.counts)[i] = rank_count;
                ((T*)coll->src.info_v.displacements)[i] = buf_count;
                buf_count += rank_count;
            }

            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->init_buf,
                                   buf_count * ucc_dt_size(dtype),
                                   UCC_MEMORY_TYPE_HOST));
            for (int i = 0; i < nprocs; i++) {
                alltoallx_init_buf(r, i, (uint8_t*)ctxs[r]->init_buf +
                               ((T*)coll->src.info_v.displacements)[i] * ucc_dt_size(dtype),
                               ((T*)coll->src.info_v.counts)[i] * ucc_dt_size(dtype));
            }
            UCC_CHECK(ucc_mc_alloc(&coll->src.info_v.buffer,
                                   buf_count * ucc_dt_size(dtype),
                                   mem_type));
            ucc_mc_memcpy(coll->src.info_v.buffer, ctxs[r]->init_buf,
                          buf_count * ucc_dt_size(dtype), mem_type,
                          UCC_MEMORY_TYPE_HOST);

            /* TODO: inplace support */

            buf_count = 0;
            for (int i = 0; i < nprocs; i++) {
                int rank_count = (nprocs - r + i) * count;
                ((T*)coll->dst.info_v.counts)[i] = rank_count;
                ((T*)coll->dst.info_v.displacements)[i] = buf_count;
                buf_count += rank_count;
            }
            ctxs[r]->rbuf_size = buf_count * ucc_dt_size(dtype);
            UCC_CHECK(ucc_mc_alloc(&coll->dst.info_v.buffer,
                                   buf_count * ucc_dt_size(dtype),
                                   mem_type));
        }
        return ctxs;
    }
    void data_validate(UccCollCtxVec ctxs)
    {
        std::vector<uint8_t *> dsts(ctxs.size());

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                UCC_CHECK(ucc_mc_alloc((void**)&dsts[r], ctxs[r]->rbuf_size,
                                       UCC_MEMORY_TYPE_HOST));
                ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info_v.buffer,
                              ctxs[r]->rbuf_size, UCC_MEMORY_TYPE_HOST, mem_type);
            }
        } else {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (uint8_t *)(ctxs[r]->args->dst.info.buffer);
            }
        }
        for (int r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t* coll = ctxs[r]->args;
            for (int i = 0; i < ctxs.size(); i++) {
                size_t rank_size = ucc_dt_size(coll->dst.info_v.datatype) *
                        (size_t)((T*)coll->dst.info_v.counts)[i];
                size_t rank_offs = ucc_dt_size(coll->dst.info_v.datatype) *
                        (size_t)((T*)coll->dst.info_v.displacements)[i];
                EXPECT_EQ(0,
                          alltoallx_validate_buf(r, i,
                          (uint8_t*)dsts[r] + rank_offs,
                          rank_size));
            }
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                ucc_mc_free((void*)dsts[r], UCC_MEMORY_TYPE_HOST);
            }
        }
    }
    void data_fini(UccCollCtxVec ctxs)
    {
        for (gtest_ucc_coll_ctx_t* ctx : ctxs) {
            ucc_coll_args_t* coll = ctx->args;
            UCC_CHECK(ucc_mc_free(coll->src.info_v.buffer, mem_type));
            free(coll->src.info_v.counts);
            free(coll->src.info_v.displacements);
            UCC_CHECK(ucc_mc_free(coll->dst.info_v.buffer, mem_type));
            free(coll->dst.info_v.counts);
            free(coll->dst.info_v.displacements);
            UCC_CHECK(ucc_mc_free(ctx->init_buf, UCC_MEMORY_TYPE_HOST));
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }
};

class test_alltoallv_0 : public test_alltoallv <uint64_t>,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoallv_0, single)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    UccTeam_h team = UccJob::getStaticJob()->create_team(size);

    coll_mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll_flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                 UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;

    UccCollCtxVec args = data_init(size, (ucc_datatype_t)dtype, 1);

    UccReq    req(team, args);
    req.start();
    req.wait();

    data_validate(args);
    data_fini(args);
}


class test_alltoallv_1 : public test_alltoallv <uint32_t>,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoallv_1, single)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    UccCollCtxVec args = data_init(size, (ucc_datatype_t)dtype, 1);

    UccReq    req(team, args);
    req.start();
    req.wait();

    data_validate(args);
    data_fini(args);
}

#ifdef HAVE_CUDA
UCC_TEST_P(test_alltoallv_0, single_cuda)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    UccTeam_h team = UccJob::getStaticJob()->create_team(size);

    coll_mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll_flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                 UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
    set_inplace(TEST_NO_INPLACE);
    set_mem_type(UCC_MEMORY_TYPE_CUDA);
    UccCollCtxVec args = data_init(size, (ucc_datatype_t)dtype, 1);

    UccReq    req(team, args);
    req.start();
    req.wait();

    data_validate(args);
    data_fini(args);
}

UCC_TEST_P(test_alltoallv_1, single_cuda)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    set_inplace(TEST_NO_INPLACE);
    set_mem_type(UCC_MEMORY_TYPE_CUDA);
    UccCollCtxVec args = data_init(size, (ucc_datatype_t)dtype, 1);

    UccReq    req(team, args);
    req.start();
    req.wait();

    data_validate(args);
    data_fini(args);
}
#endif


INSTANTIATE_TEST_CASE_P(
        64,
        test_alltoallv_0,
        ::testing::Combine(
            ::testing::Values(2,7), // nprocs
            ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1))); // dtype

INSTANTIATE_TEST_CASE_P(
        32,
        test_alltoallv_1,
        ::testing::Combine(
            ::testing::Values(2,7), // nprocs
            ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1))); // dtype

/* TODO: enable parallel teams once it is supported */
/*
class test_alltoallv_2 : public test_alltoallv<uint64_t>,
        public ::testing::WithParamInterface<int> {};

class test_alltoallv_3 : public test_alltoallv<uint32_t>,
        public ::testing::WithParamInterface<int> {};

UCC_TEST_P(test_alltoallv_2, multiple)
{
    const ucc_datatype_t dtype = (ucc_datatype_t)(GetParam());
    std::vector<UccReq> reqs;
    std::vector<UccCollCtxVec> args;

    coll_mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll_flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                 UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;

    for (auto &team : UccJob::getStaticTeams()) {
        UccCollCtxVec arg = data_init(team->procs.size(),
                                       dtype, 1);
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


UCC_TEST_P(test_alltoallv_3, multiple)
{
    const ucc_datatype_t dtype = (ucc_datatype_t)(GetParam());
    std::vector<UccReq> reqs;
    std::vector<UccCollCtxVec> args;

    for (auto &team : UccJob::getStaticTeams()) {
        UccCollCtxVec arg = data_init(team->procs.size(),
                                       dtype, 1);
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
        64,
        test_alltoallv_2,
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1)); // dtype

INSTANTIATE_TEST_CASE_P(
        32,
        test_alltoallv_3,
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1)); // dtype
*/
