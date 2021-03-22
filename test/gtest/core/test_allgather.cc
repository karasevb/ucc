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

class test_allgather : public UccCollArgs, public ucc::test
{
public:
    UccCollArgsVec data_init(int nprocs, ucc_datatype_t dtype,
                                     size_t count) {
        UccCollArgsVec args(nprocs);
        for (auto r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));
            coll->mask = 0;
            coll->coll_type = UCC_COLL_TYPE_ALLGATHER;
            coll->src.info.mem_type = UCC_MEMORY_TYPE_HOST;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dtype;
            coll->dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
            coll->dst.info.count   = (ucc_count_t)count;
            coll->dst.info.datatype = dtype;

            UCC_CHECK(ucc_mc_alloc(&coll->src.info.buffer,
                      ucc_dt_size(dtype) * count,
                      UCC_MEMORY_TYPE_HOST));
            UCC_CHECK(ucc_mc_alloc(&coll->dst.info.buffer,
                      ucc_dt_size(dtype) * count * nprocs,
                      UCC_MEMORY_TYPE_HOST));
            for (int i = 0; i < ucc_dt_size(dtype) * count; i++) {
                uint8_t *sbuf = (uint8_t*)coll->src.info.buffer;
                sbuf[i] = r;
            }
            args[r] = coll;
        }
        return args;
    }
    void data_fini(UccCollArgsVec args) {
        for (ucc_coll_args_t* coll : args) {
            ucc_memory_type_t mtype;
            UCC_CHECK(ucc_mc_type(coll->src.info.buffer, &mtype));
            UCC_CHECK(ucc_mc_free(coll->src.info.buffer, mtype));
            UCC_CHECK(ucc_mc_type(coll->dst.info.buffer, &mtype));
            UCC_CHECK(ucc_mc_free(coll->dst.info.buffer, mtype));
            free(coll);
        }
        args.clear();
    }
    void data_validate(UccCollArgsVec args) {
        for (int i = 0; i < args.size(); i++) {
            ucc_coll_args_t* coll = args[i];
            uint8_t *rbuf = (uint8_t*)coll->dst.info.buffer;
            for (int r = 0; r < args.size(); r++) {
                size_t rank_size = ucc_dt_size((args[r])->src.info.datatype) *
                        (args[r])->src.info.count;
                for (int i = 0; i < rank_size; i++) {
                    EXPECT_EQ(r, rbuf[r*rank_size + i]);
                }
            }
        }
    }
};

class test_allgather_0 : public test_allgather,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_allgather_0, single)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    const int count = std::get<2>(GetParam());

    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    UccCollArgsVec args = data_init(size, (ucc_datatype_t)dtype, count);
    UccReq    req(team, args);
    req.start();
    req.wait();
    data_validate(args);
    data_fini(args);
}

INSTANTIATE_TEST_CASE_P(
    ,
    test_allgather_0,
    ::testing::Combine(
        ::testing::Values(1,3,16), // nprocs
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1), // dtype
        ::testing::Values(1,3,8))); // count
