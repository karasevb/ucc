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

class test_allgatherv : public UccCollArgs, public ucc::test
{
public:
    UccCollArgsVec data_init(int nprocs, ucc_datatype_t dtype,
                                     size_t count) {
        UccCollArgsVec args(nprocs);
        for (auto r = 0; r < nprocs; r++) {
            int *counts;
            int *displs;
            size_t my_count = (nprocs - r) * count;
            size_t all_counts = 0;
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            counts = (int*)malloc(sizeof(int) * nprocs);
            displs = (int*)malloc(sizeof(int) * nprocs);

            for (int i = 0; i < nprocs; i++) {
                counts[i] = (nprocs - i) * count;
                displs[i] = all_counts;
                all_counts += counts[i];
            }
            coll->mask = 0;
            coll->coll_type = UCC_COLL_TYPE_ALLGATHERV;

            coll->src.info.mem_type = UCC_MEMORY_TYPE_HOST;
            coll->src.info.count = my_count;
            coll->src.info.datatype = dtype;

            coll->dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;
            coll->dst.info_v.counts   = (ucc_count_t*)counts;
            coll->dst.info_v.displacements = (ucc_aint_t*)displs;
            coll->dst.info_v.datatype = dtype;

            UCC_CHECK(ucc_mc_alloc(&(coll->src.info.buffer),
                      ucc_dt_size(dtype) * my_count,
                      UCC_MEMORY_TYPE_HOST));
            UCC_CHECK(ucc_mc_alloc(&(coll->dst.info_v.buffer),
                      ucc_dt_size(dtype) * all_counts,
                      UCC_MEMORY_TYPE_HOST));
            for (int i = 0; i < (ucc_dt_size(dtype) * my_count); i++) {
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
            free(coll->dst.info_v.displacements);
            free(coll->dst.info_v.counts);
            free(coll);
        }
        args.clear();
    }
    void data_validate(UccCollArgsVec args) {
        for (int i = 0; i < args.size(); i++) {
            size_t rank_size = 0;
            ucc_coll_args_t* coll = args[i];
            uint8_t *rbuf = (uint8_t*)coll->dst.info_v.buffer;
            for (int r = 0; r < args.size(); r++) {
                rbuf += rank_size;
                rank_size = ucc_dt_size((args[r])->src.info.datatype) *
                        (args[r])->src.info.count;
                for (int i = 0; i < rank_size; i++) {
                    EXPECT_EQ(r, rbuf[i]);
                }
            }
        }
    }
};

class test_allgatherv_0 : public test_allgatherv,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_allgatherv_0, single)
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
    test_allgatherv_0,
    ::testing::Combine(
        ::testing::Values(1,3,16), // nprocs
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1), // dtype
        ::testing::Values(1,3,8))); // count
