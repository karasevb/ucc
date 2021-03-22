/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include "test_mc_reduce.h"
//#include <components/mc/cpu/mc_cpu_reduce.h>
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

/*
INSTANTIATE_TEST_CASE_P(
    ,
    test_allreduce_0,
    ::testing::Combine(
        ::testing::Values(3), // nprocs
        ::testing::Range((int)UCC_DT_INT32, (int)UCC_DT_INT32 + 1), // dtype
        ::testing::Range((int)UCC_OP_SUM, (int)UCC_OP_SUM + 1),
        ::testing::Values(2)));
*/
/** NEW **/

template<typename T>
class test_allreduce : public UccCollArgs, public testing::Test {
  public:
    test_allreduce() {
        // tmp
        mem_type = UCC_MEMORY_TYPE_HOST;
    }

    UccCollArgsVec data_init(int nprocs, ucc_datatype_t dt, size_t count) {
        UccCollArgsVec args(nprocs);
        for (int r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));
            coll->mask = UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS;
            coll->coll_type = UCC_COLL_TYPE_ALLREDUCE;

            coll->src.info.mem_type = mem_type;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dt;

            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count   = (ucc_count_t)count;
            coll->dst.info.datatype = dt;

            UCC_CHECK(ucc_mc_alloc(&coll->src.info.buffer,
                      ucc_dt_size(dt) * count, mem_type));
            UCC_CHECK(ucc_mc_alloc(&coll->dst.info.buffer,
                      ucc_dt_size(dt) * count, mem_type));

            for (int i = 0; i < /*ucc_dt_size(dt) * */count; i++) {
                typename T::type * sbuf;
                sbuf = (typename T::type *)(coll->src.info.buffer);
                sbuf[i] = (typename T::type)(r);
                //uint8_t *sbuf = (uint8_t*)coll->src.info.buffer;
                //sbuf[i] = r+i;
            }
            args[r] = coll;
            coll->reduce.predefined_op = T::redop;
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
        size_t count = (args[0])->src.info.count;

        typename T::type * tmp;

        for (int i = 0; i < count; i++) {
            typename T::type res = ((typename T::type *)((args[0])->src.info.buffer))[i];
            for (int r = 1; r < args.size(); r++) {
                tmp = (typename T::type *)((args[r])->src.info.buffer);
                res = T::do_op(res,tmp[i]);
            }
            for (int r = 0; r < args.size(); r++) {
                //EXPECT_EQ(res, ((typename T::type *)(coll->dst.info.buffer))[i]);
                printf("%p %d: %u - %u", args[r], r,
                       res, ((typename T::type *)((args[r])->dst.info.buffer))[i]);
                if (res != ((typename T::type *)((args[r])->dst.info.buffer))[i]) {
                     printf(" <--- not equal\n");
                } else {
                    printf("\n");
                }

            }
        }
        return;
    }
};

using TypesOps = ::testing::Types<ReductionTest<UCC_DT_INT32, max>>;

TYPED_TEST_CASE(test_allreduce, TypesOps);
/*
TYPED_TEST(test_allreduce, single) {
    const int size = 2;
    const int count = 2;

    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    UccCollArgsVec args = this->data_init(size, TypeParam::dt, count);
    UccReq    req(team, args);
    req.start();
    req.wait();
    this->data_validate(args);
    this->data_fini(args);
}
*/

TYPED_TEST(test_allreduce, multi) {
    const int count = 2;

    std::vector<UccReq> reqs;
    std::vector<UccCollArgsVec> args;
    for (auto &team : UccJob::getStaticTeams()) {
        UccCollArgsVec arg = this->data_init(team->procs.size(),
                                       TypeParam::dt, count);
        args.push_back(arg);
        reqs.push_back(UccReq(team, arg));
    }
    /*
    for (int i = 0; i < args.size(); i++) {
        UccCollArgsVec arg = args[i];
        for (auto coll: arg) {
            printf("%d: ", i);
            for (int c = 0; c < count; c++) {
                printf("%d ", ((int *)(coll->src.info.buffer))[c]);
            }
            printf("\n");
        }
    }*/
    UccReq::startall(reqs);
    UccReq::waitall(reqs);

    for (auto arg : args) {
        printf("%p\n", &arg);
        this->data_validate(arg);
        this->data_fini(arg);
    }
}

