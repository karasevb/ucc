/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include <random>
#include <assert.h>

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

template<typename T>
void * TestAlltoallv::mpi_counts_to_ucc(int *mpi_counts, size_t _ncount)
{
    void *ucc_counts = (T*)malloc(sizeof(T) * _ncount);
    for (auto i = 0; i < _ncount; i++) {
        ((T*)ucc_counts)[i] = mpi_counts[i];
    }
    return ucc_counts;
}

TestAlltoallv::TestAlltoallv(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                             ucc_memory_type_t _mt, ucc_test_team_t &_team,
                             size_t _max_size,
                             ucc_test_vsize_flag_t _count_bits,
                             ucc_test_vsize_flag_t _displ_bits) :
    TestCase(_team, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count = _msgsize/dt_size;
    int rank;
    int nprocs;
    int rank_count;

    std::default_random_engine eng;
    eng.seed(test_rand_seed);
    std::uniform_int_distribution<int> urd(count/2, count);

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);

    sncounts = 0;
    rncounts = 0;
    scounts = NULL;
    sdispls = NULL;
    rcounts = NULL;
    rdispls = NULL;
    count_bits = _count_bits;
    displ_bits = _displ_bits;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);

    args.coll_type = UCC_COLL_TYPE_ALLTOALLV;

    if (TEST_INPLACE == inplace && ucc_coll_inplace_supported(args.coll_type)) {
        test_skip = TEST_SKIP_NOT_IMPL_INPLACE;
    }

    if (count_bits == TEST_FLAG_VSIZE_64BIT) {
        args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags |= UCC_COLL_ARGS_FLAG_COUNT_64BIT;
    }
    if (displ_bits == TEST_FLAG_VSIZE_64BIT) {
        args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags |= UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
    }

    scounts = (int*)malloc(sizeof(*scounts) * nprocs);
    sdispls = (int*)malloc(sizeof(*sdispls) * nprocs);
    rcounts = (int*)malloc(sizeof(*rcounts) * nprocs);
    rdispls = (int*)malloc(sizeof(*rdispls) * nprocs);

    for (auto i = 0; i < nprocs; i++) {
        rank_count = urd(eng);
        scounts[i] = rank_count;
    }

    MPI_Alltoall((void*)scounts, 1, MPI_INT,
                 (void*)rcounts, 1, MPI_INT, team.comm);

    sncounts = 0;
    rncounts = 0;
    for (auto i = 0; i < nprocs; i++) {
        assert((size_t)rcounts[i] <= count);
        sdispls[i] = sncounts;
        rdispls[i] = rncounts;
        sncounts += scounts[i];
        rncounts += rcounts[i];
    }

    if (test_max_size < (_msgsize * nprocs)) {
        test_skip = TEST_SKIP_MEM_LIMIT;
    }
    if (TEST_SKIP_NONE != skip_reduce(test_skip, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&sbuf, sncounts * dt_size, _mt));
    init_buffer(sbuf, sncounts, TEST_DT, _mt, rank);
    UCC_ALLOC_COPY_BUF(check_sbuf, UCC_MEMORY_TYPE_HOST, sbuf, _mt,
                       sncounts * dt_size);
    UCC_CHECK(ucc_mc_alloc(&rbuf, rncounts * dt_size, _mt));
    UCC_CHECK(ucc_mc_alloc(&check_rbuf, rncounts * dt_size,
                           UCC_MEMORY_TYPE_HOST));

    args.src.info_v.buffer = sbuf;
    args.src.info_v.datatype = TEST_DT;
    args.src.info_v.mem_type = _mt;

    args.dst.info_v.buffer = rbuf;
    args.dst.info_v.datatype = TEST_DT;
    args.dst.info_v.mem_type = _mt;

    if (TEST_FLAG_VSIZE_64BIT == count_bits) {
        args.src.info_v.counts =
                (ucc_count_t*)mpi_counts_to_ucc<uint64_t>(scounts, nprocs);
        args.dst.info_v.counts =
                (ucc_count_t*)mpi_counts_to_ucc<uint64_t>(rcounts, nprocs);
    } else {
        args.src.info_v.counts = (ucc_count_t*)scounts;
        args.dst.info_v.counts = (ucc_count_t*)rcounts;
    }
    if (TEST_FLAG_VSIZE_64BIT == displ_bits) {
        args.src.info_v.displacements =
                (ucc_aint_t*)mpi_counts_to_ucc<uint64_t>(sdispls, nprocs);
        args.dst.info_v.displacements =
                (ucc_aint_t*)mpi_counts_to_ucc<uint64_t>(rdispls, nprocs);
    } else {
        args.src.info_v.displacements = (ucc_aint_t*)sdispls;
        args.dst.info_v.displacements = (ucc_aint_t*)rdispls;
    }

    UCC_CHECK(ucc_collective_init(&args, &req, team.team));
}

TestAlltoallv::~TestAlltoallv()
{
    if (scounts) {
        free(scounts);
    }
    if (sdispls) {
        free(sdispls);
    }
    if (rcounts) {
        free(rcounts);
    }
    if (rdispls) {
        free(rdispls);
    }

    if (TEST_FLAG_VSIZE_64BIT == count_bits) {
        free(args.src.info_v.counts);
        free(args.dst.info_v.counts);
    }
    if (TEST_FLAG_VSIZE_64BIT == displ_bits) {
        free(args.src.info_v.displacements);
        free(args.dst.info_v.displacements);
    }
}

ucc_status_t TestAlltoallv::check()
{
    MPI_Alltoallv(check_sbuf, scounts, sdispls, ucc_dt_to_mpi(TEST_DT), check_rbuf,
                  rcounts, rdispls, ucc_dt_to_mpi(TEST_DT), team.comm);
    return compare_buffers(rbuf, check_rbuf, rncounts, TEST_DT, mem_type);
}

std::string TestAlltoallv::str()
{
    return TestCase::str() +
            " counts=" + (count_bits == TEST_FLAG_VSIZE_64BIT ? "64" : "32") +
            " displs=" + (displ_bits == TEST_FLAG_VSIZE_64BIT ? "64" : "32");
}
