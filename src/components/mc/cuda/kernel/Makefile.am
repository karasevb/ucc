#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

NVCC = nvcc
NVCCFLAGS = ${AM_CPPFLAGS} ${UCS_CPPFLAGS} -I${UCC_TOP_BUILDDIR} -I${UCC_TOP_SRCDIR}/src -I${UCC_TOP_BUILDDIR}/src --compiler-options -fno-rtti,-fno-exceptions

LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(NVCC) -c $< -o $@  $(NVCCFLAGS) $(NVCC_ARCH)

.cu.lo:
	/bin/bash $(top_srcdir)/cuda_lt.sh "$(LIBTOOL)" $@ $(NVCC) -c  $< $(NVCCFLAGS) $(NVCC_ARCH)

comp_noinst = libucc_mc_cuda_kernels.la

libucc_mc_cuda_kernels_la_SOURCES  = mc_cuda_reduce.cu
libucc_mc_cuda_kernels_la_CPPFLAGS =

noinst_LTLIBRARIES = $(comp_noinst)
