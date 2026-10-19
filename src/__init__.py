import os
import sys

# On macOS, torch and faiss-cpu each bundle their own libomp, and loading both aborts
# the process ("OMP: Error #15") in an order-dependent way. Allowing the duplicate runtime
# is safe only without competing OpenMP thread pools, so OpenMP is also pinned to one
# thread (bulk encoding runs on MPS, which does not use OpenMP). tests/test_index.py
# checks faiss results against numpy across a torch call to guard this.
# Must run before torch or faiss is imported, hence the package __init__.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
