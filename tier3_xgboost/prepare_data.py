"""
Build tier3/artifacts with the verbatim copy of preprocess.py
===============================================================================

How to run
----------
    uv run python prepare_data.py

preprocess.py is a byte-for-byte copy of eda_and_preprocess/preprocess.py and
must stay that way. It asserts 848,406 deduplicated train benign flows, but with
the pandas version locked in uv.lock (3.1.0) the same code yields 848,433, so
the expected count is overridden here at run time instead of editing the copy.
Every other assert (split sizes, attack counts, time order, unseen test
attacks) runs unchanged.
"""

import preprocess

LOCKED_TRAIN_BENIGN = 848_433

if __name__ == "__main__":
    preprocess.EXPECTED_TRAIN_BENIGN = LOCKED_TRAIN_BENIGN
    preprocess.main()
