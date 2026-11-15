"""Read the resident GSM8K parquet in the separate Arrow environment."""

import json
import sys

import pyarrow.parquet as pq


if __name__ == "__main__":
    rows = pq.read_table(sys.argv[1], columns=["question", "answer"]).to_pylist()
    json.dump(rows, sys.stdout)
