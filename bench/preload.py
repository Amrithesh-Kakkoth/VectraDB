import argparse
import os
import sys
import time
import numpy as np

# Import Python client
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "python-client"))
try:
    from vectradb_simple import VectraDB  # type: ignore
except Exception as e:
    print("Error: could not import python-client/vectradb_simple.")
    print("Make sure the python-client is present and importable. Error:", e)
    sys.exit(1)


def gen_dataset(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--prefix", default="bench")
    p.add_argument("--batch-size", type=int, default=1000)
    args = p.parse_args()

    X = gen_dataset(args.n, args.dim)
    client = VectraDB(host=args.host, port=args.port)

    t0 = time.time()
    inserted = 0
    while inserted < args.n:
        batch_end = min(inserted + args.batch_size, args.n)
        items = [
            {
                "id": f"{args.prefix}-{i}",
                "vector": X[i].tolist(),
                "tags": {"bench": "true"},
            }
            for i in range(inserted, batch_end)
        ]
        response = client.batch_upsert(items)
        failures = [status for status in response.statuses if status.code != "OK"]
        if failures:
            first = failures[0]
            raise RuntimeError(
                f"Batch upsert failed for {len(failures)} items, first={first.id}:{first.code}:{first.message}"
            )
        inserted = batch_end
        if inserted % 1000 == 0 or inserted == args.n:
            elapsed = time.time() - t0
            rate = inserted / max(elapsed, 1e-6)
            print(f"Inserted {inserted}/{args.n} in {elapsed:.1f}s ({rate:.0f} vec/s)")
    total = time.time() - t0
    print(f"Done. Inserted {args.n} vectors in {total:.1f}s ({args.n/max(total,1e-6):.0f} vec/s)")


if __name__ == "__main__":
    main()
