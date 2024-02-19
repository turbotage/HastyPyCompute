import flow_recon_simulated as frs

import asyncio
import argparse

CLI=argparse.ArgumentParser()

CLI.add_argument(
    "--wexp",
    nargs="*",
    type=float,
    default=None
)
CLI.add_argument(
    "--lamda",
    nargs="*",
    type=float,
    default=None
)

args = CLI.parse_args()

if args.wexp is None:
    raise ValueError('wexp is required')
if args.lamda is None:
    raise ValueError('lamda is required')

print(f"Running with wexp={args.wexp} and lamda={args.lamda}")

asyncio.run(frs.big_runner(args.wexp, args.lamda, True))