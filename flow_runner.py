import flow_recon_simulated as frs

import asyncio
import argparse




#if __name__ == "__main__":
#    asyncio.run(frs.big_runner([0.75], [1e-2], True))



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

CLI.add_argument(
    "--noise",
    nargs=1,
    type=float,
    default=None
)

CLI.add_argument(
    "--nspokes",
    nargs=1,
    type=int,
    default=None
)

CLI.add_argument(
    "--samp",
    nargs=1,
    type=int,
    default=None
)

CLI.add_argument(
    "--is_add",
    nargs=1,
    type=bool,
    default=[False]
)

args = CLI.parse_args()

if args.wexp is None:
    raise ValueError('wexp is required')
if args.lamda is None:
    raise ValueError('lamda is required')
if args.noise is None:
    raise ValueError('noise is required')
if args.nspokes is None:
    raise ValueError('nspokes is required')
if args.samp is None:
    raise ValueError('samp is required')
if args.is_add is None:
    raise ValueError('is_add is required')

if len(args.noise) != 1:
    raise ValueError('noise must be a single value')
if len(args.nspokes) != 1:
    raise ValueError('nspokes must be a single value')
if len(args.samp) != 1:
    raise ValueError('samp must be a single value')
if len(args.is_add) != 1:
    raise ValueError('is_add must be a single value')

print(f"Running with wexp={args.wexp}")
print(f"Running with lamda={args.lamda}")
print(f"Running with noise={args.noise[0]:.2e}")
print(f"Running with nspokes={args.nspokes[0]}")
print(f"Running with samp={args.samp[0]}")
print(f"Running with is_add={args.is_add[0]}")

asyncio.run(frs.big_runner(args.wexp, args.lamda, 
    args.noise[0], args.nspokes[0], args.samp[0], args.is_add[0]))
