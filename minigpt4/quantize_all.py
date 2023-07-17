import argparse
import minigpt4_library
from pathlib import Path
from quantize import quantize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantize minigpt4.cpp model file')
    parser.add_argument('src_path', help='Path to checkpoint file')
    parser.add_argument('dst_path', help='Path to output folder containing quantized models')
    parser.add_argument('-p', '--dst_prefix', help='Prefix for output files', type=str, default='minigpt4-13B-')
    args = parser.parse_args()

    src_path = args.src_path
    dst_path = args.dst_path
    dst_prefix = args.dst_prefix

    if not Path(args.src_path).exists():
        print(f'File does not exist: {src_path}')
        exit(1)

    lib = minigpt4_library.load_library()
    for e in minigpt4_library.DataType:
        if e == minigpt4_library.DataType.F32 or e == minigpt4_library.DataType.I32 or e == minigpt4_library.DataType.L64:
            continue
        suffix = str(e).removeprefix('DataType.').lower()
        quantize(lib, src_path, dst_path + f'{dst_prefix}{suffix}.bin', e)
        print(f'Finished quantization, wrote to {dst_path}')
