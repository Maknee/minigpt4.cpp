import argparse
import minigpt4_library
from pathlib import Path

def quantize(lib, src_path, dst_path, quantization):
    lib.minigpt4_set_verbosity(minigpt4_library.Verbosity.INFO)
    lib.minigpt4_quantize_model(src_path, dst_path, quantization)

if __name__ == "__main__":
    name_to_data_type = {}
    for e in minigpt4_library.DataType:
        name_to_data_type[str(e).removeprefix('DataType.')] = e

    parser = argparse.ArgumentParser(description='Quantize minigpt4.cpp model file')
    parser.add_argument('src_path', help='Path to checkpoint file')
    parser.add_argument('dst_path', help='Path to output file')
    parser.add_argument('quantization', help='Quantization, one of ' + ', '.join(name_to_data_type.keys()), type=str, choices=list(name_to_data_type.keys()), default='Q4_1')
    args = parser.parse_args()

    src_path = args.src_path
    dst_path = args.dst_path
    quantization = args.quantization

    quantization = name_to_data_type[quantization]

    if not Path(args.src_path).exists():
        print(f'File does not exist: {src_path}')
        exit(1)

    lib = minigpt4_library.load_library()
    quantize(lib, src_path, dst_path, quantization)
    print(f'Finished quantization, wrote to {dst_path}')
