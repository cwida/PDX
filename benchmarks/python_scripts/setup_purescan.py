import os
import numpy as np
from setup_settings import PURESCAN_DATA

np.random.seed(42)

# Setup random collections of vectors with PDX and Nary
def generate_synthetic_data(BLOCK_SIZES=(), dimensions=(), vectors=(), dtype=np.float32):
    if dtype not in [np.float32]:
        # for more dtype support, checkout the updates on the `main` branch
        raise ValueError('dtype must be np.float32')

    type_size = np.dtype(dtype).itemsize
    adaptive_block_size = int(256 / type_size)  # 256 bytes can fit in registers across architectures
    print('Block size =', adaptive_block_size)
    if len(dimensions) == 0:
        dimensions = [
            8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192
        ]
    if len(vectors) == 0:
        vectors = [
            64, 128, 512, 1024, 4096, 8192, 16384, 65536, 131072  # , 262144 #, 1048576
        ]
    if len(BLOCK_SIZES) == 0:
        BLOCK_SIZES = [adaptive_block_size]

    rng = np.random.default_rng()
    total_disk_needed = 0
    for D in dimensions:
        for V in vectors:
            # Generate Data
            if dtype is np.float32 or dtype is np.float64:
                data = rng.standard_normal(size=(V + 1, D), dtype=dtype)
            else:
                data = np.random.randint(0, 256, size=(V+1, D), dtype=dtype)

            # Write N-ary
            filename = f'{V}x{D}'
            filelocation = os.path.join(PURESCAN_DATA, filename)
            with open(filelocation, 'wb') as file:
                file.write(data.tobytes("C"))

            # Write PDX
            for BLOCK_SIZE in BLOCK_SIZES:
                print(BLOCK_SIZE, D, V, (D * V * type_size) / (1024 * 1024 * 1024))
                if V % BLOCK_SIZE != 0:
                    continue
                total_disk_needed += (D * V * type_size) / (1024 * 1024 * 1024)
                if BLOCK_SIZE != adaptive_block_size:
                    filename_pdx = f'{BLOCK_SIZE}x{V}x{D}-pdx-{dtype.__name__}'
                else:
                    filename_pdx = f'{V}x{D}-pdx-{dtype.__name__}'
                filelocation_pdx = os.path.join(PURESCAN_DATA, filename_pdx)
                print(filelocation_pdx)
                with open(filelocation_pdx, 'wb') as file:
                    file.write(data[0, :].tobytes("C"))  # Query vector
                    if V < BLOCK_SIZE:
                        file.write(data.tobytes("F"))
                    else:
                        pdx_chunks = V // BLOCK_SIZE
                        cur_offset = 1
                        for i in range(pdx_chunks):
                            file.write(data[cur_offset: cur_offset + BLOCK_SIZE, :].tobytes("F"))
                            cur_offset += BLOCK_SIZE
    print(total_disk_needed)


if __name__ == "__main__":
    generate_synthetic_data()
    # generate_synthetic_data(
    #     (16, 32, 128, 256, 512),
    #     (8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192,),
    #     (64, 128, 512, 1024, 4096, 8192, 16384, 65536, 131072,)
    # )
