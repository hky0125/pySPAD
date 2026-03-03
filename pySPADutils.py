import numpy as np
import tifffile as tiff
import os

class pySPADutils:
    '''
    Utility functions for SPAD data handling, including:
    - Reading/writing binary files
    - Writing BigTIFF files for large image stacks
    - Unpacking byte arrays from 1-bit packed SPAD data into (H, W, N) image stacks
    
    Usage:
    from pySPADutils import pySPADutils as util

    Last update: 
    2026-03-03 by Keyi Han
    '''
    
    '''
             ------------ I/O functions ------------
    ''' 
    @staticmethod
    def readBinBig(file_path):
        with open(file_path, "rb") as f:
            return f.read()
    

    @staticmethod
    def readBinFolder(folder_path):
        files = sorted(
            f for f in os.listdir(folder_path)
            if f.endswith(".bin")
        )
        file_paths = [os.path.join(folder_path, f) for f in files]

        # Compute total size
        total_size = sum(os.path.getsize(f) for f in file_paths)
        big_data = bytearray(total_size)


        offset = 0
        for path in file_paths:
            with open(path, "rb") as f:
                data = f.read()
                size = len(data)
                big_data[offset:offset+size] = data
                offset += size

        return big_data


    @staticmethod
    def writeBinBig(file_path, data):
        with open(file_path, "wb") as f:
            f.write(data)


    @staticmethod
    def writeTiffBig(file_path, img, compression_mode="zlib"):
        # img shape: (H, W, Z), values 0/1
        stack = np.transpose(img.astype(np.uint8), (2, 0, 1))  # (Z,H,W)
        tiff.imwrite(
            file_path,
            stack,
            photometric="minisblack",
            compression=compression_mode,
            metadata={"axes": "ZYX"},
        )


    '''
             ------------ Processing functions ------------
    ''' 
    @staticmethod
    def unpackBytearray(data,H=512, W=512, footer_bytes=4):

        BYTES_PER_FRAME = H * W // 8  # 32768

        data_mv = memoryview(data)
        if footer_bytes:
            data_mv = data_mv[:-footer_bytes]

        # zero-copy view onto the bytes
        u8 = np.frombuffer(data_mv, dtype=np.uint8)

        # sanity: number of full frames present
        # reshape to (frames, bytes_per_frame)
        n_frames = u8.size // BYTES_PER_FRAME
        u8 = u8[:n_frames * BYTES_PER_FRAME].reshape(n_frames, BYTES_PER_FRAME)


        # unpack all frames at once -> (frames, bytes_per_frame*8) = (frames, 262144)
        # reshape to (frames, H, W)
        bits = np.unpackbits(u8, axis=1).reshape(n_frames, H, W)

        # rot90(databit.reshape((512,512))) per frame in vectorized:
        bits = np.transpose(bits, (0, 2, 1))[:, ::-1, :]

        # put into img shaped (H, W, frames)
        img = np.transpose(bits, (1, 2, 0)).astype(np.uint8, copy=False)

        return img