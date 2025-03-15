import ctypes

import torch
import numpy as np

from model import NUM_FEATURES

# External C Library
libdataset = ctypes.CDLL("./build/libdataset.so")

CreateBatchStream = libdataset.CreateBatchStream
CreateBatchStream.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint]
CreateBatchStream.restype = ctypes.c_void_p

DestroyBatchStream = libdataset.DestroyBatchStream
DestroyBatchStream.argtypes = [ctypes.c_void_p]
DestroyBatchStream.restype = None


class SparseBatch(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("num_active_features", ctypes.c_int),
        ("stm", ctypes.POINTER(ctypes.c_int)),
        ("score", ctypes.POINTER(ctypes.c_int)),
        ("result", ctypes.POINTER(ctypes.c_float)),
        ("white_features_indices", ctypes.POINTER(ctypes.c_int)),
        ("black_features_indices", ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device):
        stm = torch.from_numpy(
            np.ctypeslib.as_array(self.score, shape=(self.size, 1))
        ).to(device)
        score = torch.from_numpy(
            np.ctypeslib.as_array(self.score, shape=(self.size, 1))
        ).to(device)
        result = torch.from_numpy(
            np.ctypeslib.as_array(self.result, shape=(self.size, 1))
        ).to(device)

        # As we said, the index tensor needs to be transposed (not the whole sparse tensor!).
        # This is just how pytorch stores indices in sparse tensors.
        # It also requires the indices to be 64-bit ints.
        white_features_indices = (
            torch.transpose(
                torch.from_numpy(
                    np.ctypeslib.as_array(
                        self.white_features_indices,
                        shape=(self.num_active_features, 2),
                    )
                ),
                0,
                1,
            )
            .long()
            .to(device)
        )
        black_features_indices = (
            torch.transpose(
                torch.from_numpy(
                    np.ctypeslib.as_array(
                        self.black_features_indices,
                        shape=(self.num_active_features, 2),
                    )
                ),
                0,
                1,
            )
            .long()
            .to(device)
        )

        # The values are all ones, so we can create these tensors in place easily.
        # No need to go through a copy.
        white_features_values = torch.ones(self.num_active_features).to(device)
        black_features_values = torch.ones(self.num_active_features).to(device)

        # Now the magic. We construct a sparse tensor by giving the indices of
        # non-zero values (active feature indices) and the values themselves (all ones!).
        # The size of the tensor is batch_size*NUM_FEATURES, which would
        # normally be insanely large, but since the density is ~0.1% it takes
        # very little space and allows for faster forward pass.
        # For maximum performance we do cheat somewhat though. Normally pytorch
        # checks the correctness, which is an expensive O(n) operation.
        # By using _sparse_coo_tensor_unsafe we avoid that.
        white_features = torch.sparse_coo_tensor(
            white_features_indices,
            white_features_values,
            (self.size, NUM_FEATURES),
            is_coalesced=True,
        ).to(device)
        black_features = torch.sparse_coo_tensor(
            black_features_indices,
            black_features_values,
            (self.size, NUM_FEATURES),
            is_coalesced=True,
        ).to(device)

        # What is coalescing?! It makes sure the indices are unique and ordered.
        # Now you probably see why we said the inputs must be ordered from the start.
        # This is normally a O(n log n) operation and takes a significant amount of
        # time. But here we **know** that the tensor is already in a coalesced form,
        # therefore we can just tell pytorch that it can use that assumption.
        white_features._coalesced_(True)
        black_features._coalesced_(True)

        return (white_features, black_features, stm, score, result)


SparseBatchPtr = ctypes.POINTER(SparseBatch)

GetNextBatch = libdataset.GetNextBatch
GetNextBatch.argtypes = [ctypes.c_void_p]
GetNextBatch.restype = SparseBatchPtr

DestroyBatch = libdataset.DestroyBatch
DestroyBatch.argtypes = [SparseBatchPtr]
DestroyBatch.restype = None
