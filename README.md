A Prediction‐Traversal Approach for Compressing Scientific Data on Unstructured Meshes with Bounded Error
=====
(C) 2024 by Congrong Ren. See LICENSE in top-level directory.

## Dependencies

To run this project, you need to have the following Python packages installed:

- `h5py`: For working with HDF5 files
- `numpy`: For numerical operations
- `networkx`: For creating and manipulating complex networks
- `trimesh`: For handling triangular meshes
- `dahuffman`: For Huffman coding (data compression)
- `zstd`: For Zstandard compression
- `matplotlib`: For plotting and visualization

## Testing Examples

```
python compress_decompress.py -dataset <dataset_name> -attribute <attribute_name>
```

## Citation

Please including the following citation if you use the code:

* Ren, C., Liang, X., and Guo, H., "[A Prediction‐Traversal Approach for Compressing Scientific Data on Unstructured Meshes with Bounded Error](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.15097)," *Computer Graphics Forum*, vol. 43, no. 3, 2024, p. e15097.
