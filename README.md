A Prediction‐Traversal Approach for Compressing Scientific Data on Unstructured Meshes with Bounded Error
=====
(C) 2024 by Congrong Ren. See LICENSE in top-level directory.

## Dependencies

To run this project, you need to have the following Python packages installed:

- `numpy`: For numerical operations
- `networkx`: For creating and manipulating complex networks
- `trimesh`: For handling triangular meshes
- `dahuffman`: For Huffman coding (data compression)
- `zstd`: For Zstandard compression
- `matplotlib`: For plotting and visualization

## Testing Examples

To run this project, execute the following command in your terminal:
```
$ python compress_decompress.py -dataset <dataset_name> -attribute <attribute_name>
```
For example:
```
$ python compress_decompress.py -dataset syn -attribute data
```
See the results in `results/<dataset_name>/<attribute_name>`.

## Citation

Please including the following citation if you use the code:

* Ren, C., Liang, X., and Guo, H., "[A Prediction‐Traversal Approach for Compressing Scientific Data on Unstructured Meshes with Bounded Error](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.15097)," *Computer Graphics Forum*, vol. 43, no. 3, 2024, p. e15097.
