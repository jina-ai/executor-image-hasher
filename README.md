# ImageHasher

An executor to encode the images into embeddings using different comparable hashing technique, such as `perceptual`,
`average`, `wavelet`, and `difference` techniques. Features in the image are used to generate a distinct
(but not unique) fingerprint, and these fingerprints are comparable.

Comparable hashes are a different concept compared to cryptographic hash functions like `MD5` and `SHA1`. With
cryptographic hashes, the hash values are random. The data used to generate the hash acts like a random seed, so the
same data will generate the same result, but different data will create different results. Comparing two SHA1 hash
values really only tells you two things. If the hashes are different, then the data is different. And if the hashes are
the same, then the data is likely the same. In contrast, comparable hashes such as `perceptual` can be compared --
giving you a sense of similarity between the two data sets. All comparable hashes have the same basic properties:
images can be scaled larger or smaller, have different aspect ratios, and even minor coloring differences (contrast,
brightness, etc.) and they will still match similar images. A good usage of this is the detection of duplicate images.

ImageHasher receives the `Documents` with `blob` attributes. The executor will encode the images into a vector of either
`dtype=np.uint8` or `dtype=np.bool` and store them in the `embedding` attribute of the `Document`.