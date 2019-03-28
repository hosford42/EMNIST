"""
# EMNIST

This package is a convenience wrapper around the EMNIST data set. Documentation on this data set, as well as manually
downloadable files, can be found [here](https://www.nist.gov/itl/iad/image-group/emnist-dataset).
"""

# TODO: Automatically downloading emnist data from various sources, including the original location, one or more git
#       repositories, and/or free file hosting services.


import gzip
import logging
import os
import re
import zipfile

import numpy
import requests


LOGGER = logging.getLogger(__name__)


SOURCE_URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
CACHE_FILE_PATH = '~/.cache/emnist/emnist.zip'
PARTIAL_EXT = '_partial'
ZIP_PATH_TEMPLATE = 'gzip/emnist-{dataset}-{usage}-{matrix}-idx{dim}-ubyte.gz'
DATASET_ZIP_PATH_REGEX = re.compile(r'gzip/emnist-(.*)-test-images-idx3-ubyte\.gz')

IDX_DATA_TYPE_CODES = {
    0x08: numpy.ubyte,
    0x09: numpy.byte,
    0x0B: numpy.int16,
    0x0C: numpy.int32,
    0x0D: numpy.float32,
    0x0E: numpy.float64,
}


def download_file(url, save_path):
    """Download a file from the requested URL to the indicated local save path. Download is done similarly to Chrome's,
    keeping the actual data in a separate file with 'partial' appended to the end of the name until the download
    completes, to ensure that an incomplete or interrupted download can always be detected."""
    if os.path.isfile(save_path):
        raise FileExistsError(save_path)
    LOGGER.info("Downloading %s to %s.", url, save_path)
    temp_path = save_path + PARTIAL_EXT
    try:
        with open(save_path, 'wb'), open(temp_path, 'wb') as temp_file:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
    except Exception:
        try:
            if os.path.isfile(temp_path):
                LOGGER.info("Removing temp file at %s due to exception during download.", temp_path)
                os.remove(temp_path)
        finally:
            if os.path.isfile(save_path):
                LOGGER.info("Removing placeholder file at %s due to exception during download.", save_path)
                os.remove(save_path)
        raise
    os.remove(save_path)
    os.rename(temp_path, save_path)
    LOGGER.info("Successfully downloaded %s to %s.", url, save_path)


def get_cached_data_path():
    """Return the path where the EMNIST data is (or will be) cached."""
    return os.path.expanduser(CACHE_FILE_PATH)


def clear_cached_data():
    """Delete the cached EMNIST data, including the temporary file that can be created by an interrupted download."""
    cache_path = get_cached_data_path()
    temp_path = cache_path + PARTIAL_EXT
    for path in (cache_path, temp_path):
        if os.path.isfile(path):
            LOGGER.info("Removing cache file %s.", path)
            os.remove(path)
    LOGGER.info("Cache is clear.")


def ensure_cached_data():
    """Check that the EMNIST data is available in the local cache, and download it if not."""
    cache_path = get_cached_data_path()
    save_folder = os.path.dirname(cache_path)
    if not os.path.isdir(save_folder):
        LOGGER.info("Creating folder %s", save_folder)
        os.makedirs(save_folder)
    if os.path.isfile(cache_path):
        LOGGER.info("Cached file found at %s.", cache_path)
        if os.path.getsize(cache_path) > 0:
            return cache_path
        else:
            LOGGER.info("Cached file %s is zero bytes and cannot be used.", cache_path)
            os.remove(cache_path)
    download_file(SOURCE_URL, cache_path)
    return cache_path


def parse_idx(data):
    """Parse binary data in IDX format, returning it as a numpy array of the correct shape."""
    data = bytes(data)
    # See http://yann.lecun.com/exdb/mnist/ for an explanation of the IDX file format.
    if data[0] != 0 or data[1] != 0:
        raise ValueError("Data is not in IDX format.")
    data_type_code = data[2]
    data_type = IDX_DATA_TYPE_CODES.get(data_type_code)
    if data_type is None:
        raise ValueError("Unrecognized data type code %s. Is the data in IDX format?" % hex(data_type_code))
    dims = data[3]
    if not dims:
        raise ValueError("Header indicates zero-dimensional data. Is the data in IDX format?")
    shape = []
    for dim in range(dims):
        offset = 4 * (dim + 1)
        dim_size = int(numpy.frombuffer(data[offset:offset + 4], dtype='>u4'))
        shape.append(dim_size)
    shape = tuple(shape)
    offset = 4 * (dims + 1)
    data = numpy.frombuffer(data[offset:], dtype=numpy.dtype(data_type).newbyteorder('>'))
    return data.reshape(shape)


def extract_data(dataset, usage, component):
    """Extract an image or label array. The dataset must be one of those listed by list_datasets(), e.g. 'digits' or
    'mnist'. Usage should be either 'train' or 'test'. Component should be either 'images' or 'labels'."""
    if usage not in ('train', 'test'):
        raise ValueError("Unrecognized value %r for usage. Expected 'train' or 'test'." % usage)
    if component == 'images':
        dim = 3
    elif component == 'labels':
        dim = 1
    else:
        raise ValueError("Unrecognized value %r for component. Expected 'images' or 'labels'." % component)
    ensure_cached_data()
    cache_path = get_cached_data_path()
    zip_internal_path = ZIP_PATH_TEMPLATE.format(dataset=dataset, usage=usage, matrix=component, dim=dim)
    with zipfile.ZipFile(cache_path) as zf:
        compressed_data = zf.read(zip_internal_path)
    data = gzip.decompress(compressed_data)
    array = parse_idx(data)
    if dim == 3:
        # Why is this necessary? Was there a formatting error when the data was packaged and released by NIST?
        return array.swapaxes(1, 2)
    else:
        return array


def extract_samples(dataset, usage):
    """Extract the samples for a given dataset and usage as a pair of numpy arrays, (images, labels). The dataset must
    be one of those listed by list_datasets(), e.g. 'digits' or 'mnist'. Usage should be either 'train' or 'test'."""
    images = extract_data(dataset, usage, 'images')
    labels = extract_data(dataset, usage, 'labels')
    if len(images) != len(labels):
        raise RuntimeError("Extracted image and label arrays do not match in size. ")
    return images, labels


def extract_training_samples(dataset):
    """Extract the training samples for a given dataset as a pair of numpy arrays, (images, labels). The dataset must be
    one of those listed by list_datasets(), e.g. 'digits' or 'mnist'."""
    return extract_samples(dataset, 'train')


def extract_test_samples(dataset):
    """Extract the test samples for a given dataset as a pair of numpy arrays, (images, labels). The dataset must be one
    of those listed by list_datasets(), e.g. 'digits' or 'mnist'."""
    return extract_samples(dataset, 'test')


def list_datasets():
    """Return a list of the names of the available datasets."""
    ensure_cached_data()
    cache_path = get_cached_data_path()
    results = []
    with zipfile.ZipFile(cache_path) as zf:
        for path in zf.namelist():
            match = DATASET_ZIP_PATH_REGEX.fullmatch(path)
            if match:
                results.append(match.group(1))
    return results


def inspect(dataset='digits', usage='test'):
    """A convenience function for visually inspecting the labeled samples to ensure they are being extracted
    correctly."""
    # NOTE: This will hang if you run it from the PyCharm python console tab, whenever you have already imported
    #       matplotlib or if you have already called the function before. It's probably related to PyCharm's use of the
    #       debugger to extract variable values for display in the right-hand panel of the console. (For a brief
    #       explanation, see https://stackoverflow.com/a/24924921/4683578) As a simple work-around, start a fresh
    #       console tab each time and run it from there.
    import matplotlib.pyplot as plt
    backend = plt.get_backend()
    interactive = plt.isinteractive()
    try:
        plt.switch_backend('TkAgg')
        plt.ioff()
        images = extract_data(dataset, usage, 'images')
        labels = extract_data(dataset, usage, 'labels')
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            print("LABEL:", label)
            plt.imshow(image)
            plt.show(block=True)
    finally:
        plt.switch_backend(backend)
        if interactive:
            plt.ion()
        else:
            plt.ioff()


if __name__ == '__main__':
    inspect()
