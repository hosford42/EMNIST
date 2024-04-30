"""
# EMNIST

This package is a convenience wrapper around the EMNIST data set. Documentation on this data set, as well as manually
downloadable files, can be found [here](https://www.nist.gov/itl/iad/image-group/emnist-dataset).
"""

import gzip
import html
import logging
import os
import re
import zipfile

import numpy
import requests
import tqdm


LOGGER = logging.getLogger(__name__)


# These are ordered from most preferred to least preferred. The file is hosted on Google Drive to be polite to the
# authors and reduce impact to the original download server.
SOURCE_URLS = [
    'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip',
    'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip',
    'https://drive.google.com/uc?id=1R0blrtCsGEVLjVL3eijHMxrwahRUDK26',
]

CACHE_FILE_PATH = '~/.cache/emnist/emnist.zip'
PARTIAL_EXT = '_partial'
ZIP_PATH_TEMPLATE = 'gzip/emnist-{dataset}-{usage}-{matrix}-idx{dim}-ubyte.gz'
DATASET_ZIP_PATH_REGEX = re.compile(r'gzip/emnist-(.*)-test-images-idx3-ubyte\.gz')
GOOGLE_DRIVE_CONFIRMATION_LINK_REGEX = re.compile(rb'href="(/uc\?export=download.*?confirm=.*?)">Download anyway</a>')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 '
                  'Safari/537.36 Edg/115.0.1901.203'
}

IDX_DATA_TYPE_CODES = {
    0x08: numpy.ubyte,
    0x09: numpy.byte,
    0x0B: numpy.int16,
    0x0C: numpy.int32,
    0x0D: numpy.float32,
    0x0E: numpy.float64,
}


class DataFileValidationError(Exception):
    """Raised if the cached data file fails validation."""


def download_file(url, save_path, session=None):
    """Download a file from the requested URL to the indicated local save path. Download is done similarly to Chrome's,
    keeping the actual data in a separate file with 'partial' appended to the end of the name until the download
    completes, to ensure that an incomplete or interrupted download can always be detected."""
    if os.path.isfile(save_path):
        raise FileExistsError(save_path)
    LOGGER.info("Downloading %s to %s.", url, save_path)
    temp_path = save_path + PARTIAL_EXT
    try:
        with open(save_path, 'wb'), open(temp_path, 'wb') as temp_file:
            with (session or requests).get(url, stream=True, headers=HEADERS) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                chunk_size = 8192
                total_chunks = total_size // chunk_size + bool(total_size % chunk_size)
                with tqdm.tqdm(total=total_chunks, unit='B', unit_scale=True, unit_divisor=1024,
                               desc="Downloading %s" % os.path.basename(save_path)) as progress:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        temp_file.write(chunk)
                        progress.update(chunk_size)
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


def download_large_google_drive_file(url, save_path):
    """Google Drive has to make things complicated. There appears to be no way to circumvent their warning that the
    file is too large to be virus-scanned, hence this otherwise unnecessary complexity. A different choice of file
    hosting is advisable in the future."""
    session = requests.session()
    with session.get(url) as response:
        response.raise_for_status()
        content = response.content

    match = re.search(GOOGLE_DRIVE_CONFIRMATION_LINK_REGEX, content)
    if not match:
        raise RuntimeError("Google appears to have changed their large file download process unexpectedly. "
                           "Please download the file manually from %s and save it to ~/.cache/emnist/emnist.zip "
                           "as a manual work-around." % url)

    confirmed_link = url.split("/uc?")[0] + html.unescape(match.group(1).decode())
    download_file(confirmed_link, save_path, session)


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


def validate_cached_data(clear_on_failure: bool = True):
    """Validate the cached EMNIST data. If the downloaded data is corrupted, raise an exception. If the
    clear_on_failure flag is set, also clear the cached data on error, to enable a fresh download."""
    cache_path = get_cached_data_path()
    save_folder = os.path.dirname(cache_path)
    if not os.path.isdir(save_folder):
        LOGGER.info("Creating folder %s", save_folder)
        os.makedirs(save_folder)
    try:
        if not os.path.isfile(cache_path):
            raise DataFileValidationError("Cached file not found at %s." % cache_path)
        if not os.path.getsize(cache_path):
            raise DataFileValidationError("Cached file %s is zero bytes and cannot be used." % cache_path)
        if not list_datasets():
            raise DataFileValidationError("No data sets found in cached file %s." % cache_path)
    except DataFileValidationError:
        if clear_on_failure:
            clear_cached_data()
        raise
    except zipfile.BadZipfile as exc:
        if clear_on_failure:
            clear_cached_data()
        raise DataFileValidationError("Zip file %s is corrupted." % cache_path) from exc


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
    first_error = None
    for source_url in SOURCE_URLS:
        try:
            if 'drive.google.com' in source_url:
                download_large_google_drive_file(source_url, cache_path)
            else:
                download_file(source_url, cache_path)
            validate_cached_data()
            break
        except Exception as e:
            LOGGER.error("Error downloading file from %s:", source_url)
            if first_error is None:
                first_error = e
    else:
        assert first_error, "No source URLs listed in SOURCE_URLS!"
        raise first_error
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
    import sys
    logging.basicConfig(stream=sys.stdout)
    logging.getLogger().setLevel(0)
    inspect()
