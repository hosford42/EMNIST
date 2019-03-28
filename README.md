# EMNIST
Extended MNIST - Python Package

## The EMNIST Dataset

The EMNIST Dataset is an extension to the original MNIST dataset to also include letters. For more details, see
the [EMNIST web page](https://www.nist.gov/itl/iad/image-group/emnist-dataset) and the 
[paper](http://arxiv.org/abs/1702.05373) associated with its release:

  Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).
  EMNIST: an extension of MNIST to handwritten letters.
  Retrieved from http://arxiv.org/abs/1702.05373

## The EMNIST Python Package

This package is a convenience wrapper around the EMNIST Dataset. The package provides functionality to 
automatically download and cache the dataset, and to load it as numpy arrays, minimizing the boilerplate 
necessary to make use of the dataset.

# Installation

To install the EMNIST Python package along with its dependencies, run the following command:

  pip install emnist

The dataset itself is automatically downloaded and cached when needed. To preemptively download the data
and avoid a delay later during the execution of your program, execute the following command after
installation:

  python -c "import emnist; emnist.ensure_cached_data()"

Alternately, if you have already downloaded the original IDX-formatted dataset from the EMNIST web page,
copy or move it to `~/.cache/emnist/`, where `~` is your home folder, and rename it to `emnist.zip`. The
package will use the existing file rather than downloading it again.

## Usage

Usage of the EMNIST Python package is designed to be very simple. 

To get a listing of the available subsets:

  >>> from emnist import list_datasets
  >>> list_datasets()
  \['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist'\]
  
(See the [EMNIST web page](https://www.nist.gov/itl/iad/image-group/emnist-dataset) for details on each of 
these subsets.)

To load the training samples for the 'digits' subset:

  >>> from emnist import extract_training_samples
  >>> images, labels = extract_training_samples('digits')
  >>> images.shape
  (240000, 28, 28)
  >>> labels.shape
  (240000,)
  >>>

To load the test samples for the 'digits' subset:

  >>> from emnist import extract_test_samples
  >>> images, labels = extract_test_samples('digits')
  >>> images.shape
  (40000, 28, 28)
  >>> labels.shape
  (40000,)
  >>>

Data is extracted directly from the downloaded compressed file to minimize disk usage, and is returned 
as standard numpy arrays.
