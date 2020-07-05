# Dog vs Cat Classifier
> Classify between cats and dogs using a simple convolution neural network

The code runs a convolution neural network which discriminates between cats and dogs with high accuracy.

## Setup

Clone this repository on your local machine or on colab;

```shell
git clone https://github.com/ogunlao/dog-cat-classifier.git
```

Download dataset from this [google drive](https://drive.google.com/file/d/1Cn0B9Zr2irUnZcHqODT9IilGHf9fZ61R/view) into your local directory or google colab.

Extract the data folder containing train and test into your `dog-cat-classifier` directory.

For a quick classification, run the `main.py` file

```shell
python main.py
```

## Usage

Two simple CNN classifiers are proposed for the problem: `model2` is similar to `model1` but with a much more deeper architecture.

Run the first model using;

```shell
python main.py model1
```

Run the second model using;

```shell
python main.py model2
```

Also a `dog_cat.ipynb` notebook is attached to examine the setup and play with the layer structure.

## License

[MIT](https://choosealicense.com/licenses/mit/)