# Transformer Drums

Playing Drums with Transformers - a series of experiments

## 1. Sequence prediction

- clone this repo w/ submodules: `git clone https://github.com/RVirmoors/transformer-drums --recurse-submodules`
- open the `transformer-drums` folder in your editor of choice.
- if running locally, follow the [startup instructions](startup.md). Else, you can load the .py scripts into Colab or something else.
- make sure the virtual environment is active.

### 1.1. Encoder -> Output

Basic experiment: can a Transformer "predict" a value if its Encoder sees the whole sequence?

- run `python sequence.py`

### 1.2. Simple time series

Standard usage model: encoder sees the history & whole time axis, decoder sees the current step & predicts the next.

- run `python time-series.py`