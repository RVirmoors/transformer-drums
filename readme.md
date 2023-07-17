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

## 2. Drum sequences

### 2.1 Drums prediction

### 2.2 Drums generation

## 3. C++ implementation with LibTorch

Download the Release version of LibTorch from the [PyTorch website](https://pytorch.org/get-started/locally/) and extract it somewhere, making note of the absolute path. See also the [official](https://pytorch.org/tutorials/advanced/cpp_frontend.html) [guides](https://pytorch.org/cppdocs/installing.html).

If you want to use CUDA, make sure you get the right version for what you have installed (run `nvcc --version` to check).

Edit `CMakeLists.txt` and replace your absolute path to your libtorch download next to `CMAKE_PREFIX_PATH` (instead of my `"E:/GitHub/libtorch"`)

Then run:

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Then you can run `drums.exe` from the `build/Release` folder, BUT ONLY after you copy (on Windows) all the .dll files from `libtorch/lib` next to `drums.exe`.