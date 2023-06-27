Instructions apply to Windows but should be similar for Linux.

1. Download Python and (if using an Nvidia GPU) CUDA

https://www.python.org/downloads/release/python-3913/

2. Create new folder, open in VS Code

3. Run in terminal:

```
python -m venv venv
```

4. Create a 'libs' folder in `venv/Scripts`

5. Copy `python39.lib` from your `Python39/libs` folder to your `venv/Scripts/libs`. Then keep executing in terminal:

```
venv/Scripts/activate.bat

python -m pip install light-the-torch
ltt install torch
```