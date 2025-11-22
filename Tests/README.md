## Create a Virtual ENV & Install Packages

```bash
pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tqdm onnxruntime
```

## Run:

```bash
cd Tests/
python test_model.py -i input.wav -o output.wav

```
