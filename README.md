# SongScribe

Transcribe piano song into MIDI and use music21 to convert it to sheet music for musicians to learn unknown pieces. 

**Visit the [Interactive Model](https://matthewnevelos.github.io/SongScribe/)**

## How to Create Environment

Different environments will be made based on system requirements (CPU, GPU, GPU-ARM)

### Create Environment

```
conda create -n songscribe python=3.10
conda activate songscribe
```

### CPU only
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### GPU
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### GPU ARM64
```
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev
pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu122
```

### Other dependancies 
Once PyTorch is installed, the other dependencies can be installed

```
pip install nnAudio librosa music21 mido mir_eval soundfile matplotlib pretty_midi onnxruntime
pip install --upgrade onnx onnxscript
```

A sheet music renderer is needed for music21. \
[Online Renderer](https://www.soundslice.com/musicxml-viewer/) \
MuseScore Studio for Windows/Mac \
LilyPond for Linux.

### MuseScore installation
Install [MuseScore](https://musescore.org/en/download/musescore.msi)
In your environments terminal, type:
```
python
import music21
music21.configure.run()
```

Follow the instructions for configuring music21 with MuseScore