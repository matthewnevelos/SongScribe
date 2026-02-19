# SongScribe

Transcribe song snippet into MIDI and use music21 to sheet music for musicians to learn unknown pieces. 

**phase 1:**\
Train model to convert piano solos to MIDI using MAESTRO dataset.

**phase 2:**\
Build classifier model on instrument recognition.\
Train multiple models on solos (drums, guitar, cello)

**Phase 3:**\
Test simultaneous instruments, potentially including unknown instruments. 


## Environment
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
pip install nnAudio librosa music21 mido mir_eval soundfile
```

A sheet music renderer is needed for music21. 
MuseScore Studio for Windows/Mac
LilyPond for Linux.