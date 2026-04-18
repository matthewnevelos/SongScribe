import torch
import torch.nn as nn
from nnAudio.features.cqt import CQT1992v2

class EndToEndModel(nn.Module):
    """wrapper which combines CQT1992v2 and model into one to prepare for ONNX export"""
    def __init__(self, original_model):
        super().__init__()
        
        self.cqt = CQT1992v2(
            sr=22050,
            hop_length=128,
            fmin=27.5,
            n_bins=88,
            bins_per_octave=12,
            output_format='Magnitude', 
            trainable=False
        )
        
        self.crnn = original_model
        
    def forward(self, waveform):
        x = self.cqt(waveform) 
        
        max_val = x.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        x = x / max_val
        
        x = x.unsqueeze(1) 
        
        onset_logits, frame_logits = self.crnn(x)
        
        onset_probs = torch.sigmoid(onset_logits)
        frame_probs = torch.sigmoid(frame_logits)
        
        return onset_probs, frame_probs

def export_onnx(model, output_filename):
    """Export model as ONNX for use in interactive web app"""
    model.cpu()
    model.eval()
    
    e2e_model = EndToEndModel(model)
    e2e_model.eval()
    
    dummy_waveform = torch.randn(1, 22050 * 5)

    torch.onnx.export(
        e2e_model, 
        dummy_waveform, #type: ignore
        output_filename,
        export_params=True,
        opset_version=17, 
        do_constant_folding=True, # This is crucial to merge repeated weights
        input_names=['raw_audio'],
        output_names=['onset_probs', 'frame_probs'], # Two distinct outputs now
        dynamic_axes={
            'raw_audio': {1: 'num_samples'}, 
            'onset_probs': {2: 'time_steps'},      
            'frame_probs': {2: 'time_steps'}
        }
    )