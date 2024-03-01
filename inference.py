import numpy as np
import torch
import socket               # Import socket module
import wave
from model import SpeechClassifierModel
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow
import torchaudio
import sys
sys.path.append('/homes/2/karrolla.1/')
from VAD.whisperX import whisperx


compute_type = "int8"
model1 = whisperx.load_model("large-v2", "cuda", compute_type=compute_type)

sample_rate = 16000
# Define a callback function to process each frame
def save_file(data):
    output_file = "my_voice.wav"
    sample_width = 2  # 2 bytes for 16-bit audio, 1 byte for 8-bit audio
    CHANNELS = 1
    SAMPLE_RATE = 16000
    #CHUNK = int(SAMPLE_RATE / 10)
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(SAMPLE_RATE)

        for chunk in data:
            wf.writeframes(chunk)
        wf.close()
    return output_file


def process_frame(indata, frames,data, time, status):
    indata = np.frombuffer(indata, np.int8)
    a = int(indata.size)
    if a < 1024: 
        indata = np.pad(indata, (0, 1024-a), mode='constant', constant_values=0)
        '''m = nn.ConstantPad1d((0,512-a), 0.0000000)
        indata = m(indata)'''
    frames.append(indata)  # Apply windowing function to the input frame
    # Check if enough frames are collected
    num_frames = 16
    if len(frames) >= num_frames:
        #import ipdb;ipdb.set_trace()
        wav_file = save_file(data)
        data.pop(0)
        #frames_array = np.array(frames[:num_frames])  # Convert frames list to numpy array
        frames.pop(0)  # Clear the frames list
        filterbank = Filterbank(n_mels=40)
        stft = STFT(sample_rate=sample_rate, win_length=25, hop_length=10, n_fft=400)
        deltas = Deltas(input_size=40)
        #self.context_window = ContextWindow(window_size=151, left_frames=75, right_frames=75)
        input_norm = InputNormalization()
        wavform, sr = torchaudio.load(wav_file)

        #wavform = self.input_norm(wavform)
        wavform = wavform.type('torch.FloatTensor')
        if sr > 16000:
            wavform = torchaudio.transforms.Resample(sr, 16000)(wavform)
        features = stft(wavform)
        features = spectral_magnitude(features)
        features = filterbank(features)
        #return features, curr['Class'], curr['AudioPath']

        output = model(features)  # Make predictions using the pretrained model
        best = np.where(output < 0.999, 0, 1)
        res = best.item()
        if res !=1:
            print("Wake word not detected")
        if res == 1:
            print("Wake word detected")
            import ipdb;ipdb.set_trace()
            audio = whisperx.load_audio(wav_file)
            result = model1.transcribe(audio, 2)
            print(result["segments"])
            #import ipdb;ipdb.set_trace()
            # return
        # Process the output as required
        # ...

# Initialize the frames list
frames = []
device = torch.device('cpu')
# Load the pretrained model checkpoint
checkpoint_path = "/homes/2/karrolla.1/KWD/saved/debug_mode/upsample_minority_debug_1_3_lstm_80.pt"
pretrained_model = torch.load(checkpoint_path, map_location=device)
model_state_dict = pretrained_model["model_state_dict"]
model_params = {'num_classes': pretrained_model['model_params']['num_classes'],
                    'feature_size': pretrained_model['model_params']['feature_size'], 
                    'hidden_size': pretrained_model['model_params']['hidden_size'], 
                    'num_layers': pretrained_model['model_params']['num_layers'], 
                    'dropout': pretrained_model['model_params']['dropout'], 
                    'bidirectional': pretrained_model['model_params']['bidirectional'], 
                    'device': device}  
model = SpeechClassifierModel(**model_params)
model.load_state_dict(model_state_dict)
model.eval()


s = socket.socket()
host = socket.gethostname()
port = 7012      # specially allocated port
print("The server is waiting on the following host and port address")
print(host, port)
s.bind((host, port))
data =[]
voiced_confidences = []

s.listen(5)    # now waiting for the client connection (This is where the pause between wake word and question/information sentence from the user input comes **I GUESS)
#while True:  # dont't know why an infinite loop is required here?
c, addr = s.accept()     # Establishes the connection with client
print("Connection Established") 
print("Receiving the audio data from client side")
#l = c.recv(1600)     # l should match the chunk size being sent on the client side
count  = 0
a = True
data = []
while a:
    l = c.recv(1024)
    data.append(l)
    if len(l) == 0:   # User tries to abruptly close the connection (user exception)
        break
    else:      #elif data(l) >= 16000:
        found = process_frame(l, frames,data, time=0, status=1)
        
    

