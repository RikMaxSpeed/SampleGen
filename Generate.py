from MakeSTFTs import *
from Train import *
from AudioUtils import *
from Graph import *


def numpify(tensor):
    return tensor.squeeze(0).detach().cpu().numpy()


def max_amp(x):
    if isinstance(x, torch.Tensor):
        return torch.max(torch.abs(x)).item()
    else:
        return np.max(np.abs(x))


# New: see whether we can interpolate interestingly between samples (ie: not just linear mixing)
class Sample_Generator():
    def __init__(self):
        super(Sample_Generator, self).__init__()
        self.load_data()
        
        
    def load_data(self):
        self.model = load_best_model()
        self.stfts, self.file_names = load_STFTs()
        
        amps =[]
        for stft in self.stfts:
            amps.append(max_amp(stft))
        plot_multiple_histograms_vs_gaussian([amps], ["STFT Max Amplitude"])
    
    
    def find_samples_matching(self, pattern, count = 1):
        pattern = pattern.lower()
        
        names = []
        stfts = []
        
        for stft, name in zip(self.stfts, self.file_names):
            if pattern in name.lower():
                name = name[:-4]
                #print(f"File {name} matches {pattern}.")
                stft = adjust_stft_length(stft, sequence_length)
                
                if count == 1:
                    return name, stft
                else:
                    names.append(name)
                    stfts.append(stft)
                    if len(names) >= count:
                        return names, stfts
        
        if len(names) > 0:
            return names, stfts
        
        raise Exception(f"No sample matches '{pattern}' :(")


    def encode_sample_matching(self, pattern):
        name, stft = self.find_samples_matching(pattern)
        input_stft = convert_stft_to_input(stft).unsqueeze(0).to(device)
        encode = self.model.encode(input_stft, False)
        print(f"Encoded {name} to {encode.shape}")
        return name, stft.numpy(), encode


    def decode_and_save(self, encode, save_to_file, play_sound):
        with torch.no_grad():
            decode = self.model.decode(encode)
            
        stft = convert_stft_to_output(decode)
    
        plot_stft(save_to_file, stft, sample_rate, stft_hop)
        save_and_play_audio_from_stft(stft, sample_rate, stft_hop, "Results/" + save_to_file + ".wav", play_sound)

    
    def interpolate_vae(self, pattern1, pattern2, play_sound=True, steps = 5):
        name1, stft1, encode1 = self.encode_sample_matching(pattern1)
        name2, stft2, encode2 = self.encode_sample_matching(pattern2)
        
        #plot_multiple_histograms_vs_gaussian([numpify(encode1), numpify(encode2)], [name1, name2])
        plot_bar_charts([numpify(encode1), numpify(encode2)], [name1, name2], "Encodings")
        
        plot_stft(name1, stft1, sample_rate, stft_hop)
        save_and_play_audio_from_stft(stft1, sample_rate, stft_hop, None, play_sound)

        for i in range(steps):
            t = i / (steps - 1)
            encode = linterp(t, encode1, encode2)
            save_file = f"interpolate {100*(1-t):.1f}% x {name1} & {100*t:.1f}% x {name2}"
            self.decode_and_save(encode, save_file, play_sound)

        plot_stft(name2, stft2, sample_rate, stft_hop)
        save_and_play_audio_from_stft(stft2, sample_rate, stft_hop, None, play_sound)


    def interpolate_no_ai(self, pattern1, pattern2, play_sound=True, steps = 5):
        name1, stft1 = self.find_samples_matching(pattern1)
        name2, stft2 = self.find_samples_matching(pattern2)
        
        stft1 = adjust_stft_length(stft1, sequence_length).numpy()
        stft2 = adjust_stft_length(stft2, sequence_length).numpy()
        
        amp1 = max_amp(stft1)
        amp2 = max_amp(stft2)
        amp = max(amp1, amp2)
        #print(f"amp1={amp1:.2f}, amp2={amp2:.2f} --> max={amp:.2f}")
        
        stft1 /= amp1
        stft2 /= amp2
        mult = stft1 * stft2
        diff = abs(stft1*stft1 - stft2*stft2)
        mid = mult + diff
                
        for i in range(steps):
            t = i / (steps - 1)
            
            if t <= 0.5:
                stft = linterp(2*t, stft1, mid)
            else:
                stft = linterp(2*t - 1, mid, stft2)
            
            stft *= amp / max_amp(stft)
            
            save_file = f"interpolate-no-AI {100*t:.1f}% - {name1} & {name2}"
            
            plot_stft(save_file, stft, sample_rate, stft_hop)
            save_and_play_audio_from_stft(stft, sample_rate, stft_hop, "Results/" + save_file + ".wav", play_sound)


    def randomise_sample(self, pattern, max_noise=1, play_sound=True, steps=5):
        name, stft, encode = self.encode_sample_matching(pattern)
        #plot_multiple_histograms_vs_gaussian([numpify(encode)], [name])
        plot_bar_charts([numpify(encode)], [name], "Encoding")

        for i in range(steps):
            amount = max_noise * i / (steps - 1)
            
            noise = (amount * (2 * torch.rand(encode.shape) - 1)).to(device)
            noisy_encode = encode * (1 + noise) # This can become extremely loud?
            save_file = f"noise={100*amount:.1f}% {name}"
            self.decode_and_save(noisy_encode, save_file, play_sound)


    def plot_encodings(self, pattern, count):
        names, stfts = self.find_samples_matching(pattern, count)
        encodes=[]
        for stft in stfts:
            input_stft = convert_stft_to_input(stft).unsqueeze(0).to(device)
            encodes.append(numpify(self.model.encode(input_stft, False)))
        plot_bar_charts(encodes, names, f"{len(names)} {pattern} encodings")
    
    
    def generate_main_encodings(self, values, play_sound=True):
        # Determine the latent size:
        stft = self.stfts[0]
        input_stft = convert_stft_to_input(stft).unsqueeze(0).to(device)
        encode = self.model.encode(input_stft, False)
        debug("encode", encode)
        latent_size = encode.shape[1]
        print(f"latent_size={latent_size}")
        
        # Decode each variable one by one
        for var in range(latent_size):
            encode = torch.zeros(encode.shape).to(device)
            for value in values:
                if value == 0 and var > 0: # we only need to generate 0,0,0,0... once
                    continue
                    
                encode[0, var] = value
                self.decode_and_save(encode, f"{encode[0]}", play_sound)
                
            
#EPiano Mrk II C3: loss=0.000102
#Kawai-K11-Dulcimer-C4: loss=0.000103
#E-Mu-Proteus-FX-Kalimba-C4: loss=0.000107
#Zither C3: loss=0.000107
#Analog 102 C4: loss=0.000110
#Electric Fat Fingers 1 C3: loss=0.000111
#Electric Fat Fingers 3 C4: loss=0.000113
#E-Piano C4: loss=0.000114
#80s Analog Solid Bass C3: loss=0.000114
#80s Analog Solid Bass C4: loss=0.000117
#Masterpiece Pluck C3: loss=0.000119
#Electric Fat Fingers 2 C4: loss=0.000120
#Electric Fat Fingers 4 C4: loss=0.000123
#Bottle Hit 17 C4: loss=0.000123
#Kalimba C4: loss=0.000125
#Electric Fat Fingers 2 C3: loss=0.000125
#Korg-M3R-Rock-Mutes-C3: loss=0.000125
#Piano St C4: loss=0.000125
#Electric Fat Fingers 4 C3: loss=0.000126
#Kithara_C3: loss=0.000126
#Guitar Attack C4: loss=0.000128
#Trad Harp C4: loss=0.000129
#Marimba C3: loss=0.000129
#Electric Fat Plect 3 C3: loss=0.000129
#Electric Fat Fingers 1 C4: loss=0.000130
#Piano Baby G C4: loss=0.000130
#Kawai-PHm-Contrabass-C3: loss=0.000133
#Mini m Dark Decay C4: loss=0.000133
#Electric Fat Plect 1 C4: loss=0.000133
#High Granular Harmonic C3: loss=0.000134

examples = [
    "EPiano Mrk II C3",
    "High Granular Harmonic C3",
    "Kawai-K11-Dulcimer-C4",
    "Electric Fat Plect 1 C4",
    "E-Mu-Proteus-FX-Kalimba-C4",
    "Mini m Dark Decay C4",
    "Zither C3",
    "Kawai-PHm-Contrabass-C3",
    "Analog 102 C4",
    "Piano Baby G C4",
]


def generate_morphs():
    g = Sample_Generator()

    for i in range(0, len(examples)-1, 2):
        g.interpolate_vae(examples[i], examples[i+1])
        #g.interpolate_no_ai(examples[i], examples[i+1]) # Linear interpolation over STFTs.


def generate_variations():
    g = Sample_Generator()

    for sample in examples[:5]:
        g.randomise_sample(sample)


def plot_encodings():
    g = Sample_Generator()
    for type in ["organ", "piano", "epiano", "string", "acoustic guitar", "marimba", "pad", "fm", "voice", "moog", ""]:
        g.plot_encodings(type, 1000)


def generate_main_encodings():
    g = Sample_Generator()
    g.generate_main_encodings([-2, -1, 0, +1, +2])


