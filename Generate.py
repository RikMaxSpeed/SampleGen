from MakeSTFTs import *
from MakeModels import load_saved_model
from SampleCategory import *
from Train import predict_stft
from Graph import *

import matplotlib.patches as patches

from IPython.display import HTML, display


def display_custom_link(file_path, display_text=None):

    if display_text is None:
        display_text = file_path

    link_str = f'<a href="{file_path}" target="_blank">{display_text}</a>'
    display(HTML(link_str))    




def numpify(tensor):
    return tensor.squeeze(0).detach().cpu().numpy()


def max_amp(x):
    if isinstance(x, torch.Tensor):
        return torch.max(torch.abs(x)).item()
    else:
        return np.max(np.abs(x))


# New: see whether we can interpolate interestingly between samples (ie: not just linear mixing)
class Sample_Generator():
    def __init__(self, model_name):
        super(Sample_Generator, self).__init__()
        self.load_data(model_name)
        
        
    def load_data(self, model_name):
        self.model_name = model_name
        self.model = load_saved_model(model_name)
        self.stfts, self.file_names = load_STFTs()
        self.categories = infer_sample_categories(self.file_names)

    def plot_amplitudes():
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
        plot_bar_charts([numpify(encode1), numpify(encode2)], [name1, name2], self.model_name + " encodings")
        
        plot_stft(name1, stft1, sample_rate, stft_hop)
        save_and_play_audio_from_stft(stft1, sample_rate, stft_hop, None, play_sound)

        for i in range(steps):
            t = i / (steps - 1)
            encode = linterp(t, encode1, encode2)
            save_file = f"{self.model_name} interpolate {100*(1-t):.1f}% x {name1} & {100*t:.1f}% x {name2}"
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
        plot_bar_charts([numpify(encode)], [name], self.model_name + " encoding")

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
            encodes.append(numpify(self.model.encode(input_stft)[0]))
        plot_bar_charts(encodes, names, f"{self.model_name}: {len(names)} {pattern} encodings")
    
    
    def generate_main_encodings(self, values, play_sound=True):
        # Determine the latent size:
        stft = self.stfts[0]
        input_stft = convert_stft_to_input(stft).unsqueeze(0).to(device)
        encode = self.model.encode(input_stft)
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
                
            
            
    def test_all(self):
        names=[]
        losses=[]
    
        noisy = False
        graphs = False
    
        for i in range(len(self.stfts)):
            stft = self.stfts[i]
            name = self.file_names[i][:-4]
            
            stft = adjust_stft_length(stft, sequence_length)
            
            if graphs:
                plot_stft(name, stft, sample_rate)
            
            if noisy:
                save_and_play_audio_from_stft(stft.cpu().numpy(), sample_rate, stft_hop, None, True)
            
            resynth, loss = predict_stft(self.model, stft)
            names.append(name)
            losses.append(loss)
            
            if graphs:
                plot_stft("Resynth " + name, resynth, sample_rate)
            
            save_and_play_audio_from_stft(resynth, sample_rate, stft_hop, "Results/" + name + " - resynth.wav", noisy)


        plot_multiple_histograms_vs_gaussian([losses], ["Resynthesis Loss"])


        indices = [i[0] for i in sorted(enumerate(losses), key=lambda x:x[1])]
        pad = max([len(x) for x in names])
        for i in indices:
            loss = losses[i]
            name = names[i]
            display_custom_link("Results/" + name + " - resynth.wav", "{}: loss={:.6f}".format(name, loss))
            
            
    def display_terms_in_file_names():
        all_words = [w for file_name in self.file_names for w in split_text_into_words(file_name) if not ignore_term(w)]
        print("Top words found in file names:")
        display_top_words(all_words, 0.005)


    def plot_categories(self, category_filter, colour_map = 'Set1'):
        # Build up a list of encoded STFTs
        encoded_dict = {}
        for category in category_filter:
            encoded_dict[category] = []
            
        encode_size = None
        for i in range(len(self.stfts)):
            if self.categories[i] in category_filter:
                input_stft = convert_stft_to_input(self.stfts[i]).unsqueeze(0).to(device)
                encode = numpify(self.model.encode(input_stft)[0])
                if encode_size is None:
                    encode_size = len(encode)
                    #print(f"Encode size={encode_size}")
                assert(len(encode) == encode_size)
                encoded_dict[self.categories[i]].append(encode)
        
        
        print("Encoded samples:")
        for category in category_filter:
            print(f"{len(encoded_dict[category]):>4} x {category}")

        cols = 3
        rows = encode_size // cols
        if cols * rows < encode_size:
            rows += 1

        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        cmap = plt.cm.get_cmap(colour_map)
        dot_size = 15
        for e in range(encode_size):
            plt.subplot(rows, cols, e + 1) # 1-based
            ne = (e + 1) % encode_size # wrap-around so we plot x6 vs x1
            plt.title(f"encode {e+1} & {ne+1}")

            for c in range(len(category_filter)):
                category = category_filter[c]
                encodes = encoded_dict[category]
                e1 = [x[e] for x in encodes]
                e2 = [x[ne] for x in encodes]
                colour = cmap(c / cmap.N)
                plt.scatter(e1, e2, label = category, s=dot_size, color=[colour], zorder=1, alpha = 0.5)
                ellipse = patches.Ellipse((np.mean(e1), np.mean(e2)), 2*np.std(e1), 2*np.std(e2), color=colour, alpha = 0.2, zorder=0)
                ax = plt.gca()
                ax.add_patch(ellipse)
                
                r = 3.0
                ax.set_xlim(-r, r)
                ax.set_ylim(-r, r)

            # Draw the X & Y axis explicitly
            plt.axhline(0, color='black', linewidth=1, zorder=2)
            plt.axvline(0, color='black', linewidth=1, zorder=2)

            # Move the axes to the centre
            plt.gca().spines['left'].set_position('zero')
            plt.gca().spines['bottom'].set_position('zero')

            # Hide the other lines:
            plt.gca().spines['right'].set_color('none')
            plt.gca().spines['top'].set_color('none')

            if e == 0:
                legend = plt.legend()
                for handle in legend.legendHandles:
                    handle.set_sizes([4 * dot_size])
        
        plt.show()
        

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


#model = "StepWiseMLP"
model = "MLPVAE_Incremental"
g = Sample_Generator(model)


def generate_morphs():
    for i in range(0, len(examples)-1, 2):
        g.interpolate_vae(examples[i], examples[i+1])
        #g.interpolate_no_ai(examples[i], examples[i+1]) # Linear interpolation over STFTs.


def generate_variations():
    for sample in examples[:5]:
        g.randomise_sample(sample)


def plot_encodings():
    for type in ["organ", "piano", "epiano", "string", "acoustic guitar", "marimba", "pad", "fm", "voice", "moog", ""]:
        g.plot_encodings(type, 1000)


def plot_categories(categories = None):
    if categories is None:
        g.plot_categories(["Vocal", "Synth", "Guitar"], "Set1")
        g.plot_categories(["Bass", "Plucked", "Bell"], "Set2")
        g.plot_categories(["Other", "Synth Makes", "Piano", "Bell"], "Dark2")
    else:
        g.plot_categories(categories)


def generate_main_encodings():
    g.generate_main_encodings([-2, -1, 0, +1, +2])

def test_all():
    g.test_all()


def demo_all():
    plot_encodings()
    plot_categories()
    generate_main_encodings()

