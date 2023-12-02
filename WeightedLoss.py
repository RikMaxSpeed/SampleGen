# Currently dead code.
# The idea was to focus the loss function on key time & frequency regions.


# Custom loss function to emphasise frequencies and the start time
class Time_Frequency_Loss(nn.Module):
    def __init__(self, time_steps, frequency_buckets):
        super(Time_Frequency_Loss, self).__init__()
        
        lowF = dB_to_amplitude(-40) # Looks like -60 is too aggressive.
        hiF  = dB_to_amplitude(-10)
        
        midF = int( (c4hz * 2 * stft_buckets) / sample_rate ) # bucket for C4
        print(f"{c4hz:.2f}Hz = bucket# {midF}")
        
        hiT  = dB_to_amplitude(10)
        lowT = dB_to_amplitude(-40)

        # Frequency weights: centred on middle-C
        freq_weights = torch.cat([log_interp(lowF, 1.0, midF), log_interp(1.0, hiF, frequency_buckets - midF)]).view(1, frequency_buckets)
        assert(freq_weights.shape[1] == frequency_buckets)
                
        # Time weights: higher for start and end times
        time_weights = log_interp(hiT, lowT, time_steps).view(time_steps, 1)
        assert(time_weights.shape[0] == time_steps)
        
        self.plot_freq_time_weights(freq_weights, time_weights)
        
        # Combined weights
        weights = time_weights * freq_weights
        
        assert(weights.shape[0] == time_steps)
        assert(weights.shape[1] == frequency_buckets)
        
#        minA = min([lowF, hiF, lowT, hiT])
#        weights[weights<minA] = minA
        
#        weights /= weights.sum() # Normalise so the total is 1.

        self.weights = weights.to(device)


    def plot_weights(self):
        plot_stft("Loss Function Weights", transpose(self.weights.cpu()), sample_rate, stft_hop)


    def plot_freq_time_weights(self, freq_weights, time_weights):
        frequencies = [i * sample_rate/(2 * stft_buckets) for i in range(1, stft_buckets + 1)]
        seconds = [i * stft_hop / sample_rate for i in range(len(time_weights))]

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Frequency Weights")
        plt.gca().set_xscale('log')
        plt.plot(frequencies, freq_weights.squeeze(0).numpy())
        plt.subplot(1, 2, 2)
        plt.title("Time Weights")
        plt.plot(seconds, time_weights.squeeze(0).numpy())
        plt.tight_layout()
        plt.show()
    

    def forward(self, y_pred, y_true):
        squared_diffs = (y_pred - y_true) ** 2
        weighted_diffs = squared_diffs * self.weights
        average = weighted_diffs.mean()
        return average


def get_criterion():

    if True:
        return nn.MSELoss() # Naive
        
    # Weighted:
    criterion = Time_Frequency_Loss(sequence_length, stft_buckets)
    criterion.plot_weights()
    return criterion
