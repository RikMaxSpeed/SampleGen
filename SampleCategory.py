from collections import Counter
import re

# Display the most common words found in the sample file names:
def split_text_into_words(text):
    return [w for w in re.split(r'[^a-zA-Z]+', text.lower()) if w]


def display_top_words(words, min_percent = 0.05):
    total = len(words)
    word_counts = Counter(words)
    sorted_words = sorted(word_counts.keys(), key=lambda x: (-word_counts[x], x))

    for word in sorted_words:
        count = word_counts[word]
        pct = count / total
        if pct >= min_percent:
            print("{:>4} = {:>5.1f}% : {}".format(count, 100*pct, word))


def ignore_term(word):
    return word == "c" or word == "wav"



category_map = [ # Important: these will be evaluated in order
    ("Bell", ["bell", "musicbox", "glock", "celesta", "vibraphone", "chime", "tubular", "glockenspiel"]),
    ("Wind", ["flute", "clarinet", "bassoon", "blown", "bottle", "breath", "calliope"]),
    ("Bass", ["bass", "pickbass", "ebass"]),
    ("E-Piano", ["epiano", "e-piano", "rhod", "road", "keys"]),
    ("Guitar", ["guitar", "eguitar", "banjo", "steel", "bandura", "hawaii", "electric", "fretless", "classicguitar"]),
    ("Plucked", ["pluck", "harpsichord", "harp", "bandura", "sitar", "dulcimer", "dulcimar", "charang", "harpsicord", "zither"]),
    ("Percussion", ["marimba", "xylo", "kalimba", "drum"]),
    ("Strings", ["string", "strng", "strings", "violin", "viola"]),
    ("Brass", ["brass", "trumpet", "sax", "flugel", "horn", "trombone"]),
    ("Organ", ["organ", "tonewheel", "hammond", "wurlitzer", "farfisa"]),
    ("Vocal", ["oh", "ah", "eh", "oo", "aa", "choir", "human", "vocoder", "vox", "vocal", "vowel", "voice", "cherry", "anastacia", "amanda", "vocalize", "formant"]),
    ("Pad", ["pad", "ambient", "dream", "aqua", "arctic", "kingdom", "iceland"]),
    ("Piano", ["piano"]),
    ("Pulsing", ["bpm", "sequence", "particle"]),
    ("Synth", ["saw", "sawtooth", "wavetable", "sine", "sines", "square", "triangle", "triangles", "pulse", "sync", "acid", "moog", "analog", "synth", "filter", "vintage", "additive", "harmonic", "resonance", "mini", "chiptone", "sonar", "trance", "lead", "polyphonic"]),
    ("Synth Makes", ["casio", "korg", "yamaha", "roland", "ensoniq", "mu", "alesis", "kawai"]) # daft 'catch-all' category!
]
    
    
# Try to automatically infer the sample category from the file name:
def infer_sample_category(name):
    words = split_text_into_words(name)
    
    for category, terms in category_map:
        if category != "":
            for term in terms:
                if term in words:
                    return category
                    
    return "Other"
    

all_categories = [x[0] for x in category_map] + ["Other"]


def infer_sample_categories(names):
    categories = [infer_sample_category(name) for name in names]
    
    print("Samples by Category:")
    display_top_words(categories, 0)
        
    return categories

