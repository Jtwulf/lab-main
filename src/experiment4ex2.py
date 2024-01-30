from external_libraries import *
from modules import *
import data_const as const

def calculate_bar_length(bpm):
    beats_per_bar = 4
    bars = 8
    beats_per_minute = bpm
    seconds_per_beat = 60.0 / beats_per_minute
    bar_length = seconds_per_beat * beats_per_bar * bars
    return bar_length

def get_bpm_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['bpm']

def count_drum_events_in_bars(drum_times, bar_length):
    bar_counts = defaultdict(int)
    for time in drum_times:
        bar_index = int(time // bar_length)
        bar_counts[bar_index] += 1
    return bar_counts

def plot_spaghetti(drum_counts_per_bar_all_songs, note_to_drum):
    for drum_note, songs_bar_counts in drum_counts_per_bar_all_songs.items():
        drum_name = note_to_drum.get(drum_note, drum_note)
        plt.figure(figsize=(10, 6))
        for song, bar_counts in songs_bar_counts.items():
            bars = sorted(bar_counts.keys())
            counts = [bar_counts[bar] for bar in bars]
            plt.plot(bars, counts, label=song, marker='o', linestyle='-', markersize=0.01, alpha=0.1, color='blue')
        plt.title(f"Drum Events per 8 Bars for {drum_name}")
        plt.xlabel("8 Bar Sections")
        plt.ylabel("Number of Events")
        plt.tight_layout()
        plt.show()

def process_file(json_path, midi_directory, drum_mapping, drum_counts_per_bar_all_songs):
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    midi_path = os.path.join(midi_directory, base_name + '.mid')
    if not os.path.exists(midi_path):
        return

    bpm = get_bpm_from_json(json_path)
    bar_length = calculate_bar_length(bpm)

    drum = Drum()
    drum_events = drum.get_drum_events(midi_path)

    '''
    pprint.pprint(base_name)
    print('Acoustic Bass Drum', len(drum_events[35]['times']))
    print('Acoustic Snare', len(drum_events[38]['times']))
    print('Closed Hi-Hat', len(drum_events[42]['times']))
    '''

    for drum_name in drum_mapping.values():
        if drum_name in drum_events:
            drum_times = drum_events[drum_name]['times']
            bar_counts = count_drum_events_in_bars(drum_times, bar_length)
            drum_counts_per_bar_all_songs[drum_name][base_name] = bar_counts

def main():
    json_directory = const.PROD_JSON_DIRECTORY_TEMPO
    midi_directory = const.PROD_MIDI_DIRECTORY
    drum_mapping = {'Kick': 35, 'Snare': 38, 'Hi-Hat': 42}
    note_to_drum = {note: drum for drum, note in drum_mapping.items()}

    drum_counts_per_bar_all_songs = defaultdict(lambda: defaultdict(dict))

    total_files = sum([len(files) for r, d, files in os.walk(json_directory) if any(file.endswith(".json") for file in files)])
    progress_bar = tqdm(total=total_files, desc="Overall Progress")

    for root, dirs, files in os.walk(json_directory):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                process_file(json_path, midi_directory, drum_mapping, drum_counts_per_bar_all_songs)
                progress_bar.update(1)

    progress_bar.close()
    plot_spaghetti(drum_counts_per_bar_all_songs, note_to_drum)

if __name__ == "__main__":
    main()
