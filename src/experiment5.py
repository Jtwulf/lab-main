from external_libraries import *
from modules import *
import data_const as const

def calculate_section_based_matching_rate(pattern_changes, section_changes, song_duration):
    matched_count = 0
    matched_times_percent = []

    for section_change in section_changes:
        if any(section_change - 1 <= pattern_change <= section_change + 1 for pattern_change in pattern_changes):
            matched_count += 1
            matched_percent = (section_change / song_duration) * 100
            matched_times_percent.append(matched_percent)

    matching_rate = (matched_count / len(section_changes)) * 100 if section_changes else 0
    return matching_rate, matched_times_percent


def calculate_drum_based_matching_rate(pattern_changes, section_changes, song_duration):
    matched_count = 0
    matched_times_percent = []

    for pattern_change in pattern_changes:
        if any(pattern_change - 1 <= section_change <= pattern_change + 1 for section_change in section_changes):
            matched_count += 1
            matched_percent = (pattern_change / song_duration) * 100
            matched_times_percent.append(matched_percent)

    matching_rate = (matched_count / len(pattern_changes)) * 100
    return matching_rate, matched_times_percent

def plot_matching_rates(all_matching_rates):
    plt.hist(all_matching_rates, bins=range(0, 101, 10), histtype="bar", edgecolor="black")
    plt.xlabel('Matching Rate (%)')
    plt.ylabel('Number of Songs')
    plt.title('Distribution of Matching Rates')
    plt.tight_layout()
    plt.show()

def plot_matched_times_percent(matched_times_percent):
    plt.hist(matched_times_percent, bins=range(0, 101, 10), histtype="bar", edgecolor="black")
    plt.xlabel('Time (%)')
    plt.ylabel('Number of Matches')
    plt.title('Distribution of Matches Over Time (%)')
    plt.tight_layout()
    plt.show()

def detect_section_changes(section_data):
    previous_label = None
    section_changes = []
    for segment in section_data['segments']:
        if segment['label'] != previous_label:
            section_changes.append(int(round(segment['start'])))
            previous_label = segment['label']
    return sorted(set(section_changes))

def process_midi_file(midi_path, json_directory, allin1, all_matching_rates, all_matched_times_percent):
    song_name = os.path.splitext(os.path.basename(midi_path))[0]
    json_path = os.path.join(json_directory, f"{song_name}.json")
    section_data = allin1.load_section_data(json_path)

    drum = Drum()
    events = drum.get_drum_events(midi_path)

    events_with_times = {note: event for note, event in events.items() if event['times']}
    if not events_with_times:
        print(f"No drum events found in {song_name}. Skipping.")
        return

    pattern_changes = drum.detect_pattern_changes(events)
    section_changes = detect_section_changes(section_data)

    song_duration = max(max(event['times']) for event in events.values())
    matching_rate, matched_times_percent = calculate_drum_based_matching_rate(pattern_changes, section_changes, song_duration)
    # matching_rate, matched_times_percent = calculate_section_based_matching_rate(pattern_changes, section_changes, song_duration)

    if all_matching_rates is not None:
        all_matching_rates.append(matching_rate)
    if all_matched_times_percent is not None:
        all_matched_times_percent.extend(matched_times_percent)

    # drum.plot_drum_with_pattern_and_sections(song_name, events, pattern_changes, section_changes)

def main(process_mode):
    midi_directory = const.PROD_MIDI_DIRECTORY
    json_directory = const.PROD_JSON_DIRECTORY
    allin1 = Allin1()

    midi_files = [os.path.join(root, file) for root, dirs, files in os.walk(midi_directory) for file in files if file.endswith(".mid")]
    progress_bar = tqdm(total=len(midi_files), desc="Overall Progress")

    all_matching_rates = []
    for midi_path in midi_files:
        process_midi_file(midi_path, json_directory, allin1, all_matching_rates, None)
        progress_bar.update(1)

    average_matching_rate = sum(all_matching_rates) / len(all_matching_rates) if all_matching_rates else 0
    print(f"Average Matching Rate: {average_matching_rate:.2f}%")

    if process_mode == 'timeseries':
        all_matched_times_percent = []
        for midi_path in midi_files:
            process_midi_file(midi_path, json_directory, allin1, None, all_matched_times_percent)
        plot_matched_times_percent(all_matched_times_percent)
    elif process_mode == 'distribution':
        plot_matching_rates(all_matching_rates)

    progress_bar.close()

if __name__ == "__main__":
    process_mode = 'distribution'  # 'timeseries' | 'distribution'
    main(process_mode)
