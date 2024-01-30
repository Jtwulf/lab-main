from external_libraries import *

class Visualizer(ABC):
    def plot(self):
        pass


class AudioSeparator:
    def __init__(self, in_path, out_path, model="mdx_q", extensions=["mp3", "wav", "ogg", "flac"], two_stems=None, mp3=True, mp3_rate=320, float32=False, int24=False):
        self.in_path = in_path
        self.out_path = out_path
        self.model = model
        self.extensions = extensions
        self.two_stems = two_stems
        self.mp3 = mp3
        self.mp3_rate = mp3_rate
        self.float32 = float32
        self.int24 = int24
        self.stems = ['bass.mp3', 'drums.mp3', 'vocals.mp3', 'other.mp3']

    def separate(self, inp=None, outp=None):
        inp = inp or self.in_path
        outp = outp or self.out_path

        cmd = ["python3", "-m", "demucs.separate", "-o", str(outp), "-n", self.model]
        if self.mp3:
            cmd += ["--mp3", f"--mp3-bitrate={self.mp3_rate}"]
        if self.float32:
            cmd += ["--float32"]
        if self.int24:
            cmd += ["--int24"]
        if self.two_stems is not None:
            cmd += [f"--two-stems={self.two_stems}"]

        files = [str(f) for f in self._find_files(inp)]
        if not files:
            print(f"No valid audio files in {self.in_path}")
            return

        print("Going to separate the files:")
        print('\n'.join(files))
        print("With command: ", " ".join(cmd))
        p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
        self._copy_process_streams(p)
        p.wait()
        if p.returncode != 0:
            print("Command failed, something went wrong.")

    def _is_separated(self, outp):
        return all(os.path.exists(Path(outp, stem)) for stem in self.stems)

    def _find_files(self, in_path):
        out = []
        in_path = Path(in_path)
        if in_path.is_dir():
            for file in in_path.iterdir():
                if file.suffix.lower().lstrip(".") in self.extensions:
                    out.append(file)
        else:
            if in_path.suffix.lower().lstrip(".") in self.extensions:
                out.append(in_path)
        return out


    def _copy_process_streams(self, process: sp.Popen):
        def _raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
            assert stream is not None
            if isinstance(stream, io.BufferedIOBase):
                stream = stream.raw
            return stream

        p_stdout, p_stderr = _raw(process.stdout), _raw(process.stderr)
        stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
                p_stdout.fileno(): (p_stdout, sys.stdout),
                p_stderr.fileno(): (p_stderr, sys.stderr),
                }
        fds = list(stream_by_fd.keys())

        while fds:
            ready, _, _ = select.select(fds, [], [])
            for fd in ready:
                p_stream, std = stream_by_fd[fd]
                raw_buf = p_stream.read(2 ** 16)
                if not raw_buf:
                    fds.remove(fd)
                    continue
                buf = raw_buf.decode()
                std.write(buf)
                std.flush()


class RMS(Visualizer):
    def __init__(self, in_path, demucs_in_path, out_path, threshold = 0.8, sr=44100, frame_length=65000, hop_length=16250, n_ignore=10):
        self.in_path = in_path
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_ignore = n_ignore
        self.demucs_in_path = demucs_in_path
        self.threshold = threshold

    def plot(self):
        rms, times = self._compute_rms(self.in_path)

        rms_data = []
        rms_data.append(self._compute_splited_rms(self.in_path, self.demucs_in_path + "/bass.mp3")[0])
        rms_data.append(self._compute_splited_rms(self.in_path, self.demucs_in_path + "/drums.mp3")[0])
        rms_data.append(self._compute_splited_rms(self.in_path, self.demucs_in_path + "/other.mp3")[0])
        rms_data.append(self._compute_splited_rms(self.in_path, self.demucs_in_path + "/vocals.mp3")[0])

        labels = ["bass", "drums", "vocals", "other"]

        self._plot_rms_with_color(times, rms_data, rms, labels)

    def compute_rms(self, file):
        y, _ = librosa.load(file, sr=self.sr, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        rms /= np.max(rms)
        times = np.floor(librosa.times_like(rms, hop_length=self.hop_length, sr=self.sr))

        return rms, times

    def _compute_splited_rms(self, file, s_file):
        s_y, s_sr = librosa.load(s_file, sr=44100, mono=True)
        y, _ = librosa.load(file, sr=44100, mono=True)
        s_rms = librosa.feature.rms(y=s_y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        s_rms /= np.max(rms)
        times = np.floor(librosa.times_like(s_rms, hop_length=self.hop_length, sr=s_sr))

        return s_rms, times

    def _plot_rms_with_color(self, times, rms_data, rms, labels):
        plt.figure(figsize=(15, 5))

        plt.plot(rms, color="black", lw=2.0)

        plt.ylabel("RMS")
        plt.xlabel("time")

        red_count = 0
        for i in range(len(times)):
            if rms[i] > self.threshold:
                red_count += 1
                plt.vlines(i, 0, 1, color="red", alpha=0.4)
            else:
                plt.vlines(i, 0, 1, color="green", alpha=0.4)
                if red_count == 0:
                    plt.vlines(i, 0, 1, color="yellow", alpha=0.4)

        red_count_2 = 0
        for i in range(len(times)):
            if red_count == red_count_2:
                plt.vlines(i, 0, 1, color="blue", alpha=0.4)
            if rms[i] > self.threshold:
                red_count_2 += 1

        colors = ["blue", "magenta", "yellow", "green"]
        for rms, label, color in zip(rms_data, labels, colors):
            plt.plot(rms, label=label, color=color ,lw=2, alpha=1)

        plt.legend()
        plt.show()


class Drum(Visualizer):
    def __init__(self):
        self.drum_mapping = {
                # 35: 'Acoustic Bass Drum',
                35: 'Kick',
                36: 'Bass Drum 1',
                37: 'Side Stick',
                # 38: 'Acoustic Snare',
                38: 'Snare',
                39: 'Hand Clap',
                40: 'Electric Snare',
                41: 'Low Floor Tom',
                # 42: 'Closed Hi-Hat',
                42: 'Hi-Hat',
                43: 'High Floor Tom',
                44: 'Pedal Hi-Hat',
                45: 'Low Tom',
                46: 'Open Hi-Hat',
                47: 'Low-Mid Tom',
                48: 'Hi-Mid Tom',
                49: 'Crash Cymbal 1',
                50: 'High Tom',
                51: 'Ride Cymbal 1',
                52: 'Chinese Cymbal',
                53: 'Ride Bell',
                54: 'Tambourine',
                55: 'Splash Cymbal',
                56: 'Cowbell',
                57: 'Crash Cymbal 2',
                58: 'Vibraslap',
                59: 'Ride Cymbal 2',
                60: 'Hi Bongo',
                61: 'Low Bongo',
                62: 'Mute Hi Conga',
                63: 'Open Hi Conga',
                64: 'Low Conga',
                65: 'High Timbale',
                66: 'Low Timbale',
                67: 'High Agogo',
                68: 'Low Agogo',
                69: 'Cabasa',
                70: 'Maracas',
                71: 'Short Whistle',
                72: 'Long Whistle',
                73: 'Short Guiro',
                74: 'Long Guiro',
                75: 'Claves',
                76: 'Hi Wood Block',
                77: 'Low Wood Block',
                78: 'Mute Cuica',
                79: 'Open Cuica',
                80: 'Mute Triangle',
                81: 'Open Triangle'
                }

    def get_drum_events(self, in_path):
        mid = mido.MidiFile(in_path)
        events = self._extract_events(mid)
        return events

    def _extract_events(self, mid):
        events = {}
        drum_counter = 0
        time = 0
        tempo = mido.bpm2tempo(120)

        for track in mid.tracks:
            for msg in track:
                time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
                new_tempo = self._extract_tempo(msg)
                if new_tempo is not None:
                    tempo = new_tempo
                elif self._is_drum_part(msg):
                    if msg.note not in events:
                        events[msg.note] = {'name': self.drum_mapping[msg.note], 'id': drum_counter, 'times': []}
                        drum_counter += 1
                    events[msg.note]['times'].append(time)
        return events

    def _plot_events(self, events):
        plt.figure(figsize=(15, 5))
        for drum_note, event in events.items():
            plt.eventplot(event['times'], orientation='horizontal', linelengths=0.08, lineoffsets=event['id'])
        plt.yticks([event['id'] for event in events.values()], [event['name'] for event in events.values()])
        plt.xlabel('Time')
        plt.ylabel('Drum elements')
        plt.title('Drum elements over time')
        plt.grid(True)
        plt.show()

    def _extract_tempo(self, message):
        if message.type == 'set_tempo':
            return message.tempo
        return None

    def _is_drum_part(self, message):
        return message.type == 'note_on' and message.note in self.drum_mapping

    def detect_pattern_changes(self, events):
        all_event_times = [time for event in events.values() for time in event['times']]
        avg_interval, std_deviation = self._calculate_similarity(all_event_times)
        pattern_changes = self._find_unique_integers(all_event_times, avg_interval, std_deviation)
        return pattern_changes

    def _find_unique_integers(self, event_times, avg_interval, std_deviation):
        unique_integers = []

        similarity_scores = []
        prev_change_time = None

        for i in range(1, len(event_times) - 1):
            prev_interval = event_times[i] - event_times[i - 1]
            next_interval = event_times[i + 1] - event_times[i]
            similarity_score = abs(prev_interval - avg_interval) + abs(next_interval - avg_interval)
            similarity_scores.append(similarity_score)

        threshold = std_deviation
        for i, similarity_score in enumerate(similarity_scores):
            if similarity_score > threshold:
                change_time = int(round(event_times[i]))
                if prev_change_time is None or change_time != prev_change_time:
                    unique_integers.append(change_time)
                prev_change_time = change_time

        for i in range(1, len(event_times) - 1):
            interval = event_times[i] - event_times[i - 1]

            if abs(interval - avg_interval) > std_deviation:
                unique_integers.append(int(round(event_times[i])))

        return list(set(unique_integers))

    def _calculate_similarity(self, times):
        intervals = np.diff(times)
        avg_interval = np.mean(intervals)
        std_deviation = np.std(intervals)
        return avg_interval, std_deviation

    def _plot_pattern_changes(self, events, pattern_changes):
        plt.figure(figsize=(15, 5))
        for drum_note, event in events.items():
            plt.eventplot(event['times'], orientation='horizontal', linelengths=0.08, lineoffsets=event['id'])
        plt.yticks([event['id'] for event in events.values()], [event['name'] for event in events.values()])
        plt.xlabel('Time')
        plt.ylabel('Drum elements')
        plt.title('Drum elements over time')
        plt.grid(True)

        for change_time in pattern_changes:
            plt.axvline(x=change_time, color='red', linestyle='--')

        plt.show()

    def plot_drum_with_pattern_changes(self, song_name, events, pattern_changes):
        plt.figure(figsize=(15, 5))
        for drum_note, event in events.items():
            plt.eventplot(event['times'], orientation='horizontal', linelengths=0.08, lineoffsets=event['id'])
        plt.yticks([event['id'] for event in events.values()], [event['name'] for event in events.values()])
        plt.xlabel('Time')
        plt.ylabel('Drum elements')
        plt.title(f'Drum elements over time - {song_name}')
        plt.grid(True)

        for change_time in pattern_changes:
            if change_time <= max(max(event['times']) for event in events.values()):
                plt.axvline(x=change_time, color='red', linestyle='--')

        plt.show()

    def plot_drum_with_pattern_and_sections(self, song_name, events, pattern_changes, section_changes):
        plt.figure(figsize=(15, 5))
        for drum_note, event in events.items():
            plt.eventplot(event['times'], orientation='horizontal', linelengths=0.08, lineoffsets=event['id'])
        plt.yticks([event['id'] for event in events.values()], [event['name'] for event in events.values()])
        plt.xlabel('Time')
        plt.ylabel('Drum elements')
        plt.title(f'Drum elements over time - {song_name}')
        plt.grid(True)

        for change_time in pattern_changes:
            if change_time <= max(max(event['times']) for event in events.values()):
                plt.axvline(x=change_time, color='red', linestyle='--')

        for time in section_changes:
            plt.axvline(x=time, color='green', linestyle=':')

        """
        for segment in section_data['segments']:
            start_time = segment['start']
            if start_time <= max(max(event['times']) for event in events.values()):
                plt.axvline(x=start_time, color='green', linestyle=':')
        """

        plt.show()


class Frequency:
    def __init__(self):
        pass

    def get_spectral_centroid(self, audio_file: str, n_fft=2048*2) -> Tuple[np.ndarray, float, np.ndarray]:
        y, sr = librosa.load(audio_file, sr=None)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
        times = librosa.times_like(spectral_centroid, sr=sr)
        return spectral_centroid, sr, times

    def get_spectrogram(self, audio_file: str) -> List[List[float]]:
        y, sr = librosa.load(audio_file, sr=None)
        spectrogram = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
        # self._plot_spectrogram(spectrogram)
        return spectrogram, sr

    def _plot_spectral_centroid(self, spectral_centroid):
        plt.figure(figsize=(10, 6))
        plt.semilogy(spectral_centroid.T, label='Spectral Centroid')
        plt.ylabel('Hz')
        plt.xticks([])
        plt.xlim([0, spectral_centroid.shape[-1]])
        plt.legend(loc='upper right')
        plt.title("Spectral Centroid")
        plt.show()

    def _plot_spectrogram(self, spectrogram):
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(spectrogram, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        plt.show()


class Allin1:
    def format_json(self, path):
        json_files = [file for file in os.listdir(path) if file.endswith('.json')]

        for json_file in json_files:
            file_path = os.path.join(path, json_file)

            with open(file_path, 'r') as file:
                data = json.load(file)

            for item in list(data.keys()):
                if item not in ('path', 'segments'):
                    del data[item]

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"{colored('format_json', 'blue')}: Update '{json_file}'.")

        print("All json files have been updated.")

    def update_path_json(self, path):
        json_files = [file for file in os.listdir(path) if file.endswith('.json')]

        for json_file in json_files:
            file_path = os.path.join(path, json_file)

            with open(file_path, 'r') as file:
                data = json.load(file)

            data['path'] = file_path

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"{colored('update_path_json', 'blue')}: Update path of '{json_file}'.")

        print("All json files have been updated.")

    def modify_label(self, label):
        label_mappings = {
                'start': 'intro',
                'end': 'outro',
                'bridge': 'break',
                'inst': 'break',
                'solo': 'break',
                'verse': 'break',
                'chorus': 'drop'
                }
        return label_mappings.get(label, label)

    def modify_json(self, path):
        json_files = [file for file in os.listdir(path) if file.endswith('.json')]

        for json_file in json_files:
            file_path = os.path.join(path, json_file)

            with open(file_path, 'r') as file:
                data = json.load(file)

            if 'segments' in data and isinstance(data['segments'], list):
                for segment in data['segments']:
                    if 'label' in segment:
                        segment['label'] = self.modify_label(segment['label'])

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"{colored('modify_json', 'blue')}: Modified '{json_file}'.")

        print("All json files have been modified.")

    def load_section_data(self, json_path: str):
        with open(json_path, 'r') as file:
            data = json.load(file)
        return data

    def seconds_to_min_sec(self, seconds):
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:05.2f}"

    def min_sec_to_seconds(self, min_sec):
        minutes, seconds = map(float, min_sec.split(':'))
        return minutes * 60 + seconds

    def convert_time_format(self, path):
        json_files = [file for file in os.listdir(path) if file.endswith('.json')]

        for json_file in json_files:
            file_path = os.path.join(path, json_file)

            with open(file_path, 'r') as file:
                data = json.load(file)

            if 'segments' in data and isinstance(data['segments'], list):
                for segment in data['segments']:
                    if 'start' in segment:
                        segment['start'] = self.seconds_to_min_sec(segment['start'])
                    if 'end' in segment:
                        segment['end'] = self.seconds_to_min_sec(segment['end'])

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"Converted time format in '{json_file}'.")

        print("All json files have been updated.")

    def revert_time_format(self, path):
        json_files = [file for file in os.listdir(path) if file.endswith('.json')]

        for json_file in json_files:
            file_path = os.path.join(path, json_file)

            with open(file_path, 'r') as file:
                data = json.load(file)

            if 'segments' in data and isinstance(data['segments'], list):
                for segment in data['segments']:
                    if 'start' in segment:
                        segment['start'] = self.min_sec_to_seconds(segment['start'])
                    if 'end' in segment:
                        segment['end'] = self.min_sec_to_seconds(segment['end'])

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"Reverted time format in '{json_file}'.")

        print("All json files have been updated.")


# matplotlibで箱ひげ図の上に検定のP値を表示する関数
def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)
