from experiment1 import *
from external_libraries import *
from modules import *
import data_const as const

def plot_bar_graph(section_averages):
    total_averages = {}
    for section, avgs in section_averages.items():
        if avgs:
            total_averages[section] = np.mean(avgs)

    plt.bar(total_averages.keys(), total_averages.values(), alpha=0.5, label='Average per section')
    for section, avgs in section_averages.items():
        if avgs:
            plt.scatter([section] * len(avgs), avgs, color='red', label='Individual averages' if section == 'intro' else "")

    plt.xlabel('Section')
    plt.ylabel('Average RMS')
    plt.title('Average RMS per Music Section')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_box_plot(section_averages):
    plt.boxplot(section_averages.values(), labels=section_averages.keys(), showmeans=True)
    plt.xlabel('Section')
    plt.ylabel('Average RMS')
    plt.title('Box Plot of Average RMS per Music Section')

    y_min, y_max = plt.ylim()
    margin = (y_max - y_min) * 0.3
    plt.ylim(y_min - margin, y_max + margin)

    plt.tight_layout()
    plt.show()

def plot_violin_plot(section_averages):
    data_to_plot = [avgs for avgs in section_averages.values() if avgs]
    plt.violinplot(data_to_plot)
    plt.xticks(range(1, len(section_averages) + 1), section_averages.keys())
    plt.xlabel('Section')
    plt.ylabel('Average RMS')
    plt.title('Violin Plot of Average RMS per Music Section')
    plt.tight_layout()
    plt.show()


def get_rms(file_path):
    y, sr = librosa.load(file_path)
    rms = librosa.feature.rms(y=y)
    times = librosa.times_like(rms)
    return rms, sr, times

def process_file(json_path, song_directory, all_section_averages, allin1):
    section_data = allin1.load_section_data(json_path)

    filename = os.path.splitext(os.path.basename(json_path))[0] + '.mp3'
    file_path = os.path.join(song_directory, filename)
    rms_values, sr, times = get_rms(file_path)
    section_averages = calculate_section_averages(section_data['segments'], rms_values, sr, times)

    for section, average in section_averages.items():
        all_section_averages[section].append(average)

def process_files(json_directory, song_directory, allin1, all_section_averages):
    for root, dirs, files in tqdm(os.walk(json_directory), desc="Processing files"):
        for file in tqdm(files, desc="Overall Progress", leave=False):
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                process_file(json_path, song_directory, all_section_averages, allin1)

def main(process_mode):
    song_directory = const.PROD_SONG_DIRECTORY
    json_directory = const.PROD_JSON_DIRECTORY
    allin1 = Allin1()
    all_section_averages = {'intro': [], 'drop': [], 'break': [], 'outro': []}

    process_files(json_directory, song_directory, allin1, all_section_averages)

    # 対数変換をした上でダゴスティーノのK^2検定(正規性の検討)
    transformed_data = reevaluate_normality(all_section_averages)
    # レヴィンの検定(等分散性の検討)
    check_homoscedasticity(all_section_averages)
    # クラスカル・ウォリス検定を行い，有意差があった場合にダンの検定
    perform_dunn_test(all_section_averages)

    if process_mode == 'bar':
        plot_bar_graph(all_section_averages)
    elif process_mode == 'box':
        plot_box_plot(all_section_averages)
    elif process_mode == 'violin':
        plot_violin_plot(all_section_averages)

if __name__ == "__main__":
    process_mode = 'box'  # 'bar' | 'box' | 'violin'
    main(process_mode)
