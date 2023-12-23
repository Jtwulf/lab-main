from external_libraries import *
from modules import *

def load_section_data(json_path: str):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_section_averages(sections, spectral_centroid, sr, times):
    section_averages = {'intro': [], 'drop': [], 'break': [], 'outro': []}
    spectral_centroid = spectral_centroid.flatten()

    for section in sections:
        label = section['label']
        if label in section_averages:
            start_index = np.argmax(times >= section['start'])
            end_index = np.argmax(times >= section['end'])
            if end_index == 0:
                end_index = len(spectral_centroid)

            section_centroid = spectral_centroid[start_index:end_index]
            if len(section_centroid) > 0:
                section_average = section_centroid.mean()
                section_averages[label].append(section_average)

    for label in section_averages:
        if section_averages[label]:
            section_averages[label] = sum(section_averages[label]) / len(section_averages[label])
        else:
            section_averages[label] = None

    return section_averages

def plot_bar_graph(section_averages, title):
    total_averages = {section: np.mean([avg for avg in avgs if avg is not None])
                      for section, avgs in section_averages.items()}

    plt.bar(total_averages.keys(), total_averages.values(), alpha=0.5, label='Average per section')
    for section, avgs in section_averages.items():
        plt.scatter([section] * len(avgs), avgs, color='red', alpha=0.6, label='Individual averages' if section == 'intro' else "")

    plt.xlabel('Section')
    plt.ylabel('Average Spectral Centroid')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_combined_bar_graph(component_averages, components):
    colors = ['blue', 'green', 'red', 'purple']
    bar_width = 0.45
    gap_width = 0.25
    group_width = len(components) * bar_width + (len(components) - 1) * gap_width
    positions = np.arange(1, len(components) * 4, 4)

    for i, comp in enumerate(components):
        for j, section in enumerate(['intro', 'drop', 'break', 'outro']):
            data = component_averages[comp][section]
            if data:
                avg = np.mean(data)
                position = positions[j] + i * (bar_width + gap_width)
                plt.bar(position, avg, color=colors[i], width=bar_width, label=comp if j == 0 else "")
                for d in data:
                    plt.scatter(position, d, color=colors[i], edgecolor='black')

    plt.axvline(x=4, color='gray', linestyle='--')
    plt.axvline(x=8, color='gray', linestyle='--')
    plt.axvline(x=12, color='gray', linestyle='--')

    plt.xticks(positions + group_width / 2, ['Intro', 'Drop', 'Break', 'Outro'])
    plt.xlabel('Section')
    plt.ylabel('Average Spectral Centroid')
    plt.title('Combined Bar Graph for Each Component')

    plt.legend()
    plt.show()

def plot_box_plot(section_averages, title):
    data_to_plot = [avgs for avgs in section_averages.values()]
    plt.boxplot(data_to_plot, labels=section_averages.keys())
    plt.xlabel('Section')
    plt.ylabel('Average Spectral Centroid')
    plt.title(title)
    plt.show()

def plot_violin_plot(section_averages, title):
    data_to_plot = [avgs for avgs in section_averages.values() if avgs]
    plt.violinplot(data_to_plot)
    plt.xticks(range(1, len(section_averages) + 1), section_averages.keys())
    plt.xlabel('Section')
    plt.ylabel('Average Spectral Centroid')
    plt.title(title)
    plt.show()

def plot_combined_box_plot(component_averages, components):
    colors = ['blue', 'green', 'red', 'purple']
    positions = np.arange(1, len(components) * 4, 4)

    for i, comp in enumerate(components):
        data = [component_averages[comp][section] for section in ['intro', 'drop', 'break', 'outro'] if component_averages[comp][section]]
        pos = positions + i
        plt.boxplot(data, positions=pos, patch_artist=True, boxprops=dict(facecolor=colors[i]))

    plt.xticks(positions + 1.5, ['Intro', 'Drop', 'Break', 'Outro'])
    plt.axvline(x=4.5, color='gray', linestyle='--')
    plt.axvline(x=8.5, color='gray', linestyle='--')
    plt.axvline(x=12.5, color='gray', linestyle='--')

    plt.xlabel('Section')
    plt.ylabel('Average Spectral Centroid')
    plt.title('Combined Box Plot for Each Component')

    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors], components)
    plt.show()

def plot_combined_violin_plot(component_averages, components):
    colors = ['blue', 'green', 'red', 'purple']
    positions = np.arange(1, len(components) * 4, 4)

    for i, comp in enumerate(components):
        data = [component_averages[comp][section] for section in ['intro', 'drop', 'break', 'outro'] if component_averages[comp][section]]
        pos = positions + i
        plt.violinplot(data, positions=pos, showmeans=True, showmedians=True, showextrema=True)

    plt.xticks(positions + 1.5, ['Intro', 'Drop', 'Break', 'Outro'])
    plt.axvline(x=4.5, color='gray', linestyle='--')
    plt.axvline(x=8.5, color='gray', linestyle='--')
    plt.axvline(x=12.5, color='gray', linestyle='--')

    plt.xlabel('Section')
    plt.ylabel('Average Spectral Centroid')
    plt.title('Combined Violin Plot for Each Component')

    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors], components)
    plt.show()

def main(process_mode):
    song_directory = "../data/demo/songs_demo/"
    demucs_directory = "../data/demo/demucs_demo/mdx_q/"
    json_directory = "../data/demo/allin1_demo/"
    freq = Frequency()

    components = ['bass', 'drums', 'other', 'vocals']
    component_averages = {component: {'intro': [], 'drop': [], 'break': [], 'outro': []} for component in components}

    for root, dirs, files in os.walk(json_directory):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                section_data = load_section_data(json_path)

                song_name = os.path.splitext(file)[0]

                for component in components:
                    file_path = os.path.join(demucs_directory, song_name, f"{component}.mp3")
                    if os.path.exists(file_path):
                        spectral_centroid, sr, times = freq.get_spectral_centroid(file_path)
                        section_averages = calculate_section_averages(section_data['segments'], spectral_centroid, sr, times)

                        for section, average in section_averages.items():
                            if average is not None:
                                component_averages[component][section].append(average)

    if process_mode == 'bar':
        for component in components:
            plot_bar_graph(component_averages[component], f"Bar Graph for {component.capitalize()}")
    elif process_mode == 'combined_bar':
        plot_combined_bar_graph(component_averages, components)
    elif process_mode == 'box':
        for component in components:
            plot_box_plot(component_averages[component], f"Box Plot for {component.capitalize()}")
    elif process_mode == 'combined_box':
        plot_combined_box_plot(component_averages, components)
    elif process_mode == 'violin':
        for component in components:
            plot_violin_plot(component_averages[component], f"Violin Plot for {component.capitalize()}")
    elif process_mode == 'combined_violin':
        plot_combined_violin_plot(component_averages, components)


if __name__ == "__main__":
    process_mode = 'combined_bar'  # 'bar' | 'combined_bar' | 'box' | 'combined_box' | 'violin' | 'combined_violin'
    main(process_mode)
