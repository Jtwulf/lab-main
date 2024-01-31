from external_libraries import *
from modules import *
import data_const as const
from experiment2 import *

def perform_kruskal_wallis_test_by_component(component_averages, component):
    data = [component_averages[component][section] for section in component_averages[component] if component_averages[component][section]]
    if len(data) > 1:
        stat, p = kruskal(*data)
        print(f"Kruskal-Wallis test for {component}: Statistics = {stat}, p-value = {p}")
        return p < 0.05
    return False

def perform_dunn_test_by_component(component_averages, component):
    data = []
    groups = []
    for section in component_averages[component]:
        for value in component_averages[component][section]:
            if value is not None:
                data.append(value)
                groups.append(section)
    if data and groups:
        df = pd.DataFrame({'Value': data, 'Group': groups})
        dunn_p_values = scikit_posthocs.posthoc_dunn(df, val_col='Value', group_col='Group', p_adjust='bonferroni')
        print(f"Dunn's test results for {component} (p-values):")
        print(dunn_p_values)

def perform_anova_on_components(component_averages):
    for component in component_averages:
        if perform_kruskal_wallis_test_by_component(component_averages, component):
            perform_dunn_test_by_component(component_averages, component)

def get_rms(file_path):
    y, sr = librosa.load(file_path)
    rms = librosa.feature.rms(y=y)
    times = librosa.times_like(rms, sr=sr)
    return rms, sr, times

def calculate_section_averages(sections, feature_values, sr, times):
    section_averages = {'intro': [], 'drop': [], 'break': [], 'outro': []}
    feature_values = feature_values.flatten()

    for section in sections:
        label = section['label']
        if label in section_averages:
            start_index = np.argmax(times >= section['start'])
            end_index = np.argmax(times >= section['end'])
            if end_index == 0:
                end_index = len(feature_values)

            section_centroid = feature_values[start_index:end_index]
            if len(section_centroid) > 0:
                section_average = section_centroid.mean()
                section_averages[label].append(section_average)

    for label in section_averages:
        if section_averages[label]:
            section_averages[label] = sum(section_averages[label]) / len(section_averages[label])
        else:
            section_averages[label] = None

    return section_averages

def plot_box_plot(section_averages, title):
    data_to_plot = [avgs for avgs in section_averages.values()]
    plt.boxplot(data_to_plot, labels=section_averages.keys())

    plt.xlabel('Section')
    plt.ylabel('Average RMS')
    plt.title(title)

    y_min, y_max = plt.ylim()
    margin = (y_max - y_min) * 0.3
    plt.ylim(y_min - margin, y_max + margin)

    plt.tight_layout()
    plt.show()

def process_file(json_path, song_directory, component_averages, allin1, components):
    section_data = allin1.load_section_data(json_path)
    song_name = os.path.splitext(os.path.basename(json_path))[0]

    for component in components:
        file_path = os.path.join(song_directory, song_name, f"{component}.mp3")
        if os.path.exists(file_path):
            rms, sr, times = get_rms(file_path)
            section_averages = calculate_section_averages(section_data['segments'], rms, sr, times)

            for section, average in section_averages.items():
                if average is not None:
                    component_averages[component][section].append(average)

def process_files(json_directory, song_directory, allin1, component_averages, components):
    for root, dirs, files in tqdm(os.walk(json_directory), desc="Processing files"):
        for file in tqdm(files, desc="Overall Progress", leave=False):
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                process_file(json_path, song_directory, component_averages, allin1, components)

def main(process_mode):
    song_directory = const.PROD_SONG_DIRECTORY
    json_directory = const.PROD_JSON_DIRECTORY
    demucs_directory = const.PROD_DEMUCS_DIRECTORY
    allin1 = Allin1()

    components = ['bass', 'drums', 'other', 'vocals']
    component_averages = {component: {'intro': [], 'drop': [], 'break': [], 'outro': []} for component in components}

    process_files(json_directory, demucs_directory, allin1, component_averages, components)

    perform_anova_on_components(component_averages)

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
    process_mode = 'box'  # 'bar' | 'combined_bar' | 'box' | 'combined_box' | 'violin' | 'combined_violin'
    main(process_mode)
