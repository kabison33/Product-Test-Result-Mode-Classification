if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import json as js
    import os

    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    from sklearn.pipeline import Pipeline

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import LabelEncoder

    from sklearn.decomposition import PCA
    from sklearn.decomposition import KernelPCA

    from sklearn.ensemble import RandomForestClassifier as rfc
    from sklearn.preprocessing import FunctionTransformer

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    from feature_sample_lib import poss_ear_sn, T1_early_sign_sample_id, T1_early_sign_sn


    target_dimension = 2
    clustering = 'none'

    dataset_opt = ['T7000 ATF'] # sample_a , sample_b

    selective_coloring_samples = T1_early_sign_sn

    feature_group_opt = ['P2T2_RELAY_CHA', 'P2T2_RELAY_CHB']


    scaler_opt = ['mm'] # mm , std , rob
    reducer_opt = ['pca'] # kpca , pca

    scaler_dict = {
                        'mm': MinMaxScaler(),
                        'std': StandardScaler()
    }

    def plotter(row, col, points, title):
        return 0

    for i in dataset_opt:
        print('Analytic Pipeline for: ' + i)
        print('--------------------------------')
        for j in feature_group_opt:
            print('\nProcessing Test Model (Feature Group): ' + j + '\n')
            for k in scaler_opt:
                print('Using %s Scaler...'%(k))
                for l in reducer_opt:
                    #reading dataset and labels, remove duplicates
                    dataset_df = pd.read_csv('../dataset/%s/%s/%s_samples.csv'%(i, j, j), usecols = ['sample_id', 'feature', 'value', 'mpv'])
                    dataset_df.drop_duplicates(subset = ['sample_id', 'feature'], keep = 'first', inplace = True)

                    labels_df = pd.read_csv('../dataset/%s/%s/%s_labels.csv'%(i, j, j), usecols = ['sample_id', 'class', 'station', 'ita_id', 'harness_a', 'harness_b', 'event_time'])
                    labels_df.drop_duplicates(subset = ['sample_id'], keep = 'first', inplace = True)

                    #setting dimension
                    original_dimension = len(dataset_df['feature'].drop_duplicates())
                    target_dimension = original_dimension if target_dimension == 0 else target_dimension

                    #collecting stats
                    sample_size = len(dataset_df['sample_id'].drop_duplicates())
                    no_of_meas = len(dataset_df['feature'].drop_duplicates())
                    no_of_sample = len(dataset_df['sample_id'].drop_duplicates())

                    x = []
                    fn = []
                    sample_list = []

                    #adding prog version column to the labels dataframe
                    program_versions = dataset_df[['sample_id', 'mpv']].drop_duplicates()
                    labels_df = labels_df.merge(program_versions, on='sample_id', how='left')

                    #checking if the sample ids for labels and samples dataframe are in the same order and not having duplicates.
                    print('all ok' if all( dataset_df['sample_id'].drop_duplicates() == labels_df['sample_id'] ) else 'SAMPLE LENGTHS DIFFER')

                    #transforming labels dataframe
                    labels_df['sn'] = labels_df['sample_id'].apply(lambda x: x.split(' - ')[0])
                    labels_df['sn_short'] = labels_df['sample_id'].apply(lambda x: x.split(' - ')[0].split('BX70')[1])

                    labels_df['failure_mode_color'] = labels_df['class'].apply(lambda x: 'green' if x == 'PASS' else ('blue' if x == 'RFT' else 'red'))
                    labels_df['failure_mode_color'] = labels_df['class'].apply(lambda x: 'blue' if x == 'RFT' else ('red' if x.startswith('FAIL - ') else 'green'))
                    labels_df['failure_mode_color'] = labels_df.apply(lambda row: 'orange' if row['sn'] in selective_coloring_samples else row['failure_mode_color'], axis = 1)

                    labels_df['result'] = labels_df['failure_mode_color'].apply(lambda x: 0 if x == 'red' else 1)

                    #creating labels arrays for labeling scatter points
                    serials = np.array(labels_df['sn'])
                    serials_short = np.array(labels_df['sn_short'])
                    failure_modes = np.array(labels_df['class'])
                    failure_mode_colors = np.array(labels_df['failure_mode_color'])
                    results = np.array(labels_df['result'])

                    #creating labels arrays for coloring the scatter plot.
                    unique_station = np.unique(labels_df['station'])
                    unique_ita = np.unique(labels_df['ita_id'])
                    unique_mpv = np.unique(labels_df['mpv'])
                    unique_harna = np.unique(labels_df['harness_a'])
                    unique_harnb = np.unique(labels_df['harness_b'])

                    station_num_dict = {cat: i for i, cat in enumerate(unique_station)}
                    ita_num_dict = {cat: i for i, cat in enumerate(unique_ita)}
                    mpv_num_dict = {cat: i for i, cat in enumerate(unique_mpv)}
                    harna_num_dict = {cat: i for i, cat in enumerate(unique_harna)}
                    harnb_num_dict = {cat: i for i, cat in enumerate(unique_harnb)}

                    station_numeric = np.array([station_num_dict[cat] for cat in labels_df['station']])
                    ita_numeric = np.array([ita_num_dict[cat] for cat in labels_df['ita_id']])
                    mpv_numeric = np.array([mpv_num_dict[cat] for cat in labels_df['mpv']])
                    harna_numeric = np.array([harna_num_dict[cat] for cat in labels_df['harness_a']])
                    harnb_numeric = np.array([harnb_num_dict[cat] for cat in labels_df['harness_b']])

                    station_cmap = plt.get_cmap('rainbow', len(unique_station))
                    ita_cmap = plt.get_cmap('rainbow', len(unique_ita))
                    mpv_cmap = plt.get_cmap('rainbow', len(unique_mpv))
                    harna_cmap = plt.get_cmap('rainbow', len(unique_harna))
                    harnb_cmap = plt.get_cmap('rainbow', len(unique_harnb))

                    station_colors = station_cmap(station_numeric)
                    ita_colors = ita_cmap(ita_numeric)
                    mpv_colors = mpv_cmap(mpv_numeric)
                    harna_colors = harna_cmap(harna_numeric)
                    harnb_colors = harnb_cmap(harnb_numeric)

                    #creating datapoints
                    for m in dataset_df['sample_id'].drop_duplicates():
                        sample_list.append(m)

                        temp = dataset_df[dataset_df['sample_id'] == m]
                        x.append( np.array(temp['value'].tolist()) )

                    #array contents checking
                    if len(x) == 0:
                        print('dataset %s is empty'%(i))
                        break

                    if len(x[0]) == 1:
                        print('Test model is only 1 dimension. Skipping.')
                        break

                    #printing test_model sample info
                    print('sample count: ', len(x))
                    print('feature count: ', len(x[0]))

                    print('color len: ', len(failure_mode_colors))

                    #convert array to np.array for easier slicing
                    x = np.array(x)

                    #Feature Selection
                    # x, fmask = feature_reducer(x, results, 'lasso', True, 0.000686648845004)

                    #creating pipeline
                    pipe = Pipeline([ ('scaler', scaler_dict[k] ), ('reducer', PCA(n_components = len(x[0]) if len(x[0]) < target_dimension else target_dimension)) ])

                    x = pipe.fit_transform(x)

                    #Creating scree plot
                    print('Creating Scree Plot..')
                    PC_values = np.arange(pipe.named_steps['reducer'].n_components_) + 1
                    plt.figure(figsize = (10,10))
                    plt.plot(PC_values, pipe.named_steps['reducer'].explained_variance_ratio_, 'o-', linewidth=2, color='blue')
                    plt.title('%s - %s feature group - %s norm - %dD Scree Plot'%(i, j, k, target_dimension))
                    plt.xlabel('Principal Component')
                    plt.ylabel('Variance Explained')
                    plt.savefig('../figures/%s/scree plot/%s_dataset_%s_%s_%dD'%(i, i, j, k, target_dimension))
                    plt.cla()
                    plt.clf()
                    plt.close()

                    print('Plotting Samples..\n')

                    mainplot_width, mainplot_height = 100,94 # 18,12 IS ORIGINAL SIZE
                    subplot_title_fontsize = 12
                    subplot_title_fontweight = 'bold'
                    subplot_legend_fontsize = 8
                    subplot_legend_handletextpad = 0
                    point_size = 10
                    point_size_fail = 20

                    bigfig, axes = plt.subplots(2,3, figsize = (mainplot_width,mainplot_height))
                    bigfig.suptitle('%s - %s - %s Plot'%(i, j, k), fontsize = 15, fontweight = 'bold', y = 0.99)

                    #preparing scatter plots
                    custom_handles = [Line2D([0], [0], color = 'green'), Line2D([0], [0], color = 'blue'), Line2D([0], [0], color = 'orange'), Line2D([0], [0], color = 'red')]
                    aaa = axes[0,0].scatter(x[:,0], x[:,1], c=labels_df['failure_mode_color'].tolist(), s=point_size )

                    # for clr in set(failure_mode_colors):
                    #     mask = failure_mode_colors == clr
                    #     legend_text = failure_modes[mask][0][:4] if failure_modes[mask][0][:4] == 'FAIL' else failure_modes[mask][0]
                    #     aaa = axes[0,0].scatter(x[mask][:,0], x[mask][:,1], c=clr, label = legend_text, s=point_size )

                    annotation = axes[0,0].annotate(
                        'Alhamdulillah',
                        xy=(0, 0),
                        xytext=(0, 0),
                        textcoords='offset points'
                    )

                    annotation.set_visible(False)


                    def motion_hover(event):
                        annotation_visibility = annotation.get_visible() #get visibility of annotation
                        if event.inaxes == axes[0,0]:
                            is_contained, annotation_index = aaa.contains(event)
                            if is_contained:
                                data_point_location = aaa.get_offsets()[annotation_index['ind'][0]]
                                annotation.xy = data_point_location

                                text_label = '  %s'%(serials[annotation_index['ind'][0]])
                                annotation.set_text(text_label)

                                annotation.set_visible(True)
                                bigfig.canvas.draw_idle()
                            else:
                                if annotation_visibility:
                                    annotation.set_visible(False)
                                    bigfig.canvas.draw_idle()

                    bigfig.canvas.mpl_connect('motion_notify_event', motion_hover)

                    # #annotating failed points with text of failure mode
                    # fail_mask = failure_mode_colors == 'red'
                    # x_failed = x[fail_mask]
                    # class_failed = failure_modes[fail_mask]

                    # for z in range(len(x_failed)):
                    #     axes[0,0].text(x_failed[z][0], x_failed[z][1], s = class_failed[z], size=7)

                    #configuring plot titles and legends
                    axes[0,0].legend(custom_handles, ['PASS', 'RFT', 'ES', 'FAIL'], fontsize = subplot_legend_fontsize, markerscale = 2, handletextpad = 0)
                    axes[0,0].legend(fontsize = subplot_legend_fontsize, markerscale = 2, handletextpad = 0)
                    axes[0,0].set_title('Failure Modes', fontsize = subplot_title_fontsize, fontweight = subplot_title_fontweight)
                    axes[0,0].set(xticklabels=[])
                    axes[0,0].set(yticklabels=[])

                    #plotting station variance
                    for st in np.unique(labels_df['station']):
                        mask = np.array(labels_df['station'] == st)
                        axes[0,1].scatter(x[mask][:,0], x[mask][:,1], c=station_colors[mask], label = st, s=point_size)

                    axes[0,1].legend(fontsize = subplot_legend_fontsize, markerscale = 2, handletextpad = 0)
                    axes[0,1].set_title('Stations', fontsize = subplot_title_fontsize, fontweight = subplot_title_fontweight)
                    axes[0,1].set(xticklabels=[])
                    axes[0,1].set(yticklabels=[])

                    #plotting harness a variance
                    for ha in np.unique(labels_df['harness_a']):
                        mask = np.array(labels_df['harness_a'] == ha)
                        axes[0,2].scatter(x[mask][:,0], x[mask][:,1], c=harna_colors[mask], label = ha, s=point_size )

                    axes[0,2].legend(fontsize = subplot_legend_fontsize, markerscale = 2, handletextpad = 0)
                    axes[0,2].set_title('Harness A', fontsize = subplot_title_fontsize, fontweight = subplot_title_fontweight)
                    axes[0,2].set(xticklabels=[])
                    axes[0,2].set(yticklabels=[])

                    #plotting ita variance
                    for ita in np.unique(labels_df['ita_id']):
                        mask = np.array(labels_df['ita_id'] == ita)
                        axes[1,0].scatter(x[mask][:,0], x[mask][:,1], c=ita_colors[mask], label = ita, s=point_size )

                    axes[1,0].legend(fontsize = subplot_legend_fontsize, markerscale = 2, handletextpad = 0)
                    axes[1,0].set_title('ITAs', fontsize = subplot_title_fontsize, fontweight = subplot_title_fontweight)
                    axes[1,0].set(xticklabels=[])
                    axes[1,0].set(yticklabels=[])

                    #plotting measurement program variance
                    for p in np.unique(labels_df['mpv']):
                        mask = np.array(labels_df['mpv'] == p)
                        axes[1,1].scatter(x[mask][:,0], x[mask][:,1], c=mpv_colors[mask], label = p, s=point_size )

                    axes[1,1].legend(fontsize = subplot_legend_fontsize, markerscale = 2, handletextpad = 0)
                    axes[1,1].set_title('Test Programs', fontsize = subplot_title_fontsize, fontweight = subplot_title_fontweight)
                    axes[1,1].set(xticklabels=[])
                    axes[1,1].set(yticklabels=[])

                    #plotting harness b variance
                    # for hb in np.unique(labels_df['harness_b']):
                    #     mask = np.array(labels_df['harness_b'] == hb)
                    #     axes[1,2].scatter(x[mask][:,0], x[mask][:,1], c=harnb_colors[mask], label = hb, s=point_size )

                    # axes[1,2].legend(fontsize = subplot_legend_fontsize, markerscale = 2, handletextpad = 0)
                    # axes[1,2].set_title('Harness B', fontsize = subplot_title_fontsize, fontweight = subplot_title_fontweight)
                    # axes[1,2].set(xticklabels=[])
                    # axes[1,2].set(yticklabels=[])

                    bigfig.subplots_adjust(left = 0.01, bottom = 0.02, top = 0.95, right = 0.99, hspace = 0.05, wspace = 0.05)

                    bigfig.savefig('../figures/%s/scatter/%s_%s_%s'%(i, i, j, k))
                    plt.show()
                    plt.cla()
                    plt.clf()
                    plt.close()
