import pandas as pd
import numpy as np
import json as js
import os
from datetime import datetime
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA

from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

x = []
sn_list = []

file_path = 'dataset/sample_a_dataset_full_mm_norm.csv'
normalizer = file_path.split('_')[-2]
sample = file_path.split('/')[1].split('_dataset')[0]
dataset_list = os.listdir('dataset')
target_dimension = 0

target_dimension = 2
# sample_opt = ['sample_b'] # sample_a , sample_b
# sample_focus_opt = ['IDR01', 'TMD01', 'ADB01'] # full , CDN01
# normalizer_opt = ['mm', 'std'] # mm , std , rob
sample_opt = list(set( ["_".join(i.split('_')[:2]) for i in dataset_list] ))
sample_focus_opt = list(set( [i.split('_')[3] for i in dataset_list] ))
normalizer_opt = list(set( [i.split('_')[4] for i in dataset_list] ))
reducer_opt = ['pca'] # kpca , pca


for i in sample_opt:
    for j in sample_focus_opt:
        for k in normalizer_opt:
            for l in reducer_opt:

                try:
                    data = pd.read_csv('dataset/%s_dataset_%s_%s_norm.csv'%(i, j, k))
                except FileNotFoundError as fnf_data:
                    #print('%s_dataset_%s_%s_norm.csv'%(i, j, k), ' file not found')
                    break

                try:
                    classes_df = pd.read_csv('raw_data/%s_label.csv'%(i))
                except FileNotFoundError as fnf_label:
                    #print(fnf_label)
                    break

                print('%s_dataset_%s_%s_norm_%s_dim'%(i, j, k, str(target_dimension)))
                print('-------------------------------------')

                sn_class_dict = dict( zip(classes_df['SN'], classes_df['LABEL']) )
                class_color_dict = dict( zip( classes_df['LABEL'].drop_duplicates().tolist() , list(range(len( classes_df['LABEL'].drop_duplicates().tolist() ))) ) )

                original_dimension = len(data['Feature'].drop_duplicates())
                target_dimension = original_dimension if target_dimension == 0 else target_dimension

                sample_size = len(data['SN'].drop_duplicates())
                no_of_meas = len(data['Feature'].drop_duplicates())
                no_of_sample = len(data['SN'].drop_duplicates())

                # PREPARING SAMPLES
                # --------------------------
                x = []
                y = []
                sn_list = []
                color_list = []

                for m in data['SN'].drop_duplicates():
                    x.append( data[data['SN'] == m].sort_values(by = 'Feature')['Value'].tolist() )
                    y.append(sn_class_dict[m])
                    sn_list.append(m)
                    color_list.append( class_color_dict[ sn_class_dict[m] ] )

                if len(x) == 0:
                    print('No analysis done on' +' %s_dataset_%s_%s_norm.csv'%(i, j, k) + ' as it has 0 feature\n')
                    break

                # DIMENSION REDUCTION
                # --------------------
                pca = PCA(n_components = target_dimension)

                if target_dimension != 0:
                    if l == 'pca':
                        x = pca.fit_transform(x)
                    else:
                        raise ValueError("no reducer have been selected")
                else:
                    pass

                PC_values = np.arange(pca.n_components_) + 1
                plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
                plt.title('Scree Plot')
                plt.xlabel('Principal Component')
                plt.ylabel('Variance Explained')
                plt.savefig('figure/pca_scree_plot/%s_dataset_%s_%s_%sD'%(i, j, k, str(target_dimension)))
                plt.clf()

                # TRAIN-TEST SPLIT
                # -----------------
                x_train, x_test, y_train, y_test, sn_train, sn_test = [], [], [], [], [], []
                for i in range(len(sn_list)):
                    if sn_list[i] in ['bx270176']:
                        x_test.append(x[i])
                        y_test.append(y[i])
                        sn_test.append(sn_list[i])
                    else:
                        x_train.append(x[i])
                        y_train.append(y[i])
                        sn_train.append(sn_list[i])

                #x_train, x_test, y_train, y_test = train_test_split(x, np.column_stack((y, sn_list)), test_size=0.3, random_state=41)


                # CLASSIFICATION
                # ---------------
                # svc = SVC(random_state=42, probability=True)
                # svc.fit(x_train, y_train)
                # prediction = svc.predict_proba(x_test)
                # print(np.array(y_test))
                # print(prediction)
                # print('\n')

                plt.scatter(x[:,0], x[:,1], c=color_list, s=100, label=y )
                plt.show()
                plt.clf()

                # VISUALISATION
                # --------------
