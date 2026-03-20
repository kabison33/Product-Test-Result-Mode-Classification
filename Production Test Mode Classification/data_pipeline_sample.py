import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import sys
import time
import numpy as np
feature_count=0

#remove any constants in this file df is the datframe we pass through, c1 and c3 are the columsn we use to check the commonality
def remove_constants(df,c1,c3):
    #make a list of all the items in c1
    ls = df[c1].unique().tolist()
    #parse through our list
    for item in ls:
        #make a temporary dataframe that stores all values associated to that item in c1
        temp = df[df[c1] == item]
        #get the first item in the row, of our temp
        value = temp.iloc[0][c3]
        #count how many times it occurs in our c3 column
        count = temp[c3].value_counts()
        #get the count of our value it should be 1 not more
        if count[value] != 1 and len(count) == 1:
            # if not one, then we remove it from our dataframe
            df = df[df[c1] != item]
        del temp
    #return this new dataframe
    return df

# funciton to create a log file with the filename as input and data for this file
def Log_file(filename,data):
    with open(filename,"a") as f:
        f.write(data)

#check for duplicate features in this log file
def check_df_duplicates(df,name):
    #get the dataframe of duplicates features
    duplicates = df[df.duplicated()]
    #if the duplicates exist check
    if duplicates.shape[0] > 0:
        # get the list of features that are duplicates
        list_duplicates = duplicates['feature'].unique().tolist()
        return list_duplicates
    else:
        return 0

#check how many features are in a sample
def check_features_num(df):
    #feature count declared as global
    global feature_count
    #number is the counting the number of a feature for a sample
    number = df.shape[0]
    #if  there are features
    if number > 0:
        #check if the feature count has been assigned
        if feature_count == 0:
            feature_count = number
            #print the number set for this feature group for a feature number per sample
            print("\n",number)
        else:
            #if the number is different to feature group return the number and add it to the log
            if number != feature_count:
                return number
    return 0

#combine the class valus if they have the same sample id
def combine_dup(df):
    combined_df = (
    df.groupby('sample_id', as_index=False)
      .agg({
          'class': lambda x: ', '.join(map(str, sorted(set(x))))  # Unique classes joined
          ,'station': lambda x: x.iloc[0]
          ,'ita_id': lambda x: x.iloc[0]
          ,'harness_a': lambda x: x.iloc[0]
          ,'harness_b': lambda x: x.iloc[0]
          ,'event_time': lambda x: x.iloc[0]
      }))

    return combined_df


#select all the relevant files required for this test
root=tk.Tk()
root.withdraw()


#prompt user to  provide files sample files, feature group file and problem dms labels file:
sample_data = filedialog.askopenfilenames(title='Select Sample Data Files',filetypes=[("csv files","*.csv")])
feature_group_labels = filedialog.askopenfilename(title='Select Feature Group CSV',filetypes=[("csv files","*.csv")])
problem_DAs = filedialog.askopenfilename(title='Select Problem Dianostic Groups',filetypes=[("csv files","*.csv")])

#prompt user to  provide area to save these files:
folder_gen_location = filedialog.askdirectory(title="Pick where to generate these CSVs")

for sampledata in sample_data:

    #reading the csv  files
    sd = pd.read_csv(sampledata)
    fgl = pd.read_csv(feature_group_labels)
    problbl = pd.read_csv(problem_DAs)

    #taking the project and test type & create a folder with our project
    sampledatabasename = os.path.splitext(os.path.basename(sampledata))[0]
    group_file_path = folder_gen_location+"/"+sampledatabasename
    os.makedirs(group_file_path, exist_ok=True)

    #droping columns that are not needed:
    sd =  sd.drop('event_type',axis='columns')
    # problbl = problbl.drop('event_time',axis='columns')
    problbl = combine_dup(problbl)

    #getting unique  feature group names
    feature_group_names =  fgl['feature_group'].unique().tolist()

    #count how many csvs done
    load = 0

    #output the name of the project test we are running through
    print('\n', sampledatabasename,":")

    #so go through the list of group names we have  and create a csv file that contains the associated features in them.
    for group_name in feature_group_names:

        #list of our group and its associated features
        features_list =  fgl[fgl['feature_group'] == group_name]

        #group all our features and also our group names
        grouped_features_df =  features_list.merge(sd, on='feature', how='left')
        grouped_sample_ids =  grouped_features_df['sample_id'].unique().tolist()

        #remove if f# is just always a number in all sample_ids
        grouped_features_df = remove_constants(grouped_features_df,'feature','value')

        #dataframe which contains the sample_id and if it passed or failed
        sample_id_results = pd.DataFrame(columns=['sample_id','group_result'])

        #create a folder within this to hold 2 csvs
        csv_folders_path = group_file_path+"/"+group_name
        os.makedirs(csv_folders_path, exist_ok=True)

        #make  new dataframe with group result inside
        for item in grouped_sample_ids:
            #make a temp dataframe of all items that have that associated sample_id
            temp_features = grouped_features_df[(grouped_features_df["sample_id"] == item )]
            #Check if
            # -there is a duplicate of feature in a sample id
            dup_result = check_df_duplicates(temp_features,item)
            # -the set of features are the same for every sample id
            num_result = check_features_num(temp_features)
            Log_file(csv_folders_path+"/"+group_name+"_log.txt",item)
            Log_file(csv_folders_path+"/"+group_name+"_log.txt",":\n")
            # if the duplicated result is true for duplicates
            if dup_result != 0:
                #print the list of duplicate features in the log file
                Log_file(csv_folders_path+"/"+group_name+"_log.txt","Duplicates:\n")
                for dup in dup_result:
                    Log_file(csv_folders_path+"/"+group_name+"_log.txt",dup)
            if num_result != 0 :
                # if the number of features per sample id is different within a feature group then add it to the log
                Log_file(csv_folders_path+"/"+group_name+"_log.txt","\nfeatures number: "+str(num_result))
            Log_file(csv_folders_path+"/"+group_name+"_log.txt","\n")

            #take all the rows numbers that have that sample_id
            row_number = grouped_features_df.index[grouped_features_df["sample_id"] == item].tolist()

            #check no 2s are in this  sample id group
            if temp_features[temp_features["result"]==2].size != 0:

                #if 2 found then we assign 2 to the entire group and assign it to the sample_id and add it to sample_id_results
                sample_id_results.loc[len(sample_id_results)] =  [item,2]
            else:

                #if no 2 found then we assign 2 to the entire group and assign it to the sample_id and add it to sample_id_results
                sample_id_results.loc[len(sample_id_results)] =  [item,1]
            del temp_features

        #take all the class labels with their associated sample_id
        labels =  problbl[['sample_id','class', 'station', 'ita_id', 'harness_a', 'harness_b', 'event_time']]

        #merge thes sample_id_results with the labels dataframe basically gets us the columns: sample_id, group_result, class
        sample_id_results = sample_id_results.merge(labels, on='sample_id',how='left')
        sample_id_results['class'].fillna(value='NO DMS RAISED', inplace=True)

        #any values that dont have rft if they have passed, will be just passed
        sample_id_results['class'] = np.where((sample_id_results['group_result'] == 1) & (sample_id_results['class'] != 'RFT'),'PASS',sample_id_results['class'])
        #any values that have no value associaetd and it has failed, then fail with no link to the dms csv
        sample_id_results['class'] = np.where((sample_id_results['group_result'] == 2) & (sample_id_results['class'] == 'NO DMS RAISED'),'Fail - NO LINK TO DMS',sample_id_results['class'])

        #merge this sample_id_results with our group features to get the feature_group, features, values, result, group_result, class all in one table
        grouped_features_df = grouped_features_df.merge(sample_id_results,on='sample_id',how='left')

        #split for csv 1: feature_group,sample_id, feature, value, mpv
        final_features_df = grouped_features_df[['feature_group','sample_id','feature','value','mpv']]
        final_features_df.to_csv(csv_folders_path+"/"+group_name+"_samples.csv",index=False)
        print()

        # #split for csv 2: feature_group,sample_id,class
        final_samples_df = grouped_features_df[['feature_group','sample_id','class','station','ita_id','harness_a','harness_b','event_time']]
        final_samples_df_v2 = final_samples_df.drop_duplicates(subset="sample_id")
        final_samples_df_v2.to_csv(csv_folders_path+"/"+group_name+"_labels.csv",index=False)

        #shows loading screen basically to show progress of our csvs
        load=load+1
        sys.stdout.write("\rCSV Sample & Labels generated: "+str(load)+"|"+str(len(feature_group_names)))
        sys.stdout.flush()
        time.sleep(0.1)
        feature_count = 0
