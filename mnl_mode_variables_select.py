import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def get_vars():
    fixed_effect = ['Trip_distance', 'Trip_time', 'Household_employeed', 'Household_children',
                    'Household_bike', 'Household_car', 'Household_licence', 'Household_settlement_1',
                    'Individual_employment_1', 'Individual_education_1',
                    'Individual_gender_1'] + purpose_columns + Individual_age_col + Individual_income_col
    return fixed_effect


def get_vars_MNL(data, var=1):

    data_1 = pd.merge(data, data.Mode.replace([2, 3, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_2 = pd.merge(data, data.Mode.replace([1, 3, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_3 = pd.merge(data, data.Mode.replace([1, 2, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_4 = pd.merge(data, data.Mode.replace([1, 2, 3, 5], [0, 0, 0, 0]), on=data.index)
    data_5 = pd.merge(data, data.Mode.replace([1, 2, 3, 4], [0, 0, 0, 0]), on=data.index)
    data_mode = [data_1, data_2, data_3, data_4, data_5]

    mode_name = get_modes()

    vars_MNL = []
    corr = np.zeros((len(mode_name), len(get_vars())))
    for i in range(len(data_mode)):
        data_corr = data_mode[i]
        corr[i] = data_corr.drop(columns=['Mode_x', 'key_0']).corr()['Mode_y'].reindex(get_vars(),
                                                                                       axis='rows').values
        corr_col = data_corr.drop(columns=['Mode_x', 'key_0']).corr()['Mode_y'].abs().sort_values(
            ascending=False).head(10)
        print(f'Columns correlated with the alternatives {mode_name[i]}:')
        print(corr_col)
        data_1_col = []
        for j in range(1, 10):
            if corr_col[corr_col.index[j]] > 0.1: data_1_col.append(corr_col.index[j])
        if len(data_1_col) < 2:
            data_1_col = list(corr_col.index[1:3])
        if get_vars()[var] not in data_1_col:
            data_1_col.append(get_vars()[var])
        print(f'Columns selected in the utility function of {mode_name[i]}:')
        print(data_1_col)
        print(' ')
        vars_MNL.append(data_1_col)

    return vars_MNL, corr.T


def get_modes():
    travel_mode = ['Walk', 'Bicycle', 'Car or van', 'Bus', 'Rail']
    return travel_mode


if __name__ == '__main__':
    path = os.getcwd()
    if not os.path.exists(path + '\\image'): os.makedirs(path + '\\image')
    if not os.path.exists(path + '\\results'): os.makedirs(path + '\\results')
    if not os.path.exists(path + '\\results\\mnl'): os.makedirs(path + '\\results\\mnl')

    # Read the preprocessed dataset
    data_stand = pd.read_csv('data/mnl_data_non_stand.csv')
    data_stand = data_stand.loc[data_stand['Household_region'] == 7]
    data_stand.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'index', 'IndividualID', 'HouseholdID',
                             'Household_region', 'Population_density', 'Trip_purpose_1', 'Trip_purpose_2',
                             'Trip_purpose_3', 'Trip_purpose_4', 'Trip_purpose_5', 'Trip_purpose_6', 'Trip_purpose_7'],
                    inplace=True)
    purpose_columns = ['Commuting', 'Business', 'Education/escort education', 'Shopping', 'Personal business',
                       'Leisure', 'Other_purposes']
    # purpose --> one-hot
    Purpose = pd.get_dummies(data_stand['purpose'])
    Purpose.columns = purpose_columns
    for item in Purpose:
        data_stand[item] = Purpose[item]

    # Individual_age --> one-hot
    Individual_age_col = ['Individual_age (0-16)', 'Individual_age (17-20)', 'Individual_age (21-29)',
                          'Individual_age (30-39)', 'Individual_age (40-49)', 'Individual_age (50-59)',
                          'Individual_age (60+)']

    Individual_age = pd.get_dummies(data_stand['Individual_age'])
    Individual_age.columns = Individual_age_col
    for item in Individual_age:
        data_stand[item] = Individual_age[item]

    # Individual_income --> one-hot
    Individual_income_col = [r'Individual_income (Less than £25,000)', 'Individual_income (£25,000 to £49,999)',
                             'Individual_income (£50,000 and over)']

    Individual_income = pd.get_dummies(data_stand['Individual_income'])
    Individual_income.columns = Individual_income_col
    for item in Individual_income:
        data_stand[item] = Individual_income[item]

    # Standardization
    standard_vars = ['Trip_distance', 'Trip_time', 'Household_employeed', 'Household_children', 'Household_bike',
                     'Household_car', 'Household_licence']
    non_standard_vars = ['Household_settlement_1', 'Individual_employment_1', 'Individual_education_1',
                         'Individual_gender_1', 'Year',
                         'Mode'] + purpose_columns + Individual_age_col + Individual_income_col

    df_stand_part = pd.DataFrame(StandardScaler().fit_transform(data_stand[standard_vars]), columns=standard_vars,
                                 index=data_stand.index)
    data_stand = pd.concat([data_stand[non_standard_vars], df_stand_part], axis=1)

    data_stand.sort_values(by=['Mode'], inplace=True)
    data_stand.reset_index(inplace=True)
    data_stand.drop(columns=['index'], inplace=True)
    data = data_stand
    corr_choice_attribute = pd.DataFrame(columns=get_modes(), index=get_vars())
    data_test_mnl = data.sample(100000, random_state=42).reset_index()
    data_test_mnl.drop(columns=['index', 'Year'], inplace=True)
    # print("data_test_mnl:\n", data_test_mnl)
    # print(data_test_mnl.columns)

    mnl_col, corr_mnl = get_vars_MNL(data_test_mnl)
    print("mnl_col:\n", mnl_col)

    """
    mnl = [
    ['Purpose(Other)', 'Trip_distance', 'Purpose(Commuting)', 'Household_car', 'Trip_time'], 

    ['Household_bike', 'Individual_gender_1', 'Trip_time'], 

    ['Household_car', 'Household_licence', 'Trip_time', 'Purpose(Commuting)', 'Individual_age_(21-29)',
     'Household_bike', 'Purpose(Leisure)'],

    ['Household_car', 'Household_licence', 'Individual_income_(Less_than_25000)', 'Household_bike', 
    'Individual_age_(17-20)', 'Purpose(Other)', 'Trip_time'], 

    ['Purpose(Commuting)', 'Trip_time', 'Trip_distance', 'Purpose(Shopping)', 'Household_car', 'Individual_education_1',
     'Individual_age_(21-29)', 'Purpose(Other)', 'Individual_income_(Less_than_25000)']]
     
    """
    # save to csv
    for i in range(len(get_modes())):
        corr_choice_attribute[get_modes()[i]] = corr_mnl.T[i]
    # print(corr_choice_attribute)
    corr_choice_attribute.to_csv('results\\mode_corr_choice_attribute.csv',encoding='windows-1252')