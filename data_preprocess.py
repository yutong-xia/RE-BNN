import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# trip
useful_col_trip = ['TripID', 'IndividualID', 'HouseholdID', 'JD', 'MainMode_B04ID', 'TripPurpose_B04ID', 'TripTravTime',
                   'PSUID']
newname_col_trip = ['TripID', 'IndividualID', 'HouseholdID', 'Trip_distance', 'Mode', 'Trip_purpose', 'Trip_time',
                    'PSUID']
# houshold
useful_col_hous = ['HouseholdID', 'HHoldEmploy_B01ID', 'HHoldGOR_B02ID', 'HHoldNumChildren', 'NumBike', 'NumCar',
                   'NumLicHolders', 'Settlement2011EW_B03ID', ]
newname_col_hous = ['HouseholdID', 'Household_employeed', 'Household_region', 'Household_children', 'Household_bike',
                    'Household_car',
                    'Household_licence', 'Household_settlement']
# individual
useful_col_indi = ['IndividualID', 'Age_B04ID', 'EcoStat_B03ID', 'EdAttn3_B01ID', 'IndIncome2002_B02ID', 'Sex_B01ID', ]
newname_col_indi = ['IndividualID', 'Individual_age', 'Individual_employment', 'Individual_education',
                    'Individual_income', 'Individual_gender', ]

# psu
useful_col_psu = ['PSUID', 'SurveyYear']
newname_col_psu = ['PSUID', 'Year']

def data_clean(data_trip, data_hous,data_indi,data_psu,statistics=False):
    data_trip_new = data_trip[useful_col_trip].rename(columns=dict(zip(useful_col_trip, newname_col_trip)))
    data_hous_new = data_hous[useful_col_hous].rename(columns=dict(zip(useful_col_hous, newname_col_hous)))
    data_indi_new = data_indi[useful_col_indi].rename(columns=dict(zip(useful_col_indi, newname_col_indi)))
    data_psu_new = data_psu[useful_col_psu].rename(columns=dict(zip(useful_col_psu, newname_col_psu)))

    ###################################### merge the datasets   #####################################
    data_t_h = pd.merge(data_trip_new, data_hous_new, on='HouseholdID')
    data_t_h_i = pd.merge(data_t_h, data_indi_new, on='IndividualID')
    data = pd.merge(data_t_h_i, data_psu_new, on='PSUID')

    ###################################### drop unuseful columns  #####################################
    data.drop(columns=['TripID', 'PSUID'], inplace=True)

    ###################################### drop rows with null values  #####################################
    data.dropna(inplace=True)
    data = data[(data['Individual_employment'] > 0)&(data['Individual_education'] > 0)]

    ###################################### drop observations in Wales and Scotland  #####################################
    data = data[(data['Household_region'] < 10)&(data['Household_settlement'] < 3)]
    data.reset_index(inplace=True)
    #################################### combine some value of variables ###################
    # Individual_employment
    # comebine 'Retired / permanently sick(3)' and 'Other non-work(4)' into 'Non-work'
    data.Individual_employment.replace((4, 3), inplace=True)
    # Travel mode
    old_mode = range(1,14)
    new_mode = [1,2,3,3,0,0,4,4,4,5,5,3,0]
    data.Mode.replace(dict(zip(old_mode, new_mode)), inplace=True)
    data = data[data['Mode'] >0]
    # Household_employeed
    old_employeed = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    new_employeed = [2, 3, 3, 3, 4, 4, 4, 4, 4]
    data.Household_employeed.replace(dict(zip(old_employeed, new_employeed)), inplace=True)
    # Trip_purpose
    old_purpose = [5, 6, 7, 8]
    new_purpose = [7, 5, 6, 7]
    data.Trip_purpose.replace(dict(zip(old_purpose, new_purpose)), inplace=True)
    # age
    old_age = range(1, 10)
    new_age = [1, 1, 1, 2, 3, 4, 5, 6, 7]
    data.Individual_age.replace(dict(zip(old_age, new_age)), inplace=True)

    ####################################### Add the population density data####################################
    data['Population_density'] = data['Household_region']
    region_number = range(1, 10)
    pop_density = [307, 510, 352, 302, 446, 321, 5590, 476, 231]
    data.Population_density.replace(dict(zip(region_number, pop_density)), inplace=True)

    if statistics==True:
        # summary statistics of the dataset
        num_vars = ['Trip_distance', 'Trip_time', 'Household_children', 'Household_bike', 'Household_car',
                    'Household_licence', 'Population_density']
        cat_vars = ['Mode', 'Trip_purpose', 'Individual_age', 'Individual_income', 'Household_employeed',
                    'Household_region', 'Household_settlement', 'Individual_employment', 'Individual_education',
                    'Individual_gender', 'Year']
        data[num_vars].describe().to_csv('summary statistics\\num.csv')
        for i in range(len(cat_vars)):
            data[cat_vars[i]].value_counts().to_csv(f'summary statistics\\cat_{cat_vars[i]}.csv')
        print('Summary statistics of variables have been saved in summary statistics folder.')
    ####################################### encode some categorical variable to numerical values #################
    # age
    cat_age = range(1, 8)
    num_age = [14, 19, 25, 35, 45, 55, 62]
    data.Individual_age.replace(dict(zip(cat_age, num_age)), inplace=True)
    # income
    cat_income = [1, 2, 3]
    num_income = [12500, 37500, 62500]
    data.Individual_income.replace(dict(zip(cat_income, num_income)), inplace=True)

    ###################################### standardisation ###########################################
    
    standard_vars = ['Trip_distance', 'Trip_time', 'Household_employeed',
                    'Household_children', 'Household_bike', 'Household_car',
                    'Household_licence', 'Individual_age', 'Individual_income',
                    'Population_density']
    non_standard_vars = ['index', 'IndividualID', 'HouseholdID', 'Mode', 'Trip_purpose', 'Household_region',
                        'Household_settlement', 'Individual_employment', 'Individual_education', 'Individual_gender',
                        'Year']

    df_stand_part = pd.DataFrame(StandardScaler().fit_transform(data[standard_vars]), columns=standard_vars,
                                index=data.index)
    data = pd.concat([data[non_standard_vars], df_stand_part], axis=1)

    ###################################### get dummies of other catagorical variables  #####################################
    dummy_variable = ['Trip_purpose', 'Household_settlement', 'Individual_employment', 'Individual_education',
                      'Individual_gender']
    for i in range(len(dummy_variable)):
        data = pd.merge(data, pd.get_dummies(data[dummy_variable[i]], prefix=dummy_variable[i]), on=data.index)
        data.drop(columns=['key_0', dummy_variable[i]], inplace=True)

    # ignore reference level
    ref_level = ['Trip_purpose_7', 'Household_settlement_2', 'Individual_employment_2', 'Individual_education_2',
                 'Individual_gender_2']
    data.drop(columns=ref_level, inplace=True)

    if standard==True:
        data.to_csv('data_stand.csv')
    else:
        data.to_csv('data_non_stand.csv')
    print(f'Done. The sample size is: {len(data)}')


