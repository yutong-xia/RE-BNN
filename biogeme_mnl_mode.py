import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.expressions as exp
import biogeme.results as res
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.enable_eager_execution()


def train_test_index(total_data):
    train_idx = total_data.sample(frac=0.7, random_state=42).index
    test_idx = total_data.drop(train_idx).index
    return train_idx, test_idx


def cat_accuracy(actual, pre):
    right = 0
    for j in range(len(actual)):
        if list(actual[j]).index(1) == list(pre[j]).index(1):
            right += 1
    return right / len(actual)


"""
    ----------------------------------------------------------------------------------------------------------------
                See code "mnl_mode_variables_select.py" for the choice of independent variables
    ----------------------------------------------------------------------------------------------------------------
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


original_data = pd.read_csv('data/mnl_data_non_stand.csv')
LD_data = original_data.loc[original_data['Household_region'] == 7]
LD_data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'index', 'IndividualID', 'HouseholdID',
                      'Household_region', 'Population_density', 'Trip_purpose_1', 'Trip_purpose_2',
                      'Trip_purpose_3', 'Trip_purpose_4', 'Trip_purpose_5', 'Trip_purpose_6', 'Trip_purpose_7'],
             inplace=True)
purpose_columns = ['Purpose_Commuting', 'Purpose_Business', 'Purpose_Education_escort_education', 'Purpose_Shopping',
                   'Purpose_Personal_business', 'Purpose_Leisure', 'Purpose_Other']
# purpose --> one-hot
Purpose = pd.get_dummies(LD_data['purpose'])
Purpose.columns = purpose_columns
for item in Purpose:
    LD_data[item] = Purpose[item]
# Individual_age --> one-hot
Individual_age_col = ['Individual_age_0_16', 'Individual_age_17_20', 'Individual_age_21_29',
                      'Individual_age_30_39', 'Individual_age_40_49', 'Individual_age_50_59',
                      'Individual_age_60+']
Individual_age = pd.get_dummies(LD_data['Individual_age'])
Individual_age.columns = Individual_age_col
for item in Individual_age:
    LD_data[item] = Individual_age[item]
# Individual_income --> one-hot
Individual_income_col = [r'Individual_income_Less_than_25000', 'Individual_income_25000_to_49999',
                         'Individual_income_50000_and_over']
Individual_income = pd.get_dummies(LD_data['Individual_income'])
Individual_income.columns = Individual_income_col
for item in Individual_income:
    LD_data[item] = Individual_income[item]
# Standardization
standard_vars = ['Trip_distance', 'Trip_time', 'Household_employeed', 'Household_children', 'Household_bike',
                 'Household_car', 'Household_licence']
non_standard_vars = ['Household_settlement_1', 'Individual_employment_1', 'Individual_education_1',
                     'Individual_gender_1', 'Year', 'Mode'] + \
                    purpose_columns + Individual_age_col + Individual_income_col
df_stand_part = pd.DataFrame(StandardScaler().fit_transform(LD_data[standard_vars]), columns=standard_vars,
                             index=LD_data.index)
LD_data_stand = pd.concat([LD_data[non_standard_vars], df_stand_part], axis=1)
LD_data_stand.sort_values(by=['Mode'], inplace=True)
LD_data_stand.reset_index(inplace=True)
LD_data_stand.drop(columns=['index'], inplace=True)
result_save = []
stop_year = 2017
for year in range(2005, stop_year):
    data = LD_data_stand[LD_data_stand.Year == year].sort_values(by=['Mode']).reset_index(drop=True)
    data['AV'] = 1
    database = db.Database("data_base", data)
    globals().update(database.variables)
    train_index, test_index = train_test_index(data)
    data_train = data.loc[train_index]
    data_test = data.loc[test_index]
    # mode label --> one-hot
    mode_train_true = pd.get_dummies(data_train['Mode']).astype('int')
    mode_test_true = pd.get_dummies(data_test['Mode']).astype('int')

    database_train = db.Database("database_train", data_train)
    database_test = db.Database("database_test", data_test)

    '''--- WALK ---'''
    A_walk = exp.Beta('A_walk', 0, None, None, 0)

    B_walk_purpose_other = exp.Beta('B_walk_purpose_other', 0, None, None, 0)                   # 'Purpose_Other'
    B_walk__trip_distance = exp.Beta('B_walk__trip_distance', 0, None, None, 0)                 # 'Trip_distance'
    B_walk_purpose_commuting = exp.Beta('B_walk_purpose_commuting', 0, None, None, 0)           # 'Purpose_Commuting'
    B_walk_household_car = exp.Beta('B_walk_household_car', 0, None, None, 0)                   # 'Household_car'
    B_walk_trip_time = exp.Beta('B_walk_trip_time', 0, None, None, 0)                           # 'Trip_time'

    '''--- BIKE ---'''
    A_bike = exp.Beta('A_bike ', 0, None, None, 0)

    B_bike_household_bike = exp.Beta('B_bike_household_bike', 0, None, None, 0)                 # 'Household_bike'
    B_bike_individual_gender = exp.Beta('B_bike_individual_gender', 0, None, None, 0)           # 'Individual_gender_1'
    B_bike_trip_time = exp.Beta('B_bike_trip_time ', 0, None, None, 0)                          # 'Trip_time'

    '''--- CAR ---'''
    A_car = exp.Beta('A_car', 0, None, None, 0)

    B_car_household_car = exp.Beta('B_car_household_car', 0, None, None, 0)                     # 'Household_car'
    B_car_household_licence = exp.Beta('B_car_household_licence', 0, None, None, 0)             # 'Household_licence'
    B_car_trip_time = exp.Beta('B_car_trip_time', 0, None, None, 0)                             # 'Trip_time'
    B_car_purpose_commuting = exp.Beta('B_car_purpose_commuting', 0, None, None, 0)             # 'Purpose_Commuting'
    B_car_individual_age_21_29 = exp.Beta('B_car_individual_age_21_29', 0, None, None, 0)       # 'Individual_age_21_29'
    B_car_household_bike = exp.Beta('B_car_household_bike', 0, None, None, 0)                   # 'Household_bike'
    B_car_purpose_leisure = exp.Beta('B_car_purpose_leisure', 0, None, None, 0)                 # 'Purpose_Leisure'

    '''--- BUS ---'''
    A_bus = exp.Beta('A_bus', 0, None, None, 0)

    B_bus_household_car = exp.Beta('B_bus_household_car', 0, None, None, 0)                     # 'Household_car'
    B_bus_household_licence = exp.Beta('B_bus_household_licence', 0, None, None, 0)             # 'Household_licence'
    B_bus_individual_income_Less_than_25000 = exp.Beta('B_bus_individual_income_Less_than_25000', 0, None, None, 0)     # 'Individual_income_Less_than_25000'
    B_bus_household_bike = exp.Beta('B_bus_household_bike', 0, None, None, 0)                   # 'Household_bike'
    B_bus_individual_age_17_20 = exp.Beta('B_bus_individual_age_17_20', 0, None, None, 0)       # 'Individual_age_17_20'
    B_bus_trip_time = exp.Beta('B_bus_trip_time', 0, None, None, 0)                             # 'Trip_time'
    B_bus_purpose_other = exp.Beta('B_bus_purpose_other', 0, None, None, 0)                     # 'Purpose_Other'

    '''--- RAIL ---'''
    A_rail = exp.Beta('A_rail', 0, None, None, 0)

    B_rail_purpose_commuting = exp.Beta('B_rail_purpose_commuting', 0, None, None, 0)           # 'Purpose_Commuting'
    B_rail_trip_time = exp.Beta('B_rail_trip_time', 0, None, None, 0)                           # 'Trip_time'
    B_rail_trip_distance = exp.Beta('B_rail_trip_distance', 0, None, None, 0)                   # 'Trip_distance'
    B_rail_purpose_shopping = exp.Beta('B_rail_purpose_shopping', 0, None, None, 0)             # 'Purpose_Shopping'
    B_rail_household_car = exp.Beta('B_rail_household_car', 0, None, None, 0)                   # 'Household_car'
    B_rail_individual_education = exp.Beta('B_rail_individual_education', 0, None, None, 0)   # 'Individual_education_1'
    B_rail_individual_age_21_29 = exp.Beta('B_rail_individual_age_21_29', 0, None, None, 0)     # 'Individual_age_21_29'
    B_rail_purpose_other = exp.Beta('B_rail_purpose_other', 0, None, None, 0)                   # 'Purpose_Other'
    B_rail_individual_income_Less_than_25000 = exp.Beta('B_rail_individual_income_Less_than_25000', 0, None, None, 0)   # 'Individual_income_Less_than_25000'

    Walk = A_walk + \
        B_walk_purpose_other * Purpose_Other + \
        B_walk_trip_distance * Trip_distance + \
        B_walk_purpose_commuting * Purpose_Commuting + \
        B_walk_household_car * Household_car + \
        B_walk_trip_time * Trip_time

    Bike = A_bike + \
        B_bike_household_bike * Household_bike + \
        B_bike_individual_gender * Individual_gender_1 + \
        B_bike_trip_time * Trip_time

    Car = A_car + \
        B_car_household_car * Household_car + \
        B_car_household_licence * Household_licence + \
        B_car_trip_time * Trip_time + \
        B_car_purpose_commuting * Purpose_Commuting + \
        B_car_individual_age_21_29 * Individual_age_21_29 + \
        B_car_household_bike * Household_bike + \
        B_car_purpose_leisure * Purpose_Leisure

    Bus = A_bus + \
        B_bus_household_car * Household_car + \
        B_bus_household_licence * Household_licence + \
        B_bus_individual_income_Less_than_25000 * Individual_income_Less_than_25000 + \
        B_bus_household_bike * Household_bike + \
        B_bus_individual_age_17_20 * Individual_age_17_20 + \
        B_bus_purpose_other * Purpose_Other + \
        B_bus_trip_time * Trip_time

    Rail = A_rail + \
        B_rail_purpose_commuting * Purpose_Commuting + \
        B_rail_trip_time * Trip_time + \
        B_rail_trip_distance * Trip_distance + \
        B_rail_purpose_shopping * Purpose_Shopping + \
        B_rail_household_car * Household_car + \
        B_rail_individual_education * Individual_education_1 + \
        B_rail_individual_age_21_29 * Individual_age_21_29 + \
        B_rail_purpose_other * Purpose_Other + \
        B_rail_individual_income_Less_than_25000 * Individual_income_Less_than_25000

    V = {1: Walk, 2: Bike, 3: Car, 4: Bus, 5: Rail}
    av = {1: AV, 2: AV, 3: AV, 4: AV, 5: AV}
    # Define the model
    log_prob = models.loglogit(V, av, Mode)
    # Define the Biogeme object
    biogeme = bio.BIOGEME(database_train, log_prob)
    biogeme.modelName = "Mode_Train_" + str(year)
    biogeme.generateHtml = True
    biogeme.generatePickle = True
    results = biogeme.estimate()
    print(f"HTML file:    {results.data.htmlFileName}")
    print(f"Pickle file:  {results.data.pickleFileName}")

    # betas = results.getBetaValues()
    # for k, v in betas.items():
    #     print(f"{k:10}=\t{v:.3g}")

    prob_walk = models.logit(V, av, 1)
    prob_bike = models.logit(V, av, 2)
    prob_car = models.logit(V, av, 3)
    prob_bus = models.logit(V, av, 4)
    prob_rail = models.logit(V, av, 5)

    simulate = {'Prob. walk': prob_walk,
                'Prob. bike': prob_bike,
                'Prob. car': prob_car,
                'Prob. bus': prob_bus,
                'Prob. rail': prob_rail}
    save_column = []
    ''' =================================================== test =================================================== '''
    biogeme_test = bio.BIOGEME(database_test, simulate)
    biogeme_test.modelName = "mode_test_" + str(year)

    betas = biogeme_test.freeBetaNames

    # print('Extracting the following variables:')
    # for k in betas:
    #     print('\t', k)

    results = res.bioResults(pickleFile="Mode_Train_" + str(year) + ".pickle")
    betaValues = results.getBetaValues()
    # print("betaValues:\n", betaValues)

    simulatedValues = biogeme_test.simulate(betaValues)
    # print(simulatedValues.head())

    # calculate ce
    test_prob = np.array(simulatedValues)
    y_test = np.zeros_like(test_prob)
    y_test[np.arange(len(test_prob)), test_prob.argmax(1)] = 1
    cce = tf.keras.losses.CategoricalCrossentropy()
    test_ce = cce(np.array(mode_test_true), test_prob).numpy()

    # confusion_matrix
    prob_max = simulatedValues.idxmax(axis=1)
    prob_max = prob_max.replace({'Prob. walk': 1, 'Prob. bike': 2, 'Prob. car': 3, 'Prob. bus': 4, 'Prob. rail': 5})
    # print("prob_max: \n", prob_max)
    data_result = {'y_Actual': data_test['Mode'],
                   'y_Predicted': prob_max
                   }
    df = pd.DataFrame(data_result, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'],
                                   dropna=False)
    # confirm dimension is 5
    if len(list(confusion_matrix.columns)) < 5:
        for i in range(1, 6):
            if i not in list(confusion_matrix.columns):
                confusion_matrix.insert(i - 1, i, 0)
    print(confusion_matrix)
    # calculate acc
    test_ac = np.diagonal(confusion_matrix.to_numpy()).sum() / confusion_matrix.to_numpy().sum()
    print(year, "Test_CE: ", test_ce)
    print(year, 'Test_ACC:', test_ac)
    save_column.append(test_ce)
    save_column.append(test_ac)
    ''' ================================================== train ================================================== '''
    # select the database
    biogeme_train = bio.BIOGEME(database_train, simulate)
    biogeme_train.modelName = "mode_train_" + str(year)
    # reload the model
    results = res.bioResults(pickleFile="Mode_Train_" + str(year) + ".pickle")
    # simulate
    betaValues = results.getBetaValues()
    simulatedValues = biogeme_train.simulate(betaValues)
    # calculate ce
    train_prob = np.array(simulatedValues)
    y_train = np.zeros_like(train_prob)
    y_train[np.arange(len(train_prob)), train_prob.argmax(1)] = 1
    cce = tf.keras.losses.CategoricalCrossentropy()
    train_ce = cce(np.array(mode_train_true), train_prob).numpy()
    # calculate acc
    Y = pd.get_dummies(data_train['Mode']).astype("int")
    train_ac = cat_accuracy(np.array(mode_train_true), y_train)

    print(year, "Train_CE: ", train_ce)
    print(year, 'Train_ACC:', train_ac)
    save_column.append(train_ce)
    save_column.append(train_ac)
    result_save.append(save_column)

idx_list = [i for i in range(2005, stop_year)]
result_save = pd.DataFrame(result_save, columns=["test_ce", "test_acc", "train_ce", "train_acc"], index=idx_list)
print(result_save)
result_save.to_csv("biogeme_mnl_mode_results.csv")
