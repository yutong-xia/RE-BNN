import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.expressions as exp
import biogeme.results as res
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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

LD_data['Trip_time_non_stand'] = LD_data['Trip_time']           # plot as X axis (1)
LD_data['Trip_distance_non_stand'] = LD_data['Trip_distance']   # plot as X axis (2)

# Standardization
standard_vars = ['Trip_distance', 'Trip_time', 'Household_employeed', 'Household_children', 'Household_bike',
                 'Household_car', 'Household_licence']
non_standard_vars = ['Household_settlement_1', 'Individual_employment_1', 'Individual_education_1',
                     'Individual_gender_1', 'Year', 'Trip_time_non_stand', 'Trip_distance_non_stand',
                     'Mode'] + purpose_columns + Individual_age_col + Individual_income_col
df_stand_part = pd.DataFrame(StandardScaler().fit_transform(LD_data[standard_vars]), columns=standard_vars,
                             index=LD_data.index)
LD_data_stand = pd.concat([LD_data[non_standard_vars], df_stand_part], axis=1)

for year in range(2005,2006):
    data = LD_data_stand[LD_data_stand.Year == year].sort_values(by=['Mode']).reset_index(drop=True)
    data['AV'] = 1
    database = db.Database("data_base", data)
    globals().update(database.variables)
    Trip_time_non_stand = data['Trip_time_non_stand']
    Trip_distance_non_stand = data['Trip_distance_non_stand']

    data_train = data
    data_train.drop(columns=['Trip_time_non_stand', 'Trip_distance_non_stand'], inplace=True)
    database_train = db.Database("database_train", data_train)

    data_ = np.array(data_train)
    data_avg = np.mean(data_, axis=0)

    # plot time as X axis
    data_feed = np.repeat(data_avg, len(data_)).reshape(np.size(data_, axis=1), len(data_)).T
    idx_time = list(data_train.columns).index('Trip_time')
    data_feed[:, idx_time] = data_[:, idx_time]
    data_plot_time = pd.DataFrame(data_feed, columns=data_train.columns)
    database_plot_time = db.Database("database_plot_time", data_plot_time)

    # plot distance as X axis
    data_feed = np.repeat(data_avg, len(data_)).reshape(np.size(data_, axis=1), len(data_)).T
    idx_distance = list(data_train.columns).index('Trip_distance')
    data_feed[:, idx_distance] = data_[:, idx_distance]
    data_plot_distance = pd.DataFrame(data_feed, columns=data_train.columns)
    database_plot_distance = db.Database("database_plot_distance", data_plot_distance)

    """ See code "mnl_mode_variables_select.py" for the choice of independent variables """

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

    # save mode
    log_prob = models.loglogit(V, av, Mode)
    biogeme = bio.BIOGEME(database_train, log_prob)
    biogeme.modelName = "Mode_plot_" + str(year)
    biogeme.generateHtml = True
    biogeme.generatePickle = True
    results = biogeme.estimate()
    print(f"HTML file:    {results.data.htmlFileName}")
    print(f"Pickle file:  {results.data.pickleFileName}")

    # load model
    results = res.bioResults(pickleFile="Mode_plot_" + str(year) + ".pickle")
    betaValues = results.getBetaValues()

    # plot
    colors = ['red', 'darkorange', 'darkgreen', 'darkorchid', 'blue']
    mode = ["Walk", "Bike", "Car or Van", "Bus", "Rail"]

    # plot Trip time
    time_plot = bio.BIOGEME(database_plot_time, simulate)
    simulatedValues = time_plot.simulate(betaValues)
    time_prob = np.array(simulatedValues)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams.update({'font.size': 16})
    for j in range(5):
        plot = sorted(zip(Trip_time_non_stand, time_prob[:, j]))
        ax.plot([x[0] for x in plot], [x[1] for x in plot], linewidth=3, color=colors[j], label=mode[j])
    ax.legend(loc='upper right')
    # ax.legend(bbox_to_anchor=(0.9, 0.3))
    ax.set_ylabel("choice probability", fontsize=16)
    ax.set_xlabel("Trip time (minutes)", fontsize=16)
    ax.set_xlim([0, np.percentile(Trip_time_non_stand, 99)])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xticklabels([0, 20, 40, 60, 80, 100, 120], fontsize=16)
    ax.set_yticklabels([0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
    plt.savefig(str(year) + "_MNL_mode_choice_probability_vs_Trip_time.png")

    # plot Trip distance
    distance_plot = bio.BIOGEME(database_plot_distance, simulate)
    simulatedValues = distance_plot.simulate(betaValues)
    distance_prob = np.array(simulatedValues)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams.update({'font.size': 16})
    for j in range(5):
        plot = sorted(zip(Trip_distance_non_stand, distance_prob[:, j]))
        ax.plot([x[0] for x in plot], [x[1] for x in plot], linewidth=3, color=colors[j], label=mode[j])
    ax.legend(loc='center right')
    ax.set_ylabel("choice probability", fontsize=16)
    ax.set_xlabel("Trip distance (miles)", fontsize=16)
    ax.set_xlim([0, np.percentile(Trip_distance_non_stand, 99)])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70], fontsize=16)
    ax.set_yticklabels([0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)

    plt.savefig(str(year) + "_MNL_mode_choice_probability_vs_Trip_distance.png")
