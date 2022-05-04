import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.expressions as exp
from biogeme.expressions import Beta, DefineVariable
import seaborn as sns
import biogeme.optimization as opt
import matplotlib.pyplot as plt
import biogeme.results as res
import pandas as pd
import numpy as np


import nn

def biogeme_mnl(data2016,year='2016'):
    data2016['AV']=1
    database2016 = db.Database("data_base2016", data2016)

    globals().update(database2016.variables)

    train_index, test_index = nn.train_test_index(data2016)
    data2016_train = data2016.loc[train_index]
    data2016_test = data2016.loc[test_index]

    database2016_train = db.Database("database2016_train", data2016_train)
    database2016_test = db.Database("database2016_test", data2016_test)

    A_walk = exp.Beta('A_walk', 0, None, None, 0)
    B_walk_cars = exp.Beta('B_walk_cars', 0, None, None, 0)
    B_walk_distance = exp.Beta('B_walk_distance', 0, None, None, 0)
    B_walk_licence = exp.Beta('B_walk_licence', 0, None, None, 0)
    B_wlak_time = exp.Beta('B_wlak_time', 0, None, None, 0)

    A_bike = exp.Beta('A_bike ', 0, None, None, 1)
    B_bike_bikes = exp.Beta('B_bike_bikes', 0, None, None, 0)
    B_bike_purpose1 = exp.Beta('B_bike_purpose1', 0, None, None, 0)
    B_bike_time = exp.Beta('B_bike_time ', 0, None, None, 0)

    A_car = exp.Beta('A_car', 0, None, None, 0)
    B_car_cars = exp.Beta('B_car_cars', 0, None, None, 0)
    B_car_pop = exp.Beta('B_car_pop', 0, None, None, 0)
    B_car_licence = exp.Beta('B_car_licence', 0, None, None, 0)
    B_car_time = exp.Beta('B_car_time', 0, None, None, 0)
    B_car_age = exp.Beta('B_car_age', 0, None, None, 0)
    B_car_set1 = exp.Beta('B_car_set1', 0, None, None, 0)

    A_bus = exp.Beta('A_bus', 0, None, None, 0)
    B_bus_cars = exp.Beta('B_bus_cars', 0, None, None, 0)
    B_bus_licence = exp.Beta('B_bus_licence', 0, None, None, 0)
    B_bus_pop = exp.Beta('B_bus_pop', 0, None, None, 0)
    B_bus_time = exp.Beta('B_bus_time', 0, None, None, 0)

    A_rail = exp.Beta('A_rail', 0, None, None, 0)
    B_rail_pop = exp.Beta('B_rail_pop', 0, None, None, 0)
    B_rail_time = exp.Beta('B_rail_time', 0, None, None, 0)
    B_rail_distance = exp.Beta('B_rail_distance', 0, None, None, 0)
    B_rail_purpose1 = exp.Beta('B_rail_purpose1', 0, None, None, 0)
    B_rail_edu1 = exp.Beta('B_rail_edu1', 0, None, None, 0)

    Walk = A_walk + \
           B_walk_cars * Household_car + \
           B_walk_distance * Trip_distance + \
           B_walk_licence * Household_licence + \
           B_wlak_time * Trip_time

    Bike = A_bike + \
           B_bike_bikes * Household_bike + \
           B_bike_purpose1 * Trip_purpose_1 + \
           B_bike_time * Trip_time

    Car = A_car + \
          B_car_cars * Household_car + \
          B_car_pop * Population_density + \
          B_car_licence * Household_licence + \
          B_car_time * Trip_time + \
          B_car_age * Individual_age + \
          B_car_set1 * Household_settlement_1

    Bus = A_bus + \
          B_bus_cars * Household_car + \
          B_bus_licence * Household_licence + \
          B_bus_pop * Population_density + \
          B_bus_time * Trip_time

    Rail = A_rail + \
           B_rail_pop * Population_density + \
           B_rail_time * Trip_time + \
           B_rail_distance * Trip_distance + \
           B_rail_purpose1 * Trip_purpose_1 + \
           B_rail_edu1 * Individual_education_1

    V = {1: Walk,
         2: Bike,
         3: Car,
         4: Bus,
         5: Rail}

    av = {1: AV,
          2: AV,
          3: AV,
          4:AV,
          5:AV
    }

    # Define the model
    logprob = models.loglogit(V, av, Mode)

    # Define the Biogeme object
    biogeme = bio.BIOGEME(database2016_train, logprob)

    biogeme.modelName = f"data{year}_train"

    biogeme.generateHtml = True
    biogeme.generatePickle = True

    results = biogeme.estimate()

    betas = results.getBetaValues()

    prob_walk = models.logit(V, av, 1)
    prob_bike = models.logit(V, av, 2)
    prob_car = models.logit(V,av, 3)
    prob_bus = models.logit(V, av, 4)
    prob_rail = models.logit(V, av, 5)

    simulate = {'Prob. walk': prob_walk,
                'Prob. bike': prob_bike,
                'Prob. car': prob_car,
                'Prob. bus': prob_bus,
                'Prob. rail': prob_rail}

    biogeme = bio.BIOGEME(database2016_test, simulate)
    biogeme.modelName = f"data{year}_test"

    betas = biogeme.freeBetaNames

    results = res.bioResults(pickleFile='data2016_train.pickle')
    betaValues = results.getBetaValues()

    simulatedValues = biogeme.simulate(betaValues)

    prob_max = simulatedValues.idxmax(axis=1)
    prob_max = prob_max.replace({'Prob. walk': 1, 'Prob. bike': 2, 'Prob. car': 3,'Prob. bus':4,'Prob. rail':5})

    data_result = {'y_Actual': data2016_test['Mode'],
                   'y_Predicted': prob_max
                   }

    df = pd.DataFrame(data_result, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    accuracy = np.diagonal(confusion_matrix.to_numpy()).sum() / confusion_matrix.to_numpy().sum()

    return accuracy
