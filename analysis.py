import numpy as np
import math
import pandas as pd
import nn
import statsmodels.api as sm
import plot


years_str = plot.get_years()
travel_mode = plot.get_modes()
region_name=plot.get_region()

def acc_region(data_year, prob_rebnn_pre,prob_rebnn_true):
    '''
    This function is used to calculate the predictive accuracy in each region
    :param data_year: a dataset for a given year
    :param prob_rebnn_pre: single training results of a given year
    :param prob_rebnn_true: true label of a given year
    :return: a list containing the predictive accuracy in different regions for a single training
    '''

    train_index, test_index = nn.train_test_index(data_year)

    # get the region column of the testing dataset
    data_year_test = data_year['Household_region'].loc[test_index].reset_index(drop=True)

    # merge the region name column and the predicted probability of modes
    data_year_test = pd.merge(data_year_test, pd.DataFrame(prob_rebnn_pre),
                             on=pd.DataFrame(prob_rebnn_pre).index)
    data_year_test.drop(columns=['key_0'], inplace=True)

    # then merge the origin df with the true mode choice
    data_year_test=pd.merge(data_year_test,pd.DataFrame(prob_rebnn_true),
                             on=pd.DataFrame(prob_rebnn_true).index)
    data_year_test.drop(columns=['key_0'], inplace=True)

    r1 = data_year_test[data_year_test.Household_region == 1].reset_index(drop=True)
    r2 = data_year_test[data_year_test.Household_region == 2].reset_index(drop=True)
    r3 = data_year_test[data_year_test.Household_region == 3].reset_index(drop=True)
    r4 = data_year_test[data_year_test.Household_region == 4].reset_index(drop=True)
    r5 = data_year_test[data_year_test.Household_region == 5].reset_index(drop=True)
    r6 = data_year_test[data_year_test.Household_region == 6].reset_index(drop=True)
    r7 = data_year_test[data_year_test.Household_region == 7].reset_index(drop=True)
    r8 = data_year_test[data_year_test.Household_region == 8].reset_index(drop=True)
    r9 = data_year_test[data_year_test.Household_region == 9].reset_index(drop=True)
    region_pre = [r1, r2, r3, r4, r5, r6, r7, r8, r9]


    acc_year = []

    x_index=['0_x','1_x','2_x','3_x','4_x']# this is the column of predictive results
    y_index = ['0_y','1_y','2_y','3_y','4_y'] # this is the column of true travel mode

    for i in region_pre:
        acc=0
        for j in range(i.shape[0]):
            # if the max probability's mode is equal to the true mode, then the 'acc' plus one
            if str(i[x_index].loc[j].sort_values(ascending=False).index[0])[0]==str(i[y_index].loc[j].sort_values(ascending=False).index[0])[0]:
                acc+=1
        acc_year.append(acc/i.shape[0])

    return acc_year




def prob_function_rebnn(xlist,year,result_para,var,mode,vars_mean,region=None,ave=False):
    ylist = []

    qw_0_value = result_para[year][0]
    qw_1_value = result_para[year][1]
    qb_0_value = np.squeeze(result_para[year][2])
    qb_1_value = np.squeeze(result_para[year][3])
    qb_r_value = result_para[year][4]

    for i in range(len(xlist)):
        x=vars_mean
        x[var]=i
        a=np.dot(x,qw_0_value)+qb_0_value
        a=(math.e**(a)-math.e**(-a))/(math.e**(a)+math.e**(-a))
        if ave==False:
            a=np.matmul(a,qw_1_value)+qb_1_value+qb_r_value[region]
        else:
            a=np.matmul(a,qw_1_value)+qb_1_value+qb_r_value.mean(axis=0)
        y=np.exp(a)/sum(np.exp(a))
        ylist.append(y[mode])
    return ylist

def prob_function_bnn(xlist,year,result_para,var,mode,vars_mean):
    ylist=[]
    qw_0_value=result_para[year][0]
    qw_1_value = result_para[year][1]
    qb_0_value = np.squeeze(result_para[year][2])
    qb_1_value = np.squeeze(result_para[year][3])
    for i in xlist:
        x=vars_mean
        x[var]=i
        a=np.dot(x,qw_0_value)+qb_0_value
        a=(math.e**(a)-math.e**(-a))/(math.e**(a)+math.e**(-a))
        a=np.dot(a,qw_1_value)+qb_1_value
        y=np.exp(a)/sum(np.exp(a))
        ylist.append(y[mode])
    return ylist

def prob_function_mnl(data,vars_MNL,xlist,var,vars_mean):
    # create a dataframe for vars_mean
    col_vars_mean=pd.DataFrame(columns=nn.get_vars())
    col_vars_mean.loc[0]=vars_mean

    #creat x_pre for each mode
    x_1=col_vars_mean[vars_MNL[0]].loc[0].values
    x_2 = col_vars_mean[vars_MNL[1]].loc[0].values
    x_3 = col_vars_mean[vars_MNL[2]].loc[0].values
    x_4 = col_vars_mean[vars_MNL[3]].loc[0].values
    x_5 = col_vars_mean[vars_MNL[4]].loc[0].values
    x_list=[x_1,x_2,x_3,x_4,x_5]

    data_1 = pd.merge(data, data.Mode.replace([2, 3, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_2 = pd.merge(data, data.Mode.replace([1, 3, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_3 = pd.merge(data, data.Mode.replace([1, 2, 4, 5], [0, 0, 0, 0]), on=data.index)
    data_4 = pd.merge(data, data.Mode.replace([1, 2, 3, 5], [0, 0, 0, 0]), on=data.index)
    data_5 = pd.merge(data, data.Mode.replace([1, 2, 3, 4], [0, 0, 0, 0]), on=data.index)
    data_mode = [data_1, data_2, data_3, data_4, data_5]

    def utility_mnl_one_mode(data_1, mode,xlist):
        train_index, test_index = nn.train_test_index(data_1)
        data_train = data_1.loc[train_index]

        X_test = np.zeros((len(xlist), len(vars_MNL[mode])))
        trip_index=vars_MNL[mode].index(nn.get_vars()[var])
        for i in range(len(xlist)):
            X_test[i] = x_list[mode]
        X_test.T[trip_index] = xlist

        model_1 = sm.OLS(exog=data_train[vars_MNL[mode]], endog=data_train.Mode_y).fit()
        utility_pred_1 = model_1.predict(X_test)
        return utility_pred_1

    utility_all_mode = np.zeros((len(data_mode), len(xlist)))
    for i in range(len(data_mode)):
        utility_all_mode[i] = utility_mnl_one_mode(data_mode[i], i,xlist)

    pre = np.exp(utility_all_mode) / np.sum(np.exp(utility_all_mode), axis=0)

    return pre.T



def dataset_year(year=2016,stand=True):
    '''
    This function use to get the sub-dataset of a given year
    :param year: The year of the dataset
    :return: a dataframe
    '''
    if stand==True:
        data=pd.read_csv('data_stand.csv')
    else:
        data=pd.read_csv('data_non_stand.csv')

    data.sort_values(by=['Mode'],inplace = True)
    data.reset_index(inplace = True)
    data.drop(columns=['level_0', 'Unnamed: 0','index'],inplace = True)

    data_year = data[data.Year == year].sort_values(by=['Mode', 'Household_region']).reset_index(drop=True)

    return data_year



def mks(prob_pre,ave=True):
    if ave==True:
        list_mks=np.zeros((len(years_str),len(travel_mode)))
        for i in range(len(years_str)):
            pre=prob_pre[years_str[i]]
            for j in range(len(travel_mode)):
                list_mks[i][j]=np.sum(pre.T[j]) / pre.shape[0]
    else:
        list_mks=[]
        for j in range(len(travel_mode)):
            list_mks.append(np.sum(prob_pre.T[j]) / prob_pre.shape[0] )
    return list_mks


def mks_region(data_year, prob_rebnn_pre,df=False):

    train_index, test_index = nn.train_test_index(data_year)

    # get the region column of the testing dataset
    data_year_test = data_year['Household_region'].loc[test_index].reset_index(drop=True)

    # merge the region name column and the predicted probability of modes
    data_year_test = pd.merge(data_year_test, pd.DataFrame(prob_rebnn_pre),
                             on=pd.DataFrame(prob_rebnn_pre).index)
    data_year_test.drop(columns=['key_0'], inplace=True)

    r1 = data_year_test[data_year_test.Household_region == 1]
    r2 = data_year_test[data_year_test.Household_region == 2]
    r3 = data_year_test[data_year_test.Household_region == 3]
    r4 = data_year_test[data_year_test.Household_region == 4]
    r5 = data_year_test[data_year_test.Household_region == 5]
    r6 = data_year_test[data_year_test.Household_region == 6]
    r7 = data_year_test[data_year_test.Household_region == 7]
    r8 = data_year_test[data_year_test.Household_region == 8]
    r9 = data_year_test[data_year_test.Household_region == 9]
    region_pre = [r1, r2, r3, r4, r5, r6, r7, r8, r9]

    pre_col = range(5)
    if df==True:
        mks_year = pd.DataFrame(index=range(len(travel_mode)))
        mks_year['Mode'] = travel_mode
        for i in range(len(region_name)):
            mks_year[region_name[i]] = mks(region_pre[i][pre_col].values, ave=False)
        return mks_year

    if df==False:
        mks_year = np.zeros((len(region_name),len(travel_mode)))
        for i in range(len(region_name)):
            mks_year[i] = mks(region_pre[i][pre_col].values, ave=False)
        return mks_year.T


def get_diff_mks(mks_rebnn_2016,mks_col):
    region_name=plot.get_region()
    mks_rebnn_2016_diff=mks_rebnn_2016.copy()
    for i in range(1,len(region_name)+1):
        mks_rebnn_2016_diff[mks_col[i]]=(mks_rebnn_2016[mks_col[i]]-mks_rebnn_2016[mks_col[i+len(region_name)]]).values
    return mks_rebnn_2016_diff.drop(columns=mks_col[10:20])


def training_result_rebnn():
    '''
    read the training results pf RE-BNN from local
    :return: a list of [predictive accuracy,
        choice probability of each testing input,
        the true choice(label) of each record in testing set,
        the parameters in each model,
        the standardised values of random effect parameter]
    '''
    accuracy_rebnn_50 = np.zeros((50, len(years_str)))
    prob_rebnn_pre_50 = dict.fromkeys(range(50))
    prob_rebnn_true_50 = dict.fromkeys(range(50))
    result_para_rebnn_50 = dict.fromkeys(range(50))
    qb_r_standard_50 = dict.fromkeys(range(50))

    for _ in range(50):
        accuracy_rebnn_50[_] = pd.read_csv(f'results\\rebnn\\train{_ + 1}\\accuracy_rebnn.csv', index_col=0).values.T
        prob_rebnn_pre_50[_] = dict.fromkeys(years_str)
        prob_rebnn_true_50[_] = dict.fromkeys(years_str)
        result_para_rebnn_50[_] = dict.fromkeys(years_str)
        qb_r_standard_50[_] = []
        for i in range(len(years_str)):
            prob_rebnn_pre_50[_][years_str[i]] = pd.read_csv(
                f'results\\rebnn\\train{_ + 1}\\prob_rebnn_pre{years_str[i]}.csv', index_col=0).values
            prob_rebnn_true_50[_][years_str[i]] = pd.read_csv(
                f'results\\rebnn\\train{_ + 1}\\prob_rebnn_true{years_str[i]}.csv', index_col=0).values
            result_para_rebnn_50[_][years_str[i]] = dict.fromkeys(range(5))
            for j in range(5):
                list2 = pd.read_csv(f'results\\rebnn\\train{_ + 1}\\result_para_rebnn{i}{j}.csv', index_col=0).values
                result_para_rebnn_50[_][years_str[i]][j] = list2
            qb_r_standard_50[_].append(
                pd.read_csv(f'results\\rebnn\\train{_ + 1}\\qb_r_standard{years_str[i]}.csv', index_col=0).values)

    return [accuracy_rebnn_50,prob_rebnn_pre_50,prob_rebnn_true_50,result_para_rebnn_50 ,qb_r_standard_50]

def training_result_bnn():
    '''
    read the training results pf BNN from local
    :return: a list of [predictive accuracy,
        choice probability of each testing input,
        the parameters in each model]
    '''
    accuracy_bnn_50 = np.zeros((50, len(years_str)))
    prob_bnn_pre_50 = dict.fromkeys(range(50))
    result_para_bnn_50 = dict.fromkeys(range(50))

    for _ in range(50):
        accuracy_bnn_50[_] = pd.read_csv(f'results\\bnn\\train{_ + 1}\\accuracy_bnn.csv', index_col=0).values.T
        prob_bnn_pre_50[_] = dict.fromkeys(years_str)
        result_para_bnn_50[_] = dict.fromkeys(years_str)
        for i in range(len(years_str)):
            prob_bnn_pre_50[_][years_str[i]] = pd.read_csv(f'results\\bnn\\train{_ + 1}\\prob_bnn_pre{years_str[i]}.csv',
                                                           index_col=0).values
            result_para_bnn_50[_][years_str[i]] = dict.fromkeys(range(5))
            for j in range(4):
                list2 = pd.read_csv(f'results\\bnn\\train{_ + 1}\\result_para_bnn{i}{j}.csv', index_col=0).values
                result_para_bnn_50[_][years_str[i]][j] = list2

    return [accuracy_bnn_50,prob_bnn_pre_50,result_para_bnn_50]

def training_result_dnn():
    '''
    read the training results pf DNN from local
    :return: a list of [predictive accuracy,
        choice probability of each testing input,
        the parameters in each model]
    '''
    accuracy_dnn_50 = np.zeros((50, len(years_str)))
    prob_dnn_pre_50 = dict.fromkeys(range(50))
    result_para_dnn_50 = dict.fromkeys(range(50))

    for _ in range(50):
        accuracy_dnn_50[_] = pd.read_csv(f'results\\dnn\\train{_ + 1}\\accuracy_dnn.csv', index_col=0).values.T
        prob_dnn_pre_50[_] = dict.fromkeys(years_str)
        result_para_dnn_50[_] = dict.fromkeys(years_str)
        for i in range(len(years_str)):
            prob_dnn_pre_50[_][years_str[i]] = pd.read_csv(f'results\\dnn\\train{_ + 1}\\prob_dnn_pre{years_str[i]}.csv',
                                                           index_col=0).values
            result_para_dnn_50[_][years_str[i]] = dict.fromkeys(range(5))
            for j in range(4):
                list2 = pd.read_csv(f'results\\dnn\\train{_ + 1}\\result_para_dnn{i}{j}.csv', index_col=0).values
                result_para_dnn_50[_][years_str[i]][j] = list2
    return [accuracy_dnn_50,prob_dnn_pre_50,result_para_dnn_50]

def training_result_mnl():

    accuracy_mnl=pd.read_csv('results\\mnl\\accuracy_mnl.csv',index_col=0).values
    prob_mnl_pre=dict.fromkeys(years_str)
    mnl_col=[['Household_car', 'Trip_distance', 'Household_licence', 'Trip_time'],
     ['Household_bike', 'Trip_purpose_1', 'Trip_time'],
     ['Household_car',
      'Population_density',
      'Household_licence',
      'Trip_time',
      'Individual_age',
      'Household_settlement_1'],
     ['Household_car', 'Household_licence', 'Population_density', 'Trip_time'],
     ['Population_density',
      'Trip_time',
      'Trip_distance',
      'Trip_purpose_1',
      'Individual_education_1']]

    for i in range(len(years_str)):
        prob_mnl_pre[years_str[i]]=pd.read_csv(f'results\\mnl\\prob_mnl_pre{years_str[i]}.csv',index_col=0).values

    return [accuracy_mnl,prob_mnl_pre,mnl_col]