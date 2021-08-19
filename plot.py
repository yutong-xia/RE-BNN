import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def get_years():
    years_str=['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
    return years_str

def get_region():
    region_name = ['North East', 'North West', 'Yorkshire and the Humber'
        , 'East Midlands', 'West Midlands', 'East of England', 'London', 'South East', 'South West']
    return region_name

def get_modes():
    travel_mode = ['Walk','Bicycle','Car or van','Bus','Rail']
    return travel_mode

years_str=get_years()
region_name =get_region()
travel_mode = get_modes()

def heatmap(data, labels,  ax, cbar_kw={}, cbarlabel="", **kwargs):
    # Plot
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # show ticks and labels
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Let the horizontal axes labeling appear on top
    ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",rotation_mode="anchor")

    # Turn spines off
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar

def annotate_heatmap(im):
    valfmt = matplotlib.ticker.StrMethodFormatter('{x:.2f}')
    textcolors=("black", "white")
    data = im.get_array()
    threshold = im.norm(data.max())/2.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None))
            texts.append(text)
    return texts


def Mode_Share_Group(results, mode_share_df,figsize=(6,10),region=False):
    group_name=mode_share_df.index

    if region==True:
        labels=['North\nEast', 'North\nWest', 'Yorkshire &\nthe Humber'
        , 'East\nMidlands', 'West\nMidlands', 'East of\nEngland', 'London', 'South\nEast', 'South\nWest']

    else: labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = ['#e6194B','#ffe119', '#4363d8', '#f58231', '#42d4f4']

    fig, ax = plt.subplots(figsize=figsize)


    ax.set_ylim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(group_name, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.bar(labels, widths, bottom=starts, width=0.5,label=colname, color=color)

    ax.legend()
    plt.ylabel('Travel Mode Shares')


def Mode_in_Region_Plot_Stand(qb_r_standard_50,num_model=50,year='2016'):
    '''
    This function is used to plot the parameters of the random effect (standardised) of each alternative in different regions
    '''
    region_name = ['North\nEast', 'North\nWest', 'Yorkshire &\nthe Humber'
        , 'East\nMidlands', 'West\nMidlands', 'East of\nEngland', 'London', 'South\nEast', 'South\nWest']
    qb_array = np.zeros((len(travel_mode), num_model, len(region_name)))
    year = year
    year_index = years_str.index(year)
    for mode in range(len(travel_mode)):
        for _ in range(num_model):
            qb_array[mode][_] = qb_r_standard_50[_][year_index].T[mode]

    fig = plt.figure(dpi=300, figsize=(12, 10))
    seq = ['a', 'b', 'c', 'd', 'e']
    for mode in range(len(travel_mode)):
        fig.add_subplot(3, 2, mode + 1)
        plt.ylim(ymax=3,ymin=-3)
        plt.xlim(xmax=9.5,xmin=0.5)
        plt.plot(range(len(region_name)+2),np.zeros((11,)),'--',alpha=0.5)
        plt.boxplot(list(qb_array[mode].T))
        plt.xticks([y + 1 for y in range(len(region_name))],
                 region_name,rotation=0, fontsize=8)
        plt.xlabel('Region')
        plt.title(f'({seq[mode]}) {travel_mode[mode]}', y=-0.35)
        plt.ylabel(f'Values of $z_r$ in {num_model} models')
    plt.tight_layout()


def Mode_in_Region_Plot_Nonstand(result_para_rebnn_50,num_model=50,year='2016',add_b=False):
    '''
    This function is used to plot the parameters of the random effect (non-standardised) of each alternative in different regions
    '''
    region_name = ['North\nEast', 'North\nWest', 'Yorkshire &\nthe Humber'
        , 'East\nMidlands', 'West\nMidlands', 'East of\nEngland', 'London', 'South\nEast', 'South\nWest']
    qb_array = np.zeros((len(travel_mode), num_model, len(region_name)))
    year = year
    for mode in range(len(travel_mode)):
        for _ in range(30):
            if add_b==True:
                qb_array[mode][_] = result_para_rebnn_50[_][year][4].T[mode] + result_para_rebnn_50[_][year][3][mode]
            else:
                qb_array[mode][_] = result_para_rebnn_50[_][year][4].T[mode]

    fig = plt.figure(dpi=300, figsize=(12, 10))
    seq = ['a', 'b', 'c', 'd', 'e']
    for mode in range(len(travel_mode)):
        fig.add_subplot(3, 2, mode + 1)
        plt.boxplot(list(qb_array[mode].T))
        plt.xlim(xmax=9.5,xmin=0.5)
        plt.xticks([y + 1 for y in range(len(region_name))],
                 region_name,rotation=0, fontsize=8)
        plt.xlabel('Region')
        plt.title(f'({seq[mode]}) {travel_mode[mode]}', y=-0.3)
        plt.ylabel(f'Values of $z_r$ in {num_model} models')
    plt.tight_layout()


def Mode_in_Region_Plot_Stand_ave(qb_r_standard_50):
    '''
    This function is used to plot the parameters of the random effect (average) in different years
    '''
    region_name = ['North\nEast', 'North\nWest', 'Yorkshire &\nthe Humber'
        , 'East\nMidlands', 'West\nMidlands', 'East of\nEngland', 'London', 'South\nEast', 'South\nWest']
    array_ave = np.zeros((len(travel_mode), len(years_str), len(region_name)))
    for year in range(len(years_str)):
        for mode in range(len(travel_mode)):
            list1 = np.mean(qb_r_standard_50, axis=0)[year].T[mode]
            array_ave[mode][year] = list1

    fig = plt.figure(dpi=300, figsize=(12, 10))
    seq = ['a', 'b', 'c', 'd', 'e']
    for mode in range(len(travel_mode)):
        fig.add_subplot(3, 2, mode + 1)
        ymax=abs(array_ave[mode]).max()+0.5
        plt.ylim(ymax=ymax,ymin=ymax*(-1))
        plt.xlim(xmax=9.5,xmin=0.5)
        plt.plot(range(len(region_name)+2),np.zeros((11,)),'--',alpha=0.5)
        plt.boxplot(array_ave[mode])
        plt.xticks([y + 1 for y in range(len(region_name))],
                 region_name,rotation=0, fontsize=8)
        plt.xlabel('Region')
        plt.title(f'({seq[mode]}) {travel_mode[mode]}', y=-0.3)
        plt.ylabel(f'Values of $z_r$ in different years')
    plt.tight_layout()
