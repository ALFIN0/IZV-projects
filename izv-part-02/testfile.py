#%%%
%matplotlib inline
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile
import datetime


#%%%
def load_data(filename : str) -> pd.DataFrame:
    """Load .csv data from zipfile.

    Keyword arguments:
    filename (str) -- name of .zip file

    Returns:
        (pd.DataFrame) loaded dataframe
    """

    # .csv file and result dataframe headers
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
                "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
                "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
                "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]

    def get_dataframe(filename: str, zipF: zipfile) -> pd.DataFrame:
        """Get dataframe from .csv file (filename) in zipfile (zipF)

        Args:
        filename (str) -- file name of .csv file
        zipF (zipfile) -- zipfile

        Returns:
            pd.DataFrame: dataframe of .csv file
        """

        # used regions
        regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

        innerZipfile = zipfile.ZipFile(zipF.open(filename, 'r'))

        outputDF = None

        # going though regions dictionary with .csv file names
        for k,v in regions.items():
            csvFileName = v + '.csv'
            
            if csvFileName not in innerZipfile.namelist():
                continue
            with innerZipfile.open(v+'.csv') as csvFile:
                if outputDF is None:
                    outputDF = pd.read_csv(csvFile, delimiter=';', encoding='cp1250', header=None, names=headers, low_memory=False)
                    outputDF['region'] = k
                else:
                    tmpDF = pd.read_csv(csvFile, delimiter=';', encoding='cp1250', header=None, names=headers, low_memory=False)
                    tmpDF['region'] = k
                    outputDF = pd.concat([outputDF, tmpDF], ignore_index=True)

        return outputDF

    
    outputDF = None
    zipF = zipfile.ZipFile(filename, 'r')

    # going though .zip files in outer .zip file
    for fileN in zipF.filelist:   
        if fileN.filename[-4:] != '.zip':
            continue
        if outputDF is None:
            outputDF = get_dataframe(fileN, zipF)
        else:
            outputDF = pd.concat([outputDF, get_dataframe(fileN, zipF)], ignore_index=True)


    return outputDF

df = load_data("data.zip")


#%%%

# Ukol 2: zpracovani dat
def parse_data(df : pd.DataFrame, verbose : bool = False) -> pd.DataFrame:
    """Parse dataframe and shrink data size.

    Args:
    df (pd.DataFrame) -- input dataframe
    verbose (bool, optional) -- display original and new dataframe size

    Returns:
        (pd.DataFrame) output dataframe
    """

    dataFrame = df.copy()
    dataFrame.rename(columns={'p2a':'date'}, inplace=True)
    dataFrame['date'] = pd.to_datetime(dataFrame['date'])

    # change type of columns with object type
    for column in dataFrame.select_dtypes([np.object0]):
        if column == 'region':
            continue
        # change type of columns to float
        elif column == 'd' or column == 'e':
            tmpArr = []
            for number in dataFrame[column]:
                try:
                    newNumber = float(str(number).replace(',','.'))
                except:
                    newNumber = 0.0
                tmpArr.append(newNumber)
            
            dataFrame[column] = tmpArr
            dataFrame[column] = dataFrame[column].astype(np.float64)
        # change type of column to category
        else:
            dataFrame[column] = dataFrame[column].astype('category')

    # drop duplicates rows
    dataFrame.drop_duplicates(subset=['p1'])

    # print original and new size of dataframe
    if verbose:
        print("orig_size=%5.2f MB" % (df.memory_usage(deep=True).sum() / 1024 ** 2))
        print("new_size=%5.2f MB" % (dataFrame.memory_usage(deep=True).sum() / 1024 ** 2))

    return dataFrame

df2 = parse_data(df, True)
print(df2.info())


# p19	VIDITELNOST	
# 1	ve dne	viditelnost nezhoršená vlivem povětrnostních podmínek
# 2	ve dne	zhoršená viditelnost (svítání, soumrak)
# 3	ve dne	zhoršená viditelnost vlivem povětrnostních podmínek (mlha, sněžení, déšť apod.)
# 4	v noci	s veřejným osvětlením, viditelnost nezhoršená vlivem povětrnostních podmínek
# 5	v noci	s veřejným osvětlením, zhoršená viditelnost vlivem povětrnostních podmínek (mlha, déšť, sněžení apod.)
# 6	v noci	bez veřejného osvětlení, viditelnost nezhoršená vlivem povětrnostních podmínek
# 7	v noci	bez veřejného osvětlení, viditelnost zhoršená vlivem povětrnostních podmínek (mlha, déšť, sněžení apod.)


#%%%
def plot_visibility(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):

    """Plot a graph of the number of accidents in 4 regions based on visibility.

    Args:
    df (pd.DataFrame) -- input dataframe
    fig_location (str, optional) -- output file location (default None)
    show_figure (bool, optional) -- show figure to output (default False)
    """

    # visibility values dictionary
    visibilityDict = {1: 'VISDAY', 2: 'NOTDAY', 3: 'NOTDAY', 4: 'VISNI', 5: 'NOTNI', 6: 'VISNI', 7: 'NOTNI'}
    
    newDF = df.copy()
    # map new visibility values 
    newDF['visibility'] = newDF['p19'].map(visibilityDict)
    # drop unused rows in dataframe
    newDF.drop(newDF.index[(newDF['region'] != 'HKK') 
        & (newDF['region'] != 'LBK') 
        & (newDF['region'] != 'PLK') 
        & (newDF['region'] != 'STC')], inplace = True)

    # group by dataframe rows based on region and pick only visibility column
    a = newDF.groupby('region')['visibility']
    # get count of unique values combinations of region-visibility
    b = a.value_counts().rename_axis(['region', 'visibility']).reset_index(name='counts')

    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    
    ax = axes.flatten()
    graphs = [
        {'ax': ax[0], 'df': b[b.visibility == 'VISNI'], 'title': 'Viditelnost v noci - nezhoršená', 'xlabel': None},
        {'ax': ax[1], 'df': b[b.visibility == 'NOTNI'], 'title': 'Viditelnost v noci - zhoršená', 'xlabel': None},
        {'ax': ax[2], 'df': b[b.visibility == 'VISDAY'], 'title': 'Viditelnost ve dne - nezhoršená', 'xlabel': 'Kraj'},
        {'ax': ax[3], 'df': b[b.visibility == 'NOTDAY'], 'title': 'Viditelnost ve dne - zhoršená', 'xlabel': 'Kraj'}
    ]

    for graph in graphs:
        # plot bar graph with region x axis and counts of accidents y axis with values of current visibility - graph['df']
        sns.barplot(graph['df'], ax=graph['ax'], x='region', y='counts', palette=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
        graph['ax'].set_title(graph['title'])

        if graph['xlabel'] is None:
            graph['ax'].xaxis.set_ticklabels([])
        graph['ax'].set_xlabel( graph['xlabel'])

        graph['ax'].set_ylabel('Počet Nehod')
        graph['ax'].locator_params(axis='y', nbins=6)

    if fig_location is not None:
        fig.savefig(fig_location)
    if show_figure:
        fig.show()
plot_visibility(df2, "01_visibility.png")

#p7	DRUH SRÁŽKY JEDOUCÍCH VOZIDEL
# 1	čelní
# 2	boční
# 3	z boku
# 4	zezadu
# 0	nepřichází v úvahu

#%%%
def plot_direction(df: pd.DataFrame, fig_location: str = None,
                   show_figure: bool = False):
    """Plot graph of number of accidents in 4 regions based on type of 
    precipitation and month.

    Args:
    df (pd.DataFrame) -- input dataframe
    fig_location (str, optional) -- output file location (default None)
    show_figure (bool, optional) -- show figure to output (default False)
    """

    # precipitation values dictionary
    precipitation = {0: 'EXCLUDED', 1: 'čelní', 2: 'boční', 3: 'boční', 4: 'zezadu'}
    
    newDF = df.copy()
    # map new precipitation values
    newDF['precipitation'] = newDF['p7'].map(precipitation)
    # drop unused columns in dataframe
    newDF.drop(newDF.index[((newDF['region'] != 'HKK') 
        & (newDF['region'] != 'LBK') 
        & (newDF['region'] != 'PLK') 
        & (newDF['region'] != 'STC')) 
        | (newDF['precipitation'] == 'EXCLUDED')], inplace = True)
    # set new index from datetime for grouping
    newDF.index = pd.to_datetime(newDF['date'],format='%m/%d/%y %I:%M%p')

    # group by dataframe by precipitation an month and choosing only region column
    a = newDF.groupby(by=[newDF.precipitation, newDF.index.month])['region']
    # get count of unique values combinations of precipitation-month-region
    b = a.value_counts().rename_axis(['precipitation','date','region']).reset_index(name='counts')

    fig, axes = plt.subplots(2, 2, figsize=(16,8))
    
    ax = axes.flatten()
    graphs = [
        {'ax': ax[0], 'df': b[b.region == 'HKK'], 'title': 'Královéhradecký kraj'},
        {'ax': ax[1], 'df': b[b.region == 'LBK'], 'title': 'Liberecký kraj'},
        {'ax': ax[2], 'df': b[b.region == 'PLK'], 'title': 'Plzeňský kraj'},
        {'ax': ax[3], 'df': b[b.region == 'STC'], 'title': 'Středočeský kraj'}
    ]


    for graph in graphs:
        # plot bar graph with date x axis and counts of accidents y axis and hue as precipitation with values of current region - graph['df']
        sns.barplot(graph['df'], ax=graph['ax'], x='date', y='counts', hue='precipitation', palette=['tab:blue', 'tab:orange', 'tab:green'])
        graph['ax'].set_title(graph['title'])

        graph['ax'].set_xlabel('Měsíc')
        graph['ax'].set_ylabel('Počet nehod')
        # step rounded to 25
        step = (int(int(graph['df']['counts'].max()/6)/25) + 1) * 25
        graph['ax'].set_ylim(top=step*6)
        graph['ax'].yaxis.set_ticks(np.arange(0, step*6, step))
        graph['ax'].legend().set_visible(False)

    # get legend values and labels from subplot and setting legend for main figure
    lines_labels = ax[0].get_legend_handles_labels()    
    legend = fig.legend(lines_labels[0], lines_labels[1], loc=7, title="Druh zrážky")
    legend.get_frame().set_alpha(None)
    
    fig.tight_layout()
    fig.subplots_adjust(right=0.93, wspace=0.15, hspace=0.32)

    if fig_location is not None:
        fig.savefig(fig_location)
    if show_figure:
        fig.show()
plot_direction(df2, "02_direction.png", True)

# p13	NÁSLEDKY NEHODY
# p13a	usmrceno osob
# p13b	těžce zraněno osob
# p13c	lehce zraněno osob

#%%%
def plot_consequences(df: pd.DataFrame, fig_location: str = None,
    show_figure: bool = False):
    
    newDF = df.copy()
    # drop unused columns in dataframe
    newDF.drop(newDF.index[(newDF['region'] != 'HKK') 
        & (newDF['region'] != 'LBK') 
        & (newDF['region'] != 'PLK') 
        & (newDF['region'] != 'STC')], inplace = True)

    def getConseq(x) -> None|str:
        """Get consequence value.

        Args:
        x -- dataframe column

        Returns:
            (None|str): value
        """

        if x['p13a']:
            return 'Usmrcení'
        if x['p13b']:
            return 'Těžké zranění'
        if x['p13c']:
            return 'Lehké zranění'
        return None

    # get value of consequences
    newDF['Následky'] = df.apply(lambda x: getConseq(x), axis=1)
    # get month and year from date
    newDF['month'] = newDF.date.dt.strftime('%Y-%m')
    newDF['month'] = pd.to_datetime(newDF['month'])

    table = pd.pivot_table(newDF, columns=['Následky'], values='p1', index=['region', 'month'], aggfunc='count')

    target = table.stack(level='Následky').unstack(level='region')

    # drop Nan values
    target.dropna(how='all', inplace=True)
    target = target.stack(level='region').reset_index(name='counts')

    # plot graphs with x value of months, y value of counts, subplot difference based
    # on region and hue based on type fo consequences
    plot = sns.relplot(data=target, x = 'month', y = 'counts', col = 'region',
        col_wrap = 2, kind='line', hue='Následky',
        palette=['tab:blue', 'tab:orange', 'tab:green'])

    for ax in plot.axes:
        if str(ax.title)[-5:-2] == 'HKK':
            ax.title.set_text('Královéhradecký kraj')
        elif str(ax.title)[-5:-2] == 'LBK':
            ax.title.set_text('Liberecký kraj')
        elif str(ax.title)[-5:-2] == 'PLK':
            ax.title.set_text('Plzeňský kraj')
        else:
            ax.title.set_text('Středočeský kraj')
    
        ax.set_xlabel('Měsíc')
        ax.set_ylabel('Počet nehod')
        ax.set_ylim(bottom=0)
        ax.set_xlim([datetime.date(2016, 1, 1), datetime.date(2022, 1, 1)])

    plot.figure.suptitle('Závažnost následků nehod v krajích')
    plot.figure.set_size_inches(14, 8)
    plot.figure.subplots_adjust(right=0.9, top=0.9, wspace=0.15, hspace=0.2)

    if fig_location is not None:
        plot.figure.savefig(fig_location)
    if show_figure:
        plot.figure.show()


plot_consequences(df2, "03_consequences.png", True)

# %%
