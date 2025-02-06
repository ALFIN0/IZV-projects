#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
import numpy as np
import datetime
import seaborn as sns

def plot_graph(df: pd.DataFrame, fig_location: str = None,
             show_figure: bool = False):
    """Plot graph with causes and consequences of accindents in 2020
    
    Args:
    df (pd.DataFrame) -- input
    fig_location (str, optional) -- output file location (default None)
    show_figure (bool, optional) -- show figure to output (default False)
    """

    newDF = df.copy()
    # set year column for separate axes plotting
    newDF['p2a'] = pd.to_datetime(newDF['p2a'])
    newDF['year'] = newDF.apply(lambda x: x.p2a.year, axis=1)

    newDF.drop(newDF.index[(newDF['year'] != datetime.datetime(2020,1,1).year)], inplace = True)

    def getCause(x) -> None|str:
        """Get cause value.

        Args:
        x -- dataframe column

        Returns:
            (None|str): value
        """

        if x['p12'] < 200:
            return 'Nezavinená vodičom'
        if x['p12'] > 200 and x['p12'] < 300:
            return 'Neprimeraná rýchlosť jazdy'
        if x['p12'] > 300 and x['p12'] < 400:
            return 'Nesprávne predchádzanie'
        if x['p12'] > 400 and x['p12'] < 500:
            return 'Nedanie prednosti v jazde'
        if x['p12'] > 500 and x['p12'] < 600:
            return 'Nesprávny spôsob jazdy'
        if x['p12'] > 600 and x['p12'] < 700:
            return 'Technická závada vozidla'
        return None

    def getConseq(x) -> None|str:
        """Get consequence value.

        Args:
        x -- dataframe column

        Returns:
            (None|str): value
        """

        if x['p13a']:
            return 'Usmrtenie'
        if x['p13b']:
            return 'Ťažké zranenie'
        if x['p13c']:
            return 'Ľahké zranenie'
        return None

    # get value of consequences
    newDF['Následky'] = df.apply(lambda x: getConseq(x), axis=1)

    # get value of causes
    newDF['Príčina'] = df.apply(lambda x: getCause(x), axis=1)

    # table with causes and consequences and their count
    table = pd.pivot_table(newDF, columns=['Následky'], values='p1', 
    index=['Príčina'], aggfunc='count', fill_value=0)
    table_print = table.style.format(decimal=',', thousands='', precision=0)
    print(table_print.to_latex(hrules=True, position_float="centering"))

    table_cols_sum = table.sum()
    print("Počet nehôd celkovo : %d" % (table_cols_sum.sum()))
    sum_of_bad_accidents = int(table_cols_sum["Usmrtenie"]) + int(table_cols_sum["Ťažké zranenie"])
    print("Počet ťažkých/smrteľných nehôd : %d" % (sum_of_bad_accidents))
    print(("Podiel ťažkých/smrteľných nehôd : %.2f " % (sum_of_bad_accidents / table_cols_sum.sum() * 100)) + "%")

    # sum of bad consequences in table
    table["bad"] = table["Usmrtenie"] + table["Ťažké zranenie"]
    # percentage of bad accidents to all accindents based on cause
    table["bat-to-all"] = table["bad"] / (table["Usmrtenie"] + table["Ťažké zranenie"] + table["Ľahké zranenie"]) * 100

    print(("Percento tragicosti nedania prednosti v jazde : %.2f " % (table["bat-to-all"]["Nedanie prednosti v jazde"])) + "%")
    print(("Percento tragicosti neprimeranej rýchlosti jazdy : %.2f " % (table["bat-to-all"]["Neprimeraná rýchlosť jazdy"])) + "%")
    print(("Percento tragicosti nesprávneho predchádzania : %.2f " % (table["bat-to-all"]["Nesprávne predchádzanie"])) + "%")
    print(("Percento tragicosti nesprávneho spôsobu jazdy : %.2f " % (table["bat-to-all"]["Nesprávny spôsob jazdy"])) + "%")
    print(("Percento tragicosti bez zavinenia vodiča : %.2f " % (table["bat-to-all"]["Nezavinená vodičom"])) + "%")
    print(("Percento tragicosti technickej závady vozidla : %.2f " % (table["bat-to-all"]["Technická závada vozidla"])) + "%")

    # group by of causes and consquences for plotting
    a = newDF.groupby(by=[newDF['Následky']])['Príčina']
    b = a.value_counts().reset_index(name='counts')

    # plot bar graph
    plot = sns.barplot(b, x='Následky', y='counts', 
            hue='Príčina')

    # set labels and title
    plot.axes.title.set_text('Počet nehôd so stupňom následkov v závsíslosti od príčiny v roku 2020')
    plot.axes.set_ylabel('Počet nehôd')

    if fig_location is not None:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

if __name__ == "__main__":
    df = pd.read_pickle("accidents.pkl.gz")
    plot_graph(df, "fig.png", True)