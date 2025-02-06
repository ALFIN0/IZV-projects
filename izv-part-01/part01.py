#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Patrik Dvorščák, (xdvors15)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

import bs4
from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def integrate(x: np.array, y: np.array) -> float:
    """Calculate definite integral.

    Keyword arguments:
    x (np.array) -- vector of x values of integral point
    y (np.array) -- vector of y values of integral point

    Returns:
        (float) value of definite integral
    """

    return np.sum(np.diff(x) * (np.add(y[:-1], y[1:]) / 2))


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None=None):
    """Create graphs of function fa(x) = a * x^2 from argument a.

    Keyword arguments:
    a (List[float]) -- list of constants a
    show_figure (bool, optional) -- show figure (default False)
    save_path (str | None, optional) -- path for storing generated graphs figure (default None)
    """

    # limits of x (t) and y axis
    x_from, x_to = -3, 3
    y_from, y_to = -20, 20

    # t axis linspace
    x = np.linspace(x_from, x_to, 1000)
    # y axis values
    y = np.array(a).reshape(-1, 1) * x**2

    figure = plt.figure(figsize=(7, 4))

    # create main subplot of figure
    axes = figure.add_subplot()
    axes.set_xlabel('$x$')
    axes.set_ylabel('$f_a(x)$')
    axes.xaxis.set_ticks(np.arange(x_from, x_to + 1, 1))

    # plot functions for all a constants
    for y_values in y:
        axes.plot(x, y_values)

    # set legend above graph subplot
    axes.legend([f'$y_{{{input_item}}}(x)$' for input_item in a], loc='upper center', ncols=len(a), bbox_to_anchor=(0.5, 1.15))

    for (input_item, y_values) in zip(a, y):
        # fill space between x and y values with visibility 10%
        axes.fill_between(x, y_values, step='mid', alpha=0.1)
        # add legend text next to function
        axes.text(x[-1], y_values[-1] - 0.5, f'$\\int f_{{{input_item}}}(x)dx$')

    # set x,y limits x limit is +1 larger because of legend
    axes.set_xlim(left=x_from, right=x_to + 1)
    axes.set_ylim(bottom=y_from, top=y_to)

    if save_path is not None:
        figure.savefig(save_path)

    if show_figure:
        figure.show()


def generate_sinus(show_figure: bool=False, save_path: str | None=None):
    """Create graphs of functions f1(t) = 0.5 * sin(pi*t/50), f2(t) = 0.25 * sin(pi*t) 
    and f3(t) = f1(t) + f2(t) in range of time <0;100>.

    Keyword arguments:
    show_figure (bool, optional) -- show figure (default False)
    save_path (str | None, optional) -- path for storing generated graphs figure (default None)
    """

    # limits of x (t) and y axis
    t_from, t_to = 0, 100
    y_from, y_to = -.8, .8

    # t axis linspace
    t = np.linspace(t_from, t_to, 8000)
    # y axis values of 3 graphs
    y = [0.5 * np.sin(np.pi * t * (1/50)), 0.25 * np.sin(np.pi * t)]
    y.append(y[0] + y[1])

    # figure with 3 subplots
    figure, axis = plt.subplots(3,1)
    figure.set_size_inches(7.5,10)
    figure.subplots_adjust(hspace=0.3)

    index = 0
    for ax, y_values in zip(axis, y):
        ax.xaxis.set_ticks(np.arange(t_from, t_to + 1, 20))
        ax.yaxis.set_ticks(np.arange(y_from, y_to + 1, 0.4))
        ax.set_xlim(left=t_from, right=t_to)
        ax.set_ylim(bottom=y_from, top=y_to)
        ax.set_xlabel('$t$')

        if (index < 2):
            ax.set_ylabel(f'$f_{{{index + 1}}}(t)$')
            # plot f1(t) and f2(t)
            ax.plot(t, y_values)
        else:
            ax.set_ylabel(f'$f_1(t) + f_2(t)$')
            y_red = []
            y_green = []

            # compare value of f1(t) + f2(t) with value of f1(t)
            for (y_1, y_12) in zip(y[0], y[2]):
                if y_1 > y_12:
                    y_red.append(y_12)
                    y_green.append(np.nan)
                else:
                    y_red.append(np.nan)
                    y_green.append(y_12)

            # plot f1(t) + f2(t) with part under values of f1(t)
            ax.plot(t, y_green, color='green')
            # plot f1(t) + f2(t) with part above values of f1(t)
            ax.plot(t, y_red, color='red')

        index += 1

    if save_path is not None:
        figure.savefig(save_path)

    if show_figure:
        figure.show()


def download_data(url="https://ehw.fit.vutbr.cz/izv/temp.html") -> list:
    """Download data from url and parse temperatures with month and year.

    Keyword arguments:
    url -- url for download (default "https://ehw.fit.vutbr.cz/izv/temp.html")

    Raises:
        requests.ConnectionError: incorrect request response status code

    Returns:
        list: array of data in format {'year': value, 'month': value, 'temp': array(float)}
    """

    # get response from url (redirect handled)
    response = requests.get(url)
    
    # check response status code
    if (int(response.status_code) < 200 or int(response.status_code) >= 300):
        raise requests.ConnectionError
    
    # parse html doc
    html = BeautifulSoup(response.content, 'xml')
    result = []
    
    row: bs4.element.Tag
    # retrieve data from html doc
    for row in html.find_all('tr'):
        cells = [cell.get_text().replace(',', '.') for cell in row.find_all('p')]
        
        result.append({
            'year' : int(cells[0]),
            'month' : int(cells[1]),
            'temp' : np.array(cells[2:], dtype='float')
        })
     
    return result


def get_avg_temp(data, year=None, month=None) -> float:
    """Compute arithmetic mean from given temperature data filtered by year and month.

    Keyword arguments:
    data -- array of data in format {'year': value, 'month': value, 'temp': array(float)}
    year -- filtered year (default None)
    month -- filtered month (default None)

    Returns:
        (float) mean average of temperatures
    """

    # concatenate all temps that match the filter
    res = np.concatenate([d['temp'] for d in data if 
    (year is None or year == d['year']) 
    and (month is None or month == d['month'])]
    , axis=0)

    return np.average(res)