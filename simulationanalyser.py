import os
import datetime
import json
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analyser import Analyser


class SimulationAnalyser(Analyser):
    def __init__(self, file):
        super().__init__()
        self.data = pd.read_csv(file, index_col='Unnamed: 0', parse_dates=True)

    def sum(self, column_names: List, total_column_name: str):
        self.data[total_column_name] = self.data[column_names].sum(axis=1)

    def calculate_total_pv(self, pv_column_names: List):
        self.sum(pv_column_names, 'pv total power')
        self.data['pv total power'] = self.data['pv total power']
        return sum(self.data['pv total power'].dropna())

    def calculate_total_demand(self, demand_column_names):
        self.sum(demand_column_names, 'eldemand total power')
        self.data['eldemand total power'] = self.data['eldemand total power']
        return sum(self.data['eldemand total power'].dropna())

    def calculate_self_consumption_pv(self, pv_column_name, demand_column_name):
        d = self.data
        d['selfconsumption pv'] = d.apply(
            lambda d: self._calculate_self_consumption_pv(d[pv_column_name], d[demand_column_name]), axis=1)
        return sum(d['selfconsumption pv'] * d[demand_column_name]) / sum(d[demand_column_name])

    def calculate_self_consumption(self, pv_column_name, battery_column_name, demand_column_name):
        d = self.data
        d['selfconsumption'] = d.apply(
            lambda d: self._calculate_self_consumption(d[pv_column_name], d[battery_column_name], d[demand_column_name]), axis=1)
        return sum(d['selfconsumption'] * d[demand_column_name]) / sum(d[demand_column_name])

    def get_battery_storage_solar(self, pv_column_name, battery_column_name, demand_column_name):
        d = self.data
        d['battery stored solar'] = d.apply(
            lambda d: self._get_battery_storage_solar(d[pv_column_name], d[battery_column_name], d[demand_column_name]), axis=1)
        return sum(d['battery stored solar'].dropna())

    def get_battery_storage_grid(self, pv_column_name, battery_column_name, demand_column_name):
        d = self.data
        d['battery stored grid'] = d.apply(
            lambda d: self._get_battery_storage_grid_from_data(d[pv_column_name], d[battery_column_name],
                                                               d[demand_column_name]), axis=1)
        if any(d['battery stored grid'] < 0.0):
            print('WARNING: one or more negative values for grid power stored in battery were calculated. This might'
                  'indicate something was wrong with the analysed simulation!')

        return sum(d['battery stored grid'].dropna())

    def get_total_grid_consumption(self, pv_column_name, demand_column_name, battery_column_name=None):
        d = self.data

        if battery_column_name:
            d['grid demand'] = d.apply(
                lambda d: self._get_demand_from_data(d[pv_column_name], d[battery_column_name],
                                                     d[demand_column_name]), axis=1)
            d['battery charging demand'] = d.apply(
                lambda d: self._get_battery_charging_energy(d[pv_column_name], d[battery_column_name],
                                                           d[demand_column_name]), axis=1)

            d['total grid demand'] = d['grid demand'] + d['battery charging demand']
        else:
            d['battery'] = [0.0] * len(d)
            d['total grid demand'] = d.apply(
                lambda d: self._get_demand_from_data(d[pv_column_name], d['battery'],
                                                     d[demand_column_name]), axis=1)
        return sum(d['total grid demand'].dropna())

    def get_injection(self, pv_column_name, demand_column_name, battery_soc_column_name=None, price_column_name=None):
        d = self.data

        if price_column_name:
            if battery_soc_column_name:
                d['injected profit'] = d.apply(
                    lambda d: self._get_injected_power(d[pv_column_name], d[demand_column_name],
                                                       d[battery_soc_column_name], d[price_column_name]), axis=1)
            else:
                d['injected profit'] = d.apply(
                    lambda d: self._get_injected_power(d[pv_column_name], d[demand_column_name],
                                                       d[price_column_name]), axis=1)
            return sum(d['injected profit'].dropna())
        else:
            if battery_soc_column_name:
                d['injected'] = d.apply(
                    lambda d: self._get_injected_power(d[pv_column_name], d[demand_column_name],
                                                       d[battery_soc_column_name]),
                    axis=1)
            else:
                d['injected'] = d.apply(
                    lambda d: self._get_injected_power(d[pv_column_name], d[demand_column_name]),
                    axis=1)
            return sum(d['injected'].dropna())

    def get_total_costs(self, pv_column_name, demand_column_name,
                        price_column_name, battery_column_name=None):
        d = self.data

        d['demand costs no ems'] = d[demand_column_name] * d[price_column_name]

        if battery_column_name:
            d['demand costs'] = d.apply(
                lambda d: self._get_demand_costs_from_data(d[pv_column_name], d[battery_column_name],
                                                           d[demand_column_name],
                                                           d[price_column_name]), axis=1)
            d['battery charging costs'] = d.apply(
                lambda d: self._get_battery_charging_costs(d[pv_column_name], d[battery_column_name],
                                                           d[demand_column_name],
                                                           d[price_column_name]), axis=1)

            d['total costs'] = d['demand costs'] + d['battery charging costs']
        else:
            d['battery'] = [0.0] * len(d)
            d['total costs'] = d.apply(
                lambda d: self._get_demand_costs_from_data(d[pv_column_name], d['battery'],
                                                           d[demand_column_name],
                                                           d[price_column_name]), axis=1)
        return sum(d['total costs'].dropna())

    def get_total_savings(self, pv_column_name, demand_column_name, price_column_name, battery_column_name=None):
        d = self.data
        if battery_column_name:
            d['avoided costs pv'] = d.apply(
                lambda d: self._get_pv_savings(d[pv_column_name], d[demand_column_name],
                                               d[price_column_name]), axis=1)

            d['avoided costs battery'] = d.apply(
                lambda d: self._get_battery_financial_savings_from_data(d[battery_column_name],
                                                                        d[price_column_name]), axis=1)

            d['total avoided costs'] = d['avoided costs pv'] + d['avoided costs battery']
        else:
            d['total avoided costs'] = d.apply(
                lambda d: self._get_pv_savings(d[pv_column_name], d[demand_column_name],
                                               d[price_column_name]), axis=1)
        return sum(d['total avoided costs'].dropna())

    def get_wpo_on_when_wasteheat(self, year):
        def _get_wpo_on_when_wasteheat(wasteheat, heatdemand, wpoheat):
            if wasteheat > heatdemand and wpoheat > 10000.0:
                return True
            else:
                return False

    def get_monthly_averages(self, sums=[], means=[]):
        self.monthly_averages = pd.DataFrame()
        for name in sums:
            self.monthly_averages[name] = self.data[name].resample('MS').sum()
        for name in means:
            self.monthly_averages[name] = self.data[name].resample('MS').mean() * 100

    def plot_monthly_energy(self):
        days = 8
        width = datetime.timedelta(hours=24 * days)

        fig, ax = plt.subplots(2, 1, figsize=(15, 8))
        ax[0].bar(self.monthly_averages['pv total power'].index - width / 2,
                  self.monthly_averages['pv total power'] / 1000, [days] * len(self.monthly_averages['pv total power']),
                  edgecolor='k', label='PV')
        ax[0].bar(self.monthly_averages['battery stored solar'].index,
                  self.monthly_averages['battery stored solar'] / 1000, [days] * len(self.monthly_averages['battery stored solar']),
                  edgecolor='k', label='Battery storage (solar)')
        ax[0].bar(self.monthly_averages['battery stored grid'].index,
                  self.monthly_averages['battery stored grid'] / 1000, [days] * len(self.monthly_averages['battery stored grid']),
                  bottom=self.monthly_averages['battery stored solar'] / 1000,
                  edgecolor='k', label='Battery storage (grid)')

        ax[1].bar(self.monthly_averages['selfconsumption'].index - width / 4,
                  self.monthly_averages['selfconsumption'], [days] * len(self.monthly_averages['selfconsumption']),
                  facecolor=sns.color_palette()[0], edgecolor='k',
                  label='Self consumption / low-priced grid electricity')
        ax[1].bar(self.monthly_averages['selfconsumption pv'].index + width / 4,
                  self.monthly_averages['selfconsumption pv'], [days] * len(self.monthly_averages['selfconsumption pv']),
                  facecolor=sns.color_palette()[1], edgecolor='k', label='Self consumption only PV')

        # ax.set_ylabel('Solar energy [kWh]', fontsize=15)
        for a in ax:
            #a.set_xlim(min(self.monthly_averages.index), min(self.monthly_averages.index))
            a.tick_params(labelsize=12)
            a.legend(fontsize=12)

        ax[1].set_ylim(0, 100)
        ax[0].set_ylim(0, )

        ax[0].set_title('Energy per month [kWh]', fontsize=15)
        ax[1].set_title('Self consumption [%]', fontsize=15)

        sns.despine()

    def plot_costs(self, year):
        """
        Costs: covering demand and charging battery with grid
        Savings: demand covered by PV and battery
        """
        days = 8
        width = datetime.timedelta(hours=24 * days)

        fig, ax = plt.subplots(2, 1, figsize=(15, 8))
        ax[0].bar(self.monthly_averages['avoided costs battery'].index - width / 2,
                  self.monthly_averages['avoided costs battery'],
                  [days] * len(self.monthly_averages['avoided costs battery']),
                  edgecolor='k', label='Savings discharging battery')
        ax[0].bar(self.monthly_averages['battery charging costs'].index,
                  self.monthly_averages['battery charging costs'],
                  [days] * len(self.monthly_averages['battery charging costs']),
                  edgecolor='k', label='Cost charging battery')
        ax[1].bar(self.monthly_averages['total costs'].index,
               self.monthly_averages['total costs'], [days] * len(self.monthly_averages['total costs']),
               edgecolor='k', label='Actual')
        ax[1].bar(self.monthly_averages['total avoided costs'].index,
               self.monthly_averages['total avoided costs'], [days] * len(self.monthly_averages['total avoided costs']),
               bottom=self.monthly_averages['total costs'],
               edgecolor='k', label='Avoided')

        for i in range(0, len(self.monthly_averages['total costs'])):
            try:
                ax[1].text(self.monthly_averages['total costs'].index[i] - width / 1.5,
                        self.monthly_averages['total avoided costs'][i] + self.monthly_averages['total costs'][i] + 100,
                        str(int(round(self.monthly_averages['total avoided costs'][i] / (self.monthly_averages['total avoided costs'][i] + self.monthly_averages['total costs'][i]) * 100,
                                      0))) + '%',
                        fontsize=15)
            except:
                pass

        for a in ax:
            a.set_xlim(datetime.datetime(year - 1, 12, 20), datetime.datetime(year, 12, 15))
            a.tick_params(labelsize=12)
            a.legend(fontsize=12)
            a.set_ylim(0, )

        ax[0].set_title('Battery charging costs/discharging savings [€]', fontsize=15)
        ax[1].set_title('Electricity costs [€] and savings [%] per month ', fontsize=15)
        # ax.set_ylabel('€', fontsize=15)

        ax[1].text(self.monthly_averages['total costs'].index[0] - width * 2,
                   -1000.0,
                   f"Cumulative: {round(self.monthly_averages['total costs'].sum())}€ spent; "
                   f"{round(self.monthly_averages['total avoided costs'].sum())}€ saved", fontsize=12)

        fig.tight_layout()
        sns.despine()
