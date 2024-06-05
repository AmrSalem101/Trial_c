#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:01:20 2022

@author: chaimdemulder
"""
import requests
import datetime
import ipydatetime
import ipywidgets as widgets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ipywidgets import interact, interact_manual
from IPython.display import display
from collections import defaultdict
from typing import List

from analyser import Analyser

style = {'description_width': 'initial'}


class DataAnalyser(Analyser):
    """
    Class for reading data
    """
    watt_to_watthour_factors = {'5m': 12.0, '15m': 4.0, 'h': 1.0, 'D': 1 / 24}

    def __init__(self, year):
        """
        :param installation_id:
        :param client_id:
        :param client_secret:
        :param use_widget:
        :param use_api:
        """
        super().__init__()
        self.battery = dict()
        self.pv = dict()
        self.demand = dict()
        self.private_demand = dict()
        self.heat_demand = dict()
        self.heatpump = dict()
        self.price = dict()
        self.temperature = dict()
        self.ems = dict()
        self.grid_load_data = dict()
        self.monthly_electricity = pd.DataFrame()
        self.monthly_heat = pd.DataFrame()
        self.joined_electricity = pd.DataFrame()
        self.joined_heat = pd.DataFrame()
        self.current_year = year

    def get_battery_data(self, start=ipydatetime.DatetimePicker(),
                         end=ipydatetime.DatetimePicker(),
                         resolution=['5m', '15m', 'h', 'D'],
                         group_function=['last', 'mean']):
        data = self._get(start, end, resolution, group_function,
                         metric_type='battery', metric_source='batterysupplies')
        dictionary = defaultdict(dict)
        keys = [k.split('_')[1] for k in data[0]['values'].keys() if '.realPower' in k or '.stateOfCharge' in k]
        for key in keys:
            for data_instance in data:
                dictionary[key][data_instance['time']] = data_instance['values'][key]
        # convert to pd.DataFrame
        self.battery_data = dictionary

    def get_pv_data(self, start=ipydatetime.DatetimePicker(),
                    end=ipydatetime.DatetimePicker(),
                    resolution=['5m', '15m', 'h', 'D'],
                    group_function=['last', 'mean']):
        data = self._get(start, end, resolution, group_function,
                         metric_type='sma')
        # convert to pd.DataFrame
        self.pv_data = data

    def get_electricity_demand_data(self, start=ipydatetime.DatetimePicker(),
                                    end=ipydatetime.DatetimePicker(),
                                    resolution=['5m', '15m', 'h', 'D'],
                                    group_function=['last', 'mean']):
        """
        Get electricity demand data from influxdb database. Get data in W and convert to kWh
        """
        name = 'FAA.A.Pe.Actief Pe Ducoop'
        key = group_function + '_' + name
        watt_to_watthour_factor = DataAnalyser.watt_to_watthour_factors[resolution]

        data = self._get(start, end, resolution, group_function,
                         metric_type='modbus', metric_source='simatics7')

        dictionary = defaultdict(dict)
        for data_instance in data:
            try:
                dictionary[key][data_instance['time']] = data_instance['values'][key] / watt_to_watthour_factor / 1000.0
            except:
                print(f"Could not format data point: {data_instance['values'][key]} - Using 'None'")
                dictionary[key][data_instance['time']] = None

        d = pd.DataFrame(dictionary).rename({key: 'electricity_demand'}, axis=1)
        d['datetime'] = d.index
        d = d.set_index(d['datetime'].apply(datetime.datetime.fromtimestamp)).drop('datetime', axis=1)
        for year in range(start.year, end.year + 1):
            self.demand[year] = d.loc[datetime.datetime(year, 1, 1): datetime.datetime(year, 12, 31)]

    def get_data(self, start=ipydatetime.DatetimePicker(), end=ipydatetime.DatetimePicker(),
                 resolution=['5m', 'h', 'D'], group_function=['last', 'mean'],
                 battery=False, pv=False, electricity_demand=False, grid_load=False):
        if battery:
            print('Getting battery data...')
            self.get_battery_data(start, end, resolution, group_function)
        if pv:
            print('Getting pv data...')
            self.get_pv_data(start, end, resolution, group_function)
        if electricity_demand:
            print('Getting electricity demand data...')
            self.get_electricity_demand_data(start, end, resolution, group_function)
        if grid_load:
            print('Getting grid load data...')
            self.read_grid_load(start.month)
        print('Data fetched!')

    def _get(self, start, end, resolution, group_function, metric_type, metric_source=None):
        if not self.token:
            print('Request failed: No token defined yet!')
            return {}
        params = {'end': int(end.timestamp()),
                  'start': int(start.timestamp()),
                  'resolution': resolution,
                  'group_function': group_function}
        if metric_source:
            params.update({'metric_source': metric_source})
        response = requests.get(
            f"https://cloud.openmotics.com/api/v1/base/installations/{self.installation_id}/metrics/type/{metric_type}",
            params=params, headers=self.header)
        if not response.ok:
            print(f'Request failed: {response.text}')
            return {}

        data = response.json().get_metrics('data')
        if not data:
            print('Response does not have expected "data" key!')
        return data

    def plot_(self):
        return None

    def get_widget(self, method):
        widget = interact_manual(method)
        widget.widget.children[-2].description = ' '.join(method.__name__.split('_'))

    def read(self, path='', data_type=['pv', 'battery', 'demand', 'heat_demand', 'heatpump', 'price', 'temperature',
                                       'private_demand'],
             year=[2021, 2022], convert_to_wh=True):
        """

        :param path: str
        :param data_type: str, options: 'pv', 'battery', 'demand', 'heat_demand', 'price', 'temperature', 'private_demand'
        :param year: int
        :param convert_to_wh: boolean
        :return:
        """
        try:
            data = pd.read_csv(path)
        except FileNotFoundError:
            print(f"File {path} not found!")
            return
        data['datetime'] = data['Time'].apply(pd.to_datetime)
        data = data.drop(['Time'], axis=1).set_index('datetime', drop=True).sort_index()
        if data_type == 'pv':
            self._format_pv(data, year, convert_to_wh)
        if data_type == 'pv_ems':
            self._format_pv_ems(data, year, convert_to_wh)
        elif data_type == 'battery':
            self._format_battery(data, year, convert_to_wh)
        elif data_type == 'demand':
            self._format_demand(data, year, convert_to_wh)
        elif data_type == 'private_demand':
            self._format_private_demand(data, year, convert_to_wh)
        elif data_type == 'heat_demand':
            self._format_heat_demand(data, year, convert_to_wh)
        elif data_type == 'price':
            self._format_price(data, year)
        elif data_type == 'temperature':
            self._format_temperature(data, year)
        elif data_type == 'heatpump':
            self._format_heatpump(data, year, convert_to_wh)

    def _format_pv(self, pv, year, convert_to_wh):
        if year == 2021:
            pv['0198B310A9C0'].fillna(pv['0198-B310A9C0'], inplace=True)
            pv.drop('0198-B310A9C0', axis=1, inplace=True)
            pv['0198B32BA4B4'].fillna(pv['0198-B32BA4B4'], inplace=True)
            pv.drop('0198-B32BA4B4', axis=1, inplace=True)
            pv['017AB330E9A1'] = pv['017AB330E9A1'].fillna(pv['017A-B330E9A1']).fillna(pv['017A-xxxxx9A1']).fillna(pv['017Axxxxx9A1'])
            pv.drop(['017A-B330E9A1', '017A-xxxxx9A1', '017Axxxxx9A1'], axis=1, inplace=True)
            pv['pv'] = pv.sum(axis=1) * 2.3  # times 2.3 because two of 5 pv installations are missing...
        elif year == 2022:
            pv['pv'] = [0.0] * len(pv)
            # times 2.3 because two of 5 pv installations are missing...
            pv.loc[datetime.datetime(2022, 1, 1):datetime.datetime(2022, 3, 10, 9, 50), 'pv'] = \
                pv.loc[datetime.datetime(2022, 1, 1):datetime.datetime(2022, 3, 10, 9, 50)].sum(axis=1, numeric_only=True) * 2.3
            # times 1.6 because one of 5 pv installations is missing.
            pv.loc[datetime.datetime(2022, 3, 10, 9, 50):, 'pv'] = \
                pv.loc[datetime.datetime(2022, 3, 10, 9, 50):].sum(axis=1, numeric_only=True) * 1.6
        elif year in [2023, 2024]:
            pv.loc[:, 'pv'] = pv.sum(axis=1, numeric_only=True)
        else:
            print(f'no specific data formatting for year {year} was defined yet! Using raw data!')
        pv = pv.loc[~pv.index.duplicated(), :]
        pv = pv.asfreq('15T')
        if convert_to_wh:
            pv = self._convert_series_to_wh(pv)
        self.pv[year] = pv

    def _format_pv_ems(self, pv, year, convert_to_wh):
        pv['pv'] = pv['Solar energy production']
        pv = pv.loc[~pv.index.duplicated(), :]
        pv = pv.asfreq('15T')
        if convert_to_wh:
            pv = self._convert_series_to_wh(pv)
        self.pv[year] = pv

    def _format_battery(self, battery, year, convert_to_wh):
        if year == 2021:
            try:
                battery['Power'].fillna(battery['Power.1'], inplace=True)
                battery['State'].fillna(battery['State.1'], inplace=True)
                battery['State of Charge'].fillna(battery['State of Charge.1'], inplace=True)
                battery.drop(['Power.1', 'State.1', 'State of Charge.1'], axis=1, inplace=True)
            except:
                pass
            battery.loc[datetime.datetime(2021, 1, 1):datetime.datetime(2021, 6, 3, 14, 15), 'Power'] = \
                battery.loc[datetime.datetime(2021, 1, 1):datetime.datetime(2021, 6, 3, 14, 15), 'Power'] * 1000

        battery.rename(columns={'Power': 'battery', 'State': 'battery_state',
                                'State of Charge': 'battery_state_of_charge'}, inplace=True)
        battery = battery.loc[~battery.index.duplicated(), :]
        try:
            battery.loc[:, 'battery_state'] = battery['battery_state'].astype(float)
        except KeyError:
            print("No 'battery state' value included in data")
        battery = battery.asfreq('15T')
        if convert_to_wh:
            battery['battery'] = self._convert_series_to_wh(battery['battery'])
        self.battery[year] = battery

    def _format_demand(self, demand, year, convert_to_wh):
        if year == 2021:
            demand.loc[datetime.datetime(2021, 1, 1):datetime.datetime(2021, 6, 3, 14, 0), 'Total measured'] = \
                demand.loc[datetime.datetime(2021, 1, 1):datetime.datetime(2021, 6, 3, 14, 15), 'Total measured'] * 1000

        demand.rename(columns={'Warmtepomp': 'heatpump',
                               'Total measured': 'demand',
                               'Total for simulation': 'demand_ems',
                               'Laadpalen': 'evcharging'}, inplace=True)
        demand = demand.loc[~demand.index.duplicated(), :]
        demand = demand.asfreq('15T')
        demand['demand'].where(demand['demand'] > 0.0, 0.1, inplace=True)
        try:
            demand['demand_ems'].where(demand['demand_ems'] > 0.0, 0.1, inplace=True)
        except KeyError:
            pass
        if convert_to_wh:
            demand = self._convert_series_to_wh(demand)
        self.demand[year] = demand

    def _format_private_demand(self, demand, year, convert_to_wh):
        self.nr_private_units = len(demand.columns)
        demand = pd.DataFrame(demand.sum(axis=1), columns=['private demand'])
        demand = demand.loc[~demand.index.duplicated(), :]
        demand = demand.asfreq('15T')
        demand = demand.where(demand['private demand'] < 1e6, 0.0)
        if convert_to_wh:
            demand = self._convert_series_to_wh(demand)
        self.private_demand[year] = demand

    def _format_heat_demand(self, heat, year, convert_to_wh):
        if year == 2021:
            heat.loc[datetime.datetime(2021, 1, 1):datetime.datetime(2021, 6, 3, 13, 0), ['Heat produced Christeyns',
                                                                              'Heat Waste', 'Heat Demand',
                                                                              'Heat WPO', 'Electricity WPO']] = \
            heat.loc[datetime.datetime(2021, 1, 1):datetime.datetime(2021, 6, 3, 13, 0), ['Heat produced Christeyns',
                                                                              'Heat Waste', 'Heat Demand',
                                                                              'Heat WPO', 'Electricity WPO']] * 1000
        if year == 2022:
            heat.loc[:, 'Driewegkraan esters'] = [100.0] * len(heat)
        if year == 2023:
            heat.loc[datetime.datetime(2023, 1, 1):datetime.datetime(2023, 11, 15), 'Driewegkraan esters'] = \
                [100.0] * len(heat[datetime.datetime(2023, 1, 1):datetime.datetime(2023, 11, 15)])
        heat.rename(columns={'Heat Waste': 'total_wasteheat',
                             'Heat produced Christeyns': 'total_external_heat',
                             'Heat WPO': 'heatpump',
                             'Heat Gas Boiler': 'gasboiler',
                             'Electricity WPO': 'heatpump_electricity',
                             'Heat Demand': 'heatdemand',
                             'Driewegkraan esters': 'wasteheat_split'}, inplace=True)
        heat = heat.loc[~heat.index.duplicated(), :]
        heat.loc[:, 'heatpump_electricity'] = heat['heatpump_electricity'].where(heat['heatpump_electricity'] > 300.0,
                                                                                 other=0.0)
        heat.loc[:, 'wasteheat_split'] = heat['wasteheat_split'].where(heat['wasteheat_split'] > 1.0,
                                                                       other=0.0)
        for heat_measurement in ['total_external_heat', 'total_wasteheat', 'heatdemand', 'heatpump']:
            heat.loc[:, heat_measurement] = heat[heat_measurement].where(heat[heat_measurement] > 0.0, other=0.0)
        heat = heat.asfreq('15T')
        if convert_to_wh:
            heat.loc[:, ['total_wasteheat', 'total_external_heat', 'heatpump', 'gasboiler', 'heatpump_electricity', 'heatdemand']] = \
                self._convert_series_to_wh(
                    heat.loc[:, ['total_wasteheat', 'total_external_heat', 'heatpump', 'gasboiler', 'heatpump_electricity', 'heatdemand']])
        self.heat_demand[year] = heat

    def _format_price(self, price, year):
        price = price.loc[~price.index.duplicated(), :]
        if year == 2021:
            price.loc[datetime.datetime(2021, 1, 1):datetime.datetime(2021, 6, 3, 14, 15), 'Belpex electricity price'] = \
                price.loc[datetime.datetime(2021, 1, 1):datetime.datetime(2021, 6, 3, 14, 15), 'Belpex electricity price'] / 1e6

        price.rename(columns={'Belpex electricity price': 'price'}, inplace=True)
        price = price.resample('15min').ffill()
        #price.loc[datetime.datetime(year, 12, 31, 23, 15), :] = price.loc[datetime.datetime(year, 12, 31, 23, 0), :]
        #price.loc[datetime.datetime(year, 12, 31, 23, 30), :] = price.loc[datetime.datetime(year, 12, 31, 23, 0), :]
        #price.loc[datetime.datetime(year, 12, 31, 23, 45), :] = price.loc[datetime.datetime(year, 12, 31, 23, 0), :]
        self.price[year] = price / 1e6  # €/Wh

    def _format_temperature(self, data, year):
        data = data.loc[~data.index.duplicated(), :]
        data.rename(columns={'davis_weatherstation.Inside Temp': "inside",
                             "davis_weatherstation.Outside Temp": "outside",
                             "Forecast": "outside_forecast"}, inplace=True)
        data = data.resample('15min').ffill()
        self.temperature[year] = data

    def _format_ems(self, data, year):
        data = data.loc[~data.index.duplicated(), :]
        data = data.resample('15min').ffill()
        self.ems[year] = data

    def _format_heatpump(self, data, year, convert_to_wh):
        if year == 2021:
            data.loc[datetime.datetime(2021, 1, 1):datetime.datetime(2021, 6, 3, 13, 0), ['Thermal power',
                                                                                          'Electric power']] = \
            data.loc[datetime.datetime(2021, 1, 1):datetime.datetime(2021, 6, 3, 13, 0), ['Thermal power',
                                                                                          'Electric power']] * 1000
        data = data.loc[~data.index.duplicated(), :]
        data.loc[:, 'Electric power'] = data['Electric power'].where(data['Electric power'] > 300.0, other=0.0)
        data.loc[:, 'Thermal power'] = data['Thermal power'].where(data['Thermal power'] > 0.0, other=0.0)
        data = data.resample('15min').ffill()
        if convert_to_wh:
            data.loc[:, ['Thermal power', 'Electric power']] = \
                self._convert_series_to_wh(data[['Thermal power', 'Electric power']])
        self.heatpump[year] = data

    def read_grid_load(self, start=ipydatetime.DatetimePicker(), end=ipydatetime.DatetimePicker(), convert_to_wh=False):
        """
        Read grid load data from excel files Eoly, convert to Wh (instead of kWh)
        """
        df = pd.DataFrame()
        all_months = pd.date_range(start, end, freq='M')
        for month in all_months:
            year = month.year
            month = month.month
            try:
                raw = pd.read_excel(f'./data/{year}/{year}{"%02d" % month}_GridConsumption_15min.xlsx')
                raw.dropna(inplace=True)
            except FileNotFoundError:
                print(f'File ./data/{year}/{year}{"%02d" % month}_GridConsumption_15min.xlsx not found!')
                continue
            data = pd.DataFrame(index=raw['End'],
                                data={'grid_load': raw['Energie kWh'].values,
                                      'datetime': raw['End'].values})
            df = pd.concat([df, data], axis=1).drop_duplicates('datetime')
        df = df.drop('datetime', axis=1)
        if convert_to_wh:
            df = self._convert_series_to_wh(df)
        for year in range(start.year, end.year + 1):
            self.grid_load_data[year] = df.loc[datetime.datetime(year, 1, 1): datetime.datetime(year, 12, 31)]

    @staticmethod
    def _convert_series_to_wh(data):
        second_frequency = pd.to_timedelta(data.index.freq).seconds
        w_to_wh_converter = second_frequency / 3600
        return data * w_to_wh_converter

    @staticmethod
    def _convert_value_to_wh(value, second_frequency):
        w_to_wh_converter = second_frequency / 3600
        return value * w_to_wh_converter

    # %% ELECTRICITY
    def join_electricity_series(self, years: List):
        """To be worked on!"""
        for year in years:
            try:
                self.joined_electricity = pd.concat([self.joined_electricity,
                                                     pd.concat([self.demand[year]['demand'],
                                                                self.demand[year]['evcharging'],
                                                                self.demand[year]['heatpump'],
                                                                self.battery[year],
                                                                self.pv[year]['pv'],
                                                                self.price[year]], axis=1)])
            except KeyError:
                print("Series could not be joined, there might be data missing. Expected data: demand, pv, battery, heatpump, price")

    def add_fixed_grid_price(self, fixed_price, type_='value'):
        if type_ == 'value':
            self.joined_electricity['price'] = self.joined_electricity['price'] + fixed_price  # €/Wh
        elif type_ == 'fraction':
            self.joined_electricity['price'] = self.joined_electricity['price'] + \
                                               self.joined_electricity['price'] * fixed_price  # €/Wh

    def get_battery_results(self, start: datetime.datetime, end: datetime.datetime, unit='Wh', method=''):
        """

        :param start:
        :param end:
        :param unit: the unit in which the data in self.joined_electricity and self.battery is given (should be the same!)
        :param method: method to use for calculation of battery results (battery_state uses battery states to determine
        when battery is (dis)charging and requires pv and demand data as well; all else uses only battery power and
        assumptions on when (dis)charging is taking place).
        
        :return:
        """
        # Don't take into account anything between -200 and 0, as this is battery loss (and unclear, because
        # the sign (i.e. the same as when charging with grid) indicates charging...)
        year = start.year
        joined_data = self.joined_electricity[start:end].copy()
        second_frequency = pd.to_timedelta(joined_data.index.freq).seconds

        if 'k' in unit:
            factor = 1 / 1000.0
        else:
            factor = 1.0

        if method == 'battery_state' and 'battery_state' in self.joined_electricity.columns:
            print('Using battery states')
            charging = joined_data['battery'].where(joined_data['battery_state'] == 2).dropna().resample(
                'M').sum() * factor
            discharging = joined_data['battery'].where(
                joined_data['battery_state'] == 1).dropna().resample(
                'M').sum() * factor
            df_charging_with_grid = joined_data.apply(
                lambda joined_data: self._get_battery_storage_grid_from_data(joined_data['pv'], joined_data['battery'],
                                                                             joined_data['demand'],
                                                                             joined_data['battery_state']),
                axis=1)
            charging_w_grid = df_charging_with_grid.dropna().resample('M').sum() * factor
        else:
            print('Using assumptions for battery results')
            battery_data = self.battery[year][start:end].copy()
            if unit in ['W', 'kW']:
                self_discharge = 200 * factor
                grid_charge = 40000 * factor
                discharge = 0.0 * factor
            elif unit in ['Wh', 'kWh']:
                self_discharge = self._convert_value_to_wh(200, second_frequency) * factor
                grid_charge = self._convert_value_to_wh(40000, second_frequency) * factor
                discharge = self._convert_value_to_wh(0.0, second_frequency) * factor

            charging = battery_data['battery'].where(battery_data['battery'] > self_discharge).dropna().resample(
                'M').sum() * factor
            charging_w_grid = battery_data['battery'].where(battery_data['battery'] > grid_charge).dropna().resample(
                'M').sum() * factor
            discharging = battery_data['battery'].where(battery_data['battery'] < discharge).dropna().resample(
                'M').sum() * factor

        if unit in ['W', 'kW']:
            print(f'Converting to {unit}h...')
            charging = self._convert_value_to_wh(charging, second_frequency)
            charging_w_grid = self._convert_value_to_wh(charging_w_grid, second_frequency)
            discharging = self._convert_value_to_wh(discharging, second_frequency)

        return {'charging': charging, 'charging with grid': charging_w_grid, 'discharging': discharging}

    def calculate_grid_use(self):
        """
        This always includes the energy for charging the battery with the grid!

        :return: None
        """
        total = self.joined_electricity
        self.joined_electricity[f'grid use'] = total.apply(
            lambda total: self._calculate_grid_use(total['pv'], total['battery'], total['demand'],
                                                   total['battery_state_of_charge']), axis=1)

    def calculate_injection(self, relative=True):
        total = self.joined_electricity
        self.joined_electricity[f'injection{"_rel" if relative else "_abs"}'] = total.apply(
            lambda total: self._calculate_grid_injection(total['pv'], total['battery'], total['demand'],
                                                         total['battery_state_of_charge'], relative), axis=1)

    def calculate_self_consumption(self, relative=True):
        """
        According to: https://nl.wikibooks.org/wiki/Energietransitie/Eigen_productie
        De zelfconsumptiegraad Zc bepaalt hoeveel van je eigen productie je onmiddellijk zelf kan verbruiken.
        Een Zc van 100 % betekent dat je nooit injecteert in het elektriciteitsnetwerk.
        Aanname: enkel PV (dus geen batterij); per tijdstip: als vraag > pv -> Zc = 1, anders Zc = vraag / pv
        """
        total = self.joined_electricity
        self.joined_electricity[f'selfconsumption{"_rel" if relative else "_abs"}'] = total.apply(
            lambda total: self._calculate_self_consumption(total['pv'], total['demand'], relative), axis=1)

    def calculate_self_sufficiency(self, relative):
        """
        According to: https://nl.wikibooks.org/wiki/Energietransitie/Eigen_productie
        De zelfvoorzieningsgraad Zv bepaalt hoeveel je van het net nog moet afnemen.
        Een Zv van 100 % betekent dat je nooit iets afneemt van het elektriciteitsnetwerk.
        include_battery: whether or not to include battery charging with grid in the total grid use when calculating
            self sufficiency
        """
        total = self.joined_electricity
        if 'grid use' not in total.columns:
            self.calculate_grid_use()
        if 'demand with battery' not in total.columns:
            self.get_battery_storage_grid()
        self.joined_electricity[f'selfsufficiency{"_rel" if relative else "_abs"}'] = total.apply(
            lambda total: self._calculate_self_sufficiency(total['grid use'],
                                                           total['demand with battery'], relative), axis=1)

    def get_electricity_fractions(self):
        total = self.joined_electricity
        self.joined_electricity['provided by pv'] = total.apply(
            lambda total: self._calculate_self_consumption(total['pv'], total['demand'], relative=False), axis=1)
        self.joined_electricity['fraction from pv'] = total.apply(
            lambda total: self._get_fraction(total['provided by pv'], total['demand']), axis=1)
        self.joined_electricity['provided by battery'] = total.apply(
            lambda total: self._get_battery_energy_savings_from_calculation(total['pv'], total['battery'],
                                                                            total['battery_state'], total['demand']), axis=1)
        self.joined_electricity['fraction from battery'] = total.apply(
            lambda total: self._get_fraction(total['provided by battery'], total['demand']), axis=1)
        self.joined_electricity['fraction from grid'] = 1 - \
                                                        self.joined_electricity['fraction from pv'] - \
                                                        self.joined_electricity['fraction from battery']

    def get_battery_storage_solar(self):
        total = self.joined_electricity
        self.joined_electricity['battery stored solar'] = total.apply(
            lambda total: self._get_battery_storage_solar(total['pv'], total['battery'], total['demand']), axis=1)

    def get_battery_storage_grid(self):
        total = self.joined_electricity
        self.joined_electricity['battery stored grid'] = total.apply(
            lambda total: self._get_battery_storage_grid_from_data(total['pv'], total['battery'], total['demand']),
            axis=1)
        self.joined_electricity['demand with battery'] = self.joined_electricity['demand'] + \
                                                         self.joined_electricity['battery stored grid']

    def get_total_costs(self):
        total = self.joined_electricity
        self.joined_electricity['battery charging costs'] = total.apply(
            lambda total: self._get_battery_charging_costs(total['pv'], total['battery'], total['demand'],
                                                           total['price']), axis=1)

        self.joined_electricity['demand costs'] = total.apply(
            lambda total: self._get_demand_costs_from_data(total['pv'], total['battery'],
                                                           total['demand'], total['price']), axis=1)

        self.joined_electricity['actual price'] = self.joined_electricity['demand costs'] / total['demand']

        self.joined_electricity['demand costs no ems'] = total['demand'] * total['price']

        self.joined_electricity['total costs'] = self.joined_electricity['demand costs'] + \
                                                 self.joined_electricity['battery charging costs']

    def get_total_savings(self, battery_costs_method: str = 'calculated'):
        total = self.joined_electricity
        self.joined_electricity['avoided costs pv'] = total.apply(
            lambda total: self._get_pv_savings(total['pv'], total['demand'], total['price']), axis=1)

        if battery_costs_method == 'calculated':
            self.joined_electricity['avoided costs battery'] = total.apply(
                lambda total: self._get_battery_financial_savings_from_calculation(total['pv'], total['battery'],
                                                                                   total['battery_state'],
                                                                                   total['demand'], total['price']), axis=1)
        elif battery_costs_method == 'data':
            self.joined_electricity['avoided costs battery'] = total.apply(
                lambda total: self._get_battery_financial_savings_from_data(total['battery'], total['price']), axis=1)
        else:
            raise Exception('battery_cost_method not known!')

        self.joined_electricity['total avoided costs'] = self.joined_electricity['avoided costs pv'] + \
                                                         self.joined_electricity['avoided costs battery']

    def get_total_emission_savings(self, emission_factor_pv, emission_factor_battery,
                                   emission_factor_grid, battery_costs_method: str = 'data'):
        """
        :param emission_factor_relative_to_pv: emission factor of grid electricity compared to pv,
        i.e. factor_grid - factor_pv
        :param emission_factor_relativeto_battery: emission factor of grid electricity compared to battery,
        i.e. factor_grid - factor_pv
        :param battery_costs_method: use either actual data, or the assumed operation state (also data) of the battery
        to make battery savings calculations
        :return:
        """
        total = self.joined_electricity
        self.joined_electricity['emission factor'] = self.joined_electricity['fraction from pv'] * emission_factor_pv + \
                                                     self.joined_electricity['fraction from battery'] * emission_factor_battery + \
                                                     self.joined_electricity['fraction from grid'] * emission_factor_grid

        self.joined_electricity['emissions pv'] = total.apply(
            lambda total: self._get_pv_emission_savings(total['pv'], total['demand'], emission_factor_pv),
            axis=1)
        self.joined_electricity['avoided emissions pv'] = total.apply(
            lambda total: self._get_pv_emission_savings(total['pv'], total['demand'], emission_factor_grid),
            axis=1)

        if battery_costs_method == 'calculated':
            self.joined_electricity['emissions battery'] = total.apply(
                lambda total: self._get_battery_emission_from_calculation(total['pv'], total['battery'],
                                                                                  total['battery_state'],
                                                                                  total['demand'],
                                                                                  emission_factor_grid),
                axis=1)
            self.joined_electricity['avoided emissions battery'] = total.apply(
                lambda total: self._get_battery_emission_savings_from_calculation(total['pv'], total['battery'],
                                                                                  total['battery_state'],
                                                                                  total['demand'],
                                                                                  emission_factor_grid,
                                                                                  emission_factor_battery),
                axis=1)
        elif battery_costs_method == 'data':
            self.joined_electricity['emissions battery'] = total.apply(
                lambda total: self._get_battery_emission_from_data(total['pv'], total['battery'], total['demand'],
                                                                   total['battery_state'],
                                                                   emission_factor_grid),
                axis=1)
            self.joined_electricity['avoided emissions battery'] = total.apply(
                lambda total: self._get_battery_emission_savings_from_data(total['battery'], emission_factor_grid,
                                                                           emission_factor_battery),
                axis=1)
        else:
            raise Exception('battery_cost_method not known!')

        self.joined_electricity['emissions heat pump'] = self.joined_electricity['heatpump'] * \
                                                        self.joined_electricity['emission factor']

        self.joined_electricity['total reference emissions'] = self.joined_electricity['emissions pv'] + \
                                                               self.joined_electricity['emissions battery']
        self.joined_electricity['total avoided emissions'] = self.joined_electricity['avoided emissions pv'] + \
                                                             self.joined_electricity['avoided emissions battery']

    def get_monthly_electricity_sums(self):
        self.monthly_electricity = pd.DataFrame()
        sums = ['pv', 'demand', 'demand with battery', 'battery stored solar', 'battery stored grid', 'total costs',
                'battery charging costs', 'demand costs', 'total avoided costs', 'avoided costs battery', 'avoided costs pv',
                'avoided emissions battery', 'avoided emissions pv', 'total avoided emissions', 'evcharging']
        for name in sums:
            try:
                self.monthly_electricity[name] = self.joined_electricity[name].resample('MS').sum()
            except KeyError:
                print(f"No value for '{name}' calculated yet!")

    def get_monthly_average_selfconsumption(self):
        means = ['selfconsumption_rel', 'selfsufficiency_rel']
        for name in means:
            try:
                self.monthly_electricity[name] = self.joined_electricity[name].resample('MS').mean() * 100
            except KeyError:
                print(f"No value for '{name}' calculated yet!")

    def get_monthly_selfconsumption(self):
        sums = ['injection', 'selfconsumption', 'selfsufficiency']
        if 'injection_abs' not in self.joined_electricity.columns:
            self.calculate_injection(relative=False)
        if 'selfconsumption_abs' not in self.joined_electricity.columns:
            self.calculate_self_consumption(relative=False)
        if 'selfsufficiency_abs' not in self.joined_electricity.columns:
            self.calculate_self_sufficiency(relative=False)

        for name in sums:
            self.monthly_electricity[name+'_cumul'] = self.joined_electricity[name+'_abs'].resample('MS').sum()
        self.monthly_electricity['grid use_cumul'] = self.joined_electricity['grid use'].resample('MS').sum()

        self.monthly_electricity['selfconsumption'] = self.monthly_electricity['selfconsumption_cumul'] / \
                                                      self.monthly_electricity['pv'] * 100
        self.monthly_electricity['selfsufficiency'] = (1 - self.monthly_electricity['grid use_cumul'] /
                                                      self.monthly_electricity['demand with battery']) * 100

    def plot_battery_storage(self):
        days = 12
        width = datetime.timedelta(hours=24 * days)

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(self.monthly_electricity['pv'].index - width / 2,
               self.monthly_electricity['pv'] / 1000, [days] * len(self.monthly_electricity['pv']),
               edgecolor='k', label='PV')
        ax.bar(self.monthly_electricity['battery stored solar'].index,
               self.monthly_electricity['battery stored solar'] / 1000, [days] * len(self.monthly_electricity['battery stored solar']),
               edgecolor='k', label='Battery storage (solar)')
        ax.bar(self.monthly_electricity['battery stored grid'].index,
               self.monthly_electricity['battery stored grid'] / 1000, [days] * len(self.monthly_electricity['battery stored grid']),
               bottom=self.monthly_electricity['battery stored solar'] / 1000,
               edgecolor='k', label='Battery storage (grid)')
        ax.set_xlim(self.monthly_electricity.index[0] - width, self.monthly_electricity.index[-1])
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        ax.set_ylim(0, )
        ax.set_title('Energy per month [kWh]', fontsize=15)
        sns.despine()
        return fig, ax

    def plot_self_consumption(self, xlim: tuple, relative=True):
        days = 12
        width = datetime.timedelta(hours=24 * days)
        data_to_use = self.monthly_electricity[xlim[0]:xlim[1]].copy()

        fig, ax = plt.subplots(2, 1, figsize=(15, 8))
        ax[0].bar(data_to_use[f'selfconsumption'].index + width / 4,
               data_to_use[f'selfconsumption'],
               [days] * len(data_to_use[f'selfconsumption']),
               facecolor=sns.color_palette()[0], edgecolor='k', label='Self consumption')

        ax[0].bar(data_to_use[f'selfsufficiency'].index - width / 4,
               data_to_use[f'selfsufficiency'],
               [days] * len(data_to_use[f'selfsufficiency']),
               facecolor=sns.color_palette()[1], edgecolor='k',
               label='Self sufficiency')

        ax[0].set_xlim(data_to_use.index[0] - width, data_to_use.index[-1] + width)
        ax[0].tick_params(labelsize=12)
        ax[0].legend(fontsize=12)
        ax[0].set_ylim(0, 100.0)
        ax[0].set_title('Self consumption/sufficiency [%]', fontsize=15)

        ax[1].bar(data_to_use[f'demand'].index - width / 2,
                  data_to_use[f'demand'] / 1e6,
                  [days] * len(data_to_use[f'demand']),
                  facecolor=sns.color_palette()[0],
                  edgecolor='k', label='Total energy demand')
        ax[1].set_xlim(data_to_use.index[0] - width, data_to_use.index[-1] + width)
        ax[1].tick_params(labelsize=12)
        ax[1].set_title('Total energy demand [MWh]', fontsize=15)

        sns.despine()
        fig.tight_layout()
        return fig, ax

    def plot_costs(self, xlim: tuple):
        """
        Costs: covering demand and charging battery with grid
        Savings: demand covered by PV and battery
        """
        days = 12
        width = datetime.timedelta(hours=24 * days)
        data_to_use = self.monthly_electricity[xlim[0]:xlim[1]].copy()

        fig, ax = plt.subplots(2, 1, figsize=(15, 8))
        ax[0].bar(data_to_use[f'avoided costs battery'].index - width / 2,
                  data_to_use[f'avoided costs battery'],
                  [days] * len(data_to_use[f'avoided costs battery']),
                  edgecolor='k', label='Savings discharging battery')
        ax[0].bar(data_to_use['battery charging costs'].index,
                  data_to_use['battery charging costs'],
                  [days] * len(data_to_use['battery charging costs']),
                  edgecolor='k', label='Cost charging battery')
        ax[1].bar(data_to_use['total costs'].index,
                  data_to_use['total costs'], [days] * len(data_to_use['total costs']),
                  edgecolor='k', label='Actual')
        ax[1].bar(data_to_use['total avoided costs'].index,
                  data_to_use['total avoided costs'], [days] * len(data_to_use['total avoided costs']),
                  bottom=data_to_use['total costs'],
                  edgecolor='k', label='Avoided')

        ax_0_max = max(max(data_to_use[f'avoided costs battery']),
                       max(data_to_use['battery charging costs']))
        ax_1_max = max(data_to_use['total costs'] + data_to_use['total avoided costs'])

        for i in range(0, len(data_to_use['total costs'].index)):
            try:
                ax[1].text(data_to_use['total costs'].index[i] - width / 1.5,
                           data_to_use['total avoided costs'][i] + data_to_use['total costs'][i] + ax_1_max/100,
                           str(int(round(data_to_use['total avoided costs'][i] / (data_to_use['total avoided costs'][i] + data_to_use['total costs'][i]) * 100,
                                         0))) + '%',
                           fontsize=15)
            except:
                pass

        for a in ax:
            a.set_xlim(data_to_use.index[0] - width, data_to_use.index[-1] + width)
            a.tick_params(labelsize=12)
            a.legend(fontsize=12)
            a.set_ylim(0, )

        ax[0].set_title('Battery charging costs/discharging savings [€]', fontsize=15)
        cumulative_battery_avoided_costs = data_to_use['avoided costs battery'][xlim[0]:xlim[1]].sum()
        cumulative_battery_charging_costs = data_to_use['battery charging costs'][xlim[0]:xlim[1]].sum()
        cumulative_battery_savings = cumulative_battery_avoided_costs - cumulative_battery_charging_costs

        ax[0].text(data_to_use['total costs'].index[0] - width * 2,
                   -ax_0_max / 5,
                   f"Cumulative: {round(cumulative_battery_charging_costs)}€ spent (battery charging); "
                   f"{round(cumulative_battery_avoided_costs)}€ avoided (avoided grid use); "
                   f"{round(cumulative_battery_savings)}€ saved", fontsize=12)

        ax[1].set_title('Electricity costs [€] and savings [%] per month ', fontsize=15)
        cumulative_electricity_avoided_costs = data_to_use['total avoided costs'][xlim[0]:xlim[1]].sum()
        cumulative_electricity_costs = data_to_use['total costs'][xlim[0]:xlim[1]].sum()

        ax[1].text(data_to_use['total costs'].index[0] - width * 2,
                   -ax_1_max / 5,
                   f"Cumulative: {round(cumulative_electricity_costs)}€ spent (grid use); "
                   f"{round(cumulative_electricity_avoided_costs)}€ avoided",
                   fontsize=12)

        fig.tight_layout()
        sns.despine()

        return fig, ax

    def plot_emissions(self):
        days = 12
        width = datetime.timedelta(hours=24 * days)

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(self.monthly_electricity['avoided emissions pv'].index,
               self.monthly_electricity['avoided emissions pv'], [days] * len(self.monthly_electricity['avoided emissions pv']),
               facecolor=sns.color_palette()[0], edgecolor='k', label='Avoided emissions PV')
        ax.bar(self.monthly_electricity['avoided emissions battery'].index + width / 2,
               self.monthly_electricity['avoided emissions battery'],
               [days] * len(self.monthly_electricity['total avoided emissions']),
               bottom=self.monthly_electricity['avoided emissions pv'],
               facecolor=sns.color_palette()[1], edgecolor='k', label='Avoided emissions battery')

        ax.set_xlim(self.monthly_electricity.index[0], self.monthly_electricity.index[-1])
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        ax_min = min(self.monthly_electricity['avoided emissions pv'] +
                     self.monthly_electricity['avoided emissions battery'])
        ax.set_ylim(ax_min,
                    max(self.monthly_electricity['avoided emissions pv'] +
                        self.monthly_electricity['avoided emissions battery']))
        ax.set_title('Avoided emissions [gCO2eq]', fontsize=15)
        ax.text(self.monthly_electricity['total avoided emissions'].index[0] - width * 2,
                2 * ax_min,
                f"Cumulative: {round(self.monthly_electricity['total avoided emissions'].sum() / 1e6, 2)} ton CO2eq saved",
                fontsize=12)

        sns.despine()
        return fig, ax

    # %% HEAT
    def join_heat_series(self, years: List):
        """To be worked on!"""
        for year in years:
            try:
                self.joined_heat = pd.concat([self.joined_heat,
                                              pd.concat([self.price[year],
                                                         self.heat_demand[year],
                                                         self.heatpump[year]], axis=1)])
            except KeyError:
                print("Series could not be joined, there might be data missing. Expected data: price, heat_demand, heatpump")

    def add_heat_prices(self, heat_prices: dict, kind='gas', unit='€/kWh'):
        """

        :param heat_prices: dict containing the heat prices for the covered period in the format {year: {month: price}, {month:  price}}
        :param kind: kind of heat prices being added, gas or waste heat
        :return: None; update the self.joined dataframe with the prices
        """
        if unit == '€/kWh':
            factor = 1e-3
        elif unit == '€/Wh':
            factor = 1
        else:
            raise Exception(f"Unit {unit} not known!")
        self.joined_heat[f'price {kind}'] = [price * factor for price in self._get_monthly_price(self.joined_heat.index, heat_prices)]

    def get_wpo_on_when_wasteheat(self):
        def _get_wpo_on_when_wasteheat(wasteheat, heatdemand, wpoheat):
            if wasteheat > heatdemand and wpoheat > 10000.0:
                return True
            else:
                return False

    def split_chp_from_wasteheat(self, column_name_wasteheat_total, column_name_splitvalue):
        """
        column_name_wasteheat_total: column name in self.joined_heat that contains the total wasteheat
        column_name_splitvalue: column name in self.joined_heat that contains the values indicating what percentage of
        energy comes from waste heat and what from chp

        This assumes the split value is between 0 and 100, with 100 indicating that 100% of the value in
        column_name_wasteheat_total is coming from actual waste heat and the rest from chp heat
        """
        self.joined_heat['wasteheat'] = self.joined_heat[column_name_wasteheat_total] * self.joined_heat[column_name_splitvalue] / 100.0
        self.joined_heat['chp'] = self.joined_heat[column_name_wasteheat_total] * (100.0 - self.joined_heat[
            column_name_splitvalue]) / 100.0

    def get_heat_fractions(self, gas_fraction_method='complement'):
        """

        :param gas_fraction_method:
            'complement': gas fraction is equal to 100 - all other fractions;
            'data': gas fraction is calculated from data
        :return:
        """
        total = self.joined_heat
        self.joined_heat['fraction from external'] = total.apply(lambda total: self._get_fraction(total['total_external_heat'],
                                                                                                  total['heatdemand']), axis=1)
        self.joined_heat['fraction from waste heat'] = total.apply(lambda total: self._get_fraction(total['wasteheat'],
                                                                                                    total['total_external_heat']), axis=1) * \
                                                       self.joined_heat['fraction from external']
        self.joined_heat['fraction from chp'] = total.apply(lambda total: self._get_fraction(total['chp'],
                                                                                             total['total_external_heat']), axis=1) * \
                                                       self.joined_heat['fraction from external']
        self.joined_heat['fraction from heat pump'] = 1.0 - self.joined_heat['fraction from external']

        if gas_fraction_method == 'complement':
            self.joined_heat['fraction from gas'] = self.joined_heat['fraction from external'] - \
                                                    self.joined_heat['fraction from waste heat'] - \
                                                    self.joined_heat['fraction from chp']
        elif gas_fraction_method == 'data':
            self.joined_heat['fraction from gas'] = total.apply(lambda total: self._get_fraction(total['gasboiler'],
                                                                                                 total['heatdemand']),
                                                                axis=1)

    def get_heat_prices(self):
        """
        Requires self.get_total_costs to have run!
        :return: None
        """
        self.joined_heat['cop'] = self.joined_heat['Thermal power'] / self.joined_heat['Electric power']
        self.joined_heat['actual electricity price'] = self.joined_electricity['actual price']
        total = self.joined_heat
        self.joined_heat['price thermal heat pump'] = total.apply(
            lambda total: self._get_thermal_value(total['actual electricity price'], total['cop'], total['Thermal power']),
            axis=1)

    def get_total_heat_costs(self):
        self.joined_heat['heat pump cost'] = self.joined_heat['heatdemand'] * \
                                             self.joined_heat['fraction from heat pump'] * \
                                             self.joined_heat['price thermal heat pump']
        self.joined_heat['waste heat cost'] = self.joined_heat['heatdemand'] * \
                                              self.joined_heat['fraction from waste heat'] * \
                                              self.joined_heat['price waste heat']
        self.joined_heat['chp cost'] = self.joined_heat['heatdemand'] * \
                                              self.joined_heat['fraction from chp'] * \
                                              self.joined_heat['price chp']
        self.joined_heat['gas cost'] = self.joined_heat['heatdemand'] * \
                                       self.joined_heat['fraction from gas'] * \
                                       self.joined_heat['price gas']
        self.joined_heat['total costs'] = self.joined_heat['heat pump cost'] + self.joined_heat['waste heat cost'] + \
                                         self.joined_heat['gas cost']
        self.joined_heat['total reference cost'] = self.joined_heat['heatdemand'] * self.joined_heat['price gas']

    def get_total_heat_savings(self):
        self.joined_heat['avoided costs heat pump'] = self.joined_heat['heatdemand'] * \
                                                      self.joined_heat['fraction from heat pump'] * \
                                                (self.joined_heat['price gas'] - self.joined_heat['price thermal heat pump'])
        self.joined_heat['avoided costs waste heat'] = self.joined_heat['heatdemand'] * \
                                                       self.joined_heat['fraction from waste heat'] * \
                                                 (self.joined_heat['price gas'] - self.joined_heat['price waste heat'])
        self.joined_heat['avoided costs chp'] = self.joined_heat['heatdemand'] * \
                                                self.joined_heat['fraction from chp'] * \
                                                (self.joined_heat['price gas'] - self.joined_heat['price chp'])

        self.joined_heat['total avoided costs'] = self.joined_heat['avoided costs heat pump'] + \
                                                  self.joined_heat['avoided costs waste heat'] + \
                                                  self.joined_heat['avoided costs chp']

    def get_total_emission_savings_heat(self, emission_factor_wasteheat, emission_factor_chp,
                                        emission_factor_electricity, emission_factor_gas):
        """

        :param emission_factor_electricity:
        :param emission_factor_gas:
        :return:
        """
        total = self.joined_heat
        self.joined_heat['emissions waste heat'] = self.joined_heat['heatdemand'] * \
                                                   self.joined_heat['fraction from waste heat'] * \
                                                   emission_factor_wasteheat
        self.joined_heat['avoided emissions waste heat'] = self.joined_heat['heatdemand'] * \
                                                                  self.joined_heat['fraction from waste heat'] * \
                                                                  emission_factor_gas
        self.joined_heat['emissions chp'] = self.joined_heat['heatdemand'] * \
                                                   self.joined_heat['fraction from chp'] * \
                                                   emission_factor_chp
        self.joined_heat['avoided emissions chp'] = self.joined_heat['heatdemand'] * \
                                                           self.joined_heat['fraction from chp'] * \
                                                           emission_factor_gas
        heatpump_emission_factor = total.apply(
            lambda total: self._get_thermal_value(emission_factor_electricity, total['cop'], total['Thermal power']), axis=1)
        self.joined_heat['emissions heat pump'] = self.joined_heat['heatdemand'] * \
                                                  self.joined_heat['fraction from heat pump'] * \
                                                  heatpump_emission_factor
        self.joined_heat['avoided emissions heat pump'] = self.joined_heat['heatdemand'] * \
                                                          self.joined_heat['fraction from heat pump'] * \
                                                          (emission_factor_gas - heatpump_emission_factor)
        self.joined_heat['emissions gas'] = self.joined_heat['heatdemand'] * \
                                            self.joined_heat['fraction from gas'] * \
                                            emission_factor_gas
        self.joined_heat['total reference emissions'] = self.joined_heat['heatdemand'] * emission_factor_gas
        self.joined_heat['total avoided emissions'] = self.joined_heat['avoided emissions waste heat'] + \
                                                      self.joined_heat['avoided emissions chp'] + \
                                                      self.joined_heat['avoided emissions heat pump']

    def get_monthly_heat_averages(self):
        self.monthly_heat = pd.DataFrame()
        sums = ['total_external_heat', 'total_wasteheat', 'wasteheat', 'chp', 'heatdemand', 'heatpump',
                'heat pump cost', 'waste heat cost', 'chp cost', 'gas cost', 'total costs',
                'total reference cost', 'avoided costs heat pump', 'avoided costs waste heat', 'total avoided costs',
                'emissions waste heat', 'avoided emissions waste heat',
                'emissions chp', 'avoided emissions chp',
                'emissions heat pump', 'avoided emissions heat pump',
                'emissions gas', 'total reference emissions', 'total avoided emissions']
        means = ['fraction from external', 'fraction from waste heat', 'fraction from chp',
                 'fraction from heat pump', 'fraction from gas']
        for name in sums:
            try:
                self.monthly_heat[name] = self.joined_heat[name].resample('MS').sum()
            except KeyError:
                print(f"No value for '{name}' calculated yet!")
        for name in means:
            try:
                self.monthly_heat[name] = self.joined_heat[name].resample('MS').mean()
            except KeyError:
                print(f"No value for '{name}' calculated yet!")

    def plot_heat_fractions(self):
        days = 12
        width = datetime.timedelta(hours=24 * days)
        total_production = self.monthly_heat['total_external_heat'] + self.monthly_heat['heatpump']

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(self.monthly_heat[f'fraction from chp'].index,
               self.monthly_heat[f'fraction from chp'] * total_production / 1000.0,
               [days] * len(self.monthly_heat[f'fraction from chp']),
               edgecolor='k', label='CHP')
        ax.bar(self.monthly_heat[f'fraction from waste heat'].index,
               self.monthly_heat[f'fraction from waste heat'] * total_production / 1000.0,
               [days] * len(self.monthly_heat[f'fraction from waste heat']),
               bottom=self.monthly_heat[f'fraction from chp'] * total_production / 1000.0,
               edgecolor='k', label='Waste heat')
        ax.bar(self.monthly_heat['fraction from heat pump'].index,
               self.monthly_heat['fraction from heat pump'] * total_production / 1000.0,
               [days] * len(self.monthly_heat['fraction from heat pump']),
               bottom=(self.monthly_heat[f'fraction from waste heat'] + self.monthly_heat[f'fraction from chp']) *
                      total_production / 1000.0,
               edgecolor='k', label='Heat pump')
        ax.bar(self.monthly_heat['fraction from gas'].index,
               self.monthly_heat['fraction from gas'] * total_production / 1000.0,
               [days] * len(self.monthly_heat['fraction from gas']),
               bottom=(self.monthly_heat[f'fraction from waste heat'] + self.monthly_heat[f'fraction from chp'] +
                       self.monthly_heat[f'fraction from heat pump']) * total_production / 1000.0,
               edgecolor='k', label='Gas')

        ax.set_xlim(self.monthly_heat.index[0] - width, self.monthly_heat.index[-1] + width)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        ax.set_title('Heat production according to source [kWh]', fontsize=15)
        sns.despine()

        return fig, ax

    def plot_heat_costs(self, xlim: tuple):
        days = 12
        width = datetime.timedelta(hours=24 * days)
        data_to_use = self.monthly_heat[xlim[0]:xlim[1]].copy()

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(data_to_use['total reference cost'].index,
               data_to_use['total reference cost'],
               [days] * len(data_to_use['total reference cost']),
               color='white',
               edgecolor='k', label='Reference cost (all gas)')
        ax.bar(data_to_use['chp cost'].index,
               data_to_use['chp cost'],
               [days] * len(data_to_use['chp cost']),
               edgecolor='k', label='Cost CHP')
        ax.bar(data_to_use['waste heat cost'].index,
               data_to_use['waste heat cost'],
               [days] * len(data_to_use['waste heat cost']),
               bottom=data_to_use['chp cost'],
               edgecolor='k', label='Cost waste heat')
        ax.bar(data_to_use['heat pump cost'].index,
               data_to_use['heat pump cost'],
               [days] * len(data_to_use['heat pump cost']),
               bottom=data_to_use['chp cost'] + data_to_use['waste heat cost'],
               edgecolor='k', label='Cost heat pump')
        ax.bar(data_to_use['gas cost'].index,
               data_to_use['gas cost'],
               [days] * len(data_to_use['gas cost']),
               bottom=data_to_use['chp cost'] + data_to_use['waste heat cost'] + data_to_use['heat pump cost'],
               edgecolor='k', label='Cost gas')

        ax.set_xlim(data_to_use.index[0] - width, data_to_use.index[-1] + width)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)

        for i in range(0, len(data_to_use['total costs'].index)):
            try:
                ax.text(data_to_use['total reference cost'].index[i] - width / 1.5,
                           data_to_use['total reference cost'][i] + 500,
                           str(int(round(data_to_use['total avoided costs'][i] / (data_to_use['total avoided costs'][i] + data_to_use['total costs'][i]) * 100,
                                         0))) + '%',
                           fontsize=15)
            except:
                pass

        ax_max = max(data_to_use['total reference cost'])

        ax.set_ylim(0, ax_max + 0.1 * ax_max)
        ax.set_title('Heating costs [€] and savings [%]', fontsize=15)
        ax.text(data_to_use['total avoided costs'].index[0] - width * 2,
                -ax_max * 0.2,
                f"Cumulative: {round(data_to_use['total avoided costs'].sum())} € saved",
                fontsize=12)

        sns.despine()
        return fig, ax

    def plot_emissions_heat(self, xlim: tuple):
        days = 12
        width = datetime.timedelta(hours=24 * days)
        data_to_use = self.monthly_heat[xlim[0]:xlim[1]].copy()

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(data_to_use['total reference emissions'].index,
               data_to_use['total reference emissions'],
               [days] * len(data_to_use['total reference emissions']),
               color='white',
               edgecolor='k', label='Reference emissions (all gas)')
        ax.bar(data_to_use['emissions gas'].index,
               data_to_use['emissions gas'],
               [days] * len(data_to_use['emissions gas']),
               facecolor=sns.color_palette()[0], edgecolor='k', label='Emissions gas')
        ax.bar(data_to_use['emissions waste heat'].index,
               data_to_use['emissions waste heat'],
               [days] * len(data_to_use['emissions waste heat']),
               bottom=data_to_use['emissions gas'],
               facecolor=sns.color_palette()[1], edgecolor='k', label='Emissions waste heat')
        ax.bar(data_to_use['emissions chp'].index,
               data_to_use['emissions chp'],
               [days] * len(data_to_use['emissions chp']),
               bottom=data_to_use['emissions gas'] + data_to_use['emissions waste heat'],
               facecolor=sns.color_palette()[2], edgecolor='k', label='Emissions chp')
        ax.bar(data_to_use['emissions heat pump'].index,
               data_to_use['emissions heat pump'],
               [days] * len(data_to_use['emissions heat pump']),
               bottom=data_to_use['emissions gas'] + data_to_use['emissions waste heat'] + data_to_use['emissions chp'],
               facecolor=sns.color_palette()[3], edgecolor='k', label='Emissions heat pump')

        ax.set_xlim(data_to_use.index[0], data_to_use.index[-1] + width)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        max_value = max(data_to_use['total reference emissions'])
        ax.set_ylim(0, max_value)
        ax.set_title('Avoided emissions [gCO2eq]', fontsize=15)
        ax.text(data_to_use['total avoided costs'].index[0] - width * 2,
                -0.1*max_value,
                f"Cumulative: {round(data_to_use['total avoided emissions'].sum() / 1e6, 2)} ton CO2eq saved",
                fontsize=12)

        sns.despine()
        return fig, ax

    # %% GENERAL
    def compare_with_previous_quarter(self, energy_type, measurement, quarter='Q1'):
        if energy_type == 'electricity':
            monthly_data = self.monthly_electricity.copy()
        elif energy_type == 'heat':
            monthly_data = self.monthly_heat.copy()
        else:
            raise Exception(f"Energy type {energy_type} not known!")
        quarters = {'Q1': [datetime.datetime(self.current_year, 1, 1), datetime.datetime(self.current_year, 3, 31)],
                    'Q2': [datetime.datetime(self.current_year, 4, 1), datetime.datetime(self.current_year, 6, 30)],
                    'Q3': [datetime.datetime(self.current_year, 7, 1), datetime.datetime(self.current_year, 9, 30)],
                    'Q4': [datetime.datetime(self.current_year, 10, 1), datetime.datetime(self.current_year, 12, 31)]}
        previous_quarters = {
            'Q1': [datetime.datetime(self.current_year - 1, 1, 1), datetime.datetime(self.current_year - 1, 3, 31)],
            'Q2': [datetime.datetime(self.current_year - 1, 4, 1), datetime.datetime(self.current_year - 1, 6, 30)],
            'Q3': [datetime.datetime(self.current_year - 1, 7, 1), datetime.datetime(self.current_year - 1, 9, 30)],
            'Q4': [datetime.datetime(self.current_year - 1, 10, 1), datetime.datetime(self.current_year - 1, 12, 31)]}

        previous_Q = monthly_data[measurement][previous_quarters[quarter][0]:previous_quarters[quarter][1]]
        previous_Q.index = [i.month for i in previous_Q.index]
        previous_Q.name = 'Previous quarter'

        current_Q = monthly_data[measurement][quarters[quarter][0]:quarters[quarter][1]]
        current_Q.index = [i.month for i in current_Q.index]
        current_Q.name = 'Current quarter'

        summary = pd.concat([current_Q, previous_Q], axis=1)
        summary['Change [%]'] = (summary['Current quarter'] - summary['Previous quarter'])/ summary['Previous quarter'] * 100.0

        return summary

    def compare_with_previous_year(self, energy_type, measurement):
        if energy_type == 'electricity':
            monthly_data = self.monthly_electricity.copy()
        elif energy_type == 'heat':
            monthly_data = self.monthly_heat.copy()
        else:
            raise Exception(f"Energy type {energy_type} not known!")
        current_year_daterange = [datetime.datetime(self.current_year, 1, 1),
                                  datetime.datetime(self.current_year, 12, 31)]
        previous_year_daterange = [datetime.datetime(self.current_year - 1, 1, 1),
                                   datetime.datetime(self.current_year - 1, 12, 31)]

        previous_year = monthly_data[measurement][previous_year_daterange[0]:previous_year_daterange[1]]
        previous_year.index = [i.month for i in previous_year.index]
        previous_year.name = 'Previous year'

        current_year = monthly_data[measurement][current_year_daterange[0]:current_year_daterange[1]]
        current_year.index = [i.month for i in current_year.index]
        current_year.name = 'Current year'

        summary = pd.concat([current_year, previous_year], axis=1)
        summary['Change [%]'] = (summary['Current year'] - summary['Previous year'])/ summary['Previous year'] * 100.0

        
        return summary
    
    
    

from dataanalyser import DataAnalyser


analyser = DataAnalyser(year=2023)
analyser.get_data(battery=True, pv=True, electricity_demand=True)
analyser.join_electricity_series([2023])
analyser.calculate_grid_use()
# ... more analysis steps

