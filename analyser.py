import math


class Analyser(object):
    def __init__(self):
        pass

    def _calculate_grid_use(self, pv, battery, demand, battery_soc):
        if battery >= 0 or battery_soc >= 99.5:  # charging/idle OR battery is full and is not consuming
            not_covered = max(0.0, demand - pv)
        else:
            not_covered = max(0.0, demand - pv - abs(battery))

        battery_charging_grid = self._get_battery_storage_grid_from_data(pv, battery, demand)
        return not_covered + battery_charging_grid

    @staticmethod
    def _calculate_grid_injection(pv, battery, demand, battery_soc, relative=True):
        if battery <= 0:
            injecting = max(0.0, pv - demand - abs(battery))
        else:
            if battery_soc >= 99.5:
                injecting = max(0.0, pv - demand)
            else: # charging
                injecting = max(0.0, pv - demand - battery)
        return injecting / demand if relative else injecting

    @staticmethod
    def _calculate_self_consumption(pv, demand, relative=True):
        if relative:
            try:
                return 1.0 if demand > pv else demand/pv
            except ZeroDivisionError:
                return 1.0
        else:
            return pv if demand > pv else demand

    @staticmethod
    def _calculate_self_sufficiency(grid_use, demand, relative=True):
        """
        The absolute value is here the total amount of energy that is provided by internal sources (could also be energy
        from the battery, even if it comes from the grid; this is compensated for at other times when grid charging is
        taken into account.

        :param grid_use:
        :param demand:
        :param relative:
        :return:
        """
        if relative:
            return (demand - grid_use) / demand
        else:
            return demand - grid_use

    @staticmethod
    def _get_battery_storage_solar(pv, battery, demand):
        if pv > demand and battery > 0.0:
            return pv - demand
        else:
            return 0.0

    @staticmethod
    def _get_battery_storage_grid_from_data(pv, battery, demand, battery_state=None):
        if not battery_state or battery_state == 'nan':
            battery_charging = battery > 0.0
        else:
            battery_charging = battery_state == 2

        if battery_charging:
            if pv > demand:  # zal in de praktijk neerkomen op battery die oplaadt met zonne-energie die over is
                pv_charging_power = pv - demand
                grid_charging_power = max(0.0, battery - pv_charging_power)  # wanneer het vermogen waarmee de batterij oplaadt groter is dan wat verwacht wordt op basis van enkel pv, dat wordt er opgeladen met de grid ook.
            else:  # wanneer de batterij opgeladen wordt, maar de geproduceerde pv energie overstijgt de vraag niet, dan wordt er opgeladen met de grid
                grid_charging_power = battery
            return grid_charging_power
        else:
            return 0.0

    @staticmethod
    def _get_battery_storage_grid_from_calculation(pv, battery, battery_state, demand):
        if isinstance(battery_state, float):
            battery_state_ = 'charging' if battery_state == 2.0 else ''

        if battery_state == 'nan':
            battery_state_ = 'charging' if battery > 0.0 else None

        if battery_state_ == 'charging':
            if pv > demand:  # zal in de praktijk neerkomen op battery die oplaadt met zonne-energie die over is
                pv_charging_power = pv - demand
                grid_charging_power = max(0.0,
                                          battery - pv_charging_power)  # wanneer het vermogen waarmee de batterij oplaadt groter is dan wat verwacht wordt op basis van enkel pv, dat wordt er opgeladen met de grid ook.
            else:  # wanneer de batterij opgeladen wordt, maar de geproduceerde pv energie overstijgt de vraag niet, dan wordt er opgeladen met de grid
                grid_charging_power = battery
            return grid_charging_power
        else:
            return 0.0

    @staticmethod
    def _get_injected_power(pv, demand, battery_soc=None, price=None):
        if price is None:
            price = 1.0

        if battery_soc:
            if battery_soc >= 100.0:
                return max(0.0, pv - demand) * price
            else:
                return 0.0

    @staticmethod
    def _get_demand_from_data(pv, battery, demand):
        if battery > 0.0:
            not_covered = max(0.0, demand - pv)
        else:
            not_covered = max(0.0, demand - pv - abs(battery))
        return not_covered

    @staticmethod
    def _get_demand_costs_from_data(pv, battery, demand, price):
        if battery > 0.0:
            not_covered = max(0.0, demand - pv)
        else:
            not_covered = max(0.0, demand - pv - abs(battery))
        return price * not_covered

    @staticmethod
    def _get_demand_costs_from_calculation(pv, battery, battery_state, demand, price):
        if isinstance(battery_state, float):
            battery_state = 'charging' if battery_state == 2.0 else ''

        if battery_state == 'nan':
            battery_state = 'charging' if battery > 0.0 else None

        if battery_state == 'charging' or battery > 0.0:
            not_covered = max(0.0, demand - pv)
        else:
            not_covered = max(0.0, demand - pv - abs(battery))
        return price * not_covered

    @staticmethod
    def _get_thermal_value(el_price, cop, thermal_power):
        return el_price / cop if thermal_power > 0.0 else 0.0  # price (and so cost) becomes zero when no heat energy is delivered

    def _get_battery_charging_energy(self, pv, battery, demand):
        return self._get_battery_storage_grid_from_data(pv, battery, demand)

    def _get_battery_charging_costs(self, pv, battery, demand, price):
        return self._get_battery_storage_grid_from_data(pv, battery, demand) * price

    @staticmethod
    def _get_pv_savings(pv, demand, price):
        if pv > demand:
            covered = demand
        else:
            covered = pv
        return price * covered

    @staticmethod
    def _get_pv_emissions(pv, demand, emission_factor_pv):
        if pv > demand:
            covered = demand
        else:
            covered = pv
        return emission_factor_pv * covered

    @staticmethod
    def _get_pv_emission_savings(pv, demand, emission_factor_grid):
        if pv > demand:
            covered = demand
        else:
            covered = pv
        return emission_factor_grid * covered

    def _get_battery_financial_savings_from_data(self, battery, price):
        return self._get_battery_energy_savings_from_data(battery) * price

    def _get_battery_financial_savings_from_calculation(self, pv, battery, battery_state, demand, price):
        return self._get_battery_energy_savings_from_calculation(battery, pv, battery_state, demand) * price

    def _get_battery_emission_savings_from_data(self, battery, emission_factor_grid, emission_factor_battery):
        return self._get_battery_energy_savings_from_data(battery) * \
               (emission_factor_grid - emission_factor_battery)

    def _get_battery_emission_from_data(self, pv, battery, demand, battery_state, emission_factor_grid):
       return self._get_battery_storage_grid_from_data(pv, battery, demand, battery_state) * emission_factor_grid

    def _get_battery_emission_savings_from_calculation(self, pv, battery, battery_state, demand,
                                                       emission_factor_grid, emission_factor_battery):
        return self._get_battery_energy_savings_from_calculation(pv, battery, battery_state, demand) * \
               (emission_factor_grid - emission_factor_battery)

    def _get_battery_emission_from_calculation(self, pv, battery, battery_state, demand, emission_factor_grid):
        return self._get_battery_storage_grid_from_calculation(pv, battery, battery_state, demand) * emission_factor_grid


    @staticmethod
    def _get_battery_energy_savings_from_data(battery):
        """
        :param battery:
        :return:
        """
        if battery < 0.0:  # discharging
            return abs(battery)
        else:
            return 0.0  # battery not discharging, so no savings

    @staticmethod
    def _get_battery_energy_savings_from_calculation(pv, battery, battery_state, demand):
        if isinstance(battery_state, float):
            battery_state_ = 'charging' if battery_state == 2.0 else ''

        if battery_state == 'nan':
            battery_state_ = 'charging' if battery > 0.0 else None

        if battery_state_ == 'charging' or battery > 0.0:  # charging
            covered = 0.0
        else:
            covered = min(max(0.0, demand - pv), abs(battery))
            # don't take just battery, as this value is apparently sometimes a lot higher than the actual demand
            # There can be situations where the total load is covered by the pv and the battery still is
            # discharging: don't take into account battery savings then! (covered = 0.0)
        return covered

    @staticmethod
    def _get_monthly_price(datetime_index, price_list):
        return [price_list[month][date] for month, date in zip(datetime_index.year, datetime_index.month)]

    @staticmethod
    def _get_fraction(part, total):
        if not part or not total:
            return 0.0
        try:
            return min(1.0, part / total)
        except ZeroDivisionError:
            return 0.0






