from dataanalyser import DataAnalyser


analyser = DataAnalyser(year=2023)
analyser.get_data(battery=True, pv=True, electricity_demand=True)
analyser.join_electricity_series([2023])
analyser.calculate_grid_use()
# ... more analysis steps

