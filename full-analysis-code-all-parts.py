# COOK COUNTY PENSION FUND SOLVENCY ANALYSIS - ALL PARTS
#------------------------------------------------------#

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SCENARIO 1: base return of 6% annually
# without any uncertainty
# expected return without uncertainty
base_return = 0.06

# import data from excel file
df = pd.read_excel('[ENTER PATH FOR DATA]', sheet_name='import-data', index_col='Year')


# extract total deductions
total_deductions = df['Total Deductions'].tolist()
# extract growth rates
# we want to ignore the first values
# we can achieve this by only iterating over the list items from index 1 to end
deductions_gr = [((deduction - total_deductions[i - 1]) / total_deductions[i - 1]) for i, deduction in enumerate(total_deductions[1:])]
# get average growth rate
avg_deductions_gr = np.average(deductions_gr)
# extract the 2016 deduction
start_deduction = total_deductions[-1]

# extract the average growth rate for additions
# there was a supplemental addition in 2016
# we exclude that for purposes of calculating average growth rate
total_additions = df['Additions - Ex. Sup.'].tolist()
# extract growth rates
additions_gr = [(addition - total_additions[i - 1]) / total_additions[i - 1] for i, addition in enumerate(total_additions[1:])]
# get average growth rate
avg_additions_gr = np.average(additions_gr)
# get the 2016 addition
starting_addition = total_additions[-1]

# import asset data to predict solvency
df2 = pd.read_excel('[ENTER PATH FOR DATA]', sheet_name='asset-data', index_col='Year')
# find average growth rate
total_assets = df2['Total Assets'].tolist()
# 2016 ending assets
starting_assets = total_assets[-1]

# CALCULATE THE TIME UNTIL ASSETS RUN OUT
#---------------------------------------#

# create function to simulate the solvency of the portfolio
def simulate_solvency(starting_assets, avg_deductions_gr, avg_additions_gr):
    # initialize variables
    available_assets = starting_assets
    curr_deduction = start_deduction
    curr_addition = starting_addition
    #results = results = pd.DataFrame(columns=['Total Assets', 'Additions', 'Deductions'])
    # create empty list to hold the dictionaries
    results_data = []
    # create period - year - incrementor
    solvent_years = 0

    # set up the loop
    # this while loop will execute until the assets are depleted
    while available_assets >= 0:
        # calculate current year's deduction and addition based on growth rates
        period_deduction = curr_deduction * (1 + avg_deductions_gr)
        # update the curr_deduction variable to reflect the growth
        curr_deduction = period_deduction
        # increase the addition based on growth rate
        period_addition = curr_addition * (1 + avg_additions_gr)
        # increase the addition variable
        curr_addition = period_addition
        # update available assets
        # increase the available assets by the given growth rate
        # subtract the deduction and add addition
        available_assets = (available_assets * (1 + base_return)) + period_addition - period_deduction
        # increment solvent year
        solvent_years += 1
        # append results to the dataframe
        # have to set index to ignore so that it can be properly appended
        #results = results.append({'Total Assets': available_assets, 'Additions': curr_addition, 'Deductions': curr_deduction}, ignore_index=True)
        # create a new row we want to add to dataframe
        # dataframe.append() has been depreciated so we need to first create a list
        # we can then convert the list of dictionaries to a df
        row = {'Total Assets': available_assets, 'Additions': curr_addition, 'Deductions': curr_deduction}
        results_data.append(row)
        # ass long as the keys have the same values, pandas will be able to convert it
        results = pd.DataFrame(results_data)
    # return the solvent_years variable as well as results df
    return [solvent_years, results]

# call the function to make calculations
results = simulate_solvency(starting_assets, avg_deductions_gr, avg_additions_gr)

# destruct the list
total_years = results[0]
fund_data = results[1]

# insert the intial values to the top of the df
# to do this, we can use concat function
# create new dataframe of starting values
# by concating the two dfs this will prepend them
# we need to pass in the single values as a list
# this is because df expect list like objects to be passed in
starting_values = pd.DataFrame({'Total Assets': [starting_assets], 'Additions': [starting_addition], 'Deductions': [start_deduction]})
# concat the starting values with the dataframe
fund_data = pd.concat([starting_values, fund_data], ignore_index=True)

# PLOT THE RESULTS
#----------------#

# extract the relevant columns
years = fund_data.index
assets = fund_data['Total Assets']
additions = fund_data['Additions']
deductions = fund_data['Deductions']

# create a figure and plot
plt.figure(figsize=(10, 6))
plt.plot(years, assets, label='Total Assets', color='gray')
plt.plot(years, additions, label='Additions', color='blue')
plt.plot(years, deductions, label='Deductions', color='purple')
plt.title(' Fund Solvency Simulation', fontsize=14)
plt.xlabel('Years', fontsize=14)
plt.ylabel('Amount', fontsize=14)
plt.xlim(left=0, right=total_years)
plt.legend(fontsize='large')
plt.grid()
plt.show()

# INTRODUCING UNCERTAINTY
#-----------------------#

# import the allocation data
allocation_data = pd.read_excel('[ENTER PATH FOR DATA]', sheet_name='fund-comp-data', index_col='Year')
# extract the weights for 2016
# last val in each column
equity_weight = allocation_data['Equities - %'].iloc[-1]
gov_bonds_weight = allocation_data['Government Obligations - %'].iloc[-1]
alts_weight = allocation_data['Alternative Investments - %'].iloc[-1]
quarterly_tbill_weight = allocation_data['Short Term Securities - %'].iloc[-1]
fixed_income_weight = allocation_data['Fixed Income and Corporate Bonds - %'].iloc[-1]
total_weights_check = equity_weight + gov_bonds_weight + alts_weight + quarterly_tbill_weight + fixed_income_weight
print(total_weights_check)
# calculate the standard deviations and covariance
# import data and extract based on asset class
equities = pd.read_excel('[ENTER PATH FOR DATA]', sheet_name='Exhibit 10', index_col='Date')
# extract short term gov debt
short_term_gov_debt = equities['3 month T-Bills']
# drop short term gov debt from equites df
equities = equities['S&P 500']
# get the average return for each month in equities
# extract standard deviations
equities_std = equities.std()
# get average std for all equities
equities_std = np.mean(equities_std)
short_term_gov_debt_std = short_term_gov_debt.std()

# import the annual data for other assets
annual_return_data = pd.read_excel('[ENTER PATH FOR DATA]', sheet_name='Exhibit 9', index_col='Year')

# create function to convert returns from annual to monthly
#---------------------------------------------------------#

# convert the annual returns to average monthly returns
# define function that we can apply to each element (cell) in the df
# this will convert the realized annual return to the avg monthly return
def annual_to_monthly(annual_return):
    # simple formula to convert annual to monthly returns
    monthly_return = (1 + annual_return) ** (1/12) - 1
    # return the value so it changes and is usable after exectution
    return monthly_return
# call function on return data
# applies function to every element in the dataframe
# we now have mean monthly return for each year in each asset
# we use the .applymap() to apply a callback function to every element in a df
avg_monthly_return_data = annual_return_data.applymap(annual_to_monthly)
# drop irrelevant columns from dataframe
annual_return_data = annual_return_data.drop(columns=['Inflation'])
annual_return_data = annual_return_data.dropna()

# MERGE TWO DATAFRAMES BASED ON THE YEAR
#--------------------------------------#

# convert both the indicies to the same datetime format
# we are converting monthly to yearly
# i.e. there are 12 * 2002 for the year 2002
# the actual dates are not relevant for our analysis
# change name to year
# for readability, not relevant to analysis
avg_monthly_return_data.index.name = 'Year'
# change index type to string to merge
avg_monthly_return_data.index = avg_monthly_return_data.index.astype(str)
# create new df of the individual returns we extracted
# assign nanes to each series
# this is needed in order to merge
equities.name = 'Equities'
short_term_gov_debt.name = 'Short Term Treasuries'
remerged_dataframe = pd.merge(equities, short_term_gov_debt, left_index=True, right_index=True, how='inner')
# change index type to string to make the merge between the two dataframes
# first change it using ˜datetime format to year
# then convert to a string to properly merge
remerged_dataframe.index.name = 'Year'
remerged_dataframe.index = pd.to_datetime(remerged_dataframe.index).year.astype(str)
# perform merge on the two dataframes
# we essentially append the remerged_dataframe to the other dataframe
# we make the merge based on the index
# this will fill all the values from the remerged_dataframe df and plugs in the corresponding value for each month
# therefore, the return values for the full_monthly_returns_df are more accurate and the remerged_dataframe are avg approximations
# however, this lets us estimate a relatively accurate cov matrix we can use for monte carlo analysis
# the inner merge matches the index values, in this case the year, and then fills in the data based on that
# since avg_monthly_return_data contains one value per year (avg return over those 12 periods), it will be the same for each year
# while this is not ideal, it lets us estamite a more accurate covariance matrix than annualizing the more accurate data
full_monthly_returns_df = remerged_dataframe.merge(avg_monthly_return_data, left_index=True, right_index=True, how='inner')

# MAKE CALCULATION
#----------------#

# get the covariance matrix
# save as np array for dot multiplication
full_cov_matrix = np.array(annual_return_data.cov())
# extract columns for later use
colum_names = ['Equities', 'Short Term Treasuries', 'Treasury Bonds', 'Corporate Bonds', 'Hedge Funds', 'Private Equity']
# define the weights that we extracted
# assume the alts allocation is half hedge fund half pe since not specified
current_portfolio_weights = np.array([quarterly_tbill_weight, gov_bonds_weight, fixed_income_weight, equity_weight, (alts_weight / 2), (alts_weight / 2)])
# calculate the portfolio variance based on the weights and cov matrix
portfolio_variance = np.dot(current_portfolio_weights, np.dot(full_cov_matrix, current_portfolio_weights))
# use sqrt to get the standard deviation
portfolio_std = np.sqrt(portfolio_variance)

# get the expected return of the portfolio
#expected_monthly_returns_of_asset = np.array([np.mean(column) for _, column in full_monthly_returns_df.items()])
expected_monthly_returns_of_asset = np.array([np.mean(column) for _, column in annual_return_data.items()])
# use dot matrix multiplication to get the expected monthly return
expected_monthly_return_of_portfolio = np.dot(expected_monthly_returns_of_asset, current_portfolio_weights)
expected_annual_return_of_portfolio = expected_monthly_return_of_portfolio
# annualize the results to simulate solvency
#expected_annual_return_of_portfolio = ((1 + expected_monthly_return_of_portfolio)**12) - 1
print(expected_annual_return_of_portfolio)
# scale the standard deviation to annual
#expected_annual_portfolio_std = portfolio_std * np.sqrt(12)
expected_annual_portfolio_std = portfolio_std
print(expected_annual_portfolio_std)

# CONDUCT MONTE CARLO ANALYSIS
#----------------------------#

# import package
from scipy.stats import norm

# set the number of years to predict
# supposed to go through 2035
time_periods = 2035 - 2016
# create functinon to calculate expected return
def calc_expected_return(std, exp_return):
    """purpose of this function is to recreate the NORMINV() Excel function"""
    # calculate a random val between 0-1
    # generate from standard normal distribution
    rand_val = norm.ppf(np.random.rand())
    # scale random val to std and add expected return
    # by adding the expected return, you are effectively shifting the mean of the distribution
    # this matches the dist with expected return
    simulated_return = rand_val * std + exp_return
    # return val
    return simulated_return
# create new function to simulate solvency
# create function to simulate the solvency of the portfolio
def simulate_solvency_with_risk(starting_assets, avg_deductions_gr, avg_additions_gr, port_std, port_exp_return):
    # initialize variables
    available_assets = starting_assets
    curr_deduction = start_deduction
    curr_addition = starting_addition
    #results = results = pd.DataFrame(columns=['Total Assets', 'Additions', 'Deductions', 'Expected Return'])
    # create empty list to hold data
    results_data = []
    # create period - year - incrementer
    solvent_years = 0

    # set up the loop
    # this while loop will execute until the assets are depleted
    while available_assets >= 0:
        # get random return to plug in
        simulated_return = calc_expected_return(port_std, port_exp_return)
        # calculate current year's deduction and addition based on growth rates
        period_deduction = curr_deduction * (1 + avg_deductions_gr)
        # update the curr_deduction variable to reflect the growth
        curr_deduction = period_deduction
        # increase the addition based on growth rate
        period_addition = curr_addition * (1 + avg_additions_gr)
        # increase the addition variable
        curr_addition = period_addition
        # update available assets
        # increase the available assets by the simulated growth rate
        # subtract the deduction and add addtion
        available_assets = (available_assets * (1 + simulated_return)) + period_addition - period_deduction
        # increment solvent year
        solvent_years += 1
        # append results to the dataframe
        # have to set index to ignore so that it can be properly appended
        #results = results.append({'Total Assets': available_assets, 'Additions': curr_addition, 'Deductions': curr_deduction, 'Expected Return': simulated_return}, ignore_index=True)
        row = {'Total Assets': available_assets, 'Additions': curr_addition, 'Deductions': curr_deduction, 'Expected Return': simulated_return}
        results_data.append(row)
        # convert the list to a dataframe
        results = pd.DataFrame(results_data)
        
    # return the solvent_years variable as well as results df
    return [solvent_years, results]
# set number of sims
sims = 1000
# create results list
sim_results = []
# create loop
for i in range(sims):
    # reset starting assets for each sim
    available_assets = starting_assets
    # run sim
    sim_result = simulate_solvency_with_risk(starting_assets, avg_deductions_gr, avg_additions_gr, expected_annual_portfolio_std, expected_annual_return_of_portfolio)
    # append sim to results
    sim_results.append(sim_result)

# MAKE PLOT OF RESULTS
#--------------------#
# remove most extreme value to normalize results
solvency_years_idx_yr = [(idx, result[0]) for idx, result in enumerate(sim_results)]
max_tuple = max(solvency_years_idx_yr, key=lambda x: x[1])
max_years = max_tuple[1]
# remove from list
sim_results = [[years, result] for years, result in sim_results if years != max_years]

# repeat
# remove most extreme value to normalize results
solvency_years_idx_yr = [(idx, result[0]) for idx, result in enumerate(sim_results)]
max_tuple = max(solvency_years_idx_yr, key=lambda x: x[1])
max_years = max_tuple[1]
# remove from list
sim_results = [[years, result] for years, result in sim_results if years != max_years]
# remove most extreme value to normalize results
solvency_years_idx_yr = [(idx, result[0]) for idx, result in enumerate(sim_results)]
max_tuple = max(solvency_years_idx_yr, key=lambda x: x[1])
max_years = max_tuple[1]
# remove from list
sim_results = [[years, result] for years, result in sim_results if years != max_years]
# extract the results dataframes
# make a plot of the total assets
asset_simulation_data = [result[1]['Total Assets'].tolist() for result in sim_results]
# insert the starting - year 0 - assets at the beginning of each list
# we can do this using a simple for loop
for i in range(len(asset_simulation_data)):
    # since converted series values to list
    # we can simply prepend as such
    asset_simulation_data[i].insert(0, available_assets)

# make a plot of the assets for each simulation
plt.figure(figsize=(10, 6))
for data in asset_simulation_data:
    plt.plot(data)
plt.ylabel('Total Assets', fontsize=14)
plt.xlabel('Years', fontsize=14)
plt.title('Fund Solvency Simulation w/ Risk', fontsize=14)
plt.ylim(0)
plt.grid()
plt.margins(x=0, y=0)
plt.show()

# CALCULATE THE PROBABILITY OF SURVIVAL
#-------------------------------------#

# extract the survival years for each sim
solvency_years_sim_data = [result[0] for result in sim_results]
# extract the mean year
mean_solvency_years = np.mean(solvency_years_sim_data)
print(mean_solvency_years)
# get years to survive
# from 2017 through 2035
target_survival_years = len(range(2017, 2036))
# create list to store results
survival_chance = []
# create loop to find proportion of survivors
for year in range(target_survival_years):
    # get a count of how many funds have survived up to that point
    # creates sum of all the funds that have survived up to given year
    num_survived = sum(1 for years in solvency_years_sim_data if years >= year)
    # calculated the proportion of funds that have survived
    survival_percentage = (num_survived / len(solvency_years_sim_data))
    # append result to list
    survival_chance.append((year, survival_percentage))
    
# convert to df so we can export to excel
survival_proportion_df = pd.DataFrame(survival_chance, columns=['Year', 'Survival Rate'])
# export to excel
survival_proportion_df.to_excel('./survival_rates.xlsx', index=False)