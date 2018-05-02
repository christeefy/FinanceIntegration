import abc

import pandas as pd
import numpy as np

import requests

import datetime
from dateutil.relativedelta import relativedelta
import re

class Asset(metaclass=abc.ABCMeta):
    pass

class Currency(Asset, metaclass=abc.ABCMeta):
    
    transaction_schema = ['datetime', 'type', 'units', 'price', 'fee']
    transaction_types = ['fund', 'buy', 'comm', 'fee', 'sell', 'transfer', 'display']
    
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol
        self.units = 0.
        self.total_profit = 0.
        self.current_acb = 0.
        self.transactions = pd.DataFrame()
        self.inception_date = None

    def __repr__(self):
        return f'{self.name} ({self.symbol})\nUnits: {self.units:.5f}\nTotal Profit: {round(self.total_profit, 2)}'
        
    def worth(self, fiat='cad'):
        ''' None -> float
        Return the value in `fiat`.
        '''
        return self.units * self.price(fiat)
    
    def cost_per_unit(self):
        ''' None -> float
        Return the average acquisition cost per unit.
        '''
        assert self.units > 0, f'Cannot calculate cost per unit for {self.name} when there\'s no funds.' 
    
        return self.current_acb / (self.units + 1e-6)
        
    def buy(self, units, price, fee, dt):
        ''' (float, float, float, datetime) -> None
        Log a purchase. Update attributes accordingly.
        
        units: Units purchased
        price: Purchase price (CAD)
        fee: Fee (in units)
        date: Transaction date
        '''
        # Modify fee when is is NaN. 
        # Occurs when funding transaction involve cryptos.
        if np.isnan(fee):
            fee = 0.

        # Update attributes
        self.units += units
        self.current_acb += price * (units + fee)
        
        # Add transaction
        self._add_transaction(transaction_type='buy', 
                              units=units, 
                              price=price, 
                              fee=fee, 
                              dt=dt)
        
    def sell(self, units, price, fee, dt):
        ''' (float, float, float, datetime) -> None
        Log a sale. Update attributes accordingly.
        
        units: Units sold
        price: Sale price (CAD)
        fee: Fee (in units)
        date: Transaction date
        '''
        assert self.units >= units, f'More \'{self.symbol}\' units sold than owned.'
        
        # Update attributes
        self.total_profit += price * (units - fee) - self.cost_per_unit() * units
        
        self.units -= units
        self.current_acb = max(0, self.current_acb - price * (units - fee))
        
         # Add transaction
        self._add_transaction(transaction_type='sell', 
                              units=units, 
                              price=price, 
                              fee=fee, 
                              dt=dt)
        
    def commission(self, units, dt):
        ''' (float, datetime) -> None
        Log a commission. Update attributes accordingly.
        
        units: Units obtained from commission
        date: Transaction date
        '''
        # Update attributes
        self.units += units
        
        # Add transaction
        self._add_transaction(transaction_type='comm', 
                              fee=0., 
                              price=Currency.price(self.symbol, timestamp=dt.timestamp()),
                              units=units,
                              dt=dt)
        
    def acb(self, dt=datetime.datetime.now()):
        ''' int -> float
        Calculate the Average Cost Basis (ACB) for a given `dt`. 
        '''
        # Obtain relevant transactions
        df = Currency._filter_transaction(self.transactions, 
                                          start=self.transactions.loc[0, 'datetime'],
                                          end=dt)
        
        # Calculate ACB
        _acb = 0.
        
        for _, row in df.iterrows():
            if row['type'] in ['buy', 'fund']:
                _acb += row['price'] * (row['units'] + row['fee'])
            elif row['type'] == 'sell':
                _acb = max(0, _acb - row['price'] * (row['units'] - row['fee']))
            elif row['type'] == 'transfer':
                _acb += row['price'] * row['fee']
                
        return _acb
    
    def fee(self, start, end, year=None):
        ''' int -> float
        Calculate the fees accumulated between `start` and `end` (inclusive). 
        
        If `year` is not None, provide fees accumulated for that year. 
        '''
        # Obtain relevant transactions
        if year is not None and isinstance(year, int):
            start = datetime.datetime(year, 1, 1)
            end = datetime.datetime(year, 12, 31)

        return (Currency
                ._filter_transaction(self.transactions, 
                                     start=datetime.datetime(dt.year, 1, 1),
                                     end=dt)['fee']
                .sum())
    
    def profit(self, start, end, year=None):
        ''' datetime, datetime -> float
        Calculate the profits for from `start` to `end` (inclusive). 

        If year is provided, profit is calculated for that year. 
        '''
        assert start <= end, 'start should precede end.'

        # If year is provided, set `start` and `end` to first and last day of the year.
        if year is not None and isinstance(year, int):
            start = datetime.datetime(year, 1, 1)
            end = datetime.datetime(year, 12, 31)

        if start == self.transactions.loc[0, 'datetime'] and end >= self.transactions.loc[len(self.transactions) - 1, 'datetime']:
            return self.total_profit
        
        # Obtain relevant transactions
        df = Currency._filter_transaction(self.transactions, start=start, end=end)
        
        # Initialise values
        _acb = self.acb(start - datetime.timedelta(days=-1))
        _profit = 0.
        _units = self._units(start - datetime.timedelta(days=-1))
        _epsilon = 1e-6
        
        # Calculate profit
        for _, row in df.iterrows():
            if row['type'] in ['buy', 'fund']:
                _acb += row['price'] * (row['units'] + row['fee'])
                _units += row['units']
                
            elif row['type'] == 'sell':
                _profit += row['price'] * (row['units'] - row['fee']) - row['units'] * _acb / (_units + _epsilon)
                _acb = max(0, _acb - row['price'] * (row['units'] - row['fee']))
                _units -= row['units']
                
            elif row['type'] == 'comm':
                _units += row['units']

            elif row['type'] == 'transfer':
                _units -= row['fee']
                _acb += row['price'] * row['fee']
                _profit -=  row['price'] * row['fee']
                
        return _profit 
    
    def _units(self, dt=datetime.datetime.now()):
        ''' int -> float
        Calculate the units at the end of a given datetime `dt`. 
        '''
        
        # Obtain relevant transactions
        df = Currency._filter_transaction(self.transactions, 
                                          start=self.transactions.loc[0, 'datetime'],
                                          end=dt)
        
        # Calculate units
        _units = 0.
        
        for _, row in df.iterrows():
            if row['type'] in ['buy', 'comm', 'fund']:
                _units += row['units']
            elif row['type'] in ['sell', 'transfer']:
                _units -= row['units']
                
        return _units

    @staticmethod
    def price(fsym, tsym='cad', timestamp=datetime.datetime.now().timestamp()):
        ''' str, str, float -> float
        Return the currency conversion rate from `fsym` to `tsym` for the specified `timestamp`.

        Data based on aggregated closing hourly price obtained from Cryptocompare.
        '''
        if fsym == tsym:
            return 1.

        payload = {
            'tsym': tsym.upper(),
            'fsym': fsym.upper(),
            'toTs': int(timestamp)
        }

        # Obtain response
        response = requests.get('https://min-api.cryptocompare.com/data/histohour', params=payload)

        assert response.status_code == 200, 'Failed currency conversion query.'

        return response.json()['Data'][-1]['close']
    

    ## Private methods for handling transactions
    def _add_transaction(self, transaction_type, units, price, fee, dt):
        # Add inception date for first transaction it encounters
        if self.inception_date is None:
            self.inception_date = dt

        self.transactions = pd.concat([self.transactions, 
                                       pd.DataFrame([(dt, transaction_type, units, price, fee)], 
                                                    columns=Currency.transaction_schema)],
                                      ignore_index=True,
                                      verify_integrity=True)
        
    @staticmethod
    def _filter_transaction(df, start, end=datetime.datetime.now(), freq=None):
        ''' df, datetime, datetime, str -> df
        Return a filtered dataframe containing dates from `start` to `end` (inclusive),
        grouped in time frequency `freq`.
        
        Inputs:
            df: A dataframe containing 'date' series of datetime objects
        '''
        # Filter transactions
        df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]

        if freq is not None:
            df = (df
                  .groupby([pd.Grouper(key='datetime', freq=freq), 'type'])
                  .sum()
                  .reset_index())

        return df


    ## Private static method for ROI calculations
    @staticmethod
    def _decouple_fees(raw_df):
        ''' DataFrame -> DataFrame
        Decouple the fees from transactions by creating a dedicated 'type'
        for fees.

        Return a new dataframe.
        '''
        keys = ['datetime', 'type', 'price', 'units']
        df = pd.DataFrame(columns=keys)

        for _, row in raw_df.iterrows():
            if row['type'] in ['buy', 'sell']:
                decoupled_df = pd.DataFrame([[row['datetime'], row['type'], row['price'], row['units'] * (-1 if row['type'] == 'sell' else 1)],
                                              [row['datetime'], 'fee', row['price'], -row['fee']]], 
                                             columns=keys)
            elif row['type'] == 'comm':
                decoupled_df = pd.DataFrame([[row['datetime'], row['type'], row['price'], row['units']]], columns=keys)

            elif row['type'] == 'transfer':
                decoupled_df = pd.DataFrame([[row['datetime'], 'fee', row['price'], -row['fee']]], columns=keys)

            # Append to main df
            df = df.append(decoupled_df)

        # Make 'type' to categorical
        df['type'] = df['type'].astype('category').cat.set_categories(Currency.transaction_types)

        # Reset index to be a monotonically increasing range
        df = df.reset_index(drop=True)

        return df

    def roi(self, duration, mode='money'):
        ''' DataFrame, str -> float, bool
        Calculate a given `mode` of return of investment. 
        ROI is annualized if the duration is over a year. 

        Simple ROI: Final Market Value / Initial Market Value
        Time ROI: Time weighted return
        Money ROI: Modified Dietz method

        Inputs:
            mode: Type of ROI to calculate. Valid options are ['simple', 'time', 'money']
            duration: String denoting appropriate duration. 
                          Follows pd time offsets. Multiples allowed.
                          Currently implemented for ['D', 'W', 'M', 'Q', 'Y', 'MTD', 'YTD'].

        Returns:
            ROI (annualized if time frame is over a year).
            Boolean on whether ROI was annualized.
        '''
        def _endpoint_dates(duration, self):
            ''' str -> Date, Date
            Return the date `duration` in the past as well as the current date.
            '''
            # Separate potential multiple from time unit
            time_unit = re.search(r'[A-Z]+$', duration).group()

            time_multiple = duration.split(time_unit)[0]
            time_multiple = 1 if time_multiple == '' else int(time_multiple)

            # Ensure that `duration` is valid
            assert time_unit in ['D', 'W', 'M', 'Q', 'Y', 'MTD', 'YTD'], "Invalid duration provided. Choose from ['D', 'W', 'M', 'Q', 'Y', 'MTD', 'YTD']."

            # Short circuit function it is `time_unit` is 'YTD'
            if time_unit in ['MTD', 'YTD']:
                end = datetime.datetime.now()
                start = datetime.datetime(end.year, 1 if time_unit == 'YTD' else end.month, 1)
                return start, end

            # Dictionary whose values are binary lists of (day, month, year)
            DAYS_IN_DURATION = {
                'D': [1, 0, 0],
                'W': [7, 0, 0],
                'M': [0, 1, 0],
                'Q': [0, 3, 0],
                'Y': [0, 0, 1]
            }

            # Get `days`, `months` and `years` variables for use in relativedelta
            days, months, years = [-time_multiple * x for x in DAYS_IN_DURATION[time_unit]]

            # Calculate start and end dates
            end = datetime.datetime.now()
            start = max(self.inception_date, end + relativedelta(days=days, months=months, years=years))

            return start, end

        def _add_endpoints(df, duration, self):
            ''' DataFrame, str, Asset -> DataFrame
            Append and prepend a row each to `df` for the current date
            and a date in the past by `duration` respectively.

            Inputs:
                df:       DataFrame with pre-computed values
                duration: String denoting appropriate duration. 
                          Follows pd time offsets. Multiples allowed.
                          Currently implemented for ['D', 'W', 'M', 'Q', 'Y', 'MTD', 'YTD'].
                obj:      Required to infer prices of Asset
            '''
            # Get start and end dates
            start, end = _endpoint_dates(duration, self)

            # Add a row for start date if inception date precedes it
            if start > self.inception_date:
                df = df.append(pd.DataFrame({
                        'datetime': [start],
                        'type': pd.Categorical(['display'], categories=Currency.transaction_types),
                        'price': [Currency.price(self.symbol, timestamp=start.timestamp())],
                        'units': [0],
                        'unit_balance': [df.at[0, 'unit_balance'] - df.at[0, 'units']]
                    }))

            # Add a row for end date, sort the dates and columns
            df = (
                df
                .append(pd.DataFrame({
                    'datetime': [end],
                    'type': pd.Categorical(['display'], categories=Currency.transaction_types),
                    'price': [Currency.price(self.symbol, timestamp=end.timestamp())],
                    'units': [0],
                    'unit_balance': [df['unit_balance'].iat[-1]]}))
                .sort_values('datetime')
                .reset_index(drop=True)
            )[['datetime', 'type', 'price', 'units', 'unit_balance']]

            return df

        def _roi_pre_computations(self, duration):
            ''' DataFrame, str -> DataFrame
            Return a DataFrame containing the necessary computations for ROI calculations.
            '''
            # Decouple fees from transactions
            df = (
                self
                .transactions
                .pipe(Currency._decouple_fees)
                .assign(unit_balance = lambda df: df['units'].cumsum())
            )

            # Filter rows with relevant dates
            df = df[df['datetime'] >= _endpoint_dates(duration, self)[0]]

            assert len(df) > 0, 'No transactions for this time window.'

            # Add start and end rows
            # Calculate remainder of states
            df = (
                df
                .reset_index(drop=True)
                .pipe(_add_endpoints, duration, self)
                .assign(cashflow = lambda df: df['units'] * df['price'])
                .assign(value = lambda df: df['unit_balance'] * df['price'])
                .assign(inter_dt_roi = lambda df: df['unit_balance'].shift(1) * df['price'] / df['value'].shift(1) - 1)
                .assign(cum_days_delta = lambda df: (df['datetime'] - df.at[0, 'datetime']).apply(lambda dt: dt.days))
                .assign(dietz_weight = lambda df: 1 - df['cum_days_delta'] / df['cum_days_delta'].iat[-1])
            )

            # Replace NaN values with 0.
            df.loc[0, 'inter_dt_roi':'cum_days_delta'] = 0

            return df
        
        
        ###################
        # Start of method #
        ###################

        # Ensure that mode is valid
        assert mode in ['simple', 'time', 'money'], 'Invalid mode provided. Choose from "simple", "time" and "money".'
        
        # Get precomputed dataframe
        df = _roi_pre_computations(self, duration)

        # Constants
        _DP = 5 # Decimal places for rounding
        _EPSILON = 1e-10 # For numerical stability

        # Infer length of `df`
        N = len(df)

        # Infer duration of dataset
        total_days = (df.at[N - 1, 'datetime'] - df.at[0, 'datetime']).days

        if mode == 'money':
            _roi = (df.at[N - 1, 'value'] - df.at[0, 'value'] - df['cashflow'].sum()) / (df.at[0, 'value'] + (df['dietz_weight'] * df['cashflow']).sum() + _EPSILON)
        elif mode == 'time':
            _roi = (df['inter_dt_roi'] + 1).agg(np.prod) - 1
        elif mode == 'simple':
            _roi = df.at[N - 1, 'value'] / df.at[0, 'value'] - 1

        # Annualize ROI if it over a year
        if total_days > 365.25:
            _roi = (_roi + 1)**(365.25 / (total_days + _EPSILON)) - 1

        # Round to appropriate decimal places
        _roi = round(_roi, _DP)

        return _roi, total_days > 365.25
    
class Crypto(Currency):
    def price(self, fiat='cad'):
        ''' str -> float
        Return the closing price in `fiat`.
        Use Quadriga Public API.
        '''
        response = requests.get('https://api.quadrigacx.com/v2/ticker?book={}_{}'\
                                .format(self.symbol, fiat))
        
        assert response.status_code == 200, 'Quadriga price query failed.'
        
        return float(response.json()['last']) 

    def transfer(self, fee, dt):
        ''' (float, float, datetime) -> None
        Log a transfer. Update attributes accordingly.
        
        units: Units sold
        fee: Fee (in units)
        date: Transaction date
        '''
        # Find price at corresponding timestamp
        price = Currency.price(self.symbol, timestamp=dt.timestamp())

        # Update attributes
        self.units -= fee
        self.current_acb += fee * price
        self.total_profit -= fee * price

        # Add transaction
        self._add_transaction(transaction_type='transfer', 
                              units=0., 
                              price=price, 
                              fee=fee, 
                              dt=dt)
    
class Fiat(Currency):
    @staticmethod
    def _correction_rate(base='cad'):
        ''' str -> float
        Helper function to fix Fixer API GBP base currency limitation.
        
        Returns the CAD/GBP exchange rate (CAD per GBP). 
        '''
        payload = {
            'access_key': '9288db303315491ac90897b70e9bf19d',
            'symbols': base.upper()
        }
        
        return requests.get('http://data.fixer.io/api/latest', params=payload).json()['rates']['CAD']
          
    def price(self, fiat='cad'):
        ''' str -> float
        Obtain currency exchange rate relative to `fiat` using Fixer's API.
        If `fiat` is the base currency, returns the instance's units.
        '''
        if self.symbol == fiat:
            return self.units
        
        payload = {
            'access_key': '9288db303315491ac90897b70e9bf19d',
            'symbols': self.symbol.upper()
        }
        
        correction_rate = Fiat._correction_rate()
        forex_rate = requests.get('http://data.fixer.io/api/latest', params=payload).json()['rates'][self.symbol.upper()]
        
        return self.units * forex_rate / correction_rate

    def fund(self, units, price, fee, dt):
        ''' (float, float, float, datetime) -> None
        Log a fund transaction. Update attributes accordingly.
        This method piggybacks on the `buy` method.
        
        units: Units purchased
        price: Purchase price (CAD)
        fee: Fee (in units)
        date: Transaction date
        '''
        # Call buy method, since fund method's logic is nearly identical
        self.buy(units=units, price=price, fee=fee, dt=dt)

        # Replace 'buy' with 'fund' in the appropriate row in transaction
        idx = (self.transactions['datetime'] == dt).idxmax()
        self.transactions.loc[idx, 'type'] = 'fund'