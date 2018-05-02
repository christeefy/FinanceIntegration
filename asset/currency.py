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
    transaction_types = ['fund', 'buy', 'comm', 'fee', 'sell', 'transfer']
    
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol
        self.units = 0.
        self.total_profit = 0.
        self.current_acb = 0.
        self.transactions = pd.DataFrame()

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