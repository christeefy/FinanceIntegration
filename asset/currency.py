import abc
import datetime
import pandas as pd
import numpy as np
import requests

class Asset(metaclass=abc.ABCMeta):
    pass

class Currency(Asset, metaclass=abc.ABCMeta):
    
    transaction_schema = ['datetime', 'type', 'units', 'price', 'fee']
    
    def __repr__(self):
        return f'{self.name} ({self.symbol})\nUnits: {self.units:.5f}\nTotal Profit: {round(self.total_profit, 2)}'
    
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
    
    def _add_transaction(self, transaction_type, units, price, fee, dt):
        self.transactions = pd.concat([self.transactions, 
                                       pd.DataFrame([(dt, transaction_type, units, price, fee)], 
                                                    columns=Currency.transaction_schema)],
                                      ignore_index=True,
                                      verify_integrity=True)
        
    @staticmethod
    def _filter_by_year(df, year, cumulative=True):
        ''' df -> df
        Return a filtered dataframe containing dates up to
        `year`, if `cumulative` is True. Otherwise, filtered dataframe
        contains entries only for that year. 
        
        Inputs:
            df: A dataframe containing 'date' series of datetime objects
        '''
        if cumulative:
            return df[df['datetime'].apply(lambda date: date.year) <= year]
        
        return df[df['datetime'].apply(lambda date: date.year) == year]
    
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol
        self.units = 0.
        self.total_profit = 0.
        self.current_acb = 0.
        self.transactions = pd.DataFrame()
        
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
                              price=np.nan,
                              units=units,
                              dt=dt)
        
    def acb(self, year=None):
        ''' int -> float
        Calculate the Average Cost Basis (ACB) at a given `year` end. 
        
        If `year` is None, provides latest ACB.
        '''
        # Obtain relevant transactions
        df = Crypto._filter_by_year(self.transactions, 
                                    year=datetime.datetime.now().year if year is None else year,
                                    cumulative=True)
        
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
    
    def fee(self, year=None):
        ''' int -> float
        Calculate the fees accumulated during a given `year`. 
        
        If `year` is None, provides total fee.
        '''
        # Obtain relevant transactions
        return (Crypto
                ._filter_by_year(self.transactions, 
                                 year=datetime.datetime.now().year if year is None else year,
                                 cumulative=False)['fee']
                .sum())
    
    def profit(self, year=None):
        ''' int -> float
        Calculate the profits for a given `year`. 
        
        If `year` is None, provides total profits.
        '''
        if year is None:
            return self.total_profit
        
        # Obtain relevant transactions
        df = Crypto._filter_by_year(self.transactions, year=year, cumulative=False)
        
        # Initalise values
        _acb = self.acb(year - 1)
        _profit = 0.
        _units = self._units(year - 1)
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
    
    def _units(self, year=None):
        ''' int -> float
        Calculate the units at the end of a given `year`. 
        
        If `year` is None, return current units.
        '''
        if year is None:
            return self.units
        
        # Obtain relevant transactions
        df = Crypto._filter_by_year(self.transactions, year=year, cumulative=True)
        
        # Calculate units
        _units = 0.
        
        for _, row in df.iterrows():
            if row['type'] in ['buy', 'comm', 'fund']:
                _units += row['units']
            elif row['type'] in ['sell', 'transfer']:
                _units -= row['units']
                
        return _units
    
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