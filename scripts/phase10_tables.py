"""Small CSV/NumPy table utility for offline phase10 figures; no pandas dependency."""
import csv
import numpy as np
class Column(np.ndarray):
    def head(self,n):return self[:n]
    def tail(self,n):return self[-n:]
class RowIndex:
    def __init__(self,table):self.table=table
    def __getitem__(self,key):return self.table.rows[key]
class LabelIndex:
    def __init__(self,table):self.table=table
    def __getitem__(self,key):
        value,column=key;return self.table.index[value][column]
class Table:
    def __init__(self,rows):
        self.rows=[dict(x) for x in rows];self.columns=list(dict.fromkeys(k for row in self.rows for k in row));self.iloc=RowIndex(self);self.loc=LabelIndex(self)
    def __len__(self):return len(self.rows)
    def __contains__(self,key):return key in self.columns
    def __getattr__(self,key):
        if key in self.columns:return self[key]
        raise AttributeError(key)
    def __getitem__(self,key):
        if isinstance(key,str):
            values=[row.get(key,np.nan) for row in self.rows]
            values=[np.nan if v is None else v for v in values]
            return np.asarray(values).view(Column)
        if isinstance(key,slice):return Table(self.rows[key])
        mask=np.asarray(key,dtype=bool);return Table([row for row,keep in zip(self.rows,mask) if keep])
    def to_csv(self,path,index=False):
        with open(path,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=self.columns);writer.writeheader();writer.writerows(self.rows)
    def assign(self,**values):return Table([dict(row,**values) for row in self.rows])
    def set_index(self,key):self.index={row[key]:row for row in self.rows};return self

def parse(value):
    if value=='':return np.nan
    if value in ('True','False'):return value=='True'
    try:return float(value)
    except ValueError:return value

def read_csv(path):
    with open(path,newline='') as f:return Table([{k:parse(v) for k,v in row.items()} for row in csv.DictReader(f)])

def concat(tables,ignore_index=True):return Table([row for table in tables for row in table.rows])
