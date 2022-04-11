from cmath import exp
import streamlit as st
import pickle
import sklearn
import numpy as np

#import the model 

pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('data.pkl','rb'))


st.title('car price predictor')

#brand

company=st.selectbox('Brand',df['name'].unique())

#year

year=int(st.selectbox('year of purchase',[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,
                                      2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]))

#km_driving

km_driving=st.number_input('No km traveling')

#fuel

fuel=st.selectbox('Fuel',df['fuel'].unique())

#seller_type

seller=st.selectbox('seller',df['seller_type'].unique())

#transmission

transmission=st.selectbox('transmission',df['transmission'].unique())

#owner

owner=st.selectbox('owner',df['owner'].unique())

#mileage	

mileage=st.number_input('mileage')

#engine

engine=int(st.number_input('engine cc capacity'))

#max_power

maxpower=st.number_input('max power of engine')


#torque	

torque=st.number_input('rotation per second')


#seats


seats=int(st.number_input('no of seats'))



#predict

if st.button('PREDICT PRICE'):
    query=np.array([company,year,km_driving,fuel,seller,transmission,owner,mileage,engine,maxpower,torque,seats])
    query=query.reshape(1,12)
    sp=pipe.predict(query)
    s=np.round(sp,2)
    st.title('The Predicitng price')
    st.title(s)
    #st.title(pipe.predict(query))

elif st.button('car detains'):
    query=np.array([company,year,km_driving,fuel,seller,transmission,owner])
    st.title(query)    














