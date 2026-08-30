import streamlit as st
import joblib as jl
import pandas as pd

model=jl.load("dtreee.pkl")

age=st.number_input("enter your age")
timespent=st.number_input("enter timespent")
pagevisited=st.number_input("enter page v")
membership=st.number_input("enter membership")

if st.button("submit"):
    data=[[age,timespent,pagevisited,membership]]
    pred=model.predict(data)[0]
    st.write(pred)