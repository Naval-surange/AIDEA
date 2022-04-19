import streamlit as st
st.set_page_config(layout="wide",page_title='AIDEA', page_icon="💡")

import sys
sys.path.insert(1,"./Frontend")
sys.path.insert(1,"./Backend")

from PIL import Image
import Frontend.classif_dashboard as classif_dashboard
import Frontend.regress_dashboard as regress_dashboard
import Frontend.ts_dashboard as ts_dashboard

from hydralit import HydraApp
from Backend.login import Login, Signup
from Backend.Session import get_session_id
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

logo = Image.open('./Frontend/logo.png')

st.image(logo,width=300,clamp=True)
st.markdown("<b>AI</b> <b>D</b>evelopment, <b>E</b>xplainability, and <b>A</b>bstraction  \n<small><i>Minimalistic No-Code platform to convert your AI Ideas into Reality</i></small>",unsafe_allow_html=True)


login_app = Login()
signup_app = Signup()

logged_in = False


placeholder = st.empty()
with placeholder.container():
    if(not logged_in):
        app = HydraApp()
        app.add_app("Login",app=login_app)
        app.add_app("Signup",app=signup_app)
        app.run()

fp = open("./Backend/logged_in_user.txt", "r")
for line in fp:
    if line.strip() == get_session_id():
        logged_in = True
fp.close()

if(logged_in):
    placeholder.empty()
    hide_streamlit_style = """
                        <style>
                        #MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}
                        </style>
                        """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    PAGES = {
        "--Select--": None,
        "Classification": classif_dashboard,
        "Regression": regress_dashboard,
        "Time Series - Univariate (experimental)": ts_dashboard
    }
    st.sidebar.title('AIDEA')
    selection = st.selectbox("Select Use Case", list(PAGES.keys()))
    page = PAGES[selection]

    if page is not None:
        page.app()
    else:
        st.write("1. Select Use Case.   ")
        st.write("2. Choose Dataset from Sidebar.")
        st.write("3. Fill Required Information.")
        st.markdown("4. <i>Run Experiment</i> and Train your model within minutes <i>(Feature Engineering & Training will be done automatically)</i>.",unsafe_allow_html=True)
        st.markdown("5. Review performance metrics and explainability through <i>XAI Dashboard.</i>",unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<small><center><i>Note: Trained Models should not be used in production using current version of AIDEA.</i></center></small>",unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.markdown("<small><center><b>A Ripik.ai Product</b></center></small>",unsafe_allow_html=True)