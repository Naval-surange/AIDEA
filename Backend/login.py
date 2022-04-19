import sqlite3
import hashlib
import streamlit as st
from Session import get_session_id
from hydralit import HydraHeadApp

# Security
# passlib,hashlib,bcrypt,scrypt


def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False


# DB Management
conn = sqlite3.connect('./Backend/data.db',check_same_thread=False)
c = conn.cursor()
# DB  Functions


def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username, password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',
			  (username, password))
	conn.commit()


def login_user(username, password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',
			  (username, password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


class Login(HydraHeadApp):
	def run(self):
		st.title("Login")
		username = st.text_input("User Name")
		password = st.text_input("Password",type='password')
		login_btn = st.button("Login")

		if login_btn:
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:
				fp = open("./Backend/logged_in_user.txt", "a")
				fp.write(str(get_session_id()) + "\n")
				fp.close()
			else:
				st.warning("Incorrect Username/Password")

class Signup(HydraHeadApp):
	def run(self):
		st.title("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")



