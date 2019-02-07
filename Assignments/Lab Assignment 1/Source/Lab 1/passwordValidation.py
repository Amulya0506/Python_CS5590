# import Regular Expressions(re) package
import re
# Take the username naame from the user
username = input("Enter Username: ")
# loop until the valid password is entered
while True:
    # Take the password naame from the user
    password = input("Enter Password: ")
    # Validating the password length
    if len(password) < 6 or len(password) > 16:
        print("\033[1;31;0mPassword length should be in range 6-16 characters")
    # Validating the password should have atleast one number
    elif not re.search("[0-9]", password):
        print("\033[1;31;0mPassword should have atleast one number")
    # Validating the password should have atleast one special character in [$@!*]
    elif not re.search("[$@!]", password):
        print("\033[1;31;0mPassword should have at least one special character in [$@!*]")
    # Validating the password should have atleast one lowercase
    elif not re.search("[a-z]", password):
        print("\033[1;31;0mPassword should have at least one lowercase")
    # Validating the password should have atleast one uppercase
    elif not re.search("[A-Z]", password):
        print("\033[1;31;0mPassword should have at least one uppercase")
    else:
        print("\033[1;32;0mValid Password")
        break
