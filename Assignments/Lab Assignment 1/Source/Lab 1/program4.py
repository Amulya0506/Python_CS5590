# Set who are attending "Python" class
python_class={'Aman','Alex','Bob'}
print(python_class)
# Set who are attending "Web Application" class
web_class={'Bob','Aman','Riyan'}
print(web_class)
#list of students who attends both the classes
coomon = python_class|web_class
#list of students who are not coomon in both the classes
notcommon = python_class^web_class
print("list of students who are attending both the classes: %s"%(coomon))
print("list of students who are not common in both the classes: %s"%(notcommon))
