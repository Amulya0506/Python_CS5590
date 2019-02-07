class Employee:
    data_member = 0
    salaryList = []
    def __init__(self, name, family, salary, department):
        Employee.data_member += 1
        self.name = name  # instance variable unique to each instance
        self.family = family
        self.salary = salary
        self.department = department
        Employee.salaryList.append(self.salary)
    def avgSalary(self):
        avgsalary = sum(Employee.salaryList)
        return avgsalary/Employee.data_member

class Fulltime_Employee(Employee):

    def __init__(self, name, family, salary, benefit_salary, department):
        total_salary = salary+benefit_salary
        super().__init__(name, family, total_salary, department)
        self.benefit_salary = benefit_salary

    # def total_sal(self):
    #     print(self.benefit_salary)
    #     total_salary = self.salary + self.benefit_salary
    #     Employee.salaryList.append(total_salary)
    #     return total_salary

employee1 = Employee("Vineetha", "family", 2000, "Manager")
employee2 = Employee("Alex", "family", 3000, "Software")
employee3 = Employee("Bob", "family", 4000, "Employee")
employee4 = Employee("Mike", "family", 5000, "Software")
employee5 = Fulltime_Employee("Mike", "family", 5000,100,"Software")

print(Employee.salaryList)
print("Average Salary: %d"%(Employee.avgSalary(Employee)))
print("Count of Employees: %d"%(Employee.data_member))

