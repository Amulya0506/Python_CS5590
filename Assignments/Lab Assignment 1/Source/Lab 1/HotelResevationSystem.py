class Occupant:
    def __init__(self, name, phone, SSN, number_of_rooms, number_of_people, check_in, number_of_days, type_of_room ):
        self.customerName=name
        self.occupantphone = phone
        self.occupantid = SSN
        self.number_of_rooms = number_of_rooms
        self.number_of_people = number_of_people
        self.check_in = check_in
        self.number_of_days= number_of_days
        self.type_of_room = type_of_room
# calculating the basic price
    def price(self):
        self.basic_price = 50
        self.doubleBed_price = 50
        self.singleBed_room = 25
        self.tax = 15
        if self.type_of_room == "Double Bed":
                self.price =(self.basic_price + self.doubleBed_price) * self.number_of_days
        elif self.type_of_room == "Single Bed":
                self.price = (self.basic_price+self.singleBed_room)* self.number_of_days
        return self.price

    def get_tax(self):
        return self.tax
# display the booking details of standard customer
    def display(self):
        print("Custome Name: %s"%(self.customerName))
        print("Check In date: %s"%(self.check_in))
        print("Number of Days: %d"%(self.number_of_days))
        print("Price: %d"%(self.price))

class RoomSelection:
    def __init_(self, no_of_people, no_of_rooms):
        self.no_of_people = no_of_people
        self.no_of_rooms = no_of_rooms
        self.twin_bed_price = 1000
        self.tax = 14
        self.price = (self.twin_bed_price + ((self.twin_bed_price * self.tax) * 100))
    def get_bed_price(self):
        self.price = self.price * self.no_of_rooms
        if self.no_of_people > 2:
            self.price = self.price + self.no_of_people * 50
        print("ordinary room", self.price)
    def set_tax(self):
        self.tax = 14
    def get_no_of_people(self):
        return self.no_of_people
    def get_no_of_rooms(self):
        return self.no_of_rooms
    def get_tax(self):
        return self.tax
    def set_twin_bed_size(self):
        self.twin_bed_price = 1000
    def get_twin_bed_size(self):
        return self.twin_bed_price
# Inheriting the Occupant class
class DeluxeRoom(Occupant):
    def __init__(self, name, phone, SSN, number_of_rooms, number_of_people, check_in, number_of_days, type_of_room):
        super().__init__( name, phone, SSN, number_of_rooms, number_of_people, check_in, number_of_days, type_of_room)
        self.price = (Occupant.price(self) + 100)*self.number_of_rooms
        self.price = self.price * self.number_of_rooms
        self.total_price = self.price+(self.price*15)/100
    def get_bed_price(self):
        print("Deluxe room price: ", self.total_price)
    def display(self):
        print("Custome Name: %s"%(self.customerName))
        print("Check In date: %s"%(self.check_in))
        print("Number of Days: %d"%(self.number_of_days))
        print("Price: %d"%(self.total_price))

class LuxuryRoom(RoomSelection):
    def __init_(self, no_of_people, no_of_rooms):

        super().__init_(no_of_people, no_of_rooms)
        self.no_of_people = no_of_people
        self.no_of_rooms = no_of_rooms
        self.price = (RoomSelection.get_twin_bed_size(self) + 300) + (
                    (RoomSelection.get_twin_bed_size(self) + 300) / RoomSelection.get_tax(self)) * 100
        self.price = self.price * self.no_of_rooms
        if self.no_of_people > 2:
            self.price = self.price + self.no_of_people * 50
    def get_bed_price(self):
        print("Luxury room price", self.price)
# inheriting the classes DeluxRoom LuxuryRoom & Ouccupant
class BookingInformation(LuxuryRoom, DeluxeRoom, Occupant):
    def __init_(self, no_of_people, no_of_rooms):
        self.no_of_people = no_of_people
        self.no_of_rooms = no_of_rooms
        super().__init__(no_of_people, no_of_rooms)
        DeluxeRoom.get_bed_price(self)
        LuxuryRoom.get_bed_price(self)
        RoomSelection.get_bed_price(self)
    pass

while True:
    choice= int(input(" 1. Book Standard Room\n 2. Book Delux Room\n 3. Display Booking\n 4. To End\n Enter The choice: "))
    if (choice == 1):
        customername= input("Enter the name of customer: ")
        customerphone = input("Enter the phone number of customer: ")
        customerid = input("Enter the id of customer: ")
        numberofrooms = int(input("Enter the number of rooms : "))
        numberofpeople= int(input("Enter the number of people: "))
        checkIn_date=input("Enter check in date: ")
        numberofdays=int(input("Number of days: "))
        roomtype = input("Enter the room type: ")
        customer = Occupant(customername,customerphone,customerid,numberofrooms,numberofpeople,checkIn_date,numberofdays,roomtype)
        print(customer.price())
    if (choice == 2):
        customername = input("Enter the name of customer: ")
        customerphone = input("Enter the phone number of customer: ")
        customerid = input("Enter the id of customer: ")
        numberofrooms = int(input("Enter the number of rooms : "))
        numberofpeople = int(input("Enter the number of people: "))
        checkIn_date = input("Enter check in date: ")
        numberofdays = int(input("Number of days: "))
        roomtype = input("Enter the room type: ")
        customer = DeluxeRoom(customername, customerphone, customerid, numberofrooms, numberofpeople, checkIn_date, numberofdays, roomtype)
        customer.get_bed_price()
    if (choice == 3):
        customer.display()
    if (choice == 4):
        break