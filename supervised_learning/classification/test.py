''' Test Script '''

class User(object):
    
    def __init__(self, name):
        self.__name = name
        print(f'My name is {self.__name}')
    
    @property     
    def Get_Name(self):
        return self.__name 
        
user1 = User("Kingsley")
print(user1.Get_Name)