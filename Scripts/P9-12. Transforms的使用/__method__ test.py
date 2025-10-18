class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __call__(self):
        print(f"Person created: {self.name}, {self.age} years old")

person = Person("Alice", 30)
person()  # This will invoke the __call__ method

person1 = Person()
person1("Bob", 25)  # This will raise an error since __init__ requires name and age