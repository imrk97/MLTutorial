class Student:
	def __init__(self, id : int, name : str, address : str):
		self.__id=id
		self.__name=name
		self.__address=address

	@property
	def id(self):
		return self.__id

	@id.setter
	def id(self, id : int):
		self.__id=id

	@property
	def name(self):
		return self.__name

	@name.setter
	def name(self, name : str):
		self.__name=name


	@property
	def address(self):
		return self.__address

	@address.setter
	def address(self, address : str):
		self.__address=address
	
	def __repr__(self):

		return (f'id: {self.__id}\t name: {self.__name}\t address: {self.__address}\t')


rohan = Student(1,'Rohan', 'Sodepur')
print(rohan.name)
rohan.address='Chanditala'
print(rohan)