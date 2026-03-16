# name = input("what is your name? ")
# file = open("name.txt","a")
# file.write(name + "\n")
# file.close()

# name = []

# with open("name.txt") as file:
#     for line in file:
#         name.append(line.rstrip())

# for name in sorted(name):
#     print('Hello ' + name + '!')    

#Simplfied
# with open("name.txt") as file:
#     for line in sorted(file):
#         print('Hello ' + line.rstrip() + '!')
 
        
# with open("student.csv") as file:
#     for line in file:
#         row = line.rstrip().split(",")
#         print(f"{row[0]} is in  {row[1]}")

#Unpacking
with open("student.csv") as file:
    for line in file:
        name, course = line.rstrip().split(",")
        print(f"{name} is in  {course}")





