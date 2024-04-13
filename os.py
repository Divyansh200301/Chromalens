import os 
path='static/colorized'
# if not os.path.exists(path):
#     os.makedirs(path)
per=os.stat(path).st_mode
print(oct(per)[-3:])