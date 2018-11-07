books = [{'h':1,'w':1,'n':1},{'h':2,'w':2,'n':2},{'h':2,'w':2,'n':3},{'h':1,'w':1,'n':4},{'h':3,'w':1,'n':5}]

shelves = [{'l':4,'high':0,'n':1}, {'l':4,'high':0,'n':2},{'l':4,'high':0,'n':3}]

s_index = 0

while 1:

 for x in books:
     if shelves[s_index]['l'] == 0:
         s_index = s_index + 1


     if shelves[s_index]['l'] - x['w'] >= 0 and x['n']!=0:
             x['n'] = 0
             shelves[s_index]['l'] = shelves[s_index]['l'] - x['w']
             if x['h'] > shelves[s_index]['high']:
                 shelves[s_index]['high'] = x['h']

 sum = 0

 for x in books:
    sum = x['n'] + sum

 if sum == 0:
    break

sum_of_high = 0



print(shelves[0]['high'] + shelves[1]['high'])














