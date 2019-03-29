
#Question 1
'''Given a string S and a list of allowed words D, find the least number of spaces to split S
into words all from D. If splitting is not possible, print 'n/a.'''

#Logic Explanation
'''
I divided the problem into 2 parts.
First I consider all prefixes of the string S one by one and check if they are present in list D
using recursion. If the prefix is present, then I add it to the list called result and store number of spaces required
After getting all the possible combinations, I see which one has the least space and print that in the output
'''


import numpy as np



def splitter(S,D,output):
    if len(S) == 0 : #baseline: if S is at the end, will add output to the result list
        result.append(output)
        size.append(len(output.split()))
        return
    for l in range(len(S)):
        prefix = S[:l+1] #consider all prefixes of the current substring
        if (prefix in D): #if prefix is present in D, will add it to the output and recurse for the remaing string
            if (len(output) == 0):
                splitter(S[l+1:],D,prefix) #to remove leading space in the output
            else:
                splitter(S[l+1:],D,output+" "+prefix)



S = ["abcdefab","cdab","abc"]
D = [["abc", "def","ab","cd","ef"],["ab","cd"],["ab","bc"]]


for s,d in zip(S,D):
    result = [] #store all possible combinations
    size = [] #store the spaces requried for each combinations
    print("S:", s , " D: ", d)
    splitter(s,d,"")
    if(len(result) == 0):
        print ("n\\a")
    else:
        least_space = np.min(size)
        print((least_space-1), ", ", result[size.index(least_space)])
