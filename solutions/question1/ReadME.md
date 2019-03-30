#Question 1
Given a string S and a list of allowed words D, find the least number of spaces to split S
into words all from D. If splitting is not possible, print 'n/a'

#Logic Explanation
I divided the problem into 2 parts.
First I consider all prefixes of the string S one by one and check if they are present in list D
using recursion. If the prefix is present, then I add it to the list called result and store number of spaces required
After getting all the possible combinations, I see which one has the least space and print that in the output

Instructions for Running:
python Q1.py

By default, it will run all the examples outlined in the question. If you want to add more samples please add them in the list S and D respectively.
