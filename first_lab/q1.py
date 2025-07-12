##Consider the given list as [2, 7, 4, 1, 3, 6]. Write a program to count pairs of elements with sum equal to 10.
def count_sum(in_list,target_sum):
            count_10 = 0
            n = len(in_list)
            for i in range(n):
                    for j in range(i+1,n):
                            if in_list[i] + in_list[j] == target_sum:
                                    count_10 +=1
            return count_10

s_list = [2,7,4,1,3,6]
target_sum = 10
p_count = count_sum(s_list,target_sum)
print("number of pairs with sum 10 are : ", p_count)
