def list_range(n_list):
    if len(n_list) < 3:
        return "Range determination not possible"
    return max(n_list) - min(n_list)
user_input = input("Enter list of numbers separated by spaces: ")
real_no = list(map(float, user_input.strip().split()))
range_result = list_range(real_no)
print("Range of the list:", range_result)