# give me the python program for look and say number sequence
def lookandsay(n):
    if n == 1:
        return "1"
    else:
        prev_term = lookandsay(n - 1)
        result = ""
        count = 1
        for i in range(1, len(prev_term)):
            if prev_term[i] == prev_term[i - 1]:
                count += 1
            else:
                result += str(count) + prev_term[i - 1]
                count = 1
        result += str(count) + prev_term[-1]
        return result
print(lookandsay(5))



