check = [4, 8, 6]
check2 = []
check3 = [1,15, 12, 14, 14, 25]

def conc(l: list):
    n = len(l)
    if n == 0: return 0
    if n == 1: return 1
    l = l[:]
    l.sort()
    n = len(set(l))
    need_len = l[-1] - l[0] + 1
    return need_len - n


print(conc(check))
print(conc(check2))
print(conc(check3))
            
