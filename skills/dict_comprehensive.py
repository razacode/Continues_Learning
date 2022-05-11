from unicodedata import name

def main():
    num = [1,2,3,4,5,6,7,8,9]

    # Creating dict
    new_num = {}
    for n in num:
        new_num[n] = n**2
    print("new_num: ",new_num)

    # TODO: Using Comprehensive
    new_num1 = {n: n * n for n in num}
    print("new_num1: ",new_num1)

    # TODO: Using Comprehensive & condition 
    new_num2 = {n: n + n for n in num if n%2 == 0}
    print("new_num2: ",new_num2)

    # TODO: Using Comprehensive & condition (range)
    new_num3 = {n: n + n for n in num if 2 < n < 8}
    print("new_num3: ", new_num3)

    # ############## More complex ###################
    states = ["Alabama", "California", "Hawaii", "Florida", " New York"]
    capitals = ["Montgomery", "Sacramento","Honolulu", "Tallahassee", "Albany" ]

    # Creating dict to assign each state their capital
    my_stat_cap_dic ={}
    for state in states:
        for capital in capitals:
            if states.index(state) == capitals.index(capital):
                my_stat_cap_dic[state] = capital
    print(my_stat_cap_dic)

    # TODO: Using zip
    my_stat_cap_zip = zip(states, capitals)
    print(dict(my_stat_cap_zip))

    # TODO: Using Comprehensive & condition 
    my_stat_cap = {s: c for s in states for c in capitals if states.index(s) == capitals.index(c)}
    print(my_stat_cap)

    # TODO: Using Comprehensive & condition  for item

    scores = {'Trento' : 10, 'Verona': 23, 'Milan': 34, 'Venice': 45, 'Rome':67}
    score_com = {key: value * 2 for (key, value) in scores.items() if value >= 11 if value <=45}
    print(score_com)

if __name__ == '__main__':
    main()