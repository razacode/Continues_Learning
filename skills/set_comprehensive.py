jan_2022_trento_temp_f = num = [15,12,13,14,15,16,17,18,19, 12, 14, 15, 13, 12, 16, 19, 18, 18, 19, 19 ,11,15]

def main():
    set_list_temp_c = set()
    for f in jan_2022_trento_temp_f:
        t_celsuis = round((f -32) *(5/9), 2)
        set_list_temp_c.add(t_celsuis)
    print(type(set_list_temp_c))
    print(set_list_temp_c)

    #TODO in comprehensive
    set_list_temp_comp = {round((f -32) * (5/9), 1) for f in jan_2022_trento_temp_f}
    print(set_list_temp_comp)


if __name__ == '__main__':
    main()