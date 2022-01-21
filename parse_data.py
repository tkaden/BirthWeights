import pandas as pd


def get_data():
    df_init = []

    with open("/Users/tkaden/Development/data/Nat2020.txt", "r") as file:
        head = [next(file) for x in range(100)]

    for line in head:

        new_row = {'YEAR':      int(line[8:12]),
                   'M_Ht':      int(line[279:281]),  # if 99, no reporting
                   'M_Age':     int(line[74:76]),
                   'Lst_Prg':   int(line[280:281]),  # 88=FstPregnancy (cat)
                   'Wt_Gain':   int(line[305:306]),  # 9=unknown (cat)
                   'SEX':       line[474:475],
                   'GEST':      int(line[489:491]),
                   'BWt':       int(line[503:507]) # measured in grams
                   }

        if list(new_row.values())[1] != 99 and list(new_row.values())[6] != 99 and list(new_row.values())[7] != 9999 and list(new_row.values())[3] != ' ':
            df_init.append(new_row)

    return pd.DataFrame(df_init)
#%%

