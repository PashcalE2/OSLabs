import pandas as pd
import matplotlib.pyplot as plt

capacity = 5
pages_order = [4, 8, 11, 2, 3, 17, 16, 22, 6, 4, 13, 7, 23, 7, 2, 18, 14, 22, 21, 5, 12, 13, 20, 10, 5, 2, 8, 6, 5, 2,
               11, 1, 6, 3, 12, 19, 18, 4, 9, 1, 23, 20, 1, 18, 20]

'''
# рандомный вариант

capacity = 6
pages_order = [7, 2, 20, 2, 5, 18, 20, 14, 18, 9, 15, 4, 5, 12, 17, 9, 4, 15, 9, 7, 18, 7, 21, 12, 8, 20, 21, 13, 1, 13,
               12, 9, 21, 20, 13, 8, 9, 15, 1, 15, 9, 4, 5]
'''

df_header = ["Кадр {}".format(i + 1) for i in range(capacity)] + ["Новая страница", "Замена"]


def FIFO(capacity: int, pages_order: list):
    output_table = []

    pages_table = [-1 for i in range(capacity)]
    priorities = [-1 for i in range(capacity)]
    insert_priority = 0
    delete_priority = 0

    for tick, page in enumerate(pages_order):
        replace = True

        if page in pages_table:
            replace = False
        else:
            page_index_to_swap = -1
            for i in range(capacity):
                if pages_table[i] == -1:
                    replace = False
                    page_index_to_swap = i
                    break

            if page_index_to_swap == -1:
                for i in range(capacity):
                    if priorities[i] == delete_priority:
                        pages_table[i] = page
                        priorities[i] = insert_priority
                        insert_priority += 1
                        delete_priority += 1
                        break
            else:
                pages_table[page_index_to_swap] = page
                priorities[page_index_to_swap] = insert_priority
                insert_priority += 1
        output_table.append(["-" if pages_table[i] == -1 else pages_table[i] for i in range(capacity)] + [page, 1 if replace else 0])

    return output_table


str_table = FIFO(capacity, pages_order)
FIFO_df = pd.DataFrame(str_table, columns=df_header)
FIFO_df.to_csv("./results/FIFO.csv")


def LRU(capacity: int, pages_order: list):
    output_table = []

    pages_table = [-1 for i in range(capacity)]
    priorities = [0 for i in range(capacity)]

    for tick, page in enumerate(pages_order):
        for i in range(capacity):
            priorities[i] += 1

        replace = True

        if page in pages_table:
            replace = False
            for i in range(capacity):
                if pages_table[i] == page:
                    priorities[i] = 0
                    break
        else:
            page_index_to_swap = -1
            for i in range(capacity):
                if pages_table[i] == -1:
                    replace = False
                    page_index_to_swap = i
                    break

            if page_index_to_swap == -1:
                max_priority_index = 0
                max_priority = priorities[max_priority_index]
                for i in range(capacity):
                    if max_priority < priorities[i]:
                        max_priority_index = i
                        max_priority = priorities[i]

                pages_table[max_priority_index] = page
                priorities[max_priority_index] = 0
            else:
                pages_table[page_index_to_swap] = page
                priorities[page_index_to_swap] = 0
        output_table.append(["-" if pages_table[i] == -1 else pages_table[i] for i in range(capacity)] + [page, 1 if replace else 0])

    return output_table


LRU_df = pd.DataFrame(LRU(capacity, pages_order), columns=df_header)
LRU_df.to_csv("./results/LRU.csv")


def optimal(capacity: int, pages_order: list):
    output_table = []

    pages_table = [-1 for i in range(capacity)]
    priorities = [0 for i in range(capacity)]

    for tick, page in enumerate(pages_order):
        replace = True

        if page in pages_table:
            replace = False
        else:
            page_index_to_swap = -1
            for i in range(capacity):
                if pages_table[i] == -1:
                    replace = False
                    page_index_to_swap = i
                    break

            if page_index_to_swap == -1:
                max_time_to_swap = 0
                for i in range(capacity):
                    found_one = False
                    for j in range(tick + 1, len(pages_order)):
                        if pages_order[j] == pages_table[i]:
                            found_one = True
                            if max_time_to_swap < j - tick:
                                max_time_to_swap = j - tick
                                page_index_to_swap = i
                            break

                    if not found_one:
                        page_index_to_swap = i
                        break

            pages_table[page_index_to_swap] = page

        output_table.append(
            ["-" if pages_table[i] == -1 else "{}{}".format(pages_table[i], "+" * (priorities[i] - 1)) for i in range(capacity)] + [page, 1 if replace else 0])

    return output_table


optimal_df = pd.DataFrame(optimal(capacity, pages_order), columns=df_header)
optimal_df.to_csv("./results/optimal.csv")

capacities = [i for i in range(1, max(pages_order) + 1)]

calculations = [optimal(i, pages_order) for i in capacities]

page_faults_ratio = [sum([calculations[i][j][-1] for j in range(len(calculations[i]))]) / len(pages_order) for i in range(len(calculations))]

print(page_faults_ratio)

max_page_faults_ratio = 0.05

for i, faults_ratio in enumerate(page_faults_ratio):
    if faults_ratio < max_page_faults_ratio:
        print(i + 1)
        break

plt.plot(capacities, page_faults_ratio, "o-")
plt.show()

