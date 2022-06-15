import pandas as pd
import os
import tkinter as ttk
from tkinter import filedialog
from tkinter import *
import re
import xlsxwriter

def load_file():
    path = filedialog.askdirectory()
    os.chdir(path)
    list_files = os.listdir('.')
    index_mac = list_files.index('.DS_Store')
    list_files = list_files[:index_mac] + list_files[index_mac + 1:]
    files = []
    for i in range(0, len(list_files)):
        files.append(path + '/' + list_files[i])
    print(files)

    time_input = [] 
    time_processing = [] 
    deadline = [] 
    task_number = []
    list_sum_delays_MINDL = []
    list_sum_delays_SF = []
    list_sum_delays_BTSF_1 = []
    list_sum_delays_BTSF_1_5 = []
    list_sum_delays_BTSF_2 = []



    for i in range(0, len(list_files)):
        df = pd.read_excel(list_files[i])
        for index, row in df.iterrows():
            time_input.append(row[0])
            time_processing.append(row[1])
            deadline.append(row[2])
        for i in range(1, len(time_input) + 1):
            task_number.append(i)
        count = len(time_processing)
        S = []
        deadline_min = max(deadline) + 1
        time, sum_delays, mark, index_for = 0, 0, 0, 0
        while count > 0:
            for i in range(0, len(time_processing)):
                if time_input[i] <= time:
                    if deadline[i] <= deadline_min:
                        deadline_min = deadline[i]
                        S_i = task_number[i]
                        index_for = i
                        mark = 1
            if mark == 0:
                time += 1
            else:
                S.append(str(S_i))
                time += time_processing[index_for]
                sum_delays += max(time - deadline[index_for], 0)
                del time_input[index_for]
                del time_processing[index_for]
                del deadline[index_for]
                del task_number[index_for]
                count -= 1
                mark, index_for = 0, 0
                if len(deadline) != 0:
                    deadline_min = max(deadline) + 1
                else:
                    deadline_min = 100000
        list_sum_delays_MINDL.append(sum_delays)
        time_processing.clear()
        time_input.clear()
        deadline.clear()
        task_number.clear()


    for i in range(0, len(list_files)):
        df = pd.read_excel(list_files[i])
        for index, row in df.iterrows():
            time_input.append(row[0])
            time_processing.append(row[1])
            deadline.append(row[2])
        for i in range(1, len(time_input) + 1):
            task_number.append(i)
        count = len(time_processing)
        S = []
        K_min = 10000000000
        time, sum_delays, mark, index_for = 0, 0, 0, 0
        while count > 0:
            for i in range(0, len(time_processing)):
                if time_input[i] <= time:
                    K_i = (deadline[i] - time) / time_processing[i]
                    if K_i <= K_min:
                        K_min = K_i
                        S_i = task_number[i]
                        index_for = i
                        mark = 1
            if mark == 0:
                time += 1
            else:
                S.append(str(S_i))
                time += time_processing[index_for]
                sum_delays += max(time - deadline[index_for], 0)
                del time_input[index_for]
                del time_processing[index_for]
                del deadline[index_for]
                del task_number[index_for]
                count -= 1
                mark, index_for = 0, 0
                K_min = 10000000000
        list_sum_delays_SF.append(sum_delays)
        time_processing.clear()
        time_input.clear()
        deadline.clear()
        task_number.clear()
        

    alpha = 1
    for i in range(0, len(list_files)):
        df = pd.read_excel(list_files[i])
        for index, row in df.iterrows():
            time_input.append(row[0])
            time_processing.append(row[1])
            deadline.append(row[2])
        for i in range(1, len(time_input) + 1):
            task_number.append(i)
        count = len(time_processing)
        S = []
        K_min = 1000000000000000
        time, sum_delays, mark, mark_1, sum_time, count_time,  = 0, 0, 0, 0, 0, 0 
        big_time, number_big, S_i, index_for, K_min_2, S_i_2, index_for_2 = 0, 0, 0, 0, 0, 0, 0
        while count > 0:
            if len(time_processing) != 1:
                for i in range(0, len(time_processing)):
                    if time_input[i] <= time:
                        sum_time += time_processing[i]
                        count_time += 1
                        if big_time < time_processing[i]:
                            big_time = time_processing[i]
                            number_big = task_number[i]
                if count_time != 1 and count_time != 0:
                    if big_time > alpha*(sum_time - big_time) / (count_time - 1):
                        mark_1 = 1
            for i in range(0, len(time_processing)):
                if time_input[i] <= time:
                    K_i = (deadline[i] - time) / time_processing[i]
                    if K_i <= K_min:
                        if K_min != 1000000000000000:
                            K_min_2 = K_min
                        else:
                            K_min_2 = K_i
                        K_min = K_i
                        if S_i != -1:
                            S_i_2 = S_i
                        else:
                            S_i_2 = task_number[i]
                        S_i = task_number[i]
                        if index_for != -1:
                            index_for_2 = index_for
                        else:
                            index_for_2 = i
                        index_for = i 
                        mark = 1
            if mark_1 == 1 and S_i == number_big:
                print('BIG')
                K_min = K_min_2
                S_i = S_i_2
                index_for = index_for_2
            if mark == 0:
                time += 1
            else:
                S.append(str(S_i))
                time += time_processing[index_for]
                sum_delays += max(time - deadline[index_for], 0)
                del time_input[index_for]
                del time_processing[index_for]
                del deadline[index_for]
                del task_number[index_for]
                count -= 1
                mark = 0
                K_min = 1000000000000000
                mark_1 = 0
                big_time = 0
                number_big = 0
                index_for_2 = -1
                index_for = -1
                K_min_2 = 0
                S_i_2 = 0
                S_i = -1
                sum_time = 0
                count_time = 0
        list_sum_delays_BTSF_1.append(sum_delays)
        time_processing.clear()
        time_input.clear()
        deadline.clear()
        task_number.clear()
        
    name_dir = path[max([m.start() for m in re.finditer('/', path)]):]
    path_1 = '/Users/elizavetatarasova/Documents/Result' + name_dir
    os.chdir(path_1)
    name = name_dir[1:] + '.xlsx'
    book = xlsxwriter.Workbook(name)
    sheet = book.add_worksheet('Sheet1')
    sheet.write(0, 0, 'Результаты MINDL')
    sheet.write(0, 1, 'Результаты SF')
    sheet.write(0, 2, 'Результаты BTSF')
    sheet.write(0, 3, 'SF относительно MINDL')
    sheet.write(0, 4, 'BTSF относительно SF')
    sheet.write(0, 5, 'BTSF относительно MINDL')
    comprasion_1, comprasion_2, comprasion_3 = [], [], []
    for j in range(0, len(list_sum_delays_MINDL)):
        comprasion_1.append(list_sum_delays_MINDL[j] / list_sum_delays_SF[j])
        comprasion_2.append(list_sum_delays_SF[j] / list_sum_delays_BTSF_1[j])
        comprasion_2.append(list_sum_delays_MINDL[j] / list_sum_delays_BTSF_1[j])
    plus_1 = [pl for pl in comprasion_1 if pl > 1]
    plus_2 = [pl for pl in comprasion_2 if pl > 1]
    plus_2 = [pl for pl in comprasion_2 if pl > 1]
    for j in range(0, len(list_sum_delays_BTSF_1)):
        sheet.write(j + 1, 0, list_sum_delays_MINDL[j])
        sheet.write(j + 1, 1, list_sum_delays_SF[j])
        sheet.write(j + 1, 2, list_sum_delays_BTSF_1[j])
        sheet.write(j + 1, 3, comprasion_1[j])
        sheet.write(j + 1, 4, comprasion_2[j])
        sheet.write(j + 1, 5, comprasion_3[j])
    book.close()

root = Tk()
root.geometry('200x150')
root.title('Report')
l1 = ttk.Label(root, text = 'Please press button', font = 'Arial 14')
l2 = ttk.Label(root, text = 'and select path', font = 'Arial 14')
btn = ttk.Button(root, text = 'Select', command = lambda : load_file())
l1.pack(side=TOP, pady=5)
l2.pack(side=TOP)
btn.pack(side=TOP, pady=20)

mainloop()
