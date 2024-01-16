import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.interpolate import CubicSpline
from scipy import interpolate
from datetime import datetime

def total_and_zeros(address):
    df = pd.read_csv(address+'.csv')
    
    if 'Author XX prior pubs (insert more columns if reqd)' in df.columns:
        df_pubs = df[['Author 1 prior pubs','Author 2 prior pubs','Author 3 prior pubs','Author 4 prior pubs','Author XX prior pubs (insert more columns if reqd)']]
    elif 'Author XX prior pubs' in df.columns:
        df_pubs = df[['Author 1 prior pubs','Author 2 prior pubs','Author 3 prior pubs','Author 4 prior pubs','Author XX prior pubs']]
    df_pubs.fillna(0,inplace = True)
    df_pubs = df_pubs.astype(int)
    df_sum = df_pubs.sum().sum()
    pzpp = []
    for i in range(df_pubs.shape[0]):
        if df_pubs.sum(axis = 1)[i] == 0:
            pzpp.append(1)
        else:
            pzpp.append(0)
    df['pzpp'] = pzpp
    df.to_csv(address+'.csv', index = False)
    zeroes  = df_pubs.shape[0] - df_sum
    total = df_pubs.shape[0]
    return [address, total, zeroes, round(zeroes/total*100,2)]

def address_prior_counts(start, i1, i2):
    address_list = []
    for i in range(3,23):
        for j in [i1,i2]:
            if i < 10:
                cur = start + f'_200{i}_{j}'
            else:
                cur = start + f'_20{i}_{j}'
            address_list.append(cur)
    prior_counts = []
    count = 0
    for address in address_list:
        count += 1
        prior_counts.append(total_and_zeros(address))
    df = pd.DataFrame(np.array(prior_counts), columns = ['Issue', 'No. of Papers', 'No. of PZPPs', '% of PZPPs'])
    df.to_excel(start+'_prior_counts.xlsx')
    return address_list, prior_counts


def pzpp_vs_issue(prior_counts):
    zeroes = [prior_counts[i][2] for i in range(len(prior_counts))]
    names = [prior_counts[i][0] for i in range(len(prior_counts))]
    pers = [prior_counts[i][3] for i in range(len(prior_counts))]
    def addlabels(x,y):
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha = 'center')
    plt.figure(figsize = (12,10))
    plt.xticks(rotation=90), plt.bar(names, zeroes, color = 'blueviolet')
    plt.title('No. of PZPP vs Issue'), plt.xlabel('Issue'), plt.ylabel('No. of PZPP')
    addlabels(names,zeroes)
    plt.show()
    
    plt.figure(figsize = (12,10))
    plt.xticks(rotation=90), plt.bar(names, pers, color = 'blueviolet')
    plt.title('% of PZPP vs Issue'), plt.xlabel('Issue'), plt.ylabel('Percentage of PZPP')
    addlabels(names, pers)
    plt.show()
    return zeroes, names, pers


def averages(prior_counts, zeroes, names, pers):
    tot_data_points = 0
    for i in range(40):
        tot_data_points += prior_counts[i][1]
    weighted_zero_avg = round(sum(zeroes)/tot_data_points*100,2)
    unweighted_zero_avg = round(sum(pers)/40,2)
    print(f"% of PZPPs (Unweighted Avg): {unweighted_zero_avg}%")
    print(f"% of PZPPs (Weighted Avg): {weighted_zero_avg}%")    
    five_w, two_w, two_p = [], [], []
    f = [prior_counts[i][1] for i in range(len(prior_counts))]
    cnt,ttl = 0, 0
    for i in range(40):
        if i%10==9:
            cnt=cnt+zeroes[i]
            ttl= ttl+f[i]
            five_w.append(round(cnt/ttl*100,2))
            cnt, ttl=0, 0    
        else:
            cnt=cnt+zeroes[i]
            ttl= ttl+f[i]
    cnt, ttl = 0, 0
    for i in range(40):
        if i%4==3:
            cnt=cnt+zeroes[i]
            ttl= ttl+f[i]
            two_w.append(round(cnt/ttl*100,2))
            two_p.append(round(ttl/4,2))
            cnt, ttl=0, 0 
        else:
            cnt=cnt+zeroes[i]
            ttl= ttl+f[i]
    tx = np.array([i for i in range(1,41,4)])
    ty = interpolate.interp1d(tx, np.array(two_w))
    txx = np.arange(1,36,0.1)
    tyy = ty(txx)
    five_w = [weighted_zero_avg] + five_w + [weighted_zero_avg]
    fx = np.array([i for i in range(-5,55,10)])
    fy = interpolate.interp1d(fx, np.array(five_w))
    fxx = np.arange(0,40,0.1)
    fyy = fy(fxx)
    two_p  = [f[0]] + two_p + [f[-1]]
    tpx = np.array([i for i in range(-3,45,4)])
    tpy = interpolate.interp1d(tpx, np.array(two_p))
    tpxx = np.arange(0,40,0.1)
    tpyy = tpy(tpxx)
    plt.figure(figsize = (12,8))
    plt.xticks(rotation = 90)
    plt.title('Total No. of Papers vs Issue')
    plt.xlabel('Issues'), plt.ylabel('No. of Papers')
    plt.plot(names, f, label = 'Issue Wise', color = 'blueviolet', linewidth = 1.5)
    plt.plot(tpxx, tpyy, label = 'Two year Average', color = 'black', linewidth = 2)
    plt.xlim(-1,39), plt.legend()
    plt.show()
    plt.figure(figsize = (12,8))
    plt.title('Temporal Evolution of PZPPs')
    plt.xticks(rotation = 90)
    wl = [weighted_zero_avg for _ in range(40) ]
    ul = [unweighted_zero_avg for _ in range(40)]
    plt.plot(names, pers, color = 'blueviolet', linewidth = 1)
    plt.plot(names, wl, linestyle = 'dashed', color = 'blue', label = 'Total Weighted Average')
    plt.plot(names, ul, linestyle = 'dotted', color = 'red', label = 'Total Unweighted Average')
    plt.xlabel('Issues'), plt.ylabel('% of PZPPs')
    plt.plot(txx, tyy,label = 'Two year Average', linewidth = 1.75, color = 'green')
    plt.plot(fxx, fyy, label = 'Five year Average', linewidth = 2.5, color = 'black')
    plt.legend()
    plt.show()


def departments_freq(start, i1, i2):
    departments_freq = {}
    for k in range(2003,2023):
        for j in range(i1,i2+1,3):
            df = pd.read_csv(start+'_'+ str(k)+'_'+ str(j)+'x.csv')
            if 'Department' in df.columns:
                deps = np.array(df['Department'])
                for i in range(df.shape[0]):
                    dep = deps[i].strip().lower()
                    if dep in departments_freq:
                        departments_freq[dep] += 1
                    else:
                        departments_freq[dep] = 1
    return departments_freq

def dept_adder(ls, dept, adders):
    for d in adders:
        if dept in ls:
            ls[dept] += ls[d]
        else:
            ls[dept] = ls[d]
        del ls[d]    
    ls = {k:v for k,v in sorted(ls.items(), key = lambda x: x[1], reverse = True)}


def dept_merger(ls, dept, mergers):
    for d in mergers:
        if dept in ls:
            ls[dept].extend(ls[d])
        else:
            ls[dept] = ls[d]
        del ls[d]
    ls = {k:v for k,v in sorted(ls.items(), key = lambda x: x[1], reverse = True)}

from datetime import datetime

def time_between_dates(date_str1, date_str2):
    dt = date_str1.split()
    if len(dt) == 3:
        date_format = "%B %d, %Y"
    else:
        if len(date_str1) == 6:
            date_format = "%b-%y"
        else:
            date_format = "%B %Y"
    date1 = datetime.strptime(date_str1, date_format)
    date2 = datetime.strptime(date_str2, date_format)
    
    time_difference = date2 - date1
    days = time_difference.days
    weeks = days//7 
    return weeks

def days_for_publication(duration_str):
    time_units = {
        'day': 1,'days': 1,
        'week': 7,'weeks': 7,
        'month': 30,'months': 30,
        'year': 365,'years': 365,
    }
    words = duration_str.split()
    days = 0
    current_value = 0
    current_unit = None

    for word in words:
        if word.isdigit() or (word.replace('.', '', 1).isdigit() and '.' in word):
            current_value = float(word) if '.' in word else int(word)
        elif word in time_units:  
            current_unit = word
            days += current_value * time_units[current_unit]
    return int(days)//7

def received_accepted_time(start, i1, i2):
    for k in range(2003,2023):
        for j in range(i1,i2+1,3):
            address = start+'_'+ str(k)+'_'+ str(j)+'x.csv'
            df = pd.read_csv(address)
            if 'Received Date' and 'Accepted Date' in df.columns:
                times = []
                for i in range(df.shape[0]):               
                    time = time_between_dates(df['Received Date'][i], df['Accepted Date'][i])
                    times.append(time)
                times = np.array(times)
                df['Time Taken'] = times
            else:
                times = []
                for i in range(df.shape[0]):        
                    time = days_for_publication(df['Time with Authors'][i])
                    times.append(time)
                times = np.array(times)
                df['Time Taken'] = times
            df.to_csv(address)

def departments_wise_time(start, i1, i2):
    departments_time = {}
    for k in range(2003,2023):
        for j in range(i1,i2+1,3):
            df = pd.read_csv(start+'_'+ str(k)+'_'+ str(j)+'x.csv')
            if 'Department' in df.columns:
                deps = np.array(df['Department'])
                time_taken = np.array(df['Time Taken'])
                for i in range(df.shape[0]):
                    dep = deps[i].strip().lower()
                    if dep in departments_time:
                        departments_time[dep].append(time_taken[i])
                    else:
                        departments_time[dep] = []
                        departments_time[dep].append(time_taken[i])
    return departments_time

def departments_wise_pzpp(start, i1, i2):
    departments_pzpp = {}
    for k in range(2003,2023):
        for j in range(i1,i2+1,3):
            df = pd.read_csv(start+'_'+ str(k)+'_'+ str(j)+'x.csv')
            if 'Department' in df.columns:
                deps = np.array(df['Department'])    
                
                if 'Author XX prior pubs (insert more columns if reqd)' in df.columns:
                    df_pubs = df[['Author 1 prior pubs','Author 2 prior pubs','Author 3 prior pubs','Author 4 prior pubs','Author XX prior pubs (insert more columns if reqd)']]
                elif 'Author XX prior pubs' in df.columns:
                    df_pubs = df[['Author 1 prior pubs','Author 2 prior pubs','Author 3 prior pubs','Author 4 prior pubs','Author XX prior pubs']]
                df_pubs.fillna(0,inplace = True)
                df_pubs = df_pubs.astype(int) 
                df_sum = df_pubs.sum(axis = 1)    
                for i in range(df.shape[0]):
                    
                    if df_sum[i] == 0:
                        if deps[i].strip().lower() in departments_pzpp:
                            departments_pzpp[deps[i].strip().lower()] += 1
                        else:
                            departments_pzpp[deps[i].strip().lower()] = 1
                    else:
                        if deps[i] not in departments_pzpp:
                            departments_pzpp[deps[i].strip().lower()] = 0
    return departments_pzpp 

def country(start, i1, i2):
    for k in range(2003,2023):
        for j in range(i1,i2+1,3):
            df = pd.read_csv(start+'_'+ str(k)+'_'+ str(j)+'x.csv')
            if 'Author XX country of affiliation (insert more columns if reqd)' in df.columns:
                df_pubs = df[['Author 1 country of affiliation','Author 2 country of affiliation','Author 3 country of affiliation','Author 4 country of affiliation','Author XX country of affiliation (insert more columns if reqd)']]
            elif 'Author XX country of affiliation' in df.columns:
                df_pubs = df[['Author 1 country of affiliation','Author 2 country of affiliation','Author 3 country of affiliation','Author 4 country of affiliation','Author XX country of affiliation']]
            df_pubs.fillna(0,inplace = True)
            df_pubs = df_pubs.astype(int)
            af = df_pubs.to_numpy()
            aff = []
            for i in range(af.shape[0]):
                if -1 in af[i]:
                    aff.append(-1)
                elif 1 in af[i]:
                    aff.append(1)
                else:
                    aff.append(0)
            df['affiliations'] = aff
            df.to_csv(start+'_'+ str(k)+'_'+ str(j)+'x.csv') 

def affiliation_pzpp(start,i1,i2):
    issue_ap = []
    global_ap = {-1:0, 0: 0, 1:0}
    tot_aff = {-1:0, 0:0, 1:0}
    for k in range(2003,2023): 
        for j in range(i1,i2+1,3):
            df = pd.read_csv(start+'_'+ str(k)+'_'+ str(j)+'x.csv')
            ap = {-1: 0, 0: 0, 1:0}
            for i in range(df.shape[0]):
                if df['pzpp'][i] == 1:
                    ap[int(df[f'affiliations'][i])] += 1
                    global_ap[int(df['affiliations'][i])] += 1
                tot_aff[int(df['affiliations'][i])] += 1
            issue_ap.append(ap)
    return global_ap, issue_ap, tot_aff

def affiliation_dept(start, i1, i2):
    departments_aff = {}
    for k in range(2003,2023):
        for j in range(i1,i2+1,3):
            df = pd.read_csv(start+'_'+ str(k)+'_'+ str(j)+'x.csv')
            if 'Department' in df.columns:
                deps = np.array(df['Department'])
                affs = np.array(df['affiliations'])
                for i in range(df.shape[0]):
                    dep = deps[i].strip().lower()
                    if dep in departments_aff:
                        departments_aff[dep].append(affs[i])
                    else:
                        departments_aff[dep] = []
                        departments_aff[dep].append(affs[i])
    return departments_aff


def dept_wise_affiliation(start, departments_aff):
    department_names = list(departments_aff.keys())
    department_names2 = list(departments_aff.keys())    
    aff_count_dep = {}
    aff_dept = []
    cnt = 0
    for deps in department_names2:
        ap = {-1: 0, 0: 0, 1:0}
        de_aff = departments_aff[deps]
        for i in range(len(de_aff)):
            ap[int(de_aff[i])]+=1 
        aff_count_dep[department_names[cnt]]=ap
        iss = []
        iss.append(department_names[cnt])
        iss.append(ap[-1])
        iss.append(ap[0])
        iss.append(ap[1]) 
        tot = ap[-1]+ap[0]+ap[1]
        iss.append(tot)
        iss.append(round(ap[-1]/tot*100,2))
        iss.append(round(ap[0]/tot*100,2))
        iss.append(round(ap[1]/tot*100,2)) 
        aff_dept.append(iss)
        cnt+=1
    df = pd.DataFrame(np.array(aff_dept), columns = ['Department','-1 affiliation','0 affiliation','1 affiliation','total', '% of -1 aff' , '% of 0 aff' ,'% of 1 aff'])
    df.to_excel(start+'_dept_wise_aff.xlsx')

def issue_wise_affiliation(start,i1,i2):
    aff_issue_wise = {}
    aff_issue = []
    aff_overall = {-1:0, 0: 0, 1:0}
    for k in range(2003,2023):
        for j in range(i1,i2+1,3):
            df = pd.read_csv(start+'_'+ str(k)+'_'+ str(j)+'x.csv')
            aff = {-1: 0, 0: 0, 1:0}
            for i in range(df.shape[0]):    
                aff[int(df['affiliations'][i])] += 1
                aff_overall[int(df['affiliations'][i])] += 1
            aff_issue_wise[start+'_'+ str(k)+'_'+ str(j)]=aff
            iss = []
            iss.append(start+'_'+ str(k)+'_'+ str(j))
            iss.append(aff[-1])
            iss.append(aff[0])
            iss.append(aff[1]) 
            tot = aff[-1]+aff[0]+aff[1]
            iss.append(tot)
            iss.append(round(aff[-1]/tot*100,2))
            iss.append(round(aff[0]/tot*100,2))
            iss.append(round(aff[1]/tot*100,2))   
            aff_issue.append(iss)
    df = pd.DataFrame(np.array(aff_issue), columns = ['Issue','-1 affiliation','0 affiliation','1 affiliation','total', '% of -1 aff' , '% of 0 aff' ,'% of 1 aff'])
    df.to_excel(start+'_issue_wise_aff.xlsx')

def year_wise_affiliation(start, i1, i2):
    aff_year_wise={}
    aff_year = []
    for k in range(2003,2023):
        for j in range(i1,i2+1,3):
            df = pd.read_csv(start+'_'+ str(k)+'_'+ str(j)+'x.csv')
            aff = {-1: 0, 0: 0, 1:0}
            for i in range(df.shape[0]):    
                aff[int(df['affiliations'][i])] += 1
            aff_year_wise[j]=aff
            iss = []
            iss.append(j)
            iss.append(aff[-1])
            iss.append(aff[0])
            iss.append(aff[1])
            tot = aff[-1]+aff[0]+aff[1]
            iss.append(tot)
            iss.append(round(aff[-1]/tot*100,2))
            iss.append(round(aff[0]/tot*100,2))
            iss.append(round(aff[1]/tot*100,2))
            aff_year.append(iss)
    df = pd.DataFrame(np.array(aff_year), columns = ['year','-1 affiliation','0 affiliation','1 affiliation','total', '% of -1 aff' , '% of 0 aff' ,'% of 1 aff'])
    df.to_excel(start+'_year_wise_aff.xlsx')


def zero_country(start, i1, i2):
    zero_country_address = []
    for k in range(2003,2023):
        for j in range(i1,i2+1,3):
            df = pd.read_csv(start+'_'+ str(k)+'_'+ str(j)+'x.csv')
            times = np.array(df['Time Taken'])
            aff = np.array(df['affiliations'])
            for i in range(df.shape[0]):
                if int(aff[i]) == 0:
                    zero_country_address.append((start+'_'+ str(k)+'_'+ str(j), i))
    return zero_country_address

def department_total(department, deps, address_list):
    dep_issue = {}
    for address in address_list:
        df = pd.read_csv(address+'x.csv')
        count = 0
        if 'Department' in df.columns:
            departments = np.array(df['Department'])
            for i in departments:
                dep = i.lower()
                if dep in deps:
                    count += 1
        dep_issue[address] = count
    dep_year = {}
    for i in range(0,40,2):
        dep_year[address_list[i][3:7]] = dep_issue[address_list[i]] + dep_issue[address_list[i+1]]
    x = list(dep_issue.keys())
    y = list(dep_issue.values())
    plt.figure(figsize = (10,8))
    plt.xticks(rotation = 90)
    plt.title(f"{department.title()} (Issue Wise)")
    plt.ylabel("Number of Papers")
    plt.xlabel("Issues")
    plt.bar(x,y, color = 'blueviolet')
    plt.show()
    x = list(dep_year.keys())
    y = list(dep_year.values())
    plt.figure(figsize = (10,8))
    plt.xticks(rotation = 90)
    plt.title(f"{department.title()} (Year Wise)")
    plt.ylabel("Number of Papers")
    plt.xlabel("Years")
    plt.bar(x,y, color = 'blueviolet') 
    plt.show()
    return dep_issue, dep_year

def department_pzpp(department, deps, address_list):
    dep_pzpp_issue = {}
    for address in address_list:
        df = pd.read_csv(address+'x.csv')
        count = 0
        if 'Department' in df.columns:
            departments = np.array(df['Department'])
            pzpp = np.array(df['pzpp'])
            for i in range(len(departments)):
                dep = departments[i].lower()
                if dep in deps:
                    if int(pzpp[i]) == 1:
                        count += 1
        dep_pzpp_issue[address] = count
    dep_pzpp_year = {}
    for i in range(0,40,2):
        dep_pzpp_year[address_list[i][3:7]] = dep_pzpp_issue[address_list[i]] + dep_pzpp_issue[address_list[i+1]]
    x = list(dep_pzpp_issue.keys())
    y = list(dep_pzpp_issue.values())
    plt.figure(figsize = (10,8))
    plt.xticks(rotation = 90)
    plt.title(f"{department.title()} PZPP (Issue Wise)")
    plt.ylabel("Number of Papers")
    plt.xlabel("Issues")
    plt.bar(x,y, color = 'blueviolet')
    plt.show()
    x = list(dep_pzpp_year.keys())
    y = list(dep_pzpp_year.values())
    plt.figure(figsize = (10,8))
    plt.xticks(rotation = 90)
    plt.title(f"{department.title()} PZPP (Year Wise)")
    plt.ylabel("Number of Papers")
    plt.xlabel("Years")
    plt.bar(x,y, color = 'blueviolet') 
    plt.show()
    return dep_pzpp_issue, dep_pzpp_year