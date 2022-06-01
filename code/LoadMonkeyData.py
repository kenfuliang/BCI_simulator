import pandas as pd
from Dataset import RigCDataset
    #if (date =='2010-10-27') & (source=='Gilja') & (monkey=='Jenkins'):
    #    ST_list = {
    #        'hand':[14],
    #        'VKF':[9],
    #        'ReFIT':[12],
    #    }
    #if (date =='2010-10-28') & (source=='Gilja') & (monkey=='Jenkins'):
    #    ST_list = {
    #        'hand':[2],
    #        'VKF':[8],
    #        'ReFIT':[11],
    #    }
    #if (date =='2010-10-29') & (source=='Gilja') & (monkey=='Jenkins'):
    #    ST_list = {
    #        'hand':[3],
    #        'VKF':[8],
    #        'ReFIT':[10],
    #    }
    #if (date =='2010-11-02') & (source=='Gilja') & (monkey=='Jenkins'):
    #    ST_list = {
    #        'hand':[3],
    #        'VKF':[8],
    #        'ReFIT':[10],
    #    }
    #if (date =='2010-10-27') & (source=='Gilja') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[8],
    #        'VKF':[12],
    #        'ReFIT':[14],
    #    }
    #if (date =='2010-10-28') & (source=='Gilja') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[2],
    #        'VKF':[12],
    #        'ReFIT':[14],
    #    }

    #if (date =='2010-10-29') & (source=='Gilja') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[2],
    #        'VKF':[7],
    #        'ReFIT':[9],
    #    }

    #if (date =='2010-11-02') & (source=='Gilja') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[2],
    #        'VKF':[8],
    #        'ReFIT':[10],
    #    }

    #if (date =='2011-05-12') & (source=='Fan') & (monkey=='Jenkins'):
    #    ST_list = {
    #        'hand':[1],
    #        'FIT':[6,7],
    #        'ReFIT':[13],
    #    }

    #if (date =='2011-05-13') & (source=='Fan') & (monkey=='Jenkins'):
    #    ST_list = {
    #        'hand':[9],
    #        'FIT':[13,16,20],
    #        'ReFIT':[11,14,18,22],
    #    }

    #if (date =='2011-05-16') & (source=='Fan') & (monkey=='Jenkins'):
    #    ST_list = {
    #        'hand':[1],
    #        'FIT':[10,14],
    #        'ReFIT':[8,12],
    #    }

    #if (date =='2011-05-17') & (source=='Fan') & (monkey=='Jenkins'):
    #    ST_list = {
    #        'hand':[2],
    #        'FIT':[14],
    #        'ReFIT':[9,16],
    #    }

    #if (date =='2011-05-19') & (source=='Fan') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[1],
    #        'VKF':[10,14,18],
    #        'FIT':[12,16,20],
    #    }

    #if (date =='2011-05-20') & (source=='Fan') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[1],
    #        'VKF':[7,11,15,19],
    #        'FIT':[5,9,13,17],
    #    }

    #if (date =='2011-05-24') & (source=='Fan') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[1],
    #        'VKF':[5,9,10,16,17],
    #        'FIT':[7,14],
    #    }
    #if (date =='2011-04-07') & (source=='Fan') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[1],
    #        'FIT':[11,15,19],
    #        'ReFIT':[13,17],
    #    }
    #if (date =='2011-04-08') & (source=='Fan') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[1],
    #        'FIT':[11,15,19,23,27],
    #        'ReFIT':[9,13,17,21,25],
    #    }
    #if (date =='2011-04-15') & (source=='Fan') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[1],
    #        'FIT':[7,9],
    #        'ReFIT':[8],
    #    }
    #if (date =='2011-04-18') & (source=='Fan') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[1],
    #        'VKF':[18],
    #        'FIT':[16],
    #    }
    #if (date =='2011-04-21') & (source=='Fan') & (monkey=='Larry'):
    #    ST_list = {
    #        'hand':[1],
    #        'VKF':[16,21],
    #        'FIT':[10,19],
    #    }




rigC_df = pd.DataFrame()
rigC_df = rigC_df.append({'date':'2010-10-27','source':'Gilja','monkey':'Jenkins','decoder':'PVKF','ST':7},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-27','source':'Gilja','monkey':'Jenkins','decoder':'hand','ST':14},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-27','source':'Gilja','monkey':'Jenkins','decoder':'VKF','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-27','source':'Gilja','monkey':'Jenkins','decoder':'ReFIT','ST':12},ignore_index=True)

rigC_df = rigC_df.append({'date':'2010-10-28','source':'Gilja','monkey':'Jenkins','decoder':'hand','ST':2},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-28','source':'Gilja','monkey':'Jenkins','decoder':'PVKF','ST':5},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-28','source':'Gilja','monkey':'Jenkins','decoder':'VKF','ST':8},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-28','source':'Gilja','monkey':'Jenkins','decoder':'ReFIT','ST':11},ignore_index=True)

rigC_df = rigC_df.append({'date':'2010-10-29','source':'Gilja','monkey':'Jenkins','decoder':'hand','ST':3},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-29','source':'Gilja','monkey':'Jenkins','decoder':'PVKF','ST':5},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-29','source':'Gilja','monkey':'Jenkins','decoder':'VKF','ST':8},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-29','source':'Gilja','monkey':'Jenkins','decoder':'ReFIT','ST':10},ignore_index=True)

rigC_df = rigC_df.append({'date':'2010-11-02','source':'Gilja','monkey':'Jenkins','decoder':'hand','ST':3},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-11-02','source':'Gilja','monkey':'Jenkins','decoder':'PVKF','ST':5},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-11-02','source':'Gilja','monkey':'Jenkins','decoder':'VKF','ST':8},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-11-02','source':'Gilja','monkey':'Jenkins','decoder':'ReFIT','ST':10},ignore_index=True)


rigC_df = rigC_df.append({'date':'2010-10-27','source':'Gilja','monkey':'Larry','decoder':'PVKF','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-27','source':'Gilja','monkey':'Larry','decoder':'hand','ST':8},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-27','source':'Gilja','monkey':'Larry','decoder':'VKF','ST':12},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-27','source':'Gilja','monkey':'Larry','decoder':'ReFIT','ST':14},ignore_index=True)

rigC_df = rigC_df.append({'date':'2010-10-28','source':'Gilja','monkey':'Larry','decoder':'hand','ST':2},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-28','source':'Gilja','monkey':'Larry','decoder':'PVKF','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-28','source':'Gilja','monkey':'Larry','decoder':'VKF','ST':12},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-28','source':'Gilja','monkey':'Larry','decoder':'ReFIT','ST':14},ignore_index=True)

rigC_df = rigC_df.append({'date':'2010-10-29','source':'Gilja','monkey':'Larry','decoder':'hand','ST':2},ignore_index=True)
#rigC_df = rigC_df.append({'date':'2010-10-29','source':'Gilja','monkey':'Larry','decoder':'PVKF','ST':5},ignore_index=True) # dirty
rigC_df = rigC_df.append({'date':'2010-10-29','source':'Gilja','monkey':'Larry','decoder':'VKF','ST':7},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-10-29','source':'Gilja','monkey':'Larry','decoder':'ReFIT','ST':9},ignore_index=True)

rigC_df = rigC_df.append({'date':'2010-11-02','source':'Gilja','monkey':'Larry','decoder':'hand','ST':2},ignore_index=True)
#rigC_df = rigC_df.append({'date':'2010-11-02','source':'Gilja','monkey':'Larry','decoder':'PVKF','ST':5},ignore_index=True) # dirty
rigC_df = rigC_df.append({'date':'2010-11-02','source':'Gilja','monkey':'Larry','decoder':'VKF','ST':8},ignore_index=True)
rigC_df = rigC_df.append({'date':'2010-11-02','source':'Gilja','monkey':'Larry','decoder':'ReFIT','ST':10},ignore_index=True)



rigC_df = rigC_df.append({'date':'2011-05-12','source':'Fan','monkey':'Jenkins','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-12','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':6},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-12','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':7},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-12','source':'Fan','monkey':'Jenkins','decoder':'PVKF','ST':11},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-12','source':'Fan','monkey':'Jenkins','decoder':'ReFIT','ST':13},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-05-13','source':'Fan','monkey':'Jenkins','decoder':'hand','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-13','source':'Fan','monkey':'Jenkins','decoder':'PVKF','ST':6},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-13','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':13},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-13','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':16},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-13','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':20},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-13','source':'Fan','monkey':'Jenkins','decoder':'ReFIT','ST':11},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-13','source':'Fan','monkey':'Jenkins','decoder':'ReFIT','ST':14},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-13','source':'Fan','monkey':'Jenkins','decoder':'ReFIT','ST':18},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-13','source':'Fan','monkey':'Jenkins','decoder':'ReFIT','ST':22},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-05-16','source':'Fan','monkey':'Jenkins','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-16','source':'Fan','monkey':'Jenkins','decoder':'PVKF','ST':4},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-16','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':10},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-16','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':14},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-16','source':'Fan','monkey':'Jenkins','decoder':'ReFIT','ST':8},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-16','source':'Fan','monkey':'Jenkins','decoder':'ReFIT','ST':12},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-05-17','source':'Fan','monkey':'Jenkins','decoder':'hand','ST':2},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-17','source':'Fan','monkey':'Jenkins','decoder':'PVKF','ST':5},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-17','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':14},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-17','source':'Fan','monkey':'Jenkins','decoder':'ReFIT','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-17','source':'Fan','monkey':'Jenkins','decoder':'ReFIT','ST':16},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-05-19','source':'Fan','monkey':'Jenkins','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-19','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':10},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-19','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':14},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-19','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':18},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-19','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':12},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-19','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':16},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-19','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':20},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-05-20','source':'Fan','monkey':'Jenkins','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-20','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':7},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-20','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':11},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-20','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':15},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-20','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':19},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-20','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':5},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-20','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-20','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':13},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-20','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':17},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-05-24','source':'Fan','monkey':'Jenkins','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-24','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':5},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-24','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-24','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':10},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-24','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':16},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-24','source':'Fan','monkey':'Jenkins','decoder':'VKF','ST':17},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-24','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':7},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-05-24','source':'Fan','monkey':'Jenkins','decoder':'FIT','ST':14},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-04-07','source':'Fan','monkey':'Larry','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-07','source':'Fan','monkey':'Larry','decoder':'PVKF','ST':7},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-07','source':'Fan','monkey':'Larry','decoder':'PVKF','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-07','source':'Fan','monkey':'Larry','decoder':'FIT','ST':11},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-07','source':'Fan','monkey':'Larry','decoder':'FIT','ST':15},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-07','source':'Fan','monkey':'Larry','decoder':'FIT','ST':19},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-07','source':'Fan','monkey':'Larry','decoder':'ReFIT','ST':13},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-07','source':'Fan','monkey':'Larry','decoder':'ReFIT','ST':17},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'PVKF','ST':4},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'FIT','ST':11},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'FIT','ST':15},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'FIT','ST':19},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'FIT','ST':23},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'FIT','ST':27},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'ReFIT','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'ReFIT','ST':13},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'ReFIT','ST':17},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'ReFIT','ST':21},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-08','source':'Fan','monkey':'Larry','decoder':'ReFIT','ST':25},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-04-15','source':'Fan','monkey':'Larry','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-15','source':'Fan','monkey':'Larry','decoder':'PVKF','ST':3},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-15','source':'Fan','monkey':'Larry','decoder':'FIT','ST':7},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-15','source':'Fan','monkey':'Larry','decoder':'FIT','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-15','source':'Fan','monkey':'Larry','decoder':'ReFIT','ST':8},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-04-18','source':'Fan','monkey':'Larry','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-18','source':'Fan','monkey':'Larry','decoder':'VKF','ST':18},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-18','source':'Fan','monkey':'Larry','decoder':'FIT','ST':16},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-04-21','source':'Fan','monkey':'Larry','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-21','source':'Fan','monkey':'Larry','decoder':'VKF','ST':16},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-21','source':'Fan','monkey':'Larry','decoder':'VKF','ST':21},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-21','source':'Fan','monkey':'Larry','decoder':'FIT','ST':10},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-04-21','source':'Fan','monkey':'Larry','decoder':'FIT','ST':19},ignore_index=True)

## Sussillo-Jenkins
rigC_df = rigC_df.append({'date':'2011-02-04','source':'Sussillo','monkey':'Jenkins','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-04','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':4},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-04','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':9},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-04','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':13},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-04','source':'Sussillo','monkey':'Jenkins','decoder':'FORCE','ST':7},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-04','source':'Sussillo','monkey':'Jenkins','decoder':'FORCE','ST':11},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Jenkins','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':4},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':10},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':14},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':15},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':16},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Jenkins','decoder':'FORCE','ST':8},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Jenkins','decoder':'FORCE','ST':12},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Jenkins','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':4},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':8},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':12},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Jenkins','decoder':'FORCE','ST':6},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Jenkins','decoder':'FORCE','ST':10},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-02-09','source':'Sussillo','monkey':'Jenkins','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-09','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':4},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-09','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':10},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-09','source':'Sussillo','monkey':'Jenkins','decoder':'VKF','ST':15},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-09','source':'Sussillo','monkey':'Jenkins','decoder':'FORCE','ST':7},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-09','source':'Sussillo','monkey':'Jenkins','decoder':'FORCE','ST':13},ignore_index=True)

## Sussillo-Larry
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Larry','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Larry','decoder':'VKF','ST':5},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Larry','decoder':'VKF','ST':10},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-07','source':'Sussillo','monkey':'Larry','decoder':'FORCE','ST':7},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Larry','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Larry','decoder':'VKF','ST':5},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Larry','decoder':'VKF','ST':12},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Larry','decoder':'VKF','ST':13},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Larry','decoder':'VKF','ST':14},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-08','source':'Sussillo','monkey':'Larry','decoder':'FORCE','ST':7},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-02-10','source':'Sussillo','monkey':'Larry','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-10','source':'Sussillo','monkey':'Larry','decoder':'VKF','ST':6},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-10','source':'Sussillo','monkey':'Larry','decoder':'FORCE','ST':8},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-10','source':'Sussillo','monkey':'Larry','decoder':'FORCE','ST':14},ignore_index=True)

rigC_df = rigC_df.append({'date':'2011-02-11','source':'Sussillo','monkey':'Larry','decoder':'hand','ST':1},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-11','source':'Sussillo','monkey':'Larry','decoder':'VKF','ST':5},ignore_index=True)
rigC_df = rigC_df.append({'date':'2011-02-11','source':'Sussillo','monkey':'Larry','decoder':'FORCE','ST':10},ignore_index=True)

 
def loadMonkeyData(dt,bDropInvisible=True,bSmoothPos=False,date=None,source=None,monkey=None,decoders=None):
    '''
    Feature: 
        * load by date
        * load by monkey
        * load by source
    '''

    valid_index = [True]*len(rigC_df)
    if date!=None:
        valid_index = valid_index & (rigC_df['date']==date)
    if source!=None:
        valid_index = valid_index & (rigC_df['source']==source)
    if monkey!=None:
        valid_index = valid_index & (rigC_df['monkey']==monkey)
    if decoders!=None:
        valid_index = valid_index & (rigC_df['decoder'].isin(decoders))

    rt_df = rigC_df[valid_index].reset_index(drop=True).copy()
    rt_df['data']=None
    for index,row in rt_df.iterrows():
        source = row['source']
        monkey = row['monkey']
        date = row['date']
        ST = int(row['ST'])
        data_folder = "../../BCI/data/{}_{}/".format(source,monkey)
        data = RigCDataset(data_folder+"/R_{}_1_ST{}.mat".format(date,ST),dt=dt,bDropInvisible=bDropInvisible,bSmoothPos=bSmoothPos)
        rt_df['data'][index] = data

    return rt_df


    
        
    #df = pd.DataFrame(columns=['date','decoder','data'])
    ##date = '2011-05-19'
    ##dt = 50
    #data_folder = "../data/{}_{}/".format(source,monkey)

 
    #ST_list = _get_ST_list(date,source,monkey)
    #decoder_list = ['hand','FIT','ReFIT','FORCE','VKF']

    #for decoder in decoder_list:
    #    for ST in ST_list[decoder]:
    #        data = RigCDataset(data_folder+"/R_{}_1_ST{}.mat".format(date,ST),dt=dt)
    #        df = df.append({
    #            'date':date,
    #            'decoder':decoder,
    #            'saveTag':ST,
    #            'data':data
    #            },ignore_index=True)

    return df
