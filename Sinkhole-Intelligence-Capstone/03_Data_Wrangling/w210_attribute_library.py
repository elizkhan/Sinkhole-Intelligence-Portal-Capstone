import numpy as np
import pandas as pd
import datetime
from datetime import datetime, timedelta

# Consolidates Satellite Data for a given period
def consolidate_sat_data(daysd, datdirsat, g10file, g2file, sh, nogood):
    daysdelta = timedelta(daysd)
    df10 = pd.read_csv(datdirsat+g10file)
    df10["Group"] =  np.where(df10["label"]==1,0,1)
    
    df2 = pd.read_csv(datdirsat+g2file)
    df2 = df2[df2["label"] == 0]
    df2["Group"] = 2
    
    df3 = df10.append(df2)
    df3 = pd.merge(df3, sh, how="left", on=["ID", "ID"])
    df3["Key"] = df3.apply(lambda row: str(row["ID"])+"_"+str(row["label"])+"_"+str(row["Group"]), axis=1)
    
    df3 = df3[~df3["Key"].isin(nogood)]
    
    return df3


# Check for tile_pair until no-good list is zero
def ws_tile_pair_final(fname, ws, dfw):

    dft = pd.read_csv(fname)
    dft["DateD"] = dft.apply(lambda row: datetime.strptime(row["DateD"],"%Y-%m-%d"), axis=1)
    
    not_finished = True
    wsnogood_list = []
    
    while not_finished:
        df = ws_tile_pairs(dft, ws)
        wsgood, wsnogood = checkwsquality(df, dfw)
        
        print("No good: ",len(wsnogood))

        if len(wsnogood) > 0:
            wsnogood_list = wsnogood_list + wsnogood
            print("Cumulative no good: ", wsnogood_list)
            ws = ws[~ws["Key"].isin(wsnogood_list)]
        else:
            not_finished = False
    
    return df


# Return list of weather stations with required historic data - one list of good WS and no-good WS
def checkwsquality(df, dfw1):

    i = 0
    wsgood = []
    wsnogood = []
    for index, row in df.iterrows():
        wsname = row["Key_y"]
        dated = row["DateD"]
        td = timedelta(365*3)
        dftemp = dfw1[(dfw1["stn_wban"] == wsname)]
        if (dftemp["DateD"].max() > dated) & (dftemp["DateD"].min() < dated - td):
            wsgood.append(wsname)
        else:
            wsnogood.append(wsname)
    
    wsng = set(wsnogood)
    wsg = set(wsgood)
    return list(wsg), list(wsng)


# To pair closest weather station to the tiles
def ws_tile_pairs(dft, ws):

    # Cross merge ws and tiledata
    dfcross = pd.merge(dft,ws, how="cross")
    
    # Calculate the distance
    dfcross['Distance'] = dfcross.apply(lambda row: 
                                        haversine_distance(row['lat'], row['lon'], 
                                                           row['lat_w'], row['lon_w'], 
                                                           earth_radius=3963.19), axis=1)
    # Find the minimum distance by tile
    dfmin1 = dfcross.groupby(['Key_x'])['Distance'].min().to_frame()
    
    # Select tiles with the minimum distance
    keysL = list(dfmin1.index)
    minD = list(dfmin1['Distance'])
    dfF = dfcross[((dfcross['Key_x'].isin(keysL)) &  (dfcross['Distance'].isin(minD)))]

    # Drop duplicates
    dfF.drop_duplicates(subset=['Key_x'], inplace=True)
    
    return dfF


def withinstates(geometries, point):
    
    in_geometry = "No"
    for geometry in geometries:
        if point.within(geometry):
            return "Yes"
    
    return in_geometry


def assignGroup1(label):
    if label == 1:
        return 0
    else:
        return 1
    
def assignGroup2(label):
    if label == 0:
        return 2
    else:
        return 0
    

def haversine_distance(lat1, long1, lat2,long2, earth_radius=6371):
    
    #earth_radius in miles = 3963.19
    
    a = np.sin((lat2/57.2957795 - lat1/57.2957795)/2) * \
        np.sin((lat2/57.2957795 - lat1/57.2957795)/2) + \
        np.cos(lat2/57.2957795) * np.cos(lat1/57.2957795) * \
        np.sin((long1/57.2957795 - long2/57.2957795)/2) * \
        np.sin((long1/57.2957795 - long2/57.2957795)/2) 
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    #distance=6371*c*3280.84 #for distance in feet
    distance = earth_radius*c
    return distance


def getdate(dated):
    
    if type(dated) == str:
        return(datetime.strptime(dated[0:10],"%Y/%m/%d"))
    else:
        return ('NaN')
    
def findname(ID, list):
    for n in list:
        if int(n.split("-")[0]) == ID:
            return n

def findAttr(lon, lat, ID, dateE, shdf, EarthRadius):
    
    dif1 = 13
    nearObj = np.zeros((26,), dtype=int)    #Change from 6 to 12 - to include difference in years
        
    for indsh, shrow in shdf.iterrows():
        
        if shrow["OBJECTID"] != ID:
            
            if dateE >= shrow["DateD"]:
            
                lon2 = shrow["X"]
                lat2 = shrow["Y"]

                if (lon == lon2) & (lat == lat2):
                    nearObj[12] += 1
                    dd = (dateE - shrow["DateD"]).days
                    nearObj[12+dif1] += dd/365
                else:
                    distance = haversine_distance(lat, lon, lat2, lon2, earth_radius=EarthRadius)

                    if distance > 10: index = 11
                    elif distance <= 0.25: index = 0
                    elif distance <= 0.50: index = 1
                    elif distance <= 0.75: index = 2
                    elif distance <= 1.0:  index = 3
                    elif distance <= 1.5:  index = 4
                    elif distance <= 2.0:  index = 5
                    elif distance <= 2.5:  index = 6
                    elif distance <= 3.0:  index = 7
                    elif distance <= 5.0:  index = 8
                    elif distance <= 7.5:  index = 9
                    elif distance <= 10.0:  index = 10
                        

                    for j in range(index,dif1-1):
                        nearObj[j] += 1
                        dd = (dateE - shrow["DateD"]).days
                        nearObj[j+dif1] += dd/365
    
    
    return(nearObj)    


def sh_attr2(finEvents, df_left, daysdelta, att_list, dist_list, col_names):

    datash = []
    for index, row in finEvents.iterrows():
#         print(row["Key"])
        refdate = row["DateD"]-daysdelta
        dft1 = df_left[(df_left["DateD"] <= refdate)]
        dft1["Distance_Miles"] = dft1.apply(lambda rowsh: haversine_distance(row["Y"], row["X"], rowsh["Y"],rowsh["X"], 3963.19), axis=1)
        dft1 = dft1[dft1["Distance_Miles"] != 0]

        for i in range(11):
            dft1["l"+att_list[i]] = dft1.apply(lambda r: 1 if (r["Distance_Miles"] <=  dist_list[i]) else 0, axis=1)
            dft1["Y"+att_list[i]] = dft1.apply(lambda r: (refdate - r["DateD"]).days / 365 if (r["Distance_Miles"] <=  dist_list[i]) else 0, axis=1)

        vals = list(row) + list(dft1[col_names].sum())

        datash.append(vals)
        
    return datash

        
def shAttributes(shdf, finEvents, daysdelta, fields):
    
    attribute_list = ["l25","l50", "l75", "l100", "l150", "l200",
                      "l250", "l300", "l500", "l750", "l1000", "l1000plus", "coloc",
                      "Y25", "Y50", "Y75", "Y100", "Y150", "Y200",
                      "Y250", "Y300", "Y500", "Y750", "Y1000", "Y1000plus", "Ycoloc"]
    
    print(len(attribute_list))

    attrdata = []
    
    for index, row in finEvents.iterrows():
        
        ref_date = row["DateD"] - daysdelta
        shattr = findAttr(row[fields[0]], row[fields[1]], row[fields[2]], ref_date, shdf, 3963.19)
        
        attrdata.append(shattr)
        

    attrdata = np.array(attrdata)
#     print("******************************")
#     print(attrdata)
    
    i = 0
    for attribute in attribute_list:
        finEvents[attribute] = attrdata[:,i]
        i +=1
           
    return(finEvents)

def getwf_final(wfl, wsname, dated, dfw1):
    
#     print(wsname, dated)
    
    dftemp = dfw1[((dfw1["stn_wban"] == wsname) & (dfw1["DateD"] <= dated))]
    dftemp = dftemp.sort_values(by='DateD', ascending=False)
    dftemp = dftemp.reset_index()
    
    # Get Rolling Average
    ndays = 90
    r1 = dftemp.iloc[0:ndays,]
    
    windowl = [7, 15, 30, 60, 90]
    
    for window in windowl:
        prcsum = r1.iloc[0:window,]['prcp'].sum()
        wfl.append(prcsum)
    
    # Get Average Metrics for the last 3 years
    ndays = 365
    
    for i in range(0,3):
        r1 = dftemp.iloc[ndays*i:ndays*(i+1),]
        wfl.append(r1["prcp"].mean())
        wfl.append(r1["prcp"].max())
        wfl.append(r1["temp"].mean())
        wfl.append(r1["temp"].max())
        wfl.append(r1["temp"].min())        
    
    return(wfl)


def weatherFeatures(dfF2, dfw1, is_scale, daysdelta, date_string):

    # LOOK to extract and join Weather features

    wsfeatures = ['Key','Key_y', 'date_ws','name', 'lon_t', 'lat_t', 'lon_w', 'lat_w', 'Distance',
                  'rolling_7_precip', 'rolling_15_precip', 'rolling_30_precip', 
                  'rolling_60_precip', 'rolling_90_precip',
                  'y1_mean_prc', 'y1_max_prc', 'y1_mean_tmp', 'y1_max_tmp', 'y1_min_tmp',
                  'y2_mean_prc', 'y2_max_prc', 'y2_mean_tmp', 'y2_max_tmp', 'y2_min_tmp', 
                  'y3_mean_prc', 'y3_max_prc', 'y3_mean_tmp', 'y3_max_tmp', 'y3_min_tmp'
                 ]

    wfdata = []

    daysdelta = timedelta(365)

    for index, row in dfF2.iterrows():

        wsname = row["Key_y"]
        
        if is_scale:
            dated = datetime.strptime(date_string,"%Y/%m/%d") - daysdelta
        else:
            dated = row["DateD"] - daysdelta
            
        wfl = [row['Key_x'], row['Key_y'], dated, row['name_y'], row['lon'], row['lat'], row['lon_w'], row['lat_w'], row['Distance']]

        wfl = getwf_final(wfl, wsname, dated, dfw1)
        wfdata.append(wfl)

    dfwf = pd.DataFrame(wfdata, columns=wsfeatures)
    
    return dfwf


# NOT USED ANYMORE

def getwf1(wfl, wsname, dated, dfw1):
    
#     print(wsname, dated)
    
    dftemp = dfw1[((dfw1["stn_wban"] == wsname) & (dfw1["DateD"] <= dated))]
    dftemp = dftemp.sort_values(by='DateD', ascending=False)
    dftemp = dftemp.reset_index()
    
    ndays = 90
    r1 = dftemp.iloc[0:ndays,]
#     r1 = r1.sort_values(by='DateD')
    
    windowl = [7, 15, 30, 60, 90]
    
    for window in windowl:
        prcsum = r1.iloc[0:window,]['prcp'].sum()
        wfl.append(prcsum)
    
#     print(wfl)
    return(wfl)


def getwf(wfl, wsname, dated, dfw1):
    
    dftemp = dfw1[((dfw1["stn_wban"] == wsname) & (dfw1["DateD"] <= dated))]
    dftemp = dftemp.sort_values(by='DateD', ascending=False)
    dftemp = dftemp.reset_index()
    ndays = 365
    
    for i in range(0,3):
        r1 = dftemp.iloc[ndays*i:ndays*(i+1),]
        wfl.append(r1["prcp"].mean())
        wfl.append(r1["prcp"].max())
        wfl.append(r1["temp"].mean())
        wfl.append(r1["temp"].max())
        wfl.append(r1["temp"].min())        
    
    return(wfl)



def findAttr_Prev(lon, lat, ID, dateE, shdf, EarthRadius):
    
    dif1 = 13
    nearObj = np.zeros((26,), dtype=int)    #Change from 6 to 12 - to include difference in years
    
    avgYear = np.zeros((dif1,), dtype=float)
    
    for indsh, shrow in shdf.iterrows():
        
        if shrow["OBJECTID"] != ID:
            
            if dateE >= shrow["DateD"]:
            
                lon2 = shrow["LONGDD"]
                lat2 = shrow["LATDD"]

                if (lon == lon2) & (lat == lat2):
                    nearObj[12] += 1
                    dd = (dateE - shrow["DateD"]).days
                    nearObj[12+dif1] = dd/365
                else:
                    distance = haversine_distance(lat, lon, lat2, lon2, earth_radius=EarthRadius)

                    if distance > 10: index = 11
                    elif distance <= 0.25: index = 0
                    elif distance <= 0.50: index = 1
                    elif distance <= 0.75: index = 2
                    elif distance <= 1.0:  index = 3
                    elif distance <= 1.5:  index = 4
                    elif distance <= 2.0:  index = 5
                    elif distance <= 2.5:  index = 6
                    elif distance <= 3.0:  index = 7
                    elif distance <= 5.0:  index = 8
                    elif distance <= 7.5:  index = 9
                    elif distance <= 10.0:  index = 10
                        

                    for j in range(index,dif1-1):
                        nearObj[j] += 1
                        dd = (dateE - shrow["DateD"]).days
                        nearObj[j+dif1] += dd/365
    
    for j in range(0, dif1):
        if nearObj[j] != 0:
            avgYear[j] = nearObj[j+dif1]/nearObj[j]
        else:
            avgYear[j] = math.nan        
    
    return(nearObj, avgYear)    

def shAttributes_Prev(shdf, finEvents):
    
    l25, l50, l75, l100, l150, l200 = [], [], [],[], [], []
    l250, l300, l500, l750, l1000, l1000plus, coloc = [], [], [],[], [], [], []
    
    Y25, Y50, Y75, Y100, Y150, Y200 = [], [], [],[], [], []
    Y250, Y300, Y500, Y750, Y1000, Y1000plus, Ycoloc = [], [], [],[], [], [], []

    AY25, AY50, AY75, AY100, AY150, AY200 = [], [], [],[], [], []
    AY250, AY300, AY500, AY750, AY1000, AY1000plus, AYcoloc = [], [], [],[], [], [], []

    
    for index, row in finEvents.iterrows():
        attribrutes, avgYr = findAttr(row["LONGDD"], row["LATDD"], row["OBJECTID"], row["DateD"], shdf, 3963.19)
        print("ObjectID: ", row["OBJECTID"], attribrutes)
        
        l25.append(attribrutes[0])
        l50.append(attribrutes[1])
        l75.append(attribrutes[2])
        l100.append(attribrutes[3])
        l150.append(attribrutes[4])
        l200.append(attribrutes[5])
        l250.append(attribrutes[6])
        l300.append(attribrutes[7])
        l500.append(attribrutes[8])
        l750.append(attribrutes[9])
        l1000.append(attribrutes[10])
        l1000plus.append(attribrutes[11])
        coloc.append(attribrutes[12])

        Y25.append(attribrutes[13])
        Y50.append(attribrutes[14])
        Y75.append(attribrutes[15])
        Y100.append(attribrutes[16])
        Y150.append(attribrutes[17])
        Y200.append(attribrutes[18])
        Y250.append(attribrutes[19])
        Y300.append(attribrutes[20])
        Y500.append(attribrutes[21])
        Y750.append(attribrutes[22])
        Y1000.append(attribrutes[23])
        Y1000plus.append(attribrutes[24])
        Ycoloc.append(attribrutes[25])

        AY25.append(avgYr[0])
        AY50.append(avgYr[1])
        AY75.append(avgYr[2])
        AY100.append(avgYr[3])
        AY150.append(avgYr[4])
        AY200.append(avgYr[5])
        AY250.append(avgYr[6])
        AY300.append(avgYr[7])
        AY500.append(avgYr[8])
        AY750.append(avgYr[9])
        AY1000.append(avgYr[10])
        AY1000plus.append(avgYr[11])
        AYcoloc.append(avgYr[12])
        
    finEvents["0.25"] = l25
    finEvents["0.5"] = l50
    finEvents["0.75"] = l75
    finEvents["1.0"] = l100
    finEvents["1.5"] = l150
    finEvents["2.0"] = l200
    finEvents["2.5"] = l250
    finEvents["3.0"] = l300
    finEvents["5.0"] = l500
    finEvents["7.5"] = l750
    finEvents["10"] = l1000
    finEvents[">10"] = l1000plus
    finEvents["Coloc"] = coloc

    finEvents["Y0.25"] = Y25
    finEvents["Y0.5"] = Y50
    finEvents["Y0.75"] = Y75
    finEvents["Y1.0"] = Y100
    finEvents["Y1.5"] = Y150
    finEvents["Y2.0"] = Y200
    finEvents["Y2.5"] = Y250
    finEvents["Y3.0"] = Y300
    finEvents["Y5.0"] = Y500
    finEvents["Y7.5"] = Y750
    finEvents["Y10"] = Y1000
    finEvents["Y>10"] = Y1000plus
    finEvents["YColoc"] = Ycoloc

    finEvents["AY0.25"] = AY25
    finEvents["AY0.5"] = AY50
    finEvents["AY0.75"] = AY75
    finEvents["AY1.0"] = AY100
    finEvents["AY1.5"] = AY150
    finEvents["AY2.0"] = AY200
    finEvents["AY2.5"] = AY250
    finEvents["AY3.0"] = AY300
    finEvents["AY5.0"] = AY500
    finEvents["AY7.5"] = AY750
    finEvents["AY10"] = AY1000
    finEvents["AY>10"] = AY1000plus
    finEvents["AYColoc"] = AYcoloc


    return(finEvents)