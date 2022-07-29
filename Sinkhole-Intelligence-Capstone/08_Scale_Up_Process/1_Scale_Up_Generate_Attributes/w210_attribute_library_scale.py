import numpy as np
import pandas as pd
import datetime
from datetime import datetime, timedelta

# IMPORTANT - SET THE DATA OF REFERENCE
date_string = '2022/06/01'


#Given a geographic point, and the list of counties in florida, return the country name and county FP
def findcounty(point, flcounty):
    for index, row in flcounty.iterrows():
        if point.within(row['geometry']):
            return (row["NAME"], row["COUNTYFP"])
        
def findcountyfp(point, flcounty):
    for index, row in flcounty.iterrows():
        if point.within(row['geometry']):
            return row["COUNTYFP"]
    return "No_Florida"

def withinstates(geometries, point):
    
    in_geometry = "No"
    for geometry in geometries:
        if point.within(geometry):
            return "Yes"
    
    return in_geometry

def getdate(dated):
    
    if type(dated) == str:
        return(datetime.strptime(dated[0:10],"%Y/%m/%d"))
    else:
        return ('NaN')
    
    
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

def tilekey(row):
    county = str(row["CountyFP"])
    return county+"_"+str(row["Unnamed: 0"])


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
    
    
    return(nearObj)    


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

    i = 0
    for attribute in attribute_list:
        finEvents[attribute] = attrdata[:,i]
        i +=1
           
    return(finEvents)

def weatherFeaturesSU(dfF2, dfw1, is_scale, daysdelta, date_string, fields):

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
            
        wfl = [row[fields[0]], row[fields[1]], dated, row[fields[2]], row[fields[3]], row[fields[4]], row[fields[5]], row[fields[6]], row[fields[7]]]

        wfl = getwf_final(wfl, wsname, dated, dfw1)
        wfdata.append(wfl)

    dfwf = pd.DataFrame(wfdata, columns=wsfeatures)
    
    return dfwf