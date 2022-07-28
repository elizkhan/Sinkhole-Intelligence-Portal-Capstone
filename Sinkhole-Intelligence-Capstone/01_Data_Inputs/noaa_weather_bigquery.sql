/*
This queries the GSOD (Global Surface Summary of the Day) open dataset from NOAA on GCP
https://data.noaa.gov/dataset/dataset/global-surface-summary-of-the-day-gsod
Weather data available here is daily
*/

SELECT
--Weather attributes
concat(a.stn, "_", a.wban) as stn_wban,                          -- station combo key
year, mo, da,                                                    -- date
min, max, temp,                                                  -- temperatures
prcp, flag_prcp, rain_drizzle,                                   -- prepcipitation
sndp, fog, snow_ice_pellets, hail, thunder,tornado_funnel_cloud, -- others
wdsp, mxpsd, gust, visib, dewp, slp,                             -- these are likley not relevant, optional

--Station attributes
concat(b.usaf, "_", b.wban) as usaf_wban,
b.wban,
b.lon,
b.lat,
b.elev,
b.name,
b.begin,
b.end,
b.state,
b.country

FROM `bigquery-public-data.noaa_gsod.gsod2015` a
--left join `w210-franny.noaa_weather_data.us_weather_stations_new` b
left join `bigquery-public-data.noaa_gsod.stations` b
  ON concat(a.stn, "_", a.wban) = concat(b.usaf, "_", b.wban)
where b.state = "FL" and b.country = "US"
--limit 10

union all

SELECT
--Weather attributes
concat(a.stn, "_", a.wban) as stn_wban,                          -- station combo key
year, mo, da,                                                    -- date
min, max, temp,                                                  -- temperatures
prcp, flag_prcp, rain_drizzle,                                   -- prepcipitation
sndp, fog, snow_ice_pellets, hail, thunder,tornado_funnel_cloud, -- others
wdsp, mxpsd, gust, visib, dewp, slp,                             -- these are likley not relevant, optional

--Station attributes
concat(b.usaf, "_", b.wban) as usaf_wban,
b.wban,
b.lon,
b.lat,
b.elev,
b.name,
b.begin,
b.end,
b.state,
b.country

FROM `bigquery-public-data.noaa_gsod.gsod2016` a
--left join `w210-franny.noaa_weather_data.us_weather_stations_new` b
left join `bigquery-public-data.noaa_gsod.stations` b
  ON concat(a.stn, "_", a.wban) = concat(b.usaf, "_", b.wban)
where b.state = "FL" and b.country = "US"
--limit 10

union all

SELECT
--Weather attributes
concat(a.stn, "_", a.wban) as stn_wban,                          -- station combo key
year, mo, da,                                                    -- date
min, max, temp,                                                  -- temperatures
prcp, flag_prcp, rain_drizzle,                                   -- prepcipitation
sndp, fog, snow_ice_pellets, hail, thunder,tornado_funnel_cloud, -- others
wdsp, mxpsd, gust, visib, dewp, slp,                             -- these are likley not relevant, optional

--Station attributes
concat(b.usaf, "_", b.wban) as usaf_wban,
b.wban,
b.lon,
b.lat,
b.elev,
b.name,
b.begin,
b.end,
b.state,
b.country

FROM `bigquery-public-data.noaa_gsod.gsod2017` a
--left join `w210-franny.noaa_weather_data.us_weather_stations_new` b
left join `bigquery-public-data.noaa_gsod.stations` b
  ON concat(a.stn, "_", a.wban) = concat(b.usaf, "_", b.wban)
where b.state = "FL" and b.country = "US"
--limit 10

union all




SELECT
--Weather attributes
concat(a.stn, "_", a.wban) as stn_wban,                          -- station combo key
year, mo, da,                                                    -- date
min, max, temp,                                                  -- temperatures
prcp, flag_prcp, rain_drizzle,                                   -- prepcipitation
sndp, fog, snow_ice_pellets, hail, thunder,tornado_funnel_cloud, -- others
wdsp, mxpsd, gust, visib, dewp, slp,                             -- these are likley not relevant, optional

--Station attributes
concat(b.usaf, "_", b.wban) as usaf_wban,
b.wban,
b.lon,
b.lat,
b.elev,
b.name,
b.begin,
b.end,
b.state,
b.country

FROM `bigquery-public-data.noaa_gsod.gsod2018` a
--left join `w210-franny.noaa_weather_data.us_weather_stations_new` b
left join `bigquery-public-data.noaa_gsod.stations` b
  ON concat(a.stn, "_", a.wban) = concat(b.usaf, "_", b.wban)
where b.state = "FL" and b.country = "US"
--limit 10

union all

SELECT
--Weather attributes
concat(a.stn, "_", a.wban) as stn_wban,                          -- station combo key
year, mo, da,                                                    -- date
min, max, temp,                                                  -- temperatures
prcp, flag_prcp, rain_drizzle,                                   -- prepcipitation
sndp, fog, snow_ice_pellets, hail, thunder,tornado_funnel_cloud, -- others
wdsp, mxpsd, gust, visib, dewp, slp,                             -- these are likley not relevant, optional

--Station attributes
concat(b.usaf, "_", b.wban) as usaf_wban,
b.wban,
b.lon,
b.lat,
b.elev,
b.name,
b.begin,
b.end,
b.state,
b.country

FROM `bigquery-public-data.noaa_gsod.gsod2019` a
--left join `w210-franny.noaa_weather_data.us_weather_stations_new` b
left join `bigquery-public-data.noaa_gsod.stations` b
  ON concat(a.stn, "_", a.wban) = concat(b.usaf, "_", b.wban)
where b.state = "FL" and b.country = "US"


union all

SELECT
--Weather attributes
concat(a.stn, "_", a.wban) as stn_wban,                          -- station combo key
year, mo, da,                                                    -- date
min, max, temp,                                                  -- temperatures
prcp, flag_prcp, rain_drizzle,                                   -- prepcipitation
sndp, fog, snow_ice_pellets, hail, thunder,tornado_funnel_cloud, -- others
wdsp, mxpsd, gust, visib, dewp, slp,                             -- these are likley not relevant, optional

--Station attributes
concat(b.usaf, "_", b.wban) as usaf_wban,
b.wban,
b.lon,
b.lat,
b.elev,
b.name,
b.begin,
b.end,
b.state,
b.country

FROM `bigquery-public-data.noaa_gsod.gsod2020` a
--left join `w210-franny.noaa_weather_data.us_weather_stations_new` b
left join `bigquery-public-data.noaa_gsod.stations` b
  ON concat(a.stn, "_", a.wban) = concat(b.usaf, "_", b.wban)
where b.state = "FL" and b.country = "US"

union all

SELECT
--Weather attributes
concat(a.stn, "_", a.wban) as stn_wban,                          -- station combo key
year, mo, da,                                                    -- date
min, max, temp,                                                  -- temperatures
prcp, flag_prcp, rain_drizzle,                                   -- prepcipitation
sndp, fog, snow_ice_pellets, hail, thunder,tornado_funnel_cloud, -- others
wdsp, mxpsd, gust, visib, dewp, slp,                             -- these are likley not relevant, optional

--Station attributes
concat(b.usaf, "_", b.wban) as usaf_wban,
b.wban,
b.lon,
b.lat,
b.elev,
b.name,
b.begin,
b.end,
b.state,
b.country

FROM `bigquery-public-data.noaa_gsod.gsod2021` a
--left join `w210-franny.noaa_weather_data.us_weather_stations_new` b
left join `bigquery-public-data.noaa_gsod.stations` b
  ON concat(a.stn, "_", a.wban) = concat(b.usaf, "_", b.wban)
where b.state = "FL" and b.country = "US"

union all

SELECT
--Weather attributes
concat(a.stn, "_", a.wban) as stn_wban,                          -- station combo key
year, mo, da,                                                    -- date
min, max, temp,                                                  -- temperatures
prcp, flag_prcp, rain_drizzle,                                   -- prepcipitation
sndp, fog, snow_ice_pellets, hail, thunder,tornado_funnel_cloud, -- others
wdsp, mxpsd, gust, visib, dewp, slp,                             -- these are likley not relevant, optional

--Station attributes
concat(b.usaf, "_", b.wban) as usaf_wban,
b.wban,
b.lon,
b.lat,
b.elev,
b.name,
b.begin,
b.end,
b.state,
b.country

FROM `bigquery-public-data.noaa_gsod.gsod2022` a
--left join `w210-franny.noaa_weather_data.us_weather_stations_new` b
left join `bigquery-public-data.noaa_gsod.stations` b
  ON concat(a.stn, "_", a.wban) = concat(b.usaf, "_", b.wban)
where b.state = "FL" and b.country = "US"
